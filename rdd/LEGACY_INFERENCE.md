# FastWan multiresolution inference

This entrypoint uses pretrained FastWan DMD/VSA weights with three model evaluations.
It is separate from the non-distilled Wan RDD training and Euler50 sampler in
[TRAINING.md](TRAINING.md). Published FastWan weights were not trained on this RDD flow path.

## Pipeline

```text
rdd.run -> upstream TransformerLoader -> WanTransformerBlock_VSA
    -> rdd.sampler.sample: low -> middle -> high resolution
    -> upstream normalization / QKV / RoPE / tiling
    -> rdd.attention.C2FController: route capture and lifting
    -> upstream sparse attention or optional SmallQ executor
    -> upstream output projection / residual / cross-attention / MLP
```

The controller replaces each instance's self-attention implementation, not the model parameters
or block architecture. Call `reset()` between videos and `close()` to restore the original hook.

## Sampling

Timesteps are `1000,757,522`, with the upstream shift-8 scheduler and clean-prediction conversion.
Each boundary upsamples the clean prediction, then applies
`(1-sigma_next)*clean_up + sigma_next*fresh_IID_Gaussian` (`noise_filling`).
Factors are `(spatial, temporal)`; latent dimensions are `T,H,W`.

| Factors | Stage latent shapes | Final output |
|---|---|---|
| (4,1) -> (2,1) -> (1,1) | 8x14x26 -> 8x28x52 -> 8x56x104 | 832x448, 29 frames |
| (4,4) -> (2,2) -> (1,1) | 2x14x26 -> 4x28x52 -> 8x56x104 | 832x448, 29 frames |

## C2F routing

1. Average Q/K within each 4x4x4-token tile and compute block scores.
2. Select `ceil(keep * block_count)` keys, replacing the final slot with the self block if needed.
3. Map each fine query block to its coarse parent and expand the selected keys to their fine children.
4. `kernels/route_lift.py` writes compact `indices[B,H,Q,Kcapacity]` and `counts[B,H,Q]` on GPU.
5. Compute sparse attention with current Q/K/V. No old attention probabilities or AV outputs are copied.

Non-integer block grids use coordinate mapping. Padding slots are not initialized;
the executor reads only the prefix specified by `counts`. Coarse-route tensors remain live
until the trajectory is reset.

| Mode | Fine-stage routing | Current coarse output | Next-stage guide |
|---|---|---|---|
| `vsa` | Online Top-K | Upstream branch retained | None |
| `c2f_route` | Previous-stage routes | Retained; current coarse QK/AV still computed | Fresh current-stage block scores |
| `c2f_route_only` | Inherited routes | Removed after the seed stage | Inherited indices expanded again |

`c2f_route` is the default. `c2f_route_only` is a structural ablation, not an equivalent
kernel replacement for VSA. With eight coarse blocks, requested 20% rounds up to two
blocks (25%); the next stage does not multiply that density by 20% again.

## Run

Use Linux, NVIDIA CUDA, BF16, and the pinned upstream environment.
The input is an official Wan T5 embedding, padded to `[1,512,D]`, saved as a trusted Tensor.
The CLI exports latents; T5/VAE are not included in its timing.

```bash
export PYTHONPATH="$PWD:$PWD/fastvideo-kernel/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
python -m rdd.run --transformer /path/to/FastWan/transformer \
  --context /path/to/context.pt --factors '4,1;2,1;1,1' \
  --mode c2f_route --keep 0.2 --warmup 1 --repeat 3 --output /tmp/rdd_c2f.pt
```

| Variant | Arguments |
|---|---|
| Fixed-resolution VSA | `--factors '1,1;1,1;1,1' --mode vsa` |
| Multiresolution VSA | `--factors '4,1;2,1;1,1' --mode vsa` |
| C2F with coarse output | `--factors '4,1;2,1;1,1' --mode c2f_route` |
| Optional SmallQ | `--executor triton_q32 --smallq-max-tokens 4096` |
| Fused noise filling | `--boundary triton_fused` |

SmallQ applies only to C2F; `vsa` always uses the upstream executor.
Keep weights, input, seed, precision, density, offload and warmup fixed for comparisons.
The CLI supports one GPU, Wan patch (1,2,2), 64-token VSA blocks and three DMD evaluations.
Custom kernels are not guaranteed to be faster at every shape.

## Validation and attribution

```bash
python -m unittest discover -s rdd/tests -v
RDD_TEST_CUDA=1 python -m unittest discover -s rdd/tests -v
```

CPU tests cover shapes, route density, noise seeds, hook restoration and route lifetime.
Opt-in GPU tests compare route expansion, masked attention, BF16 boundary arithmetic
and upstream VSA parity. These component checks do not establish image quality.

The original extraction was checked against nine real-checkpoint sampling configurations
(one warmup and two measured runs each), with identical final latents versus the reference
implementation and successful VAE decoding. This is compatibility evidence, not a fair speed
benchmark or proof of quality. Aggressive schedules can produce visible artifacts.

The controller and sampler were extracted from `sample_fastwan_native_vsa.py` and
`fastwan_native_attention.py`; kernels from `rdd_c2f_route_triton.py`,
`rdd_sparse_attn_smallq_triton.py` and `rdd_noise_filling_triton.py` in the original experiments.
SmallQ follows the upstream VSA Triton online-softmax and sparse-KV structure.
The upstream [Apache-2.0 license](../LICENSE) and copyright notices are retained.

## Diff

```bash
git diff fastvideo-base HEAD -- rdd/
python rdd/review.py
```

Open the generated `rdd/review.html` for an offline file-by-file view.
