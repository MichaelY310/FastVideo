# RDD finetuning: original Wan2.1-1.3B

Full-Transformer flow-matching finetuning of the non-distilled Wan model.
No adapter, layer skipping, DMD training, or upstream FastVideo edits.
See [results and videos](../README_RDD.md).

## Configuration

| Item | Value |
|---|---|
| Base weights | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Data | `FastVideo/mixkit_filtered_6k_wan1.3_t2v`; preencoded latents and T5 embeddings |
| Pilot split | 384 train / 32 held-out clips; disjoint IDs/files, no near-duplicate audit |
| Parameters | Train the full Transformer; freeze T5/VAE; dense FlashAttention |
| Optimizer | AdamW; LR 1e-5; betas 0.9/0.999; weight decay 0.01; grad clip 1 |
| Batch | 8-GPU FSDP; 1/GPU; accumulation 2; global 16 |
| Precision | FP32 master parameters; BF16 mixed compute |
| Schedule | Uniform time; 100-update warmup; constant LR; CFG dropout 0.1 |
| Checkpoints | Every 100 updates; retain the latest two complete training states |
| Pilot duration | 5,003 updates; before/after videos at updates 0 and 5,000 |

### Stages

`300/300/400` names the high/middle/low allocation. Generation proceeds low to high.
The 1,000-unit time coordinate is not an optimizer-update count or a sampling NFE count.

| Stage | Factor (spatial, temporal) | Data-time | Latent T x H x W | Euler updates |
|---|---|---|---|---|
| Low | (4,1) | 0 to 0.4 | 8 x 14 x 26 | 20 |
| Middle | (2,1) | 0.4 to 0.7 | 8 x 28 x 52 | 15 |
| High | (1,1) | 0.7 to 1 | 8 x 56 x 104 | 15 |

Wan normalization is applied once before constructing stage latents.
Each accumulation microbatch draws a stage; all ranks share the stage but use different samples.

## Flow path

Implemented in `flow.py`; data-time t=0 is noise and t=1 is data.
Wan receives timestep `1000 * (1 - t)`.

```text
x1 = clean latent at the current stage
P  = average-pool to the previous scale, then nearest upsample
d  = P(x1) - x1
[a,b] = current stage interval

Middle/high: f(t) = t*(b-t)/(b-a); f'(t) = (b-2*t)/(b-a)
Low:        d = 0; f(t) = 0

x(t)   = t*x1 + (1-t)*epsilon + f(t)*d
target = epsilon - x1 - f'(t)*d       # velocity with respect to sigma=1-t
Euler: x_next = x - dt*predicted_velocity
```

There is no learned stage transition, extra prediction head, or resolution SNR rescaling.
A noisy boundary latent is not directly treated as a clean image:

```text
y = x - (1-t)*predicted_velocity
c = f(t) + (1-t)*f'(t)
predicted_clean = P(y) + (y-P(y))/(1-c)
next_input = b*upsample(predicted_clean) + (1-b)*fresh_IID_Gaussian
```

This implementation calls the final operation `noise_filling`.

## Run

Linux/CUDA with the upstream FastVideo training environment is required.
Run from the repository root. Model/data downloads and training outputs are not included in Git.

```bash
export RUN_ROOT="$PWD/rdd/runs/wan_base"
python -m rdd.prepare_finetune --root "$RUN_ROOT/assets"
```

Preparation selects 48 train shards and 4 validation shards with a fixed shuffle,
records model/dataset revisions in `assets/provenance.json`, and requires 250 GiB free.
It resolves the current source revisions when run; the split counts above describe the recorded pilot,
not a guarantee about future revisions of the dataset.

Set `DEADLINE` to a future absolute ISO-8601 timestamp including its timezone,
and `TOKEN` to a unique run identifier of at least 20 characters, before launching:

```bash
bash rdd/launch_finetune_8gpu.sh
# Resume:
bash rdd/launch_finetune_8gpu.sh --training.checkpoint.resume_from_checkpoint latest
```

The launcher uses `PYTHON` (default: python), `RUN_ROOT`, and `CUDA_VISIBLE_DEVICES`.
It requires eight visible GPUs. Model/data/output paths follow RUN_ROOT; trailing
`--dotted.key value` arguments override YAML settings. The YAML defaults match the directory above.
For sampling with a custom RUN_ROOT, update the four path fields in a copy of the YAML.

### Safety

`deadline.py` requires Linux /proc and pidfd. The trainer saves/exits five minutes before the deadline;
the guard sends TERM two minutes before it, then KILL at the deadline if necessary.
Signals require the same UID and an exact run token; unrelated processes are never targeted.
The guard also stops this job if another user's GPU process appears or disk space falls below 200 GiB.
This is an OS-level safeguard, not a hard-real-time guarantee.

## Validation and limits

- CPU tests cover stage allocation, flow derivatives, clean recovery, shapes, and route lifecycle.
- Opt-in CUDA tests cover kernel/attention parity; Linux-only tests cover deadline identity and shutdown.
- Sampling uses Euler50/CFG5/shift1 plus two boundary evaluations: 104 conditional/unconditional forwards.
- Outputs are 832 x 448, 29 frames at 16 fps. The full-resolution reference uses 100 forwards.
- Periodic training previews use one held-out caption and one seed, not all 32 validation clips.
- The pilot improves that example's recognizable structure; hand/face artifacts and limited motion remain.
- No FVD/VBench or generalization claim. Training-preview timings are not inference speed benchmarks.
- Current C2F is inference-only and was not finetuned; see [C2F instructions](FINETUNED_C2F.md).
- Custom SmallQ/noise-filling kernels are forward-only legacy options, not used in this training path.
