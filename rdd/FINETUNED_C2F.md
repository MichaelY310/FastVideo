# Non-DMD Wan RDD + C2F inference (2026-09-27)

This is the dense Wan checkpoint adapter, not the historical FastWan/VSA hook.
No official FastVideo files, weights, text attention, or RDD equations are changed.

- Source: `rdd_video_finetune_20260926/train/checkpoint-5003` (full FP32 model DCP).
- Original Wan2.1-1.3B, full RDD finetuning, not DMD or VSA finetuning.
- RDD spatial factors `(4,1) -> (2,1) -> (1,1)`; data-time `[0,.4,.7,1]`.
- Euler 20/15/15, CFG5, two extra boundary velocity evaluations, 104 model calls.
- Output 832x448x29, 16fps; noise_filling only.

## Routing

`dense_c2f.py` patches only the self-attention implementation after RoPE.
The lowest stage remains dense. At a stage endpoint, average Q/K within official
4x4x4-token tiles, compute block scores, and select Top-K plus the own block
(own block replaces the final selected slot, never increases K).
Use `kernels/route_lift.py` to expand routes onto the next spatial grid. Current
Q/K/V are recomputed every call. Middle-stage endpoint produces a fresh guide for
the fine stage. Routes are separately keyed by layer and conditional/unconditional
CFG branch. Text attention is untouched. No old probabilities or AV are reused.

The source model has no VSA coarse-output gate/branch. This adapter does NOT remove
one, and does not add an untrained branch. Its baseline is dense RDD, not VSA.

## Execution

Use the FastVideo environment and the pinned upstream kernel source, from the repository root.
Set model/data paths in the YAML when using a custom asset directory.

```bash
PYTHONPATH=fastvideo-kernel/python:. CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29831 \
python -m rdd.sample_finetuned_c2f \
  --checkpoint rdd/runs/wan_base/train/checkpoint-5003 \
  --output rdd/runs/c2f/prompt0 \
  --row 0 --settings dense,c2f50,c2f30,c2f20,c2f12 --repeats 3
```

One process uses one GPU. Keep no more than four concurrent processes. The runner
refuses startup with less than40GiB available. In this run GPU0/1/2 generated
validation rows0/11/24; GPU3 checked dense versus keep100% equivalence. Later
single-GPU timing used `--timing-only-rounds 5`, rotating method order each round.
Model loading, text encoding and file/video export are excluded from the resident
denoise+VAE metric. No precision or offload advantage is given to sparse attention.

## Checks and limits

`RDD_TEST_CUDA=1 python -m unittest rdd.tests.test_dense_c2f -v` passed real edge-tile
keep100% versus FlashAttention BF16 tolerance and separate CFG cache checks.
Complete52-evaluation trajectory keep100% versus dense latent RMSE=0.014687:
numerical reduction differences are not zero. The generated keep100% video is saved.
Strict model key matching and strict DCP loading are enforced.

Initial three-repeat per-prompt timings were noisy. Use the separate five-round,
single-GPU interleaved timing for final speed conclusions. Videos and raw metrics:
[the results README](../README_RDD.md) and `../rdd_media/`.
50% retains much of the image but does not speed up this implementation. 12.5%
gains only about7% denoiser /6% denoiser+VAE latency reduction and causes severe
visible artifacts. This is not a demonstrated quality-preserving10–20% gain.
