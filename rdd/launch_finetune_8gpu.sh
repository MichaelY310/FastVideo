#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-${PWD}/rdd/runs/wan_base}"
# Required absolute deadline, no automatic next-day rollover.
DEADLINE="${DEADLINE:?Set a future absolute ISO-8601 deadline including timezone}"
TOKEN="${TOKEN:?Set unique run token (at least 20 characters)}"
mkdir -p "${RUN_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PWD}:${PWD}/fastvideo-kernel/python${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export HF_HOME="${RUN_ROOT}/hf_cache"
export TRITON_CACHE_DIR="${RUN_ROOT}/triton_cache"
export WANDB_MODE=disabled
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
exec "${PYTHON}" -m rdd.deadline --deadline "${DEADLINE}" --token "${TOKEN}" \
  --status "${RUN_ROOT}/guard_${TOKEN}.json" --disk-root "${RUN_ROOT}" \
  --min-free-gib 200 --grace-seconds 120 --check-foreign-gpus -- \
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m rdd.train --config rdd/wan_finetune_300300400.yaml \
  --models.student.init_from "${RUN_ROOT}/assets/Wan2.1-T2V-1.3B-Diffusers" \
  --method.validation_dir "${RUN_ROOT}/assets/mixkit/validation" \
  --training.data.data_path "${RUN_ROOT}/assets/mixkit/train" \
  --training.checkpoint.output_dir "${RUN_ROOT}/train" "$@"
