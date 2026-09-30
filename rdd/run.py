# SPDX-License-Identifier: Apache-2.0
"""Minimal latent-sampling entrypoint; T5/VAE remain upstream responsibilities."""

import argparse
import json
import os
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", required=True, help="FastWan checkpoint's transformer directory")
    parser.add_argument("--context", required=True, help="Trusted .pt Tensor [1,512,D], official T5 postprocessing")
    parser.add_argument("--latent-shape", default="8,56,104", help="Final T,H,W; NOT pixel dimensions")
    parser.add_argument("--factors", default="4,1;2,1;1,1", help="Three spatial,temporal pairs")
    parser.add_argument("--mode", choices=["vsa", "c2f_route", "c2f_route_only"], default="c2f_route")
    parser.add_argument("--keep", type=float, default=0.2)
    parser.add_argument("--executor", choices=["native", "triton_q16", "triton_q32"], default="native")
    parser.add_argument("--smallq-max-tokens", type=int, default=0, help="0 disables SmallQ")
    parser.add_argument("--boundary", choices=["native", "triton_fused"], default="native")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("rdd_output.pt"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup >= 0 and repeat >= 1 are required")

    # Wan chooses its block class at import time; set this BEFORE any import.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    import torch
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader

    from .attention import C2FController
    from .sampler import sample, stage_shapes

    device = torch.device("cuda:0")  # Select ONE physical GPU using CUDA_VISIBLE_DEVICES.
    torch.cuda.set_device(device)
    shape = tuple(int(x) for x in args.latent_shape.split(","))
    factors = [tuple(int(x) for x in pair.split(",")) for pair in args.factors.split(";")]
    stage_shapes(shape, factors)  # Fail before loading weights on invalid shapes.
    context = torch.load(args.context, map_location="cpu", weights_only=True).to(device, torch.bfloat16)
    if context.ndim != 3 or context.shape[:2] != (1, 512):
        raise ValueError("minimal CLI requires one padded context [1,512,D]")
    for key, value in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29619",
                       "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(key, value)
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("minimal extraction supports single-GPU inference only")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    controller = None
    try:
        path = str(Path(args.transformer).resolve())
        config = FastVideoArgs(
            model_path=path, num_gpus=1, dit_cpu_offload=False, dit_layerwise_offload=False,
            pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(), dit_precision="bf16"),
        )
        config.device = device
        model = TransformerLoader().load(path, config).to(dtype=torch.bfloat16).eval()
        if type(model.blocks[0]).__name__ != "WanTransformerBlock_VSA":
            raise RuntimeError("checkpoint must load as native WanTransformerBlock_VSA")
        if not hasattr(model.blocks[0], "to_gate_compress"):
            raise RuntimeError("VSA's learned coarse-output gate must not be dropped")
        controller = C2FController(model, args.mode, args.keep, args.executor, args.smallq_max_tokens)
        records = []
        for iteration in range(args.warmup + args.repeat):
            latent, timing = sample(model, context, (1, 16, *shape), factors, controller, args.seed, args.boundary)
            print(f"run={iteration} {timing}", flush=True)
            if iteration >= args.warmup:
                records.append(timing)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.cpu(), args.output)
        metrics = {
            "warning": "Released FastWan is not RDD-trained; this is a multiresolution sampling sanity path.",
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "dmd_timesteps": [1000, 757, 522], "transition": "noise_filling", "records": records,
            "median_denoise_seconds": statistics.median(r["denoise_and_boundary_seconds"] for r in records),
            "vae_included": False,
        }
        args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    finally:
        if controller is not None:
            controller.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
