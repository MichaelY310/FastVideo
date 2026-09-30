# SPDX-License-Identifier: Apache-2.0
"""Re-decode existing RDD preview latents without loading/training a transformer."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("RDD_JOB_TOKEN") or not os.environ.get("RDD_DEADLINE_UNIX"):
        parser.error("use rdd.deadline for this GPU decoding job too")
    import torch
    import torch.distributed as dist
    from diffusers.utils import export_to_video
    from PIL import Image
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.moduleloader import load_module_from_path
    from fastvideo.train.models.wan.wan import WanModel

    cfg = load_run_config(args.config, overrides=[])
    cfg.training.distributed.num_gpus = 1
    cfg.training.distributed.hsdp_shard_dim = 1
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    vae = load_module_from_path(model_path=cfg.models["student"]["init_from"], module_type="vae",
                                training_config=cfg.training).eval()
    holder = SimpleNamespace(vae=vae)
    with torch.no_grad():
        for path in sorted(args.root.glob("step_*/*.pt")):
            latent = torch.load(path, map_location="cuda", weights_only=True)
            media = WanModel.decode_latents(holder, latent.permute(0, 2, 1, 3, 4))
            frames = media[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()
            # This API consumes float numpy frames in [0,1], not uint8.
            export_to_video(list(frames), str(path.with_suffix(".mp4")), fps=16)
            Image.fromarray((frames[len(frames) // 2] * 255).round().astype("uint8")).save(
                path.with_name(path.stem + "_frame.png"))
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text())
            meta["export_fix"] = "float [0,1] frames; redecoded original saved latent; no resampling"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"corrected {path.name}: {frames.shape}, range={frames.min()},{frames.max()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
