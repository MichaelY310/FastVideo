# SPDX-License-Identifier: Apache-2.0
"""Download a NON-distilled Wan checkpoint and a deterministic real-video pilot.

Uses FastVideo's already VAE/T5-encoded Mixkit data. Does not synthesize training
targets with a DMD model. Sources and immutable revisions are recorded.
"""

import argparse
import json
from pathlib import Path
import random
import shutil


def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--train-shards", type=int, default=48)
    p.add_argument("--validation-shards", type=int, default=4)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 250 * 2**30:
        raise RuntimeError("keep at least 250 GiB free before preparing assets")
    api = HfApi()
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    dataset_id = "FastVideo/mixkit_filtered_6k_wan1.3_t2v"
    model_rev = api.model_info(model_id).sha
    data_rev = api.dataset_info(dataset_id).sha
    model = args.root / "Wan2.1-T2V-1.3B-Diffusers"
    snapshot_download(model_id, revision=model_rev, local_dir=model,
                      ignore_patterns=["examples/*", "*.mp4", "*.png"], max_workers=4)
    scheduler = json.loads((model / "scheduler/scheduler_config.json").read_text())
    if scheduler["num_train_timesteps"] != 1000:
        raise RuntimeError("expected the original 1000-timestep Wan model")
    files = sorted(x.path for x in api.list_repo_tree(dataset_id, repo_type="dataset", recursive=True,
                                                     revision=data_rev) if x.path.endswith(".parquet"))
    random.Random(20260926).shuffle(files)
    partitions = {"validation": files[:args.validation_shards],
                  "train": files[args.validation_shards:args.validation_shards + args.train_shards]}
    for split, selected in partitions.items():
        for file in selected:
            path = hf_hub_download(dataset_id, file, repo_type="dataset", revision=data_rev,
                                   local_dir=args.root / "mixkit_source")
            dest = args.root / "mixkit" / split / Path(file).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.hardlink_to(path)
            print(f"{split}: {dest.name}", flush=True)
    manifest = {"model_id": model_id, "model_revision": model_rev,
                "dataset_id": dataset_id, "dataset_revision": data_rev, "shards": partitions,
                "num_train_timesteps": 1000, "distilled": False,
                "warning": "Small real-video finetuning pilot; not a full benchmark dataset."}
    (args.root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
