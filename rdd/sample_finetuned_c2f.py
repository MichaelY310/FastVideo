# SPDX-License-Identifier: Apache-2.0
"""Single-GPU paired RDD/C2F inference from a modular full-model DCP checkpoint.

Run independent prompt jobs on at most four physical GPUs. No optimizer, training,
or checkpoint deletion. Original FP32 master/BF16 compute conventions are kept.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="rdd/wan_finetune_300300400.yaml")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--settings", default="dense,c2f50,c2f30,c2f20,c2f12")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--timing-only-rounds", type=int, default=0,
                        help="counterbalanced paired benchmark, does not overwrite video/metrics files")
    args = parser.parse_args()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29831")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel, get_sp_group, get_world_group
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.moduleloader import load_module_from_path
    from rdd.finetune import RDDWanModel, sample_continuous
    from rdd.dense_c2f import DenseC2FController
    import pyarrow.parquet as pq
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
    from diffusers.utils import export_to_video

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free < 40 * 1024**3:
        raise RuntimeError("less than 40 GiB free; refusing to crowd an occupied GPU")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    cfg = load_run_config(args.config, overrides=[
        "--training.distributed.num_gpus", "1", "--training.distributed.hsdp_shard_dim", "1"])
    tc = cfg.training
    model_args = dict(cfg.models["student"])
    model_args.pop("_target_")
    model_args["trainable"] = False
    model = RDDWanModel(training_config=tc, **model_args)
    model.transformer.eval()
    state = {"roles.student.transformer": get_model_state_dict(model.transformer)}
    reader = dcp.FileSystemReader(str(Path(args.checkpoint) / "dcp"))
    stored = reader.read_metadata().state_dict_metadata
    prefix = "roles.student.transformer."
    expected_keys = set(state["roles.student.transformer"])
    stored_keys = {k[len(prefix):] for k in stored if k.startswith(prefix)}
    if expected_keys != stored_keys:
        raise RuntimeError(f"checkpoint keys mismatch: missing={expected_keys-stored_keys}, extra={stored_keys-expected_keys}")
    dcp.load(state, storage_reader=reader)
    set_model_state_dict(model.transformer, model_state_dict=state["roles.student.transformer"],
                         options=StateDictOptions(strict=True))
    del state, reader
    model.vae = load_module_from_path(model_path=tc.model_path, module_type="vae", training_config=tc)
    model.world_group, model.sp_group = get_world_group(), get_sp_group()
    model._init_timestep_mechanics()
    model.ensure_negative_conditioning()
    files = sorted(Path(cfg.method["validation_dir"]).glob("*.parquet"))
    rows = [row for file in files for row in pq.read_table(file).to_pylist()]
    raw = collate_rows_from_parquet_schema([rows[args.row]], pyarrow_schema_t2v,
                                          text_padding_length=512, cfg_rate=0.)
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    with torch.no_grad():
        prepared = model.prepare_batch(raw, generator=generator, latents_source="data")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    settings = {"dense": None, "c2f100": 1., "c2f50": .5, "c2f30": .3, "c2f20": .2, "c2f12": .125}
    results = {"checkpoint": str(Path(args.checkpoint).resolve()), "row": args.row, "seed": args.seed,
               "caption": raw.get("info_list"), "factors": model.rdd_path.factors,
               "data_time_boundaries": model.rdd_path.boundaries, "steps": [20, 15, 15],
               "cfg": 5., "transition": "noise_filling", "size": [832, 448, 29], "fps": 16,
               "gpu": torch.cuda.get_device_name(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "precision": "FP32 master/residual; official BF16 autocast", "settings": {}}
    if args.timing_only_rounds:
        selected = args.settings.split(",")
        measurements = []
        with torch.no_grad():
            # Each complete setting warmed first. Rotate order on every round
            # rather than putting all dense timings before all sparse timings.
            for round_index in range(-1, args.timing_only_rounds):
                offset = max(0, round_index) % len(selected)
                order = selected[offset:] + selected[:offset]
                for name in order:
                    density = settings[name]
                    controller = None if density is None else DenseC2FController(model.transformer, density=density)
                    try:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        latent, _ = sample_continuous(model, prepared, model.rdd_path,
                                                     seed=args.seed, route_controller=controller)
                        torch.cuda.synchronize()
                        denoise = time.perf_counter()-start
                        start = time.perf_counter()
                        media = model.decode_latents(latent.permute(0, 2, 1, 3, 4))
                        torch.cuda.synchronize()
                        vae = time.perf_counter()-start
                        record = {"round": round_index, "setting": name, "denoise": denoise, "vae": vae,
                                  "total": denoise+vae}
                        print(json.dumps(record), flush=True)
                        measurements.append(record)
                        (outdir / "interleaved_timing.json").write_text(
                            json.dumps({"metadata": results, "records": measurements}, indent=2), encoding="utf-8")
                        del media, latent
                    finally:
                        if controller is not None:
                            controller.close()
        dist.destroy_process_group()
        return
    dense_latent = None
    with torch.no_grad():
        for name in args.settings.split(","):
            density = settings[name]
            controller = None if density is None else DenseC2FController(model.transformer, density=density)
            timings, decode_times = [], []
            try:
                # First complete trajectory warms all three shapes, kernels and decoder.
                for rep in range(args.repeats + 1):
                    if controller is not None:
                        controller.reset()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    latent, calls = sample_continuous(model, prepared, model.rdd_path,
                                                      seed=args.seed, route_controller=controller)
                    torch.cuda.synchronize()
                    denoise = time.perf_counter() - start
                    start = time.perf_counter()
                    media = model.decode_latents(latent.permute(0, 2, 1, 3, 4))
                    torch.cuda.synchronize()
                    decode = time.perf_counter() - start
                    if rep:
                        timings.append(denoise)
                        decode_times.append(decode)
                    print(json.dumps({"setting": name, "rep": rep, "denoise": denoise, "vae": decode}), flush=True)
                if name == "dense":
                    dense_latent = latent.detach().clone()
                item = {"requested_coarse_topk_fraction": density, "actual_lifted_block_density":
                        {} if controller is None else controller.summary(),
                        "denoise_seconds": timings, "vae_seconds": decode_times,
                        "denoise_median": statistics.median(timings),
                        "total_median": statistics.median([a+b for a, b in zip(timings, decode_times, strict=False)]),
                        "model_calls_cond_plus_uncond": calls,
                        "latent_rmse_vs_dense": None if dense_latent is None else
                        float((latent-dense_latent).square().mean().sqrt()),
                        "warning": "RMSE is paired fidelity, not a perceptual quality score; no C2F finetuning."}
                torch.save(latent.cpu(), outdir / f"{name}.pt")
                pixels = media[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()
                export_to_video(list(pixels), str(outdir / f"{name}.mp4"), fps=16)
                from PIL import Image
                frames = [Image.fromarray((pixels[i]*255).round().astype("uint8")) for i in (0, 7, 14, 21, 28)]
                sheet = Image.new("RGB", (832*5, 448))
                for i, frame in enumerate(frames):
                    sheet.paste(frame, (832*i, 0))
                sheet.save(outdir / f"{name}.jpg")
                results["settings"][name] = item
                (outdir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            finally:
                if controller is not None:
                    controller.close()
    (outdir / "complete.json").write_text(json.dumps({"completed": True, "settings": list(results["settings"])}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
