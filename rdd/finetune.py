# SPDX-License-Identifier: Apache-2.0
"""RDD extension of FastVideo's modular Wan + supervised flow-matching trainer.

No upstream model files are changed. The model, optimizer, FSDP, mixed precision,
activation checkpointing, and data decoding remain FastVideo implementations.
"""

import copy
import faulthandler
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
from rdd.flow import RDDPath

logger = init_logger(__name__)


class DeadlineReached(Exception):
    """All ranks stop between optimizer steps, before the independent hard guard."""


class RDDWanModel(WanModel):
    def __init__(self, *, factors=((4, 1), (2, 1), (1, 1)), boundaries=(0., .4, .7, 1.), **kwargs):
        self.rdd_path = RDDPath(tuple(tuple(v) for v in factors), tuple(boundaries))
        self.rdd_stage = 0
        self.rdd_fixed_t = None
        self.rdd_target = None
        super().__init__(**kwargs)

    def _prepare_dit_inputs(self, batch, generator):
        # Parent prepare_batch already normalized the Wan VAE latent ONCE.
        # Crop to the declared spatial shape before exact integer pooling.
        h = self.training_config.data.num_height // 8
        w = self.training_config.data.num_width // 8
        raw = batch.latents
        if raw.shape[-2] < h or raw.shape[-1] < w:
            raise ValueError("dataset latent is smaller than configured training crop")
        top, left = (raw.shape[-2] - h) // 2, (raw.shape[-1] - w) // 2
        full = raw[..., top:top + h, left:left + w].float()
        clean = self.rdd_path.stage_clean(full, self.rdd_stage)
        if clean.shape[-1] % 2 or clean.shape[-2] % 2:
            raise ValueError("Wan patch(1,2,2) requires even stage spatial dimensions")
        a, b = self.rdd_path.boundaries[self.rdd_stage:self.rdd_stage + 2]
        t = torch.rand(clean.shape[0], device=clean.device, generator=generator) * (b - a) + a
        if self.rdd_fixed_t is not None:
            t.fill_(self.rdd_fixed_t)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator, dtype=torch.float32)
        if int(self.training_config.distributed.sp_size or 1) > 1:
            self.sp_group.broadcast(t, src=0)
            self.sp_group.broadcast(noise, src=0)
        noisy, self.rdd_target = self.rdd_path.noisy_and_target(clean, noise, t, self.rdd_stage)
        batch.latents = clean.permute(0, 2, 1, 3, 4)
        batch.noisy_model_input = noisy
        batch.noise = noise
        batch.sigmas = (1 - t).reshape(-1, 1, 1, 1, 1)
        batch.timesteps = (1 - t) * 1000  # Real noise-time units, NOT data-time units.
        batch.raw_latent_shape = clean.shape
        batch.conditional_dict = {"encoder_hidden_states": batch.encoder_hidden_states,
                                  "encoder_attention_mask": batch.encoder_attention_mask}
        batch.unconditional_dict = {"encoder_hidden_states": self.negative_prompt_embeds,
                                    "encoder_attention_mask": self.negative_prompt_attention_mask}
        return batch


class RDDFineTuneMethod(FineTuneMethod):
    """Full-parameter, ordinary supervised FM; no DMD, teacher, or adapter."""
    def __init__(self, *, cfg, role_models):
        super().__init__(cfg=cfg, role_models=role_models)
        if not isinstance(self.student, RDDWanModel):
            raise ValueError("RDDFineTuneMethod requires RDDWanModel")
        if self.training_config.model.precondition_outputs:
            raise ValueError("use velocity MSE; ordinary clean preconditioning is invalid with drift")
        self.last_iteration = 0
        self.preview_every = int(self.method_config.get("preview_every", 250))
        self.validation_dir = self.method_config.get("validation_dir")

    def single_train_step(self, batch, iteration):
        faulthandler.dump_traceback_later(120, repeat=True)
        deadline = float(os.environ.get("RDD_DEADLINE_UNIX", "inf"))
        must_stop = torch.tensor(time.time() >= deadline - 300, device=self.student.device, dtype=torch.int32)
        dist.all_reduce(must_stop, op=dist.ReduceOp.MAX)
        if must_stop.item():
            raise DeadlineReached("save and stop five minutes before hard deadline")
        # One stage per distributed batch. Stage probabilities equal interval
        # lengths, hence the total training-time distribution is uniform [0,1].
        selector = torch.rand((), generator=self.cuda_generator, device=self.student.device)
        dist.broadcast(selector, src=0)
        edges = self.student.rdd_path.boundaries
        stage = sum(float(selector) >= b for b in edges[1:-1])
        if self.method_config.get("smoke_all_stages", False):
            stage = (iteration - 1) % len(self.student.rdd_path.factors)
        self.student.rdd_stage = stage
        prepared = self.student.prepare_batch(batch, generator=self.cuda_generator, latents_source="data")
        pred = self.student.predict_noise(prepared.noisy_model_input.permute(0, 2, 1, 3, 4),
                                           prepared.timesteps, prepared, conditional=True, attn_kind=self._attn_kind)
        target = self.student.rdd_target.permute(0, 2, 1, 3, 4)
        loss = F.mse_loss(pred.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite RDD training loss")
        self.last_iteration = iteration
        metadata = prepared.attn_metadata_vsa if self._attn_kind == "vsa" else prepared.attn_metadata
        if iteration <= 3 or iteration % 10 == 0:
            logger.info("RDD step=%d stage=%d sigma_mean=%.4f mse=%.6f shape=%s peak_GiB=%.2f",
                        iteration, stage, prepared.sigmas.mean().item(), loss.item(),
                        tuple(prepared.noisy_model_input.shape), torch.cuda.max_memory_allocated() / 2**30)
        return {"total_loss": loss, "rdd_velocity_loss": loss}, {
            "_fv_backward": (prepared.timesteps, metadata)}, {"rdd_stage": stage}

    def _validation_raw(self):
        import pyarrow.parquet as pq
        from fastvideo.dataset.utils import collate_rows_from_parquet_schema
        from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
        files = sorted(Path(self.validation_dir).glob("*.parquet"))
        if not files:
            raise ValueError("held-out validation parquet required")
        rows = pq.read_table(files[0]).slice(0, 1).to_pylist()
        return collate_rows_from_parquet_schema(rows, pyarrow_schema_t2v, text_padding_length=512, cfg_rate=0.0)

    @torch.no_grad()
    def on_validation_begin(self, iteration=0):
        if not self.validation_dir or (iteration != 0 and iteration % self.preview_every != 0):
            return {}
        faulthandler.dump_traceback_later(600, repeat=True)
        model = self.student
        was_training = model.transformer.training
        model.transformer.eval()
        gen = torch.Generator(device=model.device).manual_seed(20260926)
        raw = self._validation_raw()
        results = {}
        try:
            for stage in range(len(model.rdd_path.factors)):
                model.rdd_stage = stage
                model.rdd_fixed_t = sum(model.rdd_path.boundaries[stage:stage + 2]) / 2
                prepared = model.prepare_batch(raw, generator=gen, latents_source="data")
                pred = model.predict_noise(prepared.noisy_model_input.permute(0, 2, 1, 3, 4),
                                           prepared.timesteps, prepared, conditional=True, attn_kind=self._attn_kind)
                loss = F.mse_loss(pred.float(), model.rdd_target.permute(0, 2, 1, 3, 4))
                results[f"validation/stage_{stage}_mse"] = float(loss)
            model.rdd_fixed_t = None
            outdir = Path(self.training_config.checkpoint.output_dir) / "previews" / f"step_{iteration:07d}"
            outdir.mkdir(parents=True, exist_ok=True)
            for full in ([True, False] if iteration == 0 else [False]):
                t0 = time.monotonic()
                latent, calls = sample_continuous(model, prepared, model.rdd_path, full=full,
                                                attn_kind=self._attn_kind)
                torch.cuda.synchronize()
                seconds = time.monotonic() - t0
                if dist.get_rank() == 0:
                    name = "base_full" if full else "rdd"
                    torch.save(latent.cpu(), outdir / f"{name}.pt")
                    # VAE is replicated/frozen; all ranks wait until rank 0 finishes decoding.
                    media = model.decode_latents(latent.permute(0, 2, 1, 3, 4))
                    from diffusers.utils import export_to_video
                    # Diffusers export_to_video multiplies numpy frames by 255.
                    # Keep floats in [0,1] here; uint8 would overflow on a second multiply.
                    pixels = media[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()
                    export_to_video(list(pixels), str(outdir / f"{name}.mp4"), fps=16)
                    (outdir / f"{name}.json").write_text(json.dumps({
                        "iteration": iteration, "name": name, "denoise_seconds": seconds,
                        "nfe_cond_plus_uncond": calls, "cfg": 5.0, "seed": 20260926,
                        "steps_coarse_middle_fine": [20, 15, 15],
                        "extra_boundary_evaluations": 0 if full else 2,
                        "factors": model.rdd_path.factors, "boundaries_data_time": model.rdd_path.boundaries,
                        "full": full, "transition": "noise_filling", "validation": results,
                        "caption": raw.get("info_list"),
                        "warning": "Pilot visual check, not a FID/VBench or fair speed measurement."
                    }, default=str, indent=2), encoding="utf-8")
                dist.barrier()
            logger.info("RDD held-out preview step=%d %s", iteration, results)
            if dist.get_rank() == 0:
                with (outdir.parent / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": iteration, **results}) + "\n")
        finally:
            model.rdd_fixed_t = None
            model.transformer.train(was_training)
        return results


@torch.no_grad()
def sample_continuous(model, prepared, path, *, full=False, attn_kind="dense", seed=20260926,
                      route_controller=None):
    """Euler integration without crossing boundaries. CFG5, total 50 solver steps.

    At each nonfinal endpoint evaluate w once more for analytic clean recovery.
    Count these calls explicitly. This is NOT the old three-step DMD sampler.
    """
    gen = torch.Generator(device=model.device).manual_seed(seed)
    full_shape = (1, 16, model.training_config.data.num_latent_t,
                  model.training_config.data.num_height // 8, model.training_config.data.num_width // 8)
    factors = [(1, 1)] if full else path.factors
    edges = [0., 1.] if full else path.boundaries
    counts = [50] if full else [20, 15, 15]
    first = factors[0]
    x = torch.randn((1, 16, full_shape[2] // first[1], full_shape[3] // first[0], full_shape[4] // first[0]),
                    device=model.device, generator=gen, dtype=torch.float32)
    calls = 0
    batch = copy.copy(prepared)

    def velocity(state, t, boundary=False):
        nonlocal calls
        times = torch.full((1,), (1 - t) * 1000, device=state.device)
        batch.timesteps = times
        batch.raw_latent_shape = state.shape
        model._build_attention_metadata(batch)
        batch.attn_metadata_vsa = batch.attn_metadata
        if route_controller is not None:
            route_controller.set_call(stage, state.shape[2:], "conditional", boundary)
        cond = model.predict_noise(state.permute(0, 2, 1, 3, 4), times, batch,
                                    conditional=True, attn_kind=attn_kind).permute(0, 2, 1, 3, 4).float()
        if route_controller is not None:
            route_controller.set_call(stage, state.shape[2:], "unconditional", boundary)
        uncond = model.predict_noise(state.permute(0, 2, 1, 3, 4), times, batch,
                                      conditional=False, attn_kind=attn_kind).permute(0, 2, 1, 3, 4).float()
        calls += 2
        return uncond + 5 * (cond - uncond)

    for stage, count in enumerate(counts):
        a, b = edges[stage:stage + 2]
        dt = (b - a) / count
        for i in range(count):
            x = x - dt * velocity(x, a + i * dt)
        if stage + 1 < len(counts):
            clean = path.clean_from_velocity(x, velocity(x, b, boundary=True), torch.tensor([b], device=x.device), stage)
            x = path.transition(clean, full_shape, stage + 1, gen)
    return x, calls
