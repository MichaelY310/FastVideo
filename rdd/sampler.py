# SPDX-License-Identifier: Apache-2.0
"""Extracted multiresolution FastWan DMD-3 sampler (NOT an RDD-trained model).

This is the old noise_filling trajectory only. No redraw aliases, alternate
noise transport, output reuse, training, or VAE/T5 reimplementation.
"""

import math
import time

import torch


def stage_shapes(full_shape, factors):
    """Factors are (SPATIAL, TEMPORAL); latent axes are (T,H,W)."""
    if len(factors) != 3 or factors[-1] != (1, 1):
        raise ValueError("DMD-3 needs three factors ending in (1,1)")
    if any(s <= 0 or t <= 0 for s, t in factors):
        raise ValueError("factors must be positive")
    if any(b[0] > a[0] or b[1] > a[1] for a, b in zip(factors, factors[1:], strict=False)):
        raise ValueError("resolution may only stay constant or increase")
    time_dim, height, width = full_shape
    shapes = [(math.ceil(time_dim / t), math.ceil(height / s), math.ceil(width / s)) for s, t in factors]
    if any(h % 2 or w % 2 for _, h, w in shapes):
        raise ValueError(f"Wan patch(1,2,2) needs even latent H/W at every stage: {shapes}")
    return shapes


def noise_filling(clean, target, sigma, generator, executor="native"):
    if executor == "triton_fused":
        from .kernels.noise_filling import fused_noise_filling_transition

        return fused_noise_filling_transition(clean, target, sigma, generator)
    if executor != "native":
        raise ValueError("boundary executor must be native or triton_fused")
    # Preserve the historical repeat-and-crop rule even for noninteger ratios.
    projected = clean
    for axis, size in enumerate(target, start=2):
        projected = projected.repeat_interleave(math.ceil(size / projected.shape[axis]), axis)
    projected = projected[:, :, :target[0], :target[1], :target[2]]
    noise = torch.randn(projected.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    return (1 - sigma) * projected + sigma * noise


@torch.inference_mode()
def sample(model, context, full_shape, factors, controller, seed=20260905, boundary="native"):
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.models.utils import pred_noise_to_pred_video
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    if not context.is_cuda or context.dtype != torch.bfloat16:
        raise ValueError("context must be CUDA BF16")
    if len(full_shape) != 5 or full_shape[1] != 16 or full_shape[0] != context.shape[0]:
        raise ValueError("full_shape must be [context_batch,16,T,H,W]")
    if context.ndim != 3 or context.shape[1] != 512:
        raise ValueError("use the official Wan T5 postprocessed context padded to 512 tokens")
    if tuple(model.patch_size) != (1, 2, 2):
        raise ValueError("only Wan patch(1,2,2) is supported")
    shapes = stage_shapes(full_shape[2:], factors)
    timesteps = (1000, 757, 522)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    builder = VideoSparseAttentionMetadataBuilder()
    batch = ForwardBatch(data_type="dummy")
    device = context.device
    generator = torch.Generator(device=device).manual_seed(seed)
    controller.reset()  # Compilation may be warm; routes must never leak across videos.
    clean, events = None, []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for stage, (shape, timestep) in enumerate(zip(shapes, timesteps, strict=False)):
            if clean is None:
                state = torch.randn((*full_shape[:2], *shape), device=device,
                                    dtype=torch.bfloat16, generator=generator)
            else:
                index = torch.argmin((scheduler.timesteps.double().to(device) - timestep).abs())
                sigma = scheduler.sigmas.double().to(device)[index].to(clean.dtype)
                state = noise_filling(clean, shape, sigma, generator, boundary)
            metadata = builder.build(current_timestep=stage, raw_latent_shape=shape,
                                     patch_size=(1, 2, 2), VSA_sparsity=1 - controller.density, device=device)
            t = torch.full((full_shape[0],), float(timestep), device=device, dtype=torch.float32)
            controller.begin_stage(stage)
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            with set_forward_context(current_timestep=stage, attn_metadata=metadata, forward_batch=batch):
                pred = model(hidden_states=state, encoder_hidden_states=context, timestep=t)
            end.record()
            events.append((begin, end))
            # Use UPSTREAM's double-precision conversion and final dtype cast.
            # Replacing this by a BF16 state-sigma*pred changes the trajectory.
            state_btchw, pred_btchw = (x.permute(0, 2, 1, 3, 4) for x in (state, pred))
            clean = pred_noise_to_pred_video(
                pred_btchw.flatten(0, 1), state_btchw.flatten(0, 1), t.long(), scheduler,
            ).unflatten(0, pred_btchw.shape[:2]).permute(0, 2, 1, 3, 4)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return clean, {
        "denoise_and_boundary_seconds": elapsed,
        "stage_model_seconds": [a.elapsed_time(b) / 1000 for a, b in events],
        "latent_shapes": shapes,
    }
