# SPDX-License-Identifier: Apache-2.0
"""Fused RDD nearest-neighbour upsample + noise-filling blend.

The Gaussian draw remains PyTorch's generator-backed operation so seeds and
noise samples stay identical.  A single Triton kernel then reads the coarse
clean tensor through its real strides and overwrites that Gaussian buffer with
the fine state.  This removes three repeat_interleave launches, the materialized
upsampled tensor, and separate multiply/add temporaries at every stage boundary.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _noise_filling_fused(
    clean,
    state,
    sigma_ptr,
    clean_stride_b,
    clean_stride_c,
    clean_stride_t,
    clean_stride_h,
    clean_stride_w,
    channels: tl.constexpr,
    source_t: tl.constexpr,
    source_h: tl.constexpr,
    source_w: tl.constexpr,
    target_t: tl.constexpr,
    target_h: tl.constexpr,
    target_w: tl.constexpr,
    ratio_t: tl.constexpr,
    ratio_h: tl.constexpr,
    ratio_w: tl.constexpr,
    total: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < total
    index = offsets
    target_w_index = index % target_w
    index //= target_w
    target_h_index = index % target_h
    index //= target_h
    target_t_index = index % target_t
    index //= target_t
    channel = index % channels
    batch = index // channels

    source_w_index = tl.minimum(target_w_index // ratio_w, source_w - 1)
    source_h_index = tl.minimum(target_h_index // ratio_h, source_h - 1)
    source_t_index = tl.minimum(target_t_index // ratio_t, source_t - 1)
    clean_offset = (
        batch * clean_stride_b
        + channel * clean_stride_c
        + source_t_index * clean_stride_t
        + source_h_index * clean_stride_h
        + source_w_index * clean_stride_w
    )
    clean_value = tl.load(clean + clean_offset, mask=valid, other=0.0)
    noise_value = tl.load(state + offsets, mask=valid, other=0.0)
    # Match eager BF16 evaluation exactly: `(1-sigma)`, each multiply, and the
    # final add are materialized as BF16 tensors in the historical expression.
    # Keeping the full expression in FP32 and rounding once is locally small
    # but is amplified by later denoising calls.
    sigma = tl.load(sigma_ptr).to(tl.bfloat16)
    clean_weight = (1.0 - sigma).to(tl.bfloat16)
    clean_term = (clean_weight * clean_value).to(tl.bfloat16)
    noise_term = (sigma * noise_value).to(tl.bfloat16)
    mixed = (clean_term + noise_term).to(tl.bfloat16)
    tl.store(state + offsets, mixed, mask=valid)


def fused_noise_filling_transition(
    clean: torch.Tensor,
    target: tuple[int, int, int],
    sigma: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return `(1-sigma)*nearest(clean)+sigma*N(0,I)` without nearest materialization."""

    if clean.ndim != 5:
        raise ValueError("clean must be [batch, channels, time, height, width]")
    if not clean.is_cuda or clean.dtype != torch.bfloat16:
        raise ValueError("this inference kernel requires CUDA BF16 clean")
    if sigma.device != clean.device or sigma.dtype != clean.dtype or sigma.numel() != 1:
        raise ValueError("sigma must be one BF16 value on the same CUDA device")
    batch, channels, source_t, source_h, source_w = clean.shape
    target_t, target_h, target_w = target
    ratio_t = math.ceil(target_t / source_t)
    ratio_h = math.ceil(target_h / source_h)
    ratio_w = math.ceil(target_w / source_w)
    state = torch.randn(
        (batch, channels, target_t, target_h, target_w),
        generator=generator,
        device=clean.device,
        dtype=clean.dtype,
    )
    total = state.numel()
    block = 256
    _noise_filling_fused[(triton.cdiv(total, block),)](
        clean,
        state,
        sigma,
        clean.stride(0),
        clean.stride(1),
        clean.stride(2),
        clean.stride(3),
        clean.stride(4),
        channels=channels,
        source_t=source_t,
        source_h=source_h,
        source_w=source_w,
        target_t=target_t,
        target_h=target_h,
        target_w=target_w,
        ratio_t=ratio_t,
        ratio_h=ratio_h,
        ratio_w=ratio_w,
        total=total,
        block=block,
        num_warps=4,
        num_stages=1,
        # Preserve the separate eager multiply/add rounding boundaries.
        enable_fp_fusion=False,
    )
    return state
