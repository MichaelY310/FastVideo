# SPDX-License-Identifier: Apache-2.0
"""Fused GPU route lifting for RDD coarse-to-fine sparse attention.

Each Triton program owns one (batch, head, fine-query-block) row.  It maps the
fine query to its coarse parent, reads that parent's selected coarse key
blocks, expands every selected key to its fine children, and writes the compact
q2k list consumed directly by FastVideo's block-sparse attention kernel.

No attention scores and no dense QxK route matrix are formed.  The kernel also
supports ceil-created, non-uniform grids such as (1,2,4)->(1,4,7).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _c2f_lift_route_kernel(
    coarse_indices,
    coarse_counts,
    fine_indices,
    fine_counts,
    heads: tl.constexpr,
    coarse_q: tl.constexpr,
    fine_q: tl.constexpr,
    coarse_t: tl.constexpr,
    coarse_h: tl.constexpr,
    coarse_w: tl.constexpr,
    fine_t: tl.constexpr,
    fine_h: tl.constexpr,
    fine_w: tl.constexpr,
    max_coarse_keep: tl.constexpr,
    max_children: tl.constexpr,
    max_fine_keep: tl.constexpr,
):
    row = tl.program_id(0)
    q_fine = row % fine_q
    bh = row // fine_q

    q_t = q_fine // (fine_h * fine_w)
    q_h = (q_fine // fine_w) % fine_h
    q_w = q_fine % fine_w
    parent_t = q_t * coarse_t // fine_t
    parent_h = q_h * coarse_h // fine_h
    parent_w = q_w * coarse_w // fine_w
    parent_q = (parent_t * coarse_h + parent_h) * coarse_w + parent_w

    coarse_row = bh * coarse_q + parent_q
    selected_count = tl.load(coarse_counts + coarse_row).to(tl.int32)
    coarse_base = coarse_row * max_coarse_keep
    fine_base = row * max_fine_keep
    cursor = tl.zeros([1], dtype=tl.int32)

    for slot in tl.static_range(0, max_coarse_keep):
        selected = slot < selected_count
        parent_key = tl.load(
            coarse_indices + coarse_base + slot,
            mask=selected,
            other=0,
        ).to(tl.int32)
        key_t = parent_key // (coarse_h * coarse_w)
        key_h = (parent_key // coarse_w) % coarse_h
        key_w = parent_key % coarse_w

        start_t = (key_t * fine_t + coarse_t - 1) // coarse_t
        start_h = (key_h * fine_h + coarse_h - 1) // coarse_h
        start_w = (key_w * fine_w + coarse_w - 1) // coarse_w
        end_t = ((key_t + 1) * fine_t + coarse_t - 1) // coarse_t
        end_h = ((key_h + 1) * fine_h + coarse_h - 1) // coarse_h
        end_w = ((key_w + 1) * fine_w + coarse_w - 1) // coarse_w
        size_t = end_t - start_t
        size_h = end_h - start_h
        size_w = end_w - start_w
        child_count = size_t * size_h * size_w

        for child in tl.static_range(0, max_children):
            valid = selected & (child < child_count)
            dt = child // (size_h * size_w)
            dh = (child // size_w) % size_h
            dw = child % size_w
            fine_key = (
                ((start_t + dt) * fine_h + (start_h + dh)) * fine_w
                + start_w
                + dw
            )
            tl.store(
                fine_indices + fine_base + cursor + child,
                fine_key,
                mask=valid,
            )
        cursor += tl.where(selected, child_count, 0)

    tl.store(fine_counts + row, tl.sum(cursor, axis=0))


def lift_c2f_route(
    coarse_indices: torch.Tensor,
    coarse_counts: torch.Tensor,
    coarse_grid: tuple[int, int, int],
    fine_grid: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand compact coarse q2k routes into compact fine q2k routes."""
    if not coarse_indices.is_cuda or not coarse_counts.is_cuda:
        raise ValueError("C2F route lifting requires CUDA tensors")
    if coarse_indices.dtype != torch.int32 or coarse_counts.dtype != torch.int32:
        raise ValueError("C2F route indices/counts must be int32")
    if coarse_indices.ndim != 4 or coarse_counts.shape != coarse_indices.shape[:-1]:
        raise ValueError("expected indices [B,H,Q,K] and counts [B,H,Q]")
    if any(fine < coarse for coarse, fine in zip(coarse_grid, fine_grid, strict=False)):
        raise ValueError(f"fine grid must dominate coarse grid: {coarse_grid}->{fine_grid}")
    batch, heads, coarse_q, max_coarse_keep = coarse_indices.shape
    if coarse_q != math.prod(coarse_grid):
        raise ValueError("coarse route rows do not match coarse_grid")
    fine_q = math.prod(fine_grid)
    max_children = math.prod(
        math.ceil(fine / coarse)
        for coarse, fine in zip(coarse_grid, fine_grid, strict=False)
    )
    max_fine_keep = max_coarse_keep * max_children
    fine_indices = torch.empty(
        (batch, heads, fine_q, max_fine_keep),
        dtype=torch.int32,
        device=coarse_indices.device,
    )
    fine_counts = torch.empty(
        (batch, heads, fine_q),
        dtype=torch.int32,
        device=coarse_indices.device,
    )
    _c2f_lift_route_kernel[(batch * heads * fine_q,)](
        coarse_indices.contiguous(),
        coarse_counts.contiguous(),
        fine_indices,
        fine_counts,
        heads=heads,
        coarse_q=coarse_q,
        fine_q=fine_q,
        coarse_t=coarse_grid[0],
        coarse_h=coarse_grid[1],
        coarse_w=coarse_grid[2],
        fine_t=fine_grid[0],
        fine_h=fine_grid[1],
        fine_w=fine_grid[2],
        max_coarse_keep=max_coarse_keep,
        max_children=max_children,
        max_fine_keep=max_fine_keep,
        num_warps=1,
    )
    return fine_indices, fine_counts
