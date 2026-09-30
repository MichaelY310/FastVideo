# SPDX-License-Identifier: Apache-2.0
"""Inference-only VSA sparse attention with smaller query work tiles.

FastVideo's portable Triton fallback launches one program for every 64-token
VSA query block and (batch, head).  Very coarse RDD stages can expose only a
handful of such blocks, so an RTX PRO 6000 (188 SMs) receives too few programs.

This kernel preserves the *same* 64-token VSA route and 64-token key blocks,
but splits each routed query block into 16- or 32-token work tiles.  It is
therefore an executor change, not a sparsity or routing change.  It is forward
only and intended for inference; training continues to use FastVideo's native
operator.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _smallq_sparse_fwd(
    query,
    key,
    value,
    output,
    q2k_index,
    q2k_num,
    variable_block_sizes,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    heads: tl.constexpr,
    q_tokens: tl.constexpr,
    kv_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    fine_q_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads

    # Route metadata remains one row per original 64-token VSA block.  Several
    # smaller query programs intentionally read the same compact route row.
    query_start = fine_q_block * block_m
    route_q_block = query_start // block_n
    route_q_blocks = q_tokens // block_n
    route_row = (batch_head * route_q_blocks) + route_q_block
    kv_count = tl.load(q2k_num + route_row)
    route_ptr = q2k_index + route_row * max_kv_blocks

    q_base = batch * stride_qz + head * stride_qh
    k_base = batch * stride_kz + head * stride_kh
    v_base = batch * stride_vz + head * stride_vh
    o_base = batch * stride_oz + head * stride_oh

    q_ptr = tl.make_block_ptr(
        base=query + q_base,
        shape=(q_tokens, head_dim),
        strides=(stride_qm, stride_qd),
        offsets=(query_start, 0),
        block_shape=(block_m, head_dim),
        order=(1, 0),
    )
    k_base_ptr = tl.make_block_ptr(
        base=key + k_base,
        shape=(head_dim, kv_tokens),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(head_dim, block_n),
        order=(0, 1),
    )
    v_base_ptr = tl.make_block_ptr(
        base=value + v_base,
        shape=(kv_tokens, head_dim),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(block_n, head_dim),
        order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        base=output + o_base,
        shape=(q_tokens, head_dim),
        strides=(stride_om, stride_od),
        offsets=(query_start, 0),
        block_shape=(block_m, head_dim),
        order=(1, 0),
    )

    query_tile = tl.load(q_ptr)
    running_max = tl.full((block_m,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, head_dim), tl.float32)
    log2_scale = sm_scale * 1.4426950408889634

    for slot in range(0, kv_count):
        kv_block = tl.load(route_ptr + slot).to(tl.int32)
        valid_columns = tl.load(variable_block_sizes + kv_block)
        k_ptr = tl.advance(k_base_ptr, (0, kv_block * block_n))
        v_ptr = tl.advance(v_base_ptr, (kv_block * block_n, 0))
        key_tile = tl.load(k_ptr)
        logits = tl.dot(query_tile, key_tile)
        key_mask = tl.arange(0, block_n) < valid_columns
        logits = tl.where(key_mask[None, :], logits, -float("inf"))

        tile_max = tl.maximum(running_max, tl.max(logits, axis=1) * log2_scale)
        probabilities = tl.math.exp2(logits * log2_scale - tile_max[:, None])
        tile_sum = tl.sum(probabilities, axis=1)
        correction = tl.math.exp2(running_max - tile_max)
        running_sum = running_sum * correction + tile_sum
        accumulator *= correction[:, None]
        value_tile = tl.load(v_ptr)
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value_tile, accumulator)
        running_max = tile_max

    accumulator /= running_sum[:, None]
    tl.store(o_ptr, accumulator.to(output.type.element_ty))


def smallq_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    query_tile: int = 16,
) -> torch.Tensor:
    """Run the supplied VSA route with 16- or 32-token query work tiles."""

    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        raise RuntimeError("smallq_block_sparse_attention is inference-only")
    if query_tile not in (16, 32):
        raise ValueError("query_tile must be 16 or 32")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must use [batch, heads, tokens, dim]")
    batch, heads, q_tokens, head_dim = query.shape
    kv_tokens = key.shape[2]
    if q_tokens % 64 or kv_tokens % 64:
        raise ValueError("VSA padded query/key lengths must be multiples of 64")
    if q2k_num.shape[-1] != q_tokens // 64:
        raise ValueError("q2k_num must contain one route row per 64 query tokens")
    if variable_block_sizes.numel() != kv_tokens // 64:
        raise ValueError("variable_block_sizes must contain one value per 64 key tokens")

    output = torch.empty_like(query)
    grid = (triton.cdiv(q_tokens, query_tile), batch * heads)
    _smallq_sparse_fwd[grid](
        query,
        key,
        value,
        output,
        q2k_index,
        q2k_num,
        variable_block_sizes,
        1.0 / math.sqrt(head_dim),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads=heads,
        q_tokens=q_tokens,
        kv_tokens=kv_tokens,
        head_dim=head_dim,
        max_kv_blocks=q2k_index.shape[-1],
        block_m=query_tile,
        block_n=64,
        num_warps=4,
        num_stages=3,
    )
    return output
