# SPDX-License-Identifier: Apache-2.0
"""Only C2F routing: no step-output cache, CSBR, adapters, or training hooks.

Derived from the audited fastwan_native_attention.py implementation. We patch
only attn_impl.forward, not the Wan block forward or its parameters.
"""

import math
import types
from dataclasses import dataclass

import torch


@dataclass
class Route:
    grid: tuple[int, int, int]
    indices: torch.Tensor  # [batch, heads, query blocks, key capacity], int32
    counts: torch.Tensor  # [batch, heads, query blocks], int32


def capture_route(scores, grid, density):
    """Top-K plus a mandatory self block; self replaces, not adds, one slot."""
    blocks = math.prod(grid)
    keep = max(1, min(blocks, math.ceil(density * blocks)))
    indices = scores.topk(keep, dim=-1, sorted=True).indices
    own = torch.arange(blocks, device=scores.device).view(1, 1, blocks, 1)
    missing = ~(indices == own).any(dim=-1, keepdim=True)
    indices[..., -1:] = torch.where(missing, own, indices[..., -1:])
    indices = indices.to(torch.int32).contiguous()
    counts = torch.full(indices.shape[:-1], keep, dtype=torch.int32, device=scores.device)
    return Route(grid, indices.detach(), counts)


class C2FController:
    """Opt-in hook. Call reset() between videos; close() restores originals.

    c2f_route: inherited fine route + CURRENT coarse output; still computes
    current block scores (also captured for the next stage).
    c2f_route_only: inherited fine route only; drops the coarse-output branch
    AFTER the seed stage. This is an architecture ablation, not equivalent VSA.
    """

    def __init__(self, model, mode="vsa", density=0.2, executor="native", smallq_max_tokens=0):
        if mode not in {"vsa", "c2f_route", "c2f_route_only"}:
            raise ValueError("mode must be vsa, c2f_route, or c2f_route_only")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0,1]")
        if executor not in {"native", "triton_q16", "triton_q32"}:
            raise ValueError("unsupported sparse executor")
        if smallq_max_tokens < 0:
            raise ValueError("smallq_max_tokens must be nonnegative")
        self.mode, self.density = mode, density
        self.executor, self.smallq_max_tokens = executor, smallq_max_tokens
        self.originals = []
        self.reset()
        for layer, block in enumerate(model.blocks):
            impl = block.attn1.attn_impl
            if getattr(impl, "_minimal_c2f_hook", False):
                raise RuntimeError("close the previous controller before attaching another")
            original = impl.forward

            def forward(this, query, key, value, gate_compress, attn_metadata, _layer=layer):
                return self.forward(_layer, query, key, value, gate_compress, attn_metadata)

            self.originals.append((impl, original))
            impl.forward = types.MethodType(forward, impl)
            impl._minimal_c2f_hook = True

    def reset(self):
        self.stage = -1
        # Keep route storage alive for the whole asynchronous trajectory, as
        # the original controller did. Retaining only current/previous changed
        # route-only full-checkpoint results; component tests missed that.
        # sampler.sample synchronizes before the next reset/close.
        self._route_history = []
        self.previous = {}
        self.current = {}
        self.used_densities = []

    def begin_stage(self, stage):
        if stage != self.stage + 1:
            raise ValueError("stages must be consecutive; reset before a new video")
        self._route_history.append(self.current)
        self.previous, self.current = self.current, {}
        self.stage = stage

    def close(self):
        for impl, original in self.originals:
            impl.forward = original
            del impl._minimal_c2f_hook
        self.originals.clear()
        self.reset()

    def execute_sparse(self, q, k, v, route, sizes):
        if self.executor != "native" and 0 < q.shape[2] <= self.smallq_max_tokens:
            from .kernels.smallq import smallq_block_sparse_attention

            return smallq_block_sparse_attention(
                q, k, v, route.indices, route.counts, sizes,
                query_tile=16 if self.executor == "triton_q16" else 32,
            )
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices

        return block_sparse_attn_from_indices(q, k, v, route.indices, route.counts, sizes)[0]

    def forward(self, layer, query, key, value, gate_compress, metadata):
        if self.mode == "vsa":
            # Untouched upstream equation, including the learned coarse gate.
            return self.originals[layer][1](query, key, value, gate_compress, metadata)
        if self.stage < 0:
            raise RuntimeError("begin_stage() must precede model.forward()")
        from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean, fused_topk_mask
        from fastvideo_kernel.triton_kernels.index import map_to_index

        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        grid = tuple(int(n) for n in metadata.num_tiles)
        blocks = math.prod(grid)
        if q.shape[2] != blocks * 64:
            raise ValueError("this extraction supports FastWan VSA 4x4x4-token tiles only")
        sizes = metadata.variable_block_sizes.to(torch.int32).contiguous()
        source = self.previous.get(layer)
        if self.stage > 0 and source is None:
            raise RuntimeError("missing previous-stage route; refusing a silent online fallback")
        scores = None
        if source is None or self.mode == "c2f_route":
            qc, kc = (fused_block_mean(x, sizes, 64) for x in (q, k))
            scores = torch.matmul(qc, kc.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            self.current[layer] = capture_route(scores, grid, self.density)

        if source is None:
            # Preserve actual seed VSA top-k. Mandatory self is only part of
            # the separately captured route that will guide the next stage.
            keep = max(1, min(blocks, math.ceil((1 - float(metadata.VSA_sparsity)) * blocks)))
            indices, counts = map_to_index(fused_topk_mask(scores, keep))
            route = Route(grid, indices, counts)
        else:
            from .kernels.route_lift import lift_c2f_route

            indices, counts = lift_c2f_route(source.indices, source.counts, source.grid, grid)
            route = Route(grid, indices.contiguous(), counts.contiguous())
            if self.mode == "c2f_route_only":
                # Propagate the inherited route. No reranking by new sparse
                # probabilities and no current dense block QK at this stage.
                self.current[layer] = route
            self.used_densities.append(counts.float().mean() / blocks)

        out = self.execute_sparse(q, k, v, route, sizes)
        if scores is not None:
            vc = fused_block_mean(v, sizes, 64)
            coarse = torch.matmul(torch.softmax(scores, dim=-1), vc)
            coarse = coarse.unsqueeze(3).expand(-1, -1, -1, 64, -1).reshape_as(q)
            out = out + coarse * gate_compress.transpose(1, 2).contiguous()
        return out.transpose(1, 2).contiguous()
