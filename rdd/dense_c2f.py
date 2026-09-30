# SPDX-License-Identifier: Apache-2.0
"""Training-free C2F routes for the NON-VSA Wan checkpoint.

Only post-RoPE self-attention is replaced. No learned VSA gates, output cache,
text-attention changes, or weights are introduced. Routes are independent per
layer and CFG branch. Capture pooled Q/K scores once at each stage endpoint;
reuse their lifted Top-K routes throughout the next stage, with CURRENT Q/K/V.
"""

import math
import types

import torch

from rdd.attention import Route, capture_route
from rdd.kernels.route_lift import lift_c2f_route


class DenseC2FController:
    def __init__(self, transformer, density=0.3, dense_layers=0):
        if not 0 < density <= 1:
            raise ValueError("density must be in (0,1]")
        self.density = density
        self.dense_layers = dense_layers
        self.originals = []
        self.reset()
        for layer, block in enumerate(transformer.blocks):
            impl = block.attn1.attn_impl
            if getattr(impl, "_dense_c2f_hook", False):
                raise RuntimeError("controller already attached")
            original = impl.forward

            def forward(this, q, k, v, metadata, _layer=layer):
                return self.forward(_layer, q, k, v, metadata)

            self.originals.append((impl, original))
            impl.forward = types.MethodType(forward, impl)
            impl._dense_c2f_hook = True

    def reset(self):
        self.sources = {}
        self.lifted = {}
        self.metadata = {}
        self.stats = {}
        self.stage = -1

    def set_call(self, stage, latent_shape, branch, boundary=False):
        self.stage, self.branch, self.boundary = stage, branch, boundary
        self.latent_shape = tuple(latent_shape)

    def _metadata(self, device):
        from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
        key = (self.latent_shape, device)
        if key not in self.metadata:
            self.metadata[key] = VideoSparseAttentionMetadataBuilder().build(
                current_timestep=0, raw_latent_shape=self.latent_shape,
                patch_size=(1, 2, 2), VSA_sparsity=1-self.density, device=device)
        return self.metadata[key]

    @staticmethod
    def tile(x, meta):
        # Valid tokens are packed at the START of each tile; kernel masks by
        # sizes, not by a rectangular padding mask. Use official permutation.
        out = x.new_zeros((x.shape[0], len(meta.variable_block_sizes)*64, *x.shape[2:]))
        out[:, meta.non_pad_index] = x[:, meta.tile_partition_indices]
        return out.transpose(1, 2).contiguous()

    def forward(self, layer, q, k, v, metadata):
        if self.stage < 0:
            raise RuntimeError("set_call before forward")
        original = self.originals[layer][1]
        if layer < self.dense_layers or layer >= len(self.originals)-self.dense_layers:
            return original(q, k, v, metadata)
        if self.stage == 0 and not self.boundary:
            return original(q, k, v, metadata)
        meta = self._metadata(q.device)
        grid = tuple(meta.num_tiles)
        sizes = meta.variable_block_sizes.to(torch.int32).contiguous()
        qt, kt = self.tile(q, meta), self.tile(k, meta)
        if self.boundary:
            from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean
            qc, kc = (fused_block_mean(x, sizes, 64) for x in (qt, kt))
            # This is a block-level proxy, NOT the cached full attention matrix.
            scores = (qc.float() @ kc.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
            self.sources[(self.stage, self.branch, layer)] = capture_route(scores, grid, self.density)
        if self.stage == 0:
            return original(q, k, v, metadata)
        route_key = (self.stage, self.branch, layer)
        if route_key not in self.lifted:
            source = self.sources[(self.stage-1, self.branch, layer)]
            indices, counts = lift_c2f_route(source.indices, source.counts, source.grid, grid)
            self.lifted[route_key] = Route(grid, indices, counts)
        route = self.lifted[route_key]
        vt = self.tile(v, meta)
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices
        out = block_sparse_attn_from_indices(qt, kt, vt, route.indices, route.counts, sizes)[0]
        if route_key not in self.stats:
            self.stats[route_key] = (route.counts.float().mean() / math.prod(grid)).detach()
        return out.transpose(1, 2)[:, meta.untile_combined_index].contiguous()

    def summary(self):
        return {f"stage{s}": float(torch.stack([v for (st, _, _), v in self.stats.items()
                                               if st == s]).mean())
                for s in (1, 2) if any(st == s for st, _, _ in self.stats)}

    def close(self):
        torch.cuda.synchronize()
        for impl, original in self.originals:
            impl.forward = original
            del impl._dense_c2f_hook
        self.originals.clear()
        self.reset()
