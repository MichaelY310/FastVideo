# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity against the pinned official VSA equation, without weights."""

import os
import types
import unittest

import torch

from rdd.attention import C2FController


@unittest.skipUnless(os.environ.get("RDD_TEST_CUDA") == "1" and torch.cuda.is_available(), "opt-in upstream CUDA test")
class UpstreamTests(unittest.TestCase):
    def test_seed_and_keep_all_match_upstream(self):
        from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionImpl

        # forward() is independent of distributed initialization; no need to
        # load weights or instantiate a whole text/video pipeline for this test.
        impl = VideoSparseAttentionImpl.__new__(VideoSparseAttentionImpl)
        model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn1=types.SimpleNamespace(attn_impl=impl))])
        for density in (0.2, 1.0):
            controller = C2FController(model, mode="c2f_route", density=density)
            original = controller.originals[0][1]
            try:
                for stage, grid in enumerate([(1, 1, 2), (1, 2, 3)]):
                    blocks = grid[0] * grid[1] * grid[2]
                    q, k, v = [torch.randn(1, blocks * 64, 2, 128, device="cuda", dtype=torch.bfloat16)
                               for _ in range(3)]
                    gate = torch.randn_like(q).sigmoid()
                    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
                    sizes[-1] = 23
                    meta = types.SimpleNamespace(num_tiles=grid, variable_block_sizes=sizes,
                                                 VSA_sparsity=1 - density, total_seq_length=int(sizes.sum()))
                    controller.begin_stage(stage)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        actual = impl.forward(q, k, v, gate, meta)
                        expected = original(q, k, v, gate, meta)
                    # Seed always uses native VSA equation. At later stages
                    # only keep=100% makes C2F's chosen key set identical.
                    if stage == 0 or density == 1.0:
                        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
