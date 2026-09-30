# SPDX-License-Identifier: Apache-2.0
"""Small tests only: CPU by default; RDD_TEST_CUDA=1 enables GPU components."""

import math
import os
import types
import unittest

import torch

from rdd.attention import C2FController, capture_route
from rdd.sampler import noise_filling, stage_shapes


class CoreTests(unittest.TestCase):
    def test_factors_and_invalid_shapes(self):
        self.assertEqual(stage_shapes((8, 56, 104), [(4, 4), (2, 2), (1, 1)]),
                         [(2, 14, 26), (4, 28, 52), (8, 56, 104)])
        for factors in [[(0, 1), (2, 1), (1, 1)], [(2, 1), (4, 1), (1, 1)], [(1, 1)]]:
            with self.assertRaises(ValueError):
                stage_shapes((8, 56, 104), factors)

    def test_topk_density_and_self(self):
        scores = torch.randn(2, 3, 8, 8)
        route = capture_route(scores, (1, 2, 4), 0.2)
        self.assertEqual(route.indices.shape, (2, 3, 8, 2))  # 20% rounds to 25%.
        own = torch.arange(8).view(1, 1, 8, 1)
        self.assertTrue(((route.indices == own).any(-1)).all())

    def test_noise_filling_identity_seed(self):
        clean = torch.randn(1, 2, 2, 4, 6).to(torch.bfloat16)
        sigma = torch.tensor(0.4, dtype=torch.bfloat16)
        actual = noise_filling(clean, (4, 8, 12), sigma, torch.Generator().manual_seed(42))
        lifted = clean.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
        fresh = torch.randn(lifted.shape, dtype=clean.dtype, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(actual, (1 - sigma) * lifted + sigma * fresh))

    def test_hook_restoration_and_reset(self):
        class Backend:
            def forward(self, *args):
                return args[0]
        impl = Backend()
        original = impl.forward
        model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn1=types.SimpleNamespace(attn_impl=impl))])
        controller = C2FController(model)
        marker = torch.tensor(1)
        self.assertIs(impl.forward(marker, None, None, None, None), marker)
        controller.begin_stage(0)
        controller.current[0] = "sentinel"
        controller.reset()
        self.assertFalse(controller.current)
        controller.close()
        self.assertEqual(impl.forward, original)

    def test_route_storage_lives_until_trajectory_reset(self):
        controller = C2FController(types.SimpleNamespace(blocks=[]))
        controller.begin_stage(0)
        seed_routes = controller.current
        seed_routes[0] = capture_route(torch.randn(1, 1, 2, 2), (1, 1, 2), 0.5)
        controller.begin_stage(1)
        controller.begin_stage(2)
        self.assertIs(controller._route_history[1], seed_routes)
        controller.reset()
        self.assertEqual(controller._route_history, [])


@unittest.skipUnless(os.environ.get("RDD_TEST_CUDA") == "1" and torch.cuda.is_available(), "opt-in CUDA tests")
class KernelTests(unittest.TestCase):
    def test_route_lift_against_independent_coordinate_enumeration(self):
        from rdd.kernels.route_lift import lift_c2f_route
        coarse = (1, 2, 4)
        route = capture_route(torch.randn(2, 2, 8, 8, device="cuda"), coarse, 0.2)
        for fine in [(1, 4, 7), (2, 7, 13)]:
            indices, counts = lift_c2f_route(route.indices, route.counts, coarse, fine)
            parents = []
            for t in range(fine[0]):
                for h in range(fine[1]):
                    for w in range(fine[2]):
                        parents.append(((t * coarse[0] // fine[0]) * coarse[1]
                                        + h * coarse[1] // fine[1]) * coarse[2] + w * coarse[2] // fine[2])
            src, src_counts = route.indices.cpu(), route.counts.cpu()
            got, got_counts = indices.cpu(), counts.cpu()
            for batch in range(2):
                for head in range(2):
                    for q, parent in enumerate(parents):
                        chosen = src[batch, head, parent, :src_counts[batch, head, parent]].tolist()
                        expected = {k for k, p in enumerate(parents) if p in chosen}
                        valid = got[batch, head, q, :got_counts[batch, head, q]].tolist()
                        self.assertEqual(set(valid), expected)
                        self.assertEqual(len(valid), len(expected))
            route = types.SimpleNamespace(indices=indices, counts=counts)
            coarse = fine

    def test_smallq_against_dense_masked_math(self):
        from rdd.kernels.smallq import smallq_block_sparse_attention
        torch.manual_seed(12)
        for blocks in (2, 5):
            n = blocks * 64
            q, k, v = [torch.randn(1, 2, n, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            route = capture_route(torch.randn(1, 2, blocks, blocks, device="cuda"), (1, 1, blocks), 0.4)
            sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
            sizes[-1] = 19  # Partial final tile must not attend to padding.
            mask = torch.zeros(1, 2, blocks, blocks, device="cuda", dtype=torch.bool)
            mask.scatter_(-1, route.indices.long(), True)
            mask = mask.repeat_interleave(64, -2).repeat_interleave(64, -1)
            valid = (torch.arange(64, device="cuda")[None, :] < sizes[:, None]).flatten()
            logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(128)
            expected = logits.masked_fill(~(mask & valid), -float("inf")).softmax(-1) @ v.float()
            for query_tile in (16, 32):
                actual = smallq_block_sparse_attention(q, k, v, route.indices, route.counts, sizes,
                                                       query_tile=query_tile)
                torch.testing.assert_close(actual.float(), expected, atol=0.008, rtol=0.015)

    def test_fused_boundary_against_eager(self):
        from rdd.kernels.noise_filling import fused_noise_filling_transition
        clean = torch.randn(1, 2, 3, 4, 6, device="cuda", dtype=torch.bfloat16).transpose(3, 4)
        sigma = torch.tensor(0.6, device="cuda", dtype=torch.bfloat16)
        for shape in [(6, 12, 8), (5, 11, 7)]:
            expected = noise_filling(clean, shape, sigma, torch.Generator(device="cuda").manual_seed(10))
            actual = fused_noise_filling_transition(clean, shape, sigma, torch.Generator(device="cuda").manual_seed(10))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
