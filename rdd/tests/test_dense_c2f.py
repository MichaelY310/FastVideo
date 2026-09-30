# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA parity with real edge tiles, plus independent CFG cache tests."""
import os
import types
import unittest

import torch


@unittest.skipUnless(os.environ.get("RDD_TEST_CUDA") == "1" and torch.cuda.is_available(), "CUDA opt-in")
class DenseC2FTests(unittest.TestCase):
    def test_keep_all_and_cfg_separation(self):
        from rdd.dense_c2f import DenseC2FController
        from flash_attn import flash_attn_func

        class Impl:
            def forward(self, q, k, v, meta):
                return flash_attn_func(q, k, v)

        impl = Impl()
        model = types.SimpleNamespace(blocks=[types.SimpleNamespace(attn1=types.SimpleNamespace(attn_impl=impl))])
        controller = DenseC2FController(model, density=1.)
        try:
            for stage, shape in enumerate([(4, 6, 10), (4, 12, 20), (4, 24, 40)]):
                tokens = shape[0]*shape[1]*shape[2]//4
                for branch in ("conditional", "unconditional"):
                    q, k, v = [torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                    controller.set_call(stage, shape, branch, boundary=stage < 2)
                    actual = impl.forward(q, k, v, None)
                    expected = flash_attn_func(q, k, v)
                    torch.testing.assert_close(actual, expected, rtol=.03, atol=.015)
            self.assertIsNot(controller.sources[(0, "conditional", 0)], controller.sources[(0, "unconditional", 0)])
            self.assertTrue(all(abs(v-1.) < 1e-6 for v in controller.summary().values()))
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
