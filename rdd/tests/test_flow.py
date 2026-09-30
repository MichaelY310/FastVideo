# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from rdd.flow import RDDPath


class FlowTests(unittest.TestCase):
    def test_allocation(self):
        path = RDDPath()
        self.assertEqual([round(1000 * (b - a)) for a, b in zip(path.boundaries, path.boundaries[1:])],
                         [400, 300, 300])

    def test_derivative_inverse_endpoints(self):
        path = RDDPath()
        full = torch.randn(2, 3, 4, 16, 24, dtype=torch.float64)
        for stage in range(3):
            x = path.stage_clean(full, stage)
            self.assertTrue(torch.allclose(path.project(path.project(x, stage), stage), path.project(x, stage)))
            noise = torch.randn_like(x)
            a, b = path.boundaries[stage:stage + 2]
            for value in (a + 1e-4, (a + b) / 2, b):
                t = torch.full((2,), value, dtype=x.dtype)
                noisy, w = path.noisy_and_target(x, noise, t, stage)
                self.assertTrue(torch.allclose(path.clean_from_velocity(noisy, w, t, stage), x, atol=1e-9))
                dx = (path.noisy_and_target(x, noise, t + 1e-6, stage)[0] -
                      path.noisy_and_target(x, noise, t - 1e-6, stage)[0]) / 2e-6
                self.assertTrue(torch.allclose(dx, -w, atol=1e-8))
            t = torch.full((2,), a, dtype=x.dtype)
            actual = path.noisy_and_target(x, noise, t, stage)[0]
            self.assertTrue(torch.allclose(actual, a * path.project(x, stage) + (1 - a) * noise))

    def test_refuse_non_divisible_shape(self):
        with self.assertRaises(ValueError):
            RDDPath().stage_clean(torch.zeros(1, 16, 8, 55, 104), 0)


if __name__ == "__main__":
    unittest.main()
