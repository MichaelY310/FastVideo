# SPDX-License-Identifier: Apache-2.0
"""RDD flow math. t: noise(0)->data(1); Wan predicts w=dx/dsigma, sigma=1-t.

Default finest/middle/coarse allocation is 300/300/400, NOT 400/300/300.
No DMD time list, noise rescaling, extra prediction head, or learned transition.
Only exact integer average-pool/repeat projections are accepted.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RDDPath:
    factors: tuple = ((4, 1), (2, 1), (1, 1))  # spatial, temporal; noise -> data
    boundaries: tuple = (0.0, 0.4, 0.7, 1.0)

    def __post_init__(self):
        if len(self.factors) + 1 != len(self.boundaries) or self.factors[-1] != (1, 1):
            raise ValueError("one interval per factor; final factor must be (1,1)")
        if self.boundaries[0] != 0 or self.boundaries[-1] != 1:
            raise ValueError("boundaries must cover noise=0 to data=1")
        if any(b <= a for a, b in zip(self.boundaries, self.boundaries[1:], strict=False)):
            raise ValueError("strictly increasing boundaries required")
        for s, temporal in self.factors:
            if min(s, temporal) < 1:
                raise ValueError("positive factors required")
        for old, new in zip(self.factors, self.factors[1:], strict=False):
            if any(a % b for a, b in zip(old, new, strict=False)):
                raise ValueError("nested integer downsampling required")

    @staticmethod
    def pool(x, spatial, temporal):
        if x.ndim != 5 or x.shape[2] % temporal or x.shape[3] % spatial or x.shape[4] % spatial:
            raise ValueError("expected divisible B,C,T,H,W latent; padding is NOT an exact projection")
        return F.avg_pool3d(x, (temporal, spatial, spatial))

    def stage_clean(self, full, stage):
        s, temporal = self.factors[stage]
        return self.pool(full, s, temporal)

    def project(self, x, stage):
        if stage == 0:
            return x
        old, new = self.factors[stage - 1], self.factors[stage]
        s, temporal = old[0] // new[0], old[1] // new[1]
        low = self.pool(x, s, temporal)
        return low.repeat_interleave(temporal, 2).repeat_interleave(s, 3).repeat_interleave(s, 4)

    def coefficients(self, t, stage):
        a, b = self.boundaries[stage:stage + 2]
        if stage == 0:
            return torch.zeros_like(t), torch.zeros_like(t)
        return t * (b - t) / (b - a), (b - 2 * t) / (b - a)

    def noisy_and_target(self, clean, noise, t, stage):
        """x=t*x1+(1-t)*eps+f(t)*d; w=eps-x1-f'(t)*d."""
        t = t.reshape(-1, 1, 1, 1, 1).to(clean)
        f, df = self.coefficients(t, stage)
        drift = self.project(clean, stage) - clean
        return t * clean + (1 - t) * noise + f * drift, noise - clean - df * drift

    def clean_from_velocity(self, x, w, t, stage):
        """Invert I+[f+(1-t)f'](P-I); standard x-sigma*w is WRONG for drift stages."""
        t = t.reshape(-1, 1, 1, 1, 1).to(x)
        f, df = self.coefficients(t, stage)
        y = x - (1 - t) * w
        denom = 1 - f - (1 - t) * df
        if torch.any(denom.abs() < 1e-5):
            raise ValueError("singular clean inverse")
        low = self.project(y, stage)
        return low + (y - low) / denom

    def transition(self, clean_coarse, full_shape, next_stage, generator):
        """Noise filling at exact next-stage left endpoint; full fresh IID noise.

        A stage endpoint before t=1 is STILL NOISY. Recover clean first using
        clean_from_velocity; do not upsample an intermediate noisy state.
        """
        old, new = self.factors[next_stage - 1], self.factors[next_stage]
        s, temporal = old[0] // new[0], old[1] // new[1]
        up = clean_coarse.repeat_interleave(temporal, 2).repeat_interleave(s, 3).repeat_interleave(s, 4)
        expected = (full_shape[2] // new[1], full_shape[3] // new[0], full_shape[4] // new[0])
        if up.shape[2:] != expected:
            raise ValueError("transition shape mismatch")
        t = self.boundaries[next_stage]
        noise = torch.randn(up.shape, device=up.device, dtype=up.dtype, generator=generator)
        return t * up + (1 - t) * noise
