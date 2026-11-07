"""Adaptive source-only DURA correction with a slow-saturating gain."""
import math
import torch
from torch import nn

from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura

GAIN_KIND = 'observable_logenergy_arctan_matched_slope_v1'


def arctan_gain(logit):
    """Map a real logit to (0, 2), with unit gain and slope .5 at zero."""
    return 1 + (2 / math.pi) * torch.atan((math.pi / 4) * logit)


class ArctanLargeContextDura(StableLargeContextDura):
    def __init__(self, width=384, layers=8):
        super().__init__(width=width, layers=layers)
        self.gain = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        candidate = super().forward(received, anchor, innovation, variance, validate=validate)
        power = anchor.square().mean((1, 2)).clamp_min(1e-12)
        features = torch.stack((variance / power,
            innovation.square().mean((1, 2)) / power,
            (candidate - anchor).square().mean((1, 2)) / power,
            (received - anchor).square().mean((1, 2)) / power), dim=1)
        amplitude = arctan_gain(self.gain(torch.log1p(features)))
        return anchor + amplitude.unsqueeze(-1) * (candidate - anchor)
