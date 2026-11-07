"""Condition a global phase correction and complex local residual on source IQ."""
import math

import torch
from torch import nn

from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
from experiments.phase1_dura_orthogonal.phase_equivariant_dura import PhaseEquivariantDura


class ConditionalPhaseDura(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.local = PhaseEquivariantDura(width=width)
        self.summary = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)

    def condition_features(self, received, anchor, innovation, variance, power):
        scale = power.sqrt()[:, None]
        return torch.cat([x.mean(-1)/scale for x in (received, anchor, innovation)]
                         + [torch.log1p(variance.clamp_min(0)/power)[:, None]], dim=1)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching IQ, DURA context and variance required')
        if validate and (not all(torch.isfinite(v).all() for v in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive variance required')
        power = anchor.square().mean((1, 2)).clamp_min(1e-12)
        features = self.condition_features(received, anchor, innovation, variance, power)
        params = self.summary(features)
        phase = (math.pi/2)*params[:, 0:1].tanh()
        magnitude = 1.0 + .1*params[:, 1:2].tanh()
        mix = params[:, 2:3].sigmoid()[:, :, None]
        real, imag = anchor[:, 0], anchor[:, 1]
        cos, sin = phase.cos(), phase.sin()
        rotated = magnitude[:, None] * torch.stack((cos*real-sin*imag, sin*real+cos*imag), dim=1)
        phase_delta = noise_gate(anchor, variance)*(rotated-anchor)
        local_delta = self.local(received, anchor, innovation, variance, validate=validate)-anchor
        return anchor + mix*local_delta + (1-mix)*phase_delta
