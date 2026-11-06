"""Time-dependent complex correction around a frozen DURA anchor."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.timevarying_inverse import timevarying_filter


class DuraTimeVaryingRecovery(nn.Module):
    def __init__(self, width=32, taps=9):
        super().__init__()
        if taps < 1 or taps % 2 != 1:
            raise ValueError('Odd positive filter length required')
        self.taps = taps
        self.entry = nn.Conv1d(7, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d)
                                    for d in (1, 2, 4, 8, 16))
        self.filter_head = nn.Conv1d(width, 4*taps, 1)
        self.wave = nn.Conv1d(width, 2, 1)
        for head in (self.filter_head, self.wave):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape
                or innovation.shape != received.shape or variance.shape != (len(received),)):
            raise ValueError('Matching IQ and per-record variance required')
        if validate and (not all(torch.isfinite(v).all() for v in (received, anchor, innovation, variance))
                         or (variance <= 0).any()):
            raise ValueError('Finite IQ and positive variance required')
        scale = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        innov_scale = innovation.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        normalized_anchor = anchor/scale
        noise = ((variance.log()-2*scale[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        features = torch.cat((received/scale, normalized_anchor, innovation/innov_scale,
                              noise.expand(-1, 1, 256)), dim=1)
        hidden = F.silu(self.entry(features))
        for layer in self.layers:
            hidden = hidden + .2*F.silu(layer(hidden))
        taps = self.filter_head(hidden).tanh().reshape(len(received), 4, self.taps, 256)
        filtered = timevarying_filter(normalized_anchor, taps)/math.sqrt(self.taps)
        return anchor + .2*scale*(filtered+self.wave(hidden).tanh())
