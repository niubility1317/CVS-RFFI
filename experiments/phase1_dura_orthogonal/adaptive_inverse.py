"""Per-window adaptive widely-linear inverse with an independent IQ residual."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def widely_linear_filter(x, taps):
    """Apply each record's complex FIR and conjugate FIR without batch mixing."""
    if (x.ndim != 3 or x.shape[1] != 2 or taps.ndim != 3 or taps.shape[:2] != (len(x), 4)
            or taps.shape[-1] % 2 != 1 or taps.shape[-1] > 2*x.shape[-1]-1):
        raise ValueError('Matching IQ and odd per-record FIR taps required')
    a, b, c, d = taps.unbind(1)
    weights = torch.stack((a+c, -b+d, b+d, a-c), dim=1).reshape(2*len(x), 2, -1)
    radius = taps.shape[-1]//2
    padded = F.pad(x, (radius, radius), mode='reflect') if radius else x
    filtered = F.conv1d(padded.reshape(1, 2*len(x), -1), weights, groups=len(x))
    return filtered.reshape(len(x), 2, x.shape[-1])


class AdaptiveInverseRecovery(nn.Module):
    """One-window filter estimator and residual network; no DURA or F at training/inference."""
    def __init__(self, width=32, taps=33):
        super().__init__()
        if taps < 1 or taps % 2 != 1:
            raise ValueError('An odd positive filter length is required')
        self.taps = taps
        self.entry = nn.Conv1d(5, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, padding=2*d, dilation=d)
                                    for d in (1, 2, 4, 8, 16))
        self.filter_head = nn.Linear(width, 4*taps)
        self.wave = nn.Conv1d(width, 2, 1)
        for head in (self.filter_head, self.wave):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, received, anchor=None, innovation=None, variance=None, *, validate=True):
        if received.ndim != 3 or received.shape[1:] != (2, 256):
            raise ValueError('One [batch, 2, 256] received IQ window required')
        if validate and not torch.isfinite(received).all():
            raise ValueError('Finite received IQ required')
        scale = received.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        normalized = received/scale
        i, q = normalized[:, :1], normalized[:, 1:]
        features = torch.cat((normalized, i.square()+q.square(), i.square()-q.square(), 2*i*q), dim=1)
        hidden = F.silu(self.entry(features))
        for layer in self.layers:
            hidden = hidden + .2*F.silu(layer(hidden))
        taps = self.filter_head(hidden.mean(-1)).tanh().reshape(len(received), 4, self.taps)
        filtered = widely_linear_filter(normalized, taps)/math.sqrt(self.taps)
        wave = self.wave(hidden).tanh()
        return received + .5*scale*(filtered + wave)
