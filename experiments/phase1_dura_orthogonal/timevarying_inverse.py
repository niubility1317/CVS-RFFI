"""Time-dependent widely-linear inverse with independent waveform correction."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.adaptive_inverse import AdaptiveInverseRecovery


def timevarying_filter(x, taps):
    """Each record and time index has a separate complex/conjugate FIR."""
    if (x.ndim != 3 or x.shape[1] != 2 or taps.ndim != 4 or taps.shape[:2] != (len(x), 4)
            or taps.shape[3] != x.shape[-1] or taps.shape[2] % 2 != 1
            or taps.shape[2] > 2*x.shape[-1]-1):
        raise ValueError('Matching IQ and odd per-time FIR taps required')
    radius = taps.shape[2]//2
    padded = F.pad(x, (radius, radius), mode='reflect') if radius else x
    windows = padded.unfold(-1, taps.shape[2], 1)
    i, q = windows.unbind(1)
    a, b, c, d = taps.permute(0, 1, 3, 2).unbind(1)
    real = ((a+c)*i + (-b+d)*q).sum(-1)
    imag = ((b+d)*i + (a-c)*q).sum(-1)
    return torch.stack((real, imag), dim=1)


class TimeVaryingInverseRecovery(AdaptiveInverseRecovery):
    """Infer time-dependent filters and local residuals from each observed IQ window."""
    def __init__(self, width=32, taps=9):
        super().__init__(width=width, taps=taps)
        self.filter_head = nn.Conv1d(width, 4*taps, 1)
        nn.init.zeros_(self.filter_head.weight)
        nn.init.zeros_(self.filter_head.bias)

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
        taps = self.filter_head(hidden).tanh().reshape(len(received), 4, self.taps, 256)
        filtered = timevarying_filter(normalized, taps)/math.sqrt(self.taps)
        return received + .5*scale*(filtered+self.wave(hidden).tanh())
