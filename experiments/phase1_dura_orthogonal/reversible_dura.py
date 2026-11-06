"""Learned additive coupling with an explicit inverse on the DURA waveform."""
import torch
from torch import nn


class Coupling(nn.Module):
    def __init__(self, width=32, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.net = nn.Sequential(nn.Conv1d(2, width, 5, padding=2), nn.SiLU(),
                                 nn.Conv1d(width, width, 5, padding=2), nn.SiLU(),
                                 nn.Conv1d(width, width, 5, padding=2), nn.SiLU())
        self.head = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def delta(self, fixed):
        scale = fixed.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        return .2*scale*self.head(self.net(fixed/scale)).tanh()

    def forward(self, value, inverse=False):
        left, right = value[..., ::2], value[..., 1::2]
        if self.reverse:
            right, left = left, right
        right = right - self.delta(left) if inverse else right + self.delta(left)
        if self.reverse:
            right, left = left, right
        return torch.stack((left, right), -1).reshape_as(value)


class ReversibleDura(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        self.layers = nn.ModuleList(Coupling(width, reverse=bool(i % 2)) for i in range(depth))

    def forward(self, received, anchor, innovation, variance, *, validate=True):
        if anchor.ndim != 3 or anchor.shape[1:] != (2, 256):
            raise ValueError('DURA IQ windows of shape [batch,2,256] required')
        if validate and not torch.isfinite(anchor).all():
            raise ValueError('Finite DURA IQ required')
        value = anchor
        for layer in self.layers:
            value = layer(value)
        return value

    def inverse(self, output):
        value = output
        for layer in reversed(self.layers):
            value = layer(value, inverse=True)
        return value
