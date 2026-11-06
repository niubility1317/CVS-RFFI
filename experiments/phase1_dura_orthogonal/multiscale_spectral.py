"""Gated DURA refinement with multiscale time and global spectral context."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
from experiments.phase1_dura_orthogonal.timevarying_inverse import timevarying_filter


def deterministic_linear_2x(value):
    """Linear 2x resize, align_corners=False, using deterministic slice gradients."""
    left = torch.cat((value[..., :1], value[..., :-1]), dim=-1)
    right = torch.cat((value[..., 1:], value[..., -1:]), dim=-1)
    return torch.stack((.75*value+.25*left, .75*value+.25*right), dim=-1).flatten(-2)


def upsample_2x(value):
    if torch.are_deterministic_algorithms_enabled():
        return deterministic_linear_2x(value)
    return F.interpolate(value, size=2*value.shape[-1], mode='linear', align_corners=False)


class DuraMultiscaleSpectral(nn.Module):
    def __init__(self, width=32, taps=9):
        super().__init__()
        if taps < 1 or taps % 2 != 1:
            raise ValueError('Odd positive filter length required')
        self.taps = taps
        self.entry = nn.Conv1d(7, width, 7, padding=3)
        self.down1 = nn.Conv1d(width, 2*width, 4, stride=2, padding=1)
        self.down2 = nn.Conv1d(2*width, 2*width, 4, stride=2, padding=1)
        self.down3 = nn.Conv1d(2*width, 2*width, 4, stride=2, padding=1)
        self.bottleneck = nn.Conv1d(2*width, 2*width, 3, padding=1)
        self.spectral_real = nn.Parameter(torch.zeros(2*width, 17))
        self.spectral_imag = nn.Parameter(torch.zeros(2*width, 17))
        self.up2 = nn.Conv1d(4*width, 2*width, 3, padding=1)
        self.up1 = nn.Conv1d(4*width, 2*width, 3, padding=1)
        self.up0 = nn.Conv1d(3*width, width, 3, padding=1)
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
        normalized = anchor/scale
        noise = ((variance.log()-2*scale[:, 0, 0].log()).clamp(-16, 8)/8)[:, None, None]
        features = torch.cat((received/scale, normalized, innovation/innov_scale, noise.expand(-1, 1, 256)), 1)
        h0 = F.silu(self.entry(features))
        h1 = F.silu(self.down1(h0))
        h2 = F.silu(self.down2(h1))
        h3 = F.silu(self.down3(h2))
        spectrum = torch.fft.rfft(h3, dim=-1, norm='ortho')
        response = torch.complex(self.spectral_real.tanh(), self.spectral_imag.tanh())
        h3 = torch.fft.irfft(spectrum*(1+.1*response[None]), n=32, dim=-1, norm='ortho')
        h3 = h3 + .2*F.silu(self.bottleneck(h3))
        u2 = F.silu(self.up2(torch.cat((upsample_2x(h3), h2), 1)))
        u1 = F.silu(self.up1(torch.cat((upsample_2x(u2), h1), 1)))
        u0 = F.silu(self.up0(torch.cat((upsample_2x(u1), h0), 1)))
        taps = self.filter_head(u0).tanh().reshape(len(received), 4, self.taps, 256)
        filtered = timevarying_filter(normalized, taps)/math.sqrt(self.taps)
        correction = .2*scale*(filtered+self.wave(u0).tanh())
        return anchor + noise_gate(anchor, variance)*correction
