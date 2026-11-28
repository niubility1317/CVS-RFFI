"""Real spectral gains applied to a DURA-style physical backprojection.

The caller supplies an action estimated from the current packet and a frozen
same-dataset prior.  No classifier, label or cross-packet state is consumed.
"""

import torch
from torch import nn


class RealSpectralGain(nn.Module):
    """The LoRa real-gain formula adapted to eight 32-sample WiSig groups."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Conv1d(8, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, dilation=3, padding=12, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 1, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        bins = torch.fft.fftfreq(width).abs()
        self.register_buffer('noise_band', bins >= 1/3, persistent=False)
        self.register_buffer('edit_band', bins > 0, persistent=False)

    def forward(self, observed, prior):
        if observed.ndim != 3 or observed.shape != prior.shape or observed.shape[1:] != (8, self.width):
            raise ValueError('Expected eight matching full-rate groups')
        y, x = (torch.fft.fft(z, norm='ortho') for z in (observed, prior))
        innovation = y - x
        power = y.abs().square()
        noise = (power[..., self.noise_band].median(-1, keepdim=True).values / .69314718).clamp_min(1e-5)
        local = tuple(v.abs().square() / noise for v in (y, x, innovation))
        shared = tuple(v.mean(1, keepdim=True).expand_as(v) for v in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        relative = (x * y.conj() / noise) / ((1 + local[0]) * (1 + local[1])).sqrt()
        features = torch.cat((features, torch.stack((relative.real, relative.imag), dim=-2)), dim=-2)
        logits = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).squeeze(-2)
        gain = logits / (1 + logits.square()).sqrt()
        return torch.fft.ifft(.5 * gain * self.edit_band * innovation, norm='ortho')


class PhysicsGain(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width not in (32, 1024):
            raise ValueError('Expected a WiSig or LoRa registered group width')
        self.width = width
        self.net = nn.Sequential(
            nn.Conv1d(8, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=12, dilation=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 1, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('edit_band', torch.fft.fftfreq(width).abs() > 0, persistent=False)

    def forward(self, received, prior, action):
        if received.ndim != 2 or received.shape != prior.shape or received.shape[-1] != 8 * self.width:
            raise ValueError('Expected matching full physical packets')
        backprojection = action.adjoint(received - action(prior))
        y, p, d = (torch.fft.fft(value.unflatten(-1, (8, self.width)), norm='ortho')
                   for value in (received, prior, backprojection))
        powers = (y.abs().square(), p.abs().square(), d.abs().square())
        scale = powers[0].mean((-1, -2), keepdim=True).clamp_min(1e-6)
        local = tuple(power / scale for power in powers)
        shared = tuple(power.mean(1, keepdim=True).expand_as(power) for power in local)
        features = torch.stack((*local, *shared), dim=-2).log1p()
        relative = p * y.conj() / (scale * ((1 + local[0]) * (1 + local[1])).sqrt())
        features = torch.cat((features, torch.stack((relative.real, relative.imag), dim=-2)), dim=-2)
        logits = self.net(features.flatten(0, 1)).unflatten(0, features.shape[:2]).squeeze(-2)
        update = .25 * logits.tanh() * self.edit_band * d
        return prior + torch.fft.ifft(update, norm='ortho').flatten(-2)
