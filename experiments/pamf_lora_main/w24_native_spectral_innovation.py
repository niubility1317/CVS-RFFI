"""Observation-constrained frequency update for a coarse native-IQ estimate."""

import torch
from torch import nn


class SpectralInnovation(nn.Module):
    """Share frequency-power evidence across chirps while retaining individual phase."""

    def __init__(self, *, phase_aware=False, quadrature=False, samples=1024):
        super().__init__()
        if quadrature and not phase_aware:
            raise ValueError('Quadrature gain requires phase observations')
        self.phase_aware = phase_aware
        self.quadrature = quadrature
        self.gain = nn.Sequential(
            nn.Conv1d(8 if phase_aware else 6, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, dilation=3, padding=12, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 2 if quadrature else 1, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)
        bins = torch.fft.fftfreq(samples).abs()
        self.register_buffer('noise_band', bins >= 1/3, persistent=False)
        self.register_buffer('edit_band', bins > 0, persistent=False)

    def forward(self, observed, coarse):
        y = torch.fft.fft(observed, norm='ortho')
        x = torch.fft.fft(coarse, norm='ortho')
        innovation = y - x
        observed_power = y.abs().square()
        outside = observed_power[..., self.noise_band].median(-1, keepdim=True).values
        noise = (outside / .69314718).clamp_min(1e-5)
        local = observed_power / noise
        estimate = x.abs().square() / noise
        residual = innovation.abs().square() / noise
        shared = tuple(power.mean(1, keepdim=True).expand_as(power) for power in (local, estimate, residual))
        features = torch.stack((local, estimate, residual, *shared), dim=-2).log1p()
        if self.phase_aware:
            relative = (x * y.conj() / noise) / ((1. + local) * (1. + estimate)).sqrt()
            phase = torch.stack((relative.real, relative.imag), dim=-2)
            features = torch.cat((features,phase),dim=-2)
        logits = self.gain(features.flatten(0, 1)).unflatten(0, features.shape[:2])
        real = logits[..., 0, :]
        if self.phase_aware:
            gain = torch.complex(real, logits[..., 1, :]) if self.quadrature else real
            gain = gain / (1. + gain.abs().square()).sqrt()
        else:
            gain = real.tanh()
        update = .5 * gain * self.edit_band * innovation
        return torch.fft.ifft(update, norm='ortho')
