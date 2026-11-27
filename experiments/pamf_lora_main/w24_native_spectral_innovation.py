"""Observation-constrained frequency update for a coarse native-IQ estimate."""

import torch
from torch import nn


class SpectralInnovation(nn.Module):
    """Share frequency-power evidence across chirps while retaining individual phase."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Sequential(
            nn.Conv1d(6, 32, 7, padding=3, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 32, 9, dilation=3, padding=12, padding_mode='circular'), nn.SiLU(),
            nn.Conv1d(32, 1, 5, padding=2, padding_mode='circular'),
        )
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)
        bins = torch.fft.fftfreq(1024).abs()
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
        gains = self.gain(features.flatten(0, 1)).squeeze(1).unflatten(0, features.shape[:2])
        update = .5 * gains.tanh() * self.edit_band * innovation
        return torch.fft.ifft(update, norm='ortho')
