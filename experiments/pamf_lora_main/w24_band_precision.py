"""Receive-conditioned spectral precision in the second W24 MAP solve."""

import importlib
import math

import torch
from torch import nn


class BandPrecision(nn.Module):
    def __init__(self, *, resolved):
        super().__init__()
        self.resolved = resolved
        bins = (384 * torch.fft.fftshift(torch.fft.fftfreq(384))).round().abs()
        self.register_buffer('band', (bins >= 9) & (bins <= 32), persistent=False)
        self.net = nn.Sequential(
            nn.Conv1d(5, 24, 5, padding=2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv1d(24, 24, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(24, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observed, residual, noise):
        signal, innovation = (torch.fft.fftshift(torch.fft.fft(x, norm='ortho'), dim=-1)
                              for x in (observed, residual))
        signal_power, innovation_power = signal.abs().square(), innovation.abs().square()
        cross = signal * innovation.conj() / (signal.abs() * innovation.abs() + noise)
        fields = torch.stack(((signal_power / noise).log1p(), (innovation_power / noise).log1p(),
                              cross.real, cross.imag, noise.log().expand_as(signal_power) / 5.), dim=-2)
        logits = self.net(fields.flatten(0, 1)).unflatten(0, observed.shape[:2]).squeeze(-2)
        log_gain = math.log(8) * logits.tanh()
        if not self.resolved:
            averaged = log_gain[..., self.band].mean(dim=-1, keepdim=True)
            log_gain = averaged.expand_as(log_gain)
        return torch.fft.ifftshift((log_gain * self.band).exp(), dim=-1)


def precision_adjustment(value, inverse_rms, gain):
    """Non-diagonal part of D^(-1/2) F* diag(gain) F D^(-1/2)."""
    spectrum = torch.fft.fft(inverse_rms * value, norm='ortho')
    return inverse_rms * torch.fft.ifft((gain - 1.) * spectrum, norm='ortho')


def map_two(archive, action, current, prior, variance, remaining, gain):
    inverse_rms, weight = variance.rsqrt(), variance.reciprocal()
    residual = action.adjoint(remaining / variance + precision_adjustment(remaining, inverse_rms, gain))
    residual = residual - (current - prior) / archive.PRIOR_VARIANCE
    direction = residual
    squared = archive.dot(residual, residual)
    for step in range(2):
        observed_direction = action(direction)
        weighted = weight * observed_direction + precision_adjustment(observed_direction, inverse_rms, gain)
        normal = action.adjoint(weighted) + direction / archive.PRIOR_VARIANCE
        alpha = squared / archive.dot(direction, normal).clamp_min(1e-18)
        current = current + alpha * direction
        if step == 0:
            residual = residual - alpha * normal
            following = archive.dot(residual, residual)
            direction = residual + (following / squared.clamp_min(1e-18)) * direction
            squared = following
    return current


def restorer(training, *, resolved):
    from experiments.pamf_lora_main.w24_final_radius import projected_stages

    package = training.DuraW24.__module__.rpartition('.')[0]
    archive = importlib.import_module('.dura_capacity', package)

    class PrecisionRestorer(training.DuraW24):
        def __init__(self):
            super().__init__()
            self.requires_grad_(False)
            self.band_precision = BandPrecision(resolved=resolved)
            self.train()

        def train(self, mode=True):
            super().train(False)
            self.band_precision.train(mode)
            return self

        def solve(self, action, observed, current, prior, variance, remaining, noise):
            gain = self.band_precision(observed, remaining, noise)
            return map_two(archive, action, current, prior, variance, remaining, gain)

        def stages(self, y, quality, rms, *, condition=()):
            return projected_stages(self, archive, y, quality, rms, condition=condition,
                                    expand_radius=False, final_solver=self.solve)

    return PrecisionRestorer()
