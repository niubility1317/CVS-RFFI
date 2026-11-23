"""Confidence shrinkage for native noise PSD measured on disjoint LoRa windows."""

import torch
from torch.nn import functional as F

from .chirp_spectrum_recovery import power
from .rf_noise_transport import FFT, HOP, RFNoiseTransport


def repeatability(a, b):
    """Split-profile agreement above a frequency-shift surrogate."""
    a = a / a.mean(dim=-1, keepdim=True).clamp_min(1e-8) - 1
    b = b / b.mean(dim=-1, keepdim=True).clamp_min(1e-8) - 1
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    aligned = (a * b).sum(dim=-1)
    correlation = torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b).conj(), n=FFT)
    # Distant circular shifts preserve each profile's smoothness but break
    # agreement at matched receiver frequencies.
    null = correlation[..., 6:-5].square().mean(dim=-1).sqrt()
    power_mean = (.5 * (a + b)).square().sum(dim=-1)
    return ((aligned - null) / power_mean.clamp_min(1e-8)).clamp(0, 1)


class RFNoiseConfidence(RFNoiseTransport):
    def __init__(self, adaptive):
        super().__init__(aligned=True)
        self.adaptive = adaptive

    def confidence(self, samples):
        # Use nonoverlapping 32-point windows covering the same received sweep.
        # Centering creates two extra boundary frames. Neither fold uses them.
        first = samples[:, :, 2:-2:8].flatten(1, 2)
        second = samples[:, :, 6:-2:8].flatten(1, 2)
        profiles = []
        for fold in (first, second):
            value = fold.nanmedian(dim=-2).values
            floor = fold.flatten(1).nanmedian(dim=-1, keepdim=True).values
            enough = fold.isfinite().sum(dim=-2) >= 4
            profiles.append(torch.where(enough, value, floor))
        return repeatability(*profiles)

    def transport(self, profile, scalar, rotation):
        padded = F.pad(rotation, (FFT // 2, FFT // 2), mode='reflect')
        windows = padded.unfold(-1, FFT, HOP)
        kernel = power(torch.fft.fft(windows.conj() * self.window))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mapped = torch.fft.irfft(torch.fft.rfft(profile.unsqueeze(-2)) *
                                torch.fft.rfft(kernel), n=FFT).clamp_min(0)
        divisor = mapped.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return scalar.unsqueeze(-2) * (mapped / divisor)

    def estimate_with_confidence(self, observed, rotation):
        samples, outside, _ = self.native_samples(observed, rotation)
        profile, scalar = self.noise_profile(samples, outside)
        confidence = self.confidence(samples)
        if self.adaptive:
            profile = scalar + confidence[:, None, None] * (profile - scalar)
        return self.transport(profile, scalar, rotation), confidence

    def estimate_noise(self, observed, rotation):
        estimate, _ = self.estimate_with_confidence(observed, rotation)
        return estimate
