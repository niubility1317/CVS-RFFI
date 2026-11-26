"""Add a training-only low-SNR branch to the generic LoRa source pairing."""

import torch


STRONG_FRACTION = 1 / 3
STRONG_SNR_DB = (-8., 4.)


class StrongSourceMix:
    def __init__(self, generic, rng, device, *, enabled):
        self.generic = generic
        self.rng = rng
        self.enabled = enabled
        self.selected = torch.zeros((), dtype=torch.int64, device=device)

    def __call__(self, clean, generic_rng):
        observed = self.generic(clean, generic_rng)
        if not self.enabled:
            return observed
        options = dict(device=clean.device, dtype=clean.real.dtype, generator=self.rng)
        shape = (len(clean), 1, 1)
        strong = torch.rand(shape, **options) < STRONG_FRACTION
        low, high = STRONG_SNR_DB
        snr_db = low + (high - low) * torch.rand(shape, **options)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=self.rng)
        signal_power = clean.abs().square().mean((1, 2), keepdim=True)
        noise_power = noise.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12)
        scale = (signal_power / noise_power * 10.**(-snr_db / 10.)).sqrt()
        self.selected += strong.sum()
        return torch.where(strong, clean + noise * scale, observed)
