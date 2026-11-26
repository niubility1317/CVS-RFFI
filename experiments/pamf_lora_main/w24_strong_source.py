"""Add a training-only low-SNR branch to the generic LoRa source pairing."""

import torch


STRONG_FRACTION = 1 / 3
STRONG_SNR_DB = (-8., 4.)
COLOR_CUTOFF_HZ = (75_000., 200_000.)


class PassbandMatchedNoise:
    """Shape generic noise at capture rate and match its deployed center energy."""

    def __init__(self, tracker, rng):
        self.tracker = tracker
        self.rng = rng

    def center_power(self, waveform):
        center = self.tracker.aperture(waveform)[..., :384]
        return center.abs().square().mean((1, 2), keepdim=True)

    def __call__(self, raw_noise):
        raw = raw_noise.flatten(1)
        options = dict(device=raw.device, dtype=raw.real.dtype)
        lo, hi = COLOR_CUTOFF_HZ
        cutoff = lo + (hi-lo) * torch.rand((len(raw),1), generator=self.rng, **options)
        frequency = torch.fft.fftfreq(raw.shape[-1], d=1e-6, **options)
        response = (1. + (frequency/cutoff).pow(4)).rsqrt()
        colored = torch.fft.ifft(torch.fft.fft(raw) * response)
        scale = (self.center_power(raw) / self.center_power(colored).clamp_min(1e-12)).sqrt()
        return (colored * scale.squeeze(-1)).reshape_as(raw_noise)


class StrongSourceMix:
    def __init__(self, generic, rng, device, *, enabled, noise_shape=None):
        self.generic = generic
        self.rng = rng
        self.enabled = enabled
        self.noise_shape = noise_shape
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
        noise = noise * scale
        if self.noise_shape is not None:
            noise = self.noise_shape(noise)
        self.selected += strong.sum()
        return torch.where(strong, clean + noise, observed)
