"""Estimate LoRa noise in received RF coordinates and carry it through dechirping."""

from math import log

import torch
from torch import nn

from .channel_blind import zero_carrier
from .chirp_spectrum_recovery import power


FFT = 32
HOP = 8
EDIT = .45


def colored_observation(clean, rng):
    """Independent generic broad- and narrow-band color, without exposing latent parameters."""
    b, c, n = clean.shape
    kw = dict(device=clean.device, generator=rng)
    frequency = torch.fft.fftfreq(n, device=clean.device)
    white = torch.complex(torch.randn(b, c, n, **kw), torch.randn(b, c, n, **kw))
    centers = torch.rand((b, 2, 1), **kw) - .5
    width = .015 + .14 * torch.rand((b, 2, 1), **kw)
    circular = (frequency - centers + .5).remainder(1) - .5
    peak = torch.exp(-.5 * (circular / width).square())
    shape = 1 + (5 * torch.rand((b, 2, 1), **kw) * peak).sum(dim=1, keepdim=True)
    azimuth = 2 * torch.pi * torch.rand((b, c, 1), **kw)
    coupling = torch.polar(torch.ones_like(azimuth), azimuth)
    fraction = .65 * torch.rand((b, 1, 1), **kw)
    common = torch.complex(torch.randn(b, 1, n, **kw), torch.randn(b, 1, n, **kw))
    noise = (1 - fraction).sqrt() * white + fraction.sqrt() * coupling * common
    noise = torch.fft.ifft(torch.fft.fft(noise) * shape.sqrt())
    scale = power(clean).mean(-1, keepdim=True) / power(noise).mean(-1, keepdim=True).clamp_min(1e-8)
    dose = 1.35 * torch.rand((b, c, 1), **kw).square()
    return clean + dose * scale.sqrt() * noise


def network(size):
    layers = nn.Sequential(nn.Conv2d(size, 16, 3, padding=1), nn.SiLU(),
                           nn.Conv2d(16, 24, 3, padding=1), nn.SiLU(),
                           nn.Conv2d(24, 16, 3, padding=1), nn.SiLU(),
                           nn.Conv2d(16, 2, 1))
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return layers


def analyze(value, window):
    spectrum = torch.stft(value.flatten(0, 1), FFT, HOP, window=window, normalized=True,
                          onesided=False, return_complex=True, pad_mode='reflect')
    return spectrum.unflatten(0, value.shape[:2]).transpose(-1, -2)


def native_samples(observed, rotation, window, frequency):
    native = analyze(observed * rotation, window)
    ridge = power(analyze(rotation, window)).argmax(dim=-1, keepdim=True)
    distance = (frequency - ridge + FFT // 2).remainder(FFT) - FFT // 2
    outside = distance.abs() >= 7
    return power(native).masked_fill(~outside, torch.nan), outside, ridge


def noise_profile(samples, outside):
    # Use the current chirp PSD with a mild shrinkage to same-packet estimates.
    # Very short/ambiguous ridge support falls back to that chirp's aggregate.
    aggregate = samples.flatten(-2).nanmedian(dim=-1).values
    per_tone = samples.nanmedian(dim=-2).values
    per_tone = torch.where(outside.sum(dim=-2) >= 4, per_tone, aggregate[..., None])
    packet = per_tone.median(dim=1, keepdim=True).values
    profile = (.75 * per_tone + .25 * packet) / log(2)
    scalar = profile.mean(dim=-1, keepdim=True)
    # Both arms use the same mean noise power from the same native samples.
    color = profile / scalar.clamp_min(1e-8)
    color = color.clamp(.1, 5.)
    color = color / color.mean(dim=-1, keepdim=True)
    return scalar * color, scalar


class RFNoiseTransport(nn.Module):
    def __init__(self, aligned):
        super().__init__()
        self.aligned = aligned
        self.coarse = network(8)
        self.fine = network(11)
        self.register_buffer('window', torch.hann_window(FFT), persistent=False)
        self.register_buffer('frequency', torch.arange(FFT), persistent=False)

    def analyze(self, value):
        return analyze(value, self.window)

    def synthesize(self, spectrum, view):
        flat = spectrum.transpose(-1, -2).flatten(0, 1)
        recovered = torch.istft(flat, FFT, HOP, window=self.window, normalized=True,
                                onesided=False, return_complex=True, length=view.shape[-1])
        return recovered.reshape_as(view)

    def native_samples(self, observed, rotation):
        return native_samples(observed, rotation, self.window, self.frequency)

    noise_profile = staticmethod(noise_profile)

    def estimate_noise(self, observed, rotation):
        samples, outside, ridge = self.native_samples(observed, rotation)
        profile, scalar = self.noise_profile(samples, outside)
        if self.aligned:
            mapping = (self.frequency + ridge).remainder(FFT)
            return profile[:, :, None].expand_as(samples).gather(-1, mapping)
        return scalar[:, :, None].expand_as(samples)

    def forward(self, observed, quality, rotation):
        native_shape = self.estimate_noise(observed, rotation)
        spectrum = self.analyze(observed)
        chirps = observed.shape[1]
        common = (spectrum.sum(dim=1, keepdim=True) - spectrum) / (chirps - 1)
        signal = (power(spectrum) - native_shape).clamp_min(0)
        removal = native_shape / (native_shape + signal + .002)
        consistency = (power(common) - native_shape / (chirps - 1)).clamp_min(0)
        trust = consistency / (consistency + native_shape + .002)
        radial = -removal * spectrum
        repeat = trust * (common - spectrum)
        anchor = native_shape / (native_shape + .008)
        distance = torch.minimum(self.frequency, FFT - self.frequency) / (FFT / 2)
        fields = torch.stack((spectrum.real, spectrum.imag, common.real, common.imag,
                              native_shape.log1p(), power(spectrum).log1p(),
                              quality[..., None].expand_as(spectrum.real),
                              distance.expand_as(spectrum.real)), dim=2)

        def correction(weights):
            first, second = weights.sigmoid().unbind(2)
            delta = anchor * (first * radial + second * repeat)
            radius = EDIT * spectrum.abs()
            return delta * (radius / (radius + delta.abs()).clamp_min(1e-12))

        shape = observed.shape[:2]
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            coarse_weights = self.coarse(fields.flatten(0, 1))
        coarse_weights = coarse_weights.float().unflatten(0, shape)
        coarse_delta = correction(coarse_weights)
        prior = spectrum + coarse_delta
        evidence = torch.stack((prior.real, prior.imag, power(coarse_delta).log1p()), dim=2)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            fine_weights = self.fine(torch.cat((fields, evidence), dim=2).flatten(0, 1))
        fine_delta = correction(coarse_weights + fine_weights.float().unflatten(0, shape))
        return tuple(observed + zero_carrier(self.synthesize(edit, observed))
                     for edit in (coarse_delta, fine_delta))
