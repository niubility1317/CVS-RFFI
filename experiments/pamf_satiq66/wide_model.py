"""Supervised PAMF with an ordered raw encoder for the full common word."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import build_model, capture as capture_native


SYMBOL_COUNT = 8
SAMPLES_PER_SYMBOL = 1000
FEATURES = 160
# Both sample grids observe 320 us, so 100 kHz is exactly signed FFT bin 32.
NOISE_HALF_BAND = 32


def relative_iq_channels(iq):
    """Five full-rate channels invariant to a packet-wide complex gain and DC offset."""
    centered = iq - iq.mean(dim=-1, keepdim=True)
    power = centered.square().sum(dim=1)
    rms = power.mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    i, q = (centered / rms[:, None, :]).unbind(dim=1)
    power = i.square() + q.square()
    channels = [torch.log1p(power).unsqueeze(1)]
    # 25 samples is one microsecond at the registered 25 MS/s input rate.
    for delay in (1, 25):
        i0, i1 = i[:, :-delay], i[:, delay:]
        q0, q1 = q[:, :-delay], q[:, delay:]
        real = i1 * i0 + q1 * q0
        imag = q1 * i0 - i1 * q0
        scale = (power[:, delay:] * power[:, :-delay] + .02 ** 2).sqrt()
        correlation = torch.stack((real / scale, imag / scale), dim=1)
        channels.append(F.pad(correlation, (delay, 0)))
    return torch.cat(channels, dim=1)


class RawSymbolEncoder(nn.Module):
    def __init__(self, symbol_count=SYMBOL_COUNT, bounded=True, time_bins=1, phase_relative=False):
        super().__init__()
        self.symbol_count = symbol_count
        self.time_bins = time_bins
        self.phase_relative = phase_relative
        self.symbol_features = 64 * time_bins
        self.frontend = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=9, stride=4, padding=4, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=5, padding=4, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.mix = nn.Conv1d(self.symbol_features, self.symbol_features, kernel_size=3, padding=1,
                             groups=self.symbol_features)
        self.project = nn.Linear(symbol_count * self.symbol_features, FEATURES)
        self.output_norm = nn.LayerNorm(FEATURES)
        self.gain_raw = nn.Parameter(torch.tensor(math.atanh(.05 / .2))) if bounded else None

    def symbol_tokens(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (2, self.symbol_count * SAMPLES_PER_SYMBOL):
            raise ValueError(f"Expected two IQ channels and {self.symbol_count} complete symbols")
        if self.phase_relative:
            channels = relative_iq_channels(raw)
            tiles = channels.reshape(len(raw), 5, self.symbol_count, SAMPLES_PER_SYMBOL).transpose(1, 2)
            frontend_input = tiles.reshape(len(raw) * self.symbol_count, 5, SAMPLES_PER_SYMBOL)
        else:
            # In the original view each symbol starts a new first difference.
            tiles = raw.reshape(len(raw), 2, self.symbol_count, SAMPLES_PER_SYMBOL).transpose(1, 2)
            iq = tiles.reshape(len(raw) * self.symbol_count, 2, SAMPLES_PER_SYMBOL)
            differences = F.pad(iq.diff(dim=-1), (1, 0))
            power = iq.square().sum(dim=1, keepdim=True)
            frontend_input = torch.cat((iq, differences, power), dim=1)
        encoded = self.frontend(frontend_input)
        if self.time_bins == 1:
            moments = torch.cat((encoded.mean(-1), encoded.square().mean(-1)), dim=1)
        else:
            average = F.adaptive_avg_pool1d(encoded, self.time_bins)
            mean_square = F.adaptive_avg_pool1d(encoded.square(), self.time_bins)
            moments = torch.cat((average, mean_square), dim=1)
        return moments.reshape(len(raw), self.symbol_count, self.symbol_features)

    def forward(self, raw):
        tokens = self.symbol_tokens(raw).transpose(1, 2)
        mixed = tokens + F.silu(self.mix(tokens))
        ordered = mixed.transpose(1, 2).flatten(1)
        output = self.output_norm(self.project(ordered))
        return output if self.gain_raw is None else .2 * self.gain_raw.tanh() * output


class WidePAMF(nn.Module):
    def __init__(self, native, raw_encoder):
        super().__init__()
        self.native = native
        self.raw_encoder = raw_encoder

    def forward(self, raw, coarse, capture=False):
        if capture:
            logits, fused, branches = capture_native(self.native, coarse)
            if self.raw_encoder is not None:
                fused = fused + self.raw_encoder(raw)
                logits = self.native.classify_features(fused)
            return logits, fused, branches
        if self.raw_encoder is None:
            return self.native(coarse)
        fused = self.native.features(coarse) + self.raw_encoder(raw)
        return self.native.classify_features(fused)


def build_wide_model(runtime, seed, use_raw):
    native = build_model(runtime, seed, 1)
    raw_encoder = None
    if use_raw:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            raw_encoder = RawSymbolEncoder()
    return WidePAMF(native, raw_encoder)


def _iq_channels(signal):
    return torch.view_as_real(signal).transpose(1, 2).contiguous()


def _band_limited_noise(coefficients, samples):
    spectrum = coefficients.new_zeros((len(coefficients), samples))
    spectrum[:, :NOISE_HALF_BAND + 1] = coefficients[:, :NOISE_HALF_BAND + 1]
    spectrum[:, -NOISE_HALF_BAND:] = coefficients[:, NOISE_HALF_BAND + 1:]
    # Fourier-series coefficients have the same amplitude on either grid.
    return torch.fft.ifft(spectrum, norm="forward")


def wide_views(raw, coarse, rng, strength):
    """Generate a shared phase view and a shared additive noise view per packet."""
    if strength == 0:
        return raw, coarse, raw, coarse
    raw_signal = torch.complex(raw[:, 0], raw[:, 1])
    coarse_signal = torch.complex(coarse[:, 0], coarse[:, 1])
    options = dict(generator=rng, device=raw.device, dtype=raw.dtype)
    phase = torch.rand((len(raw), 1), **options) * (2 * torch.pi)
    rotation = torch.polar(torch.ones_like(phase), phase)
    noise_iq = torch.randn((len(raw), 2, 2 * NOISE_HALF_BAND + 1), **options)
    coefficients = torch.complex(noise_iq[:, 0], noise_iq[:, 1])
    noise_rms = coefficients.abs().square().sum(-1, keepdim=True).sqrt()
    signal_rms = raw.square().sum(1).mean(-1, keepdim=True).sqrt()
    snr = 5 + 10 * torch.rand((len(raw), 1), **options)
    coefficients = coefficients * (strength * signal_rms * torch.pow(10, -snr / 20) / noise_rms)
    raw_noise = _band_limited_noise(coefficients, raw.shape[-1])
    coarse_noise = _band_limited_noise(coefficients, coarse.shape[-1])
    return (_iq_channels(raw_signal * rotation), _iq_channels(coarse_signal * rotation),
            _iq_channels(raw_signal + raw_noise), _iq_channels(coarse_signal + coarse_noise))
