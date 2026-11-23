"""Complex spectral residual on top of the mature LoRa recovery waveform."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from experiments.pamf_unrolled_restore.denoise import complementary_context

from .denoise_components import bands, equal_rms, split_along


CAP = .06


def fft(signal):
    return torch.fft.fft(signal, norm='ortho')


def rms(signal):
    return signal.abs().square().mean(-1, keepdim=True).clamp_min(1e-16).sqrt()


def teacher_vectors(baseline, expected):
    """Both teachers have the same correction RMS before sample fitting."""
    actual, signal = fft(baseline), fft(expected)
    payload = {}
    for recipe in ('direct', 'balanced'):
        payload[recipe] = torch.zeros_like(actual)
    for band in bands(actual.shape[-1], actual.device):
        want = signal[..., band]
        error = want-actual[..., band]
        p, o = equal_rms(*split_along(error, want))
        combined = (p+o)*(.5**.5)
        payload['balanced'][..., band] = combined
        payload['direct'][..., band] = error * (rms(combined)/rms(error)).clamp_max(1.)
    return payload


class ComplexVector(nn.Module):
    """Estimate a bounded joint correction from the received and current IQ."""

    def __init__(self):
        super().__init__()
        self.spectral = Core(incoming=13, outgoing=2)
        self.temporal = Core(incoming=7, outgoing=2)

    def forward(self, observed, baseline, quality):
        original, current = (torch.fft.fftshift(fft(x), dim=-1) for x in (observed, baseline))
        nearby, dispersion = complementary_context(original, quality)
        companion, _ = complementary_context(current, quality)
        inputs = (original, current, current-original, nearby, companion)
        fields = [q for value in inputs for q in (value.real, value.imag)]
        q = quality.expand_as(current.real)
        features = torch.stack((*fields, q, current.abs().log1p(), dispersion.log1p()), dim=2)
        first = self.spectral(features.flatten(0, 1)).unflatten(0, observed.shape[:2])
        proposal = torch.complex(first[:, :, 0], first[:, :, 1])
        detail = torch.fft.ifft(torch.fft.ifftshift(proposal, dim=-1), norm='ortho')
        t = torch.stack((observed.real, observed.imag, baseline.real, baseline.imag,
                         detail.real, detail.imag, q), dim=2)
        second = self.temporal(t.flatten(0, 1)).unflatten(0, observed.shape[:2])
        proposed = .06*(torch.fft.ifftshift(proposal, dim=-1) +
                        fft(torch.complex(second[:, :, 0], second[:, :, 1])))
        output = torch.zeros_like(proposed)
        for band in bands(proposed.shape[-1], proposed.device):
            x = proposed[..., band]
            output[..., band] = x / (1.+(rms(x)/CAP).square()).sqrt()
        return output


def vector_risk(proposal, target):
    per_band = []
    for band in bands(target.shape[-1], target.device):
        wanted, received = target[..., band], proposal[..., band]
        error = (wanted-received).abs().square().mean(-1)
        per_band.append((error / (.001 + wanted.abs().square().mean(-1))).mean())
    return .55*per_band[0] + .45*per_band[1]
