"""Learn current-packet complex correction coordinates at a frozen dose."""

import torch
from torch import nn

from .dura_repair import complex_iq, no_tone, real_iq
from .repeatable_detail import basis_spec, equal_dose, expand_spectrum, pack_spectrum


class CoefficientHead(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, 512), nn.LayerNorm(512), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU())
        self.out = nn.Linear(512, outputs)

    def forward(self, features):
        value = self.encoder(features)
        return self.out(value + self.context(value))


def fft(z):
    return torch.fft.fft(z, norm='ortho')


@torch.no_grad()
def coordinates(received, reference, extra, tracker):
    observed, rotation, *_ = tracker.track(reference)
    rms = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    direct = no_tone(observed / rms)
    raw = no_tone(complex_iq(received) * rotation.conj() / rms)
    anchor = fft(complex_iq(extra - reference) * rotation.conj() / rms)
    _, mask = basis_spec(observed.device)
    inputs = [pack_spectrum(value, mask) for value in (fft(direct), fft(raw), anchor)]
    standardized = [value / (value.square().mean(-1, keepdim=True) + 1e-4).sqrt() for value in inputs]
    return torch.cat(standardized, dim=1), anchor, (direct, rotation, rms)


@torch.no_grad()
def paired_target(clean, anchor, direct, tracker):
    expected, *_ = tracker.track(clean)
    rms = expected.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    bands, _ = basis_spec(anchor.device)
    return equal_dose(anchor, fft(no_tone(expected / rms) - direct), bands)


def physical_prediction(coefficients, anchor, basis):
    bands, mask = basis_spec(anchor.device)
    if basis is not None:
        coefficients = coefficients @ basis.T
    return equal_dose(anchor, expand_spectrum(anchor, coefficients, mask), bands)


def band_loss(output, anchor, target):
    bands, _ = basis_spec(anchor.device)
    values = []
    for band in bands:
        error = (output[..., band] - target[..., band]).abs().square().sum(-1)
        dose = anchor[..., band].abs().square().sum(-1)
        values.append((error / (dose + 1e-5)).mean(-1))
    return torch.stack(values).mean(0)


def waveform(reference, correction, rotation, rms):
    restored = torch.fft.ifft(correction, norm='ortho') * rotation * rms
    return reference + real_iq(restored)
