"""Recover the complete complex error with an explicit noncarrier support."""

import torch

from .complex_vector import ComplexVector, fft, rms


CAP = .45


def partitions(n, device, wide):
    f = torch.fft.fftfreq(n, device=device).abs()
    low = ((f > 0) & (f <= 1/32), (f > 1/32) & (f <= 1/8))
    return (*low, (f > 1/8) & (f <= 1/4), f > 1/4) if wide else low


def supported(value, wide):
    support = torch.stack(partitions(value.shape[-1], value.device, wide)).any(dim=0)
    masked = value*support
    return masked * (CAP/rms(masked)).clamp_max(1.)


def target_vectors(baseline, signal):
    error = fft(signal-baseline)
    return dict(narrow=supported(error, False), broad=supported(error, True))


class BroadbandVector(ComplexVector):
    def __init__(self, *, wide):
        super().__init__()
        self.wide = wide

    def forward(self, observed, baseline, quality):
        return supported(.20*self.proposal(observed, baseline, quality), self.wide)


def band_risk(estimated, expected, wide):
    score = []
    for b in partitions(expected.shape[-1], expected.device, wide):
        error = (estimated[..., b]-expected[..., b]).abs().square().mean(-1)
        score.append((error/(.004 + expected[..., b].abs().square().mean(-1))).mean())
    return torch.stack(score).mean()
