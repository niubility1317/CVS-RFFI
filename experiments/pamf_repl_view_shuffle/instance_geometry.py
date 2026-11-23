"""Preserve relations between measured source packets under synthetic disturbances."""

import torch
from torch.nn import functional as F


def descriptor(signal):
    """Normalized log amplitude and complex phase curvature at two frequency lags."""
    spectrum = torch.fft.fft(signal, norm='ortho')
    energy = spectrum.abs().square()
    spectrum = spectrum / energy.mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    amplitude = spectrum.abs()
    unit = spectrum / (amplitude.square() + .1 ** 2).sqrt()
    curvature = []
    for lag in (1, 4):
        closure = unit.roll(lag, -1) * unit.roll(-lag, -1) * unit.conj().square()
        curvature.extend((closure.real, closure.imag))
    return torch.cat((amplitude.log1p(), *curvature), dim=-1)


def distances(restored, clean):
    """Pairwise distances scaled by the mean separation in the measured reference batch."""
    reference = descriptor(clean).detach()
    output = descriptor(restored)
    baseline = torch.cdist(reference, reference).square()
    scale = (baseline.sum() / (len(clean) * (len(clean) - 1))).clamp_min(1e-4)
    predicted = torch.cdist(output.transpose(0, 1), reference).square()
    return predicted / scale, baseline / scale


def pair_loss(predicted, baseline):
    # Similar clean packets are kept close even if they are different physical packets.
    expected = (-baseline / .2).softmax(dim=-1)
    logits = (-predicted / .2).log_softmax(dim=-1)
    divergence = F.kl_div(logits, expected.expand_as(logits), reduction='none').sum(dim=-1)
    point_error = predicted.diagonal(dim1=-2, dim2=-1)
    return divergence.mean() + point_error.mean()


def relation_loss(restored, clean):
    return pair_loss(*distances(restored, clean))
