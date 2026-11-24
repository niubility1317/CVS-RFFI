"""Probe nonlinear LoRa recovery along opposite same-packet alias perturbations."""

import torch

from .chirp_band_power import frequency_groups


def private(value):
    return value - value.mean(dim=-2, keepdim=True)


def alias_probe(y):
    bins = frequency_groups(y.shape[-1], y.device)[0]
    measured = torch.fft.fft(y, norm='ortho')
    alias = (bins + y.shape[-1] // 2) % y.shape[-1]
    probe = torch.zeros_like(measured)
    probe[..., bins] = private(measured[..., alias])
    return torch.fft.ifft(probe, norm='ortho')


def nonlinear_change(anchor, positive, negative):
    return anchor - .5 * (positive + negative)


def orient_pair(anchor, direction, rms):
    """Equal physical energy and equal edit size in opposite tangent directions."""
    weight = rms.square()
    inner = lambda a, b: (a.conj() * b * weight).sum((-1, -2), keepdim=True)
    power = inner(anchor, anchor).real
    tangent = direction - anchor * (inner(anchor, direction) / power.clamp_min(1e-12))
    tangent = torch.where(power > 1e-12, tangent, 0.)
    size = inner(tangent, tangent).real
    tangent *= (.25 * (power / size.clamp_min(1e-12)).sqrt()).clamp_max(1.)
    moved = inner(tangent, tangent).real
    scale = torch.where(power > 1e-12, (power / (power + moved).clamp_min(1e-12)).sqrt(), 1.)
    return scale*(anchor + tangent), scale*(anchor - tangent)


def substitute(recovered, difference, rms):
    bins = frequency_groups(recovered.shape[-1], recovered.device)
    base = torch.fft.fft(recovered, norm='ortho')
    curvature = torch.fft.fft(difference, norm='ortho')
    primary, opposite = base.clone(), base.clone()
    near = bins[0]
    original = private(base[..., near])
    direction = private(curvature[..., near])
    plus, minus = orient_pair(original, direction, rms)
    primary[..., near] += plus - original
    opposite[..., near] += minus - original
    wide = primary.clone()
    middle = bins[1]
    original = private(base[..., middle])
    plus, _ = orient_pair(original, private(curvature[..., middle]), rms)
    wide[..., middle] += plus - original
    return tuple(torch.fft.ifft(x, norm='ortho') for x in (primary, opposite, wide))
