"""Estimate how far to step along a frozen LoRa correction direction."""

import torch

from .w24_native_center import CENTER


GAIN_LIMIT = 1.75
REGIONS = (slice(None, CENTER), slice(CENTER, None))


def regional_power(wave):
    return torch.stack([wave[..., region].abs().square().mean((-2, -1)) for region in REGIONS], dim=-1)


def regional_product(a, b):
    return torch.stack([(a[..., region].conj() * b[..., region]).real.mean((-2, -1))
                        for region in REGIONS], dim=-1)


def optimal_gain(edit, wanted):
    """Use the legal source reference only to compute a least-squares target."""
    power = regional_power(edit)
    return (regional_product(edit, wanted) / power.clamp_min(1e-24)).clamp_min(0.)


def apply_gain(edit, gain):
    return torch.cat([edit[..., region] * gain[:, j, None, None] for j, region in enumerate(REGIONS)], dim=-1)


def gain_features(raw, observed, edit):
    """Derive packet statistics using only the received waveform and fixed edit."""
    chirps = raw.unflatten(-1, (8, 1024))
    power = chirps.abs().square().mean((-2, -1)).clamp_min(1e-24)
    freq = torch.fft.fftfreq(chirps.shape[-1], device=chirps.device).abs()
    spectrum = torch.fft.fft(chirps, norm='ortho').abs().square()
    outer = spectrum[..., freq > .30].flatten(1).median(dim=-1).values / power
    shoulder = spectrum[..., (freq > .16) & (freq <= .30)].flatten(1).median(dim=-1).values / power
    a, b = chirps[:, 1:], chirps[:, :-1]
    correlation = (a * b.conj()).sum(-1).abs()
    denominator = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    coherence = (correlation / denominator.clamp_min(1e-24)).mean(-1).clamp(0., 1.)
    shared = torch.stack((outer.add(1e-8).log(), shoulder.add(1e-8).log(), coherence), dim=-1)
    op, ep = regional_power(observed), regional_power(edit)
    local = torch.stack(((op / power[:, None]).add(1e-8).log(),
                         (ep / op.clamp_min(1e-24)).add(1e-8).log(),
                         (regional_product(observed, edit) / (op.sqrt() * ep.sqrt()).clamp_min(1e-24))
                         .clamp(-1., 1.)), dim=-1)
    return torch.cat((shared[:, None].expand(-1, len(REGIONS), -1), local), dim=-1)


def fit_gain(features, target, weight):
    """Fit two small weighted ridge regressions and matching intercept controls."""
    features, target, weight = (value.double() for value in (features, target, weight))
    target = target.clamp(0., GAIN_LIMIT)
    total = weight.sum(0)
    normalized = weight / total.clamp_min(1e-12)
    center = (features * normalized[..., None]).sum(0)
    scale = ((features - center).square() * normalized[..., None]).sum(0).sqrt().clamp_min(.05)
    intercept = torch.where(total > 0, (target * normalized).sum(0), 1.)
    standardized = (features - center) / scale
    normal = torch.einsum('bri,brj,br->rij', standardized, standardized, normalized)
    rhs = torch.einsum('bri,br,br->ri', standardized, target-intercept, normalized)
    ridge = torch.eye(normal.shape[-1], device=normal.device, dtype=normal.dtype) * .04
    slope = torch.linalg.solve(normal + ridge, rhs.unsqueeze(-1)).squeeze(-1)
    return dict(center=center.float(), scale=scale.float(), intercept=intercept.float(), slope=slope.float())


def predict_gain(parameters, features, *, adaptive=True):
    intercept = parameters['intercept']
    if adaptive:
        centered = (features - parameters['center']) / parameters['scale']
        intercept = intercept + (centered * parameters['slope']).sum(-1)
    return intercept.expand(features.shape[:2]).clamp(0., GAIN_LIMIT)
