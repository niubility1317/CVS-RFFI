"""Receive-side complex features and sufficient statistics for LoRa repeats."""

import torch

from .band_allocation import fft


OFFSETS = (-8, -4, -2, -1, 1, 2, 4, 8)
RIDGE = .1


def repeat_features(spectra, desired, *, shuffled=False):
    """Features of one fold use only the other two disjoint chirp groups."""
    # The fold order is (0,3), (1,4), (2,5). There is no target-fold IQ.
    groups = spectra.unflatten(1, (2,3)).mean(1)
    first, second = groups.roll(1, dims=1), groups.roll(-1, dims=1)
    size = spectra.shape[-1]
    minus = (-desired).remainder(size)
    adjacent_left, adjacent_right = ((desired + step).remainder(size) for step in (-1,1))
    linear = torch.stack((first[..., desired], second[..., desired],
                          first[..., minus].conj(), second[..., minus].conj(),
                          first[..., adjacent_left], second[..., adjacent_left],
                          first[..., adjacent_right], second[..., adjacent_right]), -1)
    other = second.roll(1, dims=0) if shuffled else second
    products = []
    for offset in OFFSETS:
        complement = (desired-offset).remainder(size)
        entry = (first[..., offset, None]*other[..., complement]
                 + other[..., offset, None]*first[..., complement]) * .5
        products.append(entry)
    return linear, torch.stack(products, -1)


def real_parts(values):
    return torch.view_as_real(values.resolve_conj()).flatten(-2)


def normalized_spectrum(tracked):
    # Use per-chirp carrier units; zero the DC before building any product.
    carrier = tracked.mean(-1, keepdim=True).abs()
    normalized = fft(tracked / carrier.clamp_min(.15))
    normalized[..., 0] = 0.
    return normalized


def moments(inputs, targets):
    batch = inputs.shape[0]
    ones = torch.ones_like(inputs[..., :1])
    samples = torch.cat((ones, inputs, targets), -1).permute(1,2,0,3)
    return (samples.mT @ samples).double(), batch


def solve(moment, count, features):
    means = moment[..., 0, 1:] / count
    gram = moment[..., 1:, 1:] / count - means.unsqueeze(-1)*means.unsqueeze(-2)
    standard = gram.diagonal(dim1=-2, dim2=-1)[..., :features].clamp_min(1e-10).sqrt()
    covariance = gram[..., :features, :features] / (standard.unsqueeze(-1)*standard.unsqueeze(-2))
    eye = torch.eye(features, device=moment.device, dtype=moment.dtype)
    cross = gram[..., :features, features:] / standard.unsqueeze(-1)
    coefficients = torch.linalg.solve(covariance + RIDGE*eye, cross) / standard.unsqueeze(-1)
    return means[..., :features].float(), means[..., features:].float(), coefficients.float()


def independent_noise(clean, rng):
    # An independent, target-free range of disturbance levels. The sampled
    # scale is used only to generate the observation.
    random = torch.rand((len(clean), 1, 1), device=clean.device, generator=rng)
    scale = (.02 + .28 * random) * clean.abs().square().mean((1, 2), keepdim=True).sqrt()
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=rng)
    return clean + scale*noise
