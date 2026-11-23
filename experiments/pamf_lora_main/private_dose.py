"""Swap complex correction directions with their physical IQ energy fixed."""

import torch


def action_parts(spectrum, rms):
    weight = rms.square()
    shared = (spectrum * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)
    return shared.expand_as(spectrum), spectrum - shared


def band_energy(spectrum, rms, bands):
    # Rotation has unit modulus. With an orthonormal FFT, these weighted
    # spectral energies sum to the energy of the physical time-domain edit.
    return (spectrum.abs().square() * rms.square()).sum(1) @ bands.T


def match_band_energy(template, donor, rms, bands):
    target_energy = band_energy(template, rms, bands)
    donor_energy = band_energy(donor, rms, bands)
    supported = donor_energy > 1e-14
    scale = torch.where(supported, (target_energy / donor_energy.clamp_min(1e-14)).sqrt(), 0.)
    donor_weight = scale @ bands
    keep = 1. - (supported.to(bands.dtype) @ bands)
    return donor * donor_weight[:, None] + template * keep[:, None]


def component_swaps(first, second, rms, bands):
    shared_a, private_a = action_parts(first, rms)
    shared_b, private_b = action_parts(second, rms)
    substitute_shared = match_band_energy(shared_a, shared_b, rms, bands)
    substitute_private = match_band_energy(private_a, private_b, rms, bands)
    reverse_shared = match_band_energy(shared_b, shared_a, rms, bands)
    reverse_private = match_band_energy(private_b, private_a, rms, bands)
    return (substitute_shared + substitute_private, reverse_shared + reverse_private,
            shared_a + substitute_private, substitute_shared + private_a)


def band_geometry(first, second, rms, bands):
    """Pairs of physical RMS and direction similarity for each component."""
    energy_a = band_energy(first, rms, bands)
    energy_b = band_energy(second, rms, bands)
    dot = (((first.conj() * second).real * rms.square()).sum(1) @ bands.T)
    valid = (energy_a > 1e-14) & (energy_b > 1e-14)
    cosine = torch.where(valid, dot / (energy_a * energy_b).clamp_min(1e-28).sqrt(), 0.)
    return torch.stack((energy_a.sqrt(), energy_b.sqrt(), cosine.clamp(-1., 1.), valid), dim=1)
