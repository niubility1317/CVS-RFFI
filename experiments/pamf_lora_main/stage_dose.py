"""Separate physical edit energy from the direction of a LoRa stage edit."""

import torch

from .fission_risk import band_masks
from .private_dose import band_energy, band_geometry, match_band_energy


def edit_controls(coarse, fine, rms):
    """Read out a direction/dose factorial and a conservative tangent correction."""
    spectrum = torch.fft.fft(coarse, norm='ortho')
    refined = torch.fft.fft(fine, norm='ortho')
    bands = band_masks(spectrum).to(coarse.real.dtype)
    # These edits have their carrier removed by the parent. Keep it out of all
    # matched projections even if roundoff leaves a tiny DC residual.
    spectrum = spectrum * bands.sum(0)
    refined = refined * bands.sum(0)
    fine_at_coarse = match_band_energy(spectrum, refined, rms, bands)
    coarse_at_fine = match_band_energy(refined, spectrum, rms, bands)
    energy, _, cosine, _ = band_geometry(spectrum, fine_at_coarse, rms, bands).unbind(1)
    cosine = torch.where(energy > 1e-7, cosine, 0.)
    parallel = spectrum * (cosine @ bands)[:, None]
    tangent = fine_at_coarse - parallel
    projected = match_band_energy(spectrum, spectrum + .5 * tangent, rms, bands)
    # Return measured relative band energies for validation without recomputing the network.
    targets = (spectrum, refined, spectrum)
    actual = (fine_at_coarse, coarse_at_fine, projected)
    error = torch.stack([(band_energy(a, rms, bands) - band_energy(b, rms, bands)).abs()
                         / band_energy(b, rms, bands).clamp_min(1e-10)
                         for a, b in zip(actual, targets, strict=True)], dim=1)
    edits = tuple(torch.fft.ifft(x, norm='ortho') for x in actual)
    return edits, error
