"""Train and deploy a physical edit-energy bound for LoRa refinement."""

import torch

from .fission_risk import band_masks
from .private_dose import band_energy


def bound_refinement(observed, coarse, fine, rms):
    """Preserve the fine direction, clipping each physical norm to the coarse norm."""
    prefix = torch.fft.fft((coarse - observed).detach(), norm='ortho')
    proposal = torch.fft.fft(fine - observed, norm='ortho')
    bands = band_masks(proposal).to(proposal.real.dtype)
    limit = band_energy(prefix, rms, bands).sqrt()
    size = band_energy(proposal, rms, bands).clamp_min(1e-12).sqrt()
    gain = (limit / size).clamp_max(1)
    bounded = torch.fft.ifft(proposal * (gain @ bands)[:, None], norm='ortho')
    return coarse, observed + bounded
