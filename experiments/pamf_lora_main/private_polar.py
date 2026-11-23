"""Mix amplitude and direction of two receiver-only LoRa private spectra."""

import torch

from .private_transport import contrast_basis


POLAR_MODES = ('vector_phase', 'vector_magnitude', 'modal_phase', 'modal_magnitude',
               'modal_phase_low', 'coherent_floor')


def polar_parts(midpoint, full, *, per_mode):
    """Return fixed amplitude/direction swaps, plus a frequency-wise amplitude floor."""
    if per_mode:
        b, f = midpoint.abs(), full.abs()
    else:
        b = torch.linalg.vector_norm(midpoint, dim=1, keepdim=True)
        f = torch.linalg.vector_norm(full, dim=1, keepdim=True)
    floor = 1e-10
    half_direction = midpoint / b.clamp_min(floor)
    full_direction = full / f.clamp_min(floor)
    rotation = torch.where(f > floor, full_direction, half_direction)
    original = torch.where(b > floor, half_direction, full_direction)
    phase = b * rotation
    magnitude = f * original
    return phase, magnitude


def coherent_amplitude(midpoint, full):
    """Retain lost amplitude in proportion to the squared positive complex overlap."""
    a = torch.linalg.vector_norm(midpoint, dim=1, keepdim=True)
    b = torch.linalg.vector_norm(full, dim=1, keepdim=True)
    overlap = ((full * midpoint.conj()).real.sum(1, keepdim=True) /
               (a * b).clamp_min(1e-20)).clamp(0, 1)
    retained = (a - b).clamp_min(0) * overlap.square()
    return full + torch.where(b > 1e-10, (full / b.clamp_min(1e-10)) * retained, 0.)


def polar_deltas(rank, half, full, masks):
    """Keep the public chirp mean and out-of-band readout fixed at the midpoint."""
    basis = contrast_basis(rank.shape[1], dtype=rank.real.dtype, device=rank.device).to(rank.dtype)
    midpoint = basis.T @ torch.fft.fft(rank + half, norm='ortho')
    complete = basis.T @ torch.fft.fft(rank + full, norm='ortho')
    vector_phase, vector_magnitude = polar_parts(midpoint, complete, per_mode=False)
    modal_phase, modal_magnitude = polar_parts(midpoint, complete, per_mode=True)
    protected = coherent_amplitude(midpoint, complete)
    all_bands = masks.any(0)
    low_bands = masks[:2].any(0)
    values = (vector_phase, vector_magnitude, modal_phase, modal_magnitude, modal_phase, protected)
    locations = (all_bands, all_bands, all_bands, all_bands, low_bands, all_bands)
    return tuple(half + torch.fft.ifft(basis @ ((value-midpoint)*mask), norm='ortho')
                 for value, mask in zip(values, locations, strict=True))
