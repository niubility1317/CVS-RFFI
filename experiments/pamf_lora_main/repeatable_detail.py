"""Identify repeatable packet spectral directions using source IQ pairs only."""

import torch

from .dura_repair import complex_iq, no_tone, real_iq
from .residual_dura_bandfit import band_masks


def estimate_basis(clean, first, second, rank):
    """Use clean variation and paired draw differences to resolve signal/noise directions."""
    clean = clean.double()
    second_difference = (first.double() - second.double()) / (2. ** .5)
    signal = clean - clean.mean(0)
    noise = second_difference - second_difference.mean(0)
    signal = (signal.T @ signal) / (len(signal) - 1)
    noise = (noise.T @ noise) / (len(noise) - 1)
    ridge = noise.diag().mean().clamp_min(1e-10) * .15
    regularized = noise + torch.eye(len(noise), dtype=noise.dtype, device=noise.device) * ridge
    chol = torch.linalg.cholesky(regularized)
    white = torch.linalg.solve_triangular(chol, signal, upper=False)
    white = torch.linalg.solve_triangular(chol, white.T, upper=False)
    values, vectors = torch.linalg.eigh((white + white.T) * .5)
    # Return an ordinary orthonormal basis in physical IQ coordinates, so its
    # projection and complement neither overlap nor amplify a proposal.
    basis = torch.linalg.qr(chol @ vectors[:, -rank:]).Q
    return basis.to(dtype=first.dtype), values.flip(0)


def basis_spec(device):
    near, mid = band_masks(device)
    return (near, mid), near | mid


def pack_spectrum(spectrum, mask):
    selected = spectrum[..., mask].flatten(1)
    return torch.cat((selected.real, selected.imag), dim=1)


def matched_shape(iq, tracker):
    wave, _, *_ = tracker.track(iq)
    scale = wave.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return torch.fft.fft(no_tone(wave / scale), norm='ortho')


def expand_spectrum(base, features, mask):
    real, imag = features.chunk(2, dim=1)
    spectrum = base.clone()
    spectrum[..., mask] = torch.complex(real, imag).reshape_as(base[..., mask])
    return spectrum


def equal_dose(base, replacement, masks):
    out = base.clone()
    for band in masks:
        current = base[..., band]
        proposed = replacement[..., band]
        dose = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
        length = torch.linalg.vector_norm(proposed, dim=-1, keepdim=True)
        out[..., band] = torch.where(length > 1e-7, proposed * (dose / length.clamp_min(1e-7)), current)
    return out


def clipped_dose(base, replacement, masks):
    out = base.clone()
    for band in masks:
        original = torch.linalg.vector_norm(base[..., band], dim=-1, keepdim=True)
        proposed = replacement[..., band]
        length = torch.linalg.vector_norm(proposed, dim=-1, keepdim=True)
        out[..., band] = proposed * (original / length.clamp_min(1e-7)).clamp_max(1.)
    return out


def project_corrections(reference, proposed, tracker, basis):
    wave, rotation, *_ = tracker.track(reference)
    rms = wave.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    delta = complex_iq(proposed - reference) * rotation.conj() / rms
    spectrum = torch.fft.fft(delta, norm='ortho')
    masks, mask = basis_spec(spectrum.device)
    vector = pack_spectrum(spectrum, mask)
    stable = (vector @ basis) @ basis.T
    rows = ((stable, equal_dose), (vector - stable, equal_dose), (stable, clipped_dose))
    restored = []
    drift = 0.
    for row, rescale in rows:
        projected = rescale(spectrum, expand_spectrum(spectrum, row, mask), masks)
        if rescale is equal_dose:
            for band in masks:
                norm = torch.linalg.vector_norm
                drift = max(drift, float((norm(projected[..., band], dim=-1) - norm(spectrum[..., band], dim=-1)).abs().max()))
        edit = torch.fft.ifft(projected, norm='ortho') * rotation * rms
        restored.append(reference + real_iq(edit))
    return restored, drift
