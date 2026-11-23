"""Compare MMSE and covariance-matching LoRa private spectral estimates."""

import torch

from .spectral_covariance import real_coordinates


def contrast_basis(n, *, dtype, device):
    rows = torch.arange(n, device=device)[:, None]
    counts = torch.arange(1, n, device=device)[None, :]
    steps = torch.where(rows < counts, 1, torch.where(rows == counts, -counts, 0))
    return steps.to(dtype) / (counts * (counts + 1)).to(dtype).sqrt()


def modal_transport(residual, axes, wiener_gain):
    """Apply two nonnegative gains in the same orthonormal contrast axes."""
    modal = (axes.mT @ residual.unsqueeze(-1)).squeeze(-1)
    return tuple((axes @ (modal * gain).unsqueeze(-1)).squeeze(-1)
                 for gain in (wiener_gain, wiener_gain.sqrt()))


def replace_private(full, bins, estimate):
    existing = full.index_select(-1, bins)
    new = existing.mean(1, keepdim=True) + estimate - estimate.mean(1, keepdim=True)
    return full.index_copy(-1, bins, new)


def private_modes(measured, center, band_noise, bands, prior):
    """Condition each clean-private variance on the projected receiver noise."""
    bins = prior.correlated_bins
    # Both I and Q use half of the estimated proper-complex noise variance.
    psi = contrast_basis(measured.shape[1], dtype=band_noise.dtype, device=measured.device)
    noise = .5 * band_noise.transpose(1, 2)
    covariance = psi.T @ (noise[..., None] * psi)
    eigenvalues, rotation = torch.linalg.eigh(covariance)
    band_index = bands.index_select(1, bins).argmax(0).repeat(2)
    axes = (psi @ rotation).index_select(1, band_index)
    values = eigenvalues.index_select(1, band_index).clamp_min(0.)
    variance = prior.private_real[None, :, None]
    gain = variance / (variance + values)
    subset = center.index_select(-1, bins)
    observed = measured.index_select(-1, bins)
    residual = real_coordinates(observed - subset).transpose(1, 2)
    estimates = modal_transport(residual, axes, gain)
    outputs = []
    for estimate in estimates:
        real, imag = estimate.transpose(1, 2).chunk(2, dim=-1)
        outputs.append(subset + torch.complex(real, imag))
    return tuple(outputs)
