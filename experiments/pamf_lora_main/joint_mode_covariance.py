"""Condition the joint I/Q covariance of common and differential chirp detail."""

import torch


def paired_bins(device):
    positive = torch.arange(1, 33, device=device)
    return torch.stack((positive, 256 - positive), dim=-1).reshape(8, 8)


def chirp_basis(like):
    """Orthonormal contrasts with the constant mode as the first row."""
    n = 6
    i = torch.arange(n, device=like.device)
    row, column = i[:, None], i[None, :]
    contrast = (column < row).to(like.real.dtype) - row * (column == row)
    factor = (row * (row + 1)).to(contrast.dtype).clamp_min(1).sqrt()
    return torch.where(row == 0, n ** -.5, contrast / factor)


def encode(spectrum, basis, bins):
    paired = spectrum[..., bins].transpose(1, 2)
    modes = basis.to(paired.dtype) @ paired
    return torch.view_as_real(modes).flatten(-3)


def decode(value, basis, bins, like):
    modes = torch.view_as_complex(value.reshape(*value.shape[:2], 6, bins.shape[-1], 2))
    paired = basis.T.to(modes.dtype) @ modes
    output = torch.zeros_like(like)
    output[..., bins] = paired.transpose(1, 2)
    return output


def moments(training):
    """Estimate moments once from the permitted clean source spectra."""
    mean = training.mean(0)
    centered = (training - mean).transpose(0, 1)
    covariance = centered.mT @ centered / (len(training) - 1)
    diagonal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-7)
    return mean, .75 * covariance + .25 * torch.diag_embed(diagonal)


def without_cross_covariance(covariance, bins):
    is_common = torch.arange(covariance.shape[-1], device=covariance.device) < 2 * bins.shape[-1]
    return covariance * (is_common[:, None] == is_common)


def noise_covariance(variance, basis, bins):
    """Propagate unequal per-chirp proper complex variance into the mode basis."""
    mode_cov = (basis * (.5 * variance[:, None, :])) @ basis.T
    width = 2 * bins.shape[-1]
    eye = torch.eye(width, device=basis.device, dtype=basis.dtype)
    blocks = mode_cov[:, :, None, :, None] * eye[None, None, :, None, :]
    return blocks.reshape(len(variance), 1, 6 * width, 6 * width)


def noise_edit(measurement, mean, signal_covariance, observed_covariance):
    """Subtract the conditional mean of the observation noise."""
    system = signal_covariance + observed_covariance
    factor = torch.linalg.cholesky(system)
    whitened = torch.cholesky_solve((measurement - mean).unsqueeze(-1), factor)
    return -(observed_covariance @ whitened).squeeze(-1)


def match_rank_direction(rank, correction, rms, fraction=.25):
    """Change the direction at the parent's physical IQ dose in each near band."""
    indices = torch.fft.fftfreq(rank.shape[-1], device=rank.device).abs()
    output = rank.clone()
    for band in ((indices > 0) & (indices <= 1 / 32), (indices > 1 / 32) & (indices <= 1 / 8)):
        original = rank[..., band]
        blended = torch.lerp(original, correction[..., band], fraction)
        target_norm = torch.linalg.vector_norm(original * rms, dim=(-2, -1), keepdim=True)
        new_norm = torch.linalg.vector_norm(blended * rms, dim=(-2, -1), keepdim=True)
        output[..., band] = torch.where(new_norm > 1e-10, blended * (target_norm / new_norm.clamp_min(1e-10)),
                                        original)
    return output
