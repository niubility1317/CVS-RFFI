"""A four-mode source waveform prior with the original Gaussian moments."""

import numpy as np
import torch

from .covariance_inverse import SHRINKAGE


def fit_source_modes(samples, mean, covariance):
    """Fit four clean-IQ modes without labels or validation observations."""
    eigenvalues, axes = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError('Positive original IQ covariance required')
    whiten = axes[:, -16:][:, ::-1] / np.sqrt(eigenvalues[-16:][::-1])
    coordinates = np.empty((len(samples), whiten.shape[1]), dtype=np.float64)
    for start in range(0, len(samples), 4096):
        batch = samples[start:start+4096]
        coordinates[start:start+len(batch)] = (batch - mean) @ whiten
    quadrant = 2 * (coordinates[:, 0] >= 0) + (coordinates[:, 1] >= 0)
    centers = group_means(coordinates, quadrant)
    for _ in range(8):
        groups = nearest_mode(coordinates, centers)
        centers = group_means(coordinates, groups)
    groups = nearest_mode(coordinates, centers)
    weights = np.bincount(groups, minlength=4) / len(samples)
    means = group_means(samples, groups)
    offsets = np.sqrt(1-SHRINKAGE) * (means - weights @ means)
    within = covariance - (offsets.T * weights) @ offsets
    within = .5 * (within + within.T)
    np.linalg.cholesky(within)
    return offsets, within, weights


def group_means(values, groups):
    counts = np.bincount(groups, minlength=4)
    if np.any(counts == 0):
        raise ValueError('Four occupied source IQ modes required')
    sums = np.zeros((4, values.shape[1]), dtype=np.float64)
    for start in range(0, len(values), 4096):
        batch = values[start:start+4096]
        labels = groups[start:start+len(batch)]
        for mode in range(4):
            sums[mode] += batch[labels == mode].sum(axis=0, dtype=np.float64)
    return sums / counts[:, None]


def nearest_mode(coordinates, centers):
    return (np.square(centers).sum(axis=1) - 2 * coordinates @ centers.T).argmin(axis=1)


def mixture_posterior(action, observed, mean, within, offsets, weights, variance):
    """Return exact posterior mean and mode probabilities with one FP64 factorization."""
    ac = action @ within
    observation_covariance = ac @ action.transpose(-1, -2)
    observation_covariance.diagonal(dim1=-2, dim2=-1).add_(variance[:, None])
    centered = observed - action @ mean
    projected = action @ offsets.T
    rhs = torch.cat((centered[:, :, None], projected), dim=-1)
    solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(observation_covariance))
    error = (observation_covariance @ solved - rhs).norm(dim=1) / rhs.norm(dim=1).clamp_min(1e-30)
    if not torch.isfinite(error).all() or error.max() > 1e-8:
        raise ValueError('Mixture posterior solve residual exceeds 1e-8')
    evidence = (projected * (solved[:, :, :1] - .5 * solved[:, :, 1:])).sum(dim=1)
    probabilities = (weights.log() + evidence).softmax(dim=1)
    filtered = ac.transpose(-1, -2) @ solved
    innovations = offsets.T - filtered[:, :, 1:]
    restored = mean + filtered[:, :, 0] + (innovations @ probabilities[:, :, None]).squeeze(-1)
    return restored, probabilities
