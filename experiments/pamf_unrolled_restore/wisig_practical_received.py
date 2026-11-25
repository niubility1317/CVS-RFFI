"""Blind WiSig estimation using only a 256-sample receive observation."""

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import dpss
from scipy.special import logsumexp


SAMPLES = 256
# WiSig legacy preamble carriers occupy less than +/-0.33 cycles/sample.
# Keep transition leakage outside the subspace used to measure noise.
GUARD_CUTOFF = .38
GUARD_START = 204
LOG_VARIANCE_OFFSETS = np.linspace(-.65, .65, 27)


@dataclass(frozen=True)
class Prior:
    guard: np.ndarray
    modes: np.ndarray
    power: np.ndarray
    log_noise: np.ndarray


def guard_basis():
    """Orthogonal directions outside the occupied Wi-Fi frequency band."""
    return dpss(SAMPLES, SAMPLES * GUARD_CUTOFF, Kmax=SAMPLES)[GUARD_START:].T.copy()


def guard_power(iq, guard):
    return np.mean(np.abs(iq @ guard)**2, axis=-1).clip(1e-12)


def fit(received):
    """Learn receive-waveform covariance from source observations alone."""
    received = np.asarray(received, dtype=np.complex128)
    guard = guard_basis()
    noise = guard_power(received, guard)
    covariance = received.T @ received.conj() / len(received)
    eigenvalues, modes = np.linalg.eigh(covariance)
    power = np.maximum(eigenvalues.real - np.mean(noise), 0.)
    centers = np.quantile(np.log(noise), np.linspace(.01, .99, 49))
    return Prior(guard, modes, power, centers)


def noise_posterior(iq, observed, prior):
    """Estimate packet noise from guard energy and a source-only scale prior."""
    measured = guard_power(iq, prior.guard)
    log_noise = np.log(measured[:, None]) + LOG_VARIANCE_OFFSETS
    noise = np.exp(log_noise)
    weak = prior.power <= .60 * measured[:, None]
    covariance = prior.power[None, None, :] + noise[..., None]
    log_likelihood = -np.sum(weak[:, None, :] * (
        np.log(covariance) + np.abs(observed[:, None, :])**2 / covariance), axis=-1)
    guard_likelihood = -prior.guard.shape[1] * (log_noise + measured[:, None] / noise)
    support = -.5 * ((log_noise[..., None] - prior.log_noise) / .18)**2
    # The projections may share samples. Average their log likelihoods.
    log_likelihood = .70 * log_likelihood + .30 * guard_likelihood
    logits = log_likelihood + logsumexp(support, axis=-1)
    weight = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    return np.sum(weight * noise, axis=-1)


def estimates(received, prior, scales=(1.,)):
    """Return effective in-band waveform and additive variance from receive IQ."""
    iq = np.asarray(received, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] != SAMPLES or not len(iq) or not np.isfinite(iq).all():
        raise ValueError("Expected nonempty finite complex received IQ of shape (packets, 256)")
    observed = iq @ prior.modes.conj()
    variance = noise_posterior(iq, observed, prior)
    waves = []
    for scale in scales:
        strength = scale * prior.power
        weight = strength / (strength + variance[:, None])
        waves.append((observed * weight) @ prior.modes.T)
    return np.stack(waves), variance
