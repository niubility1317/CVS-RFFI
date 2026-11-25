"""Receive-only covariance with a colored guard-band nuisance component."""

from dataclasses import dataclass

import numpy as np

from . import wisig_practical_received as baseline


@dataclass(frozen=True)
class Prior:
    guard: np.ndarray
    color: np.ndarray
    modes: np.ndarray
    power: np.ndarray


def fit_guard(received):
    guard = baseline.guard_basis()
    coefficients = received @ guard
    energy = np.mean(abs(coefficients)**2, axis=-1)
    # Learn directions of leakage from source observations with weak guard energy.
    quiet = coefficients[energy <= np.quantile(energy, .25)]
    covariance = quiet.T @ quiet.conj() / len(quiet)
    power, modes = np.linalg.eigh(covariance)
    isotropic = np.median(power[:len(power)//4])
    color = (power - isotropic).clip(0.)
    color /= max(float(color.mean()), 1e-12)
    return guard @ modes.conj(), color


def guard_variance(power, color, penalty=1.):
    measured = power.mean(axis=-1).clip(1e-12)
    contrast = color - color.mean()
    normalization = float(contrast @ contrast)
    if normalization < 1e-12:
        return measured
    amplitude = power @ contrast / normalization
    # Centered orthogonal powers have variance sigma^4 under circular white noise.
    strength = (amplitude - penalty * measured / np.sqrt(normalization)).clip(0.)
    return np.maximum(measured - strength * color.mean(), .20 * measured)


def fit(received, exponent, penalty):
    x = np.asarray(received, dtype=np.complex128)
    guard, color = fit_guard(x)
    variance = guard_variance(abs(x @ guard)**2, color, penalty)
    weight = np.maximum(variance, np.quantile(variance, .13))**(-exponent)
    weight /= weight.sum()
    values, modes = np.linalg.eigh((x.T * weight) @ x.conj())
    return Prior(guard, color, modes, (values - variance @ weight).clip(0.))


def covariance_estimate(observed, prior, scale, variance):
    strength = scale * prior.power
    gain = strength / (strength + variance[:, None])
    waveform = (observed * gain) @ prior.modes.T
    return waveform, gain


STF_START = 100
STF_LENGTH = 100
PERIOD = 20


def repeat_refinement(iq, waveform, variance, blend):
    """Use repeat consensus only where the observed short-preamble model agrees."""
    selection = slice(STF_START, STF_START + STF_LENGTH)
    blocks = iq[:, selection].reshape(-1, STF_LENGTH // PERIOD, PERIOD)
    count = blocks.shape[1]
    phase = np.angle(np.sum(blocks[:, 1:] * blocks[:, :-1].conj(), axis=(1, 2)))
    rotation = np.exp(-1j * phase[:, None, None] * np.arange(count)[None, :, None])
    aligned = blocks * rotation
    average = aligned.mean(axis=1, keepdims=True)
    model_residual = np.mean(abs(aligned - average)**2, axis=(1, 2))
    expected = variance * (1. - 1. / count)
    mismatch = (model_residual - expected).clip(0.)
    fraction = blend * expected / (expected + 2. * mismatch + 1e-12)
    proposal = (average * rotation.conj()).reshape(len(iq), STF_LENGTH)
    tapers = np.minimum(1., np.minimum(np.arange(1, STF_LENGTH + 1), np.arange(STF_LENGTH, 0, -1))/PERIOD)
    out = waveform.copy()
    out[:, selection] += fraction[:, None] * tapers * (proposal - out[:, selection])
    return out
