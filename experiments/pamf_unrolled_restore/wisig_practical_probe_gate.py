"""Combine localized and diffuse FIR estimates using observed pilot reliability."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert

from . import wisig_practical_probe_blind as blind
from . import wisig_practical_probe_relative as relative
from . import wisig_practical_receive_weighted as guard_model


REPEAT_START = 100
REPEAT_COUNT = 5
REPEAT_MODES = 40


@dataclass(frozen=True)
class NoisePrior:
    guard: np.ndarray
    color: np.ndarray
    null: np.ndarray


def repeat_basis(guard):
    period = blind.timing.PERIOD
    contrast = np.kron(helmert(REPEAT_COUNT), np.eye(period)).T
    span = np.zeros((blind.receive.SAMPLES, contrast.shape[-1]))
    span[REPEAT_START:REPEAT_START + period * REPEAT_COUNT] = contrast
    independent = span - guard @ (guard.conj().T @ span)
    left, strength, _ = np.linalg.svd(independent, full_matrices=False)
    return left[:, strength > 1e-4]


def fit_noise(received):
    samples = blind.pilot_sync(received)
    guard, color = guard_model.fit_guard(samples)
    noise = blind.receive.guard_power(samples, guard)
    span = repeat_basis(guard)
    quiet = noise <= np.quantile(noise, .3)
    residual = samples[quiet] @ span
    covariance = residual.conj().T @ residual / quiet.sum()
    _, vectors = np.linalg.eigh(covariance)
    return NoisePrior(guard, color, span @ vectors[:, :REPEAT_MODES])


def variance_candidates(synced, prior):
    power = abs(synced @ prior.guard)**2
    repeat = blind.receive.guard_power(synced, prior.null)
    return pool_variance(power, repeat, prior)


def pool_variance(power, repeat, prior):
    raw = power.mean(axis=-1).clip(1e-12)
    colored = guard_model.guard_variance(power, prior.color, 1.65)
    # Use independent repeat evidence when its estimated scale agrees with the guard.
    disagreement = np.maximum(abs(np.log(repeat / colored)) - .12, 0.)
    consistency = np.exp(-.5 * (disagreement / .42)**2)
    strength = prior.null.shape[-1] / (prior.guard.shape[-1] + prior.null.shape[-1])
    pooled = colored + strength * consistency * (repeat - colored)
    return raw, colored, pooled


def estimate(received, prior, center, scale, matrix, noise_prior):
    iq = np.asarray(received, dtype=np.complex128)
    piloted, noise = blind.estimate(iq, prior)
    broad, _ = relative.estimate(iq, prior, center, scale, matrix[None, ...])
    broad = broad[0]
    diffuse = 1. - np.interp(np.log(noise), prior.noise_knots, prior.coherence)
    anchor = piloted[blind.ARMS.index("pilot_phase")]
    blend = anchor + diffuse[:, None] * (broad - anchor)
    conservative = anchor + diffuse[:, None]**1.5 * (broad - anchor)
    _, colored, pooled = variance_candidates(blind.pilot_sync(iq), noise_prior)
    return np.stack((blend, blend, blend, conservative)), np.stack((noise, colored, pooled, pooled))
