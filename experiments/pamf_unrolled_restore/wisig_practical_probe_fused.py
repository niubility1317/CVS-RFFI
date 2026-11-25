"""Reuse the pilot projection across blind channel and noise estimates."""

from dataclasses import dataclass

import numpy as np

from . import wisig_practical_probe_gate as gate


@dataclass(frozen=True)
class Estimator:
    prior: gate.blind.Prior
    noise_prior: gate.NoisePrior
    analysis: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    matrix: np.ndarray
    protection_scale: float


def prepare(prior, center, scale, matrix, noise_prior, protection_scale):
    pilot = np.zeros((gate.blind.receive.SAMPLES, len(prior.decoder)), dtype=np.complex128)
    pilot[gate.blind.PILOT_POINTS] = prior.decoder.T
    analysis = np.column_stack((pilot, noise_prior.guard, noise_prior.null))
    return Estimator(prior, noise_prior, analysis, center, scale, matrix, protection_scale)


def components(received, cfg):
    iq = np.asarray(received, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] != gate.blind.receive.SAMPLES or not np.isfinite(iq).all():
        raise ValueError("Only finite post-channel 256-point receive IQ is accepted")
    basis = cfg.prior
    noise = gate.blind.receive.guard_power(iq, basis.guard)
    synchronized = gate.blind.pilot_sync(iq)
    projection = synchronized @ cfg.analysis
    pilot, guard, repeat = np.split(projection, [len(basis.decoder), len(basis.decoder) + cfg.noise_prior.guard.shape[1]], axis=-1)
    rx_energy = np.mean(abs(iq)**2, axis=-1)
    coherence, phase, _ = gate.blind.phase_statistics(noise, pilot, basis)
    features = gate.relative.feature_columns(pilot, rx_energy, noise)
    design = gate.relative.regressors(features, noise, cfg.center, cfg.scale)
    anchor = np.zeros((len(iq), gate.blind.KERNEL_LENGTH), dtype=np.complex128)
    anchor[:, basis.anchor] = np.sqrt((rx_energy - noise).clip(0., 1.5)) * phase
    diffuse = 1. - coherence
    repeat_power = np.mean(abs(repeat)**2, axis=-1).clip(1e-12)
    _, colored, pooled = gate.pool_variance(abs(guard)**2, repeat_power, cfg.noise_prior)
    # Reduce aggressive subtraction when the source prior predicts diffuse reception.
    trust = .25 + .75 / (1. + (noise / cfg.protection_scale)**2)
    protected = noise * (pooled / noise)**trust
    return design, anchor, diffuse, (noise, colored, pooled, protected)


def estimate(received, cfg):
    design, anchor, diffuse, (noise, colored, pooled, protected) = components(received, cfg)
    difference = design @ cfg.matrix - anchor
    blend = anchor + diffuse[:, None] * difference
    conservative = anchor + diffuse[:, None]**1.5 * difference
    return (blend, blend, blend, conservative, blend), (noise, colored, pooled, pooled, protected)
