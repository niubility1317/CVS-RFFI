"""Practical paired-IQ estimation with offline covariance and one small eigensolve."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .wisig_independent_channel_audit import N, reflect


TIME = np.arange(N)
POSITION = np.linspace(-1., 1., N)
LAGS = np.arange(-10, 75)
OFFSETS = reflect(TIME[:, None] - LAGS)
SCALES = np.array([.035, .10, .28, .70, 1.4, 3.2, 7.])
SCALE_PROBABILITY = np.array([.035, .08, .16, .24, .24, .17, .075])
MAX_RANK = 18


def rms(x):
    return max(float(np.sqrt(np.mean(abs(x)**2))), 1e-12)


def coordinates(x, y, train=TIME):
    level = rms(y[train])
    return (x / rms(x))[OFFSETS], y / level, level


def residual_power(matrix, signal):
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(float(singular[0]) * .007, 1e-12)
    removed = np.sum(abs(left[:, keep].conj().T @ signal)**2)
    return max(float((np.vdot(signal, signal).real - removed) /
                     max(len(signal) - np.count_nonzero(keep), 1)), 1e-12)


def learn(clean, received):
    grams, moments, levels = [], [], []
    for x, y in zip(clean, received):
        design, signal, _ = coordinates(x, y)
        gram = design.conj().T @ design
        grams.append(gram)
        moments.append(design.conj().T @ signal)
        levels.append(float(np.vdot(signal, signal).real))
    gram_diag = np.asarray([np.diag(gram).real for gram in grams])
    matched = np.asarray(moments)
    alignment = np.median(abs(matched)**2 / gram_diag**2, axis=0)
    weight = np.maximum(alignment - np.median(alignment), 0.)
    weight /= max(float(weight.sum()), 1e-12)
    anchor = int(np.argmax(weight))
    identity = np.eye(len(LAGS))
    cov = np.diag(.75 * weight + .25 / len(weight))
    noise = np.full(len(clean), .15)
    for _ in range(12):
        inverse = cho_solve(cho_factor(cov, lower=True), identity)
        total = np.zeros_like(cov, dtype=complex)
        for i, (gram, z, level) in enumerate(zip(grams, matched, levels)):
            system = cho_factor(gram / noise[i] + inverse, lower=True)
            conditioned = cho_solve(system, identity)
            tap = conditioned @ z / noise[i]
            total += conditioned + np.outer(tap, tap.conj())
            residual = level - 2 * np.vdot(tap, z).real + np.vdot(tap, gram @ tap).real
            residual += np.sum(gram * conditioned.T).real
            noise[i] = max(float(.35 * noise[i] + .65 * residual / N), 1e-6)
        total /= len(clean)
        cov = .995 * total + .005 * np.diag(np.diag(total).real) + 1e-9 * identity
    value, vector = np.linalg.eigh(cov)
    factors = vector[:, -MAX_RANK:][:, ::-1] * np.sqrt(value[-MAX_RANK:][::-1].clip(1e-12))
    return factors, anchor


def fit_components(x, y, train, factor, anchor, rank):
    delays, normalized, level = coordinates(x, y, train)
    core = delays @ factor[:, :rank]
    first = delays[:, anchor]
    # Linear/quadratic phase and weak receiver-image modes.
    branch = np.column_stack((.045 * first * POSITION, .020 * first * POSITION**2,
                             .024 * core[:, :3].conj(), .001 * first.conj() * POSITION))
    design = np.column_stack((core, branch))
    pieces = design[train]
    eigen, basis = np.linalg.eigh(pieces.conj().T @ pieces)
    eigen = eigen.clip(1e-16)
    coord = basis.conj().T @ (pieces.conj().T @ normalized[train])
    projected = abs(coord)**2 / eigen
    supported = eigen > eigen[-1] * 1e-6
    white_energy = max(float(np.vdot(normalized[train], normalized[train]).real -
                             projected[supported].sum()), 1e-12)
    white_count = len(train) - np.count_nonzero(supported)
    noise = white_energy / max(white_count, 1)
    return design @ basis, coord, eigen, projected, white_energy, white_count, noise, level


def posterior(components, concentration=1., variance_shrink=0.):
    rotation, coord, eigen, energy, white, count, noise, level = components
    noises = noise * np.array([.82, .90, .96, 1., 1.04, 1.10, 1.20])
    prior = SCALES * concentration
    channels = eigen[:, None, None] * prior[None, None, :]
    combined = noises[None, :, None] + channels
    is_mode = eigen > eigen[-1] * 1e-6
    covariance_loss = np.log(combined[is_mode]).sum(axis=0)
    covariance_loss += (energy[is_mode, None, None] / combined[is_mode]).sum(axis=0)
    loss = covariance_loss + (count * np.log(noises) + white / noises)[:, None]
    loss -= np.log(SCALE_PROBABILITY)[None, :]
    lik = np.exp(loss.min() - loss)
    lik /= lik.sum()
    coef = coord * np.einsum("ijk,jk->i", prior[None, None, :] / combined, lik)
    waveform = level * (rotation @ coef)
    more = float(np.sum(noises * lik.sum(axis=1)))
    power = level**2 * ((1. - variance_shrink) * noise + variance_shrink * more)
    return waveform, power
