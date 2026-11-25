"""IQ-only empirical channel covariance and posterior averaging for Low Urban."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import wisig_refined_channel as baseline
from .wisig_independent_channel_audit import N, reflect


TIME = np.arange(N)
U = np.linspace(-1., 1., N)
OFFSETS = reflect(TIME[:, None] - np.arange(-10, 75))
RATES = np.array([-.65, -.28, 0., .28, .65])
RATE_PRIOR = np.array([.02, .13, .70, .13, .02])
SCALES = 2. * np.array([.055, .16, .42, 1., 2.4, 6.])
SCALE_PRIOR = np.array([.02, .10, .21, .32, .24, .11])
RANK = 14


def normalize(clean, observed, observed_indices=TIME):
    x_rms = max(float(np.sqrt(np.mean(abs(clean)**2))), 1e-12)
    y_rms = max(float(np.sqrt(np.mean(abs(observed[observed_indices])**2))), 1e-12)
    return clean / x_rms, observed / y_rms, y_rms


def baseline_fit(x, y, config, indices):
    small = config["adaptive"]
    fits, rho, tones = baseline.adaptive.candidate_bank(x, y, indices)
    wave, variance, _ = baseline.adaptive.estimate(
        fits, rho, tones, x, y, indices, small["complexity"], small["shrink"], 0.,
    )
    bank = baseline.refined_bank(x, y, indices, wave, rho)
    prior = baseline.solutions(bank, len(indices), (config["penalty"],))[config["penalty"]]
    return wave + config["mix"] * (prior - wave), variance


def fit_covariance(clean, observed, config):
    grams, matches, powers, variances = [], [], [], []
    for original, received in zip(clean, observed):
        x, y, _ = normalize(original, received)
        _, noise = baseline_fit(x, y, config, TIME)
        design = x[OFFSETS]
        grams.append(design.conj().T @ design)
        matches.append(design.conj().T @ y)
        powers.append(float(np.vdot(y, y).real))
        variances.append(noise)
    diagonal = np.asarray([np.diag(gram).real for gram in grams])
    identity = np.eye(diagonal.shape[1])
    matches, noise = np.asarray(matches), np.asarray(variances)
    alignment = (abs(matches)**2 - diagonal * noise[:, None]) / (diagonal**2).clip(1e-15)
    energy = np.maximum(np.quantile(alignment, .65, axis=0), 0.)
    energy = np.maximum(energy - np.median(energy), 0.)
    weight = energy / max(float(energy.sum()), 1e-10)
    cov = np.diag(.5 * weight + .3 / len(weight))
    for _ in range(10):
        inverse = cho_solve(cho_factor(cov, lower=True), identity)
        update = np.zeros(cov.shape, complex)
        for i, (gram, match, power) in enumerate(zip(grams, matches, powers)):
            sigma = max(float(noise[i]), .002)
            conditional = cho_solve(cho_factor(gram / sigma + inverse, lower=True), identity)
            coef = (conditional @ match) / sigma
            update += conditional + np.outer(coef, coef.conj())
            residual = power - 2. * np.vdot(coef, match).real + np.vdot(coef, gram @ coef).real
            residual += np.sum(gram * conditional.T).real
            noise[i] = np.clip(.65 * noise[i] + .35 * residual / N, .001, 1.)
        update /= len(clean)
        cov = .98 * update + .02 * np.diag(np.diag(update).real) + 1e-8 * identity
    value, vector = np.linalg.eigh(cov)
    return vector[:, -RANK:] * np.sqrt(np.maximum(value[-RANK:], 1e-9))


def posterior(clean, observed, noise, indices, covariance):
    unit = clean[OFFSETS] @ covariance
    selected = unit[indices]
    eigen, vectors = np.linalg.eigh(selected.conj().T @ selected)
    eigen = eigen.clip(1e-16)
    rotations = np.exp(1j * U[:, None] * RATES)
    projections = vectors.conj().T @ (selected.conj().T @ (observed[indices, None] * rotations[indices].conj()))
    load = noise + eigen[:, None] * SCALES[None, :]
    coefficient = SCALES[None, :] / load
    log_weight = np.einsum("dr,ds->rs", abs(projections)**2, coefficient / noise)
    log_weight -= np.log(load / noise).sum(axis=0)[None, :]
    log_weight += np.log(RATE_PRIOR)[:, None] + np.log(SCALE_PRIOR)[None, :]
    probability = np.exp(log_weight - log_weight.max())
    probability /= probability.sum()
    average = np.einsum("dr,ds,rs->dr", projections, coefficient, probability)
    waveform = np.sum(rotations * ((unit @ vectors) @ average), axis=1)
    shrinkage = eigen[:, None] * coefficient
    dof = (2. * shrinkage - shrinkage**2).sum(axis=0) @ probability.sum(axis=0)
    residual = abs(observed[indices] - waveform[indices])**2
    fitted_noise = float(np.sum(residual) / max(len(indices) - dof, 1.))
    return waveform, float(np.clip(fitted_noise, .5 * noise, 2. * noise))


def mix_weight(y, noise, indices, rule):
    strength, knee, power = rule
    quality = max(float(np.mean(abs(y[indices])**2) / noise - 1.), 0.)
    return strength / (1. + quality / knee)**power, quality
