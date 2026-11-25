"""Evidence-weighted LOS and diffuse priors learned from paired Practical IQ."""

import numpy as np
from scipy.special import logsumexp

from . import wisig_practical_fast as fast


RANKS = (3, 9, 13, 11, 18)
# Source-U development rows are ordered as default six then strong six.
GROUPS = ((0, 3, 6, 9), (1, 2, 7, 8), (4, 10), (5, 11), tuple(range(12)))
LOG_OFFSETS = np.linspace(-.36, .36, 11)


def train(clean, received, indices, seed):
    rng = np.random.default_rng(seed)
    factors, anchors, spectra = [], [], []
    for group in GROUPS:
        subset = [rng.choice(indices, 36, replace=False) for _ in group]
        x = np.concatenate([clean[take] for take in subset])
        y = np.concatenate([received[g, take] for g, take in zip(group, subset)])
        factor, anchor = fast.learn(x, y)
        factors.append(factor)
        anchors.append(anchor)
        spectra.append(np.array([np.log(fast.fit_components(c, r, fast.TIME, factor, anchor, 18)[-2])
                                 for c, r in zip(x, y)]))
    return np.stack(factors), np.asarray(anchors), spectra


def observed_evidence(clean, received, indices, factors, anchors, spectra, *, return_basis=False):
    notes = []
    bases = []
    for factor, anchor, width, latent_noise in zip(factors, anchors, RANKS, spectra):
        fit = fast.fit_components(clean, received, indices, factor, anchor, width, return_basis=return_basis)
        if return_basis:
            fit, basis = fit
            bases.append(basis)
        rot, coef, eig, power, residual, count, scale, level = fit
        is_mode = eig > eig[-1] * 1e-6
        log_noise = np.log(scale) + LOG_OFFSETS
        noise = np.exp(log_noise)
        prior_channel = fast.SCALES[None, None, :]
        total = noise[None, :, None] + eig[:, None, None] * prior_channel
        gaussian = -np.log(total[is_mode]).sum(axis=0)
        gaussian -= (power[is_mode, None, None] / total[is_mode]).sum(axis=0)
        gaussian -= (count * log_noise + residual / noise)[:, None]
        gaussian += np.log(fast.SCALE_PROBABILITY)[None, :]
        noise_density = logsumexp(-.5 * ((log_noise[:, None] - latent_noise) / .28)**2, axis=1)
        noise_density -= np.log(len(latent_noise))
        gain = prior_channel / total
        posterior_energy = (np.einsum("ijk,i->jk", eig[:, None, None] * gain**2, abs(coef)**2) +
                            noise[:, None] * (eig[:, None, None] * gain).sum(axis=0)) / len(indices)
        notes.append((rot, coef, gain, gaussian, noise_density, posterior_energy, noise, level))
    return (notes, bases) if return_basis else notes


def estimate(evidence, noise_prior, relative, *, return_coordinates=False):
    likelihood, action = [], []
    for _, _, _, log_weight, density, power, noise, _ in evidence:
        logits = log_weight + noise_prior * density[:, None]
        likelihood.append(logits)
        action.append(logits - relative * np.log(power + .2 * noise[:, None]))
    evidence_total = logsumexp(np.stack(likelihood))
    action_total = logsumexp(np.stack(action))
    result = np.zeros(fast.N, complex)
    sigma = 0.
    coordinates = []
    for model, wt, linear in zip(evidence, likelihood, action):
        rot, coef, gain, _, _, _, noise, level = model
        weighted = coef * np.einsum("ijk,jk->i", gain, np.exp(linear - action_total))
        result += level * (rot @ weighted)
        if return_coordinates:
            coordinates.append(level * weighted)
        sigma += level**2 * float(np.sum(noise * np.exp(wt - evidence_total).sum(axis=1)))
    return (result, sigma, coordinates) if return_coordinates else (result, sigma)


def estimate_operator(clean, received, factors, anchors, spectra, config):
    """Export the frozen mixture's direct and conjugate operator before rank-one conversion."""
    evidence, bases = observed_evidence(clean, received, fast.TIME, factors, anchors, spectra, return_basis=True)
    wave, variance, coordinates = estimate(evidence, config['noise_prior'], config['relative'], return_coordinates=True)
    direct = np.zeros((fast.N, len(fast.LAGS)), complex)
    image = np.zeros_like(direct)
    for coordinate, basis, factor, anchor, rank in zip(coordinates, bases, factors, anchors, RANKS):
        coeff = basis @ coordinate / fast.rms(clean)
        direct += factor[:, :rank] @ coeff[:rank]
        direct[:, anchor] += .045 * coeff[rank] * fast.POSITION + .020 * coeff[rank+1] * fast.POSITION**2
        image += .024 * (factor[:, :3].conj() @ coeff[rank+2:rank+5])
        image[:, anchor] += .001 * coeff[rank+5] * fast.POSITION
    return wave, variance * config['variance_factor'], (fast.LAGS, direct, image)
