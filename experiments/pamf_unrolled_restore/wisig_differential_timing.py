"""Score fixed WiSig preamble shifts with locally normalized complex products."""

import numpy as np

from .wisig_joint_preamble import OBSERVED, STARTS, WEIGHT, observable, waveform
from .wisig_smooth_preamble import SMOOTH, aligned, choose, estimate


LAGS = (1, 2, 4)
ARMS = ('smooth', 'differential', 'joint')
WEIGHTS = WEIGHT.astype(np.float32)**2


def product(later, earlier):
    power = np.abs(later)**2 + np.abs(earlier)**2
    return 2. * later * earlier.conj() / np.maximum(power, 1e-10)


def bank():
    relative = OBSERVED[None] - STARTS[:, None]
    base = waveform(relative)
    return tuple(np.asarray(product(base, waveform(relative-lag)).T, dtype=np.complex64)
                 for lag in LAGS)


TEMPLATES = bank()


def differential(y, template=TEMPLATES):
    level = np.maximum(np.sqrt(np.mean(np.abs(y)**2, -1, keepdims=True)), 1e-8)
    y = y/level
    later = y[:, OBSERVED]
    scores = []
    for lag, expected in zip(LAGS, template, strict=True):
        earlier = y[:, OBSERVED-lag]
        observed = product(later, earlier)
        reliability = np.minimum(np.abs(later)*np.abs(earlier), 2.) * WEIGHTS
        candidate = (observed * reliability) @ expected.conj()
        mismatch = (np.abs(observed)**2 * reliability).sum(-1, keepdims=True)
        mismatch = mismatch + reliability @ (np.abs(expected)**2)
        norm = np.maximum(reliability.sum(-1, keepdims=True), 1e-8)
        scores.append(1. + (candidate.real - .5*mismatch)/norm)
    return np.mean(scores, axis=0)


def observations(iq, templates=TEMPLATES):
    raw, info, sync = observable(iq)
    smooth, _, response = estimate(sync, SMOOTH)
    phase_diff = differential(sync, templates)
    fits = {key: choose(value) for key, value in zip(ARMS, (smooth, phase_diff, .5*(smooth+phase_diff)), strict=True)}
    return raw, info, response, fits


def readouts(iq, thresholds):
    raw, info, response, fits = observations(iq)
    variants = {'direct': iq}
    for name, (delta, margin) in fits.items():
        selected = np.where(margin > thresholds[name], delta, 0)
        wave = aligned(raw, selected, info.coarse_rad_per_sample, response, varying=True)
        variants[name] = np.stack((wave.real, wave.imag), axis=1).astype(iq.dtype, copy=False)
    return variants, info
