"""Score public WiSig preamble shifts under a smooth received complex response."""

import numpy as np

from .wisig_joint_preamble import IN_ANCHOR, OBSERVED, RIDGE, SIZE, STARTS, TAP_OFFSETS, WEIGHT, observable, waveform
from .wisig_timing import NOMINAL, estimate_timing


MODES = ('direct', 'transition', 'fixed_three', 'fixed_three_reflect',
         'smooth_full', 'smooth_held', 'smooth_held_reflect')


def blend_weights(coordinates):
    fraction = np.clip((coordinates - OBSERVED[0]) / (OBSERVED[-1] - OBSERVED[0]), 0., 1.)
    return np.maximum(1. - np.abs(3. * fraction[..., None] - np.arange(4)), 0.).astype(np.float32)


def ridge_bank(taps, *, varying=False):
    relative = OBSERVED[None, :, None] - STARTS[:, None, None] - np.asarray(taps)
    basis = (waveform(relative) * WEIGHT[None, :, None]).astype(np.complex64)
    if varying:
        basis = basis * blend_weights(OBSERVED)[None]
        differences = np.diff(np.eye(4, dtype=np.float32), axis=0)
        penalty = .03 * (differences.T @ differences)
    else:
        penalty = np.diag([0.] + [RIDGE] * (len(taps) - 1)).astype(np.float32)
    groups = (np.arange(len(OBSERVED)), *[np.flatnonzero((OBSERVED // 8) % 2 == i) for i in (0, 1)])
    banks = []
    for columns in groups:
        sample_basis = basis[:, columns]
        adjoint = sample_basis.conj().transpose(0, 2, 1)
        gram = adjoint @ sample_basis
        projection = adjoint.reshape(-1, len(columns)).T.copy()
        banks.append((columns, projection, gram, np.linalg.inv(gram + penalty)))
    return tuple(banks)


def estimate(synced, bank):
    observed = synced[:, OBSERVED] * WEIGHT.astype(np.float32)
    fits = []
    for columns, projection, gram, inverse in bank:
        part = observed[:, columns]
        rhs = (part @ projection).reshape(len(observed), len(STARTS), gram.shape[-1])
        response = np.einsum('dkj,bdj->bdk', inverse, rhs)
        energy = (part.real**2 + part.imag**2).sum(-1, keepdims=True)
        fits.append((rhs, response, gram, energy))
    rhs, response, _, energy = fits[0]
    apparent = np.sum((response.conj() * rhs).real, axis=-1) / np.maximum(energy, 1e-8)
    explained = []
    for (_, trained, _, _), (rhs, _, gram, _) in ((fits[1], fits[2]), (fits[2], fits[1])):
        projected = np.einsum('dkj,bdj->bdk', gram, trained)
        explained.append(np.sum((trained.conj() * (2*rhs - projected)).real, axis=-1))
    predictive = sum(explained) / np.maximum(fits[1][3] + fits[2][3], 1e-8)
    return apparent, predictive, response


def choose(scores):
    outer = np.where(~IN_ANCHOR, scores, -np.inf)
    closest = np.where(IN_ANCHOR, scores, -np.inf).max(1)
    return STARTS[outer.argmax(1)] - NOMINAL, outer.max(1) - closest


FIXED = ridge_bank(TAP_OFFSETS['joint_channel'])
SMOOTH = ridge_bank((0,), varying=True)
TIME = np.arange(SIZE)


def observations(iq):
    raw, info, synced = observable(iq)
    local, margin, _ = estimate_timing(raw)
    fixed, _, fixed_response = estimate(synced, FIXED)
    full, held, smooth_response = estimate(synced, SMOOTH)
    fits = {'transition': (local, margin, None),
            'fixed_three': (*choose(fixed), fixed_response),
            'smooth_full': (*choose(full), smooth_response),
            'smooth_held': (*choose(held), smooth_response)}
    return raw, info, fits


def aligned(raw, delta, omega, fitted, *, varying=False, reflect=False):
    coordinates = TIME[None] + delta[:, None]
    inside = (coordinates >= 0) & (coordinates < SIZE)
    if reflect:
        valid = np.where(coordinates < 0, -coordinates, np.minimum(coordinates, 2*(SIZE-1)-coordinates))
        return np.take_along_axis(raw, valid, axis=1)
    y = np.take_along_axis(raw, np.clip(coordinates, 0, SIZE-1), axis=1)
    row, col = np.nonzero(~inside)
    if len(row):
        phase = np.exp(1j * omega[row] * coordinates[row, col])
        taps = fitted[row, delta[row] + len(STARTS)//2]
        if varying:
            support = waveform(col - NOMINAL) * (blend_weights(coordinates[row, col]) * taps).sum(1)
        else:
            basis = waveform(col[:, None] - NOMINAL - np.asarray(TAP_OFFSETS['joint_channel']))
            support = (basis * taps).sum(1)
        y[row, col] = phase * support
    return y


def readouts(iq, thresholds):
    raw, info, fits = observations(iq)
    result = {'direct': iq}
    for name, (shift, margin, response) in fits.items():
        delta = np.where(margin > thresholds[name], shift, 0)
        waveform_iq = aligned(raw, delta, info.coarse_rad_per_sample, response,
                              varying=name.startswith('smooth_'), reflect=name == 'transition')
        result[name] = np.stack((waveform_iq.real, waveform_iq.imag), axis=1)
        if name in ('fixed_three', 'smooth_held'):
            reflected = aligned(raw, delta, info.coarse_rad_per_sample, response, reflect=True)
            result[name + '_reflect'] = np.stack((reflected.real, reflected.imag), axis=1)
    return result, info
