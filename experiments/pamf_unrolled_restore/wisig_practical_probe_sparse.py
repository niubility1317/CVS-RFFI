"""Receive-only soft sparse relative channel targets for source training."""

import numpy as np
from scipy.special import softmax

from . import wisig_practical_probe_relative as relative


DELAYS = np.r_[0., np.arange(.75, 34.01, .25)]
DISTANCE = relative.OFFSETS[:, None] - DELAYS
ATOMS = np.sinc(DISTANCE) * np.sinc(DISTANCE / 6.) * (abs(DISTANCE) < 6.)
ECHO_DISTANCE = abs(DELAYS[1:, None] - DELAYS[None, 1:])


def fit_relative(reference, received, noise):
    """Marginalize candidate fractional delays rather than selecting an unstable peak."""
    design = reference @ ATOMS
    direct = design[:, 0]
    energy = max(float(np.vdot(direct, direct).real), 1e-12)
    normalized = direct.conj() / energy
    coupling = normalized @ design[:, 1:]
    candidates = design[:, 1:] - direct[:, None] * coupling
    precision = (np.sum(abs(candidates)**2, axis=0) + energy * .015).clip(1e-12)
    steering = ATOMS[:, 1:] - ATOMS[:, :1] * coupling
    coeff = normalized @ received
    remaining = received - direct * coeff
    first = ATOMS[:, 0] * coeff
    overlap = np.zeros(len(DELAYS) - 1)
    for _ in range(2):
        match = candidates.conj().T @ remaining
        reduction = abs(match)**2 / precision
        evidence = reduction / max(1.65 * noise, 1e-8) - 6.5 - 4. * overlap
        weight = softmax(np.r_[0., evidence])[1:]
        gains = weight * match / precision
        first = first + steering @ gains
        remaining = remaining - candidates @ gains
        overlap = (ECHO_DISTANCE < 1.2) @ weight
    return first


def target_columns(received, prior):
    packets, views, samples = received.shape
    noise = relative.blind.receive.guard_power(received.reshape(-1, samples), prior.guard).reshape(packets, views)
    start = prior.anchor + relative.OFFSETS[0]
    stop = prior.anchor + relative.OFFSETS[-1] + 1
    if start < 0 or stop > relative.blind.KERNEL_LENGTH:
        raise ValueError("Source timing anchor does not fit causal probe support")
    result = np.zeros((packets, views, relative.blind.KERNEL_LENGTH), complex)
    for i, observations in enumerate(received):
        for j, signal in enumerate(observations):
            other = np.delete(np.arange(relative.SCENE_COUNT), j % relative.SCENE_COUNT)
            selected = other[np.argsort(noise[i, other])[:3]]
            anchors = observations[selected]
            orientation = anchors @ anchors[0].conj()
            aligned = anchors * relative.blind.unit(orientation.conj())[:, None]
            reference = aligned.mean(axis=0)[relative.SOURCE_INDICES]
            result[i, j, start:stop] = fit_relative(reference, signal[relative.FIT_TIME], noise[i, j])
    return result.reshape(-1, relative.blind.KERNEL_LENGTH)
