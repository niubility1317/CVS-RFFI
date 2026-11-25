"""Learn single-receive FIR estimates from relative source-U observations."""

import numpy as np

from . import wisig_practical_probe_blind as blind


OFFSETS = np.arange(-3, 40)
FIT_TIME = np.arange(88, 253)
SOURCE_INDICES = FIT_TIME[:, None] - OFFSETS
SCENE_COUNT = 6


def target_columns(received, prior):
    """Construct relative FIR teaching signals from other Practical received views."""
    packets, views, samples = received.shape
    noise = blind.receive.guard_power(received.reshape(-1, samples), prior.guard).reshape(packets, views)
    result = np.zeros((packets, views, blind.KERNEL_LENGTH), complex)
    first = prior.anchor + OFFSETS[0]
    last = prior.anchor + OFFSETS[-1] + 1
    if first < 0 or last > blind.KERNEL_LENGTH:
        raise ValueError("Source timing anchor does not fit causal probe support")
    for i, observation in enumerate(received):
        for j, signal in enumerate(observation):
            # Exclude the same scene's views, including the other impairment profile.
            other = np.delete(np.arange(SCENE_COUNT), j % SCENE_COUNT)
            chosen = other[np.argsort(noise[i, other])[:3]]
            above = observation[chosen]
            orientation = above @ above[0].conj()
            reference = (above * blind.unit(orientation.conj())[:, None]).mean(axis=0)
            design = reference[SOURCE_INDICES]
            gram = design.conj().T @ design
            penalty = (.015 + .15 * noise[i, j]) * (1. + .05 * abs(OFFSETS))
            gram.flat[::len(OFFSETS) + 1] += penalty * np.diag(gram).real + 1e-12
            result[i, j, first:last] = np.linalg.solve(gram, design.conj().T @ signal[FIT_TIME])
    return result.reshape(-1, blind.KERNEL_LENGTH)


def observables(received, prior):
    iq = np.asarray(received, np.complex128)
    q = blind.receive.guard_power(iq, prior.guard)
    coeffs = blind.pilot_sync(iq)[:, blind.PILOT_POINTS] @ prior.decoder.T
    coeffs /= np.sqrt(np.mean(abs(iq)**2, axis=-1)).clip(1e-7)[:, None]
    return np.column_stack((coeffs.real, coeffs.imag, np.log(q))), q


def regressors(features, noise, center, scale):
    standardized = (features - center) / scale
    soft = 1. / (1. + noise / .03)
    return np.column_stack((np.ones(len(noise)), soft, standardized,
                            standardized * soft[:, None], standardized * (1. - soft[:, None])**2))


def estimate(received, prior, center, scale, matrices):
    features, noise = observables(received, prior)
    design = regressors(features, noise, center, scale)
    return design[None, :, :] @ matrices, noise
