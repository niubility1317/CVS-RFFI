"""Estimate separate Doppler and gain variations for competing multipath fits."""

import numpy as np

from . import wisig_refined_channel as paired


TIME = paired.adaptive.T
U = paired.adaptive.U[:, None]
RATES = np.arange(-5., 5.1, 1.25)
LAGS = np.arange(-42., 51., .5)
ATOM_DELAY = np.tile(LAGS, 2)
ATOM_IMAGE = np.repeat((False, True), len(LAGS))


def refine(interpolate, params, image, target, indices, weights, tones, fractional, rounds):
    def fit(current):
        delay, rate, slope, clock = current
        wave, derivative = paired.paths(interpolate, delay + clock * U, image)
        rotation = np.exp(1j * rate * U)
        modulation = rotation * (1. + slope * U)
        paths = wave * modulation
        system = np.column_stack((paths, tones))[indices] * weights[:, None]
        coefficient, _, _ = paired.adaptive.least_squares(system, target, ridge=.00025)
        residual = target - system @ coefficient
        gain = coefficient[:len(delay)]
        return (float(np.vdot(residual, residual).real), paths @ gain, system,
                residual, paths, wave, derivative, rotation, modulation, gain)

    best = fit(params)
    for _ in range(rounds):
        _, _, system, residual, paths, wave, derivative, rotation, modulation, gain = best
        jacobian = [1j * paths * U * gain, wave * rotation * U * gain]
        if fractional:
            timing = derivative * modulation * gain
            jacobian.extend((timing, timing.sum(axis=1, keepdims=True) * U))
        jacobian = np.column_stack(jacobian)[indices] * weights[:, None]
        jacobian -= system @ np.linalg.lstsq(system, jacobian, rcond=.004)[0]
        step = np.linalg.lstsq(np.r_[jacobian.real, jacobian.imag],
                              np.r_[residual.real, residual.imag], rcond=.045)[0]
        size = len(image)
        for damping in (1., .4):
            delay, rate, slope, clock = params
            if fractional:
                delay = delay + damping * np.clip(step[2 * size:3 * size], -.35, .35)
                clock = np.clip(clock + damping * np.clip(step[-1], -.3, .3), -1.5, 1.5)
            rate = np.clip(rate + damping * np.clip(step[:size], -.75, .75), -8., 8.)
            slope = np.clip(slope + damping * np.clip(step[size:2 * size], -.25, .25), -.99, .99)
            trial = delay, rate, slope, clock
            current = fit(trial)
            if current[0] < best[0]:
                params, best = trial, current
                break
        else:
            break
    return params, best[1], best[0]


def estimate_bank(clean, observed, baseline, indices):
    interpolate = paired.interpolator(clean)
    residual = observed[indices] - baseline[indices]
    tones = paired.adaptive.randomized_tones(residual, indices)
    magnitude = np.abs(residual)
    scale = max(float(np.median(magnitude)), 1e-12)
    weights = np.sqrt(np.minimum(1., 4.5 * scale / np.maximum(magnitude, 1e-12)))
    weights[magnitude > 4. * scale] *= .25
    target = observed[indices] * weights

    shifted = interpolate(indices[:, None] - LAGS)
    shifted = np.column_stack((shifted, shifted.conj()))
    coarse = shifted[:, :, None] * np.exp(1j * U[indices, :, None] * RATES)
    coarse = (coarse * weights[:, None, None]).reshape(len(indices), -1)
    if tones.shape[1]:
        q, _ = np.linalg.qr(weights[:, None] * tones[indices])
        coarse -= q @ (q.conj().T @ coarse)
    adjoint = coarse.conj().T / np.linalg.norm(coarse, axis=0).clip(1e-12)[:, None]
    params = np.empty(0), np.empty(0), np.empty(0), 0.
    images = np.empty(0, bool)
    action = np.zeros(len(clean), complex)
    candidates = []
    for size in range(1, 7):
        matches = np.abs(adjoint @ ((observed - action)[indices] * weights))
        matches = matches.reshape(-1, len(RATES))
        for delay, image in zip(params[0], images):
            unavailable = (np.abs(ATOM_DELAY - delay) < 1.2) & (ATOM_IMAGE == image)
            matches[unavailable] = -np.inf
        candidate, frequency = np.unravel_index(np.argmax(matches), matches.shape)
        params = (np.r_[params[0], ATOM_DELAY[candidate]], np.r_[params[1], RATES[frequency]],
                  np.r_[params[2], 0.], params[3])
        images = np.r_[images, ATOM_IMAGE[candidate]]
        _, fixed, error = refine(interpolate, (np.round(params[0]), *params[1:]), images,
                                 target, indices, weights, tones, fractional=False, rounds=4)
        candidates.append(paired.Fit(fixed, error, 2.8 * size + tones.shape[1]))
        params, action, error = refine(interpolate, params, images, target, indices, weights, tones,
                                       fractional=True, rounds=5)
        candidates.append(paired.Fit(action, error, 3.3 * size + tones.shape[1] + .5))
    return candidates


def packet(clean, received, prior, indices=TIME):
    baseline, variance, bank = paired.packet(clean, received, prior["adaptive"], indices)
    option = prior["penalty"]
    proposal = paired.solutions(bank, len(indices), (option,))[option]
    refined = baseline + prior["mix"] * (proposal - baseline)
    paths = estimate_bank(clean, received, baseline, indices)
    return baseline, refined, variance, paths
