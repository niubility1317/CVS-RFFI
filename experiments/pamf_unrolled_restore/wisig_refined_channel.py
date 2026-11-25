"""Source-paired channel estimation with fractional paths and smooth Doppler."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample

from . import wisig_ten_channel_estimator as adaptive


@dataclass
class Fit:
    action: np.ndarray
    error: float
    complexity: float


def interpolator(clean):
    # Reflection preserves the existing packet-edge convention. Oversampling
    # gives a differentiable fractional-delay model with exact integer samples.
    period = np.r_[clean, clean[-2:0:-1]]
    step = 4
    samples = resample(period, step * len(period))
    points = np.arange(-64 * step, (len(clean) + 65) * step) / step
    return CubicSpline(points, samples[(points * step).astype(int) % len(samples)])


def dictionary(interpolate, observation, whiten):
    delay = np.arange(-50, 51, .5)
    direct = interpolate(adaptive.T[:, None] - delay)
    atoms = np.column_stack((direct, direct.conj()))
    rates = np.arange(-8., 8.1, 2.)
    matrix = whiten(atoms[:, :, None] * np.exp(1j * adaptive.U[:, None, None] * rates))
    norms = np.sqrt(np.sum(np.abs(matrix)**2, axis=0)).clip(1e-12)
    match = np.abs(matrix.reshape(len(observation), -1).conj().T @ observation)
    index, frequency = np.unravel_index(np.argmax(match.reshape(norms.shape) / norms), norms.shape)
    return delay, atoms, delay[index % len(delay)], index >= len(delay), rates[frequency]


def paths(interpolate, delays, mirror):
    coordinate = adaptive.T[:, None] - delays
    signal = interpolate(coordinate)
    derivative = -interpolate(coordinate, 1)
    signal[:, mirror] = signal[:, mirror].conj()
    derivative[:, mirror] = derivative[:, mirror].conj()
    return signal, derivative


def optimize(interpolate, delays, mirror, phase, y, whiten, nuisance,
             weights, dynamic=False, clock=False, rounds=3):
    delays, phase = delays.copy(), phase.copy()
    times = adaptive.U[:, None]
    phase_basis = adaptive.PHASE
    drift = 0.

    def weighted(values):
        return whiten(values) * weights.reshape((-1,) + (1,) * (values.ndim - 1))

    for iteration in range(rounds + 1):
        coordinates = delays[None, :] + drift * times
        wave, derivative = paths(interpolate, coordinates, mirror)
        rotation = np.exp(1j * (phase_basis @ phase))[:, None]
        wave, derivative = wave * rotation, derivative * rotation
        design = np.column_stack((wave, wave * times)) if dynamic else wave
        matrix = weighted(np.column_stack((design, nuisance)))
        coef, _, _ = adaptive.least_squares(matrix, y, ridge=.0002)
        action = design @ coef[:design.shape[1]]
        residual = y - matrix @ coef
        if iteration == rounds:
            break
        local_gain = coef[:len(delays)]
        if dynamic:
            local_gain = local_gain + times * coef[len(delays):2 * len(delays)]
        timing = derivative * local_gain
        jacobian = [timing, 1j * action[:, None] * phase_basis]
        if clock:
            jacobian.append(timing.sum(axis=1, keepdims=True) * times)
        jacobian = weighted(np.column_stack(jacobian))
        jacobian -= matrix @ np.linalg.lstsq(matrix, jacobian, rcond=.002)[0]
        step = np.linalg.lstsq(
            np.r_[jacobian.real, jacobian.imag],
            np.r_[residual.real, residual.imag],
            rcond=.025,
        )[0]
        delays += np.clip(step[:len(delays)], -.4, .4)
        phase += np.clip(step[len(delays):len(delays) + 2], -.75, .75)
        if clock:
            drift = float(np.clip(drift + np.clip(step[-1], -.4, .4), -1.5, 1.5))
    cost = design.shape[1] + nuisance.shape[1] + 1.5 * len(delays) + 1 + .5 * clock
    return Fit(action, float(np.vdot(residual, residual).real), cost), delays, phase


def refined_bank(clean, observed, train, starting, rho):
    whiten = adaptive.whitener(train, rho)
    preliminary = observed[train] - starting[train]
    nuisance = adaptive.randomized_tones(preliminary, train)
    raw = whiten(observed)
    # Cap only clear impulsive outliers, for which white least squares spreads
    # one large sample through all estimated paths.
    residual = np.abs(whiten(observed - starting))
    limit = 4.5 * max(float(np.median(residual)), 1e-12)
    weights = np.sqrt(np.minimum(1., limit / np.maximum(residual, 1e-12)))
    y = raw * weights
    interpolate = interpolator(clean)
    delay_grid, atoms, first, image, rate = dictionary(interpolate, raw, whiten)
    delays = np.array([first])
    mirror = np.array([image])
    phase = np.array([rate, 0.])
    atoms_delay = np.tile(delay_grid, 2)
    atoms_mirror = np.repeat((False, True), len(delay_grid))
    result = []
    for size in range(1, 7):
        fit, delays, phase = optimize(interpolate, delays, mirror, phase, y, whiten,
                                      nuisance, weights, rounds=3)
        result.append(fit)
        if 2 <= size <= 4:
            for dynamic, clock in ((True, False), (False, True)):
                alternative, _, _ = optimize(interpolate, delays, mirror, phase, y,
                                            whiten, nuisance, weights, dynamic, clock)
                result.append(alternative)
        if size == 6:
            break
        residue = y - whiten(fit.action) * weights
        bank = whiten(atoms * np.exp(1j * (adaptive.PHASE @ phase))[:, None])
        scaled = bank * weights[:, None]
        if nuisance.shape[1]:
            q, _ = np.linalg.qr(whiten(nuisance) * weights[:, None])
            scaled -= q @ (q.conj().T @ scaled)
        gain = np.abs(scaled.conj().T @ residue) / np.linalg.norm(scaled, axis=0).clip(1e-12)
        for position, branch in zip(delays, mirror):
            gain[(np.abs(atoms_delay - position) < 1.35) & (atoms_mirror == branch)] = -np.inf
        index = int(np.argmax(gain))
        delays = np.r_[delays, atoms_delay[index]]
        mirror = np.r_[mirror, atoms_mirror[index]]
    return result


def packet(clean, observed, options, train=adaptive.T):
    old, rho, tones = adaptive.candidate_bank(clean, observed, train)
    signal, variance, _ = adaptive.estimate(
        old, rho, tones, clean, observed, train,
        options["complexity"], options["shrink"], options["blend"],
    )
    candidates = refined_bank(clean, observed, train, signal, rho)
    return signal, variance, candidates


def solutions(candidates, sample_count, penalties):
    error = np.array([candidate.error for candidate in candidates])
    cost = np.array([candidate.complexity for candidate in candidates])
    waves = np.stack([candidate.action for candidate in candidates])
    result = {}
    for penalty in penalties:
        score = sample_count * np.log(error.clip(1e-24)) + penalty * cost
        weights = np.exp(-.5 * (score - score.min()))
        action = weights @ waves / weights.sum()
        result[penalty] = action
    return result
