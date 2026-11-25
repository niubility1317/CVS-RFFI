"""Estimate a sparse common-Doppler channel with bounded residual precision."""

import numpy as np

from .wisig_adaptive_path_audit import FOLDS, UNCERTAINTY_THRESHOLD, packet_fit
from .wisig_independent_channel_audit import N, reflect


TIME = np.arange(N)
U = np.linspace(-1., 1., N)
LAGS = np.arange(-28, 43)
OFFSETS = reflect(TIME[:, None] - LAGS)
RATES = np.arange(-5., 5.1, .5)
PRECISION_FLOOR = .12
TAIL_OFFSET = 6.1
PATH_COST = 7.


def residual_scale(residual):
    return max(float(np.median(abs(residual)**2)) / np.log(2.), 1e-18)


def residual_precision(residual, scale):
    power = abs(residual)**2 / scale
    odds = np.exp(np.clip(TAIL_OFFSET - .92 * power, -50., 50.))
    return 1. - (1. - PRECISION_FLOOR) / (1. + odds)


def supports(gram, matched, path_count=7, width=3):
    diagonal = np.diag(gram).real
    beam = [((), 0.)]
    for _ in range(path_count):
        next_beam = {}
        for chosen, total_gain in beam:
            if chosen:
                idx = list(chosen)
                columns = gram[:, idx]
                cross = gram[np.ix_(idx, idx)]
                projection = columns @ np.linalg.solve(cross, matched[idx])
                conditional = diagonal - np.sum(columns * np.linalg.solve(cross, gram[idx]).T, axis=1).real
                gain = abs(matched - projection)**2 / np.maximum(conditional, 1e-12)
                gain[idx] = -np.inf
            else:
                gain = abs(matched)**2 / diagonal
            for index in np.argsort(gain)[-width:]:
                support = tuple(sorted((*chosen, int(index))))
                updated = total_gain + float(gain[index])
                next_beam[support] = max(next_beam.get(support, -np.inf), updated)
        beam = sorted(next_beam.items(), key=lambda item: -item[1])[:width]
        yield beam[0][0]


def coarse_rates(matrix, observed, weight, times):
    rotation = np.exp(-1j * times[:, None] * RATES)
    correlations = matrix.conj().T @ (observed[:, None] * weight[:, None] * rotation)
    power = (weight @ abs(matrix)**2).clip(1e-12)
    score = np.max(abs(correlations)**2 / power[:, None], axis=0)
    selected = []
    for index in np.argsort(score)[::-1]:
        rate = RATES[index]
        if all(abs(rate - old) >= 1. for old in selected):
            selected.append(rate)
        if len(selected) == 3:
            break
    return selected


def refine(matrix, observed, times, scale, rate, weight):
    for iteration in range(6):
        design = np.exp(1j * rate * times)[:, None] * matrix
        root = np.sqrt(weight)
        weighted = design * root[:, None]
        coefficient = np.linalg.lstsq(weighted, observed * root, rcond=.006)[0]
        prediction = design @ coefficient
        residual = observed - prediction
        if iteration == 5:
            break
        jacobian = 1j * times * prediction * root
        jacobian -= weighted @ np.linalg.lstsq(weighted, jacobian, rcond=.006)[0]
        update = np.vdot(jacobian, residual * root).real / max(np.vdot(jacobian, jacobian).real, 1e-16)
        rate = float(np.clip(rate + np.clip(update, -.55, .55), -8., 8.))
        weight = residual_precision(residual, scale)
    energy = abs(residual)**2 / scale
    loss = -np.logaddexp(-energy, -TAIL_OFFSET - energy * PRECISION_FLOOR).sum()
    return float(loss), rate, coefficient


def estimate_sparse(clean, observed, baseline, indices):
    dictionary = clean[OFFSETS]
    matrix, target = dictionary[indices], observed[indices]
    phase_time = U[indices]
    residual = target - baseline[indices]
    scale = residual_scale(residual)
    weight = residual_precision(residual, scale)
    gram = matrix.conj().T @ (matrix * weight[:, None])
    gram += np.diag(np.maximum(.0001 * np.diag(gram).real, 1e-12))
    losses, waves = [], []
    for rate in coarse_rates(matrix, target, weight, phase_time):
        matched = matrix.conj().T @ (target * weight * np.exp(-1j * rate * phase_time))
        for support in supports(gram, matched):
            columns = list(support)
            cost, phase, gain = refine(matrix[:, columns], target, phase_time, scale, rate, weight)
            losses.append(cost + PATH_COST * len(columns))
            waves.append((dictionary[:, columns] @ gain) * np.exp(1j * phase * U))
    losses = np.asarray(losses)
    confidence = np.exp(losses.min() - losses)
    return (confidence @ np.stack(waves)) / confidence.sum()


def fold_terms(clean, received, config, training, held):
    base, motion, _ = packet_fit(clean, received, config, training)
    sparse = estimate_sparse(clean, received, base, training) - base
    residue = received - base
    scale = residual_scale(residue[training])
    y, d, s = residue[held], motion[held], sparse[held]
    cap = 4. * max(float(np.median(abs(y))), 1e-12)
    motion_precision = np.minimum(1., cap / np.maximum(abs(y), 1e-12))
    motion_norm = np.sum(motion_precision * abs(d)**2)
    p = residual_precision(y, scale)
    design = np.column_stack((d, s))
    gram = (design.conj().T @ (p[:, None] * design)).real
    matched = (design.conj().T @ (p * y)).real
    return (np.vdot(motion_precision * d, y).real, motion_norm, .5 * scale * motion_norm,
            gram, matched, .5 * scale * gram)


def weights(terms, threshold):
    numerator, norm, uncertainty, gram, matched, covariance = map(sum, zip(*terms))
    motion = float(np.clip((numerator - UNCERTAINTY_THRESHOLD * np.sqrt(uncertainty)) /
                           max(norm, 1e-15), 0., 1.))
    # Estimate the extra improvement of the sparse action over the selected
    # motion action, so two branches cannot claim the same evidence.
    old = np.array([motion, 0.])
    direction = np.array([-motion, 1.])
    gain = direction @ (matched - gram @ old)
    size = direction @ gram @ direction
    uncertainty = max(float(direction @ covariance @ direction), 0.)
    sparse = float(np.clip((gain - threshold * np.sqrt(uncertainty)) / max(size, 1e-15), 0., 1.))
    return motion, sparse


def estimate(clean, received, config, threshold):
    terms = []
    for fold in range(4):
        held = FOLDS == fold
        terms.append(fold_terms(clean, received, config, TIME[~held], held))
    motion_weight, sparse_weight = weights(terms, threshold)
    start, motion, variance = packet_fit(clean, received, config, TIME)
    old = start + motion_weight * motion
    sparse = estimate_sparse(clean, received, start, TIME)
    new = old + sparse_weight * (sparse - old)
    return old, new, variance, sparse_weight
