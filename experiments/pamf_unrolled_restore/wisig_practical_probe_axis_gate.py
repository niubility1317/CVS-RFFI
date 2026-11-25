"""Calibrate direct-path and echo mixing from cross-fit received source estimates."""

import numpy as np


def gate_features(noise, impulse, anchor, knots, width):
    difference = (np.log(noise)[:, None] - knots) / width
    centers = np.exp(-.5 * difference**2)
    centers /= centers.sum(axis=1, keepdims=True).clip(1e-12)
    energy = np.sum(abs(impulse)**2, axis=-1) - abs(impulse[:, anchor])**2
    energy = energy.clip(0.)
    evidence = energy / (energy + 3. * noise + .006)
    return np.column_stack((centers, centers * evidence[:, None]))


def solve_gate(features, direct, broad, diffuse, teacher, anchor, ridge):
    direction = broad - direct
    residual = teacher - (direct + diffuse[:, None] * direction)
    rate = 1. / np.sum(abs(teacher)**2, axis=-1).clip(.1)
    axes = (np.array([anchor]), np.flatnonzero(np.arange(direction.shape[-1]) != anchor))
    coefficients = []
    for selected in axes:
        d = direction[:, selected]
        energy = np.sum(abs(d)**2, axis=-1)
        cross = np.sum((d.conj() * residual[:, selected]).real, axis=-1)
        gram = features.T @ (features * (energy * rate)[:, None])
        gram.flat[::len(gram)+1] += ridge
        coefficient = np.linalg.solve(gram, features.T @ (rate * cross))
        coefficients.append(coefficient)
    return np.stack(coefficients, axis=1)


def blend(direct, broad, diffuse, anchor, features, calibration, strength):
    change = (features @ calibration).clip(-.35, .60)
    direct_weight, echo_weight = ((diffuse[:, None] + strength * change).clip(0., 1.)).T
    difference = broad - direct
    result = direct + echo_weight[:, None] * difference
    result[:, anchor] = direct[:, anchor] + direct_weight * difference[:, anchor]
    return result
