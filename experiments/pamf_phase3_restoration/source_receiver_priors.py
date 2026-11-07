"""Matched-size IQ priors fitted only to original L_s/U_s receiver groups."""

import numpy as np

from .covariance_inverse import SHRINKAGE


def select_receivers(rx, day, roles):
    groups = tuple(sorted(set(rx.tolist())))
    if groups != (1, 3, 4, 6, 8):
        raise ValueError('Expected the original five source receivers')
    rng = np.random.default_rng(2026110707)
    selections = {}
    for held in groups:
        same = np.flatnonzero(rx == held)
        other = []
        for receiver in groups:
            if receiver == held:
                continue
            for date in (1, 2, 3):
                for role, count in (('L_s', 105), ('U_s', 945)):
                    candidates = np.flatnonzero((rx == receiver) & (day == date) & (roles == role))
                    if len(candidates) != 4*count:
                        raise ValueError('Original balanced RX/day/source roles required')
                    other.extend(rng.choice(candidates, count, replace=False).tolist())
        other = np.sort(other)
        if (len(same) != 12600 or len(other) != 12600 or len(set(other)) != 12600
                or np.intersect1d(same, other).size or (roles[same] == 'L_s').sum() != 1260):
            raise ValueError('Equal-size, label-blind receiver priors required')
        selections[held] = (same, other)
    return selections


def fit_prior(clean, indices):
    mean = np.asarray(clean[indices], dtype=np.float64).reshape(len(indices), 512).mean(0)
    covariance = np.zeros((512, 512), dtype=np.float64)
    for offset in range(0, len(indices), 1024):
        x = np.asarray(clean[indices[offset:offset+1024]], dtype=np.float64).reshape(-1, 512) - mean
        covariance += x.T @ x
    covariance /= len(indices)
    trace = np.trace(covariance)/512
    covariance *= 1-SHRINKAGE
    covariance.flat[::513] += SHRINKAGE*trace
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or trace <= 0:
        raise ValueError('Expected finite positive source IQ covariance')
    # Match the FP32 learned-model buffers of the original full-source prior.
    return mean.astype(np.float32), covariance.astype(np.float32)
