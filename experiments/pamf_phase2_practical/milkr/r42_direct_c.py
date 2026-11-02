"""Experimental R42-derived direct registration with no target-old support.

The source-selected kernel rule and full frozen residual encoder are reused;
the balanced old/new target-support calibration cannot be used without old
support and is replaced by a fixed frozen-head prior. This is a distinct
variant, not an exact execution of the original R42.
"""
from dataclasses import dataclass
import numpy as np
from . import orbit_ensemble as orbit, signature_kernel as signature, support_calibrated as base
from .prior_fusion import PRIOR_WEIGHT


def source_prototypes(features, labels):
    """Aggregate fixed labelled source-only features, never target examples."""
    x = orbit.normalize(features)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(x['z_id']),) or set(y.tolist()) != set(range(6)):
        raise ValueError('Exactly six source classes are required')
    result = {key: np.stack([value[y == c].mean(axis=0) for c in range(6)])
              for key, value in x.items()}
    for value in result.values():
        value.flags.writeable = False
    return result


def _prior(x, classes, prototypes, geometry, tau):
    p = np.asarray(x['teacher_probs'], dtype=np.float64)
    if prototypes is not None:
        sims = orbit.similarities(x, prototypes, geometry)
        affinity = sum(np.exp(-np.maximum(2. - 2. * sim, 0.) / scale) for sim, scale in zip(sims, tau)) / 2.
        p = p * affinity
        total = p.sum(axis=1, keepdims=True)
        if not np.isfinite(total).all() or (total <= 0).any():
            raise FloatingPointError('Source-prototype prior has no probability mass')
        p = p / total
    result = np.zeros((len(p), classes), dtype=np.float64)
    result[:, :6] = PRIOR_WEIGHT * (p - 1. / 6.)
    return result


@dataclass(frozen=True)
class DirectState:
    support: dict
    prototypes: object
    config: tuple
    tau: tuple
    alpha: np.ndarray
    intercept: np.ndarray
    audit: dict

    def score(self, features):
        x = orbit.normalize(features)
        geometry, _, mix, scales = self.config
        sims = orbit.similarities(x, self.support, geometry)
        K = orbit.make_kernel(sims, self.tau, mix, scales)
        value = _prior(x, self.alpha.shape[1], self.prototypes, geometry, self.tau)
        value += K @ self.alpha + self.intercept
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite direct-C scores')
        return value


class DirectPair:
    """Two independently fitted states sharing the same support kernel."""
    def __init__(self, head, proto):
        if head.prototypes is not None or proto.prototypes is None:
            raise ValueError('Expected frozen-head and source-prototype states')
        if head.config != proto.config or head.tau != proto.tau or set(head.support) != set(proto.support):
            raise ValueError('Different kernel configurations')
        if not all(np.array_equal(head.support[key], proto.support[key]) for key in head.support):
            raise ValueError('Different support feature banks')
        self.head, self.proto = head, proto

    def score(self, features):
        x = orbit.normalize(features)
        geometry, _, mix, scales = self.head.config
        sims = orbit.similarities(x, self.head.support, geometry)
        kernel = orbit.make_kernel(sims, self.head.tau, mix, scales)
        result = []
        for state in (self.head, self.proto):
            value = _prior(x, state.alpha.shape[1], state.prototypes, geometry, state.tau)
            value += kernel @ state.alpha + state.intercept
            if not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite direct-C scores')
            result.append(value)
        return tuple(result)


def fit_direct_c(features, labels, ids, *, rule, source_prototypes=None):
    x = orbit.normalize(features)
    y = np.asarray(labels, dtype=np.int64)
    ids = tuple(map(str, ids))
    if y.ndim != 1 or not len(y) or len(y) != len(x['z_id']) or (y < 6).any():
        raise ValueError('Only new class support labels are permitted')
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError('Invalid physical support IDs')
    n = int(y.max()) - 5
    counts = np.bincount(y - 6, minlength=n)
    if n not in (2, 5, 10, 20) or (counts <= 0).any() or (counts != counts[0]).any():
        raise ValueError('New class support must be balanced and contiguous')
    config = tuple(rule['config'])
    ridge = float(rule['ridge'])
    if config not in signature.CONFIGS or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Invalid source-frozen kernel rule')
    geometry, band, mix, scales = config
    sims = orbit.similarities(x, x, geometry)
    taus = []
    for sim in sims:
        distance = np.maximum(2. - 2. * sim, 0.)
        if band == 'nearest':
            tau = float(np.median(np.min(np.where(y[:, None] != y[None, :], distance, np.inf), axis=1)))
        else:
            tau = float(np.median(distance[y[:, None] != y[None, :]]))
        if not np.isfinite(tau) or tau <= 0:
            raise FloatingPointError('No positive new-support bandwidth')
        taus.append(tau)
    tau = tuple(taus)
    prototypes = None if source_prototypes is None else orbit.normalize(source_prototypes)
    if prototypes is not None and len(prototypes['z_id']) != 6:
        raise ValueError('Six frozen source prototypes required')
    K = orbit.make_kernel(sims, tau, mix, scales)
    eig, U, _, _, _, mean = base.decompose(K, y - 6)
    target = np.eye(6 + n)[y] - _prior(x, 6 + n, prototypes, geometry, tau)
    average = target.mean(axis=0)
    parts = (eig, U, target, average, target - average, mean)
    if counts[0] == 1:
        ridge *= (6 + n) / 6.
    alpha, intercept, _ = base.coefficients(parts, ridge)
    # New support can shift the old group jointly, but must not change the
    # old-class ranking supplied by the frozen head or source prototypes.
    alpha[:, :6] = alpha[:, :6].mean(axis=1, keepdims=True)
    intercept[:6] = intercept[:6].mean()
    alpha.flags.writeable = False
    intercept.flags.writeable = False
    audit = dict(actual_target_old_labels=0, actual_target_new_labels=len(y), query_fit=False,
                 old_source='frozen_head' if prototypes is None else 'source_prototypes',
                 kernel_rule='inherited_source_fixed', adaptive_strength=1.,
                 old_new_support_gate='not_applicable_no_target_old', ridge=ridge,
                 old_class_ranking='frozen_prior_unchanged_by_new_support')
    return DirectState(x, prototypes, config, tau, alpha, intercept, audit)
