"""Fixed support-only convex head; frozen Phase1 teacher, no source/query fit."""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _readonly(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    # An immutable bytes owner prevents re-enabling write access on the state.
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _mutable_copy(value):
    if isinstance(value, Mapping):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_copy(item) for item in value]
    return value


def _features(features):
    raw = np.asarray(features)
    if (raw.ndim != 2 or raw.shape[1] not in (160, 256, 288)
            or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all()):
        raise ValueError('Expected finite numeric features [N,160/256/288]')
    x = np.asarray(raw[:, :160], dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError('Identity features must have finite nonzero norms')
    return x / norm


def _objective(parameters, x, y, teacher_logprob, old_count, k, w0):
    """Return the registered loss, its analytic gradient, and measured terms."""
    c, d = w0.shape
    values = np.asarray(parameters, dtype=np.float64).reshape(c, d + 1)
    w, b = values[:, :d], values[:, d]
    logits = x @ w.T + b
    logprob = logits - logsumexp(logits, axis=1, keepdims=True)
    old = y < old_count
    weights = np.empty(len(y), dtype=np.float64)
    if c == old_count:
        weights.fill(1.0 / len(y))
    else:
        weights[old] = .5 / (old_count * k)
        weights[~old] = .5 / ((c - old_count) * k)
    support_ce = float(-np.dot(weights, logprob[np.arange(len(y)), y]))
    delta = np.exp(logprob)
    delta[np.arange(len(y)), y] -= 1
    delta *= weights[:, None]

    conditional = logits[old, :old_count]
    conditional = conditional - logsumexp(conditional, axis=1, keepdims=True)
    teacher = np.exp(teacher_logprob[old])
    kd = float(np.mean(np.sum(teacher * (teacher_logprob[old] - conditional), axis=1)))
    kd_weight = 1.0 / (k + 1)
    delta[old, :old_count] += kd_weight * (np.exp(conditional) - teacher) / (old_count * k)

    difference = w - w0
    penalty_weight = 1.0 / (2 * k * c)
    scaled_squared_distance = float(np.sum((difference / 10) ** 2) + np.sum((b / 10) ** 2))
    penalty = penalty_weight * scaled_squared_distance
    grad_w = delta.T @ x + difference / (100 * k * c)
    grad_b = delta.sum(axis=0) + b / (100 * k * c)
    gradient = np.column_stack((grad_w, grad_b)).ravel()
    total = support_ce + kd_weight * kd + penalty
    if not np.isfinite(total) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite source-free objective or gradient')
    metrics = dict(loss=float(total), support_ce=support_ce, conditional_old_kd=kd,
                   scaled_squared_distance=scaled_squared_distance, regularization=penalty,
                   support_weight=1.0, kd_weight=kd_weight,
                   squared_distance_weight=penalty_weight,
                   gradient_l2=float(np.linalg.norm(gradient)),
                   gradient_inf=float(np.max(np.abs(gradient))))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Nonfinite source-free optimization metrics')
    return float(total), gradient, metrics


@dataclass(frozen=True)
class SourceFreeHeadState:
    coefficient: np.ndarray
    intercept: np.ndarray
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'coefficient', _readonly(self.coefficient))
        object.__setattr__(self, 'intercept', _readonly(self.intercept))
        object.__setattr__(self, 'classes', tuple(self.classes))
        object.__setattr__(self, 'audit', _immutable(self.audit))

    @property
    def W(self):
        return self.coefficient

    @property
    def b(self):
        return self.intercept

    def audit_dict(self):
        """Independent JSON-safe copy; modifying it never mutates fitted state."""
        return _mutable_copy(self.audit)

    def score(self, features):
        x = _features(features)
        # Explicit per-row reduction makes scores independent of batch shape.
        scores = np.stack([np.sum(self.coefficient * row[None, :], axis=1)
                           + self.intercept for row in x]) if len(x) else np.empty((0, len(self.classes)))
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite source-free scores')
        return scores


def fit_sourcefree_head(support_features, support_labels, support_logits,
                        classes, old_count=6, max_iter=300):
    started = time.perf_counter()
    x = _features(support_features)
    y = np.asarray(support_labels)
    raw_logits = np.asarray(support_logits)
    classes = tuple(classes)
    c = len(classes)
    if (c < 2 or any(not isinstance(value, str) or not value for value in classes)
            or len(set(classes)) != c
            or type(old_count) is not int or not 1 <= old_count <= c
            or type(max_iter) is not int or max_iter < 1
            or y.shape != (len(x),) or y.dtype.kind not in 'iu'
            or set(y.tolist()) != set(range(c))
            or raw_logits.shape != (len(x), old_count)
            or raw_logits.dtype.kind not in 'fiu' or not np.isfinite(raw_logits).all()):
        raise ValueError('Support labels, registry, logits, old_count or iteration contract mismatch')
    counts = np.bincount(y.astype(np.int64), minlength=c)
    if counts.min() < 1 or np.any(counts != counts[0]):
        raise ValueError('Equal positive physical K is required for every registered class')
    k = int(counts[0])
    logits = np.asarray(raw_logits, dtype=np.float64)
    # Canonical support-only ordering; no input-order-dependent optimization.
    order = np.asarray(sorted(range(len(x)), key=lambda i: (int(y[i]), x[i].tobytes(), logits[i].tobytes())))
    x, y, logits = x[order], y[order].astype(np.int64), logits[order]
    teacher_logprob = logits - logsumexp(logits, axis=1, keepdims=True)
    means = np.stack([x[y == cls].mean(axis=0) for cls in range(c)])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError('Support class mean is degenerate')
    w0 = 10 * means / norms
    initial = np.column_stack((w0, np.zeros(c))).ravel()
    steps = []

    def objective(parameters):
        value, gradient, _ = _objective(parameters, x, y, teacher_logprob, old_count, k, w0)
        return value, gradient

    def record(parameters):
        _, _, metrics = _objective(parameters, x, y, teacher_logprob, old_count, k, w0)
        steps.append(dict(iteration=len(steps) + 1, elapsed_seconds=time.perf_counter() - started, **metrics))

    _, _, initial_metrics = _objective(initial, x, y, teacher_logprob, old_count, k, w0)
    result = minimize(objective, initial, method='L-BFGS-B', jac=True, callback=record,
                      options=dict(maxiter=max_iter, gtol=1e-6, ftol=1e-12))
    if not np.isfinite(result.x).all():
        raise FloatingPointError('Nonfinite source-free optimizer state')
    _, _, final_metrics = _objective(result.x, x, y, teacher_logprob, old_count, k, w0)
    fitted = result.x.reshape(c, 161)
    audit = dict(method='D92-SFHead-v1', k=k, old_count=old_count,
                 class_count=c, feature_dim=160, support_rows=len(x),
                 selection='fixed_preregistered_formula', query_rows_used_for_fit=0,
                 encoder_updated=False, source_runtime_access=False, ground_used_for_fit=False,
                 optimizer='scipy.optimize.L-BFGS-B', precision='float64',
                 learning_rate=None, learning_rate_reason='L-BFGS-B internal line search; scalar learning rate not measured',
                 max_iter=max_iter, gtol=1e-6, ftol=1e-12, prototype_scale=10.0,
                 converged=bool(result.success), optimizer_status=int(result.status),
                 termination_message=str(result.message),
                 status='CONVERGED' if result.success else 'NOT_CONVERGED',
                 optimizer_steps=int(result.nit), function_evaluations=int(result.nfev),
                 initial=initial_metrics, final=final_metrics, steps=steps,
                 fit_seconds=time.perf_counter() - started,
                 persistent_state_bytes=int(fitted.nbytes), new_ground_payload_bytes=0,
                 support_task_weights=dict(old=1.0 if c == old_count else .5,
                                           new=0.0 if c == old_count else .5))
    return SourceFreeHeadState(fitted[:, :160], fitted[:, 160], classes, audit)
