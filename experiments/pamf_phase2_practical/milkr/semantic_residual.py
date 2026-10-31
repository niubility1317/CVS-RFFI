"""Support-only semantic kernel ridge with an immutable B -> C function prior.

No file IO, source replay, query statistics, or gradient updates. A view is a
feature dictionary; two views represent one physical, labelled observation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular

NAMES = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')


def unit_features(features):
    if set(features) != set(NAMES):
        raise ValueError('Unexpected feature blocks')
    result = {}
    for name in NAMES:
        a = np.array(features[name], dtype=np.float64, copy=True)
        if a.ndim != 2 or a.shape[1] != (96 if name == 'fft' else 160) or not np.isfinite(a).all():
            raise ValueError('Invalid feature block: ' + name)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError('Zero feature block: ' + name)
        a /= norms
        a.flags.writeable = False
        result[name] = a
    if len({len(a) for a in result.values()}) != 1:
        raise ValueError('Feature row alignment differs')
    return result


def semantic(a, b):
    z = a['z_id'] @ b['z_id'].T
    e = a['fft'] @ b['fft'].T
    aux = sum(a[k] @ b[k].T for k in NAMES[2:]) / 3
    bg = (z + e) / 2
    return (z + (bg + aux + bg * aux) / 3) / 2


def kernel(a, b, tau):
    sim = semantic(a, b)
    distance = 2 - 2 * sim
    if (distance < -1e-12).any() or not np.isfinite(distance).all():
        raise FloatingPointError('Invalid semantic distance')
    distance = np.maximum(distance, 0)
    return .5 * sim + sum(np.exp(-distance / (scale * tau)) for scale in (.25, 1., 4.)) / 6


def gram(a, b, tau):
    return sum(kernel(x, y, tau) for x in a for y in b) / (len(a) * len(b))


def bandwidth(old_view, old_labels):
    d = np.maximum(2 - 2 * semantic(old_view, old_view), 0)
    values = d[np.asarray(old_labels)[:, None] != np.asarray(old_labels)[None, :]]
    tau = float(np.median(values))
    # No target-derived fallback and no query input is accepted.
    if not np.isfinite(tau) or tau <= 0:
        raise FloatingPointError('No positive old-support bandwidth')
    return tau


def probabilities(logits, old_count):
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != old_count or not np.isfinite(a).all():
        raise ValueError('Wrong frozen default logits')
    a = np.exp(a - a.max(1, keepdims=True))
    return a / a.sum(1, keepdims=True) - 1 / old_count


def base(logits, old_count, count, enabled):
    result = np.zeros((len(logits), count))
    if enabled:
        result[:, :old_count] = probabilities(logits, old_count)
    return result


@dataclass(frozen=True)
class State:
    views: tuple
    alpha: np.ndarray
    tau: float
    regularization: float
    k: int
    old_count: int
    count: int
    prior: bool
    ids: tuple
    labels: np.ndarray
    residual: float

    def score(self, features, logits):
        query = unit_features(features)
        if len(query['z_id']) != len(logits):
            raise ValueError('Query logits alignment differs')
        scores = base(logits, self.old_count, self.count, self.prior)
        scores += gram((query,), self.views, self.tau) @ self.alpha
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite decision')
        return scores


def fit(views, logits, labels, ids, *, old_count=6, regularization=.01,
        previous=None, prior=True, new_prototype=True, tau=None):
    """C retains its own old support; previous may only be the matching B state.

    The physical-mean loss has n rows, also with two support views. Full C
    factorization is intentional here; function inheritance is independent of
    the optional Schur speed optimization in the design.
    """
    if len(views) not in (1, 2) or len(views) != len(logits):
        raise ValueError('One or two aligned physical views required')
    v = tuple(unit_features(x) for x in views)
    y = np.array(labels, dtype=np.int64, copy=True)
    ids = tuple(map(str, ids))
    if y.ndim != 1 or not len(y) or len(set(ids)) != len(y) or len(ids) != len(y):
        raise ValueError('Invalid physical support identity')
    count = int(y.max()) + 1
    counts = np.bincount(y, minlength=count) if (y >= 0).all() else np.array([])
    if count < old_count or not len(counts) or (counts != counts[0]).any() or counts[0] <= 0:
        raise ValueError('Support must contain positive equal K per registered class')
    k = int(counts[0])
    if not np.array_equal(y, np.repeat(np.arange(count), k)):
        raise ValueError('Canonical class-major support required')
    if any(len(x['z_id']) != len(y) for x in v) or any(len(x) != len(y) for x in logits):
        raise ValueError('Physical view alignment differs')
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('Positive fixed regularization required')
    old_n = old_count * k
    if tau is None:
        tau = bandwidth({name: a[:old_n] for name, a in v[0].items()}, y[:old_n])
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('Positive fixed bandwidth required')
    coefficient = np.zeros((len(y), count))
    if previous is not None:
        if (previous.count != old_count or previous.old_count != old_count or previous.k != k
                or previous.regularization != regularization or previous.prior != prior
                or previous.ids != ids[:old_n] or len(previous.views) != len(v)):
            raise ValueError('B/C task or method identity differs')
        if not all(np.array_equal(a[name], b[name][:old_n])
                   for a, b in zip(previous.views, v) for name in NAMES):
            raise ValueError('B/C old physical features differ')
        tau = previous.tau
        coefficient[:old_n, :old_count] = previous.alpha
    if count > old_count and new_prototype:
        for col in range(old_count, count):
            coefficient[y == col, col] += 1 / k
            coefficient[:old_n, col] -= 1 / old_n
        coefficient -= coefficient.mean(1, keepdims=True)
    K = gram(v, v, tau)
    default = sum(base(a, old_count, count, prior) for a in logits) / len(logits)
    target = np.eye(count)[y] - 1 / count
    rhs = target - default - K @ coefficient
    matrix = K + regularization * k * np.eye(len(y))
    chol = np.linalg.cholesky(matrix)
    delta = solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)
    residual = float(np.linalg.norm(matrix @ delta - rhs) / max(1., np.linalg.norm(rhs)))
    if residual > 1e-10 or not np.isfinite(delta).all():
        raise FloatingPointError('Normal equation residual exceeds tolerance')
    coefficient += delta
    coefficient.flags.writeable = False
    y.flags.writeable = False
    return State(v, coefficient, tau, float(regularization), k, old_count,
                 count, prior, ids, y, residual)
