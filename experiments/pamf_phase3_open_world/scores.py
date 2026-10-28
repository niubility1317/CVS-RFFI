"""Row-local knownness scores and source-only operating points."""
import numpy as np

METHODS = ('msp', 'maxlogit', 'margin', 'energy', 'cosine')


def _matrix(value, name, columns=None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError('Nonempty finite matrix required: ' + name)
    if columns is not None and array.shape[1] != columns:
        raise ValueError('Unexpected column count: ' + name)
    return array


def fit_prototypes(features, labels, class_count):
    """Only the caller's legal labeled source training role may be supplied."""
    z = _matrix(features, 'source L features')
    y = np.asarray(labels)
    if class_count < 2 or y.shape != (len(z),) or y.dtype.kind not in 'iu':
        raise ValueError('Visible source labels and class count required')
    if np.any(y < 0) or np.any(y >= class_count):
        raise ValueError('Hidden or out-of-range source labels')
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    centers = []
    for label in range(class_count):
        if not np.any(y == label):
            raise ValueError('Missing labeled source class')
        center = z[y == label].mean(axis=0)
        norm = np.linalg.norm(center)
        if norm < 1e-12:
            raise ValueError('Degenerate source class center')
        centers.append(center / norm)
    return np.asarray(centers)


def knownness_scores(features, logits, prototypes):
    z = _matrix(features, 'features')
    raw = _matrix(logits, 'logits')
    centers = _matrix(prototypes, 'prototypes', z.shape[1])
    if len(z) != len(raw) or raw.shape[1] < 2 or len(centers) != raw.shape[1]:
        raise ValueError('Feature, class and prototype dimensions differ')
    top = raw.max(axis=1)
    shifted = raw - top[:, None]
    exp_sum = np.exp(shifted).sum(axis=1)
    msp = 1. / exp_sum
    pair = np.partition(raw, -2, axis=1)[:, -2:]
    margin = pair.max(axis=1) - pair.min(axis=1)
    # Negative free energy at fixed T=1: higher always means more known.
    energy = top + np.log(exp_sum)
    unit_z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    unit_c = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    cosine = np.clip((unit_z @ unit_c.T).max(axis=1), -1., 1.)
    return np.column_stack((msp, top, margin, energy, cosine))


def source_thresholds(scores, retention=.99):
    values = _matrix(scores, 'source V knownness', len(METHODS))
    if not 0. < retention <= 1.:
        raise ValueError('Source retention must be in (0,1]')
    # Conservative ties; no target scores, roles or truth are accepted here.
    return np.quantile(values, 1. - retention, axis=0, method='lower')


def apply_rejection(logits, scores, thresholds):
    raw = _matrix(logits, 'logits')
    values = _matrix(scores, 'knownness', len(METHODS))
    limits = np.asarray(thresholds, dtype=np.float64)
    if len(raw) != len(values) or limits.shape != (len(METHODS),) or not np.isfinite(limits).all():
        raise ValueError('Invalid frozen rejection thresholds')
    closed = raw.argmax(axis=1).astype(np.int64)
    accepted = values >= limits
    return dict(closed_prediction=closed, accepted=accepted,
                prediction=np.where(accepted, closed[:, None], -1).astype(np.int64))
