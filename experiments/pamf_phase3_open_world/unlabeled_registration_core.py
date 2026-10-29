"""Pure numerical G0/G1/G2 primitives; no IO, truth, model or optimizer access."""
from __future__ import annotations

import numpy as np


RUN = '20261029-phase3-unlabeled-registration-s2026092701-r01'
SCHEMA = 'pamf_phase3_unlabeled_registration_core_v1'
DIMENSIONS = dict(fused160=160, time160=160, frequency160=160, pa160=160,
                  reference29=29, signature336=336, spectrum128=128, pa_moments48=48)
EMBEDDING_BRANCHES = ('fused160', 'time160', 'frequency160', 'pa160', 'pa_moments48')
EMBEDDING_DIM = 688
RECEIVERS = (1, 3, 4, 6, 8)
CORE_PARAMETERS = dict(msp_min=.95, fused_cosine_min=.5, branch_votes_min=2,
    nearest_tie_gap=1e-8, per_pseudoclass_min=8, qualified_classes_min=3, bias_weight=.5, bias_norm_max=.1)


def _matrix(value, name, columns=None, *, fp32=False):
    a = np.asarray(value)
    if (a.ndim != 2 or a.shape[1] == 0 or (columns is not None and a.shape[1] != columns)
            or a.dtype.kind != 'f' or (fp32 and a.dtype != np.float32) or not np.isfinite(a).all()):
        raise ValueError('Finite floating matrix shape/dtype differs: ' + name)
    return a.astype(np.float64, copy=False)


def _vector(value, name, count, *, integers=False):
    a = np.asarray(value)
    if a.shape != (count,) or a.dtype.kind not in ('iu' if integers else 'f') or not np.isfinite(a).all():
        raise ValueError('Aligned finite vector required: ' + name)
    return a if integers else a.astype(np.float64, copy=False)


def _scalar(value, name, *, positive=False, nonnegative=False):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in 'fiu' or not np.isfinite(a):
        raise ValueError('Finite scalar required: ' + name)
    result = float(a)
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError('Frozen scalar outside its valid range: ' + name)
    return result


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _directions(value, name):
    a = _matrix(value, name, 160)
    norms = np.linalg.norm(a, axis=1)
    if not np.all(np.isclose(norms, 1., atol=5e-6, rtol=0) | (norms <= 1e-12)):
        raise ValueError('Unit F0 directions (or the zero direction) required: ' + name)
    return a


def _branches(branches):
    if not isinstance(branches, dict) or set(branches) != set(DIMENSIONS):
        raise ValueError('Exactly the eight original FP32 branch matrices are required')
    result = {key: _matrix(branches[key], key, d, fp32=True) for key, d in DIMENSIONS.items()}
    if len({len(x) for x in result.values()}) != 1 or np.any(result['reference29'][:, 27] != 0):
        raise ValueError('Aligned branch rows and strictly masked reference coordinate 27 required')
    return result


def _geometry(geometry):
    if (not isinstance(geometry, dict) or set(geometry) != {'schema', 'class_count', 'branches'}
            or geometry['schema'] != 'phase3_source_open_geometry_v1' or geometry['class_count'] != 6
            or not isinstance(geometry['branches'], dict) or set(geometry['branches']) != set(DIMENSIONS)):
        raise ValueError('Original six-class source-L aggregate geometry required')
    result = {}
    for key, d in DIMENSIONS.items():
        branch = geometry['branches'][key]
        if not isinstance(branch, dict) or set(branch) != {'mean', 'scale', 'prototypes', 'whitener'}:
            raise ValueError('Only source mean/scale/prototypes/shared whitener are allowed')
        mean, scale = (_vector(branch[k], key + ' ' + k, d) for k in ('mean', 'scale'))
        p, w = _matrix(branch['prototypes'], key + ' prototypes', d), _matrix(branch['whitener'], key + ' whitener', d)
        if p.shape != (6, d) or w.shape != (d, d) or np.any(scale <= 0):
            raise ValueError('Six source prototypes, positive scale and square shared whitener required')
        result[key] = dict(mean=mean, scale=scale, prototypes=p, whitener=w)
    ref = result['reference29']
    if ref['mean'][27] != 0 or ref['scale'][27] != 1 or np.any(ref['prototypes'][:, 27] != 0):
        raise ValueError('Source NoExplicitCFO geometry coordinate 27 must remain masked')
    return result


def _fp32(value, name):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(value, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite FP32 result: ' + name)
    return result


def embedding(branches, geometry):
    """Five equally weighted source-standardized/whitened unit branches."""
    values, fitted = _branches(branches), _geometry(geometry)
    parts = []
    for key in EMBEDDING_BRANCHES:
        g = fitted[key]
        standardized = (values[key] - g['mean']) / g['scale']
        parts.append(_unit(standardized @ g['whitener'].T) / np.sqrt(5.))
    return _fp32(np.concatenate(parts, axis=1), 'five-branch embedding')


def _nearest(unit, prototypes):
    cosine = np.clip(unit @ _unit(prototypes).T, -1., 1.)
    ordered = np.argsort(cosine, axis=1, kind='stable')
    label = ordered[:, -1]
    best = cosine[np.arange(len(unit)), label]
    unique = best - cosine[np.arange(len(unit)), ordered[:, -2]] > CORE_PARAMETERS['nearest_tie_gap']
    return label, best, unique


def _head(logits):
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return logits.argmax(axis=1), exponentials.max(axis=1) / exponentials.sum(axis=1)


def fit_bias(branches, E51_logits, F0_logits, geometry):
    """Fit one bounded standardized-space bias using unlabeled old-class cores."""
    values, fitted = _branches(branches), _geometry(geometry)
    n = len(values['fused160'])
    e, f = _matrix(E51_logits, 'E51 logits', 6), _matrix(F0_logits, 'F0 logits', 6)
    if len(e) != n or len(f) != n: raise ValueError('Dual-head rows must align with U branch rows')
    label, e_msp = _head(e); f_label, f_msp = _head(f)
    units = {key: _unit((x - fitted[key]['mean']) / fitted[key]['scale']) for key, x in values.items()}
    fused, cosine, unique = _nearest(units['fused160'], fitted['fused160']['prototypes'])
    votes = np.zeros(n, np.int64)
    for key in ('time160', 'frequency160', 'pa160'):
        nearest, _, unambiguous = _nearest(units[key], fitted[key]['prototypes'])
        votes += (nearest == label) & unambiguous
    core = ((e_msp >= .95) & (f_msp >= .95) & (label == f_label)
            & unique & (fused == label) & (cosine >= .5) & (votes >= 2))
    support = np.bincount(label[core], minlength=6).astype(np.int64)
    qualified = np.flatnonzero(support >= 8).astype(np.int64)
    selected = np.flatnonzero(core & np.isin(label, qualified)).astype(np.int64)
    applied = len(qualified) >= 3
    bias = {key: np.zeros(d, np.float64) for key, d in DIMENSIONS.items()}
    if applied:
        for key in DIMENSIONS:
            means = np.asarray([units[key][core & (label == c)].mean(axis=0) for c in qualified])
            difference = _unit(fitted[key]['prototypes'][qualified]) - means
            b = .5 * difference.mean(axis=0)
            if key == 'reference29': b[27] = 0.
            norm = float(np.linalg.norm(b))
            bias[key] = b * min(1., .1 / max(norm, 1e-12))
    weights = np.zeros(6, np.float64)
    if applied: weights[qualified] = 1. / len(qualified)
    diagnostics = dict(schema=SCHEMA, parameters=dict(CORE_PARAMETERS), applied=applied,
        reason='qualified_old_cores' if applied else 'insufficient_qualified_old_cores',
        core_mask=core, pseudo_labels=label.astype(np.int64), support_counts=support,
        qualified_classes=qualified, selected_indices=selected, class_weights=weights,
        votes=votes, core_count=int(core.sum()), selected_count=len(selected) if applied else 0,
        bias_norms={key: float(np.linalg.norm(b)) for key, b in bias.items()})
    return bias, diagnostics


def apply_bias(branches, geometry, bias):
    """Keep each standardized raw norm, shifting only its unit direction."""
    values, fitted = _branches(branches), _geometry(geometry)
    if not isinstance(bias, dict) or set(bias) != set(DIMENSIONS):
        raise ValueError('One fixed bias vector per original branch required')
    result = {}
    for key, d in DIMENSIONS.items():
        b = _vector(bias[key], key + ' bias', d).copy()
        if key == 'reference29': b[27] = 0.
        if np.linalg.norm(b) > .1 + 1e-8: raise ValueError('Fixed standardized bias norm cannot exceed .1')
        if not np.any(b):
            result[key] = np.array(branches[key], copy=True)
            continue
        g = fitted[key]; standardized = (values[key] - g['mean']) / g['scale']
        radius = np.linalg.norm(standardized, axis=1, keepdims=True)
        adapted = _unit(_unit(standardized) + b) * radius
        result[key] = _fp32(adapted * g['scale'] + g['mean'], key + ' adapted')
        if key == 'reference29': result[key][:, 27] = 0.
    return result


def _cosdist(a, b):
    return 1. - np.clip(a @ b.T, -1., 1.)


def source_density(embedding, F0_unit, y, rx, day):
    """Source-only seventh-other-neighbor eps and LOO F0 angular radius.

    Each source physical record occurs once. group_indices are temporary row
    indices for caller aggregation; deployment retains only scalar/anchor data.
    """
    z, f = _matrix(embedding, 'source embedding'), _directions(F0_unit, 'source F0 unit')
    n = len(z)
    y, rx, day = (_vector(x, name, n, integers=True) for x, name in ((y, 'source y'), (rx, 'source RX'), (day, 'source day')))
    if (len(f) != n or set(np.unique(y).tolist()) != set(range(6))
            or set(np.unique(rx).tolist()) != set(RECEIVERS) or set(np.unique(day).tolist()) != {1, 2, 3}):
        raise ValueError('Only complete source-L old-six/five-RX/three-day labels/domains allowed')
    keys, indices, anchors, seventh, angular = [], [], [], [], []
    for label in range(6):
        for receiver in RECEIVERS:
            for date in (1, 2, 3):
                rows = np.flatnonzero((y == label) & (rx == receiver) & (day == date)).astype(np.int64)
                if len(rows) < 8: raise ValueError('Every source class/RX/day group needs at least eight physical rows')
                x = z[rows]; norm2 = np.einsum('nd,nd->n', x, x)
                distance = np.sqrt(np.maximum(norm2[:, None] + norm2[None, :] - 2. * (x @ x.T), 0.))
                np.fill_diagonal(distance, np.inf)
                seventh.append(np.partition(distance, 6, axis=1)[:, 6])
                group = f[rows]; total = group.sum(axis=0, keepdims=True)
                leave_one_out = _unit((total - group) / (len(rows) - 1))
                angular.append(1. - np.clip(np.einsum('nd,nd->n', group, leave_one_out), -1., 1.))
                anchors.append(_unit(group.mean(axis=0, keepdims=True))[0])
                keys.append((label, receiver, date)); indices.append(rows)
    return dict(eps=max(float(np.quantile(np.concatenate(seventh), .95)), 1e-6),
        radius=max(float(np.quantile(np.concatenate(angular), .95)), 1e-6),
        anchor_units=_fp32(anchors, 'source F0 group anchors'), group_keys=np.asarray(keys, np.int64),
        group_indices=tuple(indices), diagnostics=dict(schema=SCHEMA, source_count=n, group_count=90,
            group_counts=np.asarray([len(rows) for rows in indices], np.int64), neighbor_rank=7,
            self_neighbor_excluded=True, source_views_per_physical=1, quantile=.95,
            radius_center='leave-one-out same-old-class/RX/day F0 unit center'))


def discover(embedding_U, F0_unit_U, knownness_U, threshold, ids, eps, radius, anchor_units, anchor_q, prefix):
    """Discover any number of density clusters among primary-MLP rejects only."""
    z, f = _matrix(embedding_U, 'U embedding'), _directions(F0_unit_U, 'U F0 unit')
    n = len(z); q = _vector(knownness_U, 'U primary knownness', n)
    threshold = _scalar(threshold, 'frozen threshold'); eps = _scalar(eps, 'source eps', positive=True)
    radius = _scalar(radius, 'source radius', positive=True)
    a = _directions(anchor_units, 'source anchors'); aq = _vector(anchor_q, 'source anchor old margins', len(a))
    opaque = np.asarray(ids)
    if (len(f) != n or opaque.shape != (n,) or opaque.dtype.kind != 'U' or np.any(np.char.strip(opaque) == '')
            or len(set(opaque.tolist())) != n or not len(a) or not isinstance(prefix, str) or not prefix.strip()):
        raise ValueError('Aligned U rows, unique physical opaque IDs and source anchors/prefix required')
    candidates = np.flatnonzero(q < threshold)
    candidates = candidates[np.argsort(opaque[candidates], kind='stable')]
    raw_labels = np.full(n, -1, np.int64)
    if len(candidates):
        from sklearn.cluster import DBSCAN
        raw_labels[candidates] = DBSCAN(eps=eps, min_samples=8, metric='euclidean', n_jobs=1).fit_predict(z[candidates])
    clusters = [np.flatnonzero(raw_labels == label) for label in np.unique(raw_labels) if label >= 0]
    registered = [rows for rows in clusters if len(rows) >= 16]
    registered.sort(key=lambda rows: min(opaque[rows].tolist()))
    assignments = np.full(n, -1, np.int64)
    for index, rows in enumerate(registered): assignments[rows] = index
    prototypes = (_fp32(_unit(np.asarray([f[rows].mean(axis=0) for rows in registered])), 'anonymous prototypes')
                  if registered else np.empty((0, 160), np.float32))
    penalty = .05
    if len(prototypes): penalty += max(0., float(np.max(1. - _cosdist(a, prototypes) / radius - aq[:, None])))
    anonymous_ids = np.asarray([prefix + '-anon-' + str(i + 1).zfill(4) for i in range(len(registered))], dtype=str)
    return dict(prototypes=prototypes, penalty=penalty, anonymous_ids=anonymous_ids, assignments=assignments,
        diagnostics=dict(schema=SCHEMA, candidate_mask=q < threshold, candidate_count=len(candidates),
            dbscan_labels=raw_labels, dbscan_cluster_count=len(clusters), registered_count=len(registered),
            cluster_sizes=np.asarray([len(rows) for rows in registered], np.int64),
            smaller_clusters_not_registered=sum(len(rows) < 16 for rows in clusters),
            min_samples=8, registration_unique_ID_min=16, eps=eps, radius=radius,
            unknown_count_input=False, forced_partition=False, zero_registered_clusters_valid=True))


def joint_margins(F0_logits, F0_unit, knownness, threshold, sigma, proto, radius, penalty):
    """All old-six and anonymous columns, on one frozen source margin scale."""
    logits, f = _matrix(F0_logits, 'F0 old-six logits', 6), _directions(F0_unit, 'query F0 unit')
    n = len(logits); q = _vector(knownness, 'query primary knownness', n)
    threshold = _scalar(threshold, 'frozen threshold'); sigma = _scalar(sigma, 'source V sigma', positive=True)
    radius = _scalar(radius, 'source radius', positive=True); penalty = _scalar(penalty, 'source-anchor penalty', nonnegative=True)
    prototypes = _directions(proto, 'anonymous prototypes')
    if len(f) != n: raise ValueError('Query default-head, F0 unit and knownness rows must align')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        q_margin = (q - threshold) / sigma
        # Preserve REG0's exact comparison even for a representable score gap
        # whose division by an extreme finite sigma would underflow to zero.
        tiny = np.nextafter(0., 1.)
        q_margin = np.where((q < threshold) & (q_margin == 0), -tiny, q_margin)
        old = (logits - logits.max(axis=1, keepdims=True)) / 30. + q_margin[:, None]
        new = 1. - _cosdist(f, prototypes) / radius - penalty
        result = np.concatenate((old, new), axis=1)
    if not np.isfinite(result).all(): raise ValueError('Frozen joint margins must remain finite')
    return result


def joint_predict(F0_logits, F0_unit, knownness, threshold, sigma, proto, radius, penalty):
    margins = joint_margins(F0_logits, F0_unit, knownness, threshold, sigma, proto, radius, penalty)
    winner = margins.argmax(axis=1)
    return np.where(margins[np.arange(len(margins)), winner] < 0., -1, winner).astype(np.int64)
