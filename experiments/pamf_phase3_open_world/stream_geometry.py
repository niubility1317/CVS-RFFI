"""Frozen source geometry and unlabeled stream decisions; no I/O or model fitting."""
from __future__ import annotations

import numpy as np


DIMENSION = 160
MAX_ROUNDS = 20
MAX_FREE = 32
SCHEMA = 'phase3_stream_geometry_v1'


def _matrix(value, name, width=DIMENSION):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Finite feature matrix required: ' + name) from exc
    if result.ndim != 2 or not len(result) or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError('Nonempty finite matrix with width %d required: %s' % (width, name))
    return result


def _unit(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def _distance2(points, centers):
    result = (np.einsum('nd,nd->n', points, points)[:, None]
              + np.einsum('kd,kd->k', centers, centers)[None, :]
              - 2. * np.einsum('nd,kd->nk', points, centers, optimize=False))
    return np.maximum(result, 0.)


def _check_bundle(bundle):
    if (not isinstance(bundle, dict) or bundle.get('schema') != SCHEMA
            or bundle.get('feature_dim') != DIMENSION or type(bundle.get('class_count')) is not int
            or bundle['class_count'] < 2):
        raise ValueError('Frozen source geometry bundle required')
    k = bundle['class_count']
    for key, shape in (('prototypes', (k, DIMENSION)), ('anchors', (k, DIMENSION)),
                       ('covariance', (DIMENSION, DIMENSION)), ('chol', (DIMENSION, DIMENSION)),
                       ('whitener', (DIMENSION, DIMENSION))):
        value = np.asarray(bundle.get(key))
        if value.shape != shape or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
            raise ValueError('Invalid source aggregate: ' + key)
    for key in ('lambda_det', 'lambda_cluster'):
        if not np.isscalar(bundle.get(key)) or not np.isfinite(bundle[key]) or bundle[key] <= 0:
            raise ValueError('Positive fixed source radius required')
    if bundle['lambda_cluster'] != bundle['lambda_det'] / 4.:
        raise ValueError('Cluster radius must be the fixed source detection radius / 4')
    return k


def fit_source(train_features, train_y, calibration_features):
    """Fit L aggregates and the caller's legal seen-RX known V detection radius."""
    train = _matrix(train_features, 'source L')
    calibration = _matrix(calibration_features, 'source known calibration')
    labels = np.asarray(train_y)
    if labels.shape != (len(train),) or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Only visible consecutive old source labels are accepted')
    observed = np.unique(labels)
    k = len(observed)
    if k < 2 or not np.array_equal(observed, np.arange(k)) or len(train) <= k:
        raise ValueError('All old source classes 0..K-1 and within-class degrees of freedom are required')
    unit = _unit(train)
    prototypes = np.asarray([unit[labels == label].mean(axis=0) for label in range(k)])
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Degenerate source class mean')
    prototypes = prototypes / norms
    residual = unit - prototypes[labels]
    scatter = residual.T @ residual / (len(train) - k)
    scale = max(float(np.trace(scatter) / DIMENSION), 1e-8)
    covariance = .8 * scatter + .2 * scale * np.eye(DIMENSION)
    chol = np.linalg.cholesky(covariance)
    whitener = np.linalg.solve(chol, np.eye(DIMENSION))
    anchors = prototypes @ whitener.T
    distances = _distance2(_unit(calibration) @ whitener.T, anchors).min(axis=1)
    radius = max(float(np.quantile(distances, .99, method='higher')), 1e-8)
    return dict(schema=SCHEMA, feature_dim=DIMENSION, class_count=k,
        source_count=len(train), source_counts=np.bincount(labels, minlength=k).astype(np.int64),
        calibration_count=len(calibration), prototypes=prototypes, covariance=covariance,
        chol=chol, whitener=whitener, anchors=anchors, lambda_det=radius, lambda_cluster=radius / 4.)


def _stream_inputs(bundle, features, logits, ids):
    k = _check_bundle(bundle)
    raw = _matrix(features, 'unlabeled stream')
    logits = _matrix(logits, 'original frozen head logits', k)
    ids = np.asarray(ids)
    if len(logits) != len(raw) or ids.shape != (len(raw),) or ids.dtype.kind != 'U':
        raise ValueError('U features, original logits and opaque Unicode IDs must align')
    if np.any(ids == '') or len(set(ids.tolist())) != len(ids):
        raise ValueError('Unique nonempty opaque U IDs are required')
    order = np.argsort(ids, kind='stable')
    return _unit(raw[order]) @ bundle['whitener'].T, logits[order].argmax(axis=1), ids[order].copy()


def _shift(q, pseudo, ids, anchors, radius):
    k = len(anchors)
    shift = np.zeros(DIMENSION)
    counts = np.zeros((10, k), dtype=np.int64)
    separation = _distance2(anchors, anchors)
    np.fill_diagonal(separation, np.inf)
    cap = .25 * np.sqrt(separation.min())
    residual = q - anchors[pseudo]
    huber_radius = .5 * np.sqrt(radius)
    history = [shift.copy()]
    for iteration in range(10):
        distances = _distance2(q - shift, anchors)
        ranked = np.argsort(distances, axis=1, kind='stable')
        nearest = ranked[:, 0]
        d1 = distances[np.arange(len(q)), nearest]
        d2 = distances[np.arange(len(q)), ranked[:, 1]]
        core = (nearest == pseudo) & (d1 <= radius) & (d1 / (d2 + 1e-12) <= .5)
        means = []
        for label in range(k):
            members = np.flatnonzero(core & (pseudo == label))
            # d1 is primary and opaque ID is the deterministic tie breaker.
            members = members[np.lexsort((ids[members], d1[members]))]
            members = members[:int(np.floor(.75 * len(members)))]
            counts[iteration, label] = len(members)
            if len(members) < 8:
                continue
            r = residual[members]
            weights = np.minimum(1., huber_radius / np.maximum(np.linalg.norm(r - shift, axis=1), 1e-12))
            means.append((r * weights[:, None]).sum(axis=0) / weights.sum())
        if len(means) >= 2:
            proposal = np.mean(means, axis=0) / 1.1
            norm = np.linalg.norm(proposal)
            if norm > cap:
                proposal *= cap / norm
            shift = .5 * shift + .5 * proposal
        history.append(shift.copy())
    return shift, counts, np.asarray(history), float(cap)


def _assign(q, anchors, free):
    distances = _distance2(q, np.concatenate((anchors, free), axis=0))
    assignment = distances.argmin(axis=1)  # Old anchors precede free components on ties.
    return assignment, distances[np.arange(len(q)), assignment]


def _canonical_free(q, anchors, free, ids, *, update):
    assignment, _ = _assign(q, anchors, free)
    components = []
    for j, center in enumerate(free):
        members = np.flatnonzero(assignment == len(anchors) + j)
        if len(members):
            components.append((ids[members[0]], q[members].mean(axis=0) if update else center.copy(), j))
    components.sort(key=lambda item: item[0])
    centers = np.asarray([item[1] for item in components], dtype=np.float64).reshape(-1, DIMENSION)
    reordered = assignment.copy()
    for new_j, item in enumerate(components):
        reordered[assignment == len(anchors) + item[2]] = len(anchors) + new_j
    return centers, reordered


def _merge_components(q, anchors, free, ids, penalty, assignment=None):
    if assignment is None:
        assignment, _ = _assign(q, anchors, free)
    groups = []
    for j in range(len(free)):
        members = np.flatnonzero(assignment == len(anchors) + j)
        if len(members):
            groups.append(dict(members=members, center=free[j].copy(), id=ids[members[0]]))
    merges = 0
    while groups:
        choices = []
        for i, group in enumerate(groups):
            for old in range(len(anchors)):
                cost = len(group['members']) * float(np.square(group['center'] - anchors[old]).sum())
                gain = penalty - cost
                if gain > 0:
                    choices.append((-gain, (group['id'], 'old:%08d' % old), i, None))
            for j in range(i + 1, len(groups)):
                other = groups[j]
                ni, nj = len(group['members']), len(other['members'])
                cost = ni * nj / (ni + nj) * float(np.square(group['center'] - other['center']).sum())
                gain = penalty - cost
                if gain > 0:
                    choices.append((-gain, tuple(sorted((group['id'], other['id']))), i, j))
        if not choices:
            break
        _, _, i, j = min(choices, key=lambda item: (item[0], item[1]))
        if j is not None:
            members = np.sort(np.r_[groups[i]['members'], groups[j]['members']])
            groups[i] = dict(members=members, center=q[members].mean(axis=0), id=ids[members[0]])
            del groups[j]
        else:
            del groups[i]
        merges += 1
    groups.sort(key=lambda group: group['id'])
    return np.asarray([group['center'] for group in groups], dtype=np.float64).reshape(-1, DIMENSION), merges


def _discover(q, anchors, ids, penalty):
    free = np.empty((0, DIMENSION), dtype=np.float64)
    previous_assignment, previous_free = None, None
    converged = cap_reached = False
    history = []
    for outer in range(MAX_ROUNDS):
        free, assigned = _canonical_free(q, anchors, free, ids, update=True)
        free, merges = _merge_components(q, anchors, free, ids, penalty, assigned)
        assignment, distance = _assign(q, anchors, free)
        births = 0
        while float(distance.max()) > penalty:
            if len(free) == MAX_FREE:
                cap_reached = True
                break
            farthest = int(np.argmax(distance))  # q/ids were globally sorted before any fit.
            free = np.vstack((free, q[farthest]))
            births += 1
            assignment, distance = _assign(q, anchors, free)
        free, _ = _canonical_free(q, anchors, free, ids, update=False)
        assignment, distance = _assign(q, anchors, free)
        history.append(dict(round=outer + 1, free_count=len(free), births=births, merges=merges,
            objective=float(distance.sum() + penalty * len(free)), max_distance2=float(distance.max())))
        same = (previous_assignment is not None and not births and not merges
                and np.array_equal(assignment, previous_assignment) and free.shape == previous_free.shape
                and np.allclose(free, previous_free, rtol=0, atol=1e-12))
        if not cap_reached and same:
            converged = True
            break
        if cap_reached:
            break
        previous_assignment, previous_free = assignment.copy(), free.copy()
    return free, assignment, converged, cap_reached, history


def _qualify(q, anchors, free, assignment, ids, penalty):
    k = len(anchors)
    qualified = np.zeros(len(free), dtype=bool)
    component_ids = []
    diagnostics = []
    for j, center in enumerate(free):
        members = np.flatnonzero(assignment == k + j)
        even, odd = members[members % 2 == 0], members[members % 2 == 1]
        radius = float(np.quantile(np.square(q[members] - center).sum(axis=1), .9, method='higher'))
        separation = float(_distance2(center[None, :], anchors).min())
        halves = float(np.square(q[even].mean(axis=0) - q[odd].mean(axis=0)).sum()) if len(even) and len(odd) else None
        tests = dict(minimum_ids=len(members) >= 16, both_halves=len(even) >= 8 and len(odd) >= 8,
            compact=radius <= penalty / 4., separated=separation >= penalty + 4. * radius,
            half_means=halves is not None and halves <= penalty / 8.)
        qualified[j] = all(tests.values())
        component_ids.append(ids[members[0]])
        diagnostics.append(dict(component_min_id=str(ids[members[0]]), count=len(members),
            even_count=len(even), odd_count=len(odd), radius90_squared=radius,
            nearest_old_distance2=separation, half_mean_distance2=halves,
            qualified=bool(qualified[j]), checks=tests))
    anonymous = np.full(len(free), -1, dtype=np.int64)
    rank = sorted(np.flatnonzero(qualified), key=lambda j: component_ids[j])
    for number, j in enumerate(rank, start=k):
        anonymous[j] = number
    return np.asarray(component_ids, dtype=ids.dtype), qualified, anonymous, diagnostics


def fit_stream(bundle, U_features, U_logits, U_ids, adapt_domain: bool):
    """Fit only the unlabeled U state; original pseudo labels stay fixed throughout."""
    if type(adapt_domain) is not bool:
        raise ValueError('adapt_domain must explicitly be bool')
    q, pseudo, ids = _stream_inputs(bundle, U_features, U_logits, U_ids)
    anchors = np.asarray(bundle['anchors'], dtype=np.float64).copy()
    if adapt_domain:
        shift, counts, shift_history, cap = _shift(q, pseudo, ids, anchors, bundle['lambda_det'])
    else:
        shift = np.zeros(DIMENSION)
        counts, shift_history, cap = np.zeros((0, len(anchors)), dtype=np.int64), shift[None, :].copy(), 0.
    corrected = q - shift
    free, assignments, converged, cap_reached, history = _discover(corrected, anchors, ids, bundle['lambda_cluster'])
    component_ids, qualified, anonymous, qualification = _qualify(corrected, anchors, free, assignments, ids, bundle['lambda_cluster'])
    return dict(schema=SCHEMA, feature_dim=DIMENSION, class_count=len(anchors), adapt_domain=adapt_domain,
        b=shift, old_centers=anchors, free_centers=free, free_ids=component_ids, qualified=qualified,
        anonymous_ids=anonymous, U_ids=ids, assignments=assignments.astype(np.int64),
        shift_iterations=10 if adapt_domain else 0, shift_core_counts=counts, shift_history=shift_history,
        shift_norm_cap=cap, rounds=len(history), converged=converged, cap_reached=cap_reached,
        limit_reached=len(free) == MAX_FREE,
        promotion_eligible=bool(converged and not cap_reached and qualified.any()),
        qualification=qualification, cluster_history=history)


def _state_shift(state):
    shift = np.asarray(state.get('b'), dtype=np.float64) if isinstance(state, dict) else np.asarray([])
    if shift.shape != (DIMENSION,) or not np.isfinite(shift).all():
        raise ValueError('Finite fitted stream shift required')
    return shift


def corrected_head_features(raw_features, bundle, state):
    """Apply the inverse-whitened shift to each raw row for the same frozen head."""
    _check_bundle(bundle)
    raw = np.asarray(raw_features)
    features = _matrix(raw, 'head features')
    shift = _state_shift(state)
    if not np.any(shift):
        return raw.copy()
    direction = bundle['chol'] @ shift
    result = features - np.linalg.norm(features, axis=1, keepdims=True) * direction
    return result.astype(raw.dtype if raw.dtype.kind == 'f' else np.float64, copy=False)


def predict_geometry(features, closed_prediction, bundle, state=None, registered: bool = False):
    """Read fixed centers independently per query row; no query statistics or updates."""
    k = _check_bundle(bundle)
    raw = _matrix(features, 'query features')
    closed = np.asarray(closed_prediction)
    if closed.shape != (len(raw),) or closed.dtype.kind not in 'iu' or np.any(closed < 0) or np.any(closed >= k):
        raise ValueError('Complete frozen-head old predictions 0..K-1 must align with queries')
    if type(registered) is not bool or (state is None and registered):
        raise ValueError('Registration requires an explicitly fitted U state')
    centers = np.asarray(bundle['anchors'], dtype=np.float64)
    slot_labels = np.arange(k, dtype=np.int64)
    shift = np.zeros(DIMENSION)
    if state is not None:
        shift = _state_shift(state)
        if (state.get('class_count') != k or not np.array_equal(state.get('old_centers'), centers)):
            raise ValueError('Stream state must retain the same frozen source anchors')
        free = np.asarray(state.get('free_centers'), dtype=np.float64)
        qualified = np.asarray(state.get('qualified'))
        anonymous = np.asarray(state.get('anonymous_ids'))
        if (free.ndim != 2 or free.shape[1] != DIMENSION or not np.isfinite(free).all()
                or qualified.shape != (len(free),) or qualified.dtype.kind != 'b'
                or anonymous.shape != (len(free),) or anonymous.dtype.kind not in 'iu'
                or np.any(anonymous[qualified] < k) or len(np.unique(anonymous[qualified])) != qualified.sum()):
            raise ValueError('Invalid qualified anonymous component registry')
        selected = np.flatnonzero(qualified)
        selected = selected[np.argsort(anonymous[selected], kind='stable')]
        centers = np.concatenate((centers, free[selected]), axis=0)
        slot_labels = np.r_[slot_labels, anonymous[selected]]
    q = _unit(raw) @ bundle['whitener'].T - shift
    distances = _distance2(q, centers)
    nearest = distances.argmin(axis=1)
    distance = distances[np.arange(len(q)), nearest]
    is_new = nearest >= k
    accepted = (distance <= bundle['lambda_det']) & (~is_new | registered)
    prediction = np.where(is_new, slot_labels[nearest], closed)
    prediction = np.where(accepted, prediction, -1).astype(np.int64)
    return dict(knownness=-distance, accepted=accepted, prediction=prediction,
        nearest_slot=slot_labels[nearest], nearest_distance2=distance, nearest_is_new=is_new)
