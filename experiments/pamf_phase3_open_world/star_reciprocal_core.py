"""Pure NumPy, U-only reciprocal pseudo-labels and anonymous registration.

All fitting functions consume only the unlabeled adaptation pool.  ``infer``
is row independent and never changes the supplied frozen registry.
"""
from __future__ import annotations

import numpy as np


REGISTRY_SCHEMA = "pamf_phase3_star_reciprocal_registry_v1"
FEATURE_DIM = 160
OLD_CLASSES = 6
NEIGHBORS = 10


def _array(value, dtype, shape, name):
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.shape != shape:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)} and shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value):
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != FEATURE_DIM:
        raise ValueError("unit must have shape [N,160]")
    result = _array(result, np.float32, result.shape, "unit")
    x64 = result.astype(np.float64)
    norm2 = np.einsum("ij,ij->i", x64, x64, optimize=False)
    if not np.allclose(norm2, 1.0, rtol=0, atol=2e-5):
        raise ValueError("unit rows must have unit Euclidean norm")
    return result


def _ids(value, n):
    result = np.asarray(value)
    if result.shape != (n,) or result.dtype.kind != "U":
        raise ValueError("ids must be a Unicode array of shape [N]")
    if len(np.unique(result)) != n or np.any(result == ""):
        raise ValueError("ids must be unique nonempty opaque IDs")
    return result


def reciprocal_graph(unit, idsU):
    """Return mutual k-nearest edges and ranked neighbors in original row order.

    Similarities use FP64 fixed-order dot products.  Equal similarities use
    opaque ID lexical order, so row permutations do not change the graph.
    """
    unit = _unit(unit)
    n = len(unit)
    ids = _ids(idsU, n)
    k = min(NEIGHBORS, max(n - 1, 0))
    if k == 0:
        return np.zeros((n, n), bool), np.empty((n, 0), np.int64)
    x64 = unit.astype(np.float64)
    similarities = np.einsum("id,jd->ij", x64, x64, optimize=False)
    np.fill_diagonal(similarities, -np.inf)
    id_order = np.argsort(ids, kind="stable")
    ranked = np.argsort(-similarities[:, id_order], axis=1, kind="stable")
    neighbors = id_order[ranked[:, :k]].astype(np.int64, copy=False)
    directed = np.zeros((n, n), bool)
    directed[np.arange(n)[:, None], neighbors] = True
    return directed & directed.T, neighbors


def pseudolabels(unit, logits, margins, ids, arm="self_pl"):
    """Freeze trusted self-labels and optional reciprocal hard-label expansion.

    Rejected rows never receive an old-class pseudo-label.  Only initially
    trusted neighbors vote; newly expanded rows cannot recursively vote.
    """
    unit = _unit(unit)
    n = len(unit)
    logits = _array(logits, np.float32, (n, OLD_CLASSES), "logits")
    margins = _array(margins, np.float64, (n,), "margins")
    ids = _ids(ids, n)
    if arm not in ("self_pl", "reciprocal"):
        raise ValueError("arm must be 'self_pl' or 'reciprocal'")
    reciprocal, neighbors = reciprocal_graph(unit, ids)
    values = logits.astype(np.float64)
    if n:
        exponential = np.exp(values - values.max(axis=1, keepdims=True))
        msp = exponential.max(axis=1) / exponential.sum(axis=1)
        original_labels = values.argmax(axis=1).astype(np.int64)
    else:
        msp = np.empty(0, np.float64)
        original_labels = np.empty(0, np.int64)
    accepted = margins >= 0
    trusted = accepted & (msp >= 0.95)
    expanded = np.zeros(n, bool)
    labels = np.full(n, -1, np.int64)
    labels[trusted] = original_labels[trusted]
    if arm == "reciprocal":
        for row in np.flatnonzero(accepted & ~trusted):
            voters = np.flatnonzero(reciprocal[row] & trusted)
            if len(voters) < 3:
                continue
            votes = np.bincount(original_labels[voters], minlength=OLD_CLASSES)
            winner = int(votes.argmax())
            if int(votes[winner]) / len(voters) >= 0.8:
                labels[row] = winner
                expanded[row] = True
    selected = trusted | expanded
    targets = np.zeros((n, OLD_CLASSES), np.float32)
    targets[np.flatnonzero(selected), labels[selected]] = 1
    statistics = {
        "arm": arm,
        "total": n,
        "initial_accepted_count": int(accepted.sum()),
        "trusted_count": int(trusted.sum()),
        "expanded_count": int(expanded.sum()),
        "selected_count": int(selected.sum()),
        "expanded_argmax_changed_count": int(np.sum(expanded & (labels != original_labels))),
    }
    return dict(targets=targets, labels=labels, selected=selected, trusted=trusted,
                expanded=expanded, neighbors=neighbors, reciprocal=reciprocal,
                statistics=statistics)


def _distances(left, right):
    delta = left - right
    return np.sqrt(np.einsum("...d,...d->...", delta, delta, optimize=False))


def fit_registry(unit_U, current_margin, original_margin, trusted, ids):
    """Register stable components rejected by both current and original gates.

    The graph is built on all U rows before taking the candidate subgraph.
    Component number and anonymous identity order come from graph structure
    and opaque IDs; no true class count is supplied.
    """
    unit = _unit(unit_U)
    n = len(unit)
    current_margin = _array(current_margin, np.float64, (n,), "current_margin")
    original_margin = _array(original_margin, np.float64, (n,), "original_margin")
    trusted = _array(trusted, np.bool_, (n,), "trusted")
    ids = _ids(ids, n)
    reciprocal, neighbors = reciprocal_graph(unit, ids)
    candidate = (current_margin < 0) & (original_margin < 0)
    id_order = np.argsort(ids, kind="stable")
    component_labels = np.full(n, -1, np.int64)
    groups = []
    for seed in id_order:
        if not candidate[seed] or component_labels[seed] >= 0:
            continue
        index = len(groups)
        component_labels[seed] = index
        pending = [int(seed)]
        members = []
        while pending:
            row = pending.pop()
            members.append(row)
            adjacent = np.flatnonzero(reciprocal[row] & candidate & (component_labels < 0))
            component_labels[adjacent] = index
            pending.extend(adjacent.tolist())
        members = np.asarray(members, np.int64)
        groups.append(members[np.argsort(ids[members], kind="stable")])
    assignments = np.full(n, -1, np.int64)
    centers, radii, registered_indices, components = [], [], [], []
    x64 = unit.astype(np.float64)
    for index, members in enumerate(groups):
        values = x64[members]
        size = len(members)
        center = values.mean(axis=0)
        if size > 1:
            other_centers = (values.sum(axis=0) - values) / (size - 1)
            radius = max(1e-6, float(np.quantile(_distances(values, other_centers), 0.95)))
        else:
            radius = 1e-6
        boundary = float(np.mean(np.any(reciprocal[members][:, trusted], axis=1)))
        halves = values[::2], values[1::2]
        half_distance = (float(_distances(halves[0].mean(axis=0), halves[1].mean(axis=0)))
                         if len(halves[1]) else None)
        reasons = []
        if size < 16:
            reasons.append("size_below_minimum")
        if boundary > 0.05:
            reasons.append("trusted_boundary_exceeds_limit")
        if min(len(halves[0]), len(halves[1])) >= 8 and half_distance > radius:
            reasons.append("unstable_halves")
        registered = not reasons
        if registered:
            assignments[members] = len(centers)
            centers.append(center)
            radii.append(radius)
            registered_indices.append(index)
        components.append(dict(index=index, minimum_id=str(ids[members[0]]), size=size,
                               trusted_boundary_fraction=boundary, half_center_distance=half_distance,
                               radius=radius, registered=registered, reasons=reasons))
    state = dict(schema=REGISTRY_SCHEMA,
                 centers=np.asarray(centers, np.float64).reshape(-1, FEATURE_DIM),
                 radii=np.asarray(radii, np.float64))
    diagnostics = dict(assignments=assignments, candidate_mask=candidate,
                       component_labels=component_labels,
                       component_sizes=np.asarray([len(group) for group in groups], np.int64),
                       registered_component_indices=np.asarray(registered_indices, np.int64),
                       components=components, neighbors=neighbors, reciprocal=reciprocal,
                       registered_count=len(centers))
    return state, diagnostics


def infer(unit_Q, logits, margins, state, registered: bool):
    """Predict each Q row using frozen state, protecting every accepted old row.

    Returns prediction, current default-head closed prediction, maximum joint
    margin (knownness), and all old/optional anonymous slot margins.
    """
    unit = _unit(unit_Q)
    n = len(unit)
    logits = _array(logits, np.float32, (n, OLD_CLASSES), "logits")
    margins = _array(margins, np.float64, (n,), "margins")
    if not isinstance(registered, (bool, np.bool_)):
        raise ValueError("registered must be boolean")
    if not isinstance(state, dict) or set(state) != {"schema", "centers", "radii"}:
        raise ValueError("registry state must contain exactly schema, centers, and radii")
    if state["schema"] != REGISTRY_SCHEMA:
        raise ValueError("unsupported registry schema")
    centers = np.asarray(state["centers"])
    if centers.ndim != 2 or centers.shape[1] != FEATURE_DIM:
        raise ValueError("centers must have shape [K,160]")
    centers = _array(centers, np.float64, centers.shape, "centers")
    radii = _array(state["radii"], np.float64, (len(centers),), "radii")
    if np.any(radii < 1e-6):
        raise ValueError("registry radii must be at least 1e-6")
    values = logits.astype(np.float64)
    if n:
        closed = values.argmax(axis=1).astype(np.int64)
        joint = (values - values.max(axis=1, keepdims=True)) / 30 + margins[:, None]
    else:
        closed = np.empty(0, np.int64)
        joint = np.empty((0, OLD_CLASSES), np.float64)
    if registered and len(centers):
        distance = _distances(unit.astype(np.float64)[:, None, :], centers[None, :, :])
        anonymous = 1 - distance / radii[None, :]
        accepted = margins >= 0
        anonymous[accepted] = -1 - np.abs(anonymous[accepted])
        joint = np.concatenate([joint, anonymous], axis=1)
    knownness = joint.max(axis=1)
    prediction = np.where(knownness >= 0, joint.argmax(axis=1), -1).astype(np.int64)
    return prediction, closed, knownness, joint
