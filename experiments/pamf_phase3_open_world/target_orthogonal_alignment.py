"""One-shot bounded row-vector alignment from reliable unlabeled old anchors.

This project hypothesis fits rotations, not encoder/head weights or thresholds.
It preserves geometry in each branch's standardized space, not its whitened
Mahalanobis geometry. Callers guarantee distinct opaque fit-role record IDs.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import source_open_readout as source
from .source_risk_calibration import METHODS


SCHEMA = 'pamf_phase3_target_orthogonal_alignment_v1'
PARAMETERS = dict(methods=list(METHODS), class_count=6, min_records_per_class=8,
    min_qualified_classes=3, minimum_branch_votes=2, tie_gap=1e-8,
    vector_norm_epsilon=1e-12, identity_regularization=.5,
    static_gates=['msp', 'cosine'], vote_branches=['time160', 'frequency160', 'pa160'],
    sample_weight='msp * (agreeing_unique_branch_votes / 3); normalized within class',
    class_weight='equal across present qualified classes; no six-class quota',
    solver='exact polar factor on span(M.T,P.T), identity on its orthogonal complement',
    row_vector_action='standardized_raw @ R; reverse source scale and mean',
    reference_constant_coordinate=27, MLP_acceptance_required=False,
    whitened_Mahalanobis_preserved=False)
EPSILON = PARAMETERS['vector_norm_epsilon']


def _standardized(values, branch):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = (values - branch['mean']) / branch['scale']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized branch input')
    return result


def _unique_winner(values):
    pair = np.partition(values, -2, axis=1)[:, -2:]
    gap = pair.max(axis=1) - pair.min(axis=1)
    return values.argmax(axis=1).astype(np.int64), gap > PARAMETERS['tie_gap']


def _nearest(values, branch):
    unit = source._unit(_standardized(values, branch))
    prototypes = source._unit(np.asarray(branch['prototypes'], dtype=np.float64))
    return _unique_winner(unit @ prototypes.T)


def _inputs(branches, logits, scores, thresholds, geometry):
    values, fitted = source._branches(branches), source._geometry(geometry)
    n = len(values['fused160'])
    raw, knownness, limits = np.asarray(logits), np.asarray(scores), np.asarray(thresholds)
    if raw.shape != (n, 6) or raw.dtype.kind != 'f' or not np.isfinite(raw).all():
        raise ValueError('Aligned finite six-class original-head logits are required')
    if knownness.shape != (n, len(METHODS)) or knownness.dtype.kind != 'f' or not np.isfinite(knownness).all():
        raise ValueError('Aligned seven finite frozen scores in fixed METHODS order are required')
    if limits.shape != (len(METHODS),) or limits.dtype != np.float64 or not np.isfinite(limits).all():
        raise ValueError('Seven FP64 unchanged source thresholds are required')
    msp = knownness[:, METHODS.index('msp')]
    if np.any(msp < 0.) or np.any(msp > 1.):
        raise ValueError('MSP sample weights must be probabilities')
    return values, raw.astype(np.float64, copy=False), knownness, limits, fitted


def _identity_branches():
    rotations, diagnostics = {}, {}
    for name, dimension in source.DIMENSIONS.items():
        rotations[name] = np.eye(dimension, dtype=np.float64)
        diagnostics[name] = dict(fitted=False, reason='global_identity', valid_core_classes=0,
            fitted_classes=[], rotation_frobenius=0., orthogonality_error=0.,
            anchor_cost_before=None, anchor_cost_after=None,
            fit_dimension=dimension - (name == 'reference29'))
    return rotations, diagnostics


def _polar(means, prototypes):
    """Exactly polar(I + .5 M.T@P/K), using its <=12-D active subspace.

    Both row/column spaces of C lie in span(M.T,P.T). On its complement A
    and its polar factor are I, so this is the full SVD formula without a
    cubic factorization in dimensions up to 336. No rank cutoff is fitted.
    """
    dimension, count = means.shape[1], len(means)
    try:
        basis, _ = np.linalg.qr(np.concatenate((means.T, prototypes.T), axis=1), mode='reduced')
        small_identity = np.eye(basis.shape[1])
        cross = (basis.T @ means.T) @ (prototypes @ basis) / count
        u, singular, vt = np.linalg.svd(small_identity + .5 * cross, full_matrices=False)
        rotation = np.eye(dimension) + basis @ (u @ vt - small_identity) @ basis.T
    except np.linalg.LinAlgError as exc:
        raise ValueError('Fixed orthogonal polar factorization failed') from exc
    if not np.isfinite(rotation).all() or not np.isfinite(singular).all():
        raise ValueError('Nonfinite orthogonal polar factorization')
    return rotation


def _fit_branch(name, values, branch, selected, labels, weights, classes):
    dimension = source.DIMENSIONS[name]
    active = np.arange(dimension)
    if name == 'reference29':
        active = active[active != PARAMETERS['reference_constant_coordinate']]
    unit = source._unit(_standardized(values[selected], branch)[:, active])
    centers = np.asarray(branch['prototypes'], dtype=np.float64)[:, active]
    kept, means, prototypes = [], [], []
    for label in classes:
        mask = labels[selected] == label
        current = weights[selected][mask]
        total = current.sum()
        if total <= EPSILON:
            continue
        mean = np.sum(unit[mask] * (current / total)[:, None], axis=0)
        prototype = centers[label]
        mean_norm, prototype_norm = np.linalg.norm(mean), np.linalg.norm(prototype)
        if mean_norm <= EPSILON or prototype_norm <= EPSILON:
            continue
        kept.append(int(label))
        means.append(mean / mean_norm)
        prototypes.append(prototype / prototype_norm)
    rotation = np.eye(dimension, dtype=np.float64)
    diagnostic = dict(fitted=False, reason='insufficient_nondegenerate_classes', valid_core_classes=len(kept),
        fitted_classes=kept, rotation_frobenius=0., orthogonality_error=0.,
        anchor_cost_before=None, anchor_cost_after=None, fit_dimension=len(active))
    if len(kept) < PARAMETERS['min_qualified_classes']:
        return rotation, diagnostic
    means, prototypes = np.asarray(means), np.asarray(prototypes)
    fitted_rotation = _polar(means, prototypes)
    rotation[np.ix_(active, active)] = fitted_rotation
    orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(dimension), ord='fro'))
    if orthogonality > 1e-8:
        raise ValueError('Computed alignment is not orthogonal')
    diagnostic.update(fitted=True, reason='fitted',
        rotation_frobenius=float(np.linalg.norm(rotation - np.eye(dimension), ord='fro')),
        orthogonality_error=orthogonality,
        anchor_cost_before=float(np.mean(np.sum((means - prototypes) ** 2, axis=1))),
        anchor_cost_after=float(np.mean(np.sum((means @ fitted_rotation - prototypes) ** 2, axis=1))))
    return rotation, diagnostic


def fit_alignment(branches, logits, scores, thresholds, geometry):
    """Fit once on unlabeled distinct records; never revisit core after rotation."""
    values, raw, knownness, limits, fitted = _inputs(branches, logits, scores, thresholds, geometry)
    pseudo, head_unique = _unique_winner(raw)
    nearest, fused_unique = _nearest(values['fused160'], fitted['fused160'])
    votes = np.zeros(len(pseudo), dtype=np.int64)
    for name in PARAMETERS['vote_branches']:
        prediction, unique = _nearest(values[name], fitted[name])
        votes += unique & (prediction == pseudo)
    msp = knownness[:, METHODS.index('msp')]
    core = (head_unique & fused_unique & (nearest == pseudo)
        & (votes >= PARAMETERS['minimum_branch_votes'])
        & (msp >= limits[METHODS.index('msp')])
        & (knownness[:, METHODS.index('cosine')] >= limits[METHODS.index('cosine')]))
    counts = np.bincount(pseudo[core], minlength=6).astype(np.int64)
    classes = np.flatnonzero(counts >= PARAMETERS['min_records_per_class'])
    selected = np.flatnonzero(core & np.isin(pseudo, classes)).astype(np.int64)
    selected_counts = np.bincount(pseudo[selected], minlength=6).astype(np.int64)
    class_weights = np.zeros(6, dtype=np.float64)
    if len(classes):
        class_weights[classes] = 1. / len(classes)
    rotations, diagnostics = _identity_branches()
    result = dict(schema=SCHEMA, parameters=deepcopy(PARAMETERS), applied=False,
        reason='insufficient_qualified_classes', sample_count=len(pseudo), core_mask=core.copy(),
        selected_indices=selected, pseudo_label_counts=counts, selected_class_counts=selected_counts,
        class_weights=class_weights, rotations=rotations,
        diagnostics=dict(core_count=int(core.sum()), selected_count=len(selected), qualified_classes=classes.tolist(),
            head_tie_count=int((~head_unique).sum()), fused_tie_count=int((~fused_unique).sum()),
            distinct_record_IDs_required_from_caller=True, branches=diagnostics,
            rotation_parameter_count=sum(r.size for r in rotations.values()),
            rotation_state_bytes=sum(r.nbytes for r in rotations.values()),
            standardized_cosine_and_Euclidean_preserved=True, whitened_Mahalanobis_preserved=False))
    if len(classes) < PARAMETERS['min_qualified_classes']:
        return result
    weights = msp * (votes / 3.)
    rotation, diagnostic = _fit_branch('fused160', values['fused160'], fitted['fused160'],
                                      selected, pseudo, weights, classes)
    diagnostics['fused160'] = diagnostic
    if not diagnostic['fitted']:
        result['reason'] = 'insufficient_nondegenerate_fused_classes'
        return result
    rotations['fused160'] = rotation
    for name in source.DIMENSIONS:
        if name == 'fused160':
            continue
        rotations[name], diagnostics[name] = _fit_branch(name, values[name], fitted[name],
                                                        selected, pseudo, weights, classes)
    result.update(applied=True, reason='fitted_from_fixed_reliable_core')
    return result


def _state(state):
    if (not isinstance(state, dict) or state.get('schema') != SCHEMA or state.get('parameters') != PARAMETERS
            or type(state.get('applied')) is not bool or not isinstance(state.get('rotations'), dict)
            or set(state['rotations']) != set(source.DIMENSIONS)):
        raise ValueError('Matching fixed orthogonal alignment state is required')
    for name, dimension in source.DIMENSIONS.items():
        rotation = state['rotations'][name]
        identity = np.eye(dimension)
        if (not isinstance(rotation, np.ndarray) or rotation.shape != (dimension, dimension)
                or rotation.dtype != np.float64 or not np.isfinite(rotation).all()):
            raise ValueError('Finite FP64 full branch rotation required: ' + name)
        exact_identity = np.array_equal(rotation, identity)
        if not state['applied'] and not exact_identity:
            raise ValueError('Disabled alignment must retain exact identity rotations')
        if not exact_identity and np.max(np.abs(rotation.T @ rotation - identity)) > 1e-8:
            raise ValueError('Frozen branch rotation is not orthogonal')
        if name == 'reference29' and (not np.array_equal(rotation[27], identity[27])
                                      or not np.array_equal(rotation[:, 27], identity[:, 27])):
            raise ValueError('Frozen reference rotation must fix the zero CFO coordinate')
    return state


def apply_alignment(branches, state, geometry):
    """Apply frozen aggregate rotations to any row count; never fit on query."""
    values, fitted = source._branches(branches), source._geometry(geometry)
    _state(state)
    result = {}
    for name, dimension in source.DIMENSIONS.items():
        rotation = state['rotations'][name]
        if not state['applied'] or np.array_equal(rotation, np.eye(dimension)):
            result[name] = branches[name].copy()
            continue
        branch = fitted[name]
        standard = _standardized(values[name], branch)
        with np.errstate(over='ignore', invalid='ignore'):
            restored = (standard @ rotation) * branch['scale'] + branch['mean']
        if name == 'reference29':
            restored[:, 27] = 0.
        result[name] = source._fp32(restored, 'adapted ' + name)
    return result


def aggregate_state(state):
    """Copy only rotations and aggregate fit diagnostics for deployment storage."""
    _state(state)
    keys = ('schema', 'parameters', 'applied', 'reason', 'sample_count', 'pseudo_label_counts',
            'selected_class_counts', 'class_weights', 'rotations', 'diagnostics')
    return {key: deepcopy(state[key]) for key in keys}
