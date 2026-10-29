"""Pure NumPy source-only, class-conditioned empirical domain projections.

RX/day group-mean residuals are first made orthogonal to the six-class mean
difference span. Retained SVD directions are actually removed, then each row
is unit-normalized again. This preserves that source span before normalization,
not every class geometry after normalization and not unknown identity signals.
There is no data IO, query fit, neural update, or source-row deployment state.
"""
from __future__ import annotations

import numpy as np

from . import star_relative_core as relative

VARIANTS = ('control', 'rx', 'rxday')
SCHEMA = 'pamf_phase3_star_nuisance_metric_v1'
BRANCHES = relative.BRANCHES
METHODS = relative.METHODS
PRIMARY = relative.PRIMARY
ENERGY_FRACTION = .9
RANK_CAP = 16
_GAUSSIAN = ('class_means', 'background_mean', 'within_whitener', 'background_whitener')
_BRANCH_FIELDS = set(_GAUSSIAN) | {'projection'}
_DIAG_TYPES = dict(rank='int', between_class_rank='int', residual_rank='int', residual_group_count='int',
    residual_energy='float', captured_energy='float', energy_coverage='coverage', energy_target_met='bool',
    between_class_damage='float', between_class_change_after_unit='float', remaining_norm_min='float',
    remaining_norm_max='float', remaining_unit_norm_anomalies='int', input_zero_unit_count='int', reason='str')


def _metadata(y, rx, day, n, variant):
    if variant not in VARIANTS:
        raise ValueError('Only control, rx and rxday source variants are supported')
    values = tuple(np.asarray(x) for x in (y, rx, day))
    if n <= 6 or any(x.shape != (n,) or x.dtype.kind not in 'iu' for x in values):
        raise ValueError('Aligned integer source class/RX/day vectors with N>6 are required')
    y, rx, day = values
    if (set(y.tolist()) != set(range(6)) or not np.isin(rx, (1, 3, 4, 6, 8)).all()
            or not np.isin(day, (1, 2, 3)).all()):
        raise ValueError('Only the old six classes and source RX/day namespace may fit a projection')
    domains = [(r,) for r in sorted(set(rx.tolist()))] if variant == 'rx' else sorted(set(zip(rx.tolist(), day.tolist())))
    if variant != 'control':
        for label in range(6):
            for domain in domains:
                mask = (y == label) & (rx == domain[0])
                if variant == 'rxday':mask &= day == domain[1]
                if not np.any(mask):
                    raise ValueError('Every observed source domain needs all six old class groups')
    return y, rx, day, domains


def _canonical_rows(values):
    # Sorting source group aggregates makes the SVD insensitive to an arbitrary
    # permutation of the six class labels, including degenerate row orderings.
    if not len(values):return values
    return values[np.lexsort(tuple(values[:, j] for j in range(values.shape[1]-1, -1, -1)))]


def _between(means):
    ordered = _canonical_rows(means)
    centered = ordered - ordered.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * max(float(np.linalg.norm(ordered)), 1.)
    return vt[singular > tolerance].T


def _project_units(u, projection):
    if np.array_equal(projection, np.eye(u.shape[1])):
        return u.copy(), np.linalg.norm(u, axis=1)
    projected = u @ projection
    norms = np.linalg.norm(projected, axis=1)
    return projected / np.maximum(norms[:, None], 1e-12), norms


def _projection(u, labels, rx, day, domains, variant):
    dimension = u.shape[1]; identity = np.eye(dimension)
    means = np.asarray([u[labels == c].mean(axis=0) for c in range(6)])
    between = _between(means)
    projection = identity.copy(); residual_rank = rank = 0
    total = captured = 0.; coverage = None; reason = 'control'; group_count = 0
    if variant != 'control':
        rows = []
        for label in range(6):
            for domain in domains:
                mask = (labels == label) & (rx == domain[0])
                if variant == 'rxday':mask &= day == domain[1]
                rows.append(u[mask].mean(axis=0) - means[label])
        residual = _canonical_rows(np.asarray(rows, np.float64))
        group_count = len(residual)
        # Remove old mean-difference directions before deciding any nuisance rank.
        residual = residual - (residual @ between) @ between.T
        _, singular, vt = np.linalg.svd(residual, full_matrices=False)
        tolerance = max(residual.shape) * np.finfo(np.float64).eps * max(float(np.linalg.norm(means)), 1.)
        keep = singular > tolerance
        singular, vt = singular[keep], vt[keep]
        residual_rank = len(singular)
        if not residual_rank:
            reason = 'zero_residual_outside_between_class_span'
        else:
            energy = singular * singular; total = float(energy.sum())
            requested = int(np.searchsorted(np.cumsum(energy), ENERGY_FRACTION * total)) + 1
            rank = min(requested, RANK_CAP)
            basis = vt[:rank].T
            projection = identity - basis @ basis.T
            # Symmetry is explicit; this is a singular orthogonal projector,
            # not invertible attenuation followed by equivalent covariance fit.
            projection = (projection + projection.T) * .5
            captured = float(energy[:rank].sum()); coverage = captured / total
            reason = 'projected' if coverage >= ENERGY_FRACTION-1e-12 else 'projected_rank_cap'
    transformed, norms = _project_units(u, projection)
    centered = means - means.mean(axis=0)
    new_means = np.asarray([transformed[labels == c].mean(axis=0) for c in range(6)])
    new_centered = new_means - new_means.mean(axis=0)
    denominator = max(float(np.linalg.norm(centered)), 1e-12)
    diagnostics = dict(rank=int(rank), between_class_rank=int(between.shape[1]), residual_rank=int(residual_rank),
        residual_group_count=int(group_count), residual_energy=total, captured_energy=captured,
        energy_coverage=coverage, energy_target_met=coverage is not None and coverage >= ENERGY_FRACTION-1e-12,
        between_class_damage=float(np.linalg.norm(centered @ projection - centered) / denominator),
        between_class_change_after_unit=float(np.linalg.norm(new_centered - centered) / denominator),
        remaining_norm_min=float(norms.min()), remaining_norm_max=float(norms.max()),
        remaining_unit_norm_anomalies=int(np.sum(np.abs(np.linalg.norm(transformed, axis=1)-1.) > 1e-9)),
        input_zero_unit_count=int(np.sum(np.linalg.norm(u, axis=1) < 1e-12)), reason=reason)
    return projection, transformed, diagnostics


def _gaussian(units, labels):
    n = len(labels); fitted = {}
    for name in BRANCHES:
        u = units[name]
        means = np.asarray([u[labels == c].mean(axis=0) for c in range(6)], np.float64)
        background = u.mean(axis=0)
        fitted[name] = dict(class_means=means, background_mean=background,
            within_whitener=relative._whitener(u-means[labels], n-6),
            background_whitener=relative._whitener(u-background, n-1))
    raw = relative._raw_scores(units, fitted)
    return fitted, raw.mean(axis=0), np.maximum(raw.std(axis=0, ddof=0), 1e-6)


def fit_metric(branches, y, rx, day, geometry, variant):
    """Fit the fixed source-only variant, returning aggregate arrays/diagnostics."""
    units = relative._units(branches, geometry)
    labels, rx, day, domains = _metadata(y, rx, day, len(units['fused160']), variant)
    projections, transformed, diagnostics = {}, {}, {}
    for name in BRANCHES:
        projections[name], transformed[name], diagnostics[name] = _projection(units[name], labels, rx, day, domains, variant)
    if variant == 'control':
        old = relative.fit_metric(branches, labels, geometry)
        fitted, mean, scale = old['branches'], old['score_mean'], old['score_scale']
    else:
        fitted, mean, scale = _gaussian(transformed, labels)
    result = dict(schema=SCHEMA, variant=variant,
        branches={name: dict(fitted[name], projection=projections[name]) for name in BRANCHES},
        score_mean=mean, score_scale=scale, diagnostics=diagnostics)
    _metric(result, verify_projection=True)
    return result


def _diagnostics(value):
    if not isinstance(value, dict) or set(value) != set(BRANCHES):
        raise ValueError('Only five branch aggregate diagnostics are accepted')
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != set(_DIAG_TYPES):
            raise ValueError('Projection diagnostics contain missing fields or source rows')
        for key, kind in _DIAG_TYPES.items():
            x = row[key]
            if kind == 'int' and (type(x) is not int or x < 0):raise ValueError('Nonnegative diagnostic count required')
            if kind == 'bool' and type(x) is not bool:raise ValueError('Boolean diagnostic required')
            if kind == 'str' and (not isinstance(x, str) or not x):raise ValueError('Diagnostic reason required')
            if kind == 'float' and (not isinstance(x, (int,float)) or not np.isfinite(x) or x < 0):raise ValueError('Finite nonnegative diagnostic required')
            if kind == 'coverage' and x is not None and (not isinstance(x, (int,float)) or not np.isfinite(x) or x < 0 or x > 1+1e-12):raise ValueError('Residual energy coverage must be defined in [0,1] or None')
        if row['rank'] > RANK_CAP:raise ValueError('Projection rank exceeds the fixed cap')
    return value


def _metric(bundle, *, verify_projection=False):
    if (not isinstance(bundle, dict) or set(bundle) != {'schema','variant','branches','score_mean','score_scale','diagnostics'}
            or bundle['schema'] != SCHEMA or bundle['variant'] not in VARIANTS
            or not isinstance(bundle['branches'], dict) or set(bundle['branches']) != set(BRANCHES)):
        raise ValueError('The frozen source nuisance aggregate schema differs')
    gaussian, projections = {}, {}
    for name in BRANCHES:
        d = relative.reg.DIMENSIONS[name]; branch = bundle['branches'][name]
        if not isinstance(branch, dict) or set(branch) != _BRANCH_FIELDS:
            raise ValueError('Only projection and Gaussian aggregate arrays may be stored')
        p = relative._array(branch['projection'], (d,d), name+' projection')
        if not np.allclose(p,p.T,rtol=0,atol=1e-10):raise ValueError('Source projection must be symmetric')
        if verify_projection and not np.allclose(p@p,p,rtol=0,atol=2e-9):raise ValueError('Source projection must be genuinely idempotent')
        projections[name] = p
        gaussian[name] = {key: branch[key] for key in _GAUSSIAN}
    _, mean, scale = relative._metric(dict(schema=relative.SCHEMA, branches=gaussian,
        score_mean=bundle['score_mean'], score_scale=bundle['score_scale']))
    _diagnostics(bundle['diagnostics'])
    return gaussian, projections, mean, scale


def metric_scores(branches, geometry, bundle):
    """Return FP64[N,2] MD/RMD without fitting any row/query statistic."""
    units = relative._units(branches, geometry)
    gaussian, projections, mean, scale = _metric(bundle)
    transformed = {name: _project_units(units[name], projections[name])[0] for name in BRANCHES}
    result = ((relative._raw_scores(transformed, gaussian)-mean[None])/scale[None]).mean(axis=2)
    if not np.isfinite(result).all():raise ValueError('Nonfinite projected MD/RMD scores')
    return result


def all_scores(branches, logits, geometry, bundle):
    """Unchanged five original scores followed by projected MD/RMD."""
    values, fitted = relative.reg._branches(branches), relative.reg._geometry(geometry)
    raw = relative.reg._matrix(logits, 'original complete head logits', 6)
    if len(raw) != len(values['fused160']):raise ValueError('Branches and complete default-head logits must align')
    static = relative.scores.knownness_scores(values['fused160'], raw, fitted['fused160']['prototypes'])
    return np.concatenate((static, metric_scores(branches, geometry, bundle)), axis=1)


def to_arrays(bundle):
    """Flatten a validated aggregate to NumPy arrays; performs no file IO."""
    _metric(bundle, verify_projection=True)
    result = dict(schema=np.asarray(bundle['schema']), variant=np.asarray(bundle['variant']),
                  score_mean=bundle['score_mean'].copy(), score_scale=bundle['score_scale'].copy())
    for name, branch in bundle['branches'].items():
        for key, value in branch.items():result[name+'__'+key] = np.asarray(value, np.float64).copy()
        for key, kind in _DIAG_TYPES.items():
            value = bundle['diagnostics'][name][key]
            dtype = np.int64 if kind == 'int' else np.bool_ if kind == 'bool' else None if kind == 'str' else np.float64
            result['diag__'+name+'__'+key] = np.asarray(-1. if kind == 'coverage' and value is None else value, dtype=dtype)
    return result


def from_arrays(arrays):
    """Reconstruct only aggregates from an in-memory flat array mapping."""
    expected = {'schema','variant','score_mean','score_scale'} | {name+'__'+key for name in BRANCHES for key in _BRANCH_FIELDS} | {
        'diag__'+name+'__'+key for name in BRANCHES for key in _DIAG_TYPES}
    if not isinstance(arrays, dict) or set(arrays) != expected:raise ValueError('Flat aggregate fields differ or contain source rows')
    if any(np.asarray(arrays[k]).shape != () or np.asarray(arrays[k]).dtype.kind != 'U' for k in ('schema','variant')):
        raise ValueError('Scalar source schema/variant strings required')
    diagnostic = {}
    for name in BRANCHES:
        row = {}
        for key, kind in _DIAG_TYPES.items():
            a = np.asarray(arrays['diag__'+name+'__'+key])
            dtype = 'i' if kind == 'int' else 'b' if kind == 'bool' else 'U' if kind == 'str' else 'f'
            if a.shape != () or a.dtype.kind != dtype:raise ValueError('Flat aggregate diagnostic scalar dtype differs')
            value = a.item(); row[key] = None if kind == 'coverage' and value == -1. else value
        diagnostic[name] = row
    result = dict(schema=np.asarray(arrays['schema']).item(), variant=np.asarray(arrays['variant']).item(),
        branches={name: {key: np.asarray(arrays[name+'__'+key]).copy() for key in _BRANCH_FIELDS} for name in BRANCHES},
        score_mean=np.asarray(arrays['score_mean']).copy(), score_scale=np.asarray(arrays['score_scale']).copy(), diagnostics=diagnostic)
    _metric(result, verify_projection=True)
    return result
