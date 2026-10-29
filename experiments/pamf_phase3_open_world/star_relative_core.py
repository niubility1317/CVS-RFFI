"""Frozen source-only multibranch MD/RMD aggregates for G10 F0 inputs.

This is the project's five-branch relative metric variant. It does not inherit
RF performance claims from a reference paper, fit an identity head, discard
directions, read data, or optimize any model. The standard eight-branch
geometry remains a separate unchanged source_open_readout geometry bundle.
"""
from __future__ import annotations

import numpy as np

from . import scores
from . import unlabeled_registration_core as reg


RUN = '20261029-phase3-star-relative-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_relative_metric_v1'
BRANCHES = reg.EMBEDDING_BRANCHES
METHODS = scores.METHODS + ('mahalanobis', 'relative_mahalanobis')
PRIMARY = 'relative_mahalanobis'
PRIMARY_INDEX = 6
_BRANCH_FIELDS = {'class_means', 'background_mean', 'within_whitener', 'background_whitener'}


def _units(branches, geometry):
    values, fitted = reg._branches(branches), reg._geometry(geometry)
    return {key: reg._unit((values[key] - fitted[key]['mean']) / fitted[key]['scale']) for key in BRANCHES}


def _whitener(residual, denominator):
    covariance = residual.T @ residual / denominator
    dimension = covariance.shape[0]
    trace_scale = max(float(np.trace(covariance) / dimension), 1e-8)
    regularized = .8 * covariance + .2 * trace_scale * np.eye(dimension)
    try:
        result = np.linalg.solve(np.linalg.cholesky(regularized), np.eye(dimension))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Source shrinkage covariance must have a finite positive-definite factor') from exc
    if not np.isfinite(result).all(): raise ValueError('Nonfinite fitted source whitener')
    return result


def _branch_scores(coordinates, branch):
    """Raw MD/RMD of fixed standardized-unit coordinates; no re-unitization.

    Center before the matrix multiply, then subtract whitened class means.
    This avoids squared-distance cancellation and an N x 6 x d row bank.
    """
    x = np.asarray(coordinates, np.float64)
    centered = x - branch['background_mean']
    white = centered @ branch['within_whitener'].T
    centers = (branch['class_means'] - branch['background_mean']) @ branch['within_whitener'].T
    distance = np.column_stack([np.einsum('nd,nd->n', white - c, white - c) for c in centers])
    minimum = distance.min(axis=1)
    background = centered @ branch['background_whitener'].T
    background_distance = np.einsum('nd,nd->n', background, background)
    result = np.column_stack((-minimum, background_distance - minimum))
    if not np.isfinite(result).all(): raise ValueError('Nonfinite raw MD/RMD scores')
    return result


def _raw_scores(units, fitted):
    return np.stack([_branch_scores(units[name], fitted[name]) for name in BRANCHES], axis=2)


def fit_metric(branches, y, geometry):
    """Fit on current path's source L only; retain aggregate Gaussian arrays.

    Actual means of unit vectors are not normalized again. Within covariance
    has N-6 degrees of freedom, background covariance N-1. Both retain all
    coordinates with fixed .2 isotropic shrinkage. No source rows/y are saved.
    """
    units = _units(branches, geometry)
    n = len(units['fused160']); labels = np.asarray(y)
    if (n <= 6 or labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or set(np.unique(labels).tolist()) != set(range(6))):
        raise ValueError('Visible source L labels must contain all six old classes with N>6')
    fitted = {}
    for name in BRANCHES:
        u = units[name]
        means = np.asarray([u[labels == label].mean(axis=0) for label in range(6)], np.float64)
        background = u.mean(axis=0)
        fitted[name] = dict(class_means=means, background_mean=background,
            within_whitener=_whitener(u - means[labels], n - 6),
            background_whitener=_whitener(u - background, n - 1))
    raw = _raw_scores(units, fitted)
    result = dict(schema=SCHEMA, branches=fitted, score_mean=raw.mean(axis=0),
                  score_scale=np.maximum(raw.std(axis=0, ddof=0), 1e-6))
    _metric(result)
    return result


def _array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError('Finite aggregate shape/dtype differs: ' + name)
    return a.astype(np.float64, copy=False)


def _metric(metric):
    if (not isinstance(metric, dict) or set(metric) != {'schema', 'branches', 'score_mean', 'score_scale'}
            or metric['schema'] != SCHEMA or not isinstance(metric['branches'], dict)
            or set(metric['branches']) != set(BRANCHES)):
        raise ValueError('The five-branch frozen relative metric aggregate schema is required')
    fitted = {}
    for name in BRANCHES:
        branch = metric['branches'][name]; d = reg.DIMENSIONS[name]
        if not isinstance(branch, dict) or set(branch) != _BRANCH_FIELDS:
            raise ValueError('Only actual class/background means and two full whiteners are allowed')
        fitted[name] = dict(class_means=_array(branch['class_means'], (6, d), name + ' class means'),
            background_mean=_array(branch['background_mean'], (d,), name + ' background mean'),
            within_whitener=_array(branch['within_whitener'], (d, d), name + ' within whitener'),
            background_whitener=_array(branch['background_whitener'], (d, d), name + ' background whitener'))
    mean = _array(metric['score_mean'], (2, 5), 'source L raw score mean')
    scale = _array(metric['score_scale'], (2, 5), 'source L raw score scale')
    if np.any(scale < 1e-6): raise ValueError('Frozen source L score scales must retain the 1e-6 floor')
    return fitted, mean, scale


def metric_scores(branches, geometry, metric):
    """FP64[N,2]: branch-standardized MD/RMD, equal mean over five branches."""
    units = _units(branches, geometry)
    fitted, mean, scale = _metric(metric)
    result = ((_raw_scores(units, fitted) - mean[None]) / scale[None]).mean(axis=2)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite normalized MD/RMD scores')
    return result


def all_scores(branches, logits, geometry, metric):
    """One immutable seven-score entry for source calibration and target Q."""
    values, fitted = reg._branches(branches), reg._geometry(geometry)
    raw = reg._matrix(logits, 'full original F0 six-class logits', 6)
    if len(raw) != len(values['fused160']): raise ValueError('F0 logits and branch rows must align')
    static = scores.knownness_scores(values['fused160'], raw, fitted['fused160']['prototypes'])
    return np.concatenate((static, metric_scores(branches, geometry, metric)), axis=1)


def fit_bias_single(branches, logits, geometry):
    """Reuse fixed core rules with one F0; this is not independent dual-F agreement."""
    bias, diagnostics = reg.fit_bias(branches, logits, logits, geometry)
    diagnostics = dict(diagnostics, single_F0_core=True, independent_dual_F_agreement=False,
                       core_encoder='Phase1_original_F0_only', head_agreement_basis='same_F0_logits_passed_twice_for_API_reuse')
    return bias, diagnostics
