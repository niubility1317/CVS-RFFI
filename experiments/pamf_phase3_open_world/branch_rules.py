"""Fixed source-only branch evidence rules; no target data or target feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scores import fit_prototypes
from .scoring import group_metrics
from .source_geometry import eligible


BRANCHES = (
    ('fused160', 160), ('time160', 160), ('frequency160', 160), ('pa160', 160),
    ('reference29', 29), ('signature336', 336), ('spectrum128', 128), ('pa_moments48', 48),
)
METHODS = tuple(f'{name}::{kind}' for name, _ in BRANCHES
                for kind in ('cosine', 'mahalanobis_s020'))
COMMON_FIELDS = frozenset({
    'train_y', 'train_ids', 'validation_logits', 'validation_y',
    'validation_ids', 'validation_receivers',
})
FEATURE_FIELDS = COMMON_FIELDS | frozenset(
    f'{role}_features__{name}' for name, _ in BRANCHES
    for role in ('train', 'validation'))
STATISTICS = ('location', 'scale', 'prototypes', 'whitener')


def _matrix(value, name):
    raw = np.asarray(value)
    if (raw.dtype.kind != 'f' or raw.ndim != 2 or not len(raw)
            or not raw.shape[1] or not np.isfinite(raw).all()):
        raise ValueError('Nonempty finite floating matrix required: ' + name)
    return raw.astype(np.float64, copy=False)


def _unit(value):
    with np.errstate(over='ignore', invalid='ignore'):
        norm = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError('Feature normalization is not finite')
    return value / np.maximum(norm, 1e-12)


def fit_branch(train_features, train_labels, *, standardize):
    """Fit global scaling, four class centers and one shared covariance on L."""
    if type(standardize) is not bool:
        raise ValueError('Branch standardization policy must be fixed boolean')
    raw = _matrix(train_features, 'source training branch')
    y = np.asarray(train_labels)
    if (y.shape != (len(raw),) or y.dtype.kind not in 'iu'
            or set(y.tolist()) != set(range(4))):
        raise ValueError('Training labels must contain exactly the four known source classes')
    dimension = raw.shape[1]
    if standardize:
        with np.errstate(over='ignore', invalid='ignore'):
            location = raw.mean(axis=0)
            scale = raw.std(axis=0, ddof=0)
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise ValueError('Source location/scale estimation is not finite')
        scale[scale <= 1e-8] = 1.
    else:
        location = np.zeros(dimension, dtype=np.float64)
        scale = np.ones(dimension, dtype=np.float64)
    processed = (raw - location) / scale
    if not np.isfinite(processed).all():
        raise ValueError('Standardized training features are not finite')
    prototypes = fit_prototypes(processed, y, 4)
    unit = _unit(processed)
    residual = unit - prototypes[y]
    covariance = residual.T @ residual / max(len(unit) - 4, 1)
    if not np.isfinite(covariance).all():
        raise ValueError('Source within-class covariance is not finite')
    scale_covariance = max(float(np.trace(covariance) / dimension), 1e-8)
    eye = np.eye(dimension)
    regularized = .8 * covariance + .2 * scale_covariance * eye
    try:
        lower = np.linalg.cholesky(regularized)
        whitener = np.linalg.solve(lower, eye)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Source shrinkage covariance is not positive definite') from exc
    if not np.isfinite(whitener).all():
        raise ValueError('Source covariance whitener is not finite')
    return dict(location=location, scale=scale, prototypes=prototypes, whitener=whitener)


def branch_scores(features, bundle):
    """Return row-local cosine / negative Mahalanobis scores; high means known."""
    raw = _matrix(features, 'source evaluation branch')
    dimension = raw.shape[1]
    if not isinstance(bundle, dict) or set(bundle) != set(STATISTICS):
        raise ValueError('Branch bundle must contain only aggregate statistics')
    location, scale, prototypes, whitener = (np.asarray(bundle[key]) for key in STATISTICS)
    if (location.shape != (dimension,) or scale.shape != (dimension,)
            or prototypes.shape != (4, dimension) or whitener.shape != (dimension, dimension)
            or any(value.dtype.kind != 'f' or not np.isfinite(value).all()
                   for value in (location, scale, prototypes, whitener))
            or np.any(scale <= 0)):
        raise ValueError('Branch aggregate dimensions, finite values or scale are invalid')
    processed = (raw - location) / scale
    if not np.isfinite(processed).all():
        raise ValueError('Standardized evaluation features are not finite')
    unit = _unit(processed)
    cosine = np.clip((unit @ prototypes.T).max(axis=1), -1., 1.)
    whitened = unit @ whitener.T
    centers = prototypes @ whitener.T
    distances = ((whitened * whitened).sum(axis=1)[:, None]
                 + (centers * centers).sum(axis=1)[None, :]
                 - 2. * whitened @ centers.T)
    result = np.column_stack((cosine, -np.maximum(distances.min(axis=1), 0.)))
    if not np.isfinite(result).all():
        raise ValueError('Branch knownness scores are not finite')
    return result


def calibrate_known(scores, labels):
    values = _matrix(scores, 'source V branch scores')
    y = np.asarray(labels)
    if (values.shape[1] != len(METHODS) or y.shape != (len(values),)
            or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y > 5)):
        raise ValueError('Source V branch labels or score dimensions differ')
    known = y < 4
    if not known.any():
        raise ValueError('Source V known calibration is empty')
    return np.quantile(values[known], .01, axis=0, method='lower')


def _marker(path, heldout_receiver):
    if type(heldout_receiver) is not int or heldout_receiver not in (1, 3, 4, 6, 8):
        raise ValueError('An integer source held-out receiver is required')
    try:
        marker = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError('Completed source branch export is required before features are opened') from exc
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_BRANCHES_EXPORTED'
            or marker.get('source_only') is not True
            or type(marker.get('target_reads')) is not int or marker['target_reads'] != 0
            or type(marker.get('optimizer_updates')) is not int or marker['optimizer_updates'] != 0
            or marker.get('head_changed') is not False
            or marker.get('parent_epochs') != 20 or marker.get('parent_updates') != 540
            or type(marker.get('heldout_source_rx')) is not int
            or marker['heldout_source_rx'] != heldout_receiver
            or any(type(marker.get(key)) is not int or marker[key] <= 0
                   for key in ('train_count', 'validation_count'))):
        raise ValueError('Source-only fixed-parent branch export is incomplete or incompatible')
    return marker


def _arrays(path, marker):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != FEATURE_FIELDS:
            raise ValueError('Source branch NPZ must contain the exact 22 fields')
        arrays = {key: stored[key].copy() for key in FEATURE_FIELDS}
    nt, nv = marker['train_count'], marker['validation_count']
    for key, count, classes in (('train_y', nt, 4), ('validation_y', nv, 6)):
        y = arrays[key]
        if (y.shape != (count,) or y.dtype.kind not in 'iu'
                or set(y.tolist()) != set(range(classes))):
            raise ValueError('Complete integer source class label coverage required: ' + key)
    for key, count in (('train_ids', nt), ('validation_ids', nv)):
        ids = arrays[key]
        if (ids.shape != (count,) or ids.dtype.kind != 'U' or np.any(ids == '')
                or len(set(ids.tolist())) != count):
            raise ValueError('Unique nonempty physical source IDs required: ' + key)
    if set(arrays['train_ids'].tolist()) & set(arrays['validation_ids'].tolist()):
        raise ValueError('Training L and validation V physical source IDs overlap')
    rx = arrays['validation_receivers']
    if (rx.shape != (nv,) or rx.dtype.kind not in 'iu'
            or not np.isin(rx, [1, 3, 4, 6, 8]).all()):
        raise ValueError('Only the original source receivers are allowed')
    if not (rx == marker['heldout_source_rx']).any():
        raise ValueError('The held-out source receiver has no validation observations')
    logits = arrays['validation_logits']
    if logits.dtype != np.dtype('float32') or logits.shape != (nv, 4) or not np.isfinite(logits).all():
        raise ValueError('Finite FP32 complete-head four-class validation logits required')
    for name, dimension in BRANCHES:
        for role, count in (('train', nt), ('validation', nv)):
            values = arrays[f'{role}_features__{name}']
            if (values.dtype != np.dtype('float32') or values.shape != (count, dimension)
                    or not np.isfinite(values).all()):
                raise ValueError('Finite raw FP32 source branch shape required: ' + name)
            if name == 'reference29' and np.any(values[:, 27] != 0):
                raise ValueError('Reference coordinate27 must preserve the NoExplicitCFO zero')
    return arrays


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def evaluate(features_path, output_root, heldout_receiver=8):
    features, output = Path(features_path), Path(output_root)
    if output.exists():
        raise FileExistsError('Source branch rule output must not exist')
    marker = _marker(features.parent / 'complete.json', heldout_receiver)
    arrays = _arrays(features, marker)
    values = np.empty((marker['validation_count'], len(METHODS)), dtype=np.float64)
    aggregate = {}
    for i, (name, _) in enumerate(BRANCHES):
        bundle = fit_branch(arrays['train_features__' + name], arrays['train_y'],
                            standardize=name != 'fused160')
        values[:, 2 * i:2 * i + 2] = branch_scores(arrays['validation_features__' + name], bundle)
        for key in STATISTICS:
            aggregate[key + '__' + name] = bundle[key]
    y = arrays['validation_y']
    rx = arrays['validation_receivers']
    thresholds = calibrate_known(values, y)
    closed = arrays['validation_logits'].argmax(axis=1).astype(np.int64)
    held = rx == heldout_receiver
    methods = {}
    for i, name in enumerate(METHODS):
        accepted = values[:, i] >= thresholds[i]
        overall = group_metrics(y, closed, values[:, i], accepted, 4)
        heldout = group_metrics(y[held], closed[held], values[held, i], accepted[held], 4)
        per_receiver = {str(r): group_metrics(y[rx == r], closed[rx == r],
            values[rx == r, i], accepted[rx == r], 4) for r in np.unique(rx)}
        methods[name] = dict(overall=overall, by_receiver=per_receiver,
            heldout_receiver=heldout, threshold=float(thresholds[i]),
            eligible=eligible(overall, heldout))
    allowed = [name for name in METHODS if methods[name]['eligible']]
    next_step = 'SOURCE_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if allowed else 'NO_PROMOTION'
    result = dict(status='SOURCE_BRANCH_RULES_EVALUATED', source_only=True,
        evidence_scope='source_proxy_only_not_target', target_reads=0,
        target_results_used=False, optimizer_updates=0, head_changed=False,
        source_unknown_used_for_network_training=False,
        statistics_role='selected_source_L_only',
        standardization='fused160_identity_other7_train_global_ddof0_then_L2',
        covariance='shared_class_residual_shrinkage0.2_trace_over_d_floor1e-8',
        threshold_role='source_V_known_only_99pct_retention_quantile_lower',
        closed_class_rule='same_complete_validation_logits_argmax_for_all16',
        parent_epochs=marker['parent_epochs'], parent_updates=marker['parent_updates'],
        train_count=marker['train_count'], validation_count=marker['validation_count'],
        heldout_receiver=heldout_receiver, method_order=list(METHODS), methods=methods,
        eligible_methods=allowed, next_step=next_step, final_six_class_method_selected=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'bundle.npz').open('xb') as stream:
        np.savez(stream, **aggregate, thresholds=thresholds, method_names=np.asarray(METHODS))
    _write(output / 'results.json', result)
    _write(output / 'complete.json', dict(status=result['status'], source_only=True,
        target_reads=0, optimizer_updates=0, head_changed=False, eligible_methods=allowed,
        next_step=next_step, final_six_class_method_selected=False,
        evidence_scope=result['evidence_scope']))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heldout-receiver', type=int, default=8)
    args = parser.parse_args(argv)
    result = evaluate(args.features, args.output, args.heldout_receiver)
    print(json.dumps({key: result[key] for key in ('status', 'eligible_methods', 'next_step')}))


if __name__ == '__main__':
    main()
