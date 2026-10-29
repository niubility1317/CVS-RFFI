"""Four-view source-only binary readouts and known-only global risk calibration.

The two arms share the immutable clean-L geometry and frozen E51/F0/DURA.
Only augmented_readout fits parameters, using L/OE membership; V calibrates
thresholds and never trains a readout. Novel descriptors are prediction-only,
after both arms' thresholds are frozen. Private novel TX truth is scorer-only.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import source_open_readout as core
from . import source_embedding_readout_run as serialization
from .source_consensus import DESCRIPTOR_SCHEMA
from .source_risk_calibration import METHODS, RECEIVERS, _macro
from .source_baseline_risk import _metrics as _risk_metrics
from .source_open_run import _read, _write, _new, _npz


RUN_ID = '20261029-phase3-source-practical-readout-s2026092701-r01'
ARMS = ('calibration_only', 'augmented_readout')
VIEWS = ('clean', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
COUNTS = dict(source_L=6300, source_V=27000, exposure_train=4200, novel_validation=3150)
EXPORT_SCHEMA = 'pamf_phase3_source_practical_export_v1'
EXPORT_STATUS = 'SOURCE_PRACTICAL_DESCRIPTORS_EXPORTED'
FIT_SCHEMA = 'pamf_phase3_source_practical_readout_fit_v1'
CALIBRATION_SCHEMA = 'pamf_phase3_source_practical_risk_v1'
PREDICTION_SCHEMA = 'pamf_phase3_source_practical_predictions_v1'
FIT_STATUS = 'SOURCE_PRACTICAL_READOUTS_FROZEN'
CALIBRATION_STATUS = 'SOURCE_PRACTICAL_THRESHOLDS_FROZEN'
PREDICTION_STATUS = 'ALL_SOURCE_PRACTICAL_PREDICTIONS_COMPLETE'
PRIMARY = 'augmented_readout::mlp16_oe'
FRR = Fraction(1, 20)
TOTAL_LOSS = Fraction(1, 50)


def _export_marker(root):
    from .source_practical_export import export_marker
    return export_marker(root)


def _load_role_view(root, role, view, include_y=False):
    from .source_practical_export import load_role_view
    return load_role_view(root, role, view, include_y=include_y)


def _old_source(root):
    marker = _read(Path(root) / 'complete.json')
    if (marker.get('status') != 'SOURCE_EMBEDDING_READOUTS_FROZEN'
            or marker.get('arm') != 'uniform_oe'
            or marker.get('descriptor_schema') != DESCRIPTOR_SCHEMA
            or marker.get('readout_training_regime') != serialization.REGIME
            or marker.get('geometry_fitted') is not True
            or marker.get('source_only') is not True
            or marker.get('readout_steps_per_model') != 200
            or marker.get('method_names') != list(METHODS)
            or any(marker.get(k) != 0 for k in ('target_reads', 'new_encoder_updates',
                'new_identity_head_updates', 'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads'))):
        raise ValueError('The fixed E51 clean-L geometry and original source readouts are required')
    geometry, state = serialization.serialization._load_new(root)
    return marker, geometry, state


def _identity(marker):
    return {k: marker[k] for k in ('encoder_state_identity', 'parent_export_identity', 'descriptor_schema', 'view_order')}


def _arrays(value):
    if isinstance(value, dict):
        return {k: _arrays(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _unchanged(value, before):
    if isinstance(value, dict):
        return set(value) == set(before) and all(_unchanged(value[k], before[k]) for k in value)
    if isinstance(value, np.ndarray):
        return np.array_equal(value, before)
    return value == before


def _role(data, role, view):
    keys = {'ids', 'descriptor', 'static_scores', 'logits', 'baseline_closed', 'rx_i', 'day_i'}
    if role in ('source_L', 'source_V'):
        keys.add('y')
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError('Only compact declared role fields are accepted; unknown TX/y are forbidden')
    n = COUNTS[role]
    ids = data['ids']
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == '')
            or len(set(ids.tolist())) != n):
        raise ValueError('Unique compact physical IDs/count required: ' + role)
    for key, shape, dtype in (('descriptor', (n, 100), np.float32),
            ('static_scores', (n, 5), np.float64), ('logits', (n, 6), np.float32)):
        x = data[key]
        if x.shape != shape or x.dtype != dtype or not np.isfinite(x).all():
            raise ValueError('Finite compact shape/dtype required: ' + key)
    for key in ('rx_i', 'day_i', 'baseline_closed'):
        if data[key].shape != (n,) or data[key].dtype != np.int64:
            raise ValueError('Aligned compact integer metadata required: ' + key)
    if set(data['rx_i'].tolist()) != set(RECEIVERS):
        raise ValueError('Only the complete original source five RX are allowed')
    baseline = data['baseline_closed']
    if role in ('source_L', 'exposure_train'):
        if not np.all(baseline == -2):
            raise ValueError('No baseline identity forwards are used for readout training roles')
    elif np.any((baseline < 0) | (baseline >= 6)):
        raise ValueError('Same-input original F six-class baseline required')
    if 'y' in keys:
        y = data['y']
        if y.shape != (n,) or y.dtype != np.int64 or set(y.tolist()) != set(range(6)):
            raise ValueError('Only the original six known source labels are allowed')
    return data


def _role_views(root, role):
    rows = {view: _role(_load_role_view(root, role, view,
        include_y=role in ('source_L', 'source_V')), role, view) for view in VIEWS}
    first = rows[VIEWS[0]]
    for row in rows.values():
        for key in ('ids', 'rx_i', 'day_i') + (('y',) if 'y' in first else ()):
            if not np.array_equal(row[key], first[key]):
                raise ValueError('Four views must preserve identical physical ID order and metadata')
    return rows


def score_compact(descriptor, static_scores, state):
    """Row-local [N,100]+[N,5] -> [N,7], with immutable source readout state."""
    d, static = np.asarray(descriptor), np.asarray(static_scores)
    if (d.ndim != 2 or d.shape[1] != 100 or not len(d) or d.dtype != np.float32
            or not np.isfinite(d).all() or static.shape != (len(d), 5)
            or static.dtype.kind != 'f' or not np.isfinite(static).all()):
        raise ValueError('Finite FP32 compact descriptors and five aligned static scores required')
    result = np.concatenate((static, core.score_readouts(d, state)), axis=1).astype(np.float64)
    if result.shape != (len(d), 7) or not np.isfinite(result).all():
        raise ValueError('Seven finite fixed-order knownness scores required')
    return result


def fit(export_root, source_fit_root, output, model_seed=2026092701):
    """Fit only L/OE four-view descriptors; copy the control state exactly."""
    output = _new(output)
    if type(model_seed) is not int or model_seed != 2026092701:
        raise ValueError('The predeclared single readout seed is fixed')
    marker = _export_marker(export_root)
    old, geometry, control = _old_source(source_fit_root)
    if (old.get('encoder_state_identity') != marker['encoder_state_identity']
            or old.get('parent_export_identity') != marker['parent_export_identity']
            or ('geometry_source_fit_root' in marker and
                Path(marker['geometry_source_fit_root']).resolve() != Path(source_fit_root).resolve())):
        raise ValueError('Source encoder, baseline and immutable clean-L geometry must match the export')
    before_geometry, before_control = _arrays(geometry), _arrays(control)
    known = _role_views(export_root, 'source_L')
    oe = _role_views(export_root, 'exposure_train')
    if set(known[VIEWS[0]]['ids'].tolist()) & set(oe[VIEWS[0]]['ids'].tolist()):
        raise ValueError('Source L and OE training physical records overlap')
    a = np.concatenate([known[v]['descriptor'] for v in VIEWS])
    b = np.concatenate([oe[v]['descriptor'] for v in VIEWS])
    started = time.perf_counter()
    augmented = core.fit_readouts(a, b, model_seed)
    training = augmented.get('training', {})
    if (training.get('steps') != 200 or training.get('known_count') != 4 * COUNTS['source_L']
            or training.get('oe_count') != 4 * COUNTS['exposure_train']
            or training.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or set(training.get('losses', {})) != set(core.METHODS)
            or any(len(x) != 200 or not np.isfinite(x).all() for x in training['losses'].values())):
        raise ValueError('Both fixed 200-step CPU readouts must complete on L/OE only')
    if not _unchanged(geometry, before_geometry) or not _unchanged(control, before_control):
        raise ValueError('The original geometry/control readout state changed')
    output.mkdir(parents=True, exist_ok=False)
    identity = _identity(marker)
    identity.update(source_fit_root=str(source_fit_root), geometry_source='immutable original clean source_L',
        dura_identity=marker.get('dura_identity'),run_id=RUN_ID,export_identity=_identity(marker))
    for arm, state in zip(ARMS, (control, augmented)):
        root = output / arm
        root.mkdir()
        core._geometry(geometry); core._readout_state(state)
        serialization._save(root, geometry, state)
        loaded_g, loaded_s = serialization.serialization._load_new(root)
        if (not _unchanged(geometry, loaded_g)
                or not np.array_equal(score_compact(a[:2], np.zeros((2, 5)), state),
                    score_compact(a[:2], np.zeros((2, 5)), loaded_s))):
            raise ValueError('Source aggregate/readout serialization roundtrip differs')
        _write(root / 'complete.json', dict(status=FIT_STATUS, schema=FIT_SCHEMA, arm=arm,
            identities=identity, descriptor_schema=DESCRIPTOR_SCHEMA, source_only=True, target_reads=0,
            method_names=list(METHODS), readout_training_regime='four_view_source_membership_v1',
            geometry_fitted=False, geometry_source='original clean L only; copied unchanged',
            new_readout_steps_per_model=0 if arm == ARMS[0] else 200,
            readout_parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633},
            readout_mean_std_source='original clean readout' if arm == ARMS[0] else 'L4views + OE4views only',
            source_L_training_count=0 if arm == ARMS[0] else len(a),
            source_OE_training_count=0 if arm == ARMS[0] else len(b),
            source_L_training_physical_count=0 if arm == ARMS[0] else COUNTS['source_L'],
            source_OE_training_physical_count=0 if arm == ARMS[0] else COUNTS['exposure_train'],
            training_count_scope='four derived view exposures, not additional physical records',
            source_V_fit_reads=0, novel_validation_reads=0, unknown_TX_labels_read=0,
            new_encoder_updates=0, new_identity_head_updates=0, new_G_updates=0,
            source_membership_supervision=True, geometry_ref='geometry.npz', readouts_ref='readouts.npz'))
    _write(output / 'training.json', dict(augmented_readout=training, seconds=time.perf_counter()-started,
        calibration_only_new_steps=0, source_V_fit_reads=0, novel_validation_reads=0,
        geometry_refitted=False, membership_loss='original equal-weight known/OE binary softplus',
        encoder_head_G_frozen=True))
    result = dict(status=FIT_STATUS, schema=FIT_SCHEMA, run_id=RUN_ID, completed_arms=list(ARMS),
        identities=identity, view_order=list(VIEWS), primary_candidate=PRIMARY, source_only=True,
        target_reads=0, source_V_fit_reads=0, novel_validation_reads=0,
        new_encoder_updates=0, new_identity_head_updates=0, new_G_updates=0)
    _write(output / 'complete.json', result)
    return result


def _fits(root):
    root = Path(root)
    marker = _read(root / 'complete.json')
    if (marker.get('status') != FIT_STATUS or marker.get('schema') != FIT_SCHEMA
            or marker.get('completed_arms') != list(ARMS) or marker.get('view_order') != list(VIEWS)
            or marker.get('source_only') is not True or marker.get('target_reads') != 0
            or marker.get('primary_candidate') != PRIMARY):
        raise ValueError('Both new-regime source-only readout arms must be frozen')
    rows = {}
    for arm in ARMS:
        row = _read(root / arm / 'complete.json')
        if (row.get('status') != FIT_STATUS or row.get('schema') != FIT_SCHEMA or row.get('arm') != arm
                or row.get('identities') != marker['identities'] or row.get('method_names') != list(METHODS)
                or row.get('geometry_fitted') is not False or row.get('source_only') is not True
                or row.get('new_readout_steps_per_model') != (0 if arm == ARMS[0] else 200)
                or any(row.get(k) != 0 for k in ('target_reads', 'source_V_fit_reads', 'novel_validation_reads',
                    'unknown_TX_labels_read', 'new_encoder_updates', 'new_identity_head_updates', 'new_G_updates'))):
            raise ValueError('Arm fit contract/identity differs: ' + arm)
        rows[arm] = row
    return marker, rows


def _calibration_inputs(ids, y, closed_new, closed_original, knownness, rx_i, view_i):
    ids, y, new, old, scores, rx, view = map(np.asarray,
        (ids, y, closed_new, closed_original, knownness, rx_i, view_i))
    n = len(ids) if ids.ndim else 0
    if not n or ids.shape != (n,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == ''):
        raise ValueError('Nonempty source-known IDs are required')
    for value in (y, new, old, rx, view):
        if value.shape != (n,) or value.dtype.kind not in 'iu':
            raise ValueError('Aligned integer known-only calibration metadata required')
    if (np.any((y < 0) | (y >= 6)) or np.any((new < 0) | (new >= 6))
            or np.any((old < 0) | (old >= 6)) or set(rx.tolist()) != set(RECEIVERS)
            or set(view.tolist()) != set(range(4))):
        raise ValueError('Calibration is restricted to old six classes, source RX and four views')
    if scores.shape != (n, 7) or scores.dtype.kind != 'f' or not np.isfinite(scores).all():
        raise ValueError('Seven finite source-known scores are required')
    reference = None
    for v in range(4):
        mask = view == v
        subset = ids[mask]
        if len(set(subset.tolist())) != len(subset):
            raise ValueError('Calibration view physical IDs must be unique')
        order = np.argsort(subset, kind='stable')
        aligned = (subset[order], y[mask][order], rx[mask][order])
        if reference is not None and any(not np.array_equal(a, b) for a, b in zip(reference, aligned)):
            raise ValueError('Known source physical IDs/labels/RX must be shared across all views')
        reference = aligned
        for r in RECEIVERS:
            if set(y[mask & (rx == r)].tolist()) != set(range(6)):
                raise ValueError('Every source RX-by-view cell must contain six known classes')
    return ids, y.astype(np.int64), new.astype(np.int64), old.astype(np.int64), scores.astype(np.float64), rx, view


def _group_masks(rx, view):
    yield 'overall', np.ones(len(rx), dtype=bool)
    for r in RECEIVERS:
        yield 'rx/' + str(r), rx == r
    for v, name in enumerate(VIEWS):
        yield 'view/' + name, view == v
    for r in RECEIVERS:
        for v, name in enumerate(VIEWS):
            yield 'rxview/' + str(r) + '/' + name, (rx == r) & (view == v)


def _local_maximum(y, new, old, scores):
    """O(N*6) temporary prefix memory for one group, never Nlevels*30*6."""
    order = np.argsort(scores, kind='stable')
    levels, sizes = np.unique(scores[order], return_counts=True)
    labels = y[order]
    block = np.repeat(np.arange(len(levels)), sizes)
    counts = np.zeros((len(levels), 6), dtype=np.int64)
    correct = np.zeros_like(counts)
    np.add.at(counts, (block, labels), 1)
    ok = new[order] == labels
    np.add.at(correct, (block[ok], labels[ok]), 1)
    prefix = np.vstack((np.zeros((1, 6), np.int64), np.cumsum(counts, axis=0)))
    correct_prefix = np.vstack((np.zeros((1, 6), np.int64), np.cumsum(correct, axis=0)))
    total, new_count = prefix[-1], correct_prefix[-1]
    old_count = np.bincount(y[old == y], minlength=6)
    fixed = _macro(old_count, total) - _macro(new_count, total)
    def feasible(i):
        return _macro(prefix[i], total) <= FRR and fixed + _macro(correct_prefix[i], total) <= TOTAL_LOSS
    left, right = 0, len(levels)-1
    no_rejection = feasible(0)
    if no_rejection:
        while left < right:
            mid = (left + right + 1)//2
            if feasible(mid): left = mid
            else: right = mid - 1
    return float(levels[left]), no_rejection


def _risk_group(y, new, old, scores, threshold):
    total = np.bincount(y, minlength=6)
    rejected = scores < threshold
    row = _risk_metrics(np.bincount(y[rejected], minlength=6),
        np.bincount(y[rejected & (new == y)], minlength=6), total,
        np.bincount(y[new == y], minlength=6), np.bincount(y[old == y], minlength=6))
    row['encoder_protection_diagnostic_passed'] = row.pop('encoder_protection_passed')
    return row


def calibrate_known(ids, y, closed_new, closed_original, knownness, rx_i, view_i):
    """One maximal global scalar/method, exact Fraction risks, intact ties."""
    _, y, new, old, values, rx, view = _calibration_inputs(
        ids, y, closed_new, closed_original, knownness, rx_i, view_i)
    masks = list(_group_masks(rx, view))
    methods = {}
    for j, name in enumerate(METHODS):
        score = values[:, j]
        local = {key: _local_maximum(y[mask], new[mask], old[mask], score[mask]) for key, mask in masks}
        zero_ok = all(row[1] for row in local.values())
        threshold = min(row[0] for row in local.values()) if zero_ok else float(score.min())
        groups = {key: _risk_group(y[mask], new[mask], old[mask], score[mask], threshold) for key, mask in masks}
        levels = np.unique(score)
        selected = int(np.searchsorted(levels, threshold))
        next_level = float(levels[selected+1]) if selected+1 < len(levels) else None
        if next_level is None:
            with np.errstate(over='ignore'):
                sentinel = float(np.nextafter(levels[-1], np.inf))
            if np.isfinite(sentinel): next_level = sentinel
        next_pass = None if next_level is None else all(_risk_group(
            y[mask], new[mask], old[mask], score[mask], next_level)['constraints_passed'] for _, mask in masks)
        passed = all(x['constraints_passed'] for x in groups.values())
        with np.errstate(over='ignore'):
            has_sentinel = bool(np.isfinite(np.nextafter(levels[-1], np.inf)))
        methods[name] = dict(threshold=threshold, selected_candidate_index=selected,
            candidate_count=len(levels)+int(has_sentinel),
            next_candidate_threshold=next_level, next_candidate_passed=next_pass,
            no_rejection_feasible=zero_ok, constraints_passed=passed,
            encoder_protection_diagnostic_passed=all(x['encoder_protection_diagnostic_passed'] for x in groups.values()),
            overall=groups['overall'], by_receiver={str(r): groups['rx/'+str(r)] for r in RECEIVERS},
            by_view={v: groups['view/'+v] for v in VIEWS},
            by_receiver_view={str(r): {v: groups['rxview/'+str(r)+'/'+v] for v in VIEWS} for r in RECEIVERS},
            local_maximum_thresholds={key: row[0] for key,row in local.items()})
    return dict(schema=CALIBRATION_SCHEMA, method_names=list(METHODS), known_count=len(y),
        original_physical_count=len(y)//4, class_count=6, receivers=list(RECEIVERS), view_order=list(VIEWS),
        thresholds=np.asarray([methods[n]['threshold'] for n in METHODS],np.float64), methods=methods,
        limits=dict(known_macro_frr=.05,original_closed_minus_new_open=.02),
        encoder_closed_loss_diagnostic_limit=.005,
        constraints_passed=all(x['constraints_passed'] for x in methods.values()),
        encoder_protection_diagnostic_passed=all(x['encoder_protection_diagnostic_passed'] for x in methods.values()),
        source_known_labels_used=True, unknown_inputs_used=False,
        acceptance_rule='knownness >= maximal legal global scalar; entire ties together',
        group_scope='overall + each RX + each view + each RX-by-view, equal known-class macro',
        risk_scope='empirical protection on calibration V itself, not independent target guarantee')


def calibrate(export_root, fit_root, output):
    output = _new(output)
    marker = _export_marker(export_root)
    fitted, _ = _fits(fit_root)
    if any(fitted['identities'][k] != v for k,v in _identity(marker).items()):
        raise ValueError('Calibration export and fitted source identities differ')
    known = _role_views(export_root, 'source_V')
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        geometry, state = serialization.serialization._load_new(Path(fit_root)/arm)
        before = _arrays((dict(geometry=geometry,state=state)))
        rows = [known[v] for v in VIEWS]
        risk = calibrate_known(np.concatenate([r['ids'] for r in rows]),
            np.concatenate([r['y'] for r in rows]),np.concatenate([r['logits'].argmax(axis=1) for r in rows]),
            np.concatenate([r['baseline_closed'] for r in rows]),
            np.concatenate([score_compact(r['descriptor'],r['static_scores'],state) for r in rows]),
            np.concatenate([r['rx_i'] for r in rows]),
            np.repeat(np.arange(4,dtype=np.int64),COUNTS['source_V']))
        if not _unchanged(dict(geometry=geometry,state=state),before):
            raise ValueError('Calibration modified frozen source aggregates/readouts')
        root = output/arm; root.mkdir()
        with (root/'thresholds.npy').open('xb') as stream:
            np.save(stream,risk.pop('thresholds'),allow_pickle=False)
        risk.update(status=CALIBRATION_STATUS,arm=arm,identities=fitted['identities'],source_only=True,
            target_reads=0,source_V_labels_read=True,source_known_label_calibration=True,
            unknown_calibration_scores_read=0,unknown_calibration_labels_read=0,novel_validation_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,new_readout_updates=0,
            thresholds_ref='thresholds.npy')
        _write(root/'complete.json',risk)
    result=dict(status=CALIBRATION_STATUS,schema=CALIBRATION_SCHEMA,completed_arms=list(ARMS),
        identities=fitted['identities'],view_order=list(VIEWS),method_names=list(METHODS),
        source_only=True,target_reads=0,novel_validation_reads=0,unknown_calibration_scores_read=0,
        unknown_calibration_labels_read=0,source_V_physical_count=COUNTS['source_V'],
        known_calibration_count=4*COUNTS['source_V'])
    _write(output/'complete.json',result)
    return result


def _calibrations(root, fitted):
    root=Path(root); marker=_read(root/'complete.json')
    if (marker.get('status')!=CALIBRATION_STATUS or marker.get('schema')!=CALIBRATION_SCHEMA
            or marker.get('completed_arms')!=list(ARMS) or marker.get('view_order')!=list(VIEWS)
            or marker.get('identities')!=fitted['identities'] or marker.get('method_names')!=list(METHODS)
            or marker.get('source_only') is not True
            or any(marker.get(k)!=0 for k in ('target_reads','novel_validation_reads',
                'unknown_calibration_scores_read','unknown_calibration_labels_read'))):
        raise ValueError('Both source-known-only arm thresholds must be frozen before novel/target input')
    rows={}
    for arm in ARMS:
        row=_read(root/arm/'complete.json')
        if (row.get('status')!=CALIBRATION_STATUS or row.get('schema')!=CALIBRATION_SCHEMA
                or row.get('arm')!=arm or row.get('identities')!=fitted['identities']
                or row.get('method_names')!=list(METHODS) or row.get('view_order')!=list(VIEWS)
                or row.get('known_count')!=4*COUNTS['source_V']
                or row.get('source_V_labels_read') is not True
                or any(row.get(k)!=0 for k in ('target_reads','novel_validation_reads','unknown_calibration_scores_read',
                    'unknown_calibration_labels_read','new_encoder_updates','new_identity_head_updates','new_readout_updates'))):
            raise ValueError('Known-only calibration arm identity/coverage differs')
        thresholds=np.load(root/arm/'thresholds.npy',allow_pickle=False)
        if thresholds.shape!=(7,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all():
            raise ValueError('Seven finite frozen global thresholds required')
        if any(thresholds[j]!=row['methods'][name]['threshold'] for j,name in enumerate(METHODS)):
            raise ValueError('Threshold artifact and known-only calibration differ')
        rows[arm]=(row,thresholds)
    return marker,rows


def load_bundle(fit_root, calibration_root, arm):
    """Deployment loader: aggregate files and markers only, no source row arrays."""
    if arm not in ARMS:raise ValueError('Unknown predeclared readout arm')
    fitted,rows=_fits(fit_root)
    calibrated,risks=_calibrations(calibration_root,fitted)
    geometry,state=serialization.serialization._load_new(Path(fit_root)/arm)
    return dict(geometry=geometry,state=state,thresholds=risks[arm][1],identities=dict(fitted['identities'],arm=arm),
        fit_marker=rows[arm],calibration_marker=risks[arm][0],arm=arm,
        method_names=list(METHODS),descriptor_schema=DESCRIPTOR_SCHEMA)


def predict_source(export_root, fit_root, calibration_root, output):
    output=_new(output)
    fitted,_=_fits(fit_root)
    _calibrations(calibration_root,fitted)  # Both arms freeze before ANY novel read.
    marker=_export_marker(export_root)
    if any(fitted['identities'][k]!=v for k,v in _identity(marker).items()):
        raise ValueError('Prediction export and frozen source state differ')
    known=_role_views(export_root,'source_V')
    novel=_role_views(export_root,'novel_validation')
    if set(known[VIEWS[0]]['ids'].tolist()) & set(novel[VIEWS[0]]['ids'].tolist()):
        raise ValueError('Known validation and novel physical IDs overlap')
    output.mkdir(parents=True,exist_ok=False)
    refs={}
    for arm in ARMS:
        bundle=load_bundle(fit_root,calibration_root,arm)
        before=_arrays(dict(geometry=bundle['geometry'],state=bundle['state'],thresholds=bundle['thresholds']))
        refs[arm]={}
        for view in VIEWS:
            k,n=known[view],novel[view]
            ids=np.r_[k['ids'],n['ids']]
            closed=np.r_[k['logits'].argmax(axis=1),n['logits'].argmax(axis=1)].astype(np.int64)
            scores=score_compact(np.concatenate((k['descriptor'],n['descriptor'])),
                np.concatenate((k['static_scores'],n['static_scores'])),bundle['state'])
            accepted=scores>=bundle['thresholds']
            root=output/arm/view;root.mkdir(parents=True)
            _npz(root/'predictions.npz',dict(ids=ids,method_names=np.asarray(METHODS),
                thresholds=bundle['thresholds'],closed_prediction=closed,
                baseline_closed_prediction=np.r_[k['baseline_closed'],n['baseline_closed']],
                knownness=scores,accepted=accepted,prediction=np.where(accepted,closed[:,None],-1).astype(np.int64),
                rx_i=np.r_[k['rx_i'],n['rx_i']],day_i=np.r_[k['day_i'],n['day_i']],
                known_y=np.r_[k['y'],np.full(len(n['ids']),-1,np.int64)]))
            refs[arm][view]=f'{arm}/{view}/predictions.npz'
        if not _unchanged(dict(geometry=bundle['geometry'],state=bundle['state'],thresholds=bundle['thresholds']),before):
            raise ValueError('Prediction modified the frozen source state or thresholds')
    result=dict(status=PREDICTION_STATUS,schema=PREDICTION_SCHEMA,run_id=RUN_ID,source_only=True,target_reads=0,
        completed_arms=list(ARMS),view_order=list(VIEWS),method_names=list(METHODS),primary_candidate=PRIMARY,
        identities=fitted['identities'],known_count=COUNTS['source_V'],novel_count=COUNTS['novel_validation'],
        count_per_view=COUNTS['source_V']+COUNTS['novel_validation'],prediction_refs=refs,
        truth_read=False,unknown_truth_read=False,query_fit=False,new_encoder_updates=0,
        new_identity_head_updates=0,new_readout_updates=0,new_threshold_calibrations=0,
        default_head_replaced=False,source_novel_evaluation_reused=True,independent_confirmation=False,
        final_phase3_complete=False,anonymous_registration_performed=False)
    _write(output/'complete.json',result)
    return result


def _predictions(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('status')!=PREDICTION_STATUS or marker.get('schema')!=PREDICTION_SCHEMA
            or marker.get('completed_arms')!=list(ARMS) or marker.get('view_order')!=list(VIEWS)
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_candidate')!=PRIMARY
            or marker.get('source_only') is not True or marker.get('target_reads')!=0
            or marker.get('known_count')!=COUNTS['source_V'] or marker.get('novel_count')!=COUNTS['novel_validation']
            or marker.get('truth_read') is not False or marker.get('unknown_truth_read') is not False
            or marker.get('query_fit') is not False):
        raise ValueError('Private truth closed: all two-arm/four-view predictions must be complete')
    expected={'ids','method_names','thresholds','closed_prediction','baseline_closed_prediction',
        'knownness','accepted','prediction','rx_i','day_i','known_y'}
    rows={};reference=None
    for arm in ARMS:
        rows[arm]={}
        for view in VIEWS:
            ref=f'{arm}/{view}/predictions.npz'
            if marker.get('prediction_refs',{}).get(arm,{}).get(view)!=ref:
                raise ValueError('Private truth closed: prediction view coverage differs')
            with np.load(root/ref,allow_pickle=False) as f:
                if set(f.files)!=expected:raise ValueError('Prediction schema differs')
                data={k:f[k].copy() for k in f.files}
            n=COUNTS['source_V']+COUNTS['novel_validation'];nk=COUNTS['source_V']
            ids=data['ids']
            if ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)==''):
                raise ValueError('Prediction physical ID/count mismatch')
            if data['method_names'].tolist()!=list(METHODS):raise ValueError('Prediction method order differs')
            for key in ('closed_prediction','baseline_closed_prediction'):
                a=data[key]
                if a.shape!=(n,) or a.dtype!=np.int64 or np.any((a<0)|(a>=6)):
                    raise ValueError('Complete default/original six-class predictions required')
            for key in ('rx_i','day_i','known_y'):
                if data[key].shape!=(n,) or data[key].dtype!=np.int64:raise ValueError('Prediction metadata differs')
            if (set(data['known_y'][:nk].tolist())!=set(range(6)) or not np.all(data['known_y'][nk:]==-1)
                    or set(data['rx_i'].tolist())!=set(RECEIVERS)):
                raise ValueError('Only source V known labels are public; novel labels remain hidden')
            t,s,a=data['thresholds'],data['knownness'],data['accepted']
            if (t.shape!=(7,) or t.dtype!=np.float64 or not np.isfinite(t).all()
                    or s.shape!=(n,7) or s.dtype.kind!='f' or not np.isfinite(s).all()
                    or a.shape!=(n,7) or a.dtype.kind!='b' or not np.array_equal(a,s>=t)
                    or not np.array_equal(data['prediction'],np.where(a,data['closed_prediction'][:,None],-1))):
                raise ValueError('Frozen thresholds and complete accepted predictions disagree')
            metadata={key:data[key] for key in ('ids','rx_i','day_i','known_y')}
            if reference is not None and not _unchanged(metadata,reference):
                raise ValueError('All arms/views must share exact source IDs and metadata')
            reference=metadata
            if view!=VIEWS[0] and not np.array_equal(t,rows[arm][VIEWS[0]]['thresholds']):
                raise ValueError('Each arm/method must use a single global threshold across views')
            if arm==ARMS[1]:
                for key in ('closed_prediction','baseline_closed_prediction'):
                    if not np.array_equal(data[key],rows[ARMS[0]][view][key]):
                        raise ValueError('Both readout arms must retain the same F/F0 predictions')
            rows[arm][view]=data
    return marker,rows


def _novel_truth(path, expected_ids):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'ids','tx'}:raise ValueError('Only independent source novel IDs/TX truth is allowed')
        ids,tx=f['ids'].copy(),f['tx'].copy()
    if (ids.shape!=expected_ids.shape or ids.dtype.kind!='U' or tx.shape!=ids.shape or tx.dtype.kind!='U'
            or len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(expected_ids.tolist())
            or np.any(np.char.strip(tx)=='')):
        raise ValueError('Independent novel source truth exact physical-ID coverage differs')
    lookup=dict(zip(ids.tolist(),tx.tolist()))
    return np.asarray([lookup[x] for x in expected_ids.tolist()])


def _goals(row, unknown_min):
    checks=dict(unknown_rejection=row['unknown_macro_rejection'] is not None and row['unknown_macro_rejection']>=unknown_min,
        known_acceptance=row['known_macro_acceptance'] is not None and row['known_macro_acceptance']>=.95-1e-12,
        known_FRR=row['known_macro_frr'] is not None and row['known_macro_frr']<=.05+1e-12,
        total_original_closed_open_loss=row['original_closed_minus_new_open'] is not None and
            row['original_closed_minus_new_open']<=.02+1e-12)
    return dict(checks=checks,passed=all(checks.values()),unknown_rejection_min=unknown_min,
        encoder_half_point_is_diagnostic_only=True)


def _known_goals(row):
    checks = _goals(row, 0.)['checks']
    checks.pop('unknown_rejection')
    return dict(checks=checks,passed=all(checks.values()),encoder_half_point_is_diagnostic_only=True)


def score_source(predictions_root, novel_truth, output):
    output=_new(output)
    marker,pred=_predictions(predictions_root)  # All arrays/arms/views before truth IO.
    first=pred[ARMS[0]][VIEWS[0]];nk=COUNTS['source_V']
    tx=_novel_truth(novel_truth,first['ids'][nk:])
    tx_names=sorted(set(tx.tolist()));mapping={t:6+j for j,t in enumerate(tx_names)}
    y=np.r_[first['known_y'][:nk],np.asarray([mapping[t] for t in tx.tolist()],np.int64)]
    known_tx=['14-10','14-7','20-15','20-19','6-15','8-20']
    label_to_tx={i:t for i,t in enumerate(known_tx)}|{v:k for k,v in mapping.items()}
    from .development_transfer import _metrics
    arms={}
    for arm in ARMS:
        methods={}
        for j,name in enumerate(METHODS):
            def metrics(data,mask):
                return _metrics(y[mask],data['closed_prediction'][mask],data['baseline_closed_prediction'][mask],
                    data['knownness'][mask,j],data['accepted'][mask,j])
            by_view={};by_rx_view={str(r):{} for r in RECEIVERS};by_tx_view={t:{} for t in label_to_tx.values()}
            for view in VIEWS:
                d=pred[arm][view];by_view[view]=metrics(d,np.ones(len(y),bool))
                for r in RECEIVERS:by_rx_view[str(r)][view]=metrics(d,d['rx_i']==r)
                for label,t in label_to_tx.items():by_tx_view[t][view]=metrics(d,y==label)
            joined={key:np.concatenate([pred[arm][v][key] for v in VIEWS])
                for key in ('closed_prediction','baseline_closed_prediction','knownness','accepted','rx_i')}
            yy=np.tile(y,4)
            def pooled(mask):
                row=_metrics(yy[mask],joined['closed_prediction'][mask],joined['baseline_closed_prediction'][mask],
                    joined['knownness'][mask,j],joined['accepted'][mask,j])
                # These four views share physical records. Keep descriptive
                # counts/ranking, never a binomial CI with 4x independent N.
                row['confidence_intervals95']=None
                for key in ('known_per_class','unknown_per_class'):
                    for record in row[key]:record['rejection_interval95']=None
                row['pooled_views_CI_not_reported']=True
                row['count_scope']='derived condition exposures; physical records repeat across views'
                return row
            overall=pooled(np.ones(len(yy),bool))
            by_rx={str(r):pooled(joined['rx_i']==r) for r in RECEIVERS}
            by_tx={t:pooled(yy==label) for label,t in label_to_tx.items()}
            view_goals={v:_goals(z,.95 if v=='clean' else .90) for v,z in by_view.items()}
            cross_goals={r:{v:_goals(z,.95 if v=='clean' else .90) for v,z in vv.items()} for r,vv in by_rx_view.items()}
            overall_known=_known_goals(overall)
            receiver_known={r:_known_goals(z) for r,z in by_rx.items()}
            passed=(overall_known['passed'] and all(g['passed'] for g in receiver_known.values())
                and all(g['passed'] for g in view_goals.values())
                and all(g['passed'] for vv in cross_goals.values() for g in vv.values()))
            methods[name]=dict(overall=overall,by_view=by_view,by_receiver=by_rx,by_transmitter=by_tx,
                by_receiver_view=by_rx_view,by_transmitter_view=by_tx_view,
                goals=dict(overall_known=overall_known,by_receiver_known=receiver_known,
                    by_view=view_goals,by_receiver_view=cross_goals,passed=passed),
                diagnostic_only=arm!=ARMS[1] or name!='mlp16_oe')
        arms[arm]=dict(methods=methods,primary_method='mlp16_oe',
            primary_source_goals=methods['mlp16_oe']['goals'],diagnostic_only=arm==ARMS[0])
    result=dict(status='SCORED_SOURCE_PRACTICAL_COMPLETE',schema='pamf_phase3_source_practical_scores_v1',
        run_id=RUN_ID,source_only=True,target_reads=0,truth_last=True,unknown_truth_read=True,
        primary_candidate=PRIMARY,method_names=list(METHODS),view_order=list(VIEWS),arms=arms,
        identities=marker['identities'],known_count_per_view=nk,novel_count_per_view=len(tx),
        unique_physical_validation_count=len(y),derived_view_count=4*len(y),
        novel_label_to_transmitter={str(v):k for k,v in mapping.items()},
        independent_confirmation=False,source_novel_evaluation_reused=True,
        target_feedback_forbidden=True,source_scores_do_not_select_target_arm=True,
        final_phase3_complete=False,domain_adaptation_performed=False,anonymous_registration_performed=False,
        next_step='SOURCE_SCORED_NO_TARGET_CLAIM',
        evidence_scope='same-source RX; repeated source novel; empirical known-V calibration; not cross-RX confirmation',
        metric_definitions=dict(known_macro='equal six-known-class mean; acceptance does not imply identity correctness',
            unknown_macro='equal private source novel TX mean',
            total_damage='same-input original F closed macro minus current open macro',
            encoder_half_point='diagnostic only, not a hard success goal',
            views='same physical records in four derived conditions, not independent samples'))
    output.mkdir(parents=True,exist_ok=False)
    _write(output/'scores.json',result)
    _write(output/'complete.json',dict(status=result['status'],schema=result['schema'],truth_last=True,
        primary_candidate=PRIMARY,source_only=True,target_reads=0,final_phase3_complete=False,
        primary_source_goals={a:arms[a]['primary_source_goals'] for a in ARMS},
        next_step=result['next_step']))
    return result


def main(argv=None):
    actual=list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(dict(event='SOURCE_PRACTICAL_READOUT_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=[sys.argv[0]]+actual)),flush=True)
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    for action in ('fit','calibrate','predict-source','score-source'):
        command=commands.add_parser(action);command.add_argument('--output',required=True)
        if action!='score-source':command.add_argument('--export-root',required=True)
        if action=='fit':
            command.add_argument('--source-fit-root',required=True);command.add_argument('--model-seed',type=int,default=2026092701)
        if action in ('calibrate','predict-source'):command.add_argument('--fit-root',required=True)
        if action=='predict-source':command.add_argument('--calibration-root',required=True)
        if action=='score-source':
            command.add_argument('--predictions-root',required=True);command.add_argument('--novel-truth',required=True)
    args=parser.parse_args(actual)
    if args.action=='fit':result=fit(args.export_root,args.source_fit_root,args.output,args.model_seed)
    elif args.action=='calibrate':result=calibrate(args.export_root,args.fit_root,args.output)
    elif args.action=='predict-source':result=predict_source(args.export_root,args.fit_root,args.calibration_root,args.output)
    else:result=score_source(args.predictions_root,args.novel_truth,args.output)
    print(json.dumps(dict(status=result['status'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
