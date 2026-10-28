"""Domain-balanced source OE readers over frozen head-aligned branch evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import source_open_run as parent
from . import source_consensus_run as previous
from . import source_consensus as consensus
from . import source_domain_oe_readout as domain_core
from .source_risk_run import _known_labels
from .source_risk_calibration import calibrate_known, METHODS


OE_COUNT = 6134
FEATURE_REF = 'exposure_train/public/features.npz'
REGIME = domain_core.REGIME
_score = previous._score


def _diverse_marker(root, identity):
    marker = parent._read(Path(root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_DIVERSE_OE_FEATURES_EXPORTED'
            or marker.get('source_only') is not True or marker.get('head_changed') is not False
            or any(marker.get(key) != 0 for key in ('target_reads', 'optimizer_updates', 'original_U_s_reads', 'OE_truth_reads'))
            or marker.get('model_classes') != 6 or marker.get('parameters') != 232993
            or marker.get('count') != OE_COUNT or marker.get('feature_ref') != FEATURE_REF
            or marker.get('parent_export_identity') != identity):
        raise ValueError('Complete matching frozen-F diverse source OE export required')
    handle = marker.get('source_OE_handle')
    if (not isinstance(handle, dict) or set(handle) != {'capsule_id', 'split_id', 'count'}
            or type(handle['count']) is not int or handle['count'] != OE_COUNT
            or any(not isinstance(handle[key], str) or not handle[key] for key in ('capsule_id', 'split_id'))):
        raise ValueError('Diverse source OE data handle differs')
    return handle


def _load_diverse_features(root):
    with np.load(Path(root) / FEATURE_REF, allow_pickle=False) as stored:
        if set(stored.files) != parent.COMMON:
            raise ValueError('Diverse source OE features must contain only the twelve public fields')
        arrays = {key: stored[key].copy() for key in parent.COMMON}
    ids, logits = arrays['ids'], arrays['logits']
    if (ids.shape != (OE_COUNT,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == '')
            or len(set(ids.tolist())) != OE_COUNT or logits.shape != (OE_COUNT, 6)
            or logits.dtype != np.float32 or not np.isfinite(logits).all()):
        raise ValueError('Diverse source OE ID/count/complete-head schema differs')
    for name, allowed in [('rx_i', [1, 3, 4, 6, 8]), ('day_i', [1, 2, 3])]:
        if arrays[name].shape != (OE_COUNT,) or arrays[name].dtype != np.int64 or not np.isin(arrays[name], allowed).all():
            raise ValueError('Diverse source OE Main domain metadata differs')
    parent.core._branches(parent._branches(arrays))
    if any(len(arrays['features__' + name]) != OE_COUNT for name in parent.DIMENSIONS):
        raise ValueError('Diverse source OE branch count differs')
    return arrays


def _save_new(output, root, state):
    with (root / 'fit/geometry.npz').open('rb') as source, (output / 'geometry.npz').open('xb') as target:
        shutil.copyfileobj(source, target)
    arrays = dict(schema=np.asarray(state['schema']), descriptor_schema=np.asarray(consensus.DESCRIPTOR_SCHEMA),
        readout_training_regime=np.asarray(REGIME), descriptor_dim=np.asarray(100, dtype=np.int64),
        method_names=np.asarray(parent.core.METHODS), mean=state['mean'], scale=state['scale'])
    for name, model in state['models'].items(): arrays.update({name + '__' + key: value for key, value in model.items()})
    parent._npz(output / 'readouts.npz', arrays)


def _load_new(root):
    root = Path(root)
    expected = {'schema', 'class_count'} | {key + '__' + name for name in parent.DIMENSIONS for key in ('mean', 'scale', 'prototypes', 'whitener')}
    with np.load(root / 'geometry.npz', allow_pickle=False) as data:
        if set(data.files) != expected or data['schema'].shape != () or data['class_count'].shape != () or data['class_count'].dtype != np.int64:
            raise ValueError('Inherited source geometry artifact differs')
        geometry = dict(schema=data['schema'].item(), class_count=int(data['class_count'].item()),
            branches={name: {key: data[key + '__' + name].copy() for key in ('mean', 'scale', 'prototypes', 'whitener')} for name in parent.DIMENSIONS})
    expected = {'schema', 'descriptor_schema', 'readout_training_regime', 'descriptor_dim', 'method_names', 'mean', 'scale'} | {name + '__' + key for name, keys in parent.WEIGHT_KEYS.items() for key in keys}
    with np.load(root / 'readouts.npz', allow_pickle=False) as data:
        if (set(data.files) != expected or any(data[key].shape != () or data[key].dtype.kind != 'U'
                for key in ('descriptor_schema', 'readout_training_regime'))
                or data['descriptor_schema'].item() != consensus.DESCRIPTOR_SCHEMA or data['readout_training_regime'].item() != REGIME
                or data['descriptor_dim'].shape != () or data['descriptor_dim'].dtype != np.int64
                or data['method_names'].dtype.kind != 'U' or data['method_names'].tolist() != list(parent.core.METHODS)):
            raise ValueError('Both the head-aligned descriptor and domain-balanced training regime must match')
        state = dict(schema=data['schema'].item(), descriptor_dim=int(data['descriptor_dim'].item()),
            method_names=data['method_names'].tolist(), mean=data['mean'].copy(), scale=data['scale'].copy(),
            models={name: {key: data[name + '__' + key].copy() for key in keys} for name, keys in parent.WEIGHT_KEYS.items()}, training={})
    parent.core._geometry(geometry); parent.core._readout_state(state)
    return geometry, state


def fit(parent_run_root, export_root, model_seed, output):
    output = parent._new(output)
    root, identity, geometry, old_readouts = previous._parent_state(parent_run_root)
    handle = _diverse_marker(export_root, identity)
    if type(model_seed) is not int or model_seed != 2026092701 or model_seed != identity['model']['model_seed']:
        raise ValueError('The fixed parent/readout seed2026092701 is required')
    output.mkdir(parents=True, exist_ok=False)
    known = parent._load_features(root / 'export', 'source_L', True)
    two = {key: value[:2] for key, value in known.items()}
    smoke = parent._score(two, geometry, old_readouts)
    desc = consensus.descriptor(parent._branches(two), two['logits'], geometry)
    if smoke.shape != (2, 7) or desc.shape != (2, 100) or not np.isfinite(desc).all():
        raise ValueError('Real frozen parent readout/source-L descriptor smoke failed')
    parent._write(output / 'smoke.json', dict(status='PASS', actual_frozen_parent_readouts=True,
        source_L_smoke_samples=2, parent_knownness_shape=[2, 7], descriptor_shape=[2, 100],
        descriptor_schema=consensus.DESCRIPTOR_SCHEMA, readout_training_regime=REGIME,
        encoder_forwards=0, source_V_reads=0, novel_validation_reads=0, target_reads=0))
    print('SOURCE_DIVERSE_PARENT_SMOKE_PASS', 2, 100, flush=True)
    oe = _load_diverse_features(export_root)
    if set(known['ids'].tolist()) & set(oe['ids'].tolist()): raise ValueError('Known and diverse source training IDs overlap')
    kd = domain_core.domains_from_metadata(known['rx_i'], known['day_i'])
    od = domain_core.domains_from_metadata(oe['rx_i'], oe['day_i'])
    started = time.perf_counter()
    known_desc = consensus.descriptor(parent._branches(known), known['logits'], geometry)
    oe_desc = consensus.descriptor(parent._branches(oe), oe['logits'], geometry)
    print('SOURCE_DIVERSE_FIT_START', len(known_desc), len(oe_desc), 200, 'cpu', flush=True)
    state = domain_core.fit_readouts(known_desc, oe_desc, kd, od, model_seed)
    training = state['training']
    if (training.get('regime') != REGIME or training.get('steps') != 200
            or training.get('known_count') != parent.COUNTS['source_L'] or training.get('oe_count') != OE_COUNT
            or training.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or training.get('normalizer_role_prior') != {'known': .6, 'oe': .4}
            or training.get('loss_role_prior') != {'known': .5, 'oe': .5} or training.get('domain_count') != 15
            or any(len(trace) != 200 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('The fixed role/domain weighting and 200-step readout budget must complete')
    _save_new(output, root, state)
    loaded_geometry, loaded_state = _load_new(output)
    if not np.isfinite(_score(two, loaded_geometry, loaded_state)).all(): raise ValueError('Serialized readout source-L forward failed')
    parent._write(output / 'training.json', dict(training=training, descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        readout_training_regime=REGIME, seconds=time.perf_counter() - started, source_only=True, target_reads=0,
        geometry_fitted=False, encoder_updates=0, identity_head_updates=0, source_validation_reads=0, novel_validation_reads=0,
        source_OE_membership_is_additional_supervision=True, domain_metadata_used_for_deployed_scores=False))
    result = dict(status='SOURCE_DIVERSE_READOUTS_FROZEN', descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        readout_training_regime=REGIME, source_only=True, target_reads=0, encoder_updates=0, identity_head_updates=0,
        readout_steps_per_model=200, source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0,
        method_names=list(METHODS), learned_method_names=list(parent.core.METHODS),
        parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633}, model_seed=model_seed, geometry_fitted=False,
        parent_export_identity=identity, source_OE_handle=handle, parent_run_root=str(root), diverse_export_root=str(export_root),
        source_L_count=parent.COUNTS['source_L'], source_OE_count=OE_COUNT,
        geometry_ref='geometry.npz', readouts_ref='readouts.npz', training_ref='training.json',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(output / 'complete.json', result)
    print('SOURCE_DIVERSE_READOUTS_FROZEN', model_seed, 200, flush=True)
    return result


def _new_fit(root, identity, handle):
    marker = parent._read(Path(root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_DIVERSE_READOUTS_FROZEN'
            or marker.get('descriptor_schema') != consensus.DESCRIPTOR_SCHEMA or marker.get('readout_training_regime') != REGIME
            or marker.get('parent_export_identity') != identity or marker.get('source_OE_handle') != handle
            or marker.get('source_only') is not True or marker.get('geometry_fitted') is not False
            or marker.get('model_seed') != 2026092701 or marker.get('readout_steps_per_model') != 200
            or marker.get('source_L_count') != parent.COUNTS['source_L'] or marker.get('source_OE_count') != OE_COUNT
            or any(marker.get(key) != 0 for key in ('target_reads', 'encoder_updates', 'identity_head_updates',
                'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads'))
            or marker.get('method_names') != list(METHODS)
            or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}):
        raise ValueError('A matching complete diverse/domain-balanced fit must precede validation')


def predict(parent_run_root, export_root, fit_root, output):
    output = parent._new(output)
    root, identity, _, _ = previous._parent_state(parent_run_root)
    handle = _diverse_marker(export_root, identity)
    _new_fit(fit_root, identity, handle)
    geometry, state = _load_new(fit_root)
    known = parent._load_features(root / 'export', 'source_V', False)
    knownness = _score(known, geometry, state); closed_known = known['logits'].argmax(axis=1).astype(np.int64)
    labels = _known_labels(root / 'export', known)
    risk = calibrate_known(known['ids'], labels, closed_known, knownness, known['rx_i']); thresholds = risk['thresholds']
    output.mkdir(parents=True, exist_ok=False); calibration = output / 'calibration'; calibration.mkdir()
    with (calibration / 'thresholds.npy').open('xb') as stream: np.save(stream, thresholds, allow_pickle=False)
    summary = {key: value for key, value in risk.items() if key != 'thresholds'}
    summary.update(status='SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        readout_training_regime=REGIME, source_only=True, target_reads=0, source_V_labels_read=True,
        source_known_label_calibration=True, unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
        novel_validation_reads=0, encoder_updates=0, identity_head_updates=0, readout_updates=0,
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(calibration / 'complete.json', summary)
    print('SOURCE_DIVERSE_RISK_THRESHOLDS_FROZEN', len(known['ids']), .04, .015, flush=True)
    novel = parent._load_features(root / 'export', 'novel_validation', False)
    if set(known['ids'].tolist()) & set(novel['ids'].tolist()): raise ValueError('Known/novel source IDs overlap')
    novelness = _score(novel, geometry, state)
    ids = np.r_[known['ids'], novel['ids']]; values = np.concatenate((knownness, novelness))
    closed = np.r_[closed_known, novel['logits'].argmax(axis=1)].astype(np.int64)
    accepted = values >= thresholds; order = np.argsort(ids, kind='stable')
    parent._npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=closed[order], knownness=values[order], accepted=accepted[order],
        prediction=np.where(accepted, closed[:, None], -1).astype(np.int64)[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']][order], day_i=np.r_[known['day_i'], novel['day_i']][order]))
    result = dict(status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        readout_training_regime=REGIME, source_only=True, target_reads=0, truth_read=False,
        truth_read_scope='unknown/private truth only; known source labels used for fixed risk calibration',
        query_fit=False, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
        known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids), method_names=list(METHODS),
        encoder_updates=0, identity_head_updates=0, readout_updates=0, OE_truth_reads=0,
        limits=risk['limits'], constraints_passed=risk['constraints_passed'], thresholds_ref='calibration/thresholds.npy',
        predictions_ref='predictions.npz', parent_export_identity=identity, source_OE_handle=handle,
        parent_run_root=str(root), diverse_export_root=str(export_root), fit_root=str(fit_root),
        domain_metadata_used_for_deployed_scores=False, independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='same-known-V empirical risk and repeated source novel evaluation; not independent confirmation')
    parent._write(output / 'complete.json', result)
    print('ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', len(ids), 7, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    for action in ('fit', 'predict'):
        command = commands.add_parser(action); command.add_argument('--parent-run-root', required=True)
        command.add_argument('--export-root', required=True); command.add_argument('--output', required=True)
        if action == 'fit': command.add_argument('--model-seed', type=int, default=2026092701)
        else: command.add_argument('--fit-root', required=True)
    args = parser.parse_args(argv)
    result = fit(args.parent_run_root, args.export_root, args.model_seed, args.output) if args.action == 'fit' else predict(args.parent_run_root, args.export_root, args.fit_root, args.output)
    print(json.dumps(dict(status=result['status'], readout_training_regime=REGIME))); return 0


if __name__ == '__main__': raise SystemExit(main())
