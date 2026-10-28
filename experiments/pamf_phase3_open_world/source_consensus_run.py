"""Head-aligned source OE readouts with inherited geometry and fixed risk limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import source_open_run as parent
from . import source_consensus as consensus
from .source_risk_run import PARENT_RUN, _known_labels
from .source_risk_calibration import calibrate_known, METHODS


def _parent_state(root):
    root = Path(root)
    if root.name != PARENT_RUN:
        raise ValueError('The original full-six-class source-open parent is required')
    marker = parent._export_marker(root / 'export')
    identity = parent._identity(marker)
    parent._frozen_fit(root / 'fit', identity)
    geometry, readouts = parent._load_aggregates(root / 'fit')
    return root, identity, geometry, readouts


def _save_new(output, root, state):
    with (root / 'fit/geometry.npz').open('rb') as source, (output / 'geometry.npz').open('xb') as target:
        shutil.copyfileobj(source, target)
    arrays = dict(schema=np.asarray(state['schema']), descriptor_schema=np.asarray(consensus.DESCRIPTOR_SCHEMA),
        descriptor_dim=np.asarray(100, dtype=np.int64), method_names=np.asarray(parent.core.METHODS),
        mean=state['mean'], scale=state['scale'])
    for name, model in state['models'].items():
        arrays.update({name + '__' + key: value for key, value in model.items()})
    parent._npz(output / 'readouts.npz', arrays)


def _load_new(root):
    root = Path(root)
    expected = {'schema', 'class_count'} | {key + '__' + name for name in parent.DIMENSIONS for key in ('mean', 'scale', 'prototypes', 'whitener')}
    with np.load(root / 'geometry.npz', allow_pickle=False) as data:
        if set(data.files) != expected or data['schema'].shape != () or data['class_count'].shape != () or data['class_count'].dtype != np.int64:
            raise ValueError('Inherited source geometry artifact differs')
        geometry = dict(schema=data['schema'].item(), class_count=int(data['class_count'].item()),
            branches={name: {key: data[key + '__' + name].copy() for key in ('mean', 'scale', 'prototypes', 'whitener')} for name in parent.DIMENSIONS})
    expected = {'schema', 'descriptor_schema', 'descriptor_dim', 'method_names', 'mean', 'scale'} | {name + '__' + key for name, keys in parent.WEIGHT_KEYS.items() for key in keys}
    with np.load(root / 'readouts.npz', allow_pickle=False) as data:
        if (set(data.files) != expected or data['descriptor_schema'].shape != ()
                or data['descriptor_schema'].dtype.kind != 'U' or data['descriptor_schema'].item() != consensus.DESCRIPTOR_SCHEMA
                or data['descriptor_dim'].shape != () or data['descriptor_dim'].dtype != np.int64
                or data['method_names'].dtype.kind != 'U' or data['method_names'].tolist() != list(parent.core.METHODS)):
            raise ValueError('Head-aligned readout descriptor identity differs; old descriptors cannot be reused')
        state = dict(schema=data['schema'].item(), descriptor_dim=int(data['descriptor_dim'].item()),
            method_names=data['method_names'].tolist(), mean=data['mean'].copy(), scale=data['scale'].copy(),
            models={name: {key: data[name + '__' + key].copy() for key in keys} for name, keys in parent.WEIGHT_KEYS.items()}, training={})
    parent.core._geometry(geometry); parent.core._readout_state(state)
    return geometry, state


def _score(data, geometry, state):
    branches = parent._branches(data)
    controls = parent.scores.knownness_scores(branches['fused160'], data['logits'], geometry['branches']['fused160']['prototypes'])
    learned = parent.core.score_readouts(consensus.descriptor(branches, data['logits'], geometry), state)
    result = np.concatenate((controls, learned), axis=1)
    if result.shape != (len(data['ids']), 7) or not np.isfinite(result).all():
        raise ValueError('Seven finite frozen head-aligned source scores are required')
    return result


def fit(parent_run_root, model_seed, output):
    output = parent._new(output)
    root, identity, geometry, old_readouts = _parent_state(parent_run_root)
    if type(model_seed) is not int or model_seed != 2026092701 or model_seed != identity['model']['model_seed']:
        raise ValueError('The fixed parent/readout seed2026092701 is required')
    output.mkdir(parents=True, exist_ok=False)
    known = parent._load_features(root / 'export', 'source_L', True)
    two = {key: value[:2] for key, value in known.items()}
    old_scores = parent._score(two, geometry, old_readouts)
    new_descriptor = consensus.descriptor(parent._branches(two), two['logits'], geometry)
    if old_scores.shape != (2, 7) or new_descriptor.shape != (2, 100) or not np.isfinite(new_descriptor).all():
        raise ValueError('Real parent readout checkpoint and new descriptor source-L smoke failed')
    parent._write(output / 'smoke.json', dict(status='PASS', actual_frozen_parent_readouts=True,
        source_L_smoke_samples=2, parent_knownness_shape=[2, 7], descriptor_shape=[2, 100],
        descriptor_schema=consensus.DESCRIPTOR_SCHEMA, encoder_forwards=0, source_V_reads=0,
        novel_validation_reads=0, exposure_train_reads=0, target_reads=0,
        encoder_updates=0, identity_head_updates=0, new_readout_updates=0))
    print('SOURCE_CONSENSUS_PARENT_SMOKE_PASS', 2, 100, flush=True)
    oe = parent._load_features(root / 'export', 'exposure_train', False)
    if set(known['ids'].tolist()) & set(oe['ids'].tolist()):
        raise ValueError('Known/source OE training roles overlap')
    started = time.perf_counter()
    known_desc = consensus.descriptor(parent._branches(known), known['logits'], geometry)
    oe_desc = consensus.descriptor(parent._branches(oe), oe['logits'], geometry)
    print('SOURCE_CONSENSUS_FIT_START', len(known_desc), len(oe_desc), 200, 'cpu', flush=True)
    state = parent.core.fit_readouts(known_desc, oe_desc, model_seed)
    training = state['training']
    if (training.get('steps') != 200 or training.get('known_count') != parent.COUNTS['source_L']
            or training.get('oe_count') != parent.COUNTS['exposure_train']
            or training.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or any(len(trace) != 200 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('The unchanged two-readout 200-update budget must complete')
    _save_new(output, root, state)
    # Basic same-fit serialization readback; no additional source or validation role.
    loaded_geometry, loaded_state = _load_new(output)
    if not np.isfinite(_score(two, loaded_geometry, loaded_state)).all():
        raise ValueError('Serialized new readout source-L forward is nonfinite')
    parent._write(output / 'training.json', dict(training=training, descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        seconds=time.perf_counter() - started, source_only=True, target_reads=0,
        geometry_fitted=False, geometry_source='parent L-only aggregate; unchanged byte copy',
        encoder_updates=0, identity_head_updates=0, source_validation_reads=0, novel_validation_reads=0,
        source_OE_membership_is_additional_supervision=True))
    result = dict(status='SOURCE_CONSENSUS_READOUTS_FROZEN', descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        source_only=True, target_reads=0, encoder_updates=0, identity_head_updates=0,
        readout_steps_per_model=200, source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0,
        method_names=list(METHODS), learned_method_names=list(parent.core.METHODS),
        parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633}, model_seed=model_seed,
        geometry_fitted=False, export_identity=identity, parent_run_root=str(root),
        geometry_ref='geometry.npz', readouts_ref='readouts.npz', training_ref='training.json',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(output / 'complete.json', result)
    print('SOURCE_CONSENSUS_READOUTS_FROZEN', model_seed, 200, flush=True)
    return result


def _new_fit(root, identity):
    marker = parent._read(Path(root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_CONSENSUS_READOUTS_FROZEN'
            or marker.get('descriptor_schema') != consensus.DESCRIPTOR_SCHEMA or marker.get('export_identity') != identity
            or marker.get('source_only') is not True or marker.get('geometry_fitted') is not False
            or marker.get('model_seed') != 2026092701 or marker.get('readout_steps_per_model') != 200
            or any(marker.get(key) != 0 for key in ('target_reads', 'encoder_updates', 'identity_head_updates',
                'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads'))
            or marker.get('method_names') != list(METHODS)
            or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}):
        raise ValueError('A complete matching head-aligned fit must precede validation')


def predict(parent_run_root, fit_root, output):
    output = parent._new(output)
    root, identity, _, _ = _parent_state(parent_run_root)
    _new_fit(fit_root, identity)
    geometry, state = _load_new(fit_root)
    known = parent._load_features(root / 'export', 'source_V', False)
    knownness = _score(known, geometry, state)
    closed_known = known['logits'].argmax(axis=1).astype(np.int64)
    labels = _known_labels(root / 'export', known)
    risk = calibrate_known(known['ids'], labels, closed_known, knownness, known['rx_i'])
    thresholds = risk['thresholds']
    output.mkdir(parents=True, exist_ok=False)
    calibration = output / 'calibration'; calibration.mkdir()
    with (calibration / 'thresholds.npy').open('xb') as stream: np.save(stream, thresholds, allow_pickle=False)
    summary = {key: value for key, value in risk.items() if key != 'thresholds'}
    summary.update(status='SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', descriptor_schema=consensus.DESCRIPTOR_SCHEMA,
        source_only=True, target_reads=0, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0, novel_validation_reads=0,
        encoder_updates=0, identity_head_updates=0, readout_updates=0, thresholds_ref='thresholds.npy',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(calibration / 'complete.json', summary)
    print('SOURCE_CONSENSUS_RISK_THRESHOLDS_FROZEN', len(known['ids']), .04, .015, flush=True)
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
        source_only=True, target_reads=0, truth_read=False,
        truth_read_scope='unknown/private truth only; source known labels are used for risk calibration',
        query_fit=False, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
        known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids), method_names=list(METHODS),
        encoder_updates=0, identity_head_updates=0, readout_updates=0, OE_truth_reads=0,
        limits=risk['limits'], constraints_passed=risk['constraints_passed'],
        thresholds_ref='calibration/thresholds.npy', predictions_ref='predictions.npz',
        export_identity=identity, parent_run_root=str(root), fit_root=str(fit_root),
        independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='same-known-V empirical risk and repeated source novel evaluation; not independent confirmation')
    parent._write(output / 'complete.json', result)
    print('ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', len(ids), 7, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    for action in ('fit', 'predict'):
        command = commands.add_parser(action); command.add_argument('--parent-run-root', required=True)
        command.add_argument('--output', required=True)
        if action == 'fit': command.add_argument('--model-seed', type=int, default=2026092701)
        else: command.add_argument('--fit-root', required=True)
    args = parser.parse_args(argv)
    result = fit(args.parent_run_root, args.model_seed, args.output) if args.action == 'fit' else predict(args.parent_run_root, args.fit_root, args.output)
    print(json.dumps(dict(status=result['status'], descriptor_schema=consensus.DESCRIPTOR_SCHEMA))); return 0


if __name__ == '__main__': raise SystemExit(main())
