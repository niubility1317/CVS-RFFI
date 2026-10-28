"""L-only geometry and original unweighted consensus readouts on frozen E51."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import source_open_run as parent, source_consensus as consensus
from . import source_consensus_run as serialization, source_embedding_eval as embedding
from . import source_embedding_readout_export as exporter
from .source_risk_calibration import METHODS, calibrate_known
from .source_risk_run import _known_labels


REGIME = 'source_unweighted_membership_v1'
PARENT_RUN = embedding.PARENT_RUN


def _export_marker(root):
    marker = parent._read(Path(root) / 'complete.json')
    if (marker.get('status') != 'SOURCE_EMBEDDING_BRANCH_FEATURES_EXPORTED' or marker.get('source_only') is not True
            or marker.get('counts') != parent.COUNTS or marker.get('feature_refs') != parent.FEATURE_REFS
            or marker.get('model_classes') != 6 or marker.get('parameters') != 232993
            or marker.get('head_changed') is not False or marker.get('default_head_replaced') is not False
            or any(marker.get(key) != 0 for key in ('target_reads', 'original_U_s_reads', 'OE_truth_reads',
                'optimizer_updates', 'new_encoder_updates', 'new_identity_head_updates'))):
        raise ValueError('A complete matching frozen E51 branch export is required')
    exporter.validate_encoder_identity(marker.get('encoder_state_identity'), marker.get('arm'))
    if not isinstance(marker.get('parent_export_identity'), dict) or not marker['parent_export_identity']:
        raise ValueError('The original full-F6 baseline identity is required')
    return marker


def _identity(marker):
    return {key: marker[key] for key in ('arm', 'encoder_state_identity', 'parent_export_identity', 'source_OE_handles', 'counts')}


def _save(output, geometry, state):
    arrays = dict(schema=np.asarray(geometry['schema']), class_count=np.asarray(6, dtype=np.int64))
    for name, branch in geometry['branches'].items(): arrays.update({key + '__' + name: value for key, value in branch.items()})
    parent._npz(output / 'geometry.npz', arrays)
    arrays = dict(schema=np.asarray(state['schema']), descriptor_schema=np.asarray(consensus.DESCRIPTOR_SCHEMA),
        descriptor_dim=np.asarray(100, dtype=np.int64), method_names=np.asarray(parent.core.METHODS), mean=state['mean'], scale=state['scale'])
    for name, model in state['models'].items(): arrays.update({name + '__' + key: value for key, value in model.items()})
    parent._npz(output / 'readouts.npz', arrays)


def _score(data, geometry, state):
    return serialization._score(data, geometry, state)


def fit(export_root, model_seed, output):
    output = parent._new(output)
    marker = _export_marker(export_root); identity = _identity(marker)
    if type(model_seed) is not int or model_seed != 2026092701: raise ValueError('Fixed readout seed required')
    known = parent._load_features(export_root, 'source_L', True)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    geometry = parent.core.fit_geometry(parent._branches(known), known['y'])
    descriptor = consensus.descriptor(parent._branches(known), known['logits'], geometry)
    parent._write(output / 'smoke.json', dict(status='PASS', source_feature_descriptor=True,
        descriptor_shape=[2, 100], source_L_samples=2, source_V_reads=0, novel_validation_reads=0,
        encoder_forwards=0, encoder_checkpoint_smoke='completed by export first', optimizer_updates=0,
        finite=bool(np.isfinite(descriptor[:2]).all())))
    oe = parent._load_features(export_root, 'exposure_train', False)
    if set(known['ids'].tolist()) & set(oe['ids'].tolist()): raise ValueError('Known L and OE train roles overlap')
    state = parent.core.fit_readouts(descriptor, consensus.descriptor(parent._branches(oe), oe['logits'], geometry), model_seed)
    training = state['training']
    if (training.get('steps') != 200 or training.get('known_count') != parent.COUNTS['source_L']
            or training.get('oe_count') != parent.COUNTS['exposure_train']
            or training.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or any(len(trace) != 200 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('Both original 200-update readouts must complete')
    _save(output, geometry, state)
    loaded_geometry, loaded_state = serialization._load_new(output)
    if not np.isfinite(_score({key: value[:2] for key, value in known.items()}, loaded_geometry, loaded_state)).all():
        raise ValueError('Serialized new readouts are nonfinite on source L')
    parent._write(output / 'training.json', dict(training=training, seconds=time.perf_counter() - started,
        descriptor_schema=consensus.DESCRIPTOR_SCHEMA, readout_training_regime=REGIME, geometry_fitted=True,
        geometry_source='this encoder legal L_s only', source_OE_membership_is_additional_supervision=True,
        source_only=True, target_reads=0, new_encoder_updates=0, new_identity_head_updates=0,
        source_validation_reads=0, novel_validation_reads=0))
    result = dict(status='SOURCE_EMBEDDING_READOUTS_FROZEN', source_only=True, target_reads=0,
        arm=marker['arm'], descriptor_schema=consensus.DESCRIPTOR_SCHEMA, readout_training_regime=REGIME,
        geometry_fitted=True, new_encoder_updates=0, new_identity_head_updates=0,
        historical_encoder_updates=222, historical_identity_head_updates=222, readout_steps_per_model=200,
        source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0, model_seed=model_seed,
        parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633}, method_names=list(METHODS),
        encoder_state_identity=marker['encoder_state_identity'], parent_export_identity=marker['parent_export_identity'],
        export_identity=identity, export_root=str(export_root), geometry_ref='geometry.npz', readouts_ref='readouts.npz',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    parent._write(output / 'complete.json', result); return result


def _fit_marker(root, marker):
    fitted = parent._read(Path(root) / 'complete.json')
    if (fitted.get('status') != 'SOURCE_EMBEDDING_READOUTS_FROZEN' or fitted.get('export_identity') != _identity(marker)
            or fitted.get('encoder_state_identity') != marker['encoder_state_identity']
            or fitted.get('descriptor_schema') != consensus.DESCRIPTOR_SCHEMA or fitted.get('readout_training_regime') != REGIME
            or fitted.get('geometry_fitted') is not True or fitted.get('source_only') is not True
            or fitted.get('readout_steps_per_model') != 200 or fitted.get('method_names') != list(METHODS)
            or any(fitted.get(key) != 0 for key in ('target_reads', 'new_encoder_updates', 'new_identity_head_updates',
                'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads'))):
        raise ValueError('Matching new-encoder L-only geometry and frozen consensus readouts required before V')
    return fitted


def predict(export_root, fit_root, output):
    output = parent._new(output); marker = _export_marker(export_root)
    _fit_marker(fit_root, marker)
    geometry, state = serialization._load_new(fit_root)
    known = parent._load_features(export_root, 'source_V', False)
    knownness = _score(known, geometry, state); closed_known = known['logits'].argmax(axis=1).astype(np.int64)
    risk = calibrate_known(known['ids'], _known_labels(export_root, known), closed_known, knownness, known['rx_i'])
    baseline_root = Path(marker['config']['parent_run_root']) / 'export'
    baseline_known = embedding._baseline(baseline_root / parent.FEATURE_REFS['source_V'], known['ids'], 'source_V')
    output.mkdir(parents=True, exist_ok=False)
    calibration = output / 'calibration'; calibration.mkdir()
    with (calibration / 'thresholds.npy').open('xb') as stream: np.save(stream, risk['thresholds'], allow_pickle=False)
    summary = {key: value for key, value in risk.items() if key != 'thresholds'}
    summary.update(status='SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', source_only=True, target_reads=0,
        source_V_labels_read=True, source_known_label_calibration=True, unknown_calibration_scores_read=0,
        unknown_calibration_labels_read=0, novel_validation_reads=0, new_encoder_updates=0,
        new_identity_head_updates=0, readout_updates=0, thresholds_ref='thresholds.npy')
    parent._write(calibration / 'complete.json', summary)
    novel = parent._load_features(export_root, 'novel_validation', False)
    if set(known['ids'].tolist()) & set(novel['ids'].tolist()): raise ValueError('Known/novel source roles overlap')
    novelness = _score(novel, geometry, state)
    baseline_novel = embedding._baseline(baseline_root / parent.FEATURE_REFS['novel_validation'], novel['ids'], 'novel_validation')
    ids = np.r_[known['ids'], novel['ids']]; values = np.concatenate((knownness, novelness))
    closed = np.r_[closed_known, novel['logits'].argmax(axis=1)].astype(np.int64)
    baseline = np.r_[baseline_known, baseline_novel].astype(np.int64)
    accepted = values >= risk['thresholds']; order = np.argsort(ids, kind='stable')
    parent._npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=closed[order], baseline_closed_prediction=baseline[order], knownness=values[order],
        accepted=accepted[order], prediction=np.where(accepted, closed[:, None], -1).astype(np.int64)[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']][order], day_i=np.r_[known['day_i'], novel['day_i']][order]))
    result = dict(status='SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE', source_only=True, target_reads=0,
        arm=marker['arm'], encoder_state_identity=marker['encoder_state_identity'], parent_export_identity=marker['parent_export_identity'],
        encoder_updates=222, identity_head_updates=222, updates_scope='historical r02 source increment',
        new_encoder_updates=0, new_identity_head_updates=0, readout_steps_per_model=200, default_head_replaced=False,
        descriptor_schema=consensus.DESCRIPTOR_SCHEMA, readout_training_regime=REGIME,
        known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids), method_names=list(METHODS),
        source_V_labels_read=True, source_known_label_calibration=True, unknown_truth_read=False, truth_read=False,
        query_fit=False, unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
        thresholds_ref='calibration/thresholds.npy', predictions_ref='predictions.npz',
        independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='same-known-V empirical risk and repeated source novel evaluation; no formal target evidence')
    parent._write(output / 'complete.json', result); return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='action', required=True)
    for action in ('fit', 'predict'):
        command = commands.add_parser(action); command.add_argument('--export-root', required=True); command.add_argument('--output', required=True)
        if action == 'fit': command.add_argument('--model-seed', type=int, default=2026092701)
        else: command.add_argument('--fit-root', required=True)
    args = parser.parse_args(argv)
    result = fit(args.export_root, args.model_seed, args.output) if args.action == 'fit' else predict(args.export_root, args.fit_root, args.output)
    print(json.dumps(dict(status=result['status'], arm=result['arm']))); return 0


if __name__ == '__main__': raise SystemExit(main())
