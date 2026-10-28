"""Fit frozen-feature source OE readouts, then calibrate and predict without TX truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import source_open_readout as core
from . import scores
from .source_open_export import COUNTS as EXPORT_COUNTS, FEATURE_REFS, DIMENSIONS


COUNTS = dict(EXPORT_COUNTS)
METHODS = scores.METHODS + core.METHODS
COMMON = {'ids', 'logits', 'rx_i', 'day_i'} | {'features__' + name for name in DIMENSIONS}
MODEL_FIELDS = ('model_seed', 'source_output', 'source_contract', 'base_release',
                'frozen_sources', 'registered_classes', 'code_commit')
WEIGHT_KEYS = dict(linear_oe=('weight', 'bias'), mlp16_oe=('weight1', 'bias1', 'weight2', 'bias2'))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def _new(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('A new immutable source readout output root is required: ' + str(output))
    return output


def _identity(marker):
    config = marker.get('config')
    if not isinstance(config, dict) or any(key not in config for key in MODEL_FIELDS):
        raise ValueError('The frozen complete six-class encoder identity is missing')
    if type(config['model_seed']) is not int or config['model_seed'] <= 0:
        raise ValueError('The frozen encoder seed differs')
    classes = config['registered_classes']
    if not isinstance(classes, list) or len(classes) != 6 or len(set(classes)) != 6:
        raise ValueError('The six registered old source classes are required')
    handles = marker.get('source_OE_handles')
    if not isinstance(handles, dict) or set(handles) != {'exposure_train', 'novel_validation'}:
        raise ValueError('The two source OE capsule handles are required')
    for role, handle in handles.items():
        if (not isinstance(handle, dict) or set(handle) != {'capsule_id', 'split_id', 'count'}
                or handle['count'] != COUNTS[role]
                or any(not isinstance(handle[key], str) or not handle[key] for key in ('capsule_id', 'split_id'))):
            raise ValueError('Source OE capsule identity differs')
    return dict(model={key: config[key] for key in MODEL_FIELDS}, source_OE_handles=handles, counts=COUNTS.copy())


def _export_marker(export_root):
    marker = _read(Path(export_root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_OPEN_FEATURES_EXPORTED'
            or marker.get('source_only') is not True or marker.get('head_changed') is not False
            or any(marker.get(key) != 0 for key in ('target_reads', 'optimizer_updates', 'original_U_s_reads', 'OE_truth_reads'))
            or marker.get('model_classes') != 6 or marker.get('parameters') != 232993
            or marker.get('counts') != COUNTS or marker.get('feature_refs') != FEATURE_REFS):
        raise ValueError('The complete frozen full six-class source feature export is required')
    _identity(marker)
    return marker


def _load_features(export_root, role, include_y=False):
    if role not in COUNTS or type(include_y) is not bool or (include_y and role != 'source_L'):
        raise ValueError('Only source-L labels may enter the readout runner')
    path = Path(export_root) / FEATURE_REFS[role]
    expected = COMMON | ({'y'} if role in ('source_L', 'source_V') else set())
    wanted = COMMON | ({'y'} if include_y else set())
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != expected:
            raise ValueError('Exported source/OE feature field whitelist differs: ' + role)
        # In predict, source_V.y is never materialized, including for calibration.
        arrays = {key: stored[key].copy() for key in wanted}
    count = COUNTS[role]
    ids = arrays['ids']
    if (ids.shape != (count,) or ids.dtype.kind != 'U' or np.any(ids == '')
            or len(set(ids.tolist())) != count):
        raise ValueError('Unique aligned source/OE Unicode IDs required')
    logits = arrays['logits']
    if logits.shape != (count, 6) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError('Complete frozen six-class FP32 logits required')
    for key, allowed in (('rx_i', [1, 3, 4, 6, 8]), ('day_i', [1, 2, 3])):
        value = arrays[key]
        if value.shape != (count,) or value.dtype != np.int64 or not np.isin(value, allowed).all():
            raise ValueError('Source Main domain metadata differs: ' + key)
    branches = _branches(arrays)
    core._branches(branches)
    if any(len(value) != count for value in branches.values()):
        raise ValueError('Source branch count differs')
    if include_y:
        y = arrays['y']
        if y.shape != (count,) or y.dtype != np.int64 or set(y.tolist()) != set(range(6)):
            raise ValueError('All six visible source L labels are required')
    return arrays


def _branches(arrays):
    return {name: arrays['features__' + name] for name in DIMENSIONS}


def _npz(path, arrays):
    values = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind == 'O' for value in values.values()):
        raise ValueError('Object arrays are forbidden in frozen artifacts')
    with Path(path).open('xb') as stream:
        np.savez(stream, **values)


def _save_aggregates(output, geometry, readouts):
    core._geometry(geometry); core._readout_state(readouts)
    arrays = dict(schema=np.asarray(geometry['schema']), class_count=np.asarray(6, dtype=np.int64))
    for name in DIMENSIONS:
        arrays.update({key + '__' + name: value for key, value in geometry['branches'][name].items()})
    _npz(output / 'geometry.npz', arrays)
    arrays = dict(schema=np.asarray(readouts['schema']), descriptor_dim=np.asarray(100, dtype=np.int64),
        method_names=np.asarray(core.METHODS), mean=readouts['mean'], scale=readouts['scale'])
    for name in core.METHODS:
        arrays.update({name + '__' + key: value for key, value in readouts['models'][name].items()})
    _npz(output / 'readouts.npz', arrays)


def _load_aggregates(fit_root):
    root = Path(fit_root)
    expected = {'schema', 'class_count'} | {key + '__' + name for name in DIMENSIONS for key in ('mean', 'scale', 'prototypes', 'whitener')}
    with np.load(root / 'geometry.npz', allow_pickle=False) as stored:
        if set(stored.files) != expected or stored['schema'].shape != () or stored['schema'].dtype.kind != 'U' or stored['class_count'].shape != () or stored['class_count'].dtype != np.int64:
            raise ValueError('Frozen source geometry aggregate schema differs')
        geometry = dict(schema=stored['schema'].item(), class_count=int(stored['class_count'].item()),
            branches={name: {key: stored[key + '__' + name].copy() for key in ('mean', 'scale', 'prototypes', 'whitener')} for name in DIMENSIONS})
    expected = {'schema', 'descriptor_dim', 'method_names', 'mean', 'scale'} | {name + '__' + key for name, keys in WEIGHT_KEYS.items() for key in keys}
    with np.load(root / 'readouts.npz', allow_pickle=False) as stored:
        if (set(stored.files) != expected or stored['schema'].shape != () or stored['schema'].dtype.kind != 'U'
                or stored['descriptor_dim'].shape != () or stored['descriptor_dim'].dtype != np.int64
                or stored['method_names'].dtype.kind != 'U' or stored['method_names'].tolist() != list(core.METHODS)):
            raise ValueError('Frozen source readout aggregate schema differs')
        readouts = dict(schema=stored['schema'].item(), descriptor_dim=int(stored['descriptor_dim'].item()),
            method_names=stored['method_names'].tolist(), mean=stored['mean'].copy(), scale=stored['scale'].copy(),
            models={name: {key: stored[name + '__' + key].copy() for key in keys} for name, keys in WEIGHT_KEYS.items()},
            training={})
    core._geometry(geometry); core._readout_state(readouts)
    return geometry, readouts


def fit(export_root, model_seed, output):
    output = _new(output)
    marker = _export_marker(export_root)
    identity = _identity(marker)
    if type(model_seed) is not int or model_seed != identity['model']['model_seed']:
        raise ValueError('Readout seed must match the declared full six-class source model seed')
    known = _load_features(export_root, 'source_L', True)
    oe = _load_features(export_root, 'exposure_train', False)
    if set(known['ids'].tolist()) & set(oe['ids'].tolist()):
        raise ValueError('Known L and source OE train IDs overlap')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print('SOURCE_OPEN_FIT_GEOMETRY', len(known['ids']), flush=True)
    geometry = core.fit_geometry(_branches(known), known['y'])
    known_desc = core.descriptor(_branches(known), known['logits'], geometry)
    oe_desc = core.descriptor(_branches(oe), oe['logits'], geometry)
    print('SOURCE_OPEN_FIT_READOUTS_START', len(known_desc), len(oe_desc), 200, 'cpu', flush=True)
    readouts = core.fit_readouts(known_desc, oe_desc, model_seed)
    training = readouts['training']
    if (training.get('steps') != 200 or training.get('model_seed') != model_seed
            or training.get('known_count') != COUNTS['source_L'] or training.get('oe_count') != COUNTS['exposure_train']
            or training.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or training.get('source_membership_supervision') is not True
            or training.get('oe_transmitter_labels_used') is not False or training.get('identity_head_trained') is not False
            or set(training.get('losses', {})) != set(core.METHODS)
            or any(len(trace) != 200 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('Both fixed 200-update source OE readouts must complete')
    _save_aggregates(output, geometry, readouts)
    _write(output / 'training.json', dict(status='SOURCE_OPEN_READOUT_TRAINING_COMPLETE', training=training,
        seconds=time.perf_counter() - started, source_only=True, target_reads=0,
        encoder_updates=0, identity_head_updates=0, source_validation_reads=0, novel_validation_reads=0,
        geometry_role='original source L only', normalizer_role='source L plus source exposure_train only',
        source_OE_membership_is_additional_supervision=True, formal_target_evidence=False))
    result = dict(status='SOURCE_OPEN_READOUTS_FROZEN', source_only=True, target_reads=0,
        encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200,
        source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0,
        model_seed=model_seed, export_identity=identity, export_root=str(export_root),
        method_names=list(METHODS), learned_method_names=list(core.METHODS),
        parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633},
        geometry_ref='geometry.npz', readouts_ref='readouts.npz', training_ref='training.json',
        source_OE_membership_is_additional_supervision=True)
    _write(output / 'complete.json', result)
    print('SOURCE_OPEN_READOUTS_FROZEN', model_seed, 200, flush=True)
    return result


def _frozen_fit(fit_root, identity):
    marker = _read(Path(fit_root) / 'complete.json')
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_OPEN_READOUTS_FROZEN'
            or marker.get('source_only') is not True or marker.get('export_identity') != identity
            or any(marker.get(key) != 0 for key in ('target_reads', 'encoder_updates', 'identity_head_updates',
                'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads'))
            or marker.get('readout_steps_per_model') != 200 or marker.get('model_seed') != identity['model']['model_seed']
            or marker.get('method_names') != list(METHODS) or marker.get('learned_method_names') != list(core.METHODS)
            or marker.get('parameter_counts') != {'linear_oe': 101, 'mlp16_oe': 1633}
            or marker.get('geometry_ref') != 'geometry.npz' or marker.get('readouts_ref') != 'readouts.npz'):
        raise ValueError('Matching complete frozen source readouts must precede all validation reads')
    return marker


def _score(data, geometry, readouts):
    branches = _branches(data)
    controls = scores.knownness_scores(branches['fused160'], data['logits'], geometry['branches']['fused160']['prototypes'])
    learned = core.score_readouts(core.descriptor(branches, data['logits'], geometry), readouts)
    values = np.concatenate((controls, learned), axis=1)
    if values.shape != (len(data['ids']), 7) or not np.isfinite(values).all():
        raise ValueError('All seven fixed source knownness scores must be finite')
    return values


def predict(export_root, fit_root, output):
    output = _new(output)
    marker = _export_marker(export_root)
    identity = _identity(marker)
    _frozen_fit(fit_root, identity)
    geometry, readouts = _load_aggregates(fit_root)
    known = _load_features(export_root, 'source_V', False)
    knownness = _score(known, geometry, readouts)
    # The old helper fixes five columns. Both learned columns use its identical
    # 1-retention/lower formula; only known source V enters either calculation.
    thresholds = np.r_[scores.source_thresholds(knownness[:, :5], .99),
                       np.quantile(knownness[:, 5:], 1. - .99, axis=0, method='lower')]
    if thresholds.shape != (7,) or not np.isfinite(thresholds).all():
        raise ValueError('Seven fixed known-source thresholds required')
    output.mkdir(parents=True, exist_ok=False)
    calibration = output / 'calibration'; calibration.mkdir()
    with (calibration / 'thresholds.npy').open('xb') as stream:
        np.save(stream, thresholds, allow_pickle=False)
    _write(calibration / 'complete.json', dict(status='SOURCE_OPEN_THRESHOLDS_FROZEN', source_only=True,
        target_reads=0, encoder_updates=0, identity_head_updates=0, readout_updates=0,
        threshold_role='known source V only; permitted source threshold calibration',
        known_count=COUNTS['source_V'], retention=.99, quantile=1. - .99, quantile_method='lower',
        source_V_labels_read=False, novel_validation_reads=0, method_names=list(METHODS), thresholds_ref='thresholds.npy'))
    print('SOURCE_OPEN_THRESHOLDS_FROZEN', COUNTS['source_V'], 7, flush=True)
    novel = _load_features(export_root, 'novel_validation', False)
    if set(known['ids'].tolist()) & set(novel['ids'].tolist()):
        raise ValueError('Known V and novel source validation IDs overlap')
    novelness = _score(novel, geometry, readouts)
    ids = np.r_[known['ids'], novel['ids']]
    values = np.concatenate((knownness, novelness))
    closed = np.r_[known['logits'].argmax(axis=1), novel['logits'].argmax(axis=1)].astype(np.int64)
    accepted = values >= thresholds
    prediction = np.where(accepted, closed[:, None], -1).astype(np.int64)
    order = np.argsort(ids, kind='stable')
    _npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=closed[order], knownness=values[order], accepted=accepted[order], prediction=prediction[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']][order], day_i=np.r_[known['day_i'], novel['day_i']][order]))
    result = dict(status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', source_only=True, target_reads=0,
        truth_read=False, query_fit=False, known_count=COUNTS['source_V'], novel_count=COUNTS['novel_validation'],
        count=COUNTS['source_V'] + COUNTS['novel_validation'], method_names=list(METHODS),
        encoder_updates=0, identity_head_updates=0, readout_updates=0, source_L_reads=0, exposure_train_reads=0,
        source_V_labels_read=False, OE_truth_reads=0, source_known_V_threshold_calibration=True,
        threshold_role='known source V only', threshold_retention=.99, thresholds_ref='calibration/thresholds.npy',
        predictions_ref='predictions.npz', export_identity=identity, export_root=str(export_root), fit_root=str(fit_root))
    _write(output / 'complete.json', result)
    print('ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', result['count'], 7, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    fitter = commands.add_parser('fit'); fitter.add_argument('--export-root', required=True)
    fitter.add_argument('--model-seed', type=int, default=2026092701); fitter.add_argument('--output', required=True)
    predictor = commands.add_parser('predict'); predictor.add_argument('--export-root', required=True)
    predictor.add_argument('--fit-root', required=True); predictor.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = fit(args.export_root, args.model_seed, args.output) if args.action == 'fit' else predict(args.export_root, args.fit_root, args.output)
    print(json.dumps(dict(status=result['status'], output=args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
