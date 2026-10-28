"""Frozen PAMF source calibration and truth-blind Phase3 prediction."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from .scores import METHODS, apply_rejection, fit_prototypes, knownness_scores, source_thresholds


COMMON = {'run_id', 'row_id', 'model_seed', 'base_release', 'frozen_sources', 'source_output',
          'source_contract', 'registered_classes', 'code_commit', 'batch_size', 'output_root'}
SOURCE = {'source_cache', 'retention'}
TARGET = {'public_root', 'calibration_root'}
FROZEN_SOURCE_KEYS = {'F', 'AF', 'BASE', 'FOLD', 'G'}
ORDER = [0, 5, 3, 6, 2, 1, 4, 7]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(action, config):
    if action not in ('smoke', 'calibrate', 'predict') or not isinstance(config, dict):
        raise ValueError('Unknown Phase3 runner mode')
    expected = COMMON | (TARGET if action == 'predict' else SOURCE)
    if set(config) != expected:
        raise ValueError('Mode input whitelist differs: ' + str(sorted(set(config) ^ expected)))
    for key in expected - {'registered_classes', 'frozen_sources', 'model_seed', 'batch_size', 'retention'}:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('Nonempty configuration string required: ' + key)
    classes = config['registered_classes']
    if (not isinstance(classes, list) or len(classes) != 6 or len(set(classes)) != 6
            or any(not isinstance(item, str) or not item for item in classes)):
        raise ValueError('Ordered six-class registered source roster required')
    roots = config['frozen_sources']
    if (not isinstance(roots, dict) or set(roots) != FROZEN_SOURCE_KEYS
            or any(not isinstance(value, str) or not value for value in roots.values())):
        raise ValueError('Exact frozen historical dependency roots required')
    if (isinstance(config['model_seed'], bool) or not isinstance(config['model_seed'], int)
            or config['model_seed'] <= 0 or config['batch_size'] != 128):
        raise ValueError('Fixed model seed and batch_size=128 required')
    if action != 'predict' and (isinstance(config['retention'], bool) or config['retention'] != .99):
        raise ValueError('Source retention is preregistered at 0.99')
    return config


def _prepare(action, config):
    validate_config(action, config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('A new immutable output directory is required: ' + str(output))
    contract = _read(config['source_contract'])
    if not isinstance(contract, dict) or contract.get('classes') != config['registered_classes']:
        raise ValueError('Checkpoint contract and registered class order differ')
    for role in ('L_s', 'V'):
        count = contract.get('counts', {}).get(role)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError('Existing source role count missing: ' + role)
    return contract, output


def _activate(config):
    base = Path(config['base_release']) / 'experiments'
    if not base.is_dir():
        raise ValueError('Immutable three-view base release is missing')
    package = importlib.import_module('experiments')
    current = Path(__file__).resolve().parents[1]
    previous = list(package.__path__)
    package.__path__ = [str(base)] + [str(value) for value in previous if Path(value) != base]
    historical_activate = importlib.import_module('experiments.pamf_repl_completion.runner').activate
    historical_activate(config)
    roots = [str(current), str(base)] + list(package.__path__)
    package.__path__ = list(dict.fromkeys(roots))


def _load_model(config, contract, device):
    _activate(config)
    from experiments.pamf_repl_view_shuffle.source import load_reader
    reader, actual = load_reader(config['source_output'], contract, device)
    row = actual.get('view_comparison', {})
    if (actual.get('model_seed') != config['model_seed'] or actual.get('num_classes') != 6
            or Path(actual.get('output_root', '')).resolve() != Path(config['source_output']).resolve()
            or row.get('view_count') != 3 or row.get('model_seed') != config['model_seed']):
        raise ValueError('Frozen source row, model seed or three-view architecture differs')
    pool = reader.sequence_pool
    if (len(pool.views) != 3 or tuple(pool.output_projection.weight.shape) != (160, 336)
            or sum(parameter.numel() for parameter in reader.parameters()) != 232993
            or any(view.window_order.tolist() != ORDER for view in pool.views)):
        raise ValueError('Current three-view PAMF parameter/temporal structure differs')
    reader.eval().requires_grad_(False)
    if any(module.training for module in reader.modules()) or any(parameter.requires_grad for parameter in reader.parameters()):
        raise ValueError('Frozen inference state required')
    return reader


def _source_dataset(config, role, contract):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    return SourceDataset(config['source_cache'], role, contract)


def _load_public(root):
    from .data import load_public
    return load_public(root)


def _encode(reader, values, device):
    import torch
    from experiments.cric_dura_comparison.io import tensor
    reader.eval().requires_grad_(False)
    if torch.is_tensor(values):
        x = values.to(device=device, dtype=torch.float32)
    else:
        array = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite fixed received IQ')
        x = tensor(array, device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, 256) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite FP32 I/Q [B,2,256] required')
    with torch.inference_mode():
        features = reader.features(x)
        logits = reader.classify_features(features)
        if (features.shape != (len(x), 160) or logits.shape != (len(x), 6)
                or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(logits).all())):
            raise ValueError('Invalid frozen PAMF features or default-head logits')
        # The deployed Torch/NumPy combination must not use its NumPy bridge.
        z = np.asarray(features.detach().cpu().tolist(), dtype=np.float64)
        raw = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
    return z, raw


def _encode_source(reader, dataset, config, device):
    zs, logits = [], []
    for start in range(0, len(dataset.x), config['batch_size']):
        z, raw = _encode(reader, dataset.x[start:start + config['batch_size']], device)
        zs.append(z)
        logits.append(raw)
    if not zs:
        raise ValueError('Empty source role')
    return np.concatenate(zs), np.concatenate(logits)


def _identity(config):
    return dict(run_id=config['run_id'], row_id=config['row_id'], model_seed=config['model_seed'],
                source_output=config['source_output'], source_contract=config['source_contract'],
                base_release=config['base_release'], frozen_sources=config['frozen_sources'],
                contract_classes=config['registered_classes'], code_commit=config['code_commit'])


def smoke(config, device='cuda:0'):
    contract, output = _prepare('smoke', config)
    reader = _load_model(config, contract, device)
    labeled = _source_dataset(config, 'L_s', contract)
    if len(labeled.x) < 2:
        raise ValueError('Two legal source-L samples required for real-checkpoint smoke')
    z, raw = _encode(reader, labeled.x[:2], device)
    result = dict(_identity(config), status='PASS', actual_checkpoint=True, source_samples=2,
                  features_shape=list(z.shape), logits_shape=list(raw.shape),
                  target_reads=0, U_truth_reads=0, optimizer_updates=0, query_fit=False)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'smoke.json', result)
    _write(output / 'complete.json', result)
    return result


def calibrate(config, device='cuda:0'):
    contract, output = _prepare('calibrate', config)
    reader = _load_model(config, contract, device)
    labeled = _source_dataset(config, 'L_s', contract)
    if len(labeled.x) != contract['counts']['L_s']:
        raise ValueError('Source L physical sample count differs')
    z_l, _ = _encode_source(reader, labeled, config, device)
    prototypes = fit_prototypes(z_l, np.asarray(labeled.metadata['y']), 6)
    del labeled, z_l
    validation = _source_dataset(config, 'V', contract)
    if len(validation.x) != contract['counts']['V']:
        raise ValueError('Source V physical sample count differs')
    z_v, raw_v = _encode_source(reader, validation, config, device)
    values = knownness_scores(z_v, raw_v, prototypes)
    thresholds = source_thresholds(values, config['retention'])
    result = dict(_identity(config), status='SOURCE_CALIBRATION_FROZEN', retention=config['retention'],
                  source_counts={role: contract['counts'][role] for role in ('L_s', 'V')},
                  method_names=list(METHODS), target_reads=0, U_truth_reads=0,
                  optimizer_updates=0, prototypes_from='L_s_only', thresholds_from='all_source_V_only', query_fit=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'bundle.npz').open('xb') as stream:
        np.savez(stream, prototypes=prototypes, thresholds=thresholds, method_names=np.asarray(METHODS))
    _write(output / 'complete.json', result)
    return result


def _calibration(config, contract):
    root = Path(config['calibration_root'])
    metadata = _read(root / 'complete.json')
    if not isinstance(metadata, dict) or metadata.get('status') != 'SOURCE_CALIBRATION_FROZEN':
        raise ValueError('Complete frozen source calibration required')
    expected = _identity(config)
    for key in ('model_seed', 'source_output', 'source_contract', 'base_release', 'frozen_sources', 'contract_classes', 'code_commit'):
        if metadata.get(key) != expected[key]:
            raise ValueError('Calibration and frozen model identity differ: ' + key)
    if (metadata.get('retention') != .99 or metadata.get('target_reads') != 0 or metadata.get('optimizer_updates') != 0
            or metadata.get('query_fit') is not False or metadata.get('method_names') != list(METHODS)
            or metadata.get('prototypes_from') != 'L_s_only' or metadata.get('thresholds_from') != 'all_source_V_only'
            or metadata.get('source_counts') != {role: contract['counts'][role] for role in ('L_s', 'V')}):
        raise ValueError('Source-only calibration roles or operating point differ')
    with np.load(root / 'bundle.npz', allow_pickle=False) as data:
        if set(data.files) != {'prototypes', 'thresholds', 'method_names'}:
            raise ValueError('Calibration bundle schema differs')
        prototypes, thresholds, names = (data[key].copy() for key in ('prototypes', 'thresholds', 'method_names'))
    if (prototypes.shape != (6, 160) or prototypes.dtype.kind != 'f' or not np.isfinite(prototypes).all()
            or not np.allclose(np.linalg.norm(prototypes, axis=1), 1., atol=1e-5)
            or thresholds.shape != (len(METHODS),) or thresholds.dtype.kind != 'f' or not np.isfinite(thresholds).all()
            or names.dtype.kind != 'U' or names.tolist() != list(METHODS)):
        raise ValueError('Frozen calibration dimensions, class centers or method order differ')
    return prototypes, thresholds


def predict(config, device='cuda:0'):
    contract, output = _prepare('predict', config)
    prototypes, thresholds = _calibration(config, contract)
    reader = _load_model(config, contract, device)
    manifest, ids, received = _load_public(config['public_root'])
    if manifest.get('registered_classes') != config['registered_classes']:
        raise ValueError('Public input and frozen registered class roster differ')
    if (ids.ndim != 1 or ids.dtype.kind != 'U' or not len(ids) or len(set(ids.tolist())) != len(ids)
            or received.shape != (len(ids), 2, 256) or received.dtype.kind != 'f'):
        raise ValueError('Public opaque IDs or fixed received shape differ')
    output.mkdir(parents=True, exist_ok=False)
    knownness_parts, raw_parts = [], []
    for start in range(0, len(ids), config['batch_size']):
        z, raw = _encode(reader, received[start:start + config['batch_size']], device)
        knownness_parts.append(knownness_scores(z, raw, prototypes))
        raw_parts.append(raw)
        print('PHASE3_FROZEN_BATCH', config['row_id'], min(start + config['batch_size'], len(ids)), len(ids), flush=True)
    values = np.concatenate(knownness_parts)
    raw = np.concatenate(raw_parts)
    decisions = apply_rejection(raw, values, thresholds)
    with (output / 'predictions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, method_names=np.asarray(METHODS), knownness=values, **decisions)
    result = dict(_identity(config), status='ALL_PREDICTIONS_COMPLETE', count=len(ids),
                  method_names=list(METHODS), truth_read=False, query_fit=False,
                  optimizer_updates=0, source_IQ_reads=0, public_capsule_id=manifest.get('capsule_id'),
                  public_split_id=manifest.get('split_id'), calibration_root=config['calibration_root'])
    _write(output / 'complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'calibrate', 'predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = globals()[args.action](_read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], row_id=result['row_id'], output=_read(args.config)['output_root'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
