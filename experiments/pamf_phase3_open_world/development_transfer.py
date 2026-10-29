"""Frozen E51/OE transfer to development public IQ, with independent scoring.

Practical prediction is paired P0 received/P1 frozen DURA. P0-only capability
has an explicit incomplete-comparison status. Clean C0 uses an independent
public input and never supplies a reference to a practical prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from .data import PUBLIC_FILES, load_public
from .scoring import _truth, group_metrics
from .source_risk_calibration import METHODS


PRIMARY = 'mlp16_oe'
READOUT_RUN = '20261029-phase3-source-embedding-readout-s2026092701-r01'
RISK_RUN = '20261029-phase3-source-baseline-risk-s2026092701-r01'
FIELDS = {'run_id', 'code_commit', 'public_root', 'capsule_id', 'split_id', 'expected_count',
          'source_row_root', 'risk_prediction_root', 'output_root', 'input_paths',
          'batch_size', 'view_family', 'dura_checkpoint', 'dura_release'}
PREDICTION_FIELDS = {'ids', 'method_names', 'input_paths', 'thresholds', 'closed_prediction',
                     'baseline_closed_prediction', 'knownness', 'accepted', 'prediction'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only declared frozen transfer inputs are allowed; no truth, U or fitting inputs')
    strings = FIELDS - {'expected_count', 'batch_size', 'input_paths', 'dura_checkpoint', 'dura_release'}
    if any(not isinstance(config[key], str) or not config[key].strip() for key in strings):
        raise ValueError('Nonempty transfer identities and paths are required')
    if (not re.fullmatch('[0-9a-f]{40}', config['code_commit'])
            or type(config['expected_count']) is not int or config['expected_count'] <= 0
            or type(config['batch_size']) is not int or config['batch_size'] != 128):
        raise ValueError('Actual commit, positive public count and batch128 are required')
    public = Path(config['public_root'])
    row, risk = Path(config['source_row_root']), Path(config['risk_prediction_root'])
    if (public.name != 'public' or public.parent.name != 'evaluation'
            or row.name != 'uniform_oe' or row.parent.name != 'rows' or row.parent.parent.name != READOUT_RUN
            or risk.name != 'predict' or risk.parent.name != 'uniform_oe' or risk.parent.parent.name != 'rows'
            or risk.parent.parent.parent.name != RISK_RUN):
        raise ValueError('Only development evaluation public and fixed uniform-OE source states are allowed')
    if Path(config['output_root']).resolve().is_relative_to(public.resolve()):
        raise ValueError('Prediction output cannot modify the public input directory')
    family, paths, checkpoint = config['view_family'], config['input_paths'], config['dura_checkpoint']
    release = config['dura_release']
    if family == 'practical_cross_receiver':
        if paths not in (['P0'], ['P0', 'P1']):
            raise ValueError('Practical input paths must be P0 or the complete ordered P0/P1 pair')
        if paths == ['P0', 'P1'] and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ValueError('Paired practical prediction requires the frozen DURA checkpoint; no fallback')
        if paths == ['P0', 'P1'] and (not isinstance(release, str) or not release.strip()):
            raise ValueError('Paired practical prediction requires the complete frozen G code directory')
    elif family == 'clean_cross_receiver':
        if paths != ['C0'] or checkpoint is not None or release is not None:
            raise ValueError('Clean is an isolated C0 public row without DURA or a practical reference')
    else:
        raise ValueError('Only the two explicitly separate cross-receiver conditions are supported')
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError('DURA checkpoint must be a path or explicit null')
    if release is not None and (not isinstance(release, str) or not release.strip()):
        raise ValueError('Frozen DURA release must be a path or explicit null')
    return config


def _source_bundle(config):
    """Load only immutable aggregates/markers, never source feature rows or IQ."""
    from . import source_embedding_readout_run as readout
    from . import source_baseline_risk_run as risk_run, source_baseline_risk as calibration
    row, risk = Path(config['source_row_root']), Path(config['risk_prediction_root'])
    marker = readout._export_marker(row / 'export')
    if marker['arm'] != 'uniform_oe':
        raise ValueError('The predeclared uniform-OE E51 encoder is required')
    readout._fit_marker(row / 'fit', marker)
    completed = _read(risk / 'complete.json')
    risk_run._validate_marker('predict', completed, 'uniform_oe')
    if (completed.get('run_id') != RISK_RUN
            or Path(completed.get('parent_row_root', '')).resolve() != row.resolve()
            or completed.get('encoder_state_identity') != marker['encoder_state_identity']
            or completed.get('descriptor_schema') != readout.consensus.DESCRIPTOR_SCHEMA
            or completed.get('readout_training_regime') != readout.REGIME
            or completed.get('thresholds_ref') != 'calibration/thresholds.npy'):
        raise ValueError('Original-baseline risk state and the fixed E51/readout identity differ')
    frozen = _read(risk / 'calibration/complete.json')
    expected = dict(status='SOURCE_ORIGINAL_BASELINE_RISK_THRESHOLDS_FROZEN',
        calibration_schema=calibration.SCHEMA, source_only=True, source_V_labels_read=True,
        source_known_label_calibration=True, method_names=list(METHODS), known_count=27000,
        class_count=6, thresholds_ref='thresholds.npy')
    expected.update({key: 0 for key in ('target_reads', 'unknown_calibration_scores_read',
        'unknown_calibration_labels_read', 'novel_validation_reads', 'new_encoder_updates',
        'new_identity_head_updates', 'new_readout_updates')})
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError('Only unchanged source-known original-baseline risk thresholds are allowed')
    thresholds = np.load(risk / 'calibration/thresholds.npy', allow_pickle=False)
    if thresholds.shape != (7,) or thresholds.dtype != np.float64 or not np.isfinite(thresholds).all():
        raise ValueError('Seven finite frozen source thresholds are required')
    geometry, state = readout.serialization._load_new(row / 'fit')
    return dict(marker=marker, geometry=geometry, state=state, thresholds=thresholds,
        source_row_root=str(row), risk_prediction_root=str(risk))


def _public(config):
    # Check only public metadata/opaque IDs before opening received bytes. This
    # prevents a wrongly supplied formal pool or U role from being consumed.
    root = Path(config['public_root'])
    if {path.name for path in root.iterdir()} != PUBLIC_FILES:
        raise ValueError('Only the three public development files are allowed')
    preflight = _read(root / 'manifest.json')
    opaque = np.load(root / 'ids.npy', allow_pickle=False)
    from .source_open_embedding_train import CLASSES
    prefix = r'p3dev-[0-9a-f]{24}' if config['view_family'] == 'practical_cross_receiver' else r'p3(?:dev|clean)-[0-9a-f]{24}'
    if (preflight.get('capsule_id') != config['capsule_id'] or preflight.get('split_id') != config['split_id']
            or preflight.get('count') != config['expected_count'] or preflight.get('registered_classes') != CLASSES
            or opaque.shape != (config['expected_count'],) or opaque.dtype.kind != 'U'
            or any(re.fullmatch(prefix, str(item)) is None for item in opaque)):
        raise ValueError('The declared development metadata or opaque namespace differs; received stays closed')
    if config['view_family'] == 'clean_cross_receiver':
        from .development_clean import load_clean_public
        manifest, ids, iq = load_clean_public(config['public_root'])
    else:
        manifest, ids, iq = load_public(config['public_root'])
    try:
        if (manifest['capsule_id'] != config['capsule_id'] or manifest['split_id'] != config['split_id']
                or manifest['count'] != config['expected_count'] or manifest['registered_classes'] != CLASSES
                or any(re.fullmatch(prefix, str(item)) is None for item in ids)):
            raise ValueError('The declared development public identity, count, old-six order or opaque namespace differs')
    except BaseException:
        iq._mmap.close()
        raise
    return manifest, ids, iq


def _load_readers(source, device):
    from . import source_embedding_readout_export as exporter, source_open_embedding_train as trainer
    from . import runner
    reader, train_config, identity = exporter._encoder(source['marker']['config'], device)
    if identity != source['marker']['encoder_state_identity']:
        raise ValueError('Loaded E51 encoder differs from the readout source identity')
    original = runner._load_model(train_config, trainer._contract(train_config), device)
    source['train_config'] = train_config
    return reader, original


def _load_dura(source, config, device):
    import importlib
    # This is the complete code snapshot used for the frozen G, including
    # ancestry skeleton dependencies absent from some current release subsets.
    release = Path(config['dura_release'])
    modules = release / 'experiments'
    if not (modules / 'pamf_r65_four_route/runtime.py').is_file() or not (modules / 'pamf_stage2_inference/engine.py').is_file():
        raise ValueError('The existing complete frozen DURA code directory is unavailable')
    package = importlib.import_module('experiments')
    package.__path__ = [str(modules)] + [str(path) for path in package.__path__ if Path(path) != modules]
    sys.path[:0] = [str(release), str(release / 'code')]
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from . import source_open_embedding_train as trainer
    # The current F contract and original G physical roles were already matched
    # by the source-only readiness run. This does not open a source IQ cache.
    trainer._contract(source['train_config'])
    model, payload = load_checkpoint(config['dura_checkpoint'], device)
    if ((payload['arm'], payload['epoch'], payload['updates'], payload['code_commit']) !=
            ('stage2', 20, 4940, 'b2a2ce937a64654e17e84c2aa817b9da4257a48f')
            or sum(p.numel() for p in model.parameters()) != 266951):
        raise ValueError('Only the original frozen DURA v1 R69 stage2 is allowed')
    model.eval().requires_grad_(False)
    return model, InferenceEngine(model, reuse=True, graph=True, native=False)


def _snapshot_model(model):
    import torch
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Only frozen eval FP32 modules may enter development inference')
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _check_model(model, before):
    import torch
    after = model.state_dict()
    _snapshot_model(model)
    if set(after) != set(before) or any(not torch.equal(before[key], value.detach().cpu()) for key, value in after.items()):
        raise ValueError('Frozen source parameters or buffers changed during query inference')


def _array_snapshot(value, prefix=''):
    if isinstance(value, np.ndarray):
        return {prefix: value.copy()}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_array_snapshot(item, prefix + '/' + key))
    return result


def _encode_new(reader, x):
    from .source_open_export import extract_branches
    branches, logits = extract_branches(reader, x)
    values = {'features__' + key: np.asarray(value.detach().cpu().tolist(), dtype=np.float32)
              for key, value in branches.items()}
    values['logits'] = np.asarray(logits.detach().cpu().tolist(), dtype=np.float32)
    return values


def _original_closed(reader, x):
    import torch
    with torch.no_grad():
        logits = reader.classify_features(reader.features(x))
    if logits.dtype != torch.float32 or tuple(logits.shape) != (len(x), 6) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Same-input original F6 must return complete finite default-head logits')
    return np.asarray(logits.argmax(1).detach().cpu().tolist(), dtype=np.int64)


def _scores(data, source):
    from .source_embedding_readout_run import _score
    return _score(data, source['geometry'], source['state'])


def _synthetic_smoke(source, reader, original, engine, paths, device):
    import torch
    n = torch.arange(256, dtype=torch.float32, device=device)
    x = torch.stack((torch.stack((torch.cos(.11 * n), torch.sin(.11 * n))),
                     torch.stack((torch.cos(.19 * n + .3), torch.sin(.19 * n + .3)))))
    for input_path in paths:
        current = x
        if input_path == 'P1':
            with torch.no_grad():
                current = engine(x)[0].detach().clone()
        if current.dtype != torch.float32 or tuple(current.shape) != (2, 2, 256) or not bool(torch.isfinite(current).all()):
            raise ValueError('Real-checkpoint synthetic smoke returned invalid IQ')
        data = _encode_new(reader, current)
        data['ids'] = np.asarray(['synthetic-a', 'synthetic-b'])
        values = _scores(data, source)
        if values.shape != (2, 7) or not np.isfinite(values).all():
            raise ValueError('Real-checkpoint synthetic descriptor/readout smoke failed')
        _original_closed(original, current)
    print(json.dumps(dict(status='DEVELOPMENT_TRANSFER_SYNTHETIC_SMOKE_PASS', synthetic_rows=2,
        input_paths=paths, descriptor_dimension=100, default_head_classes=6,
        source_IQ_reads=0, development_IQ_reads=0, formal_target_reads=0, optimizer_updates=0)), flush=True)


def predict(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Development prediction output cannot be overwritten')
    source = _source_bundle(config)
    import torch
    from .source_open_export import _tensor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader, original = _load_readers(source, device)
    models = [reader, original]
    engine = None
    if 'P1' in config['input_paths']:
        dura, engine = _load_dura(source, config, device)
        models.append(dura)
    before = [_snapshot_model(model) for model in models]
    frozen_arrays = _array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
    _synthetic_smoke(source, reader, original, engine, config['input_paths'], device)
    for model, snapshot in zip(models, before):
        _check_model(model, snapshot)
    smoke_arrays = _array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
    if any(not np.array_equal(value, smoke_arrays[key]) for key, value in frozen_arrays.items()):
        raise ValueError('Frozen source aggregates changed during no-query smoke')
    manifest, ids, received = _public(config)
    try:
        count, paths = len(ids), config['input_paths']
        values = np.empty((len(paths), count, 7), dtype=np.float64)
        closed = np.empty((len(paths), count), dtype=np.int64)
        baseline = np.empty_like(closed)
        output.mkdir(parents=True, exist_ok=False)
        for path_index, input_path in enumerate(paths):
            for start in range(0, count, config['batch_size']):
                stop = min(start + config['batch_size'], count)
                x = _tensor(received[start:stop], device)
                if input_path == 'P1':
                    # Engine output ownership and this normal-tensor clone
                    # preserve the native E51 Sinc cache's version semantics.
                    with torch.no_grad():
                        x = engine(x)[0].detach().clone()
                if x.dtype != torch.float32 or tuple(x.shape) != (stop - start, 2, 256) or not bool(torch.isfinite(x).all()):
                    raise ValueError('Finite fixed FP32[B,2,256] current input required')
                data = _encode_new(reader, x)
                data['ids'] = ids[start:stop].copy()
                score_values = _scores(data, source)
                logits = data['logits']
                if (score_values.shape != (stop - start, 7) or not np.isfinite(score_values).all()
                        or logits.shape != (stop - start, 6) or logits.dtype != np.float32
                        or not np.isfinite(logits).all()):
                    raise ValueError('Seven aligned finite frozen scores and the six-class head are required')
                values[path_index, start:stop] = score_values
                closed[path_index, start:stop] = logits.argmax(1)
                baseline[path_index, start:stop] = _original_closed(original, x)
                if start == 0 or stop == count:
                    print('DEVELOPMENT_TRANSFER_PROGRESS', input_path, stop, count, flush=True)
        for model, snapshot in zip(models, before):
            _check_model(model, snapshot)
        after_arrays = _array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
        if set(after_arrays) != set(frozen_arrays) or any(not np.array_equal(value, after_arrays[key]) for key, value in frozen_arrays.items()):
            raise ValueError('Frozen source geometry, readout or thresholds changed on query input')
        thresholds = source['thresholds']
        accepted = values >= thresholds
        prediction = np.where(accepted, closed[:, :, None], -1).astype(np.int64)
        with (output / 'predictions.npz').open('xb') as stream:
            np.savez(stream, ids=ids, method_names=np.asarray(METHODS), input_paths=np.asarray(paths),
                thresholds=thresholds, closed_prediction=closed, baseline_closed_prediction=baseline,
                knownness=values, accepted=accepted, prediction=prediction)
        paired = paths == ['P0', 'P1']
        clean = paths == ['C0']
        status = ('ALL_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE' if paired else
                  'CLEAN_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE' if clean else
                  'P0_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE_UNPAIRED')
        result = dict(status=status, schema='pamf_phase3_development_transfer_predictions_v1',
            run_id=config['run_id'], code_commit=config['code_commit'], development_only=True,
            paired_complete=paired, clean_condition_complete=clean, view_family=config['view_family'],
            input_paths=paths, primary_method=PRIMARY, method_names=list(METHODS), count=count,
            public_capsule_id=manifest['capsule_id'], public_split_id=manifest['split_id'],
            encoder_state_identity=source['marker']['encoder_state_identity'],
            source_row_root=config['source_row_root'], risk_prediction_root=config['risk_prediction_root'],
            descriptor_schema='head_aligned_branch_summary_v1',
            calibration_schema='phase3_source_original_baseline_risk_v1',
            dura_version='dura_v1' if paired else None,
            dura_release=config['dura_release'] if paired else None,
            dura_execution=dict(reuse=True, graph=True, native=False, FP32=True, TF32=False) if paired else None,
            same_input_original_F6_baseline=True, truth_read=False, query_fit=False,
            U_dev_reads=0, formal_target_reads=0, source_IQ_reads=0, optimizer_updates=0,
            new_encoder_updates=0, new_readout_updates=0, new_threshold_calibrations=0,
            default_head_replaced=False, frozen_parameters_buffers_unchanged=True,
            frozen_source_aggregates_unchanged=True, real_checkpoint_synthetic_smoke_passed=True,
            synthetic_smoke_rows=2, final_phase3_complete=False)
        _write(output / 'complete.json', result)
        return result
    finally:
        received._mmap.close()


def _predictions(root):
    root = Path(root)
    try:
        done = _read(root / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: complete frozen prediction marker required') from exc
    paired = done.get('status') == 'ALL_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE' and done.get('paired_complete') is True
    clean = done.get('status') == 'CLEAN_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE' and done.get('clean_condition_complete') is True
    expected_paths = ['P0', 'P1'] if paired else ['C0'] if clean else None
    if (expected_paths is None or done.get('input_paths') != expected_paths
            or done.get('view_family') != ('practical_cross_receiver' if paired else 'clean_cross_receiver')
            or done.get('primary_method') != PRIMARY or done.get('method_names') != list(METHODS)
            or done.get('schema') != 'pamf_phase3_development_transfer_predictions_v1'
            or done.get('truth_read') is not False or done.get('query_fit') is not False
            or done.get('same_input_original_F6_baseline') is not True
            or done.get('frozen_parameters_buffers_unchanged') is not True
            or done.get('frozen_source_aggregates_unchanged') is not True
            or any(done.get(key) != 0 for key in ('U_dev_reads', 'formal_target_reads', 'source_IQ_reads',
                'optimizer_updates', 'new_encoder_updates', 'new_readout_updates', 'new_threshold_calibrations'))):
        raise ValueError('Truth closed: full paired practical or isolated clean frozen predictions required')
    with np.load(root / 'predictions.npz', allow_pickle=False) as stored:
        if set(stored.files) != PREDICTION_FIELDS:
            raise ValueError('Truth closed: prediction artifact schema differs')
        pred = {key: stored[key].copy() for key in stored.files}
    ids = pred['ids']
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind != 'U' or np.any(ids == '')
            or len(set(ids.tolist())) != len(ids) or done.get('count') != len(ids)
            or pred['input_paths'].dtype.kind != 'U' or pred['input_paths'].tolist() != expected_paths
            or pred['method_names'].dtype.kind != 'U' or pred['method_names'].tolist() != list(METHODS)):
        raise ValueError('Truth closed: IDs, methods or input paths do not align')
    count, paths = len(ids), len(expected_paths)
    for key in ('closed_prediction', 'baseline_closed_prediction'):
        value = pred[key]
        if value.shape != (paths, count) or value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value >= 6):
            raise ValueError('Truth closed: complete aligned original/new six-class predictions required')
    values, thresholds, accepted, prediction = (pred[key] for key in ('knownness', 'thresholds', 'accepted', 'prediction'))
    shape = (paths, count, 7)
    if (values.shape != shape or values.dtype.kind != 'f' or not np.isfinite(values).all()
            or thresholds.shape != (7,) or thresholds.dtype != np.float64 or not np.isfinite(thresholds).all()
            or accepted.shape != shape or accepted.dtype.kind != 'b'
            or prediction.shape != shape or prediction.dtype.kind not in 'iu'
            or not np.array_equal(accepted, values >= thresholds)
            or not np.array_equal(prediction, np.where(accepted, pred['closed_prediction'][:, :, None], -1))):
        raise ValueError('Truth closed: frozen scores, thresholds, acceptances and decisions disagree')
    return done, pred


def _metrics(y, closed, baseline, values, accepted):
    result = group_metrics(y, closed, values, accepted, 6)
    labels = np.unique(y[y < 6])
    original = float(np.mean([np.mean(baseline[y == label] == label) for label in labels])) if len(labels) else None
    opened, new = result['known_macro_open_accuracy'], result['known_macro_closed_accuracy']
    result.update(original_closed_macro_accuracy=original,
        known_macro_acceptance=None if result['known_macro_frr'] is None else 1. - result['known_macro_frr'],
        encoder_closed_loss=None if original is None else original - new,
        original_closed_minus_new_open=None if original is None else original - opened)
    return result


def _targets(path, family):
    targets = _read(path)
    if (targets.get('schema') != 'pamf_phase3_scientific_targets_v2'
            or targets.get('cross_receiver_required') is not True
            or targets.get('each_preregistered_main_scene_required') is not True
            or targets.get('unknown_macro_rejection_min') != {'clean_cross_receiver': .95, 'practical_cross_receiver': .90}
            or targets.get('known_macro_acceptance', {}).get('minimum') != .95
            or targets.get('known_open_accuracy_drop_from_same_input_original_closed_max') != .02):
        raise ValueError('Current separate clean/practical scientific targets are required for reporting')
    return targets, targets['unknown_macro_rejection_min'][family]


def _goals(overall, scenes, unknown_min, family):
    required = ({'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban'}
                if family == 'practical_cross_receiver' else {'clean'})
    def passes(row):
        return (row['unknown_macro_rejection'] is not None and row['unknown_macro_rejection'] >= unknown_min
            and row['known_macro_acceptance'] is not None and row['known_macro_acceptance'] >= .95
            and row['original_closed_minus_new_open'] is not None and row['original_closed_minus_new_open'] <= .02 + 1e-12)
    checks = dict(overall_condition_targets=passes(overall), main_scene_coverage=set(scenes) == required,
        each_preregistered_main_scene_targets=set(scenes) == required and all(passes(row) for row in scenes.values()))
    return dict(checks=checks, **{'pass': all(checks.values())})


def score(predictions_root, truth, output, targets):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Development scoring output cannot be overwritten')
    done, pred = _predictions(predictions_root)
    target_spec, unknown_min = _targets(targets, done['view_family'])
    # Only after every route's complete predictions have passed the gate.
    actual = _truth(truth, pred['ids'])
    y, scenarios = actual['y'], actual['scenario']
    if not bool((actual['receiver'] == 26).all()):
        raise ValueError('Only the RX24-1 development truth may be scored here')
    tx_names = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20', '19-15', '19-16', '7-1']
    if np.any(y >= len(tx_names)):
        raise ValueError('Independent development truth TX mapping differs')
    rows = {}
    for path_index, input_path in enumerate(pred['input_paths'].tolist()):
        methods = {}
        for j, name in enumerate(METHODS):
            def metrics(mask):
                return _metrics(y[mask], pred['closed_prediction'][path_index, mask],
                    pred['baseline_closed_prediction'][path_index, mask], pred['knownness'][path_index, mask, j],
                    pred['accepted'][path_index, mask, j])
            overall = metrics(np.ones(len(y), dtype=bool))
            by_scene = {scene: metrics(scenarios == scene) for scene in sorted(set(scenarios.tolist()))}
            by_tx = {tx_names[int(label)]: metrics(y == label) for label in np.unique(y)}
            methods[name] = dict(overall=overall, by_scenario=by_scene, by_transmitter=by_tx,
                diagnostic_only=name != PRIMARY)
        primary = methods[PRIMARY]
        rows[input_path] = dict(methods=methods, primary_goals=_goals(primary['overall'],
            primary['by_scenario'], unknown_min, done['view_family']))
    deltas = None
    if done['paired_complete']:
        keys = ('unknown_macro_rejection', 'known_macro_frr', 'known_macro_acceptance',
                'known_macro_open_accuracy', 'original_closed_macro_accuracy', 'encoder_closed_loss',
                'original_closed_minus_new_open', 'auroc_known', 'oscr')
        deltas = {name: {key: None if rows['P0']['methods'][name]['overall'][key] is None or rows['P1']['methods'][name]['overall'][key] is None
            else rows['P1']['methods'][name]['overall'][key] - rows['P0']['methods'][name]['overall'][key]
            for key in keys} for name in METHODS}
    result = dict(status='SCORED_DEVELOPMENT_TRANSFER_COMPLETE', schema='pamf_phase3_development_transfer_scores_v1',
        truth_last=True, development_only=True, view_family=done['view_family'], paired_complete=done['paired_complete'],
        primary_method=PRIMARY, method_selection='predeclared MLP16; other six scores diagnostic; no target selection',
        count=len(y), input_paths=rows, P1_minus_P0=deltas, condition_unknown_rejection_min=unknown_min,
        targets_schema=target_spec['schema'], known_macro_acceptance_min=.95, original_closed_open_loss_max=.02,
        query_fit=False, new_threshold_calibrations=0, target_feedback_forbidden=True,
        final_phase3_complete=False, anonymous_registration_evaluated=False,
        evidence_scope='Independent RX development observations; source-novel TX were previously evaluated; '
            'not formal future query, new-TX blind confirmation, adaptation or registration success',
        metric_definitions=dict(known_macro_acceptance='1 - class-macro FRR; accepted wrong identities remain errors',
            original_closed_minus_new_open='Same-input original F6 class-macro closed accuracy minus E51 open accuracy',
            ranking='Known-positive AUROC and OSCR; score ties enter together',
            deltas='P1 minus P0 on identical physical records; fractions, not percentage points'))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'scores.json', result)
    _write(output / 'complete.json', dict(status=result['status'], truth_last=True, primary_method=PRIMARY,
        paired_complete=done['paired_complete'], count=len(y), final_phase3_complete=False,
        condition_unknown_rejection_min=unknown_min,
        primary_goals={key: value['primary_goals'] for key, value in rows.items()}))
    return result


def main(argv=None):
    print(json.dumps(dict(event='DEVELOPMENT_TRANSFER_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('predict')
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda:0')
    s = actions.add_parser('score')
    s.add_argument('--predictions-root', required=True)
    s.add_argument('--truth', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--targets', required=True)
    args = parser.parse_args(argv)
    result = (predict(_read(args.config), args.device) if args.action == 'predict'
              else score(args.predictions_root, args.truth, args.output, args.targets))
    print(json.dumps(dict(status=result['status'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
