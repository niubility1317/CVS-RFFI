"""U-only orthogonal adaptation of frozen E51, followed by future-only scoring.

Each invocation owns one isolated C0 or P1 condition. Source aggregates, F/G,
the native identity head and source thresholds stay frozen. No target truth or
query score is available to fitting; D0 is loaded only after adapter freeze.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import development_transfer as transfer
from .data import PUBLIC_FIELDS, PUBLIC_FILES, load_public
from .source_open_readout import DIMENSIONS


RUN_ID = '20261029-phase3-development-alignment-s2026092701-r01'
OLD_RUNS = dict(C0='20261029-phase3-development-clean-transfer-s2026092701-r01',
                P1='20261029-phase3-development-transfer-s2026092701-r01')
REGISTERED = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
METHODS, PRIMARY = transfer.METHODS, transfer.PRIMARY
FIELDS = {'run_id', 'code_commit', 'condition', 'view_family', 'source_row_root',
          'risk_prediction_root', 'd0_prediction_root', 'output_root', 'batch_size',
          'dura_checkpoint', 'dura_release'} | {prefix + suffix for prefix in ('U', 'Q')
          for suffix in ('_public_root', '_capsule_id', '_split_id', '_expected_count')}
PREDICTION_FIELDS = (transfer.PREDICTION_FIELDS - {'input_paths'}) | {'states'}
STATE_NAMES = ['D0', 'D1']
_read, _write = transfer._read, transfer._write
_source_bundle, _load_readers, _load_dura = transfer._source_bundle, transfer._load_readers, transfer._load_dura
_encode_new, _scores, _synthetic_smoke = transfer._encode_new, transfer._scores, transfer._synthetic_smoke
_truth = transfer._truth


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only the declared U/Q handles, frozen source, condition and D0 reference are allowed')
    strings = FIELDS - {'batch_size', 'U_expected_count', 'Q_expected_count', 'dura_checkpoint', 'dura_release'}
    if any(not isinstance(config[key], str) or not config[key].strip() for key in strings):
        raise ValueError('Nonempty identities and paths are required')
    condition = config['condition']
    if (config['run_id'] != RUN_ID or re.fullmatch('[0-9a-f]{40}', config['code_commit']) is None
            or condition not in OLD_RUNS or type(config['batch_size']) is not int or config['batch_size'] != 128
            or type(config['U_expected_count']) is not int or config['U_expected_count'] != 384
            or type(config['Q_expected_count']) is not int or config['Q_expected_count'] != 864):
        raise ValueError('The fixed independent run, condition, batch128 and U384/Q864 are required')
    family = 'clean_cross_receiver' if condition == 'C0' else 'practical_cross_receiver'
    if config['view_family'] != family:
        raise ValueError('Condition and clean/practical family differ')
    for prefix, role in [('U', 'discovery'), ('Q', 'evaluation')]:
        public = Path(config[prefix + '_public_root'])
        if public.name != 'public' or public.parent.name != role:
            raise ValueError('Only the current development discovery/evaluation public roles are accepted')
        cap = config[prefix + '_capsule_id']
        if (not cap.startswith('p3dev-') or (condition == 'C0') != cap.startswith('p3dev-clean-')):
            raise ValueError('Only matching isolated development capsule namespaces are accepted')
        if Path(config['output_root']).resolve().is_relative_to(public.resolve()):
            raise ValueError('Output cannot modify public inputs')
    if (config['U_capsule_id'] == config['Q_capsule_id']
            or config['U_split_id'] == config['Q_split_id']):
        raise ValueError('U and future Q require distinct role capsule and split handles')
    row, risk, old = (Path(config[key]) for key in ('source_row_root', 'risk_prediction_root', 'd0_prediction_root'))
    if (row.name != 'uniform_oe' or row.parent.name != 'rows' or row.parent.parent.name != transfer.READOUT_RUN
            or risk.name != 'predict' or risk.parent.name != 'uniform_oe' or risk.parent.parent.name != 'rows'
            or risk.parent.parent.parent.name != transfer.RISK_RUN
            or old.name != 'predict' or old.parent.name != OLD_RUNS[condition]):
        raise ValueError('Only the unchanged E51/OE source and exact completed condition D0 may be reused')
    if (Path(config['output_root']).resolve().is_relative_to(old.resolve())
            or old.resolve().is_relative_to(Path(config['output_root']).resolve())):
        raise ValueError('New output and old D0 prediction root must be separate')
    for key in ('dura_checkpoint', 'dura_release'):
        value = config[key]
        if condition == 'C0' and value is not None:
            raise ValueError('C0 has no DURA or star reference')
        if condition == 'P1' and (not isinstance(value, str) or not value.strip()):
            raise ValueError('P1 requires the original frozen DURA checkpoint and code')
    return config


def _metadata(config, role):
    """Read only declared public metadata/IDs before any received bytes."""
    root = Path(config[role + '_public_root'])
    if {path.name for path in root.iterdir()} != PUBLIC_FILES:
        raise ValueError('The public role must contain exactly three files and no truth')
    manifest = _read(root / 'manifest.json')
    ids = np.load(root / 'ids.npy', allow_pickle=False)
    count, clean = config[role + '_expected_count'], config['condition'] == 'C0'
    if (not isinstance(manifest, dict) or set(manifest) != PUBLIC_FIELDS
            or manifest['schema'] != ('phase3_clean_min_v1' if clean else 'phase3_min_v1')
            or manifest['data_status'] != 'VALIDATED_ONCE'
            or manifest['capsule_id'] != config[role + '_capsule_id']
            or manifest['split_id'] != config[role + '_split_id']
            or type(manifest['count']) is not int or manifest['count'] != count
            or manifest['iq_shape'] != [count, 2, 256] or manifest['registered_classes'] != REGISTERED
            or manifest['processing_route'] != ('clean' if clean else 'residual')
            or manifest['mode'] != ('clean' if clean else 'post_sync')
            or manifest['equalization_enabled'] is not False or manifest['single_observation'] is not True
            or manifest['fs_hz'] != 25000000 or ids.shape != (count,) or ids.dtype.kind != 'U'
            or len(set(ids.tolist())) != count or any(re.fullmatch('p3dev-[0-9a-f]{24}', str(i)) is None for i in ids)):
        raise ValueError('The declared development role, namespace, six classes or observation family differs; IQ closed')
    return manifest, ids


def _open_public(config, role):
    manifest, ids = _metadata(config, role)
    if config['condition'] == 'C0':
        from .development_clean import load_clean_public
        actual, loaded_ids, received = load_clean_public(config[role + '_public_root'])
    else:
        actual, loaded_ids, received = load_public(config[role + '_public_root'])
    if actual != manifest or not np.array_equal(ids, loaded_ids):
        received._mmap.close()
        raise ValueError('The loaded public role no longer matches its declared handles')
    return actual, loaded_ids, received


def _same(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _source_state(source):
    return {key: source[key] for key in ('geometry', 'state', 'thresholds')}


def _guard(models, snapshots, source, before, state=None, state_before=None):
    for model, snapshot in zip(models, snapshots):
        transfer._check_model(model, snapshot)
    if not _same(_source_state(source), before):
        raise ValueError('Frozen source geometry/readout/thresholds changed during target processing')
    if state is not None and not _same(state, state_before):
        raise ValueError('Frozen U alignment state changed on future Q')


def _fit_alignment(branches, logits, scores, thresholds, geometry):
    from .target_orthogonal_alignment import fit_alignment, aggregate_state
    return aggregate_state(fit_alignment(branches, logits, scores, thresholds, geometry))


def _apply_alignment(branches, state, geometry):
    from .target_orthogonal_alignment import apply_alignment
    return apply_alignment(branches, state, geometry)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _adapter_arrays(state):
    allowed = {'schema', 'parameters', 'applied', 'reason', 'sample_count', 'pseudo_label_counts',
               'selected_class_counts', 'class_weights', 'rotations', 'diagnostics'}
    if (not isinstance(state, dict) or not set(state) <= allowed
            or not {'schema', 'applied', 'rotations', 'sample_count'} <= set(state)
            or not isinstance(state['schema'], str) or not state['schema']
            or type(state['applied']) is not bool or state['sample_count'] != 384
            or not isinstance(state['rotations'], dict) or set(state['rotations']) != set(DIMENSIONS)):
        raise ValueError('Only R and aggregate alignment diagnostics may be persisted')
    arrays = {}
    for name, dimension in DIMENSIONS.items():
        value = state['rotations'][name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != (dimension, dimension) or not np.isfinite(value).all()):
            raise ValueError('Every branch must have a finite frozen float64 rotation')
        arrays['rotation__' + name] = value
    metadata = _jsonable({key: value for key, value in state.items() if key != 'rotations'})
    arrays['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    return arrays


def _load_alignment(root):
    with np.load(Path(root) / 'alignment.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'metadata_json'} | {'rotation__' + name for name in DIMENSIONS}:
            raise ValueError('Saved U adapter must contain only rotations and aggregate metadata')
        state = json.loads(str(saved['metadata_json'].item()))
        state['rotations'] = {name: saved['rotation__' + name].copy() for name in DIMENSIONS}
    _adapter_arrays(state)
    return state


def _branches(data, count):
    if set(data) != {'logits'} | {'features__' + name for name in DIMENSIONS}:
        raise ValueError('The full eight-branch default-head encoder output is required')
    result = {name: data['features__' + name] for name in DIMENSIONS}
    for name, value in result.items():
        if value.shape != (count, DIMENSIONS[name]) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Finite FP32 branch rows required')
    if data['logits'].shape != (count, 6) or data['logits'].dtype != np.float32 or not np.isfinite(data['logits']).all():
        raise ValueError('The complete finite FP32 native six-class logits are required')
    return result


def _current(received, start, stop, device, engine):
    import torch
    from .source_open_export import _tensor
    x = _tensor(received[start:stop], device)
    if engine is not None:
        x = engine(x)[0].detach().clone()
    if x.dtype != torch.float32 or tuple(x.shape) != (stop - start, 2, 256) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite fixed FP32 current-condition IQ is required')
    return x


def _reference(config, source, ids):
    """Called only after the U adapter's complete marker has been written."""
    done, pred = transfer._predictions(config['d0_prediction_root'])
    expected_G = (dict(reuse=True, graph=True, native=False, FP32=True, TF32=False)
                  if config['condition'] == 'P1' else None)
    if (done.get('run_id') != OLD_RUNS[config['condition']] or done['view_family'] != config['view_family']
            or done.get('encoder_state_identity') != source['marker']['encoder_state_identity']
            or done.get('source_row_root') != config['source_row_root']
            or done.get('risk_prediction_root') != config['risk_prediction_root']
            or done.get('public_capsule_id') != config['Q_capsule_id']
            or done.get('public_split_id') != config['Q_split_id']
            or done.get('dura_version') != ('dura_v1' if config['condition'] == 'P1' else None)
            or done.get('dura_release') != config['dura_release'] or done.get('dura_execution') != expected_G
            or done['count'] != config['Q_expected_count'] or not np.array_equal(pred['ids'], ids)
            or not np.array_equal(pred['thresholds'], source['thresholds'])):
        raise ValueError('Completed D0 identity, same-input baseline, IDs/order or thresholds differ')
    paths = pred['input_paths'].tolist()
    if config['condition'] not in paths:
        raise ValueError('The declared prior condition is absent from D0')
    index = paths.index(config['condition'])
    return done, {key: pred[key][index].copy() for key in
        ('closed_prediction', 'baseline_closed_prediction', 'knownness', 'accepted', 'prediction')}


def predict(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('Independent U-alignment output cannot be overwritten')
    _metadata(config, 'U')  # Wrong role/formal namespace fails without opening received IQ.
    source = _source_bundle(config)
    import torch
    from .source_open_export import _tensor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader, original = _load_readers(source, device)
    models, engine = [reader, original], None
    if config['condition'] == 'P1':
        dura, engine = _load_dura(source, config, device)
        models.append(dura)
    snapshots = [transfer._snapshot_model(model) for model in models]
    source_before = copy.deepcopy(_source_state(source))
    with torch.no_grad():
        _synthetic_smoke(source, reader, original, engine, [config['condition']], device)
        _guard(models, snapshots, source, source_before)
        _, U_ids, received = _open_public(config, 'U')
        try:
            rows, logits, values = {name: [] for name in DIMENSIONS}, [], []
            for start in range(0, len(U_ids), 128):
                stop = min(start + 128, len(U_ids))
                data = _encode_new(reader, _current(received, start, stop, device, engine))
                branches = _branches(data, stop - start)
                data['ids'] = U_ids[start:stop].copy()
                score_values = _scores(data, source)
                if score_values.shape != (stop - start, 7) or not np.isfinite(score_values).all():
                    raise ValueError('Seven finite frozen U scores required')
                for name in DIMENSIONS:
                    rows[name].append(branches[name])
                logits.append(data['logits']); values.append(score_values)
            state = _fit_alignment({name: np.concatenate(value) for name, value in rows.items()},
                np.concatenate(logits), np.concatenate(values), source['thresholds'], source['geometry'])
            del rows, logits, values
        finally:
            received._mmap.close()
        _guard(models, snapshots, source, source_before)
        arrays = _adapter_arrays(state)
        output.mkdir(parents=True, exist_ok=False)
        adapter = output / 'adapter'; adapter.mkdir()
        with (adapter / 'alignment.npz').open('xb') as stream:
            np.savez(stream, **arrays)
        fit = dict(status='U_ALIGNMENT_FROZEN', schema='pamf_phase3_U_alignment_complete_v1',
            condition=config['condition'], U_count=len(U_ids), U_capsule_id=config['U_capsule_id'],
            U_split_id=config['U_split_id'], encoder_state_identity=source['marker']['encoder_state_identity'],
            method_names=list(METHODS), thresholds=source['thresholds'].tolist(),
            adapter_applied=state['applied'], reason=state.get('reason'), rotations_ref='alignment.npz',
            source_IQ_reads=0, Q_IQ_reads=0, Q_prediction_reference_reads=0, truth_read=False,
            new_encoder_updates=0, new_readout_updates=0, new_threshold_calibrations=0,
            frozen_source_aggregates_unchanged=True, diagnostics=_jsonable(state.get('diagnostics', {})))
        _write(adapter / 'complete.json', fit)
        state = _load_alignment(adapter)
        state_before = copy.deepcopy(state)
        print(json.dumps(dict(status='U_ALIGNMENT_FROZEN', condition=config['condition'],
            adapter_applied=state['applied'], U_count=len(U_ids), Q_reads=0)), flush=True)
        # No query IDs, IQ, reference predictions or scores have entered fit.
        manifest, Q_ids = _metadata(config, 'Q')
        if set(U_ids.tolist()) & set(Q_ids.tolist()):
            raise ValueError('U and future Q opaque IDs overlap')
        old_done, old = _reference(config, source, Q_ids)
        values = np.stack((old['knownness'], old['knownness'].copy()))
        closed = np.stack((old['closed_prediction'], old['closed_prediction'].copy()))
        baseline = np.stack((old['baseline_closed_prediction'], old['baseline_closed_prediction'].copy()))
        Q_forwards = 0
        if state['applied']:
            _, loaded_ids, received = _open_public(config, 'Q')
            try:
                if not np.array_equal(loaded_ids, Q_ids):
                    raise ValueError('Future Q order differs')
                for start in range(0, len(Q_ids), 128):
                    stop = min(start + 128, len(Q_ids))
                    data = _encode_new(reader, _current(received, start, stop, device, engine))
                    branches = _branches(data, stop - start)
                    adapted = _apply_alignment(branches, state, source['geometry'])
                    if not isinstance(adapted, dict) or set(adapted) != set(DIMENSIONS):
                        raise ValueError('Alignment must return exactly all eight branches')
                    if any(not isinstance(adapted[name], np.ndarray)
                           or adapted[name].shape != (stop - start, dimension)
                           or adapted[name].dtype != np.float32 or not np.isfinite(adapted[name]).all()
                           for name, dimension in DIMENSIONS.items()):
                        raise ValueError('All aligned branches must remain finite FP32 before the native head')
                    check = {'features__' + name: adapted[name] for name in DIMENSIONS}
                    # Recompute the original native head after alignment. Old
                    # logits cannot describe the transformed fused feature.
                    fused = _tensor(adapted['fused160'], device)
                    logits = reader.classify_features(fused)
                    if logits.dtype != torch.float32 or tuple(logits.shape) != (stop - start, 6) or not bool(torch.isfinite(logits).all()):
                        raise ValueError('The unchanged native default head must classify the aligned fused feature')
                    check['logits'] = np.asarray(logits.detach().cpu().tolist(), dtype=np.float32)
                    _branches(check, stop - start)
                    check['ids'] = Q_ids[start:stop].copy()
                    scored = _scores(check, source)
                    if scored.shape != (stop - start, 7) or not np.isfinite(scored).all():
                        raise ValueError('Finite aligned descriptor/readout scores required')
                    closed[1, start:stop] = check['logits'].argmax(1)
                    values[1, start:stop] = scored
                    Q_forwards += stop - start
            finally:
                received._mmap.close()
        _guard(models, snapshots, source, source_before, state, state_before)
        accepted = values >= source['thresholds']
        with (output / 'predictions.npz').open('xb') as stream:
            np.savez(stream, ids=Q_ids, method_names=np.asarray(METHODS), states=np.asarray(STATE_NAMES),
                thresholds=source['thresholds'], closed_prediction=closed, baseline_closed_prediction=baseline,
                knownness=values, accepted=accepted, prediction=np.where(accepted, closed[:, :, None], -1).astype(np.int64))
        done = dict(status='DEVELOPMENT_ALIGNMENT_PREDICTIONS_COMPLETE',
            schema='pamf_phase3_development_alignment_predictions_v1', run_id=config['run_id'],
            code_commit=config['code_commit'], condition=config['condition'], view_family=config['view_family'],
            states=STATE_NAMES, primary_method=PRIMARY, method_names=list(METHODS), count=len(Q_ids),
            public_capsule_id=manifest['capsule_id'], public_split_id=manifest['split_id'],
            U_capsule_id=config['U_capsule_id'], U_split_id=config['U_split_id'], U_count=len(U_ids),
            encoder_state_identity=source['marker']['encoder_state_identity'],
            source_row_root=config['source_row_root'], risk_prediction_root=config['risk_prediction_root'],
            adapter_ref='adapter/alignment.npz', adapter_applied=state['applied'],
            identity_fallback=not state['applied'], Q_IQ_forward_samples=Q_forwards,
            D0_prediction_root=config['d0_prediction_root'], D0_run_id=old_done['run_id'],
            D0_input_path=config['condition'], D0_same_input_original_F6_baseline_reused=True,
            source_IQ_reads=0, U_dev_reads=len(U_ids), formal_target_reads=0, truth_read=False, query_fit=False,
            optimizer_updates=0, new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0,
            new_threshold_calibrations=0, default_head_replaced=False,
            frozen_parameters_buffers_unchanged=True, frozen_source_aggregates_unchanged=True,
            frozen_alignment_unchanged=True, real_checkpoint_synthetic_smoke_passed=True,
            synthetic_smoke_rows=2, anonymous_registration_performed=False, final_phase3_complete=False)
        _write(output / 'complete.json', done)
        return done


def _predictions(root):
    root = Path(root)
    try:
        done, fit = _read(root / 'complete.json'), _read(root / 'adapter/complete.json')
        state = _load_alignment(root / 'adapter')
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: frozen U adapter and complete D0/D1 predictions required') from exc
    condition = done.get('condition')
    if (done.get('status') != 'DEVELOPMENT_ALIGNMENT_PREDICTIONS_COMPLETE'
            or done.get('schema') != 'pamf_phase3_development_alignment_predictions_v1'
            or condition not in OLD_RUNS or done.get('run_id') != RUN_ID
            or done.get('view_family') != ('clean_cross_receiver' if condition == 'C0' else 'practical_cross_receiver')
            or done.get('states') != STATE_NAMES or done.get('method_names') != list(METHODS)
            or done.get('primary_method') != PRIMARY or done.get('count') != 864 or done.get('U_count') != 384
            or done.get('truth_read') is not False or done.get('query_fit') is not False
            or any(done.get(k) is not True for k in ('D0_same_input_original_F6_baseline_reused',
                'frozen_parameters_buffers_unchanged', 'frozen_source_aggregates_unchanged',
                'frozen_alignment_unchanged', 'real_checkpoint_synthetic_smoke_passed'))
            or any(done.get(k) != 0 for k in ('formal_target_reads', 'source_IQ_reads', 'optimizer_updates',
                'new_encoder_updates', 'new_identity_head_updates', 'new_readout_updates', 'new_threshold_calibrations'))
            or fit.get('status') != 'U_ALIGNMENT_FROZEN' or fit.get('condition') != condition or fit.get('U_count') != 384
            or fit.get('Q_IQ_reads') != 0 or fit.get('Q_prediction_reference_reads') != 0 or fit.get('truth_read') is not False
            or fit.get('encoder_state_identity') != done.get('encoder_state_identity')
            or fit.get('U_capsule_id') != done.get('U_capsule_id') or fit.get('U_split_id') != done.get('U_split_id')
            or fit.get('adapter_applied') != state['applied'] or done.get('adapter_applied') != state['applied']
            or done.get('identity_fallback') != (not state['applied'])
            or done.get('Q_IQ_forward_samples') != (864 if state['applied'] else 0)):
        raise ValueError('Truth closed: U freeze or complete D0/D1 prediction contract differs')
    with np.load(root / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != PREDICTION_FIELDS:
            raise ValueError('Truth closed: D0/D1 prediction field whitelist differs')
        pred = {key: saved[key].copy() for key in saved.files}
    ids = pred['ids']
    if (ids.shape != (864,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != 864
            or any(re.fullmatch('p3dev-[0-9a-f]{24}', str(i)) is None for i in ids)
            or pred['states'].tolist() != STATE_NAMES or pred['method_names'].tolist() != list(METHODS)):
        raise ValueError('Truth closed: prediction IDs/order/states/methods differ')
    for key in ('closed_prediction', 'baseline_closed_prediction'):
        value = pred[key]
        if value.shape != (2, 864) or value.dtype.kind not in 'iu' or np.any(value < 0) or np.any(value >= 6):
            raise ValueError('Truth closed: original/current complete six-class predictions required')
    values, thresholds, accepted, predicted = (pred[key] for key in ('knownness', 'thresholds', 'accepted', 'prediction'))
    if (values.shape != (2, 864, 7) or values.dtype.kind != 'f' or not np.isfinite(values).all()
            or thresholds.shape != (7,) or thresholds.dtype != np.float64 or not np.isfinite(thresholds).all()
            or not np.array_equal(thresholds, np.asarray(fit.get('thresholds'), dtype=np.float64))
            or accepted.shape != values.shape or accepted.dtype.kind != 'b' or predicted.shape != values.shape
            or predicted.dtype.kind not in 'iu' or not np.array_equal(accepted, values >= thresholds)
            or not np.array_equal(predicted, np.where(accepted, pred['closed_prediction'][:, :, None], -1))
            or not np.array_equal(pred['baseline_closed_prediction'][0], pred['baseline_closed_prediction'][1])):
        raise ValueError('Truth closed: scores, thresholds, decisions or same-input original baseline disagree')
    if not state['applied'] and any(not np.array_equal(pred[key][0], pred[key][1]) for key in
            ('knownness', 'closed_prediction', 'accepted', 'prediction')):
        raise ValueError('Truth closed: identity fallback must exactly reuse D0 without Q inference')
    return done, pred, fit


def score(predictions_root, truth, output, targets):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Independent alignment score output cannot be overwritten')
    done, pred, fit = _predictions(predictions_root)
    target_spec, unknown_min = transfer._targets(targets, done['view_family'])
    actual = _truth(truth, pred['ids'])  # First private truth connection, after every prediction is fixed.
    y, scenarios = actual['y'], actual['scenario']
    if not bool((actual['receiver'] == 26).all()) or np.any(y < 0) or np.any(y >= 9):
        raise ValueError('Only the RX24-1 private development scoring mapping is accepted')
    # These private group definitions never enter predict or fit_alignment.
    tx_names = REGISTERED + ['19-15', '19-16', '7-1']
    roles = dict(seen_in_U=np.isin(y, [6, 7]), later_unknown=y == 8)
    rows = {}
    for index, name in enumerate(STATE_NAMES):
        methods = {}
        for j, method in enumerate(METHODS):
            def metrics(mask):
                return transfer._metrics(y[mask], pred['closed_prediction'][index, mask],
                    pred['baseline_closed_prediction'][index, mask], pred['knownness'][index, mask, j],
                    pred['accepted'][index, mask, j])
            methods[method] = dict(overall=metrics(np.ones(len(y), dtype=bool)),
                by_scenario={str(scene): metrics(scenarios == scene) for scene in sorted(set(scenarios.tolist()))},
                by_transmitter={tx_names[int(label)]: metrics(y == label) for label in np.unique(y)},
                by_unknown_role={key: metrics(mask) for key, mask in roles.items()},
                by_scenario_unknown_role={str(scene): {key: metrics(mask & (scenarios == scene))
                    for key, mask in roles.items()} for scene in sorted(set(scenarios.tolist()))},
                diagnostic_only=method != PRIMARY)
        primary = methods[PRIMARY]
        later = primary['by_unknown_role']['later_unknown']
        later_checks = dict(later_unknown_present=later['unknown_count'] > 0,
            overall_later_rejection=(later['unknown_macro_rejection'] is not None
                                     and later['unknown_macro_rejection'] >= unknown_min),
            each_present_scene_later_rejection=all(row['later_unknown']['unknown_count'] == 0
                or (row['later_unknown']['unknown_macro_rejection'] is not None
                    and row['later_unknown']['unknown_macro_rejection'] >= unknown_min)
                for row in primary['by_scenario_unknown_role'].values()))
        rows[name] = dict(methods=methods, primary_goals=transfer._goals(primary['overall'],
            primary['by_scenario'], unknown_min, done['view_family']),
            primary_later_unknown_goals=dict(checks=later_checks, **{'pass': all(later_checks.values())}))
    keys = ('unknown_macro_rejection', 'known_macro_frr', 'known_macro_acceptance', 'known_macro_open_accuracy',
            'original_closed_macro_accuracy', 'encoder_closed_loss', 'original_closed_minus_new_open', 'auroc_known', 'oscr')
    delta = {method: {key: None if any(rows[name]['methods'][method]['overall'][key] is None for name in STATE_NAMES)
        else rows['D1']['methods'][method]['overall'][key] - rows['D0']['methods'][method]['overall'][key]
        for key in keys} for method in METHODS}
    gain = delta[PRIMARY]['known_macro_open_accuracy']
    later_protection = rows['D1']['primary_later_unknown_goals']['pass']
    success = bool(done['adapter_applied'] and rows['D1']['primary_goals']['pass'] and later_protection
                   and gain is not None and gain > 0)
    result = dict(status='SCORED_DEVELOPMENT_ALIGNMENT_COMPLETE',
        schema='pamf_phase3_development_alignment_scores_v1', condition=done['condition'],
        view_family=done['view_family'], truth_last=True, development_only=True, count=len(y),
        primary_method=PRIMARY, method_selection='Predeclared MLP16; six diagnostic scores cannot replace it',
        states=rows, D1_minus_D0=delta, adapter_applied=done['adapter_applied'],
        identity_fallback=done['identity_fallback'], adaptation_success=success,
        success_checks=dict(alignment_applied=done['adapter_applied'], all_D1_protection=rows['D1']['primary_goals']['pass'],
                            later_unknown_protection=later_protection,
                            old_open_gain_positive=gain is not None and gain > 0),
        condition_unknown_rejection_min=unknown_min, targets_schema=target_spec['schema'],
        known_macro_acceptance_min=.95, known_macro_FRR_max=.05, original_closed_open_loss_max=.02,
        U_dev_reads=done['U_dev_reads'], formal_target_reads=0, source_IQ_reads=0, query_fit=False,
        new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0, new_threshold_calibrations=0,
        target_feedback_forbidden=True, anonymous_registration_evaluated=False, anonymous_registration_performed=False,
        final_phase3_complete=False, next_step='U_ALIGNMENT_PASSED_DEVELOPMENT_ONLY' if success else 'NO_PROMOTION',
        metric_definitions=dict(known_acceptance='1 - known class-macro FRR; accepted wrong identities are still wrong',
            baseline='Copied original F6 on identical current-condition received input from completed D0',
            deltas='D1 minus exact reused D0 on the same IDs; fractions, not percentage points',
            unknown_roles='Private scorer only: labels6/7 seen_in_U but unregistered; label8 later_unknown; '
                          'all three TX were previously source-development evaluated, not source-new TX'))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'scores.json', result)
    _write(output / 'complete.json', {key: result[key] for key in ('status', 'truth_last', 'condition',
        'primary_method', 'count', 'adapter_applied', 'identity_fallback', 'adaptation_success',
        'success_checks', 'next_step', 'final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='DEVELOPMENT_ALIGNMENT_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('predict'); p.add_argument('--config', required=True); p.add_argument('--device', default='cuda:0')
    s = actions.add_parser('score'); s.add_argument('--predictions-root', required=True)
    s.add_argument('--truth', required=True); s.add_argument('--output', required=True); s.add_argument('--targets', required=True)
    args = parser.parse_args(argv)
    result = predict(_read(args.config), args.device) if args.action == 'predict' else score(
        args.predictions_root, args.truth, args.output, args.targets)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
