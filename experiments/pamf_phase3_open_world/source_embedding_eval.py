"""Evaluate a frozen source embedding increment, then freeze known-V risk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import runner, scores
from . import source_open_embedding_train as trainer
from . import source_open_export as source
from . import source_open_run as parent
from .source_risk_calibration import calibrate_known


KEYS = {'run_id', 'row_id', 'code_commit', 'train_config', 'checkpoint', 'parent_run_root',
        'novel_public_root', 'output_root'}
COUNTS = dict(source_L=6300, source_V=27000, novel_validation=3150)
PARENT_RUN = '20261029-phase3-source-open-readout-s2026092701-r01'
METHODS = tuple(scores.METHODS)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the eight declared source embedding evaluation inputs are accepted')
    if any(not isinstance(config[key], str) or not config[key].strip() for key in KEYS):
        raise ValueError('Nonempty evaluation strings required')
    if len(config['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in config['code_commit']):
        raise ValueError('Actual execution Git commit required')
    return config


def _training(config):
    train_config = runner._read(config['train_config'])
    trainer.validate_config(train_config)
    if (any(train_config[key] != config[key] for key in ('run_id', 'row_id', 'code_commit'))
            or Path(config['checkpoint']).resolve() != (Path(train_config['output_root']) / 'final.pt').resolve()):
        raise ValueError('Evaluation must use its own row and final increment checkpoint')
    marker = runner._read(Path(train_config['output_root']) / 'complete.json')
    if (marker.get('status') != 'SOURCE_OPEN_EMBEDDING_FROZEN' or marker.get('schema') != trainer.SCHEMA
            or marker.get('config') != train_config or marker.get('arm') != train_config['arm']
            or marker.get('updates') != 222 or marker.get('parent_updates') != 11100
            or marker.get('cumulative_updates') != 11322 or marker.get('optimizer_reset') is not True
            or marker.get('source_only') is not True or marker.get('default_head_joint') is not True
            or marker.get('identity_head_replaced') is not False or marker.get('model_classes') != 6
            or marker.get('parameters') != 232993
            or any(marker.get(key) != 0 for key in ('U_reads', 'validation_reads', 'novel_validation_reads', 'target_reads', 'OE_truth_reads'))):
        raise ValueError('Only a complete own-row 222-update source-only increment may be evaluated')
    return train_config, marker


def _parent(config, train_config):
    root = Path(config['parent_run_root'])
    if root.name != PARENT_RUN:
        raise ValueError('The original source-open full-F6 export is the fixed baseline')
    export_root = root / 'export'
    marker = parent._export_marker(export_root)
    identity = parent._identity(marker)
    for key in ('model_seed', 'source_output', 'source_contract', 'base_release', 'frozen_sources', 'registered_classes'):
        if identity['model'][key] != train_config[key]:
            raise ValueError('New increment and original baseline model/data parent differ: ' + key)
    return export_root, identity


def _encode(reader, values, device):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('The new full identity reader must be frozen before evaluation')
    x = source._tensor(values, device)
    if x.ndim != 3 or tuple(x.shape[1:]) != (2, 256) or not len(x) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite source FP32[B,2,256] required')
    # Native trainable Sinc is intentionally retained by the new loader. Normal
    # no_grad tensors keep its versioned evaluation cache valid across batches.
    with torch.no_grad():
        z = reader.features(x)
        logits = reader.classify_features(z)
    if (tuple(z.shape) != (len(x), 160) or tuple(logits.shape) != (len(x), 6)
            or z.dtype != torch.float32 or logits.dtype != torch.float32
            or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(logits).all())):
        raise ValueError('New fused160 and complete unchanged six-class default head required')
    return (np.asarray(z.detach().cpu().tolist(), dtype=np.float32),
            np.asarray(logits.detach().cpu().tolist(), dtype=np.float32))


def _encode_role(reader, values, train_config, role, device):
    z, logits = [], []
    for start in range(0, len(values), train_config['batch_size']):
        part, raw = _encode(reader, values[start:start + train_config['batch_size']], device)
        z.append(part); logits.append(raw)
        if start == 0 or start // train_config['batch_size'] % 32 == 0 or start + train_config['batch_size'] >= len(values):
            print('SOURCE_EMBEDDING_EVAL_PROGRESS', train_config['row_id'], role,
                  min(start + train_config['batch_size'], len(values)), len(values), flush=True)
    if not z:
        raise ValueError('A complete nonempty source evaluation role is required')
    return np.concatenate(z), np.concatenate(logits)


def _baseline(path, expected_ids, role):
    if role not in ('source_V', 'novel_validation'):
        raise ValueError('Only the two fixed source evaluation baselines may be read')
    expected = parent.COMMON | ({'y'} if role == 'source_V' else set())
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != expected:
            raise ValueError('Original source baseline export schema differs')
        # No old embeddings, TX labels, or other per-row fields are materialized.
        ids, logits = stored['ids'].copy(), stored['logits'].copy()
    n = len(expected_ids)
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or np.any(ids == '')
            or len(set(ids.tolist())) != n or set(ids.tolist()) != set(expected_ids.tolist())
            or logits.shape != (n, 6) or logits.dtype != np.float32 or not np.isfinite(logits).all()):
        raise ValueError('Original and new default-head prediction IDs must align exactly')
    location = {value: i for i, value in enumerate(ids.tolist())}
    order = np.asarray([location[value] for value in expected_ids.tolist()], dtype=np.int64)
    return logits.argmax(axis=1).astype(np.int64)[order]


def _load_novel(public_root, identity):
    public = Path(public_root)
    if public.name != 'public' or public.parent.name != 'novel_validation':
        raise ValueError('Only the fixed novel-validation public role is allowed')
    data_root = public.parent.parent
    handles = source._oe_handles(data_root)
    if handles['novel_validation'] != identity['source_OE_handles']['novel_validation']:
        raise ValueError('The validated source novel capsule/split identity differs')
    values, metadata = source._oe_public(data_root, 'novel_validation')
    return values, metadata, handles['novel_validation']


def _calibration(output, prototypes, risk):
    calibration = output / 'calibration'
    calibration.mkdir(parents=False, exist_ok=False)
    for name, array in (('prototypes.npy', prototypes), ('thresholds.npy', risk['thresholds'])):
        with (calibration / name).open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    marker = {key: value for key, value in risk.items() if key != 'thresholds'}
    marker.update(status='SOURCE_KNOWN_RISK_THRESHOLDS_FROZEN', source_only=True, target_reads=0,
        source_V_labels_read=True, source_known_label_calibration=True,
        unknown_calibration_scores_read=0, unknown_calibration_labels_read=0, novel_validation_reads=0,
        eval_optimizer_updates=0, source_V_model_updates=0, prototypes_from='new_model_legal_L_s_only',
        prototype_ref='prototypes.npy', thresholds_ref='thresholds.npy',
        independent_confirmation=False, source_novel_evaluation_reused=True)
    runner._write(calibration / 'complete.json', marker)
    print('SOURCE_EMBEDDING_RISK_THRESHOLDS_FROZEN', risk['known_count'], len(METHODS), flush=True)


def evaluate(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('A new immutable embedding evaluation root is required')
    train_config, train_marker = _training(config)
    export_root, parent_identity = _parent(config, train_config)
    wrapper = trainer.load_trained_model(train_config, config['checkpoint'], device)
    reader = wrapper.id_backbone.encoder
    source._validate_model(reader)
    contract = trainer._contract(train_config)
    labeled = runner._source_dataset(train_config, 'L_s', contract)
    lmeta = source._source_meta(labeled, 'L_s', contract)
    output.mkdir(parents=True, exist_ok=False)
    smoke_z, smoke_logits = _encode(reader, labeled.x[:2], device)
    runner._write(output / 'smoke.json', dict(status='PASS', actual_checkpoint=True,
        checkpoint=config['checkpoint'], source_role='own_legal_L_s', source_samples=2,
        feature_shape=list(smoke_z.shape), logit_shape=list(smoke_logits.shape),
        validation_reads=0, novel_validation_reads=0, OE_train_reads=0, U_reads=0,
        target_reads=0, optimizer_updates=0, default_head_replaced=False))
    print('SOURCE_EMBEDDING_OWN_CHECKPOINT_SMOKE_PASS', config['row_id'], 2, flush=True)
    z_l, _ = _encode_role(reader, labeled.x, train_config, 'L_s', device)
    prototypes = scores.fit_prototypes(z_l, lmeta['y'], 6)
    del labeled, z_l
    validation = runner._source_dataset(train_config, 'V', contract)
    known = source._source_meta(validation, 'V', contract)
    if set(lmeta['ids'].tolist()) & set(known['ids'].tolist()):
        raise ValueError('Source L/V physical IDs overlap')
    z_v, logits_v = _encode_role(reader, validation.x, train_config, 'V', device)
    knownness = scores.knownness_scores(z_v, logits_v, prototypes)
    closed_known = logits_v.argmax(axis=1).astype(np.int64)
    baseline_known = _baseline(export_root / parent.FEATURE_REFS['source_V'], known['ids'], 'source_V')
    risk = calibrate_known(known['ids'], known['y'], closed_known, knownness, known['rx_i'], method_names=METHODS)
    thresholds = risk['thresholds']
    _calibration(output, prototypes, risk)
    del validation, z_v
    # The only novel input is opened after the global thresholds are persisted.
    novel_iq = None
    try:
        novel_iq, novel, novel_handle = _load_novel(config['novel_public_root'], parent_identity)
        if (set(novel['ids'].tolist()) & (set(known['ids'].tolist()) | set(lmeta['ids'].tolist()))):
            raise ValueError('Novel source observations overlap known L/V')
        z_n, logits_n = _encode_role(reader, novel_iq, train_config, 'novel_validation', device)
        novelness = scores.knownness_scores(z_n, logits_n, prototypes)
        baseline_novel = _baseline(export_root / parent.FEATURE_REFS['novel_validation'], novel['ids'], 'novel_validation')
    finally:
        if isinstance(novel_iq, np.memmap):
            novel_iq._mmap.close()
    source._validate_model(reader)
    ids = np.r_[known['ids'], novel['ids']]
    values = np.concatenate((knownness, novelness))
    closed = np.r_[closed_known, logits_n.argmax(axis=1)].astype(np.int64)
    baseline = np.r_[baseline_known, baseline_novel].astype(np.int64)
    accepted = values >= thresholds
    order = np.argsort(ids, kind='stable')
    parent._npz(output / 'predictions.npz', dict(ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=closed[order], knownness=values[order], accepted=accepted[order],
        prediction=np.where(accepted, closed[:, None], -1).astype(np.int64)[order],
        rx_i=np.r_[known['rx_i'], novel['rx_i']].astype(np.int64)[order],
        day_i=np.r_[known['day_i'], novel['day_i']].astype(np.int64)[order],
        baseline_closed_prediction=baseline[order]))
    result = dict(status='SOURCE_EMBEDDING_PREDICTIONS_COMPLETE', config=config,
        run_id=config['run_id'], row_id=config['row_id'], code_commit=config['code_commit'],
        arm=train_config['arm'], updates=222, encoder_updates=222, identity_head_updates=222,
        update_scope='inherited source training increment; evaluation performs no optimizer updates',
        eval_optimizer_updates=0, readout_updates=0, default_head_replaced=False,
        source_only=True, target_reads=0, U_reads=0, OE_train_reads=0,
        known_count=len(known['ids']), novel_count=len(novel['ids']), count=len(ids), method_names=list(METHODS),
        source_V_labels_read=True, source_known_label_calibration=True, unknown_truth_read=False,
        truth_read=False, truth_read_scope='unknown/private TX truth; known-source labels calibrate source risk',
        query_fit=False, source_V_model_updates=0, unknown_calibration_scores_read=0,
        unknown_calibration_labels_read=0, parent_export_identity=parent_identity,
        source_OE_novel_handle=novel_handle, source_L_prototype_count=COUNTS['source_L'],
        prototype_ref='calibration/prototypes.npy', calibration_ref='calibration/complete.json',
        thresholds_ref='calibration/thresholds.npy', predictions_ref='predictions.npz',
        model_classes=6, parameters=232993, model_forward_samples=sum(COUNTS.values()) + 2,
        model_forward_roles=dict(L_smoke=2, source_L=COUNTS['source_L'], source_V=COUNTS['source_V'],
            novel_validation=COUNTS['novel_validation']), training_forward_counts_separate=True,
        risk_scope='empirical risk on the same known-source calibration samples',
        independent_confirmation=False, source_novel_evaluation_reused=True,
        evidence_scope='repeated source development; no formal target evidence', limits=risk['limits'],
        constraints_passed=risk['constraints_passed'])
    runner._write(output / 'complete.json', result)
    print('SOURCE_EMBEDDING_PREDICTIONS_COMPLETE', config['row_id'], len(ids), len(METHODS), flush=True)
    return result


run = evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = evaluate(runner._read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], count=result['count'], arm=result['arm'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
