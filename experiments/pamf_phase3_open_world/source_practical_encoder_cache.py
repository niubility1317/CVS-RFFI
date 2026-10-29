"""Immutable clean/frozen-DURA computational IQ views for source L and OE21.

This cache contains no source V, original U, source novel or target records.
The clean view copies already-normalized source IQ; no RMS/EQ is reapplied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import development_transfer as frozen
from . import source_practical_export as common
from . import source_open_embedding_train as oldnative
from . import source_diverse_export as diverse
from . import source_open_export as original


RUN_ID = '20261029-phase3-source-practical-encoder-s2026092701-r01'
SCHEMA = 'pamf_phase3_source_practical_encoder_cache_v1'
COUNTS = dict(source_L=6300, source_OE=6134)
VIEWS = common.VIEWS
FIELDS = {'run_id', 'code_commit', 'source_row_root', 'risk_prediction_root', 'diverse_data_root',
          'dura_checkpoint', 'dura_release', 'output_root', 'batch_size'}
METADATA_FIELDS = dict(source_L={'y', 'domain', 'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index'},
                       source_OE={'rx_i', 'day_i', 'domain'})
ZERO_READS = ('source_V_reads', 'novel_validation_reads', 'original_U_s_reads', 'target_reads', 'OE_truth_reads')
ZERO_UPDATES = ('optimizer_updates', 'encoder_updates', 'identity_head_updates', 'DURA_updates',
                'readout_updates', 'threshold_calibrations')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only the nine fixed source-L/OE21 cache inputs are allowed')
    if any(not isinstance(config[key], str) or not config[key].strip() for key in FIELDS - {'batch_size'}):
        raise ValueError('Immutable original source, OE21 and frozen model paths are required')
    if (config['run_id'] != RUN_ID or not re.fullmatch('[0-9a-f]{40}', config['code_commit'])
            or type(config['batch_size']) is not int or config['batch_size'] != 128):
        raise ValueError('Fixed source-practical-encoder run, actual commit and batch128 required')
    row, risk = Path(config['source_row_root']), Path(config['risk_prediction_root'])
    if (row.name != 'uniform_oe' or row.parent.name != 'rows' or row.parent.parent.name != frozen.READOUT_RUN
            or risk.name != 'predict' or risk.parent.name != 'uniform_oe' or risk.parent.parent.name != 'rows'
            or risk.parent.parent.parent.name != frozen.RISK_RUN):
        raise ValueError('Only the original E51 uniform-OE row and original baseline-risk state are allowed')
    output = Path(config['output_root']).resolve()
    if any(output.is_relative_to(Path(config[key]).resolve()) for key in
           ('source_row_root', 'risk_prediction_root', 'diverse_data_root', 'dura_release')):
        raise ValueError('Cache output cannot modify a frozen source/code input root')
    return config


def _refs():
    return {role: dict(ids=role + '/ids.npy', metadata=role + '/metadata.npz',
                       views={view: role + '/' + view + '.npy' for view in VIEWS}) for role in COUNTS}


def _dura_identity(config):
    return dict(deployment_id='dura_v1', checkpoint=config['dura_checkpoint'], release_root=config['dura_release'],
        arm='stage2', epoch=20, updates=4940, training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f',
        parameters=266951, reuse=True, graph=True, native=False, precision='FP32', TF32=False)


def _contract_handle(contract):
    return {key: contract[key] for key in ('classes', 'source_rxs', 'source_days', 'split_seed', 'counts')}


def _oe_handle(handle):
    if (not isinstance(handle, dict) or set(handle) != {'capsule_id', 'split_id', 'count'}
            or handle.get('capsule_id') != diverse.CAPSULE_ID or handle.get('split_id') != diverse.SPLIT_ID
            or type(handle.get('count')) is not int or handle['count'] != COUNTS['source_OE']):
        raise ValueError('Only the existing VALIDATED_ONCE OE21 capsule/split/count may be reused')
    return handle


def _metadata(meta, role):
    """Validate this compact cache's fields, not the original source IQ again."""
    count = COUNTS[role]
    if set(meta) != METADATA_FIELDS[role] | {'ids'}:
        raise ValueError('Source cache metadata contains missing or forbidden label fields')
    ids = meta['ids']
    if (ids.shape != (count,) or ids.dtype.kind != 'U' or np.any(np.char.strip(ids) == '')
            or len(set(ids.tolist())) != count):
        raise ValueError('Aligned unique original source physical IDs required')
    for key in METADATA_FIELDS[role]:
        value = meta[key]
        if value.shape != (count,) or value.dtype != np.int64:
            raise ValueError('Source cache metadata must retain aligned int64 fields: ' + key)
    rx, day = meta['rx_i'], meta['day_i']
    if not np.isin(rx, list(common.RX_NAMES)).all() or not np.isin(day, list(common.DAYS)).all():
        raise ValueError('Only the original five source receivers and three source days are allowed')
    expected = np.searchsorted(list(common.RX_NAMES), rx) * 3 + day - 1
    if (not np.array_equal(meta['domain'], expected)
            or set(np.unique(meta['domain']).tolist()) != set(range(15))):
        raise ValueError('Original fifteen source RX/day domains must retain their mapping and coverage')
    if role == 'source_L' and (set(meta['y'].tolist()) != set(range(6)) or not bool((meta['eq_i'] == 1).all())):
        raise ValueError('Source L must retain original six labels and eq1 representation')


def _load_l(config, source, contract):
    labeled = oldnative._load_labeled(source['train_config'], contract)
    if labeled.role != 'L_s' or set(labeled.metadata) != METADATA_FIELDS['source_L'] | {'ids'}:
        raise ValueError('Only the complete original native L_s metadata is allowed')
    # This existing helper checks original IDs/order, six known labels, eq1 and
    # domain mapping. It drops native sig/base/eq fields, so retain those from
    # the dataset rather than inventing them or using _source_meta as storage.
    original._source_meta(labeled, 'L_s', contract)
    meta = {key: np.asarray(labeled.metadata[key], dtype=np.int64).copy() for key in METADATA_FIELDS['source_L']}
    meta['ids'] = np.asarray(labeled.metadata['ids']).copy()
    _metadata(meta, 'source_L')
    return labeled.x, meta


def _view(raw, meta, view, channels, api, sessions, engine, device):
    import torch
    current = common._raw(raw)
    if view == 'clean':
        return current
    if view not in VIEWS[1:]:
        raise ValueError('Only the four declared source computational views are allowed')
    received = common._synthesize(current, meta, view, channels, api, sessions)
    # The frozen engine internally uses inference mode. Clone its result after
    # that call returns, so a later trainable F can consume an ordinary tensor.
    with torch.no_grad():
        restored = engine(original._tensor(received, device))[0].detach().clone()
    if tuple(restored.shape) != current.shape:
        raise ValueError('Frozen DURA must preserve every source row and all 256 IQ points')
    return common._raw(restored)


def _save_role(output, role, raw, meta, config, channels, api, sessions, engine, device):
    count = COUNTS[role]
    _metadata(meta, role)
    if len(raw) != count:
        raise ValueError('Source role IQ and opaque IDs must have the same complete row count')
    folder = output / role
    folder.mkdir(exist_ok=False)
    with (folder / 'ids.npy').open('xb') as stream:
        np.save(stream, meta['ids'], allow_pickle=False)
    with (folder / 'metadata.npz').open('xb') as stream:
        np.savez(stream, **{key: meta[key] for key in METADATA_FIELDS[role]})
    for view in VIEWS:
        mapped = np.lib.format.open_memmap(folder / (view + '.npy'), mode='w+', dtype=np.float32, shape=(count, 2, 256))
        try:
            for start in range(0, count, config['batch_size']):
                stop = min(start + config['batch_size'], count)
                # Scene/seed generation only receives opaque ID and RX/day.
                # Known labels, role names and OE membership never select it.
                current_meta = {key: meta[key][start:stop] for key in ('ids', 'rx_i', 'day_i')}
                mapped[start:stop] = _view(raw[start:stop], current_meta, view, channels, api, sessions, engine, device)
                if start == 0 or start % (config['batch_size'] * 32) == 0 or stop == count:
                    print('SOURCE_PRACTICAL_ENCODER_CACHE', role, view, stop, count, flush=True)
        finally:
            mapped.flush()
            mapped._mmap.close()


def build(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('A new immutable source practical encoder cache root is required')
    api = common._canonical_channel_api()
    channels = {view: common._channel_config(view) for view in VIEWS[1:]}
    source = frozen._source_bundle(config)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader, baseline = frozen._load_readers(source, device)
    dura, engine = frozen._load_dura(source, config, device)
    contract = common._source_contract(source)
    oe_handle = _oe_handle(diverse._diverse_handle(config['diverse_data_root']))
    models = [reader, baseline, dura]
    snapshots = [frozen._snapshot_model(model) for model in models]
    aggregate_before = frozen._array_snapshot({key: source[key] for key in ('geometry', 'state', 'thresholds')})
    sessions = {}
    smoke = common._smoke(reader, baseline, engine, source, channels, api, sessions, device)
    common._check_frozen(models, snapshots, source, aggregate_before)
    # No source role IQ is opened until the actual-checkpoint synthetic smoke
    # passes. It continues here immediately; no approval artifact is involved.
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'smoke.json', smoke)
    raw_l, meta_l = _load_l(config, source, contract)
    _save_role(output, 'source_L', raw_l, meta_l, config, channels, api, sessions, engine, device)
    del raw_l
    raw_oe, meta_oe = diverse._public(config['diverse_data_root'])
    try:
        meta_oe = {key: np.asarray(value, dtype=None if key == 'ids' else np.int64).copy()
                   for key, value in meta_oe.items()}
        _metadata(meta_oe, 'source_OE')
        if set(meta_l['ids'].tolist()) & set(meta_oe['ids'].tolist()):
            raise ValueError('Original L and source21 OE physical role IDs must remain disjoint')
        _save_role(output, 'source_OE', raw_oe, meta_oe, config, channels, api, sessions, engine, device)
    finally:
        if isinstance(raw_oe, np.memmap):
            raw_oe._mmap.close()
    common._check_frozen(models, snapshots, source, aggregate_before)
    physical = sum(COUNTS.values())
    result = dict(status='SOURCE_PRACTICAL_ENCODER_CACHE_COMPLETE', schema=SCHEMA,
        run_id=config['run_id'], code_commit=config['code_commit'], config=dict(config), source_only=True,
        counts=dict(COUNTS), view_order=list(VIEWS), data_refs=_refs(),
        encoder_state_identity=source['marker']['encoder_state_identity'],
        parent_export_identity=source['marker']['parent_export_identity'],
        source_contract_handle=_contract_handle(contract), source_OE_handle=dict(oe_handle), dura_identity=_dura_identity(config),
        physical_record_count=physical, derived_view_record_count=physical * 4,
        new_independent_physical_records=0, same_physical_ID_order_across_views=True,
        metadata_fields={role: sorted(fields) for role, fields in METADATA_FIELDS.items()},
        model_forward_rows=dict(E51=8, original_F=8, DURA=physical * 3 + 6),
        cache_DURA_forward_rows=physical * 3, cache_channel_rows=physical * 3,
        source_state_unchanged=True, DURA_state_unchanged=True, second_RMS_applied=False, original_EQ_reapplied=False,
        raw_branches_written=False, logits_written=False, precision='FP32', TF32=False,
        channel_configs={view: asdict(cfg) for view, cfg in channels.items()},
        seeds=dict(receiver=common.RECEIVER_SEED, source_channel=common.CHANNEL_SEED),
        channel_seed_rule='stable_seed(source_channel_seed, scene, opaqueID); identical across both roles',
        receiver_session_rule='physical source RX/date/1; first-scene receiver reused across three scenes',
        canonical_channel_code=str(Path(__file__).resolve().parents[2] / 'code/leo_practical'),
        validation_reused='Existing VALIDATED_ONCE source roles and source21 capsule; no raw-data revalidation',
        smoke=smoke, interpretation='Source-only computational IQ views; not additional independent observations')
    result.update({key: 0 for key in ZERO_READS + ZERO_UPDATES})
    _write(output / 'complete.json', result)
    return result


def cache_marker(root):
    try:
        marker = _read(Path(root) / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Complete immutable source-L/OE21 cache required; partial output cannot be reused') from exc
    config = marker.get('config')
    validate_config(config)
    handle = marker.get('source_contract_handle', {})
    if (marker.get('status') != 'SOURCE_PRACTICAL_ENCODER_CACHE_COMPLETE' or marker.get('schema') != SCHEMA
            or marker.get('run_id') != config['run_id'] or marker.get('code_commit') != config['code_commit']
            or marker.get('counts') != COUNTS or marker.get('view_order') != list(VIEWS) or marker.get('data_refs') != _refs()
            or marker.get('source_only') is not True or marker.get('source_state_unchanged') is not True
            or marker.get('DURA_state_unchanged') is not True or marker.get('same_physical_ID_order_across_views') is not True
            or marker.get('physical_record_count') != sum(COUNTS.values())
            or marker.get('derived_view_record_count') != sum(COUNTS.values()) * 4
            or marker.get('metadata_fields') != {role: sorted(fields) for role, fields in METADATA_FIELDS.items()}
            or marker.get('dura_identity') != _dura_identity(config)
            or handle.get('classes') != oldnative.CLASSES or handle.get('source_rxs') != oldnative.RX
            or handle.get('source_days') != [1, 2, 3] or handle.get('split_seed') != 392005
            or handle.get('counts') != dict(L_s=COUNTS['source_L'], U_s=56700, V=27000)
            or any(not isinstance(marker.get(key), dict) or not marker[key] for key in
                   ('encoder_state_identity', 'parent_export_identity'))
            or any(marker.get(key) != 0 for key in ZERO_READS + ZERO_UPDATES)):
        raise ValueError('Source cache identity, fixed roles/views, metadata or frozen-state boundary differs')
    _oe_handle(marker.get('source_OE_handle'))
    return marker


def load_role(root, role, views=VIEWS):
    """Return read-only per-view IQ memmaps and legal metadata including IDs."""
    if (role not in COUNTS or not isinstance(views, (list, tuple)) or not views
            or any(view not in VIEWS for view in views) or len(set(views)) != len(views)):
        raise ValueError('Only source_L/source_OE and unique declared computational views may be read')
    marker = cache_marker(root)
    refs = marker['data_refs'][role]
    ids = np.load(Path(root) / refs['ids'], allow_pickle=False)
    with np.load(Path(root) / refs['metadata'], allow_pickle=False) as stored:
        if set(stored.files) != METADATA_FIELDS[role]:
            raise ValueError('Source cache metadata includes missing or forbidden fields')
        meta = {key: stored[key].copy() for key in stored.files}
    meta['ids'] = ids
    _metadata(meta, role)
    arrays = {}
    try:
        for view in views:
            mapped = np.load(Path(root) / refs['views'][view], allow_pickle=False, mmap_mode='r')
            arrays[view] = mapped
            if mapped.shape != (COUNTS[role], 2, 256) or mapped.dtype != np.float32:
                raise ValueError('Source cache IQ must retain complete FP32[N,2,256] views')
    except BaseException:
        for mapped in arrays.values():
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()
        raise
    return arrays, meta


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_PRACTICAL_ENCODER_CACHE_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = build(_read(args.config), args.device)
    print(json.dumps({key: result[key] for key in ('status', 'physical_record_count', 'derived_view_record_count')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
