"""One G10 observation of each already-selected source-heldout novel record.

Only the parent completion/mapping, opaque IDs and domain metadata are reused.
Old normalized IQ, old truth, OE training records, models and target data are
never read. This pool is source-heldout validation, not a target input pool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import re
import sys
import zipfile

import numpy as np

from . import source_oe_data as parent_data
from .data import _identity
from .full_pool import _read_json, _read_numpy_pickle, _write_json


RUN_ID = '20261029-phase3-star-source-novel-g10-r01'
SCHEMA = 'pamf_phase3_star_source_novel_builder_v1'
PUBLIC_SCHEMA = 'phase3_star_source_novel_min_v1'
COUNT = 3150
EPISODE = dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001)
FIELDS = frozenset({'run_id', 'parent_root', 'full_root', 'mixture_config_path', 'episode', 'code_commit'})
PUBLIC_FILES = frozenset({'manifest.json', 'ids.npy', 'received.npy', 'metadata.npz'})
PUBLIC_FIELDS = frozenset({'schema', 'data_status', 'capsule_id', 'split_id', 'count', 'iq_shape', 'iq_dtype',
    'role', 'source_only', 'processing_route', 'mode', 'equalization_enabled', 'fs_hz', 'single_observation'})
METADATA_FIELDS = frozenset({'rx_i', 'day_i', 'domain'})


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != FIELDS or config['run_id'] != RUN_ID
            or any(not isinstance(config[k], str) or not config[k].strip()
                   for k in ('parent_root', 'full_root', 'mixture_config_path'))
            or not isinstance(config['code_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', config['code_commit'])
            or config['episode'] != EPISODE
            or any(type(config['episode'].get(k)) is not int for k in ('scene_seed', 'channel_seed'))):
        raise ValueError('Only the fixed source-novel run, six fields, code commit and MIX01 seeds are accepted')
    return config


def _ids(ids):
    if (ids.shape != (COUNT,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != COUNT
            or any(not re.fullmatch(r'p3oe-[0-9a-f]{24}', str(v)) for v in ids)):
        raise ValueError('The original 3150 distinct source OE opaque IDs are required')


def _metadata(metadata):
    if (set(metadata) != METADATA_FIELDS
            or any(a.dtype != np.dtype('int64') or a.shape != (COUNT,) for a in metadata.values())):
        raise ValueError('Source metadata permits only int64 rx_i/day_i/domain vectors')
    rx, day, domain = (metadata[k] for k in ('rx_i', 'day_i', 'domain'))
    if (not np.isin(rx, parent_data.MAIN_RX).all() or not np.isin(day, [1, 2, 3]).all()
            or not np.array_equal(domain, np.searchsorted(parent_data.MAIN_RX, rx)*3 + day - 1)
            or not np.array_equal(np.bincount(domain, minlength=15), np.full(15, 210))):
        raise ValueError('The fixed five Main source RX indices and fifteen RX/day domains differ')


def _parent(config):
    root = Path(config['parent_root'])
    complete, report = _read_json(root/'complete.json'), _read_json(root/'builder_report.json')
    if (not isinstance(complete, dict) or complete.get('status') != 'SOURCE_OE_DATA_VALIDATED'
            or complete.get('data_status') != 'VALIDATED_ONCE' or complete.get('source_only') is not True
            or complete.get('novel_validation_count') != COUNT
            or complete.get('counts', {}).get('novel_validation') != COUNT
            or complete.get('canonical_physical_unique') is not True
            or any(complete.get(k) != 0 for k in ('target_reads', 'original_L_s_reads', 'original_U_s_reads',
                                                 'original_V_reads', 'model_reads', 'model_predictions', 'optimizer_updates'))):
        raise ValueError('The completed VALIDATED_ONCE source-only parent is required')
    previous = parent_data.validate_config(complete.get('config'))
    if (not isinstance(report, dict) or report.get('schema') != 'pamf_phase3_source_oe_data_v1'
            or report.get('config') != previous or report.get('canonical_physical_unique') is not True
            or Path(previous['full_root']).resolve() != Path(config['full_root']).resolve()
            or not Path(config['full_root']).is_dir()):
        raise ValueError('Parent mapping and the previously used Full handle must match')
    handle = complete.get('novel_validation', {})
    public = root/'novel_validation/public'
    if (not isinstance(handle, dict) or handle.get('count') != COUNT
            or handle.get('data_status') != 'VALIDATED_ONCE' or handle.get('iq_shape') != [COUNT, 2, 256]
            or handle.get('iq_dtype') != 'float32'
            or any(not isinstance(handle.get(k), str) or not handle[k] for k in ('capsule_id', 'split_id'))
            or Path(handle.get('public_root', '')).resolve() != public.resolve()):
        raise ValueError('The original heldout source capsule/split handle is missing or incompatible')
    # No parent IQ or truth file is opened, including when one is malformed.
    ids = np.load(public/'ids.npy', allow_pickle=False)
    with np.load(public/'metadata.npz', allow_pickle=False) as stored:
        metadata = {k: stored[k] for k in stored.files}
    _ids(ids); _metadata(metadata)
    members, locations = report.get('members'), {}
    if not isinstance(members, list) or len(members) != 15:
        raise ValueError('Exactly the fifteen already-indexed source members are required')
    for item in members:
        if not isinstance(item, dict):
            raise ValueError('An existing source member locator is malformed')
        rx, day, archive = item.get('receiver'), item.get('date'), item.get('archive')
        if (rx not in parent_data.RECEIVERS or day not in parent_data.DATES or (rx, day) in locations
                or not isinstance(archive, str) or Path(archive).name != archive
                or not archive.startswith('pkl_wifi_eq_'+day+'-') or not archive.endswith('.zip')
                or item.get('member') != f'pkl_wifi_eq_{day}/dataset_{day}_node{rx}.pkl'):
            raise ValueError('An existing source RX/date member locator differs')
        locations[rx, day] = item
    records = report.get('mapping', {}).get('novel_validation')
    if not isinstance(records, list) or len(records) != COUNT:
        raise ValueError('The exact original source novel mapping is required')
    physical, cells = set(), Counter()
    for i, (identifier, row) in enumerate(zip(ids, records)):
        canonical = row.get('canonical_record') if isinstance(row, dict) else None
        if (not isinstance(canonical, list) or len(canonical) != 6 or canonical[0] != 'FullWiSigEq'
                or canonical[1] not in parent_data.NOVEL or canonical[2] not in parent_data.RECEIVERS
                or canonical[3] not in parent_data.DATES or type(canonical[4]) is not int or canonical[4] != 1
                or type(canonical[5]) is not int or canonical[5] < 0
                or row.get('opaque_id') != str(identifier)
                or str(identifier) != 'p3oe-'+_identity(['source_oe_physical', canonical])):
            raise ValueError('Original opaque ID, order or source-only canonical record differs')
        _, tx, rx, day, _, _ = canonical
        location = locations[rx, day]
        if (row.get('source_archive') != location['archive'] or row.get('source_member') != location['member']
                or tuple(canonical) in physical
                or metadata['rx_i'][i] != parent_data.MAIN_RX[parent_data.RECEIVERS.index(rx)]
                or metadata['day_i'][i] != parent_data.DATES.index(day)+1):
            raise ValueError('Original physical uniqueness, member or public domain alignment differs')
        physical.add(tuple(canonical)); cells[tx, rx, day] += 1
    expected = {(tx, rx, day): 70 for tx in parent_data.NOVEL
                for rx in parent_data.RECEIVERS for day in parent_data.DATES}
    if cells != expected:
        raise ValueError('Retain all original 3150 records; no resampling or role changes')
    ordered = [locations[rx, day] for day in parent_data.DATES for rx in parent_data.RECEIVERS]
    return handle, records, ids, metadata, ordered


def _read_member(config, location, records):
    """Read one indexed member and copy only its exact mapped raw packet rows."""
    try:
        with zipfile.ZipFile(Path(config['full_root'])/location['archive']) as stored:
            with stored.open(location['member'], 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (OSError, KeyError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('An already-indexed source Full member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full source member must contain only data/node_list')
    names, arrays = payload['node_list'], payload['data']
    if (not isinstance(names, list) or not isinstance(arrays, list) or len(names) != len(arrays)
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError('Full source node_list/data association differs')
    lookup, selected = dict(zip(names, arrays)), []
    for row in records:
        _, tx, _, _, _, index = row['canonical_record']
        values = lookup.get(tx)
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
                or values.ndim != 3 or values.shape[1:] != (256, 2) or index >= len(values)):
            raise ValueError('An original mapped source packet index, dtype or shape differs')
        raw = values[index].T.copy()
        if not np.isfinite(raw).all() or np.max(np.abs(raw)) > np.finfo(np.float32).max:
            raise ValueError('A selected source raw packet is nonfinite or overflows FP32')
        selected.append(raw)
    return selected


def _observe(raw, **kwargs):
    from .star_channel import observe
    return observe(raw, **kwargs)


def _save(path, value):
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def _save_npz(path, **values):
    with path.open('xb') as stream:
        np.savez(stream, **values)


def load_public(public_root, cap, split):
    """Load only the four public files with read-only received NPY mapping.

    This is a lightweight source-heldout handle/array-header check. It does
    not revisit Full, parent records, private truth or channel validation.
    """
    public = Path(public_root)
    if {p.name for p in public.iterdir()} != PUBLIC_FILES:
        raise ValueError('Source-heldout public input permits exactly four public files')
    manifest = _read_json(public/'manifest.json')
    required = dict(schema=PUBLIC_SCHEMA, data_status='VALIDATED_ONCE', count=COUNT,
        iq_shape=[COUNT, 2, 256], iq_dtype='float32', role='source_heldout_validation', source_only=True,
        processing_route='residual', mode='post_sync', equalization_enabled=False,
        fs_hz=25000000, single_observation=True, capsule_id=cap, split_id=split)
    if (not isinstance(manifest, dict) or set(manifest) != PUBLIC_FIELDS
            or any(manifest.get(k) != v for k, v in required.items())
            or any(not isinstance(value, str) or not value for value in (cap, split))
            or manifest['source_only'] is not True or manifest['single_observation'] is not True
            or manifest['equalization_enabled'] is not False):
        raise ValueError('Source-heldout public manifest/capsule/split contract differs')
    ids = np.load(public/'ids.npy', allow_pickle=False)
    with np.load(public/'metadata.npz', allow_pickle=False) as stored:
        metadata = {k: stored[k] for k in stored.files}
    _ids(ids); _metadata(metadata)
    received = np.load(public/'received.npy', mmap_mode='r', allow_pickle=False)
    if not isinstance(received, np.memmap) or received.shape != (COUNT, 2, 256) or received.dtype != np.dtype('float32'):
        if isinstance(received, np.memmap):
            received._mmap.close()
        raise ValueError('Source-heldout received must be a read-only FP32[3150,2,256] NPY mapping')
    return manifest, ids, received, metadata


def build(config, output_root):
    """CPU-only exact-record G10 builder; immutable output, completion last."""
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Source G10 output cannot overwrite or resume history: '+str(output))
    handle, records, ids, metadata, members = _parent(config)
    from .star_channel import identity, validate_mixture, SCENARIOS
    mixture = validate_mixture(_read_json(config['mixture_config_path']))
    output.mkdir(parents=True, exist_ok=False)
    public, truth = output/'public', output/'truth'
    public.mkdir(); truth.mkdir()
    received = np.lib.format.open_memmap(public/'received.npy', mode='w+', dtype=np.float32, shape=(COUNT, 2, 256))
    scenarios, mapping = np.empty(COUNT, dtype='U32'), [None]*COUNT
    try:
        for number, location in enumerate(members, start=1):
            indices = [i for i, row in enumerate(records)
                       if row['canonical_record'][2:4] == [location['receiver'], location['date']]]
            selected = [records[i] for i in indices]
            raw = _read_member(config, location, selected)
            for i, row, value in zip(indices, selected, raw):
                canonical = row['canonical_record']
                _, tx, rx, day, _, index = canonical
                physical_id = identity(['WiSig/Full/eq1', tx, rx, day, index])
                observation, meta = _observe(value, physical_id=physical_id, receiver=rx, day=day,
                                             episode=config['episode'], mixture=mixture)
                if (not isinstance(observation, np.ndarray) or observation.dtype != np.dtype('float32')
                        or observation.shape != (2, 256) or not np.isfinite(observation).all()
                        or not isinstance(meta, dict) or set(meta) != {'scenario', 'overlay_seed', 'input_rms', 'geometry'}
                        or meta['scenario'] not in {s[0] for s in SCENARIOS}):
                    raise ValueError('One finite FP32[2,256] G10 observation and private diagnostics required')
                received[i] = observation; scenarios[i] = meta['scenario']
                mapping[i] = dict(**row, physical_id=physical_id, **meta)
            print('STAR_SOURCE_NOVEL_PROGRESS', number, 15, location['receiver'], location['date'], number*210, COUNT, flush=True)
        received.flush()
    finally:
        received._mmap.close()
    if any(row is None for row in mapping):
        raise ValueError('Every original source-heldout ID must receive exactly one observation')
    capsule = 'p3source-novel-g10-'+_identity([RUN_ID, config['episode'], ids.tolist()])
    split = _identity([RUN_ID, 'source_heldout_validation', capsule])
    manifest = dict(schema=PUBLIC_SCHEMA, data_status='VALIDATED_ONCE', capsule_id=capsule, split_id=split,
        count=COUNT, iq_shape=[COUNT, 2, 256], iq_dtype='float32', role='source_heldout_validation', source_only=True,
        processing_route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000, single_observation=True)
    _save(public/'ids.npy', ids); _save_npz(public/'metadata.npz', **metadata)
    _save_npz(truth/'truth.npz', ids=ids, tx=np.asarray([r['canonical_record'][1] for r in records]),
              rx_i=metadata['rx_i'], day_i=metadata['day_i'], scenario=scenarios)
    _write_json(public/'manifest.json', manifest)
    _, _, checked, _ = load_public(public, capsule, split)
    checked._mmap.close()
    counters = dict(target_reads=0, formal_target_reads=0, exposure_train_reads=0, original_L_s_reads=0,
        original_U_s_reads=0, original_V_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0)
    parent_handle = {k: handle[k] for k in ('capsule_id', 'split_id', 'count', 'data_status')}
    report = dict(schema=SCHEMA, config=config, source_only=True, source_heldout_validation=True,
        count=COUNT, canonical_physical_unique=True, opaque_ID_order_unchanged=True, parent_handle=parent_handle,
        members=members, mapping=mapping, archive_reads=15, pickle_reads=15,
        full_archive_rescan=False, archive_revalidation=False, old_normalized_IQ_read=False, old_truth_read=False,
        new_observation_count=COUNT, input_RMS_normalizations_per_record=1, output_RMS_normalizations_per_record=0,
        equalization_reapplied=False, source_rx_indices=parent_data.MAIN_RX, source_dates=parent_data.DATES,
        receiver_seed=2027, all_new_IQ_satellite=True, whole_project_never_exposed_claim=False, **counters)
    _write_json(output/'builder_report.json', report)
    complete = dict(status='STAR_SOURCE_NOVEL_VALIDATED', schema=SCHEMA, data_status='VALIDATED_ONCE',
        run_id=RUN_ID, code_commit=config['code_commit'], config=config, source_only=True, source_heldout_validation=True,
        count=COUNT, unique_canonical_count=COUNT, new_observation_count=COUNT, canonical_physical_unique=True,
        capsule_id=capsule, split_id=split, public_root=str(public), truth_path=str(truth/'truth.npz'),
        parent_handle=parent_handle, single_observation=True, all_new_IQ_satellite=True, **counters)
    _write_json(output/'complete.json', complete)
    print('STAR_SOURCE_NOVEL_VALIDATED', COUNT, flush=True)
    return complete


def main(argv=None):
    actual = list(sys.argv[1:] if argv is None else argv)
    print(json.dumps(dict(pid=os.getpid(), cwd=os.getcwd(), argv=actual), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(actual)
    done = build(_read_json(Path(args.config)), args.output)
    print(json.dumps(dict(status=done['status'], count=done['count'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
