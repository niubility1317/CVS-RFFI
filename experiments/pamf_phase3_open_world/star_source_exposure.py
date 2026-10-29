"""G10 observations of exactly the completed 6134 source OE training records.

The public supervision is membership outside the registered six only. Real TX
identities remain in the offline builder report; no training truth is emitted.
Existing normalized IQ, truth, novel validation, target data and models are not
read. Physical overlap with historical OE4 is retained, not called independent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import source_oe_data as source, source_oe_diverse_data as parent_data
from .data import _identity
from .full_pool import _read_json, _write_json
from .star_source_novel import _read_member, _save, _save_npz, PUBLIC_FILES


RUN_ID = '20261029-phase3-star-source-exposure-g10-r01'
SCHEMA = 'pamf_phase3_star_source_exposure_builder_v1'
PUBLIC_SCHEMA = 'phase3_star_source_exposure_min_v1'
COUNT = 6134
EPISODE = dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001)
FIELDS = frozenset({'run_id', 'parent_root', 'full_root', 'mixture_config_path', 'episode', 'code_commit'})
PUBLIC_FIELDS = frozenset({'schema', 'data_status', 'capsule_id', 'split_id', 'count', 'iq_shape', 'iq_dtype',
    'role', 'source_only', 'outside_registered_six', 'processing_route', 'mode', 'equalization_enabled',
    'fs_hz', 'single_observation'})
METADATA_FIELDS = frozenset({'rx_i', 'day_i', 'domain'})


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != FIELDS or config['run_id'] != RUN_ID
            or any(not isinstance(config[k], str) or not config[k].strip()
                   for k in ('parent_root', 'full_root', 'mixture_config_path'))
            or not isinstance(config['code_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', config['code_commit'])
            or config['episode'] != EPISODE
            or any(type(config['episode'].get(k)) is not int for k in ('scene_seed', 'channel_seed'))):
        raise ValueError('Only the fixed source-exposure run, six fields, code commit and MIX01 seeds are accepted')
    return config


def _ids(ids):
    if (ids.shape != (COUNT,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != COUNT
            or any(not re.fullmatch(r'p3oe-[0-9a-f]{24}', str(v)) for v in ids)):
        raise ValueError('The original 6134 distinct source OE opaque IDs are required')


def _metadata(metadata):
    if (set(metadata) != METADATA_FIELDS
            or any(a.dtype != np.dtype('int64') or a.shape != (COUNT,) for a in metadata.values())):
        raise ValueError('Source metadata permits only int64 rx_i/day_i/domain vectors')
    rx, day, domain = (metadata[k] for k in ('rx_i', 'day_i', 'domain'))
    if (not np.isin(rx, source.MAIN_RX).all() or not np.isin(day, [1, 2, 3]).all()
            or not np.array_equal(domain, np.searchsorted(source.MAIN_RX, rx)*3 + day-1)
            or not (np.bincount(domain, minlength=15) > 0).all()):
        raise ValueError('The existing five source RX and fifteen nonuniform RX/day domains must be retained')


def _allocation(report, actual):
    """Check the existing mapping against its allocation; never select records."""
    allocation = report.get('allocation')
    if (not isinstance(allocation, dict) or allocation.get('count') != COUNT
            or allocation.get('nonempty_cells') != 220 or allocation.get('zero_cells') != 95
            or not isinstance(allocation.get('cells'), list) or len(allocation['cells']) != 315):
        raise ValueError('The completed source allocation must describe 6134 original records')
    expected = {}
    for cell in allocation['cells']:
        if not isinstance(cell, dict):
            raise ValueError('An original source allocation cell is malformed')
        tx, rx, day = (cell.get(k) for k in ('tx', 'receiver', 'date'))
        available, selected = cell.get('available'), cell.get('selected')
        key = (tx, rx, day)
        if (tx not in parent_data.CLASSES or rx not in source.RECEIVERS or day not in source.DATES
                or key in expected or type(available) is not int or available < 0
                or type(selected) is not int or selected != min(available, 32)
                or actual[key] != selected):
            raise ValueError('Original per-cell counts differ; resampling or padding is forbidden')
        expected[key] = selected
    if sum(expected.values()) != COUNT or sum(v > 0 for v in expected.values()) != 220:
        raise ValueError('The original nonuniform source allocation differs')
    return allocation


def _parent(config):
    root = Path(config['parent_root'])
    complete, report = _read_json(root/'complete.json'), _read_json(root/'builder_report.json')
    if (not isinstance(complete, dict) or complete.get('status') != 'SOURCE_OE_DIVERSE_DATA_VALIDATED'
            or complete.get('data_status') != 'VALIDATED_ONCE' or complete.get('source_only') is not True
            or complete.get('count') != COUNT or complete.get('exposure_train_count') != COUNT
            or complete.get('counts', {}).get('exposure_train') != COUNT
            or complete.get('canonical_physical_unique') is not True or complete.get('unique_canonical_count') != COUNT
            or any(complete.get(k) != 0 for k in ('target_reads', 'original_L_s_reads', 'original_U_s_reads',
                        'original_V_reads', 'novel_validation_IQ_reads', 'model_reads', 'model_predictions', 'optimizer_updates'))):
        raise ValueError('The completed VALIDATED_ONCE source-only exposure parent is required')
    if not isinstance(report, dict) or report.get('schema') != 'phase3_source_oe_diverse_builder_v1':
        raise ValueError('The existing source exposure mapping report is required')
    # This parent stores its config in the report, not complete.json.
    previous = parent_data.validate_config(report.get('config'))
    if (Path(previous['full_root']).resolve() != Path(config['full_root']).resolve()
            or not Path(config['full_root']).is_dir()):
        raise ValueError('Reuse the original Full root; no Full revalidation is performed')
    public, handle = root/'exposure_train/public', complete.get('exposure_train', {})
    if (not isinstance(handle, dict) or handle.get('count') != COUNT or handle.get('data_status') != 'VALIDATED_ONCE'
            or handle.get('iq_shape') != [COUNT, 2, 256] or handle.get('iq_dtype') != 'float32'
            or any(not isinstance(handle.get(k), str) or not handle[k] for k in ('capsule_id', 'split_id'))
            or Path(handle.get('public_root', '')).resolve() != public.resolve()):
        raise ValueError('The completed original exposure capsule/split handle differs')
    ids = np.load(public/'ids.npy', allow_pickle=False)
    with np.load(public/'metadata.npz', allow_pickle=False) as stored:
        metadata = {k: stored[k] for k in stored.files}
    _ids(ids); _metadata(metadata)
    locations, members = {}, report.get('members')
    if not isinstance(members, list) or len(members) != 15:
        raise ValueError('Exactly fifteen already-indexed source member locators are required')
    for item in members:
        if not isinstance(item, dict):
            raise ValueError('An existing source member locator is malformed')
        rx, day, archive = item.get('receiver'), item.get('date'), item.get('archive')
        if (rx not in source.RECEIVERS or day not in source.DATES or (rx, day) in locations
                or not isinstance(archive, str) or Path(archive).name != archive
                or not archive.startswith('pkl_wifi_eq_'+day+'-') or not archive.endswith('.zip')
                or item.get('member') != f'pkl_wifi_eq_{day}/dataset_{day}_node{rx}.pkl'):
            raise ValueError('An original source member RX/date locator differs')
        locations[rx, day] = item
    records = report.get('mapping')
    if not isinstance(records, list) or len(records) != COUNT:
        raise ValueError('The original flat 6134-record exposure mapping is required')
    physical, cells = set(), Counter()
    for i, (identifier, row) in enumerate(zip(ids, records)):
        canonical = row.get('canonical_record') if isinstance(row, dict) else None
        if (not isinstance(canonical, list) or len(canonical) != 6 or canonical[0] != 'FullWiSigEq'
                or canonical[1] not in parent_data.CLASSES or canonical[2] not in source.RECEIVERS
                or canonical[3] not in source.DATES or type(canonical[4]) is not int or canonical[4] != 1
                or type(canonical[5]) is not int or canonical[5] < 0
                or row.get('opaque_id') != str(identifier)
                or str(identifier) != 'p3oe-'+_identity(['source_oe_physical', canonical])):
            raise ValueError('Original opaque ID, order or source-only canonical record differs')
        _, tx, rx, day, _, _ = canonical
        locator = locations[rx, day]
        if (tuple(canonical) in physical or row.get('source_archive') != locator['archive']
                or row.get('source_member') != locator['member']
                or metadata['rx_i'][i] != source.MAIN_RX[source.RECEIVERS.index(rx)]
                or metadata['day_i'][i] != source.DATES.index(day)+1):
            raise ValueError('Original physical uniqueness, indexed member or metadata alignment differs')
        physical.add(tuple(canonical)); cells[tx, rx, day] += 1
    allocation = _allocation(report, cells)
    ordered = [locations[rx, day] for day in source.DATES for rx in source.RECEIVERS]
    return handle, records, ids, metadata, ordered, allocation


def _observe(raw, **kwargs):
    from .star_channel import observe
    return observe(raw, **kwargs)


def load_public(public_root, cap, split):
    """Read only four source-auxiliary public files and map received read-only."""
    public = Path(public_root)
    if {p.name for p in public.iterdir()} != PUBLIC_FILES:
        raise ValueError('Source exposure public input permits exactly four public files')
    manifest = _read_json(public/'manifest.json')
    required = dict(schema=PUBLIC_SCHEMA, data_status='VALIDATED_ONCE', count=COUNT, iq_shape=[COUNT, 2, 256],
        iq_dtype='float32', role='source_auxiliary_training', source_only=True, outside_registered_six=True,
        processing_route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000,
        single_observation=True, capsule_id=cap, split_id=split)
    if (not isinstance(manifest, dict) or set(manifest) != PUBLIC_FIELDS
            or any(manifest.get(k) != v for k, v in required.items())
            or any(not isinstance(value, str) or not value for value in (cap, split))
            or any(manifest[k] is not True for k in ('source_only', 'outside_registered_six', 'single_observation'))
            or manifest['equalization_enabled'] is not False):
        raise ValueError('Source auxiliary public manifest/capsule/split contract differs')
    ids = np.load(public/'ids.npy', allow_pickle=False)
    with np.load(public/'metadata.npz', allow_pickle=False) as stored:
        metadata = {k: stored[k] for k in stored.files}
    _ids(ids); _metadata(metadata)
    received = np.load(public/'received.npy', mmap_mode='r', allow_pickle=False)
    if not isinstance(received, np.memmap) or received.shape != (COUNT, 2, 256) or received.dtype != np.dtype('float32'):
        if isinstance(received, np.memmap):
            received._mmap.close()
        raise ValueError('Source exposure received must be a read-only FP32[6134,2,256] NPY mapping')
    return manifest, ids, received, metadata


def build(config, output_root):
    """Generate one G10 received per original record, completion marker last."""
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Source G10 exposure output cannot overwrite or resume history: '+str(output))
    handle, records, ids, metadata, members, allocation = _parent(config)
    from .star_channel import identity, validate_mixture, SCENARIOS
    mixture = validate_mixture(_read_json(config['mixture_config_path']))
    output.mkdir(parents=True, exist_ok=False)
    public = output/'public'; public.mkdir()
    received = np.lib.format.open_memmap(public/'received.npy', mode='w+', dtype=np.float32, shape=(COUNT, 2, 256))
    mapping, generated = [None]*COUNT, 0
    try:
        for number, location in enumerate(members, start=1):
            indices = [i for i, row in enumerate(records)
                       if row['canonical_record'][2:4] == [location['receiver'], location['date']]]
            selected = [records[i] for i in indices]
            raw = _read_member(config, location, selected)
            for i, row, value in zip(indices, selected, raw):
                _, tx, rx, day, _, index = row['canonical_record']
                physical_id = identity(['WiSig/Full/eq1', tx, rx, day, index])
                observation, meta = _observe(value, physical_id=physical_id, receiver=rx, day=day,
                                             episode=config['episode'], mixture=mixture)
                if (not isinstance(observation, np.ndarray) or observation.dtype != np.dtype('float32')
                        or observation.shape != (2, 256) or not np.isfinite(observation).all()
                        or not isinstance(meta, dict) or set(meta) != {'scenario', 'overlay_seed', 'input_rms', 'geometry'}
                        or meta['scenario'] not in {s[0] for s in SCENARIOS}):
                    raise ValueError('One finite FP32[2,256] G10 observation and private diagnostics required')
                received[i] = observation
                mapping[i] = dict(**row, physical_id=physical_id, **meta)
                generated += 1
            print('STAR_SOURCE_EXPOSURE_PROGRESS', number, 15, location['receiver'], location['date'], generated, COUNT, flush=True)
        received.flush()
    finally:
        received._mmap.close()
    if generated != COUNT or any(row is None for row in mapping):
        raise ValueError('Every original source exposure ID must receive exactly one observation')
    capsule = 'p3source-exposure-g10-'+_identity([RUN_ID, config['episode'], ids.tolist()])
    split = _identity([RUN_ID, 'source_auxiliary_training', capsule])
    manifest = dict(schema=PUBLIC_SCHEMA, data_status='VALIDATED_ONCE', capsule_id=capsule, split_id=split,
        count=COUNT, iq_shape=[COUNT, 2, 256], iq_dtype='float32', role='source_auxiliary_training',
        source_only=True, outside_registered_six=True, processing_route='residual', mode='post_sync',
        equalization_enabled=False, fs_hz=25000000, single_observation=True)
    _save(public/'ids.npy', ids); _save_npz(public/'metadata.npz', **metadata)
    _write_json(public/'manifest.json', manifest)
    _, _, checked, _ = load_public(public, capsule, split); checked._mmap.close()
    counters = dict(target_reads=0, formal_target_reads=0, original_L_s_reads=0, original_U_s_reads=0,
        original_V_reads=0, novel_validation_IQ_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0)
    parent_handle = {k: handle[k] for k in ('capsule_id', 'split_id', 'count', 'data_status')}
    report = dict(schema=SCHEMA, config=config, source_only=True, count=COUNT, allocation=allocation,
        mapping=mapping, members=members, parent_handle=parent_handle, archive_reads=15, pickle_reads=15,
        full_archive_rescan=False, archive_revalidation=False, old_normalized_IQ_read=False, old_truth_read=False,
        samples_replicated=False, nonuniform_domain_counts_unchanged=True, opaque_ID_order_unchanged=True,
        canonical_physical_unique=True, new_observation_count=COUNT, input_RMS_normalizations_per_record=1,
        output_RMS_normalizations_per_record=0, equalization_reapplied=False, receiver_seed=2027,
        all_new_IQ_satellite=True, historical_old_OE4_physical_overlap_count=537, historical_overlap_recomputed=False,
        independent_from_previous_source_OE4=False, whole_project_never_exposed_claim=False, **counters)
    _write_json(output/'builder_report.json', report)
    complete = dict(status='STAR_SOURCE_EXPOSURE_VALIDATED', schema=SCHEMA, data_status='VALIDATED_ONCE',
        run_id=RUN_ID, code_commit=config['code_commit'], config=config, count=COUNT, unique_canonical_count=COUNT,
        new_observation_count=COUNT, source_only=True, outside_registered_six=True, role='source_auxiliary_training',
        capsule_id=capsule, split_id=split, public_root=str(public), parent_handle=parent_handle,
        canonical_physical_unique=True, single_observation=True, all_new_IQ_satellite=True, **counters)
    _write_json(output/'complete.json', complete)
    print('STAR_SOURCE_EXPOSURE_VALIDATED', COUNT, flush=True)
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
