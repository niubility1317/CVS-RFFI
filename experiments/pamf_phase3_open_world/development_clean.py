"""Materialize a clean control of the completed RX24-1 development mapping.

Only the builder opens Full records and private parent truth. Model consumers
use load_clean_public on one condition's three public files. This module never
loads the parent's received IQ, samples records, or generates a channel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

from .data import PUBLIC_FIELDS, PUBLIC_FILES, _identity, _validate_arrays
from . import development_pool as paired
from .full_pool import _read_json, _read_numpy_pickle, _write_json


ROLES = ('discovery', 'evaluation')
COUNTS = dict(discovery=384, evaluation=864)
SCHEMA = 'phase3_clean_min_v1'


def validate_config(config):
    """Paths locate the completed pair; there are no new sampling controls."""
    if (not isinstance(config, dict) or set(config) != {'full_root', 'paired_pool'}
            or any(not isinstance(value, str) or not value.strip() for value in config.values())):
        raise ValueError('Clean control requires only full_root and completed paired_pool paths')
    return config


def _manifest_contract(manifest, *, clean):
    count = manifest.get('count') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or set(manifest) != PUBLIC_FIELDS
            or type(count) is not int or count <= 0
            or manifest['schema'] != (SCHEMA if clean else 'phase3_min_v1')
            or manifest['data_status'] != 'VALIDATED_ONCE'
            or manifest['registered_classes'] != paired.REGISTERED
            or manifest['iq_shape'] != [count, 2, 256]
            or manifest['processing_route'] != ('clean' if clean else 'residual')
            or manifest['mode'] != ('clean' if clean else 'post_sync')
            or manifest['equalization_enabled'] is not False
            or manifest['single_observation'] is not True or manifest['fs_hz'] != 25000000
            or any(not isinstance(manifest[key], str) or not manifest[key]
                   for key in ('capsule_id', 'split_id'))):
        raise ValueError('The public clean/paired observation contract differs')
    return count


def _ids(ids, count):
    if (ids.ndim != 1 or ids.dtype.kind != 'U' or len(ids) != count
            or any(not str(value) for value in ids) or len(set(ids.tolist())) != count):
        raise ValueError('Opaque IDs must be a unique count-matched Unicode vector')


def load_clean_public(root):
    """Return (manifest, opaque IDs, read-only FP32 memmap) for clean only.

    The exact whitelist excludes parent paths, truth and per-record references.
    This intentionally does not relax data.load_public's residual contract.
    """
    root = Path(root)
    if {path.name for path in root.iterdir()} != PUBLIC_FILES:
        raise ValueError('Clean public input must contain only manifest.json, ids.npy and received.npy')
    manifest = _read_json(root / 'manifest.json')
    count = _manifest_contract(manifest, clean=True)
    ids = np.load(root / 'ids.npy', allow_pickle=False)
    iq = np.load(root / 'received.npy', mmap_mode='r', allow_pickle=False)
    try:
        if not isinstance(iq, np.memmap):
            raise ValueError('Clean received input must be an NPY memory map')
        _validate_arrays(ids, iq, count)
    except Exception:
        if isinstance(iq, np.memmap):
            iq._mmap.close()
        raise
    return manifest, ids, iq


def _parent(config):
    """Check the completed parent's handles and its already-selected mapping."""
    root = Path(config['paired_pool'])
    complete = _read_json(root / 'complete.json')
    report = _read_json(root / 'builder_report.json')
    if (not isinstance(complete, dict) or complete.get('status') != 'PHASE3_DEVELOPMENT_POOL_VALIDATED'
            or complete.get('data_status') != 'VALIDATED_ONCE'
            or complete.get('development_only') is not True
            or complete.get('discovery_count') != 384 or complete.get('evaluation_count') != 864
            or complete.get('unique_canonical_count') != 1248
            or complete.get('canonical_physical_unique') is not True
            or complete.get('single_observation') is not True
            or any(complete.get(key) != 0 for key in ('formal_target_reads', 'model_reads',
                                                     'model_predictions', 'optimizer_updates'))):
        raise ValueError('A completed validated RX24-1 development parent is required')
    if (not isinstance(report, dict) or report.get('schema') != 'pamf_phase3_development_builder_v1'
            or report.get('development_only') is not True
            or report.get('formal_target_reads') != 0
            or report.get('canonical_physical_unique') is not True
            or report.get('scene_and_role_canonical_disjoint') is not True):
        raise ValueError('The paired development builder report differs')
    parent_config = paired.validate_config(report.get('config'))
    if (Path(parent_config['full_root']).resolve() != Path(config['full_root']).resolve()
            or not Path(config['full_root']).is_dir()):
        raise ValueError('The Full root must match the completed paired parent')
    reused = report.get('validation_reused', {})
    if (reused.get('status') != 'VERIFIED' or reused.get('members_checked') != 158
            or reused.get('count_mismatch_cells') != 0):
        raise ValueError('Reuse the completed Full validation reported by the paired parent')
    mapping = report.get('mapping')
    if not isinstance(mapping, dict) or set(mapping) != set(ROLES):
        raise ValueError('Both completed parent role mappings are required')
    role_data, physical, all_ids = {}, set(), set()
    labels = {name: index for index, name in enumerate(paired.REGISTERED + paired.DISCOVERY + paired.FUTURE)}
    for role in ROLES:
        public = root / role / 'public'
        if {path.name for path in public.iterdir()} != PUBLIC_FILES:
            raise ValueError('The paired parent public whitelist differs')
        manifest = _read_json(public / 'manifest.json')
        count = _manifest_contract(manifest, clean=False)
        handle = complete.get(role, {})
        if (count != COUNTS[role] or handle.get('count') != count
                or any(handle.get(key) != manifest[key] for key in ('capsule_id', 'split_id'))):
            raise ValueError('The completed parent role and public handles differ')
        # Deliberately do not load parent received.npy: clean is not a model
        # reference paired into a star predictor, and star IQ is not re-used.
        ids = np.load(public / 'ids.npy', allow_pickle=False)
        _ids(ids, count)
        if all_ids & set(ids.tolist()):
            raise ValueError('Parent opaque IDs overlap between roles')
        all_ids.update(ids.tolist())
        records = mapping[role]
        if not isinstance(records, list) or len(records) != count:
            raise ValueError('Parent mapping count differs')
        day = paired.DAYS[role]
        archive, member = paired.MEMBERS[day]
        allowed_tx = paired.REGISTERED + paired.DISCOVERY + (paired.FUTURE if role == 'evaluation' else [])
        for opaque, record in zip(ids, records):
            canonical = record.get('canonical_record') if isinstance(record, dict) else None
            if (not isinstance(record, dict) or record.get('opaque_id') != str(opaque)
                    or not isinstance(canonical, list) or len(canonical) != 6
                    or canonical[0] != 'FullWiSigEq' or canonical[1] not in allowed_tx
                    or canonical[2] != '24-1' or canonical[3] != day
                    or type(canonical[4]) is not int or canonical[4] != 1
                    or type(canonical[5]) is not int or canonical[5] < 0
                    or record.get('source_archive') != archive or record.get('source_member') != member):
                raise ValueError('Parent IDs/order, canonical record or selected member differs')
            identity = tuple(canonical)
            if identity in physical:
                raise ValueError('Parent canonical records are duplicated across roles')
            physical.add(identity)
        with np.load(root / role / 'truth/truth.npz', allow_pickle=False) as stored:
            if set(stored.files) != {'ids', 'y', 'receiver', 'scenario'}:
                raise ValueError('Parent private truth fields differ')
            truth = {key: stored[key].copy() for key in stored.files}
        expected_y = np.asarray([labels[record['canonical_record'][1]] for record in records])
        if (not np.array_equal(truth['ids'], ids) or truth['y'].shape != (count,)
                or truth['y'].dtype.kind not in 'iu' or not np.array_equal(truth['y'], expected_y)
                or truth['receiver'].shape != (count,) or truth['receiver'].dtype.kind not in 'iu'
                or not bool((truth['receiver'] == 26).all()) or truth['scenario'].shape != (count,)):
            raise ValueError('Parent private truth does not match its public IDs and mapping')
        role_data[role] = dict(ids=ids, manifest=manifest, records=records, truth=truth)
    if len(physical) != 1248:
        raise ValueError('The paired control must retain exactly 1248 unique physical records')
    return complete, report, role_data


def _read_role(config, role, records):
    from experiments.phase1_external_dg.data import iq_record
    archive, member = paired.MEMBERS[paired.DAYS[role]]
    print('PHASE3_DEVELOPMENT_CLEAN_MEMBER', role, archive, member, flush=True)
    try:
        with zipfile.ZipFile(Path(config['full_root']) / archive) as stored:
            with stored.open(member, 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (OSError, KeyError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The mapped Full member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Mapped Full member must contain data/node_list only')
    names, arrays = payload['node_list'], payload['data']
    if (not isinstance(names, list) or not isinstance(arrays, list) or len(names) != len(arrays)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Mapped Full member has an invalid node_list/data association')
    lookup = dict(zip(names, arrays))
    received = np.empty((len(records), 2, 256), dtype=np.float32)
    for position, record in enumerate(records):
        _, transmitter, _, _, _, index = record['canonical_record']
        raw = lookup.get(transmitter)
        if (not isinstance(raw, np.ndarray) or raw.dtype != np.dtype('float64')
                or raw.ndim != 3 or raw.shape[1:] != (256, 2) or index >= len(raw)):
            raise ValueError('A mapped transmitter or original record is absent or incompatible')
        value = iq_record(raw[index])
        if value.dtype != np.dtype('float32') or value.shape != (2, 256) or not np.isfinite(value).all():
            raise ValueError('Selected clean IQ must be finite FP32[2,256]')
        received[position] = value
    return received


def build(config, output_root):
    """Create one immutable clean condition, with complete.json written last."""
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Paired clean output cannot be overwritten: ' + str(output))
    parent_complete, parent_report, roles = _parent(config)
    # Finish both exact member reads before creating an output. No random
    # generator, selection helper, channel, F or G is used along this route.
    clean = {role: _read_role(config, role, roles[role]['records']) for role in ROLES}
    output.mkdir(parents=True, exist_ok=False)
    pools = {}
    for role in ROLES:
        item = roles[role]
        capsule = 'p3dev-clean-' + _identity([SCHEMA, item['manifest']['capsule_id'], item['ids'].tolist()])
        manifest = dict(schema=SCHEMA, data_status='VALIDATED_ONCE', capsule_id=capsule,
            split_id=_identity([SCHEMA, item['manifest']['split_id'], capsule]), count=COUNTS[role],
            iq_shape=[COUNTS[role], 2, 256], registered_classes=paired.REGISTERED.copy(),
            processing_route='clean', mode='clean', equalization_enabled=False, fs_hz=25000000,
            single_observation=True)
        public, truth_root = output / role / 'public', output / role / 'truth'
        public.mkdir(parents=True); truth_root.mkdir()
        for name, value in [('ids', item['ids']), ('received', clean[role])]:
            with (public / (name + '.npy')).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
        _write_json(public / 'manifest.json', manifest)
        truth = item['truth']
        with (truth_root / 'truth.npz').open('xb') as stream:
            np.savez(stream, ids=truth['ids'], y=truth['y'], receiver=truth['receiver'],
                     scenario=np.full(COUNTS[role], 'clean'))
        _, checked_ids, checked = load_clean_public(public)
        checked._mmap.close()
        if not np.array_equal(checked_ids, item['ids']):
            raise ValueError('Clean output must retain the exact parent opaque ID order')
        pools[role] = dict(public_root=str(public), truth_path=str(truth_root / 'truth.npz'),
            count=COUNTS[role], capsule_id=capsule, split_id=manifest['split_id'])
    report = dict(schema='pamf_phase3_development_clean_builder_v1', config=config,
        source_parent=str(Path(config['paired_pool'])), parent_status=parent_complete['status'],
        parent_mapping_ref=str(Path(config['paired_pool']) / 'builder_report.json') + '#mapping',
        parent_role_handles={role: {key: roles[role]['manifest'][key] for key in ('capsule_id', 'split_id')}
                             for role in ROLES},
        validation_reused=parent_report['validation_reused'], development_only=True,
        archive_reads=2, pickle_reads=2, formal_target_reads=0, full_archive_rescan=False,
        archive_revalidation=False, new_sampling=False, quality_selection=False,
        paired_original_not_new_records=True, canonical_physical_unique=True, unique_canonical_count=1248,
        ids_and_order_equal_parent=True, private_truth_copied=True, private_scenario='clean',
        input_processing_state='WiSig_equalized1_center256_unit_rms',
        input_RMS_normalizations_per_record=1, original_eq_reapplied=False,
        channel_generated=False, channel_records=0, F_reads=0, G_reads=0, model_reads=0,
        model_predictions=0, optimizer_updates=0,
        single_observation_scope='one fixed clean observation within this condition; same original '
            'physical records as star control, not additional independent physical observations',
        model_input_isolation='clean and star are separate public roots; a star model must not '
            'read clean per-record references or consume clean as restoration truth')
    _write_json(output / 'builder_report.json', report)
    result = dict(status='PHASE3_DEVELOPMENT_CLEAN_VALIDATED', data_status='VALIDATED_ONCE',
        development_only=True, discovery=pools['discovery'], evaluation=pools['evaluation'],
        discovery_count=384, evaluation_count=864, canonical_physical_unique=True,
        unique_canonical_count=1248, single_observation=True, paired_original_not_new_records=True,
        source_parent=str(Path(config['paired_pool'])), archive_reads=2, pickle_reads=2,
        formal_target_reads=0, channel_records=0, F_reads=0, G_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, builder_report=str(output / 'builder_report.json'))
    _write_json(output / 'complete.json', result)
    print('PHASE3_DEVELOPMENT_CLEAN_VALIDATED', 384, 864, 1248, flush=True)
    return result


def main(argv=None):
    actual_argv = list(sys.argv) if argv is None else list(argv)
    print(json.dumps(dict(event='PHASE3_DEVELOPMENT_CLEAN_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
                          argv=actual_argv), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build(_read_json(args.config), args.output)
    print(json.dumps({key: result[key] for key in ('status', 'discovery_count', 'evaluation_count')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
