"""Build one G10 MIX01 received observation for the fixed development U/Q IDs.

The completed legacy mapping supplies physical records and roles only. Legacy
truth, legacy received IQ, scores, models, and the formal receiver are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

from . import development_pool as legacy
from .data import PUBLIC_FIELDS, PUBLIC_FILES, _identity, load_public
from .development_clean import _ids, _manifest_contract
from .full_pool import _read_json, _read_numpy_pickle, _write_json


RUN_ID = '20261029-phase3-star-g10-pool-rx24-1-r01'
SCHEMA = 'pamf_phase3_star_g10_development_builder_v1'
ROLES = ('discovery', 'evaluation')
COUNTS = dict(discovery=384, evaluation=864)
EPISODE = dict(id='MIX01', condition='MIX', scene_seed=2026103001, channel_seed=2026104001)
FIELDS = {'legacy_pool_root', 'legacy_config_path', 'mixture_config_path', 'episode', 'full_root', 'validation_root'}


def validate_config(config):
    """Only existing data handles and the fixed MIX01 realization are inputs."""
    if (not isinstance(config, dict) or set(config) != FIELDS
            or any(not isinstance(config[k], str) or not config[k].strip() for k in FIELDS - {'episode'})
            or config['episode'] != EPISODE
            or any(type(config['episode'].get(k)) is not int for k in ('scene_seed', 'channel_seed'))):
        raise ValueError('Fixed G10 MIX01 and only the six declared data configuration fields are required')
    return config


def _observe(raw, **kwargs):
    from .star_channel import observe
    return observe(raw, **kwargs)


def _channel_mixture(path):
    from .star_channel import validate_mixture
    mixture = _read_json(path)
    validate_mixture(mixture)
    return mixture


def _parent(config):
    """Reuse handles and exact mapping; never open legacy received or truth."""
    root = Path(config['legacy_pool_root'])
    complete, report = _read_json(root / 'complete.json'), _read_json(root / 'builder_report.json')
    previous = legacy.validate_config(_read_json(config['legacy_config_path']))
    if (not isinstance(complete, dict) or complete.get('status') != 'PHASE3_DEVELOPMENT_POOL_VALIDATED'
            or complete.get('data_status') != 'VALIDATED_ONCE' or complete.get('development_only') is not True
            or complete.get('discovery_count') != 384 or complete.get('evaluation_count') != 864
            or complete.get('unique_canonical_count') != 1248 or complete.get('canonical_physical_unique') is not True
            or complete.get('single_observation') is not True
            or any(complete.get(k) != 0 for k in ('formal_target_reads', 'model_reads', 'model_predictions', 'optimizer_updates'))):
        raise ValueError('Completed validated legacy development handles are required')
    if (not isinstance(report, dict) or report.get('schema') != 'pamf_phase3_development_builder_v1'
            or report.get('development_only') is not True or report.get('config') != previous
            or report.get('formal_target_reads') != 0 or report.get('canonical_physical_unique') is not True
            or report.get('scene_and_role_canonical_disjoint') is not True):
        raise ValueError('The completed legacy mapping/configuration differs')
    if (Path(previous['full_root']).resolve() != Path(config['full_root']).resolve()
            or Path(previous['validation_root']).resolve() != Path(config['validation_root']).resolve()
            or not Path(config['full_root']).is_dir()):
        raise ValueError('The completed Full and validation handles must be reused')
    reused = report.get('validation_reused', {})
    if (reused.get('status') != 'VERIFIED' or reused.get('members_checked') != 158
            or reused.get('count_mismatch_cells') != 0):
        raise ValueError('Completed Full validation handle is absent')
    summary = _read_json(Path(config['validation_root']) / 'summary.json')
    if (not isinstance(summary, dict) or summary.get('status') != 'VERIFIED'
            or Path(summary.get('dataset_directory', '')).resolve() != Path(config['full_root']).resolve()
            or summary.get('members_checked') != 158 or summary.get('expected_nonempty_RX_date_members') != 158
            or summary.get('count_mismatch_cells') != 0 or summary.get('errors') != []):
        raise ValueError('The existing completed Full validation handle differs')
    mapping = report.get('mapping')
    if not isinstance(mapping, dict) or set(mapping) != set(ROLES):
        raise ValueError('Both original U/Q role mappings are required')
    roles, physical, opaque = {}, set(), set()
    for role in ROLES:
        public = root / role / 'public'
        if {p.name for p in public.iterdir()} != PUBLIC_FILES:
            raise ValueError('Legacy public input must retain the three-file whitelist')
        manifest = _read_json(public / 'manifest.json')
        count = _manifest_contract(manifest, clean=False)
        handle = complete.get(role, {})
        if (count != COUNTS[role] or handle.get('count') != count
                or any(handle.get(k) != manifest[k] for k in ('capsule_id', 'split_id'))):
            raise ValueError('Original role/public capsule or split differs')
        ids = np.load(public / 'ids.npy', allow_pickle=False)
        _ids(ids, count)
        if opaque & set(ids.tolist()):
            raise ValueError('Legacy IDs must be unique across U/Q roles')
        opaque.update(ids.tolist())
        records = mapping[role]
        if not isinstance(records, list) or len(records) != count:
            raise ValueError('Legacy role mapping count differs')
        day = legacy.DAYS[role]
        archive, member = legacy.MEMBERS[day]
        allowed = legacy.REGISTERED + legacy.DISCOVERY + (legacy.FUTURE if role == 'evaluation' else [])
        for identifier, record in zip(ids, records):
            canonical = record.get('canonical_record') if isinstance(record, dict) else None
            if (not isinstance(record, dict) or record.get('opaque_id') != str(identifier)
                    or not isinstance(canonical, list) or len(canonical) != 6
                    or canonical[0] != 'FullWiSigEq' or canonical[1] not in allowed
                    or canonical[2] != '24-1' or canonical[3] != day
                    or type(canonical[4]) is not int or canonical[4] != 1
                    or type(canonical[5]) is not int or canonical[5] < 0
                    or record.get('source_archive') != archive or record.get('source_member') != member):
                raise ValueError('Original order, role, canonical record or indexed member differs')
            identity = tuple(canonical)
            if identity in physical:
                raise ValueError('Duplicate canonical physical record in original U/Q mapping')
            physical.add(identity)
        expected = {tx: 48 if role == 'discovery' else 96 for tx in allowed}
        if Counter(row['canonical_record'][1] for row in records) != expected:
            raise ValueError('Original role/TX composition differs; no resampling or Q-to-U movement is allowed')
        roles[role] = dict(records=records, manifest=manifest)
    if len(physical) != 1248:
        raise ValueError('Exactly the original 1248 distinct physical records must be retained')
    return complete, report, roles


def _read_role(config, role, records):
    archive, member = legacy.MEMBERS[legacy.DAYS[role]]
    print('PHASE3_STAR_G10_MEMBER', role, archive, member, flush=True)
    try:
        with zipfile.ZipFile(Path(config['full_root']) / archive) as stored:
            with stored.open(member, 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (OSError, KeyError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The already-indexed development Full member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full member must contain data/node_list only')
    names, arrays = payload['node_list'], payload['data']
    if (not isinstance(names, list) or not isinstance(arrays, list) or len(names) != len(arrays)
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError('Full member node/data association differs')
    lookup = dict(zip(names, arrays))
    result = []
    for record in records:
        _, tx, _, _, _, index = record['canonical_record']
        values = lookup.get(tx)
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
                or values.ndim != 3 or values.shape[1:] != (256, 2) or index >= len(values)):
            raise ValueError('An original mapped packet is missing or incompatible')
        # Adapter owns the one Phase2 FP32+normalize_input operation. No iq_record,
        # EQ, RMS, scene selection, or role-dependent transformation occurs here.
        result.append(values[index].T.copy())
    return result


def _materialize(config, role, records, raw, mixture):
    from .star_channel import identity as phase2_reprhash
    episode = {k: v for k, v in config['episode'].items() if k != 'condition'}
    received = np.empty((len(records), 2, 256), dtype=np.float32)
    mapping, ids, labels, scenes = [], [], [], []
    label_map = {tx: i for i, tx in enumerate(legacy.REGISTERED + legacy.DISCOVERY + legacy.FUTURE)}
    for i, (record, value) in enumerate(zip(records, raw)):
        canonical = record['canonical_record']
        _, tx, rx, day, _, index = canonical
        physical_id = phase2_reprhash(['WiSig/Full/eq1', tx, rx, day, index])
        observation, meta = _observe(value, physical_id=physical_id, receiver=rx, day=day,
                                     episode=episode, mixture=mixture)
        if (not isinstance(observation, np.ndarray) or observation.dtype != np.dtype('float32')
                or observation.shape != (2, 256) or not np.isfinite(observation).all()
                or not isinstance(meta, dict) or not {'scenario', 'overlay_seed', 'input_rms', 'geometry'}.issubset(meta)):
            raise ValueError('G10 adapter must return one finite FP32[2,256] observation and private diagnostics')
        opaque_id = 'p3dev-star-g10-' + _identity([RUN_ID, episode, canonical])
        received[i] = observation
        ids.append(opaque_id); labels.append(label_map[tx]); scenes.append(meta['scenario'])
        mapping.append(dict(opaque_id=opaque_id, canonical_record=canonical.copy(), physical_id=physical_id,
            source_archive=record['source_archive'], source_member=record['source_member'], **meta))
        if (i + 1) % 256 == 0 or i + 1 == len(records):
            print('PHASE3_STAR_G10_PROGRESS', role, i + 1, len(records), flush=True)
    if len(set(ids)) != len(ids):
        raise ValueError('New opaque observation IDs collide')
    return dict(received=received, ids=np.asarray(ids), labels=np.asarray(labels, dtype=np.int64),
                scenes=np.asarray(scenes), mapping=mapping)


def build(config, output_root):
    """Build the fixed small CPU-only G10 pool; complete.json is written last."""
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('G10 development output must not overwrite history: ' + str(output))
    parent_complete, parent_report, roles = _parent(config)
    mixture = _channel_mixture(config['mixture_config_path'])
    raw = {role: _read_role(config, role, roles[role]['records']) for role in ROLES}
    observations = {role: _materialize(config, role, roles[role]['records'], raw[role], mixture) for role in ROLES}
    output.mkdir(parents=True, exist_ok=False)
    pools = {}
    for role in ROLES:
        data = observations[role]
        capsule = 'p3dev-star-g10-' + _identity([RUN_ID, EPISODE, role, data['ids'].tolist()])
        manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE', capsule_id=capsule,
            split_id=_identity([RUN_ID, role, capsule]), count=COUNTS[role], iq_shape=[COUNTS[role], 2, 256],
            registered_classes=legacy.REGISTERED.copy(), processing_route='residual', mode='post_sync',
            equalization_enabled=False, fs_hz=25000000, single_observation=True)
        if set(manifest) != PUBLIC_FIELDS:
            raise ValueError('Public field whitelist differs')
        public, truth = output / role / 'public', output / role / 'truth'
        public.mkdir(parents=True); truth.mkdir()
        for name, value in [('ids', data['ids']), ('received', data['received'])]:
            with (public / (name + '.npy')).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
        _write_json(public / 'manifest.json', manifest)
        with (truth / 'truth.npz').open('xb') as stream:
            np.savez(stream, ids=data['ids'], y=data['labels'], receiver=np.full(COUNTS[role], 26, dtype=np.int64), scenario=data['scenes'])
        _, _, checked = load_public(public)
        checked._mmap.close()
        pools[role] = dict(public_root=str(public), truth_path=str(truth / 'truth.npz'), count=COUNTS[role],
                           capsule_id=capsule, split_id=manifest['split_id'])
    report = dict(schema=SCHEMA, run_id=RUN_ID, config=config, episode=EPISODE.copy(), development_only=True,
        mixture_configuration=mixture,
        realized_scene_counts={role: dict(Counter(observations[role]['scenes'].tolist())) for role in ROLES},
        source_parent=config['legacy_pool_root'], parent_status=parent_complete['status'],
        parent_mapping_ref=str(Path(config['legacy_pool_root']) / 'builder_report.json') + '#mapping',
        validation_reused=parent_report['validation_reused'], archive_reads=2, pickle_reads=2,
        full_archive_rescan=False, archive_revalidation=False, new_sampling=False, quality_selection=False,
        old_received_read=False, old_truth_read=False, old_scene_used=False, old_score_read=False,
        original_role_and_canonical_order_retained=True, old_Q_became_training=False,
        canonical_physical_unique=True, unique_canonical_count=1248, new_observation_count=1248,
        same_role_new_channel_realization_not_new_physical_records=True, single_observation=True,
        physical_identity_rule="Phase2 repr hash of ['WiSig/Full/eq1',TX,RX,day,original_packet_index]",
        opaque_ID_rule='fixed run + episode + canonical; no public TX, role or scene',
        channel_rule='same G10 adapter formula for all records; scene draw independent of known/unknown/role; no class-scene quotas',
        input_processing_state='WiSig equalized1; transpose only before Phase2 FP32 normalize_input exactly once',
        input_RMS_normalizations_per_record=1, original_eq_reapplied=False, output_RMS_normalization=False,
        channel_records=1248, clean_reference_written=False, F_reads=0, G_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, formal_target_reads=0, foreign_processes_stopped=0,
        mapping={role: observations[role]['mapping'] for role in ROLES})
    _write_json(output / 'builder_report.json', report)
    result = dict(status='PHASE3_STAR_G10_POOL_VALIDATED', data_status='VALIDATED_ONCE', run_id=RUN_ID,
        development_only=True, discovery=pools['discovery'], evaluation=pools['evaluation'],
        discovery_count=384, evaluation_count=864, canonical_physical_unique=True, unique_canonical_count=1248,
        new_observation_count=1248, single_observation=True, source_parent=config['legacy_pool_root'],
        same_role_new_channel_realization_not_new_physical_records=True,
        source_development_formal_receivers_disjoint=True, phase2_development_receivers_disjoint=True,
        archive_reads=2, pickle_reads=2, channel_records=1248, formal_target_reads=0, F_reads=0, G_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, foreign_processes_stopped=0,
        builder_report=str(output / 'builder_report.json'))
    _write_json(output / 'complete.json', result)
    print('PHASE3_STAR_G10_POOL_VALIDATED', 384, 864, 1248, flush=True)
    return result


def main(argv=None):
    actual = list(sys.argv) if argv is None else list(argv)
    print(json.dumps(dict(event='PHASE3_STAR_G10_POOL_PROCESS', pid=os.getpid(), cwd=os.getcwd(), argv=actual), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build(_read_json(args.config), args.output)
    print(json.dumps({k: result[k] for k in ('status', 'discovery_count', 'evaluation_count')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
