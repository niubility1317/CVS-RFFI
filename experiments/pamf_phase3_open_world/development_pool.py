"""Build the independent RX24-1 Phase3 development-only received pools.

This CPU builder reuses completed Full WiSig validation. It never opens the
formal RX24-18 pool, models, or compact IQ. Only private builder/scorer files
contain canonical TX/scene/role mappings; consumers receive phase3_min_v1.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

from .data import PUBLIC_FIELDS, PUBLIC_FILES, _identity, load_public
from .full_pool import (
    _apply_channel, _channel_api, _channel_config, _channel_seed,
    _read_json, _read_numpy_pickle, _select_rows, _write_json,
)


REGISTERED = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
DISCOVERY = ['19-15', '19-16']
FUTURE = ['7-1']
SOURCE_RECEIVERS = ['1-19', '18-2', '19-2', '2-19', '3-19']
PHASE2_RECEIVERS = ['1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8']
SCENES = ['practical_low_suburban', 'practical_mid_urban', 'practical_low_urban']
DAYS = dict(discovery='2021_03_08', evaluation='2021_03_15')
PER_SCENE = dict(discovery=16, evaluation=32)
SEEDS = dict(selection=2026102911, receiver=2026102912, channel=2026102913, shuffle=2026102914)
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip',
                  'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-1.pkl'),
    '2021_03_15': ('pkl_wifi_eq_2021_03_15-20261028T144306Z-1-004.zip',
                  'pkl_wifi_eq_2021_03_15/dataset_2021_03_15_node24-1.pkl'),
}
FIELDS = {'validation_root', 'full_root', 'compact_receiver_evidence', 'source_receivers',
          'formal_receiver', 'receiver', 'receiver_index', 'registered_classes',
          'discovery_classes', 'future_unknown_classes', 'days', 'per_scene', 'scenes', 'seeds'}


def validate_config(config):
    """Accept only the fixed development matrix; paths locate existing inputs."""
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only the declared development pool configuration is accepted')
    if any(not isinstance(config[key], str) or not config[key].strip()
           for key in ('validation_root', 'full_root', 'compact_receiver_evidence')):
        raise ValueError('Existing Full, validation and compact RX evidence paths are required')
    if (config['receiver'] != '24-1' or type(config['receiver_index']) is not int
            or config['receiver_index'] != 26 or config['formal_receiver'] != '24-18'
            or config['source_receivers'] != SOURCE_RECEIVERS
            or config['registered_classes'] != REGISTERED or config['discovery_classes'] != DISCOVERY
            or config['future_unknown_classes'] != FUTURE or config['scenes'] != SCENES
            or config['days'] != DAYS or config['per_scene'] != PER_SCENE or config['seeds'] != SEEDS):
        raise ValueError('The fixed development classes, RX namespace, dates, counts or seeds differ')
    if (not isinstance(config['per_scene'], dict) or any(type(v) is not int for v in config['per_scene'].values())
            or not isinstance(config['seeds'], dict) or any(type(v) is not int for v in config['seeds'].values())
            or config['receiver'] in config['source_receivers']
            or config['receiver'] == config['formal_receiver']
            or config['receiver'] in PHASE2_RECEIVERS):
        raise ValueError('Integer counts/seeds and disjoint development receivers are required')
    return config


def _compact_receivers(config):
    evidence = _read_json(config['compact_receiver_evidence'])
    expected = {'ManyRx.pkl', 'ManySig.pkl', 'ManyTx.pkl', 'SingleDay.pkl'}
    if not isinstance(evidence, dict) or evidence.get('status') != 'VERIFIED':
        raise ValueError('Completed compact receiver header evidence is required')
    rows = evidence.get('rows')
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('All four compact receiver lists are required')
    lists = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('file') not in expected or row['file'] in lists:
            raise ValueError('Compact receiver evidence has a missing or duplicate file')
        names = row.get('receiver_names')
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or type(row.get('receiver_count')) is not int
                or row['receiver_count'] != len(names) or config['receiver'] in names):
            raise ValueError('Development receiver is present in compact or RX evidence is invalid')
        lists[row['file']] = names
    union = set().union(*(set(names) for names in lists.values()))
    if set(lists) != expected or not set(PHASE2_RECEIVERS).issubset(union):
        raise ValueError('Compact evidence must also cover the seven current Phase2 receivers')
    return lists


def _metadata_capacity(config):
    """Check the existing small count index without rescanning Full archives."""
    metadata = _read_json(Path(config['validation_root']) / 'counts.json')
    if not isinstance(metadata, dict) or set(metadata) != {'dates', 'receivers', 'transmitters', 'records'}:
        raise ValueError('Existing Full count metadata is invalid')
    for name, size in [('dates', 4), ('receivers', 41), ('transmitters', 174)]:
        values = metadata[name]
        if (not isinstance(values, list) or len(values) != size
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != size):
            raise ValueError('Full count metadata namespace or dimensions differ')
    receivers = metadata['receivers']
    if (receivers[config['receiver_index']] != config['receiver']
            or not set(SOURCE_RECEIVERS + PHASE2_RECEIVERS + [config['formal_receiver']]).issubset(receivers)
            or not set(DAYS.values()).issubset(metadata['dates'])
            or not set(REGISTERED + DISCOVERY + FUTURE).issubset(metadata['transmitters'])):
        raise ValueError('Full count metadata does not identify the declared RX/day/TX namespace')
    try:
        counts = np.asarray(metadata['records'])
    except (TypeError, ValueError) as exc:
        raise ValueError('Full count metadata must retain the 4x41x174 integer table') from exc
    if counts.shape != (4, 41, 174) or counts.dtype.kind not in 'iu' or bool((counts < 0).any()):
        raise ValueError('Full count metadata must retain the 4x41x174 nonnegative integer table')
    capacity = {}
    for role, day in DAYS.items():
        day_index = metadata['dates'].index(day)
        required = PER_SCENE[role] * len(SCENES)
        # The nine-device capacity decision was fixed before IQ access. The
        # later-unknown device is checked here but never selected for U_dev.
        cells = {tx: int(counts[day_index, config['receiver_index'], metadata['transmitters'].index(tx)])
                 for tx in REGISTERED + DISCOVERY + FUTURE}
        if min(cells.values()) < required:
            raise ValueError('Insufficient existing Full metadata capacity for development ' + role)
        capacity[role] = dict(day=day, required_per_selected_tx=required,
                              available_per_tx=cells, all_nine_tx_min=min(cells.values()))
    return capacity


def _validated_members(config):
    validation, full = Path(config['validation_root']), Path(config['full_root'])
    summary = _read_json(validation / 'summary.json')
    if (not isinstance(summary, dict) or summary.get('status') != 'VERIFIED'
            or Path(summary.get('dataset_directory', '')).resolve() != full.resolve()
            or type(summary.get('members_checked')) is not int or summary['members_checked'] != 158
            or summary.get('expected_nonempty_RX_date_members') != 158
            or type(summary.get('count_mismatch_cells')) is not int or summary['count_mismatch_cells'] != 0
            or summary.get('errors') != [] or not full.is_dir()):
        raise ValueError('The matching completed Full validation must be reused before development selection')
    compact, capacity = _compact_receivers(config), _metadata_capacity(config)
    wanted = {pair: day for day, pair in MEMBERS.items()}
    found = {}
    # Locate exactly these two PASS members in the existing index. No ZIP
    # directory listing, CRC pass, archive hash, or raw Full revalidation.
    with (validation / 'members.jsonl').open('r', encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Existing Full member index contains an invalid row')
            pair = (row.get('archive'), row.get('member'))
            if pair not in wanted:
                continue
            day = wanted[pair]
            if (day in found or row.get('date') != day or row.get('receiver') != config['receiver']
                    or row.get('status') != 'PASS'):
                raise ValueError('Indexed development Full member identity is duplicate or incompatible')
            found[day] = dict(archive=pair[0], member=pair[1])
    if set(found) != set(MEMBERS):
        raise ValueError('The two development Full members are absent from the existing validation index')
    return summary, found, compact, capacity


def _selected_member(config, locator, day, transmitters, per_scene):
    print('PHASE3_DEVELOPMENT_MEMBER', day, locator['archive'], locator['member'], flush=True)
    try:
        with zipfile.ZipFile(Path(config['full_root']) / locator['archive']) as stored:
            with stored.open(locator['member'], 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (KeyError, FileNotFoundError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The indexed development Full NumPy member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full PKL must contain data/node_list only')
    names, arrays = payload['node_list'], payload['data']
    if (not isinstance(names, list) or not isinstance(arrays, list) or len(names) != len(arrays)
            or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names)):
        raise ValueError('Full node_list/data association is invalid')
    lookup = {name: index for index, name in enumerate(names)}
    if not set(transmitters).issubset(lookup):
        raise ValueError('Indexed development Full member lacks a declared transmitter')
    selected = []
    for tx in transmitters:
        selected.extend(_select_rows(arrays[lookup[tx]], tx, config['receiver'], day, per_scene,
                                     config['scenes'], config['seeds']['selection']))
    return selected


def _write_pool(config, root, role, records, channels, locator):
    from experiments.phase1_external_dg.data import iq_record
    _, receiver_for_session, stable_seed, _ = _channel_api()
    day = config['days'][role]
    session = config['receiver'] + '/' + day + '/1'
    receiver = receiver_for_session(channels[SCENES[0]], config['seeds']['receiver'], session)
    label_map = {name: i for i, name in enumerate(REGISTERED + DISCOVERY + FUTURE)}
    received = np.empty((len(records), 2, 256), dtype=np.float32)
    ids, mapping, labels, scenes = [], [], [], []
    tx_seen = Counter()
    per_tx = config['per_scene'][role] * len(SCENES)
    for i, record in enumerate(records):
        canonical, scene = record['canonical_record'], record['scene']
        normalized = iq_record(record.pop('raw'))
        seed = _channel_seed(config['seeds']['channel'], scene, canonical)
        value, _ = _apply_channel(normalized, channels[scene], seed, receiver)
        if value.dtype != np.dtype('float32') or value.shape != (2, 256) or not np.isfinite(value).all():
            raise ValueError('Development received observation must be finite FP32[2,256]')
        received[i] = value
        opaque = 'p3dev-' + _identity(['FullWiSigEq_development_opaque', config['seeds']['shuffle'], list(canonical)])
        ids.append(opaque)
        labels.append(label_map[canonical[1]])
        scenes.append(scene)
        mapping.append(dict(opaque_id=opaque, canonical_record=list(canonical), scene=scene,
            channel_seed=int(seed), receiver_session=session, receiver_seed=config['seeds']['receiver'],
            source_archive=locator['archive'], source_member=locator['member']))
        tx_seen[canonical[1]] += 1
        if tx_seen[canonical[1]] == per_tx:
            print('PHASE3_DEVELOPMENT_PROGRESS', role, canonical[1], per_tx, i + 1, len(records), flush=True)
    ids = np.asarray(ids)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Opaque ID collision in development pool')
    order = np.random.default_rng(stable_seed(config['seeds']['shuffle'], config['receiver'], day)).permutation(len(ids))
    ids, received = ids[order], received[order]
    labels, scenes = np.asarray(labels, dtype=np.int64)[order], np.asarray(scenes)[order]
    mapping = [mapping[int(index)] for index in order]
    capsule = 'p3dev-' + _identity([config['receiver'], day, config['seeds'],
                                   [record['canonical_record'] for record in mapping]])
    manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE', capsule_id=capsule,
        split_id=_identity(['FullWiSigEq_development_single_received', capsule]), count=len(ids),
        iq_shape=[len(ids), 2, 256], registered_classes=REGISTERED.copy(), processing_route='residual',
        mode='post_sync', equalization_enabled=False, fs_hz=25000000, single_observation=True)
    if set(manifest) != PUBLIC_FIELDS:
        raise ValueError('Public field whitelist changed')
    public, truth = root / role / 'public', root / role / 'truth'
    public.mkdir(parents=True)
    truth.mkdir()
    with (public / 'ids.npy').open('xb') as stream:
        np.save(stream, ids, allow_pickle=False)
    with (public / 'received.npy').open('xb') as stream:
        np.save(stream, received, allow_pickle=False)
    _write_json(public / 'manifest.json', manifest)
    with (truth / 'truth.npz').open('xb') as stream:
        np.savez(stream, ids=ids, y=labels,
                 receiver=np.full(len(ids), config['receiver_index'], dtype=np.int64), scenario=scenes)
    if {path.name for path in public.iterdir()} != PUBLIC_FILES:
        raise ValueError('Private data appeared in the public development pool')
    _, _, checked = load_public(public)
    checked._mmap.close()
    return dict(public_root=str(public), truth_path=str(truth / 'truth.npz'), count=len(ids),
                capsule_id=capsule, split_id=manifest['split_id']), mapping


def build(config, output_root):
    """Build one immutable development U/V slice, without model activity."""
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Independent Phase3 development output cannot be overwritten: ' + str(output))
    summary, members, compact, capacity = _validated_members(config)
    selected = {}
    for role in ('discovery', 'evaluation'):
        day = config['days'][role]
        transmitters = REGISTERED + DISCOVERY + (FUTURE if role == 'evaluation' else [])
        selected[role] = _selected_member(config, members[day], day, transmitters, config['per_scene'][role])
    canonical = [record['canonical_record'] for records in selected.values() for record in records]
    if len(canonical) != 1248 or len(set(canonical)) != len(canonical):
        raise ValueError('Development physical identities must be disjoint across every role and scene')
    channels = {scene: _channel_config(scene) for scene in SCENES}
    output.mkdir(parents=True, exist_ok=False)
    pools, mapping = {}, {}
    for role in ('discovery', 'evaluation'):
        pools[role], mapping[role] = _write_pool(config, output, role, selected[role], channels,
                                               members[config['days'][role]])
    report = dict(schema='pamf_phase3_development_builder_v1', config=config, development_only=True,
        validation_reused=dict(root=config['validation_root'], status=summary['status'],
            members_checked=summary['members_checked'], count_mismatch_cells=summary['count_mismatch_cells']),
        metadata_capacity=capacity, compact_receiver_lists=compact,
        receiver_index_namespace='FullWiSigEq official Full RX list; not Main compact receiver indices',
        source_development_formal_receivers_disjoint=True, phase2_development_receivers_disjoint=True,
        archive_reads=2, pickle_reads=2, formal_target_reads=0,
        full_archive_rescan=False, archive_revalidation=False,
        physical_identity='(FullWiSigEq, transmitter, receiver, date, equalized=1, original_packet_index)',
        compact_prefix_identity_used=False, packet_index_cutoff_used=False,
        canonical_physical_unique=True, scene_and_role_canonical_disjoint=True,
        selection_rule='per TX/RX/date fixed permutation; selected j maps to scenes[j % 3]',
        channel_seed_rule='stable_seed(channel_seed, scene, *canonical_record); no class role',
        receiver_session_rule='RX/date/eq; no TX, scene or known/unknown role',
        input_processing_state='WiSig_equalized1_center256_unit_rms', input_RMS_normalizations_per_record=1,
        clean_reference_written=False, original_eq_reapplied=False,
        channel_route='direct CPU complex128 input to ResidualChannel, FP32 received output',
        channel_configs={scene: asdict(value) for scene, value in channels.items()},
        model_reads=0, model_predictions=0, optimizer_updates=0, channel_records=1248,
        interpretation='V_dev development domain, not formal future query or a new-TX blind test; '
                       'the three development novel TX were previously evaluated on source RX; '
                       'data construction does not establish adaptation, registration or 95% rejection',
        mapping=mapping)
    _write_json(output / 'builder_report.json', report)
    result = dict(status='PHASE3_DEVELOPMENT_POOL_VALIDATED', data_status='VALIDATED_ONCE',
        development_only=True, discovery=pools['discovery'], evaluation=pools['evaluation'],
        discovery_count=384, evaluation_count=864, canonical_physical_unique=True,
        unique_canonical_count=1248, single_observation=True,
        source_development_formal_receivers_disjoint=True, phase2_development_receivers_disjoint=True,
        formal_target_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0,
        builder_report=str(output / 'builder_report.json'))
    _write_json(output / 'complete.json', result)
    print('PHASE3_DEVELOPMENT_POOL_VALIDATED', 384, 864, 1248, flush=True)
    return result


def main(argv=None):
    actual_argv = list(sys.argv) if argv is None else list(argv)
    print(json.dumps(dict(event='PHASE3_DEVELOPMENT_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
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
