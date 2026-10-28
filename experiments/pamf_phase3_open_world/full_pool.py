"""Build the authorized independent Full WiSig eq Phase3 received pools."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

from .data import PUBLIC_FIELDS, PUBLIC_FILES, load_public, _identity


REGISTERED = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
DISCOVERY = ['11-4', '3-13']
FUTURE = ['12-20', '5-20']
SOURCE_RECEIVERS = ['1-19', '18-2', '19-2', '2-19', '3-19']
SCENES = ['practical_low_suburban', 'practical_mid_urban', 'practical_low_urban']
DAYS = dict(discovery='2021_03_08', evaluation='2021_03_23')
PER_SCENE = dict(discovery=32, evaluation=64)
SEEDS = dict(selection=2026102901, receiver=2026102902, channel=2026102903, shuffle=2026102904)
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip',
                  'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-18.pkl'),
    '2021_03_23': ('pkl_wifi_eq_2021_03_23-20261028T144307Z-1-005.zip',
                  'pkl_wifi_eq_2021_03_23/dataset_2021_03_23_node24-18.pkl'),
}
FIELDS = {'validation_root', 'full_root', 'source_receivers', 'receiver', 'receiver_index',
          'registered_classes', 'discovery_classes', 'future_unknown_classes',
          'days', 'per_scene', 'scenes', 'seeds'}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('Only the declared independent Full pool configuration is accepted')
    if any(not isinstance(config[key], str) or not config[key] for key in ('validation_root', 'full_root')):
        raise ValueError('Existing Full and cached validation directories are required')
    if (config['receiver'] != '24-18' or type(config['receiver_index']) is not int or config['receiver_index'] != 30
            or config['source_receivers'] != SOURCE_RECEIVERS
            or config['registered_classes'] != REGISTERED or config['discovery_classes'] != DISCOVERY
            or config['future_unknown_classes'] != FUTURE or config['scenes'] != SCENES
            or config['days'] != DAYS or config['per_scene'] != PER_SCENE or config['seeds'] != SEEDS):
        raise ValueError('The preregistered Full classes, RX namespace, dates, counts or seeds differ')
    if (not isinstance(config['per_scene'], dict) or any(type(v) is not int for v in config['per_scene'].values())
            or not isinstance(config['seeds'], dict) or any(type(v) is not int for v in config['seeds'].values())
            or config['receiver'] in config['source_receivers']):
        raise ValueError('Integer allocation/seed values and disjoint receivers are required')
    return config


def _validated_members(config):
    validation = Path(config['validation_root'])
    full = Path(config['full_root'])
    summary = _read_json(validation / 'summary.json')
    if (not isinstance(summary, dict) or summary.get('status') != 'VERIFIED'
            or Path(summary.get('dataset_directory', '')).resolve() != full.resolve()
            or type(summary.get('members_checked')) is not int or summary['members_checked'] != 158
            or summary.get('expected_nonempty_RX_date_members') != 158
            or type(summary.get('count_mismatch_cells')) is not int or summary['count_mismatch_cells'] != 0
            or summary.get('errors') != []):
        raise ValueError('The matching completed Full validation must be reused before selection')
    if not full.is_dir():
        raise ValueError('Validated Full directory is unavailable')
    wanted = {pair: day for day, pair in MEMBERS.items()}
    found = {}
    # This small existing index locates two members; no ZIP directory scan occurs.
    with (validation / 'members.jsonl').open('r', encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (row.get('archive'), row.get('member'))
            if pair not in wanted:
                continue
            day = wanted[pair]
            if (day in found or row.get('date') != day or row.get('receiver') != config['receiver']
                    or row.get('status') != 'PASS'):
                raise ValueError('Indexed Full member identity is missing, duplicate or incompatible')
            found[day] = dict(archive=pair[0], member=pair[1])
    if set(found) != set(MEMBERS):
        raise ValueError('The two declared Full members are absent from the existing validation index')
    return summary, found


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            from numpy._core.multiarray import _reconstruct
        except ImportError:
            from numpy.core.multiarray import _reconstruct
        allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
                   ('numpy.core.multiarray', '_reconstruct'): _reconstruct,
                   ('numpy._core.multiarray', '_reconstruct'): _reconstruct}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Unexpected Full NumPy class: ' + module + '.' + name)
        return allowed[module, name]


def _read_numpy_pickle(stream):
    value = _NumpyOnly(stream).load()
    # Normal EOF on just the selected member; never a separate CRC/hash pass.
    if stream.read(1):
        raise ValueError('Trailing bytes after the declared Full NumPy object')
    return value


def _channel_api():
    code = str(Path(__file__).resolve().parents[2] / 'code')
    if code not in sys.path:
        sys.path.insert(0, code)
    from leo_practical.channel import Config, receiver_for_session, stable_seed
    from leo_practical.residual import ResidualChannel
    return Config, receiver_for_session, stable_seed, ResidualChannel


def _channel_config(scene):
    Config, _, _, _ = _channel_api()
    return Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
        mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect',
        input_processing_state='WiSig_equalized1_center256_unit_rms')


def _channel_seed(seed, scene, canonical):
    _, _, stable_seed, _ = _channel_api()
    return stable_seed(seed, scene, *canonical)


def _select_rows(array, tx, receiver, day, per_scene, scenes, selection_seed):
    if (not isinstance(array, np.ndarray) or array.dtype != np.dtype('float64')
            or array.ndim != 3 or array.shape[1:] != (256, 2)):
        raise ValueError('Selected Full TX must preserve float64[N,256,2] raw rows')
    count = per_scene * len(scenes)
    if len(array) < count:
        raise ValueError('Insufficient original Full rows for ' + tx + '/' + day)
    _, _, stable_seed, _ = _channel_api()
    stream = stable_seed(selection_seed, tx, receiver, day)
    indices = np.random.default_rng(stream).permutation(len(array))[:count]
    result = []
    for j, index in enumerate(indices):
        raw = np.array(array[int(index)], dtype=np.float64, copy=True)
        if not np.isfinite(raw).all() or np.max(np.abs(raw)) > np.finfo(np.float32).max:
            raise ValueError('Selected original Full IQ is nonfinite or overflows FP32')
        result.append(dict(canonical_record=('FullWiSigEq', tx, receiver, day, 1, int(index)),
                           scene=scenes[j % len(scenes)], raw=raw))
    return result


def _selected_member(config, locator, day, transmitters, per_scene):
    archive = Path(config['full_root']) / locator['archive']
    print('PHASE3_FULL_POOL_MEMBER', day, locator['archive'], locator['member'], flush=True)
    try:
        with zipfile.ZipFile(archive) as stored:
            with stored.open(locator['member'], 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (KeyError, FileNotFoundError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The indexed Full NumPy member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full PKL must contain data/node_list only')
    names, arrays = payload['node_list'], payload['data']
    if (not isinstance(names, list) or not isinstance(arrays, list) or len(names) != len(arrays)
            or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names)):
        raise ValueError('Full node_list/data association is invalid')
    lookup = {name: index for index, name in enumerate(names)}
    if not set(transmitters).issubset(lookup):
        raise ValueError('The indexed Full member lacks a declared transmitter')
    selected = []
    for tx in transmitters:
        selected.extend(_select_rows(arrays[lookup[tx]], tx, config['receiver'], day, per_scene,
                                     config['scenes'], config['seeds']['selection']))
    # Only these small selected raw rows escape; all other TX arrays are released.
    return selected


def _apply_channel(normalized, cfg, seed, receiver):
    if normalized.dtype != np.dtype('float32') or normalized.shape != (2, 256) or not np.isfinite(normalized).all():
        raise ValueError('Single iq_record FP32 normalized input required')
    _, _, _, ResidualChannel = _channel_api()
    signal = np.asarray(normalized[0], dtype=np.float64) + 1j * np.asarray(normalized[1], dtype=np.float64)
    received, metadata = ResidualChannel(cfg, seed, receiver).process(signal.astype(np.complex128, copy=False))
    if received.shape != (256,) or not np.isfinite(received).all():
        raise ValueError('Residual channel returned invalid fixed received IQ')
    # Do not apply another RMS normalization, crop or equalization here.
    return np.stack((received.real, received.imag), axis=0).astype(np.float32), metadata


def _write_pool(config, root, role, records, channels):
    from experiments.phase1_external_dg.data import iq_record
    _, receiver_for_session, stable_seed, _ = _channel_api()
    day = config['days'][role]
    session = config['receiver'] + '/' + day + '/1'
    receiver = receiver_for_session(channels[SCENES[0]], config['seeds']['receiver'], session)
    label_map = {name: i for i, name in enumerate(REGISTERED + DISCOVERY + FUTURE)}
    received = np.empty((len(records), 2, 256), dtype=np.float32)
    ids, ledger, labels, scenes = [], [], [], []
    tx_seen = Counter()
    per_tx = config['per_scene'][role] * len(SCENES)
    for i, record in enumerate(records):
        canonical, scene = record['canonical_record'], record['scene']
        normalized = iq_record(record.pop('raw'))
        seed = _channel_seed(config['seeds']['channel'], scene, canonical)
        value, metadata = _apply_channel(normalized, channels[scene], seed, receiver)
        if value.dtype != np.dtype('float32') or value.shape != (2, 256) or not np.isfinite(value).all():
            raise ValueError('Fixed residual received observation must be finite FP32[2,256]')
        received[i] = value
        opaque = 'p3f-' + _identity(['FullWiSigEq_opaque', config['seeds']['shuffle'], list(canonical)])
        ids.append(opaque)
        labels.append(label_map[canonical[1]])
        scenes.append(scene)
        ledger.append(dict(opaque_id=opaque, canonical_record=list(canonical), scene=scene,
            channel_seed=int(seed), receiver_session=session, receiver_seed=config['seeds']['receiver'],
            source_archive=MEMBERS[day][0], source_member=MEMBERS[day][1]))
        tx_seen[canonical[1]] += 1
        if tx_seen[canonical[1]] % 128 == 0 or tx_seen[canonical[1]] == per_tx:
            print('PHASE3_FULL_POOL_PROGRESS', role, canonical[1], tx_seen[canonical[1]], per_tx, i + 1, len(records), flush=True)
    ids = np.asarray(ids)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Opaque ID collision in independent Full pool')
    order = np.random.default_rng(stable_seed(config['seeds']['shuffle'], config['receiver'], day)).permutation(len(ids))
    ids, received = ids[order], received[order]
    labels, scenes = np.asarray(labels, dtype=np.int64)[order], np.asarray(scenes)[order]
    ledger = [ledger[int(index)] for index in order]
    capsule = 'p3full-' + _identity([config['receiver'], day, config['seeds'],
                                   [record['canonical_record'] for record in ledger]])
    manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE', capsule_id=capsule,
        split_id=_identity(['FullWiSigEq_single_received', capsule]), count=len(ids), iq_shape=[len(ids), 2, 256],
        registered_classes=REGISTERED.copy(), processing_route='residual', mode='post_sync',
        equalization_enabled=False, fs_hz=25000000, single_observation=True)
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
        np.savez(stream, ids=ids, y=labels, receiver=np.full(len(ids), config['receiver_index'], dtype=np.int64), scenario=scenes)
    if set(path.name for path in public.iterdir()) != PUBLIC_FILES:
        raise ValueError('Private data appeared in the public pool')
    load_public(public)
    return dict(public_root=str(public), truth_path=str(truth / 'truth.npz'),
                count=len(ids), capsule_id=capsule, split_id=manifest['split_id']), ledger


def build(config, output_root):
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Independent Phase3 Full output cannot be overwritten: ' + str(output))
    summary, members = _validated_members(config)
    selected = {}
    for role in ('discovery', 'evaluation'):
        day = config['days'][role]
        transmitters = REGISTERED + DISCOVERY + (FUTURE if role == 'evaluation' else [])
        selected[role] = _selected_member(config, members[day], day, transmitters, config['per_scene'][role])
    canonical = [record['canonical_record'] for records in selected.values() for record in records]
    if len(canonical) != 2688 or len(set(canonical)) != len(canonical):
        raise ValueError('Full record identities must be disjoint across every role and scene')
    channels = {scene: _channel_config(scene) for scene in SCENES}
    output.mkdir(parents=True, exist_ok=False)
    pools, mapping = {}, {}
    for role in ('discovery', 'evaluation'):
        pools[role], mapping[role] = _write_pool(config, output, role, selected[role], channels)
    report = dict(schema='pamf_phase3_full_eq_builder_v1', config=config,
        validation_reused=dict(root=config['validation_root'], status=summary['status'],
            members_checked=summary['members_checked'], count_mismatch_cells=summary['count_mismatch_cells']),
        receiver_index_namespace='FullWiSigEq official Full RX list; not Main compact receiver indices',
        source_target_receivers_disjoint=True, archive_reads=2, pickle_reads=2,
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
        model_reads=0, model_predictions=0, optimizer_updates=0, channel_records=2688, mapping=mapping)
    _write_json(output / 'builder_report.json', report)
    result = dict(status='PHASE3_FULL_POOL_VALIDATED', data_status='VALIDATED_ONCE',
        discovery=pools['discovery'], evaluation=pools['evaluation'], discovery_count=768, evaluation_count=1920,
        canonical_physical_unique=True, unique_canonical_count=2688, single_observation=True,
        source_target_receivers_disjoint=True, model_reads=0, model_predictions=0, optimizer_updates=0,
        builder_report=str(output / 'builder_report.json'))
    _write_json(output / 'complete.json', result)
    print('PHASE3_FULL_POOL_VALIDATED', 768, 1920, 2688, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build(_read_json(args.config), args.output)
    print(json.dumps(dict(status=result['status'], discovery_count=768, evaluation_count=1920)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
