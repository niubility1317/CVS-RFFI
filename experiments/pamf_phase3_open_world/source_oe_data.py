"""Build separately supervised source OE data without model or channel access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

from .data import _identity
from .full_pool import _read_numpy_pickle


KNOWN = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
RESERVED = ['11-4', '3-13', '12-20', '5-20']
TRAIN = ['11-14', '14-19', '17-20', '18-19']
NOVEL = ['19-15', '19-16', '7-1']
RECEIVERS = ['1-19', '18-2', '19-2', '2-19', '3-19']
MAIN_RX = [1, 3, 4, 6, 8]
DATES = ['2021_03_08', '2021_03_15', '2021_03_23']
ROLES = ('exposure_train', 'novel_validation')
COUNTS = dict(exposure_train=4200, novel_validation=3150)
KEYS = {'full_root', 'validation_root', 'compact_memberships', 'known_classes',
    'reserved_target_classes', 'receivers', 'dates', 'exposure_train_classes',
    'novel_validation_classes', 'per_cell', 'selection_seed'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the declared source OE data configuration is accepted')
    for key in ('full_root', 'validation_root', 'compact_memberships'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('Existing data/metadata location required: ' + key)
    if (config['known_classes'] != KNOWN or config['reserved_target_classes'] != RESERVED
            or config['receivers'] != RECEIVERS or config['dates'] != DATES
            or config['exposure_train_classes'] != TRAIN or config['novel_validation_classes'] != NOVEL
            or type(config['per_cell']) is not int or config['per_cell'] != 70
            or type(config['selection_seed']) is not int or config['selection_seed'] != 2026102905):
        raise ValueError('The fixed source OE classes, domains, allocation or seed differ')
    if set(TRAIN) & set(NOVEL) or set(TRAIN + NOVEL) & set(KNOWN + RESERVED):
        raise ValueError('Source OE training/novel/registered/reserved TX must be disjoint')
    return config


def _memberships(config):
    evidence = _read(config['compact_memberships'])
    if not isinstance(evidence, dict) or not isinstance(evidence.get('compact'), dict):
        raise ValueError('Existing compact TX membership metadata required')
    expected = dict(ManySig=6, ManyTx=150, ManyRx=10, SingleDay=28)
    if set(evidence['compact']) != set(expected):
        raise ValueError('All four compact TX memberships are required')
    union = set()
    for name, count in expected.items():
        row = evidence['compact'][name]
        values = row.get('tx_list') if isinstance(row, dict) else None
        if (not isinstance(values, list) or len(values) != count or len(set(values)) != count
                or any(not isinstance(tx, str) or not tx for tx in values)
                or type(row.get('count')) is not int or row['count'] != count):
            raise ValueError('Compact TX membership count/identity differs: ' + name)
        union.update(values)
    absent = evidence.get('full_tx_absent_all_compact')
    if (len(union) != 150 or evidence.get('compact_union_count') != 150
            or evidence.get('compact_union_not_in_full') != [] or evidence.get('full_tx_count') != 174
            or not isinstance(absent, list) or len(absent) != 24 or len(set(absent)) != 24
            or evidence.get('old_six') != KNOWN or evidence.get('formal_unknown_excluded') != RESERVED
            or evidence.get('source_receivers') != RECEIVERS or evidence.get('source_dates') != DATES):
        raise ValueError('The completed compact/Full source OE metadata identity differs')
    if set(TRAIN + NOVEL) & union or not set(TRAIN + NOVEL).issubset(absent):
        raise ValueError('An OE TX occurs in a compact dataset or is absent from the declared Full difference')
    capacity = evidence.get('candidate_capacity')
    if not isinstance(capacity, dict):
        raise ValueError('Existing source cell capacities are missing')
    for tx in TRAIN + NOVEL:
        for day in DATES:
            for receiver in RECEIVERS:
                count = capacity.get(tx, {}).get(day, {}).get(receiver)
                if type(count) is not int or count < 70:
                    raise ValueError('Insufficient declared source OE TX/RX/date capacity')
    return evidence, union


def _locations(config):
    root, full = Path(config['validation_root']), Path(config['full_root'])
    summary = _read(root / 'summary.json')
    if (not isinstance(summary, dict) or summary.get('status') != 'VERIFIED'
            or Path(summary.get('dataset_directory', '')).resolve() != full.resolve()
            or type(summary.get('members_checked')) is not int or summary['members_checked'] != 158
            or summary.get('expected_nonempty_RX_date_members') != 158
            or type(summary.get('count_mismatch_cells')) is not int or summary['count_mismatch_cells'] != 0
            or summary.get('errors') != [] or not full.is_dir()):
        raise ValueError('Reuse of the matching completed Full validation is required')
    wanted = {(receiver, day) for receiver in RECEIVERS for day in DATES}
    found = {}
    # Read the existing small index only; never enumerate or revalidate all ZIPs.
    with (root / 'members.jsonl').open('r', encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Invalid existing Full member index row')
            key = (row.get('receiver'), row.get('date'))
            if key not in wanted:
                continue
            receiver, day = key
            member = 'pkl_wifi_eq_' + day + '/dataset_' + day + '_node' + receiver + '.pkl'
            archive = row.get('archive')
            if (key in found or row.get('status') != 'PASS' or row.get('member') != member
                    or not isinstance(archive, str) or Path(archive).name != archive
                    or not archive.startswith('pkl_wifi_eq_' + day + '-') or not archive.endswith('.zip')):
                raise ValueError('The source Full member locator is duplicate or mismatched')
            found[key] = dict(archive=archive, member=member, receiver=receiver, date=day)
    if set(found) != wanted:
        raise ValueError('All fifteen existing source RX/date members must be indexed')
    return summary, [found[receiver, day] for day in DATES for receiver in RECEIVERS]


def _indices(count, seed, receiver, day, tx, per_cell):
    if count < per_cell:
        raise ValueError('Insufficient original Full rows for the source OE allocation')
    rng_seed = int(_identity(['source_oe_selection', seed, receiver, day, tx])[:16], 16)
    return np.random.default_rng(rng_seed).permutation(count)[:per_cell]


def _selected(config, location):
    archive = Path(config['full_root']) / location['archive']
    try:
        with zipfile.ZipFile(archive) as stored:
            with stored.open(location['member'], 'r') as stream:
                payload = _read_numpy_pickle(stream)
    except (KeyError, FileNotFoundError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The indexed source Full NumPy member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full source PKL must contain data/node_list only')
    arrays, names = payload['data'], payload['node_list']
    if (not isinstance(arrays, list) or not isinstance(names, list) or len(arrays) != len(names)
            or any(not isinstance(tx, str) or not tx for tx in names) or len(set(names)) != len(names)):
        raise ValueError('Full source node_list/data association differs')
    lookup = {tx: i for i, tx in enumerate(names)}
    if not set(TRAIN + NOVEL).issubset(lookup):
        raise ValueError('The indexed Full source member lacks a declared OE TX')
    result = []
    for tx in TRAIN + NOVEL:
        values = arrays[lookup[tx]]
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
                or values.ndim != 3 or values.shape[1:] != (256, 2)):
            raise ValueError('Selected Full source IQ must preserve float64[N,256,2]')
        indices = _indices(len(values), config['selection_seed'], location['receiver'], location['date'], tx, 70)
        for index in indices:
            raw = np.array(values[int(index)], copy=True)
            if not np.isfinite(raw).all() or np.max(np.abs(raw)) > np.finfo(np.float32).max:
                raise ValueError('Selected Full source IQ is nonfinite or overflows FP32')
            result.append(dict(tx=tx, raw=raw,
                canonical_record=('FullWiSigEq', tx, location['receiver'], location['date'], 1, int(index))))
    # Only 7*70 selected rows escape; all other arrays are released on return.
    return result


def _save(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def _save_npz(path, **values):
    with Path(path).open('xb') as stream:
        np.savez(stream, **values)


def _write_role(output, role, records):
    if len(records) != COUNTS[role]:
        raise ValueError('The fixed source OE role count differs')
    ids = np.asarray(['p3oe-' + _identity(['source_oe_physical', list(item['canonical_record'])]) for item in records])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Source OE opaque ID collision')
    order = np.argsort(ids, kind='stable')
    records = [records[int(i)] for i in order]
    ids = ids[order]
    iq = np.stack([item.pop('iq') for item in records]).astype(np.float32, copy=False)
    receiver = np.asarray([MAIN_RX[RECEIVERS.index(item['canonical_record'][2])] for item in records], dtype=np.int64)
    day = np.asarray([DATES.index(item['canonical_record'][3]) + 1 for item in records], dtype=np.int64)
    domain = (np.searchsorted(MAIN_RX, receiver) * 3 + day - 1).astype(np.int64)
    tx = np.asarray([item['tx'] for item in records])
    if iq.shape != (COUNTS[role], 2, 256) or not np.isfinite(iq).all():
        raise ValueError('Finite fixed source OE FP32 layout required')
    public, truth = output / role / 'public', output / role / 'truth'
    public.mkdir(parents=True); truth.mkdir()
    _save(public / 'iq.npy', iq); _save(public / 'ids.npy', ids)
    _save_npz(public / 'metadata.npz', rx_i=receiver, day_i=day, domain=domain)
    _save_npz(truth / 'truth.npz', ids=ids, tx=tx)
    capsule = 'source-oe-' + _identity(['source_oe', role, ids.tolist()])
    split = _identity(['source_oe_split', capsule, 70])
    mapping = [dict(opaque_id=str(opaque), canonical_record=list(record['canonical_record']),
                    source_archive=record['source_archive'], source_member=record['source_member'])
               for opaque, record in zip(ids, records)]
    return dict(public_root=str(public), truth_path=str(truth / 'truth.npz'), count=COUNTS[role],
        capsule_id=capsule, split_id=split, data_status='VALIDATED_ONCE',
        iq_shape=[COUNTS[role], 2, 256], iq_dtype='float32'), mapping


def build(config, output_root):
    validate_config(config)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError('Source OE output cannot be overwritten: ' + str(output))
    evidence, union = _memberships(config)
    summary, locations = _locations(config)
    from experiments.phase1_external_dg.data import iq_record
    output.mkdir(parents=True, exist_ok=False)
    records = {role: [] for role in ROLES}
    canonical = set()
    for number, location in enumerate(locations, start=1):
        selected = _selected(config, location)
        for record in selected:
            physical = record['canonical_record']
            if physical in canonical:
                raise ValueError('A source OE physical record was allocated twice')
            canonical.add(physical)
            # The row is already length256/equalized; this is its only RMS pass.
            record['iq'] = iq_record(record.pop('raw'))
            record['source_archive'], record['source_member'] = location['archive'], location['member']
            records['exposure_train' if record['tx'] in TRAIN else 'novel_validation'].append(record)
        print('PHASE3_SOURCE_OE_DATA_PROGRESS', number, 15, location['receiver'], location['date'], len(canonical), 7350, flush=True)
    if len(canonical) != 7350:
        raise ValueError('The complete source OE allocation must contain 7350 unique physical records')
    handles, mapping = {}, {}
    for role in ROLES:
        handles[role], mapping[role] = _write_role(output, role, records[role])
    opaque = [row['opaque_id'] for role in ROLES for row in mapping[role]]
    if len(set(opaque)) != 7350:
        raise ValueError('Source OE public IDs must remain disjoint across roles')
    domains = dict(receiver_names=RECEIVERS, receiver_indices=MAIN_RX, receiver_index_namespace='original Main compact source RX',
        dates=DATES, day_indices=[1, 2, 3], domain_rule='receiver position * 3 + day_i - 1', domain_count=15)
    report = dict(schema='pamf_phase3_source_oe_data_v1', config=config,
        compact_membership_metadata=config['compact_memberships'], compact_union_count=len(union),
        full_tx_count=evidence['full_tx_count'], compact_tx_union_excluded=True,
        known_six_excluded=True, reserved_target_four_excluded=True, reserved_target_classes_used_as_OE=False,
        source_OE_membership_is_additional_supervision=True,
        target_labels_used=False, source_exposure_train_TX_labels_public=False,
        train_and_novel_TX_disjoint=True, canonical_physical_unique=True, physical_records=7350,
        validation_reused=dict(root=config['validation_root'], status=summary['status'], members_checked=158, count_mismatch_cells=0),
        members=locations, archive_reads=15, pickle_reads=15, full_archive_rescan=False, archive_revalidation=False,
        physical_identity='(FullWiSigEq, TX, RX, date, equalized=1, original_packet_index)',
        selection_rule='fixed permutation seeded by selection_seed/RX/date/TX; first70 without quality resampling',
        opaque_ID_rule='physical canonical tuple only; no role or TX text', source_domains=domains,
        input_processing_state='WiSig_equalized1_center256_unit_rms', input_RMS_normalizations_per_record=1,
        crop_or_padding_applied=False, equalization_reapplied=False, channel_observations_generated=0,
        original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0,
        whole_project_never_exposed_claim=False,
        exposure_scope='TX absent from the four existing compact datasets; not an exhaustive project exposure audit',
        mapping=mapping)
    _write(output / 'builder_report.json', report)
    complete = dict(status='SOURCE_OE_DATA_VALIDATED', data_status='VALIDATED_ONCE',
        counts=COUNTS.copy(), exposure_train_count=4200, novel_validation_count=3150,
        exposure_train=handles['exposure_train'], novel_validation=handles['novel_validation'],
        source_domains=domains, source_only=True, target_reads=0,
        original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, channel_observations_generated=0,
        train_and_novel_TX_disjoint=True, canonical_physical_unique=True, unique_canonical_count=7350,
        source_OE_membership_is_additional_supervision=True,
        whole_project_never_exposed_claim=False, config=config, builder_report=str(output / 'builder_report.json'))
    _write(output / 'complete.json', complete)
    print('SOURCE_OE_DATA_VALIDATED', 4200, 3150, 7350, flush=True)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build(_read(args.config), args.output)
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
