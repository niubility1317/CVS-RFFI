"""Cap source OE cells without replication, channel simulation or model access."""
from __future__ import annotations

from collections import Counter
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

from . import source_oe_data as source
from .data import _identity


CLASSES = ['1-20', '1-9', '10-20', '11-11', '11-14', '13-8', '14-19', '15-15', '17-20', '17-3',
    '18-19', '18-6', '19-5', '2-18', '2-2', '20-10', '20-11', '20-17', '20-6', '3-3', '6-20']
KEYS = {'full_root', 'validation_root', 'compact_memberships', 'known_classes', 'reserved_target_classes',
    'receivers', 'dates', 'exposure_train_classes', 'excluded_novel_classes', 'per_cell', 'selection_seed'}
COUNT = 6134
_read, _write = source._read, source._write


def validate_config(config):
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('Only the fixed source OE diversity configuration is accepted')
    for key in ('full_root', 'validation_root', 'compact_memberships'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError('Existing Full/validated metadata locations are required')
    if (config['known_classes'] != source.KNOWN or config['reserved_target_classes'] != source.RESERVED
            or config['receivers'] != source.RECEIVERS or config['dates'] != source.DATES
            or config['exposure_train_classes'] != CLASSES or config['excluded_novel_classes'] != source.NOVEL
            or type(config['per_cell']) is not int or config['per_cell'] != 32
            or type(config['selection_seed']) is not int or config['selection_seed'] != 2026102906):
        raise ValueError('The fixed 21 source TX, excluded TX, cap32 or selection seed differ')
    return config


def _plan(config):
    evidence = _read(config['compact_memberships'])
    if not isinstance(evidence, dict) or not isinstance(evidence.get('compact'), dict):
        raise ValueError('Completed compact TX membership metadata is required')
    expected = dict(ManySig=6, ManyTx=150, ManyRx=10, SingleDay=28)
    if set(evidence['compact']) != set(expected): raise ValueError('All four compact TX lists are required')
    union = set()
    for name, count in expected.items():
        row = evidence['compact'][name]
        txs = row.get('tx_list') if isinstance(row, dict) else None
        if (not isinstance(txs, list) or len(txs) != count or any(not isinstance(tx, str) or not tx for tx in txs)
                or len(set(txs)) != count or type(row.get('count')) is not int or row['count'] != count):
            raise ValueError('Existing compact TX metadata differs')
        union.update(txs)
    absent = evidence.get('full_tx_absent_all_compact')
    if (len(union) != 150 or evidence.get('compact_union_count') != 150
            or evidence.get('compact_union_not_in_full') != [] or evidence.get('full_tx_count') != 174
            or not isinstance(absent, list) or len(absent) != 24 or len(set(absent)) != 24
            or sorted(set(absent) - set(source.NOVEL)) != CLASSES
            or evidence.get('old_six') != source.KNOWN or evidence.get('formal_unknown_excluded') != source.RESERVED
            or evidence.get('source_receivers') != source.RECEIVERS or evidence.get('source_dates') != source.DATES
            or set(CLASSES) & (union | set(source.KNOWN + source.RESERVED + source.NOVEL))):
        raise ValueError('The predeclared compact-excluded 21-TX source identity differs')
    capacities = evidence.get('candidate_capacity')
    if not isinstance(capacities, dict): raise ValueError('Existing source capacities are required')
    cells, per_tx, per_rx, per_day, per_rx_day = [], Counter(), Counter(), Counter(), Counter()
    for tx in CLASSES:
        if not isinstance(capacities.get(tx), dict): raise ValueError('Missing declared source TX capacity')
        for day in source.DATES:
            row = capacities[tx].get(day)
            if not isinstance(row, dict): raise ValueError('Missing declared source date capacity')
            for receiver in source.RECEIVERS:
                available = row.get(receiver)
                if type(available) is not int or available < 0: raise ValueError('Source capacities must be nonnegative integers')
                selected = min(available, 32)
                cells.append(dict(tx=tx, receiver=receiver, date=day, available=available, selected=selected))
                per_tx[tx] += selected; per_rx[receiver] += selected; per_day[day] += selected
                per_rx_day[receiver + '/' + day] += selected
    nonempty = sum(cell['available'] > 0 for cell in cells)
    original = sum(per_tx[tx] for tx in source.TRAIN)
    total = sum(per_tx.values())
    if (total != COUNT or nonempty != 220 or len(cells) - nonempty != 95
            or original != 1920 or total - original != 4214 or any(per_tx[tx] <= 0 for tx in CLASSES)):
        raise ValueError('Predeclared cap32 allocation differs from 6134/220 nonempty/95 zero cells')
    return capacities, dict(count=total, nonempty_cells=nonempty, zero_cells=len(cells) - nonempty,
        original4_count=original, new17_count=total - original, by_transmitter=dict(per_tx),
        by_receiver=dict(per_rx), by_date=dict(per_day), by_receiver_date=dict(per_rx_day), cells=cells)


def _selected(config, location, capacities):
    try:
        with zipfile.ZipFile(Path(config['full_root']) / location['archive']) as stored:
            with stored.open(location['member'], 'r') as stream: payload = source._read_numpy_pickle(stream)
    except (KeyError, FileNotFoundError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        raise ValueError('The indexed validated source Full member cannot be read') from exc
    if not isinstance(payload, dict) or set(payload) != {'data', 'node_list'}:
        raise ValueError('Full source member must contain data/node_list only')
    arrays, names = payload['data'], payload['node_list']
    if (not isinstance(arrays, list) or not isinstance(names, list) or len(arrays) != len(names)
            or any(not isinstance(tx, str) or not tx for tx in names) or len(set(names)) != len(names)):
        raise ValueError('Source member TX/array association differs')
    lookup = {tx: i for i, tx in enumerate(names)}; result = []
    receiver, day = location['receiver'], location['date']
    for tx in CLASSES:
        available = capacities[tx][day][receiver]
        if not available: continue  # Empty cells create no copies, padding or placeholder samples.
        if tx not in lookup: raise ValueError('Nonempty declared source TX is absent from its indexed member')
        values = arrays[lookup[tx]]
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype('float64')
                or values.shape != (available, 256, 2)):
            raise ValueError('The declared source capacity or original float64[N,256,2] layout differs')
        indices = source._indices(available, config['selection_seed'], receiver, day, tx, min(available, 32))
        for index in indices:
            raw = np.array(values[int(index)], copy=True)
            if not np.isfinite(raw).all() or np.max(np.abs(raw)) > np.finfo(np.float32).max:
                raise ValueError('Selected original source IQ is nonfinite or overflows FP32')
            result.append(dict(raw=raw, tx=tx, canonical_record=('FullWiSigEq', tx, receiver, day, 1, int(index)),
                source_archive=location['archive'], source_member=location['member']))
    return result


def run(config, output_root):
    validate_config(config)
    output = Path(output_root)
    if output.exists(): raise FileExistsError('A new immutable source OE diversity root is required')
    capacities, allocation = _plan(config)
    summary, locations = source._locations(config)
    from experiments.phase1_external_dg.data import iq_record
    output.mkdir(parents=True, exist_ok=False)
    records, seen = [], set()
    for number, location in enumerate(locations, start=1):
        selected = _selected(config, location, capacities)
        for record in selected:
            physical = record['canonical_record']
            if physical in seen: raise ValueError('A source OE physical record was selected more than once')
            seen.add(physical); record['iq'] = iq_record(record.pop('raw')); records.append(record)
        print('SOURCE_OE_DIVERSE_DATA_PROGRESS', number, 15, location['receiver'], location['date'], len(records), COUNT, flush=True)
    if len(records) != COUNT: raise ValueError('The realized diverse source OE count differs')
    actual_cells = Counter((record['tx'], record['canonical_record'][2], record['canonical_record'][3]) for record in records)
    if any(actual_cells[cell['tx'], cell['receiver'], cell['date']] != cell['selected'] for cell in allocation['cells']):
        raise ValueError('The realized source TX/RX/day cap allocation differs')
    ids = np.asarray(['p3oe-' + _identity(['source_oe_physical', list(record['canonical_record'])]) for record in records])
    if len(set(ids.tolist())) != COUNT: raise ValueError('Diverse source OE public ID collision')
    order = np.argsort(ids, kind='stable'); ids = ids[order]; records = [records[int(i)] for i in order]
    iq = np.stack([record.pop('iq') for record in records]).astype(np.float32, copy=False)
    rx = np.asarray([source.MAIN_RX[source.RECEIVERS.index(record['canonical_record'][2])] for record in records], dtype=np.int64)
    day = np.asarray([source.DATES.index(record['canonical_record'][3]) + 1 for record in records], dtype=np.int64)
    domain = (np.searchsorted(source.MAIN_RX, rx) * 3 + day - 1).astype(np.int64)
    public, truth = output / 'exposure_train/public', output / 'exposure_train/truth'
    public.mkdir(parents=True); truth.mkdir()
    source._save(public / 'iq.npy', iq); source._save(public / 'ids.npy', ids)
    source._save_npz(public / 'metadata.npz', rx_i=rx, day_i=day, domain=domain)
    source._save_npz(truth / 'truth.npz', ids=ids, tx=np.asarray([record['tx'] for record in records]))
    mapping = [dict(opaque_id=str(identifier), canonical_record=list(record['canonical_record']),
        source_archive=record['source_archive'], source_member=record['source_member']) for identifier, record in zip(ids, records)]
    capsule = 'source-oe-diverse-' + _identity([32, config['selection_seed'], ids.tolist()])
    handle = dict(public_root=str(public), truth_path=str(truth / 'truth.npz'), count=COUNT,
        capsule_id=capsule, split_id=_identity(['source_oe_diverse_split', capsule]),
        data_status='VALIDATED_ONCE', iq_shape=[COUNT, 2, 256], iq_dtype='float32')
    domains = dict(receiver_names=source.RECEIVERS, receiver_indices=source.MAIN_RX, dates=source.DATES,
        day_indices=[1, 2, 3], domain_count=15, domain_rule='source RX position * 3 + day_i - 1')
    _write(output / 'builder_report.json', dict(schema='phase3_source_oe_diverse_builder_v1', config=config,
        allocation=allocation, mapping=mapping, members=locations, archive_reads=15, pickle_reads=15,
        validation_reused=dict(root=config['validation_root'], status=summary['status'], members_checked=158),
        full_archive_rescan=False, archive_revalidation=False, samples_replicated=False, sample_weights_assigned=False,
        original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0, novel_validation_IQ_reads=0, target_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, channel_observations_generated=0,
        input_processing_state='WiSig_equalized1_center256_unit_rms', input_RMS_normalizations_per_record=1,
        crop_or_padding_applied=False, equalization_reapplied=False, source_OE_membership_is_additional_supervision=True,
        whole_project_never_exposed_claim=False,
        physical_identity='(FullWiSigEq, TX, RX, date, equalized=1, original_packet_index)'))
    complete = dict(status='SOURCE_OE_DIVERSE_DATA_VALIDATED', data_status='VALIDATED_ONCE',
        count=COUNT, counts={'exposure_train': COUNT}, exposure_train_count=COUNT, exposure_train=handle,
        source_domains=domains, source_only=True, target_reads=0,
        original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0, novel_validation_IQ_reads=0,
        model_reads=0, model_predictions=0, optimizer_updates=0, channel_observations_generated=0,
        canonical_physical_unique=True, unique_canonical_count=COUNT,
        source_OE_membership_is_additional_supervision=True, builder_report=str(output / 'builder_report.json'))
    _write(output / 'complete.json', complete)
    print('SOURCE_OE_DIVERSE_DATA_VALIDATED', COUNT, flush=True)
    return complete


build = run


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_OE_DIVERSE_RUNTIME', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv), ensure_ascii=False), flush=True)
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv); result = run(_read(args.config), args.output)
    print(json.dumps(dict(status=result['status'], count=result['count']))); return 0


if __name__ == '__main__': raise SystemExit(main())
