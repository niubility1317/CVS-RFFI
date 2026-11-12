"""Build an independent clean-IQ bank and physical-ID mappings for Phase2.

This offline data builder reads identity metadata on the truth side but never
opens query labels or produces predictions. The original capsule and score
files remain untouched. The model-facing output contains no physical IDs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


RECEIVERS = ('1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8')
POOLS = ('P01', 'P02', 'P03')
EPISODES = tuple([f'MIX{i:02d}' for i in range(1, 9)] +
                 [f'{p}{i:02d}' for p in ('LS', 'MU', 'LU') for i in range(1, 6)])
KS = (1, 5, 10, 20)
NS = (0, 2, 5, 10, 20)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def map_ids(bank_ids, target_ids):
    bank_ids = np.asarray(bank_ids)
    target_ids = np.asarray(target_ids)
    check(bank_ids.ndim == target_ids.ndim == 1, 'IDs must be vectors')
    check(len(np.unique(bank_ids)) == len(bank_ids), 'duplicate source physical IDs')
    check(len(np.unique(target_ids)) == len(target_ids), 'duplicate target physical IDs')
    order = np.argsort(bank_ids)
    pos = np.searchsorted(bank_ids[order], target_ids)
    check((pos < len(order)).all(), 'unknown physical ID')
    check(np.array_equal(bank_ids[order[pos]], target_ids), 'unknown physical ID')
    return order[pos].astype(np.int32)


def normalize_batch(raw):
    raw = np.asarray(raw)
    check(raw.ndim == 3 and raw.shape[1:] == (2, 256), 'invalid IQ shape')
    check(np.isfinite(raw).all(), 'nonfinite ground IQ')
    z = raw[:, 0, :].astype(np.float64) + 1j * raw[:, 1, :].astype(np.float64)
    rms = np.sqrt(np.mean(np.abs(z) ** 2, axis=1))
    check(np.isfinite(rms).all() and (rms > 0).all(), 'invalid IQ RMS')
    z = z / rms[:, None]
    return np.stack((z.real, z.imag), axis=1).astype(np.float32)


def validate_splits(capsule, manifest):
    specs = [read_json(p) for p in (capsule / 'splits').glob('*.json')]
    lookup = {(s['k'], s['new_count']): s for s in specs}
    check(len(specs) == 20 and set(lookup) == {(k, n) for k in KS for n in NS}, 'wrong split matrix')
    count = manifest['received_count']
    for n in NS:
        registered = np.load(capsule / 'stage_rows' / f'N{n:02d}.npy', allow_pickle=False)
        check(registered.ndim == 1 and registered.dtype == np.int32, 'bad registered indices')
        check(len(np.unique(registered)) == len(registered) and
              ((registered >= 0) & (registered < count)).all(), 'invalid registered indices')
        previous = None
        for k in KS:
            s = lookup[k, n]
            check(s['capsule_id'] == manifest['capsule_id'] and s['protocol_schema'] == 'p2_min_v1'
                  and s['phase2_data_status'] == 'VALIDATED_ONCE', 'split identity/contract mismatch')
            support = np.asarray(s['support_indices'], dtype=np.int64)
            check(len(support) == k * (6 + n) and len(np.unique(support)) == len(support), 'bad support')
            check(np.isin(support, registered).all(), 'support outside registered pool')
            check(s['query_count'] == len(registered) - len(support), 'bad query complement count')
            labels = np.load(capsule / 'support_labels' / (s['split_id'] + '.npy'), allow_pickle=False)
            check(np.array_equal(labels, np.repeat(np.arange(6 + n), k)), 'bad support labels')
            if previous is not None:
                prev_k, prev = previous
                check(np.array_equal(prev, support.reshape(6+n, k)[:, :prev_k]), 'K prefix mismatch')
            previous = (k, support.reshape(6+n, k))
            if n:
                old = np.asarray(lookup[k, 0]['support_indices'])
                check(np.array_equal(old, support[:6*k]), 'old support differs across B/C')
    return len(specs)


def build(raw_root, task_root, out, batch_size=1024):
    started = time.monotonic()
    raw_root, task_root, out = map(Path, (raw_root, task_root, out))
    check(raw_root.is_dir() and task_root.is_dir(), 'input roots do not exist')
    check(batch_size > 0, 'invalid batch size')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'banks').mkdir()
    (out / 'capsules').mkdir()
    per_receiver = {}
    mapping_count = split_count = 0
    for rx in RECEIVERS:
        raw = np.load(raw_root / rx / 'raw_iq.npy', mmap_mode='r', allow_pickle=False)
        with np.load(raw_root / rx / 'records.npz', allow_pickle=False) as source:
            source_ids = source['physical_ids']
        check(raw.dtype == np.float32 and raw.shape == (len(source_ids), 2, 256), 'bad source IQ')
        with np.load(task_root / 'builder_only' / 'partition' / rx / 'partition.npz', allow_pickle=False) as part:
            partition_ids = part['physical_ids']
            partition = part['pool_indices']
        check(np.array_equal(source_ids, partition_ids), 'partition/source physical order mismatch')
        check(partition.shape == (len(source_ids),) and set(np.unique(partition)) == {0, 1, 2}, 'invalid pools')
        target = out / 'banks' / rx
        target.mkdir()
        bank = np.lib.format.open_memmap(target / 'clean.npy', mode='w+', dtype=np.float32, shape=raw.shape)
        for start in range(0, len(raw), batch_size):
            bank[start:start+batch_size] = normalize_batch(raw[start:start+batch_size])
        bank.flush()
        per_receiver[rx] = len(source_ids)
        for pool_index, pool in enumerate(POOLS):
            pool_rows = np.flatnonzero(partition == pool_index)
            for episode in EPISODES:
                cap = task_root / 'capsules' / pool / episode / rx
                manifest = read_json(cap / 'manifest.json')
                check(manifest['protocol_schema'] == 'p2_min_v1' and manifest['phase2_data_status'] == 'VALIDATED_ONCE', 'bad capsule')
                ids = np.load(cap / 'ids.npy', allow_pickle=False)
                with np.load(task_root / 'score_only' / pool / episode / (rx + '.npz'), allow_pickle=False) as score:
                    physical_ids = score['physical_ids']
                    observation_ids = score['observation_ids']
                check(np.array_equal(ids, observation_ids), 'capsule/score observation IDs differ')
                check(len(ids) == manifest['received_count'] == len(pool_rows), 'capsule row count mismatch')
                rows = map_ids(source_ids, physical_ids)
                check(np.array_equal(np.sort(rows), pool_rows), 'episode is not the declared physical pool')
                split_count += validate_splits(cap, manifest)
                dest = out / 'capsules' / pool / episode / rx
                dest.mkdir(parents=True)
                np.save(dest / 'row_to_clean.npy', rows, allow_pickle=False)
                write_json(dest / 'manifest.json', {'capsule_id': manifest['capsule_id'], 'receiver': rx,
                    'pool': pool, 'episode': episode, 'rows': len(rows),
                    'input_condition': 'clean_ground_pre_satellite_normalized'})
                mapping_count += 1
        print('CLEAN_RECEIVER_READY', rx, len(source_ids), 'mappings', mapping_count, flush=True)
    check(mapping_count == 483 and split_count == 9660, 'incomplete full task matrix')
    result = {'status': 'CLEAN_PAIRED_COMPLETE', 'input_condition': 'clean_ground_pre_satellite_normalized',
              'raw_root': str(raw_root), 'parent_task_root': str(task_root), 'output_root': str(out),
              'receivers': per_receiver, 'physical_records': sum(per_receiver.values()),
              'capsules': mapping_count, 'splits': split_count, 'no_satellite_overlay': True,
              'prediction_or_training_performed': False, 'query_labels_read': False,
              'elapsed_seconds': time.monotonic() - started}
    write_json(out / 'completion.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--task-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--batch-size', type=int, default=1024)
    a = p.parse_args()
    build(a.raw_root, a.task_root, a.output_root, a.batch_size)


if __name__ == '__main__':
    main()
