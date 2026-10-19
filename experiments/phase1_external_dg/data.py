"""Build source roles and shared received-IQ views once, before any training.

Builders may see dataset metadata to assign roles. Training configs receive only
L_s/V caches, never the dataset or target/truth paths. Dataset IDs are identities,
not an additional release-hash or authorization mechanism.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

from .plan import VIEWS, TARGET_F0

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def sample_id(dataset, tx, rx, day, eq, packet):
    key = '%s/%s/%s/%s/%s' % (tx, rx, day, eq, packet)
    if dataset != 'ManySig': key = dataset + '/' + key
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def check_protocol(p):
    if p['dataset'] not in ('ManySig', 'ManyRx') or p['equalized'] != 1:
        raise ValueError('Unregistered dataset or representation')
    if (p['out_len'] != 256 or not p['normalize'] or p['U_s_use'] or
        p['split_seed'] != 392005 or set(p['source_rxs']) & set(p['target_rxs'])):
        raise ValueError('Illegal source/target role contract')
    if p['fixed_labels'] is not None and (p['dataset'] != 'ManySig' or p['fixed_labels'] != 1260):
        raise ValueError('Unregistered fixed label budget')
    if p['label_per_thousand'] not in (7, 35, 70):
        raise ValueError('Unregistered training label budget')


def iq_record(raw):
    """Same per-packet float32 crop and RMS arithmetic as existing data code."""
    import numpy as np
    x = np.asarray(raw, dtype=np.float32).T
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError('Expected packet x IQ layout')
    if x.shape[1] > 256:
        start = (x.shape[1] - 256) // 2; x = x[:, start:start + 256]
    elif x.shape[1] < 256:
        out = np.zeros((2, 256), dtype=x.dtype)
        start = (256 - x.shape[1]) // 2; out[:, start:start + x.shape[1]] = x; x = out
    power = np.mean(x[0] * x[0] + x[1] * x[1])
    if not np.isfinite(power) or power <= 0:
        raise ValueError('Nonfinite or zero-energy IQ')
    return x / np.sqrt(power + 1e-12)


def selections(data, p):
    """Return disjoint source physical records. Never enumerate target IQ."""
    check_protocol(p)
    classes = len(data['tx_list']); eq = list(data['equalized_list']).index(1)
    cells = [(rx, day) for rx in sorted(p['source_rxs']) for day in p['source_days']]
    fixed = p['fixed_labels']
    if fixed is not None:
        if fixed % classes: raise ValueError('Fixed budget cannot be class-balanced')
        quotient, remainder = divmod(fixed // classes, len(cells))
    result = {r: [] for r in ('L_s', 'U_s', 'V')}
    for tx in range(classes):
        for cell_index, (rx, day) in enumerate(cells):
            array = data['data'][tx][rx][day][eq]
            n = 0 if array is None else len(array)
            natural_l, natural_u = n * 7 // 100, n * 63 // 100
            validation_start = natural_l + natural_u
            label_count = n * p['label_per_thousand'] // 1000
            if fixed is not None:
                label_count = quotient + (cell_index < remainder)
            if label_count > natural_l:
                raise ValueError('Cannot enlarge visible labels beyond original L_s')
            for packet in range(n):
                role = 'L_s' if packet < label_count else 'U_s' if packet < validation_start else 'V'
                sid = sample_id(p['dataset'], tx, rx, day, eq, packet)
                result[role].append((tx, rx, day, eq, packet, sid))
    sets = {r: {v[-1] for v in records} for r, records in result.items()}
    if any(len(sets[r]) != len(result[r]) for r in result):
        raise ValueError('Repeated physical source record')
    if sets['L_s'] & sets['U_s'] or sets['L_s'] & sets['V'] or sets['U_s'] & sets['V']:
        raise ValueError('Source physical roles overlap')
    if fixed is not None and len(result['L_s']) != fixed:
        raise ValueError('Fixed source label count changed')
    return result


def build_source(data, p, out):
    import numpy as np
    records = selections(data, p); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    classes = [str(x) for x in data['tx_list']]
    counts = {r: len(v) for r, v in records.items()}
    if not counts['L_s'] or not counts['V']:
        raise ValueError('Empty source train/validation set')
    contract = dict(schema='phase1_external_roles_v1', dataset=p['dataset'],
        source_rxs=p['source_rxs'], source_days=p['source_days'], classes=classes,
        num_classes=len(classes), counts=counts, split_seed=392005,
        split_rule='per_tx_rx_day_eq_contiguous_signal_index',
        label_per_thousand=p['label_per_thousand'], fixed_labels=p['fixed_labels'],
        role_ids={r: [x[-1] for x in items] for r, items in records.items()},
        checkpoint_init='scratch_only', target_access_before_freeze=False,
        equalized=1, out_len=256, normalize=True, U_s_use=False)
    for role in ('L_s', 'V'):
        items = records[role]
        x = np.lib.format.open_memmap(out / (role + '_x.npy'), mode='w+',
            dtype=np.float32, shape=(len(items), 2, 256))
        y = np.empty(len(items), dtype=np.int64); rx = np.empty_like(y)
        ids = []
        for i, (txi, rxi, day, eq, packet, sid) in enumerate(items):
            x[i] = iq_record(data['data'][txi][rxi][day][eq][packet])
            y[i] = txi; rx[i] = p['source_rxs'].index(rxi); ids.append(sid)
        x.flush(); del x
        if set(y.tolist()) != set(range(len(classes))):
            raise ValueError('A source class is absent')
        np.savez(out / (role + '_index.npz'), y=y, rx=rx, ids=np.asarray(ids))
    write(out / 'source_contract.json', contract)
    write(out / 'metadata.json', dict(source_contract=contract, classes=classes,
        equalized=1, normalize=True, out_len=256, counts=counts,
        U_s_exported=False, target_constructed=False))
    print('SOURCE_PREPARED', p['protocol_id'], counts, flush=True)
    return str(out / 'source_contract.json')


def inventory(data, dataset, receivers):
    eq = list(data['equalized_list']).index(1); records = []
    for tx in range(len(data['tx_list'])):
        for rx in receivers:
            for day in range(len(data['capture_date_list'])):
                array = data['data'][tx][rx][day][eq]
                for packet in range(0 if array is None else len(array)):
                    records.append((tx, rx, day, eq, packet,
                        sample_id(dataset, tx, rx, day, eq, packet)))
    if len({x[-1] for x in records}) != len(records):
        raise ValueError('Union physical identities repeat')
    return records


def generate_chunk(task):
    import numpy as np
    from leo_practical import Config, apply_leo_practical_channel_batch
    root, scene, positions, configuration = task
    root = Path(root)
    with np.load(root / 'index.npz', allow_pickle=False) as ix:
        ids = ix['ids'][positions].tolist()
    x = np.load(root / 'clean.npy', mmap_mode='r', allow_pickle=False)[positions]
    y, meta, _ = apply_leo_practical_channel_batch(np.asarray(x, dtype=np.float64),
        Config(**configuration), seed=392005, sample_ids=ids,
        session_ids=['virtual_target_session0'] * len(ids),
        realization_namespace='phase1_full_six_fixed_20261002_v1',
        receiver_seed=2027, return_meta=True)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError('Invalid received observations')
    if [m['sample_id'] for m in meta] != ids or any(m['channel_equalization_applied'] for m in meta):
        raise ValueError('Observation identity or receiver route mismatch')
    return positions, y, dict(Counter(m['receiver_lock_status'] for m in meta))


def build_union(data, dataset, root, existing_views):
    """Copy existing ManySig observations; synthesize only previously absent IDs."""
    import numpy as np
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    receivers = list(range(12)) if dataset == 'ManySig' else list(range(1, 32, 2))
    records = inventory(data, dataset, receivers)
    ids = np.asarray([x[-1] for x in records]); lookup = {sid: i for i, sid in enumerate(ids)}
    np.savez(root / 'index.npz', ids=ids)
    original = read(Path(existing_views) / 'manifest.json')
    if original['status'] != 'VALIDATED_ONCE' or original['scenes'] != list(VIEWS[1:]):
        raise ValueError('Existing sixscene handles differ')
    old_positions = new_positions = None
    if dataset == 'ManySig':
        with np.load(Path(existing_views) / 'index.npz', allow_pickle=False) as ix:
            old_ids = ix['ids'].tolist()
        if not set(old_ids) <= set(lookup):
            raise ValueError('Existing observations not covered by union')
        old_positions = np.arange(len(old_ids)); new_positions = np.asarray([lookup[x] for x in old_ids])
    clean = np.lib.format.open_memmap(root / 'clean.npy', mode='w+',
        dtype=np.float32, shape=(len(ids), 2, 256))
    for i, (tx, rx, day, eq, packet, _) in enumerate(records):
        clean[i] = iq_record(data['data'][tx][rx][day][eq][packet])
    if new_positions is not None:
        old_clean = np.load(original['clean_ref'], mmap_mode='r', allow_pickle=False)
        for start in range(0, len(old_positions), 1000):
            clean[new_positions[start:start + 1000]] = old_clean[old_positions[start:start + 1000]]
    clean.flush(); del clean
    unknown = np.arange(len(ids))
    if new_positions is not None: unknown = np.setdiff1d(unknown, new_positions)
    evidence = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for item in original['evidence']:
            scene = item['scene']
            dest = np.lib.format.open_memmap(root / (scene + '.npy'), mode='w+',
                dtype=np.float32, shape=(len(ids), 2, 256))
            copied = 0
            if new_positions is not None:
                prior = np.load(Path(existing_views) / (scene + '.npy'), mmap_mode='r', allow_pickle=False)
                for start in range(0, len(old_positions), 1000):
                    dest[new_positions[start:start + 1000]] = prior[old_positions[start:start + 1000]]
                copied = len(new_positions)
            tasks = [(str(root), scene, unknown[start:start + 1000], item['config'])
                     for start in range(0, len(unknown), 1000)]
            generated = 0; lock_counts = Counter()
            for positions, values, locks in pool.map(generate_chunk, tasks, chunksize=1):
                dest[positions] = values; generated += len(positions); lock_counts.update(locks)
            dest.flush(); del dest
            if copied + generated != len(ids):
                raise ValueError('Incomplete observations')
            evidence.append(dict(scene=scene, copied=copied, generated=generated,
                generated_locks=dict(lock_counts), config=item['config']))
            print('UNION_VIEW_COMPLETE', dataset, scene, copied, generated, flush=True)
    write(root / 'manifest.json', dict(status='OBSERVATIONS_COMPLETE', dataset=dataset,
        classes=[str(v) for v in data['tx_list']], count=len(ids), scenes=list(VIEWS[1:]),
        seed=392005, receiver_seed=2027, rows_share_observations=True,
        seconds=time.monotonic() - started, evidence=evidence,
        note='Union storage has no source/target role; each protocol wrapper validates its own roles.'))
    return records


def make_wrapper(data, p, source_contract, union_root, records, root, *, diagnostic=False):
    import numpy as np
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    receivers = p['source_rxs'] if diagnostic else p['target_rxs']
    days = [p['diagnostic_day']] if diagnostic else p['target_days']
    if diagnostic and (days[0] is None or days[0] in p['source_days']):
        raise ValueError('Diagnostic date was present during source training')
    positions = [i for i, r in enumerate(records) if r[1] in receivers and r[2] in days]
    ids = [records[i][-1] for i in positions]
    contract = read(source_contract)
    source_ids = {sid for values in contract['role_ids'].values() for sid in values}
    if not ids or len(set(ids)) != len(ids) or source_ids & set(ids):
        raise ValueError('Invalid target physical roles')
    np.savez(root / 'index.npz', ids=np.asarray(ids), positions=np.asarray(positions, dtype=np.int64))
    truth = {records[i][-1]: dict(label=records[i][0],
        receiver=str(data['rx_list'][records[i][1]]),
        day=str(data['capture_date_list'][records[i][2]])) for i in positions}
    # The predictor receives only views/; truth resides outside that folder.
    truth_path = root.parent / (root.name + '_truth.json'); write(truth_path, truth)
    write(root / 'manifest.json', dict(status='VALIDATED_ONCE', count=len(ids),
        classes=[str(v) for v in data['tx_list']], scenes=list(VIEWS[1:]),
        clean_ref=str(Path(union_root) / 'clean.npy'), observations_root=str(union_root),
        source_target_disjoint=True, receiver_disjoint=not diagnostic,
        diagnostic_unseen_date=diagnostic, source_days=p['source_days'],
        target_days=days, target_rxs=receivers, dataset=p['dataset'], unique_ids=True,
        seed=392005, receiver_seed=2027, channel='residual/post_sync/noeq', truth_read=False))
    return dict(views_root=str(root), truth=str(truth_path),
        kind='source_unseen_date' if diagnostic else 'unseen_receiver')


def prepare(rt, plan):
    """One builder execution. Existing F0 data is reused without revalidation."""
    root = Path(rt['run_root']); data_root = root / 'data'
    data_root.mkdir(parents=True, exist_ok=False)
    handles = {}
    datasets = {}
    for dataset in ('ManySig', 'ManyRx'):
        with Path(rt['datasets'][dataset]).open('rb') as stream:
            datasets[dataset] = pickle.load(stream)
        actual = datasets[dataset]
        expected = (6, 12) if dataset == 'ManySig' else (10, 32)
        if (len(actual['tx_list']), len(actual['rx_list'])) != expected:
            raise ValueError('Dataset class/RX metadata changed')
        if [str(v) for v in actual['capture_date_list']] != ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23']:
            raise ValueError('Date metadata changed')
    for pid, p in plan['protocols'].items():
        check_protocol(p)
        if pid == 'F0':
            meta = read(Path(rt['existing_source_cache']) / 'metadata.json')
            if meta['counts'] != dict(L_s=6300, U_s=56700, V=27000) or meta['U_s_exported']:
                raise ValueError('Existing F0 source handle mismatch')
            contract_path = rt['existing_source_contract']
            source_cache = rt['existing_source_cache']
        else:
            source_cache = str(data_root / 'source' / pid)
            contract_path = build_source(datasets[p['dataset']], p, source_cache)
        handles[pid] = dict(source_cache=source_cache, source_contract=contract_path,
            counts=read(Path(source_cache) / 'metadata.json')['counts'])
    unions = {}
    for dataset in ('ManySig', 'ManyRx'):
        union_root = data_root / 'observations' / dataset
        records = build_union(datasets[dataset], dataset, union_root, rt['existing_views_root'])
        unions[dataset] = (union_root, records)
    for pid, p in plan['protocols'].items():
        if p['dataset'] == 'ManySig' and p['target_rxs'] == list(TARGET_F0) and p['target_days'] == [0, 1, 2, 3]:
            main = dict(views_root=rt['existing_views_root'], truth=rt['existing_truth'], kind='unseen_receiver')
        else:
            union_root, records = unions[p['dataset']]
            main = make_wrapper(datasets[p['dataset']], p, handles[pid]['source_contract'],
                union_root, records, data_root / 'views' / (pid + '-main'))
        handles[pid]['tests'] = dict(main=main)
        if p['diagnostic_day'] is not None:
            union_root, records = unions[p['dataset']]
            handles[pid]['tests']['source_unseen_date'] = make_wrapper(
                datasets[p['dataset']], p, handles[pid]['source_contract'], union_root,
                records, data_root / 'views' / (pid + '-source-unseen-date'), diagnostic=True)
    write(root / 'data_handles.json', handles)
    print('DATA_PREPARED', len(handles), 'source protocols; no training target access', flush=True)
    return handles
