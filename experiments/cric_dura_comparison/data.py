"""Bind existing validated roles and add TX-blind source metadata/U caches once."""
import pickle
import hashlib
from pathlib import Path
import numpy as np
from .io import read, write


def bind_classes(contract, metadata):
    """Legacy F0 stores the ordered TX map in cache metadata, not its contract."""
    classes = metadata.get('classes')
    if not isinstance(classes, list) or len(classes) != contract['num_classes'] or len(set(classes)) != len(classes):
        raise ValueError('Validated source class map differs')
    if 'classes' in contract and contract['classes'] != classes:
        raise ValueError('Contract/cache class order differs')
    return dict(contract, classes=classes)


def sample_id(dataset, tx, rx, day, eq, packet):
    key = '%s/%s/%s/%s/%s' % (tx, rx, day, eq, packet)
    if dataset != 'ManySig':
        key = dataset + '/' + key
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def iq_record(raw):
    x = np.asarray(raw, dtype=np.float32).T
    if x.shape[0] != 2:
        raise ValueError('Expected IQ channels')
    if x.shape[1] > 256:
        begin = (x.shape[1] - 256) // 2
        x = x[:, begin:begin + 256]
    elif x.shape[1] < 256:
        value = np.zeros((2, 256), x.dtype)
        begin = (256 - x.shape[1]) // 2
        value[:, begin:begin + x.shape[1]] = x
        x = value
    power = np.mean(x[0] * x[0] + x[1] * x[1])
    if not np.isfinite(power) or power <= 0:
        raise ValueError('Finite nonzero source power required')
    return x / np.sqrt(power + 1e-12)


def prepare(rt, plan):
    root = Path(rt['run_root']) / 'data'
    if (root / 'complete.json').exists():
        return read(root / 'handles.json')
    root.mkdir(parents=True, exist_ok=True)
    previous = read(rt['old_data_handles'])
    counted = read(rt['count_data_handles'])
    handles = {}
    for dataset in ('ManySig', 'ManyRx'):
        with Path(rt['datasets'][dataset]).open('rb') as stream:
            source_data = pickle.load(stream)
        eq = list(source_data['equalized_list']).index(1)
        for pid, p in plan['protocols'].items():
            if p['dataset'] != dataset:
                continue
            old = counted[pid + '-supervised'] if pid.startswith('m') else previous[pid]
            contract = read(old['source_contract'])
            if contract['counts'] != p['expected_counts'] or contract['source_rxs'] != sorted(p['source_rxs']) or contract['source_days'] != p['source_days']:
                raise ValueError('Existing validated source contract differs: ' + pid)
            destination = root / pid
            destination.mkdir(exist_ok=False)
            # The builder alone can enumerate TX to locate IQ. U labels are
            # never written, and train workers do not receive the dataset path.
            lookup = {}
            serial = 0
            for tx in range(len(source_data['tx_list'])):
                for rx in sorted(p['source_rxs']):
                    for day in p['source_days']:
                        cell = source_data['data'][tx][rx][day][eq]
                        for packet in range(0 if cell is None else len(cell)):
                            sid = sample_id(dataset, tx, rx, day, eq, packet)
                            lookup[sid] = (tx, rx, day, eq, packet, serial)
                            serial += 1
            meta = read(Path(old['source_cache']) / 'metadata.json')
            contract = bind_classes(contract, meta)
            role_roots = meta.get('role_cache_roots', {})
            if pid.startswith('m') and '-natural' in pid:
                u_ref = counted.get(pid + '-pl')
            else:
                u_ref = previous.get('F0_PL') if pid == 'F0' else None
            for role in ('L_s', 'U_s', 'V'):
                ids = contract['role_ids'][role]
                records = [lookup[sid] for sid in ids]
                if len(records) != contract['counts'][role] or len(set(ids)) != len(ids):
                    raise ValueError('Physical role mismatch')
                values = {k: np.empty(len(records), np.int64) for k in ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain')}
                values['ids'] = np.asarray(ids)
                if role != 'U_s':
                    values['y'] = np.empty(len(records), np.int64)
                for i, (tx, rx, day, equalized, packet, serial) in enumerate(records):
                    for name, value in zip(('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain'),
                        (rx, day, equalized, packet, serial, sorted(p['source_rxs']).index(rx) * len(p['source_days']) + p['source_days'].index(day))):
                        values[name][i] = value
                    if role != 'U_s':
                        values['y'][i] = tx
                np.savez(destination / (role + '_metadata.npz'), **values)
                existing = Path(role_roots.get(role, old['source_cache'])) / (role + '_x.npy')
                if role == 'U_s' and u_ref:
                    existing = Path(u_ref['source_cache']) / 'U_s_x.npy'
                if existing.is_file():
                    x = np.load(existing, mmap_mode='r', allow_pickle=False)
                    if x.shape != (len(ids), 2, 256):
                        raise ValueError('Reused source IQ shape differs')
                    (destination / (role + '_x.npy')).symlink_to(existing.resolve())
                else:
                    if role != 'U_s':
                        raise ValueError('Existing labeled/validation cache missing')
                    x = np.lib.format.open_memmap(destination / 'U_s_x.npy', mode='w+', dtype=np.float32, shape=(len(records), 2, 256))
                    for i, (tx, rx, day, equalized, packet, _) in enumerate(records):
                        x[i] = iq_record(source_data['data'][tx][rx][day][equalized][packet])
                    x.flush()
                    del x
            # The source contract itself records role identities, not U labels.
            write(destination / 'source_contract.json', contract)
            handles[pid] = dict(source_cache=str(destination), source_contract=str(destination / 'source_contract.json'),
                counts=contract['counts'], tests=old['tests'], classes=contract['classes'])
            print('CRIC_SOURCE_BOUND', pid, contract['counts'], flush=True)
        del source_data
    if len(handles) != 17:
        raise ValueError('All seventeen data contracts required')
    write(root / 'handles.json', handles)
    write(root / 'complete.json', dict(status='VALIDATED_ONCE_HANDLES_REUSED', protocols=17,
        U_TX_labels_exported=False, target_IQ_regenerated=False))
    return handles
