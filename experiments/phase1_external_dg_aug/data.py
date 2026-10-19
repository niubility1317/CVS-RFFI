"""Reuse validated observations and export source U IQ without transmitter labels."""
from copy import deepcopy
from pathlib import Path
import pickle

from experiments.phase1_external_dg.data import read, write, selections, iq_record


def build_unlabeled(data, parent, protocol, destination):
    import numpy as np
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    p = dict(protocol, U_s_use=False)
    records = selections(data, p)['U_s']
    original = read(parent['source_contract'])
    expected = original['role_ids']['U_s']
    ids = [r[-1] for r in records]
    if len(ids) != 56700 or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('PL IQ does not match the existing legal source U role')
    x = np.lib.format.open_memmap(destination / 'U_s_x.npy', mode='w+', dtype=np.float32,
                                  shape=(len(records), 2, 256))
    receivers = np.empty(len(records), dtype=np.int64)
    for i, (tx, rx, day, eq, packet, _) in enumerate(records):
        x[i] = iq_record(data['data'][tx][rx][day][eq][packet])
        receivers[i] = protocol['source_rxs'].index(rx)
    x.flush(); del x
    # TX is used only by the role builder to locate IQ. It is never written.
    np.savez(destination / 'U_s_index.npz', ids=np.asarray(ids), rx=receivers)
    contract = deepcopy(original)
    contract.update(U_s_use=True, U_s_labels_hidden=True)
    metadata = deepcopy(read(Path(parent['source_cache']) / 'metadata.json'))
    metadata.update(source_contract=contract, U_s_exported=True, U_s_labels_exported=False,
        target_constructed=False, labeled_cache_ref=parent['source_cache'],
        role_cache_roots=dict(L_s=parent['source_cache'], V=parent['source_cache'], U_s=str(destination)))
    write(destination / 'source_contract.json', contract)
    write(destination / 'metadata.json', metadata)
    with np.load(destination / 'U_s_index.npz', allow_pickle=False) as index:
        if set(index.files) != {'ids', 'rx'}:
            raise ValueError('Unlabeled export contains forbidden fields')
    return dict(source_cache=str(destination), source_contract=str(destination / 'source_contract.json'),
                counts=metadata['counts'], tests=deepcopy(parent['tests']))


def prepare(rt):
    from .plan import make_plan
    root = Path(rt['run_root'])
    original = read(Path(rt['reused_suite_root']) / 'data_handles.json')
    plan = make_plan()
    # These same physical source and received-IQ handles have already passed
    # VALIDATED_ONCE. No target IQ regeneration, hashing or revalidation occurs.
    if set(original) != set(plan['protocols']) - {'F0_PL'}:
        raise ValueError('Missing registered reusable source/test handles')
    handles = deepcopy(original)
    with Path(rt['datasets']['ManySig']).open('rb') as stream:
        data = pickle.load(stream)
    handles['F0_PL'] = build_unlabeled(data, original['F0'], plan['protocols']['F0_PL'],
                                      root / 'data/source/F0_PL')
    del data
    write(root / 'data_handles.json', handles)
    print('DATA_PREPARED_REUSED', len(handles), 'protocols; U TX labels hidden; target unchanged', flush=True)
    return handles
