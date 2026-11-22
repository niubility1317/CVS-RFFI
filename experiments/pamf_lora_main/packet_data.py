"""Bind an immutable LoRa pilot to the original roles and full-rate IQ."""
import argparse
from pathlib import Path
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import CLASSES, ROLES, SOURCE, TARGET, packet_indices, recording


PILOT_SEED = 2026112207
TARGET_LIMIT = 128
PACKET_LEN = 8192
CHIRPS = (1, 2, 3, 4, 5, 6)
SINGLE_INDEX = CHIRPS.index(3)
ROLE_COUNTS = {'L_s': 1600, 'U_s': 2560, 'V': 800}


def selection(m, role):
    if ('y' in m.files) != (role != 'U_s'):
        raise ValueError('Original source role and label visibility differ')
    rng = np.random.default_rng([PILOT_SEED, ROLES.index(role)])
    if role == 'U_s':
        groups = [np.flatnonzero(m['rx_i'] == r) for r in range(len(SOURCE))]
        count = 512
    else:
        groups = [np.flatnonzero((m['rx_i'] == r) & (m['y'] == y))
                  for r in range(len(SOURCE)) for y in range(len(CLASSES))]
        count = 32 if role == 'L_s' else 16
    return np.sort(np.concatenate([rng.choice(g, size=count, replace=False) for g in groups]))


def read_iq(h5, rows):
    data = h5['data']
    if data.shape[1:] != (PACKET_LEN * 2,):
        raise ValueError('Expected official I then Q packet')
    order = np.argsort(rows)
    raw = data[rows[order]].astype(np.float32)
    z = raw[:, :PACKET_LEN] + np.complex64(1j) * raw[:, PACKET_LEN:]
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite native-rate LoRa IQ')
    return z[np.argsort(order)]


def prepare_source(archives, previous, out, scratch):
    contract = read(previous/'contract.json')
    if contract['source_rxs'] != list(range(len(SOURCE))) or contract['receiver_names'] != list(SOURCE):
        raise ValueError('Original LoRa source definition differs')
    roles = {}
    for role in ROLES:
        with np.load(previous/f'{role}_metadata.npz', allow_pickle=False) as m:
            position = selection(m, role)
            meta = {k: m[k][position] for k in m.files}
        if (len(position) != ROLE_COUNTS[role] or
                meta['ids'].tolist() != [contract['role_ids'][role][i] for i in position]):
            raise ValueError('LoRa pilot roles not bound to original physical IDs')
        roles[role] = meta
        np.savez(out/f'{role}_metadata.npz', **meta)
    values = {r: np.empty((len(m['ids']), PACKET_LEN), np.complex64) for r, m in roles.items()}
    with zipfile.ZipFile(archives/'multiple_receiver_train.zip') as archive:
        for rx, name in enumerate(SOURCE):
            with recording(archive, f'{name}_train.h5', scratch) as h5:
                order = packet_indices(8000, rx)
                if len(h5['data']) != len(order):
                    raise ValueError('Source physical packet count differs')
                for role, m in roles.items():
                    mask = m['rx_i'] == rx
                    base = m['base_index'][mask] - rx*len(order)
                    values[role][mask] = read_iq(h5, order[base])
            print('PACKET_SOURCE', name, flush=True)
    for role, iq in values.items():
        np.save(out/f'{role}_packets.npy', iq)
    write(out/'contract.json', dict(protocol='lora_packet_pilot_v1', source_only=True, chirps=list(CHIRPS),
          source_receivers=list(SOURCE), counts=ROLE_COUNTS, seed=PILOT_SEED,
          role_ids={r:m['ids'].tolist() for r,m in roles.items()}))


def prepare_target(archives, out, scratch):
    with zipfile.ZipFile(archives/'multiple_receiver_test.zip') as archive:
        for rx, name in enumerate(TARGET):
            with recording(archive, f'{name}_test.h5', scratch) as h5:
                if len(h5['data']) != 2000:
                    raise ValueError('Original target packet count differs')
                z = read_iq(h5, packet_indices(2000, rx)[:TARGET_LIMIT])
            np.save(out/f'{name}_packets.npy', z)
            print('PACKET_TARGET_BLIND', name, flush=True)
    write(out/'contract.json', dict(protocol='lora_packet_pilot_v1', receivers=list(TARGET),
          per_receiver=TARGET_LIMIT, TX_truth_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=('source','target'))
    for name in ('archives','previous','output','scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.part == 'source':
        prepare_source(a.archives, a.previous, a.output, a.scratch)
    else:
        prepare_target(a.archives, a.output, a.scratch)
