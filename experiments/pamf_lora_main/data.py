"""Convert the official LoRa archives into separate source and blind target inputs."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import write


SOURCE = ('b200_1', 'b200_mini_1', 'b210_1', 'pluto_1', 'rtl_1')
TARGET = ('b200_2', 'b200_mini_2', 'b210_2', 'pluto_2', 'n210_1', 'n210_2', 'n210_3',
          *(f'rtl_{i}' for i in range(2, 10)))
CLASSES = tuple(range(31, 41))
SPLIT_SEED = 392005
FC_HZ = 868_100_000
FS_HZ = 256_000
BANDWIDTH_HZ = 125_000
SYMBOL_START = 3 * 1024
WINDOW_START = 12
SAMPLE_COUNT = 256
ROLES = ('L_s', 'U_s', 'V')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


@contextmanager
def recording(archive, member, scratch):
    import h5py

    with tempfile.TemporaryDirectory(prefix='lora_h5_', dir=scratch) as name:
        path = Path(name) / 'recording.h5'
        with archive.open(member) as source, path.open('wb') as dest:
            shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
        with h5py.File(path, 'r') as handle:
            yield handle


def iq_window(h5, order=None):
    from scipy.signal import resample_poly

    data = h5['data']
    if data.ndim != 2 or data.shape[1] != 2 * 8192:
        raise ValueError('Expected official data rows with I then Q and 8192 complex samples')
    # Extra samples from the interior of the preamble cover the polyphase
    # filter support. The central input spans exactly 1 ms at 256 kHz.
    start = SYMBOL_START + WINDOW_START - 125
    end = start + 1250
    iq = np.stack((data[:, start:end], data[:, 8192 + start:8192 + end]), axis=1)
    x = resample_poly(iq, up=32, down=125, axis=-1)[..., 32:288].astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite LoRa IQ')
    return x if order is None else x[order]


def packet_indices(count, receiver_index):
    return np.random.default_rng([SPLIT_SEED, 1, receiver_index]).permutation(count)


def physical_ids(receiver, count):
    return np.asarray([f'{receiver}/p{i:05}' for i in range(count)])


def label_ids(h5, count):
    labels = h5['label'][:].reshape(-1)
    if labels.shape != (count,) or set(labels.tolist()) != set(CLASSES):
        raise ValueError('Official LoRa transmitter set differs')
    return labels.astype(np.int64) - CLASSES[0]


def source_role_ordinals(tx,source_limit):
    order=np.random.default_rng([SPLIT_SEED,2,tx]).permutation(source_limit)
    return dict(zip(ROLES,np.split(order,[56,560]),strict=True))


def make_source(archives, out, scratch):
    contract = dict(protocol='lora_rx5_main_v1', classes=list(map(str, CLASSES)),
                    source_rxs=list(range(len(SOURCE))), source_days=[0],
                    receiver_names=list(SOURCE), window_samples=SAMPLE_COUNT,
                    sample_rate_hz=FS_HZ, carrier_hz=FC_HZ, label_fraction=.07,
                    source_dates_observed=False, split_seed=SPLIT_SEED)
    iq, fields = defaultdict(list), defaultdict(lambda: defaultdict(list))
    source_limit = 800
    role_ordinals={tx:source_role_ordinals(tx,source_limit) for tx in range(len(CLASSES))}
    with zipfile.ZipFile(archives / 'multiple_receiver_train.zip') as archive:
        for rx, receiver in enumerate(SOURCE):
            with recording(archive, f'{receiver}_train.h5', scratch) as h5:
                x = iq_window(h5)
                y = label_ids(h5, len(x))
            order = packet_indices(len(x), rx)
            inverse = np.argsort(order)
            ids = physical_ids(receiver, len(x))
            positions = {tx: np.flatnonzero(y == tx) for tx in range(len(CLASSES))}
            if any(len(pos) != source_limit for pos in positions.values()):
                raise ValueError('Unexpected source packet population per transmitter')
            for role in ROLES:
                chosen = np.concatenate([positions[tx][role_ordinals[tx][role]] for tx in positions])
                chosen = chosen[np.argsort(inverse[chosen])]
                m = len(chosen)
                iq[role].append(x[chosen])
                meta = fields[role]
                base = inverse[chosen]
                metadata = dict(ids=ids[base], rx_i=np.full(m, rx), day_i=np.zeros(m, np.int64),
                                eq_i=np.ones(m, np.int64), sig_i=base, base_index=rx * len(x) + base,
                                domain=np.full(m, rx))
                if role != 'U_s':
                    metadata['y'] = y[chosen]
                for key, values in metadata.items():
                    meta[key].append(values)
            print('LORA_SOURCE_PREPARED', receiver, len(x), flush=True)
    contract['counts'] = {}
    contract['role_ids'] = {}
    for role in ROLES:
        metadata = {k: np.concatenate(v) for k, v in fields[role].items()}
        np.save(out / f'{role}_x.npy', np.concatenate(iq[role]))
        np.savez(out / f'{role}_metadata.npz', **metadata)
        contract['counts'][role] = len(metadata['ids'])
        contract['role_ids'][role] = metadata['ids'].tolist()
    all_ids = [pid for role in ROLES for pid in contract['role_ids'][role]]
    if len(all_ids) != 40_000 or len(set(all_ids)) != len(all_ids):
        raise ValueError('Source role partition must cover 40,000 disjoint records')
    write(out / 'contract.json', contract)


def make_target(archives, out, scratch):
    with zipfile.ZipFile(archives / 'multiple_receiver_test.zip') as archive:
        for rx, receiver in enumerate(TARGET):
            with recording(archive, f'{receiver}_test.h5', scratch) as h5:
                x = iq_window(h5, packet_indices(h5['data'].shape[0], rx))
            if len(x) != 2000:
                raise ValueError('Unexpected controlled target packet population')
            np.save(out / f'{receiver}.npy', x)
            print('LORA_BLIND_TARGET_PREPARED', receiver, len(x), flush=True)
    write(out / 'contract.json', dict(receivers=list(TARGET), count=len(TARGET)*2000,
                                     views=list(VIEWS), truth_read=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['source', 'target'])
    parser.add_argument('--archives', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepare = make_source if args.part == 'source' else make_target
    prepare(args.archives, args.output, args.scratch)


if __name__ == '__main__':
    main()
