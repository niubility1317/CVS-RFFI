"""Create physically paired LoRa views with the immutable project simulator."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from experiments.cric_dura_comparison.io import tensor, write
from .data import SOURCE, TARGET, VIEWS
from .physical import practical_iq, rms_normalize


def simulate(namespace, receiver, scene, indexed):
    index, x = indexed
    return practical_iq(x, scene, receiver, index, namespace)


def populate(pool, rows, indices, namespace, receiver, scene):
    call = partial(simulate, namespace, receiver, scene)
    return np.stack(list(pool.map(call, zip(indices, rows, strict=True), chunksize=64)))


def make_source_views(root, output):
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=8) as pool:
        l = np.load(root / 'L_s_x.npy', allow_pickle=False)
        u = np.load(root / 'U_s_x.npy', allow_pickle=False)
        pairs = np.empty((len(l) + len(u), 2, 256), dtype=np.float32)
        l_views = np.empty((len(VIEWS)-1, len(l), 2, 256), dtype=np.float32)
        for role, iq, offset in (('L_s', l, 0), ('U_s', u, len(l))):
            with np.load(root / f'{role}_metadata.npz', allow_pickle=False) as m:
                if role == 'U_s' and 'y' in m.files:
                    raise ValueError('Blind U metadata required')
                ids, rx = m['base_index'], m['rx_i']
            for r, receiver in enumerate(SOURCE):
                for s, scene in enumerate(VIEWS[1:]):
                    positions = np.flatnonzero(rx == r)
                    if role == 'L_s':
                        l_views[s, positions] = populate(pool, iq[positions], ids[positions],
                                                        'lora-source', receiver, scene)
                    chosen = positions[ids[positions] % 10 < 9]
                    chosen = chosen[(ids[chosen] // 10) % (len(VIEWS)-1) == s]
                    if role == 'L_s':
                        pairs[chosen] = l_views[s, chosen]
                    else:
                        pairs[offset + chosen] = populate(pool, iq[chosen], ids[chosen],
                                                          'lora-source', receiver, scene)
                print('LORA_SOURCE_PRACTICAL', role, receiver, flush=True)
            identity = np.flatnonzero(ids % 10 == 9)
            clean = rms_normalize(tensor(iq[identity]))
            pairs[offset + identity] = np.asarray(clean.tolist(), dtype=np.float32)
        np.save(output / 'L_practical.npy', l_views)
        np.save(output / 'LU_DURA_practical.npy', pairs)
    write(output / 'complete.json', dict(status='SOURCE_PRACTICAL_READY',
            views=list(VIEWS[1:]), source_L=len(l), source_U=len(u), target_access=False, U_truth_read=False))


def make_target_views(root, output):
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for receiver in TARGET:
            iq = np.load(root / f'{receiver}.npy', allow_pickle=False)
            views = [populate(pool, iq, range(len(iq)), 'lora-test', receiver, scene) for scene in VIEWS[1:]]
            np.save(output / f'{receiver}.npy', np.stack(views))
            print('LORA_TARGET_PRACTICAL', receiver, len(iq), flush=True)
    write(output / 'complete.json', dict(status='TARGET_PRACTICAL_READY', views=list(VIEWS[1:]),
          receivers=list(TARGET), target_truth_read=False))
