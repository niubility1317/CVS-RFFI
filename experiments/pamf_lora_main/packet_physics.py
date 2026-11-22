"""Generate sync-aligned practical LoRa packets at the capture sample rate."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, write
from .data import FC_HZ, ROLES, SOURCE, TARGET, VIEWS, WINDOW_START
from .packet_data import CHIRPS, PACKET_LEN, ROLE_COUNTS, TARGET_LIMIT


NATIVE_RATE = 1_000_000
CONTEXT = 128


def chirp_samples(z, offset=0):
    starts = np.asarray(CHIRPS) * 1024 + WINDOW_START - 125 + offset
    if starts[0] < 0 or starts[-1] + 1250 > len(z):
        raise ValueError('LoRa chirp crop needs real input context on both sides')
    x = z[starts[:, None] + np.arange(1250)]
    return resample_poly(x, up=32, down=125, axis=-1)[:, 32:288]


def chirp_windows(z, offset=0):
    values = chirp_samples(z, offset)
    scale = np.sqrt(np.mean(np.abs(values)**2))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Invalid LoRa packet power')
    return np.asarray(np.stack((values.real, values.imag), axis=1) / scale, dtype=np.float32)


@lru_cache(maxsize=None)
def channel(scene):
    from leo_practical.channel import Config
    return Config(fs_hz=NATIVE_RATE, fc_hz=FC_HZ, scenario=scene, processing_route='full',
                  mode='post_sync', equalization_enabled=False,
                  reference_snr_db=30., boundary='require_context')


@lru_cache(maxsize=None)
def hardware(scene, session):
    from leo_practical.channel import receiver_for_session
    return receiver_for_session(channel(scene), 20261016423, session)


def aligned_view(z, scene, receiver, pid, namespace, *, trace=False, channel_override=None, realization=None):
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed
    z, _ = normalize_input(z)
    session = f'{namespace}/{receiver}'
    cfg = channel_override or channel(scene)
    rx = hardware(scene, session)
    parts=(namespace,scene,receiver,str(pid))
    if realization is not None:
        parts=(*parts,'draw',realization)
    seed = stable_seed(2026112207, *parts)
    if trace:
        y, meta, _ = ChannelStream(cfg, seed, rx).process(
            z[CONTEXT:], preceding_context=z[:CONTEXT], return_traces=True)
    else:
        y, meta = ChannelStream(cfg, seed, rx).process(z[CONTEXT:], preceding_context=z[:CONTEXT])
    latency = meta['numerical_filter_latency_samples']
    if (meta['channel_equalization_applied'] or meta['boundary'] != 'provided_context'
            or latency != cfg.fractional_delay_half_length):
        raise ValueError('Aligned practical must use known filter delay and real IQ context')
    result = chirp_windows(y, offset=latency-CONTEXT)
    return (result, meta) if trace else result


def job(scene, namespace, item):
    z, rx, pid = item
    if scene == 'clean':
        return chirp_windows(z)
    return aligned_view(z, scene, rx, pid, namespace)


def save_views(pool, path, packets, receivers, ids, namespace, scenes):
    shape = (len(scenes), len(packets), len(CHIRPS), 2, 256)
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    for i, scene in enumerate(scenes):
        run = partial(job, scene, namespace)
        seq = zip(packets, receivers, ids, strict=True)
        for index, record in enumerate(pool.map(run, seq, chunksize=24)):
            output[i, index] = record
        print('ALIGNED_VIEW_READY', path.name, scene, len(packets), flush=True)
    output.flush()


def dura_pair(item):
    z, rx, pid, slot = item
    clean = chirp_windows(z)
    if slot % 8 == 7:
        return clean, clean
    scene = VIEWS[1 + (slot // 8) % 6]
    return clean, aligned_view(z, scene, rx, pid, 'source')


def make_source(data, out):
    c = read(data/'contract.json')
    if (c['protocol'] != 'lora_packet_pilot_v1' or not c['source_only']
            or c['counts'] != ROLE_COUNTS or c['chirps'] != list(CHIRPS)):
        raise ValueError('Source packet protocol differs')
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=10) as pool:
        d = {}
        for role in ROLES:
            packets = np.load(data/f'{role}_packets.npy', mmap_mode='r', allow_pickle=False)
            with np.load(data/f'{role}_metadata.npz', allow_pickle=False) as m:
                if m['ids'].tolist() != c['role_ids'][role] or ('y' in m.files) != (role != 'U_s'):
                    raise ValueError('Source physical order or labels differ')
                rx = np.asarray(SOURCE)[m['rx_i']]
                ids, slots = m['ids'], m['base_index']
            if packets.shape != (ROLE_COUNTS[role], PACKET_LEN) or packets.dtype != np.complex64:
                raise ValueError('Invalid source packet shape')
            scenes = VIEWS if role != 'U_s' else VIEWS[:1]
            namespace = 'source-V' if role == 'V' else 'source'
            save_views(pool, out/f'{role}.npy', packets, rx, ids, namespace, scenes)
            if role != 'V':
                d[role] = (packets, rx, ids, slots)

        # Restoration input excludes V, labels and the recognition network.
        n = ROLE_COUNTS['L_s'] + ROLE_COUNTS['U_s']
        output = np.lib.format.open_memmap(out/'G_pairs.npy', mode='w+', dtype=np.float32,
                                           shape=(2, n, len(CHIRPS), 2, 256))
        offset = 0
        for role in ('L_s','U_s'):
            for j, (clean, received) in enumerate(pool.map(dura_pair, zip(*d[role], strict=True), chunksize=24)):
                output[0, offset+j], output[1, offset+j] = clean, received
            offset += ROLE_COUNTS[role]
            print('ALIGNED_DURA_PAIRS', role, offset, flush=True)
        output.flush()
    write(out/'complete.json', dict(status='SOURCE_READY', counts=ROLE_COUNTS,
          input_fs=NATIVE_RATE, views=list(VIEWS), target_read=False))


def make_target(data, out):
    c = read(data/'contract.json')
    if (c['receivers'] != list(TARGET) or c['per_receiver'] != TARGET_LIMIT or c['TX_truth_read']):
        raise ValueError('Blind aligned target contract differs')
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=10) as pool:
        for name in TARGET:
            packets = np.load(data/f'{name}_packets.npy', mmap_mode='r', allow_pickle=False)
            if packets.shape != (TARGET_LIMIT, PACKET_LEN):
                raise ValueError('Target packet shape differs')
            save_views(pool, out/f'{name}.npy', packets, [name]*TARGET_LIMIT, range(TARGET_LIMIT), 'target', VIEWS)
    write(out/'complete.json', dict(status='TARGET_READY', target_truth_read=False, receivers=list(TARGET),
                                  per_receiver=TARGET_LIMIT, views=list(VIEWS)))
