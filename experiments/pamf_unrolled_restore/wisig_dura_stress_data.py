"""Generate a matched harder practical channel at an IQ-only boundary."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
from pathlib import Path
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .wisig_lora_source_fit import source_iq


ROUTES = ('full', 'residual')
PROFILES = ('default', 'strong')
EPOCHS = (1, 2, 3, 4)
SNR_DROP_DB = 6.0
PHASE_TRACKING_MULTIPLIER = 1.5
WORKERS = 12


def harder(config):
    return replace(config, reference_snr_db=config.reference_snr_db - SNR_DROP_DB,
                   phase_tracking_residual_std_deg=(
                       config.phase_tracking_residual_std_deg * PHASE_TRACKING_MULTIPLIER))


def channel(iq, config, ids, sessions, seed, receiver_seed, namespace, normalize):
    from leo_practical import apply_leo_practical_channel_batch

    observed, _, _ = apply_leo_practical_channel_batch(
        iq, config, seed=seed, sample_ids=ids, session_ids=sessions,
        realization_namespace=namespace, receiver_seed=receiver_seed,
        normalize_input_rms=normalize, return_meta=False)
    if observed.shape != iq.shape or observed.dtype != np.float32 or not np.isfinite(observed).all():
        raise ValueError('Practical simulator returned invalid IQ')
    return observed


def init_factory(clean_path, ids, channel_code):
    global CLEAN, IDS
    CLEAN = np.load(clean_path, mmap_mode='r', allow_pickle=False)
    IDS = ids
    sys.path.insert(0, str(channel_code))


def sample(job):
    indices, config, sessions, seed, receiver_seed, namespace, normalize = job
    rows = CLEAN[indices].astype(np.float64) if normalize else CLEAN[indices]
    observed = channel(rows, config, IDS[indices], sessions,
                       seed, receiver_seed, namespace, normalize)
    return indices, observed


def pool(clean, ids, channel_code):
    sys.path.insert(0, str(channel_code))
    return ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                               initializer=init_factory, initargs=(clean, ids, channel_code))


def batches(indices, config, sessions, seed, receiver_seed, namespace, normalize):
    for start in range(0, len(indices), 512):
        part = indices[start:start + 512]
        yield part, config, sessions[part], seed, receiver_seed, namespace, normalize


def read_plan(cache, epoch):
    folder = Path(cache) / 'epochs' / f'{epoch:02}'
    marker = read(folder / 'complete.json')
    if (marker['epoch'], marker['count'], marker['TX_labels_read'], marker['target_reads']) != (
            epoch, 63000, 0, 0) or marker['latent_shared'] is not True:
        raise ValueError('Expected paired source practical IQ from the fixed cache')
    quantile = np.load(folder / 'quantile.npy', allow_pickle=False)
    return folder, quantile


def use_route(quantile, route):
    return (quantile >= .575) if route == 'full' else ((quantile >= .15) & (quantile < .575))


def source_cache(rt):
    cache = Path(rt['source_root'])
    clean, fit, _ = source_iq(cache)
    train = cache / 'train'
    ids = np.load(train / 'physical_ids.npy', allow_pickle=False)
    rx = np.load(train / 'rx.npy', allow_pickle=False)
    day = np.load(train / 'day.npy', allow_pickle=False)
    sessions = np.asarray([f'rx{r}:day{d}:eq1' for r, d in zip(rx, day)])
    root = Path(rt['run_root']) / 'strong_source_iq'
    root.mkdir()
    with pool(train / 'clean.npy', ids, Path(__file__).resolve().parents[2] / 'code') as workers:
        from leo_practical import Config

        for epoch in EPOCHS:
            folder, quantile = read_plan(cache, epoch)
            scenes = np.load(folder / 'scene.npy', allow_pickle=False)
            path = root / f'{epoch:02}.npy'
            output = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=clean.shape)
            for route in ROUTES:
                cached = np.load(folder / f'{route}.npy', mmap_mode='r', allow_pickle=False)
                for scene_index, scene in enumerate(VIEWS[1:]):
                    selected = fit[use_route(quantile[fit], route) & (scenes[fit] == scene_index)]
                    config = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route=route,
                                    mode='post_sync', equalization_enabled=False, boundary='reflect')
                    namespace = f'reception-restoration/train/epoch{epoch:02}'
                    common = sessions, 2026101901, 2026101902, namespace, False
                    _, probe = workers.submit(sample, next(batches(selected[:8], config, *common))).result()
                    np.testing.assert_allclose(probe, cached[selected[:8]], rtol=2e-5, atol=2e-6)
                    for indices, iq in workers.map(sample, batches(selected, harder(config), *common)):
                        output[indices] = iq
                print('STRONG_SOURCE_IQ', epoch, route, flush=True)
            output.flush()
            del output
    write(root / 'complete.json', dict(epochs=list(EPOCHS), TX_reads=0, target_reads=0))


def target_cache(rt):
    parent = Path(rt['residual_reference'])
    manifest = read(parent / 'manifest.json')
    if (manifest['status'], manifest['channel'], manifest['count'], manifest['truth_read']) != (
            'VALIDATED_ONCE', 'residual/post_sync/noeq', 168000, False):
        raise ValueError('Expected the original truth-blind Main split simulator contract')
    with np.load(parent / 'index.npz', allow_pickle=False) as index:
        ids = index['ids']
    root = Path(rt['run_root']) / 'strong_target_iq'
    root.mkdir()
    ind = np.arange(len(ids))
    sessions = np.repeat(manifest['session'], len(ids))
    common = sessions, manifest['seed'], manifest['receiver_seed'], manifest['namespace'], True
    with pool(manifest['clean_ref'], ids, Path(rt['channel_release']) / 'code') as workers:
        from leo_practical import Config

        for route in ROUTES:
            out = root / route
            out.mkdir()
            default = Path(rt['baseline_root']) / 'full_views' if route == 'full' else parent
            for row in manifest['evidence']:
                scene = row['scene']
                config = replace(Config(**row['config']), processing_route=route)
                previous = np.load(default / f'{scene}.npy', mmap_mode='r', allow_pickle=False)
                _, probe = workers.submit(sample, next(batches(ind[:16], config, *common))).result()
                np.testing.assert_allclose(probe, previous[:16], rtol=2e-5, atol=2e-6)
                output = np.lib.format.open_memmap(out / f'{scene}.npy', mode='w+', dtype='float32',
                                                    shape=previous.shape)
                for indices, iq in workers.map(sample, batches(ind, harder(config), *common)):
                    output[indices] = iq
                output.flush()
                del output
                print('STRONG_TARGET_IQ', route, scene, len(ids), flush=True)
    write(root / 'complete.json', dict(status='IQ_COMPLETE', count=len(ids), truth_read=False))
