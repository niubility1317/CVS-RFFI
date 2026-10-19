"""Source IQ only, including L_s and U_s. TX label files are never opened."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)


def seed_for(seed, *parts):
    payload = json.dumps([seed, *parts], ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**32-1)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class IQSource:
    def __init__(self, cache, role):
        if role not in ('L_s', 'U_s', 'V'):
            raise ValueError('Source roles only; target/query/truth are not inputs')
        root = Path(cache)
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'READY' or manifest['native_role_ids'] != 'EXACT_MATCH':
            raise ValueError('Existing validated source cache required')
        if manifest['source_receivers'] != list(SOURCE_RX) or manifest['source_days'] != list(SOURCE_DAYS):
            raise ValueError('Original source receiver/day contract required')
        self.role = role
        folder = root/role
        # Deliberately do not instantiate SourceRoleDataset: that general
        # classifier loader also opens L/V label files. Restoration never does.
        self.iq = np.load(folder/'iq.npy', mmap_mode='r', allow_pickle=False)
        self.ids = np.load(folder/'physical_ids.npy', mmap_mode='r', allow_pickle=False)
        self.rx = np.load(folder/'rx_i.npy', mmap_mode='r', allow_pickle=False)
        self.day = np.load(folder/'day_i.npy', mmap_mode='r', allow_pickle=False)
        if self.iq.shape != (len(self.ids), 2, 256) or len(self.rx) != len(self.ids) or len(self.day) != len(self.ids):
            raise ValueError('Malformed source role arrays')
        if not set(self.rx.tolist()) <= set(SOURCE_RX) or not set(self.day.tolist()) <= set(SOURCE_DAYS):
            raise ValueError('Source role contains forbidden receiver/date')

    def stratified_indices(self, per_session, seed):
        selected = []
        for rx in SOURCE_RX:
            for day in SOURCE_DAYS:
                available = np.flatnonzero((self.rx == rx) & (self.day == day))
                if len(available) < per_session:
                    raise ValueError('Insufficient source session samples')
                rng = np.random.default_rng(seed_for(seed, self.role, rx, day))
                selected.extend(rng.permutation(available)[:per_session].tolist())
        return np.asarray(selected, np.int64)


_WORKER_CONFIGS = None


def corrupt_one(item):
    global _WORKER_CONFIGS
    from leo_practical.channel import Config
    from leo_practical.batch import apply_leo_practical_channel_batch
    if _WORKER_CONFIGS is None:
        _WORKER_CONFIGS = {s: Config(fs_hz=25e6, fc_hz=2462e6, scenario=s, processing_route='residual',
                                    mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect') for s in SCENES}
    index, clean, scene, pid, session, group, seed = item
    value, _, _ = apply_leo_practical_channel_batch(
        clean[None], _WORKER_CONFIGS[scene], seed=seed, sample_ids=[pid], session_ids=[session],
        realization_namespace='restoration/'+group+'/draw0', receiver_seed=2026101902,
        normalize_input_rms=False, return_meta=False)
    return index, value[0], scene


def prepare_cache(cache, output, train_l_per_session=420, train_u_per_session=3780,
                  validation_per_session=120, seed=2026101901, workers=1):
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(workers) as pool:
            return _prepare_cache(cache, output, train_l_per_session, train_u_per_session,
                                  validation_per_session, seed, pool)
    return _prepare_cache(cache, output, train_l_per_session, train_u_per_session,
                          validation_per_session, seed, None)


def _prepare_cache(cache, output, train_l_per_session, train_u_per_session,
                   validation_per_session, seed, pool):
    """All legal L/U and fixed V panel, independent of labels or predictions.

    One training channel per physical sample; fresh diffusion time/noise each
    optimizer step. Validation includes all six views, source only. This is
    not a Phase2 capsule and cannot be passed to a Phase2 method.
    """
    import time
    from leo_practical.channel import Config, normalize_input
    from leo_practical.batch import apply_leo_practical_channel_batch
    output = Path(output)
    if output.exists():
        raise FileExistsError('Immutable restoration cache already exists')
    output.mkdir(parents=True)
    configs = {s: Config(fs_hz=25e6, fc_hz=2462e6, scenario=s, processing_route='residual',
                          mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect') for s in SCENES}
    selected_ids = {}
    for group, roles in (('train', [('L_s', train_l_per_session), ('U_s', train_u_per_session)]),
                         ('validation', [('V', validation_per_session)])):
        xs, ids, rxs, days, origin = [], [], [], [], []
        for role, count in roles:
            source = IQSource(cache, role)
            indices = source.stratified_indices(count, seed)
            xs.append(np.asarray(source.iq[indices]))
            ids.extend(source.ids[indices].tolist())
            rxs.extend(source.rx[indices].tolist())
            days.extend(source.day[indices].tolist())
            origin.extend([role]*len(indices))
        raw = np.concatenate(xs)
        if group == 'train' and (len(xs[0]) != 6300 or len(xs[1]) != 56700):
            raise ValueError('All 6300 L_s and 56700 U_s must participate')
        # The original cache is normalized. Match exactly the simulator's
        # clean input units rather than claiming denoising original RX noise.
        clean = np.stack([np.stack((z.real, z.imag)) for z in
                          (normalize_input(v[0]+1j*v[1])[0] for v in raw)]).astype(np.float32)
        folder = output/group
        folder.mkdir()
        np.save(folder/'clean.npy', clean, allow_pickle=False)
        np.save(folder/'physical_ids.npy', np.asarray(ids, dtype='U24'), allow_pickle=False)
        np.save(folder/'rx.npy', np.asarray(rxs, np.int64), allow_pickle=False)
        np.save(folder/'day.npy', np.asarray(days, np.int64), allow_pickle=False)
        np.save(folder/'roles.npy', np.asarray(origin, dtype='U3'), allow_pickle=False)
        selected_ids[group] = set(ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate physical source ID')
        sessions = [f'rx{rx}:day{day}:eq1' for rx, day in zip(rxs, days)]
        totals = {}
        requested = ['mixed'] if group == 'train' else SCENES
        for view in requested:
            received = np.lib.format.open_memmap(folder/(view+'.npy'), mode='w+', dtype=np.float32, shape=clean.shape)
            scene_counts = {}
            items = ((index, clean[index],
                      SCENES[seed_for(seed, 'scene', ids[index]) % len(SCENES)] if view == 'mixed' else view,
                      ids[index], sessions[index], group, seed) for index in range(len(clean)))
            generated = pool.imap(corrupt_one, items, chunksize=32) if pool is not None else map(corrupt_one, items)
            for index, value, scene in generated:
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite native practical channel')
                received[index] = value
                scene_counts[scene] = scene_counts.get(scene, 0)+1
                if (index+1) % 1000 == 0:
                    print(json.dumps(dict(event='CACHE_PROGRESS', group=group, view=view, count=index+1, total=len(clean))), flush=True)
            received.flush()
            totals[view] = scene_counts
        write_json(folder/'manifest.json', dict(count=len(clean), roles={r: origin.count(r) for r in set(origin)},
                   scenes=totals, label_reads=0, created_at=time.time()))
    if selected_ids['train'] & selected_ids['validation']:
        raise ValueError('Training and validation physical IDs overlap')
    result = dict(status='SOURCE_CACHE_COMPLETE', schema='pamf_restoration_source_v1',
                  source_cache=str(Path(cache).resolve()), seed=seed, source_only=True, TX_labels_read=0,
                  target_reads=0, query_reads=0, independent_validation=True, scenes=list(SCENES),
                  channel=dict(route='residual', mode='post_sync', equalization=False,
                               reference_snr_db=30., fs_hz=25e6, boundary='reflect'),
                  limitations=['Fixed one training draw per physical sample for initial screening',
                               'Clean means original terrestrial RX record, not ideal TX waveform',
                               'Source screening only; no target performance claim'])
    write_json(output/'complete.json', result)
    return result


def fixed_noise(ids, seed, shape, device):
    # Sample IDs set random draws only. No TX/domain/scoring information is
    # encoded, and chunk/order changes cannot change an individual prediction.
    noise = [torch.randn(shape[1:], generator=torch.Generator().manual_seed(seed_for(seed, str(pid)))) for pid in ids]
    return torch.stack(noise).to(device)
