"""Source IQ generates action teachers. No label/F/target loader is imported."""
import json
import math
import multiprocessing
from pathlib import Path
import numpy as np
from leo_practical.channel import Config, receiver_for_session, stable_seed
from leo_practical.residual import ResidualChannel
from leo_practical.equalization import linear_filter
from experiments.pamf_restoration.data import SCENES, seed_for, write_json
from .models import KERNEL_LENGTH
from . import design as d


def teacher(clean, cfg, seed, receiver):
    if (cfg.processing_route != 'residual' or cfg.mode != 'post_sync' or
            cfg.equalization_enabled or cfg.boundary != 'reflect'):
        raise ValueError('Teacher is residual/post_sync/noEQ/reflect only')
    clean = np.asarray(clean)
    x = clean[0].astype(float)+1j*clean[1].astype(float)
    observed, meta, trace = ResidualChannel(cfg, seed, receiver).process(x, return_traces=True)
    kernel = trace['residual_kernel']
    if len(kernel) > KERNEL_LENGTH:
        raise ValueError('Native action exceeds registered FIR support; never truncate')
    rng = np.random.default_rng(2026101907)
    probe = (rng.normal(size=len(x))+1j*rng.normal(size=len(x)))/math.sqrt(2)
    traces = [ResidualChannel(cfg, seed, receiver).process(value, return_traces=True,
                 quality_snr_db=meta['quality_snr_db'])[2]['pre_agc_signal'] for value in (probe, 1j*probe)]
    response = linear_filter(probe, kernel, cfg.boundary)[0]
    if np.min(abs(response)) < 1e-12:
        raise FloatingPointError('Teacher probe is numerically unidentifiable')
    gain = 10**(meta['agc_gain_db']/20)
    main = gain*(traces[0]-1j*traces[1])/(2*response)
    image = gain*(traces[0]+1j*traces[1])/(2*response.conj())
    filtered = linear_filter(x, kernel, cfg.boundary)[0]
    replay = main*filtered+image*filtered.conj()
    error = float(np.linalg.norm(replay-gain*trace['pre_agc_signal'])/max(np.linalg.norm(replay), 1e-12))
    if error > 1e-9:
        raise FloatingPointError('Source teacher action differs from native signal')
    padded = np.pad(kernel, (0, KERNEL_LENGTH-len(kernel)))
    noise = float((abs(main)**2+abs(image)**2).mean()*10**(-cfg.reference_snr_db/10)/2)
    if not 1e-8 <= noise <= 1.:
        raise ValueError('Actual real noise variance exceeds registered estimator range')
    return dict(received=np.stack((observed.real, observed.imag)).astype(np.float32),
        kernel=np.stack((padded.real, padded.imag)).astype(np.float32),
        coefficients=np.stack((main.real, main.imag, image.real, image.imag)).astype(np.float32),
        variance=np.float32(noise)), dict(state=meta['state_start'], lock=meta['receiver_lock_status'], replay_error=error)


def clean_teacher(clean):
    kernel = np.zeros((2, KERNEL_LENGTH), np.float32)
    kernel[0, 0] = 1
    coefficients = np.zeros((4, 256), np.float32)
    coefficients[0] = 1
    return dict(received=np.asarray(clean, np.float32), kernel=kernel, coefficients=coefficients,
                variance=np.float32(1e-8)), dict(state='clean', lock='not_applicable', replay_error=0.)


def initialize(reference):
    global _CLEAN, _IDS, _RX, _DAY
    reference = Path(reference)
    _CLEAN = np.load(reference/'clean.npy', mmap_mode='r', allow_pickle=False)
    _IDS = np.load(reference/'physical_ids.npy', allow_pickle=False)
    _RX = np.load(reference/'rx.npy', allow_pickle=False)
    _DAY = np.load(reference/'day.npy', allow_pickle=False)


def generate(item):
    index, scene, namespace, seed, receiver_seed = item
    if scene == 'clean':
        values, meta = clean_teacher(_CLEAN[index])
    else:
        cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                     mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
        receiver = receiver_for_session(cfg, receiver_seed, f'rx{_RX[index]}:day{_DAY[index]}:eq1')
        latent = stable_seed(seed, namespace, scene, str(_IDS[index]))
        values, meta = teacher(_CLEAN[index], cfg, latent, receiver)
    return index, values, meta


def generate_folder(pool, reference, output, scenes, namespace, seed, receiver_seed, reuse=None):
    count = len(scenes)
    output.mkdir(parents=True)
    shapes = dict(received=(count, 2, 256), kernel=(count, 2, KERNEL_LENGTH),
                  coefficients=(count, 4, 256), variance=(count,))
    arrays = {key: np.lib.format.open_memmap(output/(key+'.npy'), mode='w+', dtype=np.float32, shape=shape)
              for key, shape in shapes.items()}
    items = ((i, scene, namespace, seed, receiver_seed) for i, scene in enumerate(scenes))
    states, locks, maximum = [], [], 0.
    for done, (index, values, meta) in enumerate(pool.imap(generate, items, chunksize=32), 1):
        if reuse is not None and not np.array_equal(values['received'], reuse[index]):
            raise ValueError('Source action teacher belongs to a different fixed observation')
        for key in arrays:
            arrays[key][index] = values[key]
        states.append(meta['state']); locks.append(meta['lock'])
        maximum = max(maximum, meta['replay_error'])
        if done % 10000 == 0:
            print(json.dumps(dict(event='ACTION_CACHE_PROGRESS', group=output.name, count=done, total=count)), flush=True)
    for array in arrays.values():
        array.flush()
    np.savez(output/'groups.npz', state=np.asarray(states), lock=np.asarray(locks), scene=np.asarray(scenes))
    write_json(output/'complete.json', dict(count=count, namespace=namespace, max_native_replay_error=maximum,
        TX_labels_read=0, F_outputs_read=0, target_reads=0, native_engine_modified=False))
    print(json.dumps(dict(event='ACTION_CACHE_GROUP_COMPLETE', group=str(output.relative_to(output.parents[1])), count=count)), flush=True)


def prepare_cache(parent, output):
    parent, output = Path(parent), Path(output)
    previous = json.loads((parent/'complete.json').read_text(encoding='utf-8'))
    if (previous['status'] != 'RECEPTION_SOURCE_CACHE_COMPLETE' or previous['train_count'] != 63000 or
            previous['L_count'] != 6300 or previous['U_count'] != 56700 or previous['TX_labels_read'] or previous['target_reads']):
        raise ValueError('Validated parent source waveform handles required')
    roles = np.load(parent/'train/roles.npy', allow_pickle=False)
    if len(roles) != 63000 or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('Original full L/U source pool required')
    output.mkdir()
    (output/'train').symlink_to((parent/'train').resolve(), target_is_directory=True)
    (output/'validation').symlink_to((parent/'validation').resolve(), target_is_directory=True)
    context = multiprocessing.get_context('spawn')
    with context.Pool(d.configuration()['workers'], initializer=initialize, initargs=(str(parent/'train'),)) as pool:
        for epoch in range(1, d.EPOCHS+1):
            rng = np.random.default_rng(seed_for(d.AUGMENTATION_SEED, 'action-observations', epoch))
            scenes = np.asarray(SCENES)[rng.integers(len(SCENES), size=len(roles))]
            scenes = np.where(rng.random(len(roles)) < .15, 'clean', scenes)
            generate_folder(pool, parent/'train', output/'epochs'/f'{epoch:02d}', scenes,
                            f'learned-action/train/epoch{epoch:02d}', d.AUGMENTATION_SEED, d.RECEIVER_SEED)
    count = len(np.load(parent/'validation/physical_ids.npy', allow_pickle=False))
    if count != 1800:
        raise ValueError('Original fixed source V panel required')
    with context.Pool(d.configuration()['workers'], initializer=initialize, initargs=(str(parent/'validation'),)) as pool:
        for scene in ('clean', *SCENES):
            existing = np.load(parent/'validation'/(scene+'.npy'), mmap_mode='r', allow_pickle=False)
            generate_folder(pool, parent/'validation', output/'validation_actions'/scene, [scene]*count,
                            'restoration/validation/draw0', d.DATA_SEED, 2026101902, reuse=existing)
    write_json(output/'complete.json', dict(status='ACTION_SOURCE_CACHE_COMPLETE', train_count=63000,
        source_V_count=count, epochs=d.EPOCHS, original_parent=str(parent), TX_labels_read=0, target_reads=0))
