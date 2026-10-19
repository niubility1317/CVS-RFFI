"""Native full/residual observations share one clean input and latent seed."""
from dataclasses import replace
import numpy as np
from experiments.pamf_restoration.data import SCENES, seed_for


def route_for(candidate, quantile):
    """Generated training composition only; never an inference input/router."""
    if not 0 <= quantile < 1:
        raise ValueError('Training observation quantile must be in [0,1)')
    if quantile < .15:
        return 'clean'
    if candidate == 'unet_residual':
        return 'residual'
    if candidate == 'unet_full':
        return 'full'
    if candidate in ('unet_mixed', 'tf_unet_mixed', 'diffusion_mixed', 'flow_mixed'):
        return 'residual' if quantile < .75 else 'full'
    raise ValueError('Registered candidate required')


def route_array(candidate, quantile):
    from .models import NAMES
    quantile=np.asarray(quantile)
    if candidate not in NAMES or not np.isfinite(quantile).all() or (quantile<0).any() or (quantile>=1).any():
        raise ValueError('Registered candidate and valid observation quantiles required')
    result=np.full(quantile.shape,'residual',dtype='U8')
    if candidate=='unet_full':result[:]='full'
    elif candidate!='unet_residual':result[quantile>=.75]='full'
    result[quantile<.15]='clean'
    return result


def shared_pair(clean, scene, pid, session, seed, epoch, receiver_seed, namespace=None):
    from leo_practical.channel import Config
    from leo_practical.batch import apply_leo_practical_channel_batch
    cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                 mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
    common = dict(seed=seed, sample_ids=[str(pid)], session_ids=[str(session)],
                  realization_namespace=namespace or f'reception-restoration/train/epoch{epoch:02d}',
                  receiver_seed=receiver_seed, normalize_input_rms=False, return_meta=True)
    # Both start from clean. Never apply residual to the full output.
    residual, residual_meta, _ = apply_leo_practical_channel_batch(clean[None], cfg, **common)
    full, full_meta, _ = apply_leo_practical_channel_batch(clean[None], replace(cfg, processing_route='full'), **common)
    r, f = residual_meta[0], full_meta[0]
    if (r['seed'] != f['seed'] or r['receiver'] != f['receiver'] or r['state_start'] != f['state_start']
            or r['geometry'] != f['geometry']):
        raise ValueError('Full/residual must share initial physical latent realization')
    if not np.isfinite(residual).all() or not np.isfinite(full).all():
        raise FloatingPointError('Nonfinite native paired reception observations')
    return residual[0], full[0], dict(seed=r['seed'],
        residual={key:r[key] for key in ('receiver_lock_status','signal_to_added_noise_db','state_start')},
        full={key:f[key] for key in ('receiver_lock_status','signal_to_added_noise_db','state_start')})


_CLEAN = _IDS = _RX = _DAY = None


def initialize_worker(reference):
    from pathlib import Path
    global _CLEAN, _IDS, _RX, _DAY
    reference = Path(reference)
    _CLEAN = np.load(reference/'clean.npy', mmap_mode='r', allow_pickle=False)
    _IDS = np.load(reference/'physical_ids.npy', allow_pickle=False)
    _RX = np.load(reference/'rx.npy', allow_pickle=False)
    _DAY = np.load(reference/'day.npy', allow_pickle=False)


def paired_worker(item):
    index, scene, seed, epoch, receiver_seed, namespace = item
    r, f, meta = shared_pair(_CLEAN[index], scene, _IDS[index],
        f'rx{_RX[index]}:day{_DAY[index]}:eq1', seed, epoch, receiver_seed, namespace)
    return index, r, f, meta


def prepare_cache(parent, output, epochs=20, workers=16, data_seed=2026101901,
                  augmentation_seed=2026101902):
    """Fresh paired native channels each epoch; common exposure across six rows."""
    import json
    import multiprocessing
    from pathlib import Path
    from experiments.pamf_restoration.data import write_json
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Immutable reception-restoration cache exists')
    previous = json.loads((parent/'complete.json').read_text(encoding='utf-8'))
    if previous['status'] != 'SOURCE_CACHE_COMPLETE' or previous['seed'] != data_seed:
        raise ValueError('Original source waveform cache required')
    clean = np.load(parent/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(parent/'train/roles.npy', allow_pickle=False)
    if clean.shape != (63000, 2, 256) or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('All original legal L/U physical records required')
    output.mkdir(parents=True)
    # Read-only links preserve the original source references and V panel.
    (output/'train').symlink_to(parent/'train', target_is_directory=True)
    (output/'validation').symlink_to(parent/'validation', target_is_directory=True)
    epoch_root = output/'epochs';epoch_root.mkdir()
    lock_names, state_names = ('locked', 'degraded', 'unlocked'), ('LOS', 'PARTIAL', 'BLOCKED')
    context = multiprocessing.get_context('spawn')
    with context.Pool(workers, initializer=initialize_worker, initargs=(str(parent/'train'),)) as pool:
        for epoch in range(1, epochs+1):
            folder=epoch_root/f'{epoch:02d}';folder.mkdir()
            rng=np.random.default_rng(seed_for(augmentation_seed, 'observations', epoch))
            quantile=rng.random(len(clean))
            scene_indices=rng.integers(len(SCENES), size=len(clean))
            np.save(folder/'quantile.npy', quantile, allow_pickle=False)
            np.save(folder/'scene.npy', scene_indices, allow_pickle=False)
            arrays={route:np.lib.format.open_memmap(folder/(route+'.npy'), mode='w+',
                     dtype=np.float32, shape=clean.shape) for route in ('residual','full')}
            active=np.flatnonzero(quantile>=.15)
            seed_values=np.zeros(len(clean), np.uint64)
            diagnostics={route:dict(lock=np.full(len(clean), -1, np.int8),
                         state=np.full(len(clean), -1, np.int8),
                         signal_to_added_noise_db=np.zeros(len(clean), np.float32)) for route in arrays}
            items=((int(i),SCENES[scene_indices[i]],data_seed,epoch,augmentation_seed,None) for i in active)
            for count,(index,residual,full,meta) in enumerate(pool.imap(paired_worker,items,chunksize=32),1):
                arrays['residual'][index]=residual;arrays['full'][index]=full
                seed_values[index]=meta['seed']
                for route in arrays:
                    info=meta[route]
                    diagnostics[route]['lock'][index]=lock_names.index(info['receiver_lock_status'])
                    diagnostics[route]['state'][index]=state_names.index(info['state_start'])
                    diagnostics[route]['signal_to_added_noise_db'][index]=info['signal_to_added_noise_db']
                if count%10000==0:
                    print(json.dumps(dict(event='PAIRED_CACHE_PROGRESS',epoch=epoch,count=count,total=len(active))),flush=True)
            for value in arrays.values():value.flush()
            np.save(folder/'latent_seed.npy',seed_values,allow_pickle=False)
            np.savez(folder/'diagnostics.npz',**{route+'_'+key:value for route,info in diagnostics.items() for key,value in info.items()})
            write_json(folder/'complete.json',dict(epoch=epoch,count=len(clean),
                clean_count=int((quantile<.15).sum()),residual_mixed_count=int(((quantile>=.15)&(quantile<.75)).sum()),
                full_mixed_count=int((quantile>=.75).sum()),latent_shared=True,TX_labels_read=0,target_reads=0,
                physical_exposure_per_row=len(clean),fresh_epoch_namespace=True))
            print(json.dumps(dict(event='PAIRED_EPOCH_CACHE_COMPLETE',epoch=epoch)),flush=True)
    # Full/post-sync V uses the exact latent namespace/seed of original V.
    # Residual replay is checked against the existing source input, then
    # discarded; no new formal residual test observation is introduced.
    validation=output/'validation_full';validation.mkdir()
    for file in ('clean.npy','physical_ids.npy','rx.npy','day.npy','roles.npy'):
        (validation/file).symlink_to(parent/'validation'/file)
    vclean=np.load(parent/'validation/clean.npy',mmap_mode='r',allow_pickle=False)
    with context.Pool(workers,initializer=initialize_worker,initargs=(str(parent/'validation'),)) as pool:
        for scene in SCENES:
            old=np.load(parent/'validation'/(scene+'.npy'),mmap_mode='r',allow_pickle=False)
            full_array=np.lib.format.open_memmap(validation/(scene+'.npy'),mode='w+',dtype=np.float32,shape=vclean.shape)
            items=((i,scene,data_seed,0,augmentation_seed,'restoration/validation/draw0') for i in range(len(vclean)))
            for index,r,f,_ in pool.imap(paired_worker,items,chunksize=32):
                if not np.array_equal(r,old[index]):
                    raise ValueError('Full/residual V latent pairing differs from original source observation')
                full_array[index]=f
            full_array.flush()
    write_json(output/'complete.json',dict(status='RECEPTION_SOURCE_CACHE_COMPLETE',epochs=epochs,
               train_count=63000,L_count=6300,U_count=56700,source_V_count=len(vclean),
               TX_labels_read=0,target_reads=0,parent_source_cache=str(parent),
               paired_full_residual=True,observations_per_physical_record_per_row_per_epoch=1,
               available_training_routes=['clean','residual/post_sync/noeq','full/post_sync/noeq']))
