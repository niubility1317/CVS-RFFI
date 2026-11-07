"""Cross the simulated RX profiles with fixed source physical packets."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from leo_practical.channel import Config, receiver_for_session, stable_seed
from leo_practical.residual import ResidualChannel

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_parameter_source import G_COMMIT, load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_RUN
from .preamble_component_source import logits
from .source_action_factor_probe import references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source


RUN = '20261107-diagnostic-Restore-receiver-session-sourceV-s2026092701-r01'
VALIDATION_SEED = 2026101902
TRAINING_SEED = 2026101911
CHANNEL_SEED = 2026101901
NAMESPACE = 'restoration/validation/draw0'
ARMS = ('original', 'train_profiles', 'rotated_val_profiles')


def scene_config(scene):
    return Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                  mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')


def profile_assignments(rx, day):
    groups = list(zip(rx.tolist(), day.tolist()))
    unique = sorted(set(groups))
    names = [f'rx{receiver}:day{date}:eq1' for receiver, date in unique]
    native = {group: name for group, name in zip(unique, names)}
    rotated = {group: name for group, name in zip(unique, names[1:]+names[:1])}
    return [[native[group] for group in groups], [rotated[group] for group in groups]]


def replay(x, pid, cfg, profiles):
    seed = stable_seed(CHANNEL_SEED, NAMESPACE, cfg.scenario, str(pid))
    raw = x[0].astype(np.float64) + 1j*x[1].astype(np.float64)
    result = {}
    state = kernel = noise_start = noise_end = None
    for name, profile in zip(ARMS, profiles):
        channel = ResidualChannel(cfg, seed, profile)
        before = channel.latent.rngs['noise'].bit_generator.state
        received, meta, trace = channel.process(raw, return_traces=True)
        after = channel.latent.rngs['noise'].bit_generator.state
        current = (meta['state_start'], meta['geometry'], meta['quality_snr_db'], meta['receiver_lock_status'])
        if state is None:
            state, kernel, noise_start, noise_end = current, trace['propagation_kernel'], before, after
        elif (current != state or not np.array_equal(trace['propagation_kernel'], kernel)
              or before != noise_start or after != noise_end):
            raise ValueError('Receiver-only intervention changed propagation or base-noise draws')
        result[name] = (np.stack((received.real, received.imag)), meta['signal_to_added_noise_db'],
                        meta['agc_gain_db'], meta['output_frequency_hz'])
    return result


def forward(reader, model, x):
    observed = tensor(x, 'cpu')
    restored, aux = model(observed, return_aux=True)
    if not torch.isfinite(restored).all() or aux['relative_solve_residual'].max() > 1e-8:
        raise ValueError('Invalid frozen BF recovery')
    return logits(reader, observed).argmax(1), logits(reader, restored).argmax(1), restored


def worker(args, out):
    started = time.perf_counter()
    model = load_model(args.root, 'cpu')
    reader, identity = load_F0(validate_source(read(args.inputs)), 'cpu')
    valid_identity(identity)
    snapshots = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in (model, reader)]
    cache = args.root/'runs'/R07/'cache'
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    source = np.flatnonzero(roles == 'L_s')[:2]
    if len(source) != 2:
        raise ValueError('Original source L_s smoke required')
    smoke = np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    with torch.inference_mode():
        forward(reader, model, smoke[source])
    write(out/'smoke.json', dict(status='PASS', real_L_s=2, original_BF=True, original_F0=True, target_inputs=0))
    print('RECEIVER_SESSION_REAL_SMOKE_PASS', flush=True)

    ref, take = references(args.root, cache, identity)
    clean = tensor(ref['clean'], 'cpu')
    native, rotated = profile_assignments(ref['rx'], ref['day'])
    if len(set(native)) != 15 or sorted(native) != sorted(rotated) or any(a == b for a, b in zip(native, rotated)):
        raise ValueError('Expected 15 equally represented sessions and a complete rotation')
    with (out/'reference.npz').open('xb') as file:
        np.savez(file, **{key: ref[key] for key in ('ids', 'rx', 'day', 'logits')}, source_index=take)

    origin = args.root/'runs'/REFERENCE_RUN
    with torch.inference_mode():
        for scene in SCENES[1:]:
            config = scene_config(scene)
            hardware = {seed: {name: receiver_for_session(config, seed, name) for name in set(native)}
                        for seed in (VALIDATION_SEED, TRAINING_SEED)}
            generated = [replay(x, pid, config, (hardware[VALIDATION_SEED][a], hardware[TRAINING_SEED][a],
                         hardware[VALIDATION_SEED][b])) for x, pid, a, b in zip(ref['clean'], ref['ids'], native, rotated)]
            arrays = {arm: np.stack([row[arm][0] for row in generated]) for arm in ARMS}
            target = np.load(cache/'validation_actions'/scene/'received.npy', mmap_mode='r', allow_pickle=False)
            if not np.array_equal(arrays['original'], target[take]):
                raise ValueError('Original validation received must replay exactly')
            with np.load(origin/(scene+'.npz'), allow_pickle=False) as old:
                if not np.array_equal(old['ids'][take], ref['ids']):
                    raise ValueError('Original predictions must share exact source IDs')
                payload = dict(ids=ref['ids'], original__raw=old['received__prediction'][take],
                    original__BF=old['physical__prediction'][take], original__NMSE=old['physical__iq_nmse'][take])
            denominator = clean.square().mean((1, 2)).clamp_min(1e-4)
            for arm in ARMS:
                payload[arm+'__telemetry'] = np.asarray([row[arm][1:] for row in generated])
                if not np.isfinite(payload[arm+'__telemetry']).all():
                    raise ValueError('Expected finite paired native telemetry')
            for arm in ARMS[1:]:
                raw, restored, nmse = [], [], []
                for offset in range(0, len(take), 32):
                    batch = slice(offset, offset+32)
                    a, b, iq = forward(reader, model, arrays[arm][batch])
                    raw.extend(a.tolist())
                    restored.extend(b.tolist())
                    nmse.extend(((iq-clean[batch]).square().mean((1, 2))/denominator[batch]).tolist())
                payload.update({arm+'__raw': np.asarray(raw), arm+'__BF': np.asarray(restored), arm+'__NMSE': np.asarray(nmse)})
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file, **payload)
            print('RECEIVER_SESSION_VIEW_COMPLETE', scene, len(take), flush=True)
    for m, snapshot in zip((model, reader), snapshots):
        check_F0_state(m, snapshot)
    write(out/'results.json', dict(status='SOURCE_RECEIVER_SESSIONS_PREDICTIONS_COMPLETE', code_commit=args.commit,
        frozen_BF_code=G_COMMIT, F0_identity=identity, arms=list(ARMS), views=list(SCENES[1:]),
        clean_reference_reused=True, source_V_records=len(take), replay_IQ_exact=True,
        shared_propagation_kernel_geometry_lock_and_base_noise_rng=True, original_receiver_seed=VALIDATION_SEED,
        train_profile_seed=TRAINING_SEED, unchanged_val_hardware_multiset_rotation=1,
        BF_new_rows=5760, F0_new_rows=11520, TX_labels_read=0, target_inputs=0, optimizer_updates=0,
        seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('CPU-only release with exact code commit required')
    os.nice(10)
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json', dict(pid=os.getpid(), code_commit=args.commit, CPU_threads=2, CUDA_empty=True))
    worker(args, out)
    write(out/'complete.json', dict(status='ARTIFACTS_COMPLETE', TX_labels_read=0, target_inputs=0))


if __name__ == '__main__':
    main()
