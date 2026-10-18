"""Group scene sampling in every cell; optional propagation-only dependence.

One native personal snapshot per packet and one unused environment snapshot
per RX/day/eq group are constructed in every cell.  Only seven established
propagation fields move; receiver/noise/phase/sync/tracking/equalizer remain
personal.  Every waveform is produced by one complete native process call.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math
import re

import numpy as np
import torch

from experiments.cvs_noise_resample75.channel import (
    PHYSICAL_FIELDS, FIT, TRAIN_SCENES, TRAIN_SEED, _integers, serial, tensor_iq)

RECEIVER_SEED = 20261016423
PROPAGATION = ('geometry', 'probs', 'state', 'z', 'ray_hz', 'ray_coef', 'f_doppler')
PRIVATE_STREAMS = ('noise', 'phase', 'sync', 'tracking', 'equalizer')
PRIVATE_SCALARS = ('phase', 'tracking_z', 'residual_hz', 'coarse_doppler_error_hz')
_NAMESPACE = re.compile(r'FcrossRX/train/e(\d+)/step(\d+)/L_S')


def _private(snapshot):
    return {name: float(getattr(snapshot.latent, name)) for name in PRIVATE_SCALARS}


def _streams(snapshot):
    values = snapshot.latent.rngs
    if not isinstance(values, dict) or any(name not in values for name in PRIVATE_STREAMS):
        raise ValueError('Original personal native RNG streams are required')
    return {key: (id(value), serial(deepcopy(value.bit_generator.state))) for key, value in values.items()}


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def mix_propagation(personal, environment):
    """The P20 field transplant; preserve all personal RNG objects and states."""
    if personal.used or environment.used or personal.cfg != environment.cfg:
        raise ValueError('Unused matching native residual snapshots required')
    before_private, before_streams = _private(personal), _streams(personal)
    for name in PROPAGATION:
        setattr(personal.latent, name, deepcopy(getattr(environment.latent, name)))
    a = personal.latent
    a.f_total = a.f_doppler + personal.receiver.oscillator_cfo_hz
    a.f_hat = a.f_total - a.residual_hz
    if _private(personal) != before_private or _streams(personal) != before_streams:
        raise ValueError('Propagation sharing changed personal receiver/random streams')
    if any(not _equal(getattr(a, name), getattr(environment.latent, name)) for name in PROPAGATION):
        raise ValueError('All seven propagation fields must match the environment')
    for name in ('probs', 'ray_hz', 'ray_coef'):
        if np.shares_memory(getattr(a, name), getattr(environment.latent, name)):
            raise ValueError('Propagation array copies must not alias environment storage')
    return personal


def _config(cfg, scene):
    if (cfg.scenario != scene or cfg.processing_route != 'residual' or cfg.mode != 'post_sync'
            or cfg.equalization_enabled or cfg.reference_snr_db != 30.
            or cfg.boundary != 'reflect' or cfg.fs_hz != 25000000. or cfg.fc_hz != 2462000000.):
        raise ValueError('Unchanged complete practical residual/post_sync/EQoff/reflect config required')


def grouped_view(primary, raw64, meta, namespace, share_propagation):
    """Training-only grouped As64; the existing second mixed pass copies it.

    This helper owns the original adapter's calls/packets/coverage/sessions
    accounting.  Environment construction counts are additional bookkeeping,
    never waveform counts.  No label is read from meta or used in any seed.
    """
    match = _NAMESPACE.fullmatch(namespace) if isinstance(namespace, str) else None
    if (not match or type(share_propagation) is not bool
            or not torch.is_tensor(raw64) or raw64.shape != (64, 2, 256)
            or raw64.dtype != torch.float32 or not bool(torch.isfinite(raw64).all())
            or not isinstance(meta, dict) or any(key not in meta for key in PHYSICAL_FIELDS)):
        raise ValueError('One finite float32 L As64 pack, train namespace and fixed sharing flag required')
    epoch, step = (int(value) for value in match.groups())
    if epoch not in range(21, 31) or step not in range(4441, 6661) or epoch != (step-1)//222+1:
        raise ValueError('Only original cumulative E21..30/4441..6660 training namespaces')
    values = {key: _integers(meta[key]) for key in PHYSICAL_FIELDS}
    planned = []; groups = {}
    for index in range(64):
        pid = tuple(values[key][index] for key in PHYSICAL_FIELDS)
        # P43 pid is eq/RX/day/sig/base, unlike the old P20 RX/day/eq ordering.
        group = (pid[1], pid[2], pid[0])
        if pid[0] != 1 or pid[1] not in FIT or pid[2] not in (1, 2, 3) or min(pid[3:]) < 0:
            raise ValueError('Original five-source-RX/day1..3/eq1 physical metadata required')
        if group not in groups:
            scene_seed = primary.ch.stable_seed(TRAIN_SEED, namespace, 'scene', group)
            scene = str(np.random.default_rng(scene_seed).choice(TRAIN_SCENES))
            if scene not in primary.configs:
                raise ValueError('All original nominal training scenes must be available')
            cfg = primary.configs[scene]; _config(cfg, scene)
            environment_seed = primary.ch.stable_seed(TRAIN_SEED, namespace, scene, 'environment', group)
            groups[group] = dict(scene=scene, scene_seed=scene_seed, cfg=cfg,
                environment_seed=environment_seed, index=len(groups), snapshot=None)
        info = groups[group]
        personal_seed = primary.ch.stable_seed(TRAIN_SEED, namespace, info['scene'], pid)
        planned.append((pid, group, personal_seed))

    raw_cpu = np.asarray(raw64.detach().cpu().tolist(), dtype=np.float64)
    waves, records, coverage = [], [], []
    for index, (pid, group, personal_seed) in enumerate(planned):
        info = groups[group]; cfg, scene = info['cfg'], info['scene']
        session = f'rx{group[0]}:day{group[1]}:eq{group[2]}'
        # Exactly64 calls are required by the existing first-view hardware
        # capture wrapper.  Every personal process gets its actual returned RX.
        hardware = primary.ch.receiver_for_session(cfg, RECEIVER_SEED, session)
        if isinstance(hardware, type) or not is_dataclass(hardware):
            raise ValueError('The original physical session receiver is required')
        hardware_dict = asdict(hardware)
        if primary.sessions.setdefault(session, hardware_dict) != hardware_dict:
            raise ValueError('Hardware changed inside a TX-independent RX/day/eq session')
        personal = primary.residual.ResidualChannel(cfg, personal_seed, hardware)
        if info['snapshot'] is None:
            info['snapshot'] = primary.residual.ResidualChannel(cfg, info['environment_seed'], hardware)
        environment = info['snapshot']
        if personal.cfg is not cfg or personal.receiver is not hardware or personal.used or environment.used:
            raise ValueError('Fresh personal snapshot and unprocessed group environment required')
        before_private, before_streams = _private(personal), _streams(personal)
        if share_propagation:
            mix_propagation(personal, environment)
        if _private(personal) != before_private or _streams(personal) != before_streams:
            raise ValueError('Private primitive receiver/noise streams must be unchanged')
        a = personal.latent
        if (a.f_total != a.f_doppler+hardware.oscillator_cfo_hz
                or a.f_hat != a.f_total-a.residual_hz):
            raise ValueError('Orbital propagation must retain the personal receiver CFO and sync error')
        matches = all(_equal(getattr(a, name), getattr(environment.latent, name)) for name in PROPAGATION)
        normalized, _ = primary.ch.normalize_input(raw_cpu[index, 0]+1j*raw_cpu[index, 1])
        signal, native = personal.process(normalized)
        if (np.shape(signal) != (256,) or not np.isfinite(signal).all() or not isinstance(native, dict)
                or native.get('channel_equalization_applied') is not False
                or native.get('seed') != personal_seed or native.get('receiver') != hardware_dict
                or native.get('boundary') != 'synthetic_reflect'
                or any(key not in native for key in ('state_end', 'receiver_lock_status',
                    'quality_snr_db', 'signal_to_added_noise_db', 'agc_gain_db', 'agc_limited'))):
            raise ValueError('Finite complete native residual metadata/output required; no dropping or boundary changes')
        if not all(math.isfinite(float(value)) for value in before_private.values()):
            raise ValueError('Finite original personal receiver primitives required')
        waves.append(np.stack((signal.real, signal.imag)))
        coverage.append((group[0], group[1], scene, native['state_end'], native['receiver_lock_status']))
        records.append(serial(dict(native, scene=scene, sample_id=pid, session_id=session,
            namespace=namespace, group=list(group), group_id=session, environment_group=list(group),
            scene_seed=int(info['scene_seed']), personal_seed=int(personal_seed),
            environment_seed=int(info['environment_seed']), packet_seed=int(personal_seed),
            share_propagation=share_propagation, propagation_source='environment' if share_propagation else 'personal',
            propagation_fields=list(PROPAGATION), propagation_matches_environment=matches,
            propagation_arrays_not_aliased=True, private_streams_preserved=True,
            private_stream_seeds={name:int(primary.ch.stable_seed(personal_seed, name)) for name in PRIVATE_STREAMS},
            packet_phase=before_private['phase'], packet_tracking_z=before_private['tracking_z'],
            packet_residual_hz=before_private['residual_hz'], packet_coarse_error_hz=before_private['coarse_doppler_error_hz'],
            f_total=float(a.f_total), initial_f_hat=float(a.f_hat),
            receiver_seed=RECEIVER_SEED, TX_used_in_group=False, sig_i_is_TX=False,
            same_raw_input=True, native_noise_chain_retained=True, native_agc_recomputed=True,
            environment_initialized_not_processed=True, environment_index=info['index'],
            environment_initializations_this_step=len(groups), personal_initializations_this_step=64,
            native_waveforms_this_step=64, environment_processes_this_step=0)))
    if any(info['snapshot'] is None or info['snapshot'].used for info in groups.values()):
        raise ValueError('Exactly one unused constructed environment per group required')
    output = tensor_iq(waves, raw64.device)
    if output.shape != raw64.shape or not bool(torch.isfinite(output).all()):
        raise ValueError('All64 finite received waveforms must survive')
    for key in coverage:
        primary.coverage[key] += 1
    primary.calls += 1; primary.packets += 64
    primary._prop76_environment_initializations = getattr(primary, '_prop76_environment_initializations', 0)+len(groups)
    primary._prop76_personal_initializations = getattr(primary, '_prop76_personal_initializations', 0)+64
    primary._prop76_native_process_calls = getattr(primary, '_prop76_native_process_calls', 0)+64
    return output, records
