"""Same complete practical realization; only a new native added-noise stream.

The native receiver recomputes quality, compensation and AGC.  Its quality
proxy precedes noise generation, so non-noise metadata must replay exactly;
AGC follows signal+noise and is deliberately allowed to change.
"""
from dataclasses import asdict, is_dataclass
import math
import re

import numpy as np
import torch

PHYSICAL_FIELDS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
FIT = (1, 3, 4, 6, 8)
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
TRAIN_SEED = 20261016421
SECOND_SUFFIX = '/view1-noise-independent'
NOISE_DEPENDENT_FIELDS = frozenset((
    'measured_added_noise_power_pre_agc', 'signal_to_added_noise_db',
    'agc_gain_db', 'agc_limited', 'output_metrics'))
_NAMESPACE = re.compile(r'FcrossRX/train/e(\d+)/step(\d+)/L_S')


def serial(value):
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    if isinstance(value, complex): return [value.real, value.imag]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def tensor_iq(array, device):
    array = np.ascontiguousarray(array, dtype=np.float32)
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch.float32).reshape(array.shape).to(device)


def _integers(value):
    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('Physical metadata must be an integer vector')
        value = value.detach().cpu().tolist()
    else:
        value = list(value)
    if len(value) != 64 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError('Exactly64 integer physical metadata entries required')
    return [int(v) for v in value]


def second_noise_view(primary, raw64, meta, first_records, hardware64):
    """Return As64 with unchanged physics/hardware and independent added noise.

    No global RNG, monkeypatch, extra physical proposal or first-view process
    is used.  Exactly one new native ResidualChannel.process is called per
    packet; input normalization is exactly the primary adapter's operation.
    """
    if (not torch.is_tensor(raw64) or raw64.shape != (64, 2, 256)
            or raw64.dtype != torch.float32 or not bool(torch.isfinite(raw64).all())
            or not isinstance(meta, dict) or any(k not in meta for k in PHYSICAL_FIELDS)
            or not isinstance(first_records, (tuple, list)) or len(first_records) != 64
            or not isinstance(hardware64, (tuple, list)) or len(hardware64) != 64):
        raise ValueError('Exactly64 finite float32 raw L packets, first records and exact RX objects required')
    values = {key: _integers(meta[key]) for key in PHYSICAL_FIELDS}
    planned = []; namespaces = set()
    for index, first in enumerate(first_records):
        if not isinstance(first, dict):
            raise ValueError('Original full native first-view metadata required')
        pid = tuple(values[key][index] for key in PHYSICAL_FIELDS)
        session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
        namespace = first.get('namespace'); match = _NAMESPACE.fullmatch(namespace) if isinstance(namespace, str) else None
        if (not match or tuple(first.get('sample_id', ())) != pid
                or first.get('session_id') != session or pid[0] != 1
                or pid[1] not in FIT or pid[2] not in (1, 2, 3)
                or pid[3] < 0 or pid[4] < 0):
            raise ValueError('Original full-source physical IDs/session/namespace required')
        epoch, step = (int(v) for v in match.groups())
        if epoch not in range(21, 31) or step not in range(4441, 6661) or epoch != (step-1)//222+1:
            raise ValueError('Only the existing cumulative E21..30 physical namespace')
        namespaces.add(namespace)
        scene_rng = np.random.default_rng(primary.ch.stable_seed(TRAIN_SEED, namespace, 'scene', pid))
        scene = str(scene_rng.choice(TRAIN_SCENES))
        if first.get('scene') != scene or scene not in primary.configs:
            raise ValueError('First nominal scene draw must remain unchanged')
        cfg = primary.configs[scene]
        if (cfg.scenario != scene or cfg.processing_route != 'residual' or cfg.mode != 'post_sync'
                or cfg.equalization_enabled or cfg.fs_hz != 25000000. or cfg.fc_hz != 2462000000.
                or cfg.reference_snr_db != 30. or cfg.boundary != 'reflect'):
            raise ValueError('Unchanged full practical residual/post_sync/EQoff configuration required')
        hardware = hardware64[index]
        if (isinstance(hardware, type) or not is_dataclass(hardware)
                or serial(asdict(hardware)) != serial(primary.sessions.get(session))
                or serial(first.get('receiver')) != serial(asdict(hardware))):
            raise ValueError('Pass the exact first-view stable physical hardware object')
        physics_seed = primary.ch.stable_seed(TRAIN_SEED, namespace, scene, pid)
        if first.get('seed') != physics_seed or first.get('channel_equalization_applied') is not False:
            raise ValueError('Original first physical seed and EQoff evidence required')
        noise_namespace = namespace + SECOND_SUFFIX
        noise_seed = primary.ch.stable_seed(TRAIN_SEED, noise_namespace, scene, pid)
        planned.append((pid, session, namespace, scene, cfg, physics_seed, noise_namespace, noise_seed, hardware))
    if len(namespaces) != 1:
        raise ValueError('One existing native L step must supply the entire64 pack')

    raw_cpu = np.asarray(raw64.detach().cpu().tolist(), dtype=np.float64)
    waves, records = [], []
    for index, plan in enumerate(planned):
        pid, session, namespace, scene, cfg, physics_seed, noise_namespace, noise_seed, hardware = plan
        normalized, _ = primary.ch.normalize_input(raw_cpu[index, 0] + 1j*raw_cpu[index, 1])
        # Recreate all original streams from the original seed and receiver.
        # Never change the constructor root seed to obtain new noise.
        channel = primary.residual.ResidualChannel(cfg, physics_seed, hardware)
        if channel.cfg is not cfg or channel.seed != physics_seed or channel.receiver is not hardware:
            raise ValueError('The new native instance must retain cfg/physical seed/exact hardware')
        streams = channel.latent.rngs
        if not isinstance(streams, dict) or 'noise' not in streams:
            raise ValueError('Actual native latent.rngs noise interface is required')
        original_streams = dict(streams)
        streams['noise'] = np.random.default_rng(noise_seed)
        if any(streams[key] is not value for key, value in original_streams.items() if key != 'noise'):
            raise ValueError('Only the native added-noise RNG may change')
        signal, actual = channel.process(normalized)
        if (not isinstance(actual, dict) or not np.isfinite(signal).all()
                or np.shape(signal) != (256,) or actual.get('channel_equalization_applied') is not False):
            raise ValueError('Full finite native residual output required; no sample dropping')
        native = serial(actual); first = first_records[index]
        if set(native) - set(first):
            raise ValueError('Complete comparable first-view native metadata required')
        checked = sorted(set(native) - NOISE_DEPENDENT_FIELDS)
        if any(native[key] != serial(first[key]) for key in checked):
            raise ValueError('Non-noise physics/receiver metadata did not replay exactly')
        if not NOISE_DEPENDENT_FIELDS.issubset(native):
            raise ValueError('Full native added-noise and independently recomputed AGC evidence required')
        power = native['measured_added_noise_power_pre_agc']
        if not isinstance(power, (int, float)) or not math.isfinite(power) or power <= 0:
            raise ValueError('Finite positive native measured added-noise power required')
        waves.append(np.stack((signal.real, signal.imag)))
        records.append(serial(dict(actual, scene=scene, sample_id=pid, session_id=session,
            namespace=noise_namespace, physics_namespace=namespace,
            physics_seed=int(physics_seed), noise1_seed=int(noise_seed), noise1_namespace=noise_namespace,
            scope='same complete practical realization; independent native added-noise RNG only',
            noise_stream_only_resampled=True, non_noise_metadata_equal=True,
            non_noise_metadata_fields_checked=checked, hardware_object_reused=True, from_same_raw=True,
            native_noise_chain_retained=True, output_white_noise_injection=False, exact_noise_energy_calibration=False,
            receiver_state_recomputed=True, compensation_recomputed=True,
            native_agc_recomputed=True, first_agc_reused=False,
            added_noise_energy_pre_agc=float(power)*256)))
    result = tensor_iq(waves, raw64.device)
    if result.shape != raw64.shape or result.dtype != raw64.dtype or not bool(torch.isfinite(result).all()):
        raise ValueError('Complete finite float32 As64 output required')
    return result, records
