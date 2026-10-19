"""Post-receiver IQ erasure with named, packet-local random streams.

The native practical channel and AGC must already have run. This function
never rerenders physics, changes receiver quality, renormalizes IQ, or reads
TX labels. The reference window [80, 160) remains observed in both modes.
"""
import hashlib
import json
from collections.abc import Mapping
from numbers import Integral

import numpy as np

try:
    import torch
except ImportError:  # NumPy-only diagnostic/test environments remain usable.
    torch = None


REFERENCE = (80, 160)
PROBABILITY = 0.5
_FIELDS = ('eq_i', 'rx_i', 'day_i', 'sig_i')
_STREAM = 'FcrossRX/train/post-practical-prefix81/v1'


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer')
    result = int(value)
    if result < minimum:
        raise ValueError(name + ' is out of range')
    return result


def _physical_key(record):
    # Existing native sample_id: (eq, rx, day, sig, base_index). Exclude
    # base_index and labels: masking is not conditional on a TX-ordered index.
    present = [field in record for field in _FIELDS]
    if any(present) and not all(present):
        raise ValueError('Explicit physical metadata must contain all four fields')
    if all(present):
        key = tuple(_integer(record[field], field) for field in _FIELDS)
        value = record.get('sample_id')
        if isinstance(value, (tuple, list)) and len(value) == 5:
            native_key = tuple(_integer(v, name) for v, name in zip(value[:4], _FIELDS))
            if native_key != key:
                raise ValueError('Explicit physical fields differ from native sample_id')
    else:
        value = record.get('sample_id')
        if not isinstance(value, (tuple, list)) or len(value) != 5:
            raise ValueError('Native sample_id or explicit eq/rx/day/sig fields required')
        key = tuple(_integer(v, name) for v, name in zip(value[:4], _FIELDS))
    if key[0] != 1 or key[1] not in (1, 3, 4, 6, 8) or key[2] not in (1, 2, 3):
        raise ValueError('Only fullsource eq1 RX1/3/4/6/8 day1/2/3 metadata')
    return key


def _rng(seed, epoch, step, key, stream):
    # Local streams do not move Python, NumPy, torch or CUDA global RNGs.
    payload = json.dumps([_STREAM, seed, epoch, step, list(key), stream],
                         separators=(',', ':'), ensure_ascii=True).encode('ascii')
    local_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), 'big')
    return np.random.Generator(np.random.PCG64(local_seed))


def _length(record, explicit):
    native = None
    source = None
    if 'residual_kernel_length' in record:
        native = _integer(record['residual_kernel_length'], 'residual_kernel_length', 2) - 1
        source = 'native_residual_kernel_length_minus_one'
    elif 'propagation_kernel_length' in record:
        if record.get('channel_equalization_applied') is not False:
            raise ValueError('Propagation length requires explicit native EQ-off evidence')
        native = _integer(record['propagation_kernel_length'], 'propagation_kernel_length', 2) - 1
        source = 'native_EQoff_propagation_kernel_length_minus_one'
    if explicit is not None:
        explicit = _integer(explicit, 'explicit mask length', 1)
        if native is not None and explicit != native:
            raise ValueError('Explicit mask length differs from actual native kernel')
        if native is None:
            native, source = explicit, 'wrapper_actual_kernel_length_minus_one'
    if native is None:
        raise ValueError('Actual native kernel length or explicit wrapper-derived length required')
    if not 1 <= native <= REFERENCE[0] or native > 256 - REFERENCE[1]:
        raise ValueError('Mask length must preserve the fixed reference window')
    return native, source


def apply_mask(x, metadata, *, mode, epoch, step, seed=2026092701, lengths=None):
    """Return (new_IQ, records) without modifying input or physical metadata.

    x is finite float32 NumPy or torch IQ with shape (N, 2, 256). Metadata is
    the corresponding list of complete native practical records. If light
    metadata omits kernel lengths, lengths must be the actual K-1 supplied
    by the wrapper (one scalar or N integers). Numerical group delay is
    never used as a substitute. Repeated physical handles are allowed.

    Epoch/step use cumulative E21..30 / 4441..6660 namespaces. PREFIX erases
    [0,K-1). RANDOM samples every integer start whose equal-length segment
    is wholly outside [80,160), uniformly. Both modes use the same local
    Bernoulli(0.5) enable draw. Call once per As view and reuse that result
    in both existing RDrop forwards.

    Powers are full-record means of I**2+Q**2. removed_power includes the
    selected segment only when enabled; no output correction is made.
    """
    if mode not in ('PREFIX', 'RANDOM'):
        raise ValueError('Mode must be PREFIX or RANDOM')
    epoch = _integer(epoch, 'epoch', 21)
    step = _integer(step, 'step', 4441)
    seed = _integer(seed, 'seed')
    if epoch > 30 or step > 6660 or epoch != (step - 1) // 222 + 1:
        raise ValueError('Only matched cumulative E21..30 steps4441..6660')
    is_tensor = torch is not None and torch.is_tensor(x)
    if is_tensor:
        valid_dtype = x.dtype == torch.float32
        finite = bool(torch.isfinite(x).all()) if valid_dtype else False
    elif isinstance(x, np.ndarray):
        valid_dtype = x.dtype == np.float32
        finite = bool(np.isfinite(x).all()) if valid_dtype else False
    else:
        raise ValueError('IQ must be a float32 NumPy array or torch tensor')
    if not valid_dtype or x.ndim != 3 or tuple(x.shape[1:]) != (2, 256) or x.shape[0] < 1 or not finite:
        raise ValueError('Finite float32 IQ with shape(N,2,256) required')
    if not isinstance(metadata, (tuple, list)) or len(metadata) != len(x):
        raise ValueError('One native metadata record is required per packet')
    if any(not isinstance(record, Mapping) for record in metadata):
        raise ValueError('Native metadata entries must be dictionaries')
    if lengths is None:
        supplied = [None] * len(x)
    elif isinstance(lengths, Integral) and not isinstance(lengths, (bool, np.bool_)):
        supplied = [lengths] * len(x)
    elif ((isinstance(lengths, (tuple, list))
           or isinstance(lengths, np.ndarray) and lengths.ndim == 1)
          and len(lengths) == len(x)):
        supplied = list(lengths)
    else:
        raise ValueError('Explicit lengths must be one integer or one per packet')
    # Validate all plans first; no stream or mutation is shared across rows.
    plans = []
    for record, explicit in zip(metadata, supplied):
        key = _physical_key(record)
        length, source = _length(record, explicit)
        enable_draw = float(_rng(seed, epoch, step, key, 'enable').random())
        enabled = enable_draw < PROBABILITY
        if mode == 'PREFIX':
            start = 0
        else:
            left_count = REFERENCE[0] - length + 1
            right_count = 256 - length - REFERENCE[1] + 1
            choice = int(_rng(seed, epoch, step, key, 'start').integers(left_count + right_count))
            start = choice if choice < left_count else REFERENCE[1] + choice - left_count
        plans.append((key, length, source, enabled, enable_draw, start))
    result = x.clone() if is_tensor else x.copy()
    numeric = (np.asarray(x.detach().to(device='cpu', dtype=torch.float64).tolist(), dtype=np.float64)
               if is_tensor else x.astype(np.float64, copy=False))
    power = np.sum(numeric * numeric, axis=1)
    records = []
    for index, (key, length, source, enabled, enable_draw, start) in enumerate(plans):
        original_power = float(power[index].sum() / 256)
        removed_power = float(power[index, start:start + length].sum() / 256) if enabled else 0.0
        if enabled:
            result[index, :, start:start + length] = 0.0
        records.append(dict(
            mode=mode, enabled=enabled, enable_draw=enable_draw, probability=PROBABILITY,
            begin=start, length=length, end=start + length, length_source=source,
            physical_key=list(key), epoch=epoch, step=step,
            namespace=_STREAM, original_power=original_power, removed_power=removed_power,
            removed_power_fraction=removed_power / original_power if original_power > 0.0 else 0.0,
            output_power=original_power - removed_power,
            after_complete_practical_and_AGC=True, IQ_renormalized=False,
            receiver_state_recomputed=False, physical_metadata_modified=False,
            reference_window_preserved=True, TX_label_used=False,
        ))
    return result, records
