"""Deterministic independent snapshots; NumPy core with optional Torch adapter."""
from __future__ import annotations

import numpy as np
from .execution import REFERENCE

from .channel import ChannelStream, receiver_for_session, stable_seed, normalize_input, STATES


def apply_leo_practical_channel_batch(
    x_iq, cfg, *, seed, sample_ids, session_ids, realization_namespace,
    receiver_seed=0, normalize_input_rms=True, return_meta=True,
    receiver_processor=None, execution=REFERENCE,
):
    """Apply independent channels to (B,2,T) real IQ or (B,T) complex IQ.

    Returns (output, metadata_or_None, terminal_state_ids), matching the broad
    CVS channel return convention. This function does NOT register a scene in
    historical sat_channel.py. Explicit IDs/seeds replace an order-dependent
    global RNG. Use separate realization_namespace values for train/evaluation
    and include epoch/view in training when fresh augmentation is desired.

    Hardware is stable per session_id + receiver_seed across scenarios, seeds,
    sample order and batch partition. Never pass TX labels as session IDs.
    Torch inputs are detached, copied to CPU and returned to the original device
    and dtype. This reference implementation is not a GPU-optimized kernel and
    is not differentiable. Metadata are ordinary Python objects, not tensors.
    receiver_processor must be None: external receiver transforms/equalizers
    are forbidden under the RFF-safe policy.
    """
    is_torch = hasattr(x_iq, "detach") and hasattr(x_iq, "device")
    if is_torch:
        import torch
        if not (x_iq.is_floating_point() or x_iq.is_complex()):
            raise TypeError("Torch IQ must be floating point or complex")
        dtype, device = x_iq.dtype, x_iq.device
        arr = x_iq.detach().to(device="cpu", dtype=(torch.complex128 if x_iq.is_complex() else torch.float64)).numpy()
    else:
        arr = np.asarray(x_iq)
        if not (np.issubdtype(arr.dtype, np.floating) or np.iscomplexobj(arr)):
            raise TypeError("IQ must be floating point or complex")
    complex_layout = np.iscomplexobj(arr)
    if complex_layout:
        if arr.ndim != 2:
            raise ValueError("complex input requires (B,T)")
        raw = arr
    else:
        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError("real input requires (B,2,T)")
        raw = arr[:, 0] + 1j*arr[:, 1]
    if raw.shape[0] == 0 or raw.shape[1] < 4 or not np.isfinite(raw).all():
        raise ValueError("empty, too-short or nonfinite IQ batch")
    count = raw.shape[0]
    if len(sample_ids) != count or len(session_ids) != count:
        raise ValueError("sample/session IDs must have one entry per record")
    ids = [str(v) for v in sample_ids]
    if len(set(ids)) != count or any(not v for v in ids):
        raise ValueError("sample IDs must be nonempty and unique within a batch")
    if not isinstance(realization_namespace, str) or not realization_namespace.strip():
        raise ValueError("an explicit realization namespace is required")
    outputs, metadata, states = [], [], []
    # Receiver is frozen and depends only on cfg/receiver_seed/session, not the
    # independent per-record stream. Reuse within this call, never stream state.
    hardware_by_session = {}
    for i, record in enumerate(raw):
        session = str(session_ids[i])
        if not session:
            raise ValueError("empty receiver session ID")
        if normalize_input_rms:
            record, input_rms = normalize_input(record)
        else:
            input_rms = float(np.sqrt(np.mean(np.abs(record)**2)))
            if input_rms == 0:
                raise ValueError("zero-energy input record")
        if session not in hardware_by_session:
            hardware_by_session[session] = receiver_for_session(cfg, receiver_seed, session)
        hardware = hardware_by_session[session]
        derived_seed = stable_seed(seed, realization_namespace, cfg.scenario, ids[i])
        if cfg.processing_route == "residual":
            from .residual import ResidualChannel
            stream = ResidualChannel(cfg, derived_seed, hardware, receiver_processor=receiver_processor, execution=execution)
        else:
            stream = ChannelStream(cfg, derived_seed, hardware, receiver_processor=receiver_processor, execution=execution)
        y, meta = stream.process(record)
        meta.update(sample_id=ids[i], independent_snapshot=True,
                    realization_namespace=realization_namespace,
                    augmentation_seed=int(seed), receiver_seed=int(receiver_seed),
                    input_original_rms=input_rms,
                    input_normalization="unit_rms" if normalize_input_rms else "shared_reference_units")
        outputs.append(y)
        metadata.append(meta)
        states.append(STATES.index(meta["state_end"]))
    out = np.stack(outputs)
    if not complex_layout:
        out = np.stack([out.real, out.imag], axis=1).astype(np.float32)
    state_ids = np.asarray(states, dtype=np.int64)
    if is_torch:
        out = torch.from_numpy(out).to(device=device, dtype=dtype)
        state_ids = torch.from_numpy(state_ids).to(device=device)
    return out, metadata if return_meta else None, state_ids
