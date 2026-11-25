"""Source-only receive extension of the registered LoRa Residual snapshot."""
from dataclasses import replace
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np

from .data import FS_HZ, VIEWS, WINDOW_START
from .packet_data import PACKET_LEN
from .packet_physics import CONTEXT, NATIVE_RATE, RESIDUAL_LATENCY, hardware, residual_channel


STRIDE = 1024
ROUTE = 'native_residual_w24_v1'
LATENCY = round(RESIDUAL_LATENCY * NATIVE_RATE / FS_HZ)


@lru_cache(maxsize=1)
def residual_snapshot_type():
    # Explicitly load this release's aperture extension while the original
    # legacy generator continues to use its pinned propagation library.
    path = Path(__file__).resolve().parents[2] / 'code' / 'leo_practical' / 'residual.py'
    spec = importlib.util.spec_from_file_location('leo_practical.residual_w24_aperture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResidualChannel


@lru_cache(maxsize=None)
def native_residual_config(scene):
    return replace(residual_channel(scene), fs_hz=NATIVE_RATE, fractional_delay_half_length=LATENCY)


def native_residual(packet, scene, receiver, physical_index, namespace, *, trace=False, component_trace=False):
    """Return eight compensated snapshots with receiver-side IQ at both edges."""
    from leo_practical.channel import normalize_input, sample_geometry, stable_seed

    if scene not in VIEWS or packet.shape != (PACKET_LEN,):
        raise ValueError('Expected a registered LoRa scene and one 8192-point packet')
    if scene == 'clean':
        if component_trace:
            return packet, packet, np.zeros_like(packet)
        return (packet, ()) if trace else packet
    x, _ = normalize_input(packet)
    padded = np.pad(x, (CONTEXT, LATENCY))
    parts = (namespace, scene, receiver, str(physical_index))
    cfg = native_residual_config(scene)
    rx = hardware(scene, f'{namespace}/{receiver}')
    geometry = sample_geometry(cfg, np.random.default_rng(stable_seed(2026112207, *parts, 'geometry')))
    implementation = residual_snapshot_type()
    rows, records = [], []
    signals, noises = [], []
    for chirp, start in enumerate(range(0, PACKET_LEN, STRIDE)):
        shifted = CONTEXT + start + LATENCY
        snapshot = implementation(cfg, stable_seed(2026112207, *parts, 'chirp', chirp), rx, geometry)
        received, meta, *components = snapshot.process(
            padded[shifted:shifted + STRIDE], preceding_context=padded[shifted - CONTEXT:shifted],
            aperture_context_samples=WINDOW_START, noise_reference_fs_hz=FS_HZ, return_traces=component_trace,
        )
        if (meta['processing_route'] != 'residual' or meta['full_channel_waveform_synthesized']
                or meta['channel_equalization_applied'] or meta['boundary'] != 'provided_context'
                or meta['numerical_filter_latency_samples'] != LATENCY
                or meta['nominal_snapshot_samples'] != 1000 or len(received) != STRIDE):
            raise ValueError('Unexpected native Residual snapshot')
        rows.append(received)
        if trace:
            records.append(meta)
        if component_trace:
            source, = components
            gain = 10 ** (meta['agc_gain_db'] / 20)
            signals.append(source['pre_agc_signal'] * gain)
            noises.append(source['pre_agc_noise'] * gain)
    result = np.concatenate(rows)
    if component_trace:
        signal, noise = np.concatenate(signals), np.concatenate(noises)
        if not np.allclose(result, signal + noise, rtol=2e-6, atol=1e-7):
            raise ValueError('Residual additive components do not reproduce the received IQ')
        return result, signal, noise
    return (result, tuple(records)) if trace else result
