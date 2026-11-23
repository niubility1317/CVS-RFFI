"""Add a fresh noise draw without changing the registered reception operator."""

import numpy as np

from .packet_data import CHIRPS
from .packet_physics import (CONTEXT, RESIDUAL_LATENCY, channel, chirp_grid, chirp_samples,
                             hardware, pack_windows, residual_channel)


def replace_noise(observed, meta, trace, rng):
    signal, noise = (trace[k] for k in ('pre_agc_signal', 'pre_agc_noise'))
    gain = 10.**(meta['agc_gain_db']/20.)
    if not np.array_equal(((signal+noise)*gain).astype(observed.dtype), observed):
        raise ValueError('Fixed-action traces must reconstruct the native observation')
    raw = rng.normal(size=(len(signal), 2))
    raw = (raw[:, 0]+1j*raw[:, 1]) * np.sqrt(meta['added_noise_variance_reference']/2.)
    added = trace['operator_main']*raw + trace['operator_image']*raw.conj()
    second = ((signal+added)*gain).astype(observed.dtype)
    return second, (gain*signal).astype(observed.dtype)


def pack_views(registered, second, signal):
    size = np.sqrt(np.mean(np.abs(registered)**2))
    want = np.stack((second, signal))
    scaled = np.stack((want.real, want.imag), axis=2) / size
    return pack_windows(registered), scaled.astype(np.float32)


def paired(row, route, namespace='source-U'):
    """Return registered observation, second observation, and shared noiseless signal."""
    from leo_practical.channel import ChannelStream, normalize_input, sample_geometry, stable_seed
    from leo_practical.residual import ResidualChannel

    raw, scene, rx, identity = row
    x, _ = normalize_input(raw)
    parts = (namespace, scene, rx, str(identity))
    receiver = hardware(scene, f'{namespace}/{rx}')
    rng = np.random.default_rng(stable_seed(2026112701, *parts, 'fresh-noise'))
    if route == 'full':
        cfg = channel(scene)
        stream = ChannelStream(cfg, stable_seed(2026112207, *parts), receiver)
        y, meta, trace = stream.process(x[CONTEXT:], preceding_context=x[:CONTEXT],
                                        return_traces=True, return_operator=True)
        if meta['channel_equalization_applied']:
            raise ValueError('Fixed-action re-noising requires the registered no-EQ input')
        new, signal = replace_noise(y, meta, trace, rng)
        offset = meta['numerical_filter_latency_samples'] - CONTEXT
        return pack_views(*(chirp_samples(wave, offset) for wave in (y, new, signal)))
    if route != 'residual':
        raise ValueError('A registered practical route is required')
    cfg = residual_channel(scene)
    geometry = sample_geometry(cfg, np.random.default_rng(stable_seed(2026112207, *parts, 'geometry')))
    start = 32 + RESIDUAL_LATENCY
    values = []
    for chirp, grid in zip(CHIRPS, chirp_grid(x), strict=True):
        seed = stable_seed(2026112207, *parts, 'chirp', chirp)
        stream = ResidualChannel(cfg, seed, receiver, geometry)
        y, meta, trace = stream.process(grid[start:start+256], preceding_context=grid[:start],
                                        return_traces=True, return_operator=True)
        if meta['channel_equalization_applied']:
            raise ValueError('Fixed-action re-noising requires the registered no-EQ snapshot')
        values.append((y, *replace_noise(y, meta, trace, rng)))
    return pack_views(*(np.stack(v) for v in zip(*values, strict=True)))
