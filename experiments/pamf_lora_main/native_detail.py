"""Oversampled LoRa sidebands and a source-trained two-component correction."""

import numpy as np
import torch
from scipy.signal import resample_poly
from torch import nn

from experiments.pamf_reception_restore.models import Core

from .data import WINDOW_START
from .dura_repair import complex_iq, no_tone, real_iq
from .packet_data import CHIRPS
from .packet_physics import CONTEXT, channel, chirp_samples, chirp_windows, hardware


CAP = .075
CENTERS = (.25, -.25, .5)


def band_windows(wave, offset):
    starts = np.asarray(CHIRPS) * 1024 + WINDOW_START - 125 + offset
    if starts[0] < 0 or starts[-1] + 1250 > len(wave):
        raise ValueError('Sidebands need real preamble crop context')
    time = starts[:, None] + np.arange(1250)[None]
    window = wave[time]
    phase = np.exp(-2j * np.pi * np.asarray(CENTERS)[None, :, None] * time[:, None])
    translated = resample_poly(window[:, None] * phase, up=32, down=125, axis=-1)[..., 32:288]
    base = chirp_samples(wave, offset)
    scale = np.sqrt(np.mean(np.abs(base)**2))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Missing finite power in the registered crop')
    return np.asarray(np.stack((translated.real, translated.imag), axis=2) / scale, dtype=np.float32)


def native_pair(row, *, sidebands=True):
    """Source simulator truth is returned separately and never supplied to the head."""
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed

    raw, rx, pid, scene, namespace, draw = row
    if scene == 'clean':
        y = chirp_windows(raw)
        return y, band_windows(raw, 0) if sidebands else None, np.zeros_like(y)
    tx, _ = normalize_input(raw)
    pieces = (namespace, scene, rx, str(pid))
    if draw is not None:
        pieces = (*pieces, 'draw', draw)
    cfg = channel(scene)
    stream = ChannelStream(cfg, stable_seed(2026112207, *pieces), hardware(scene, f'{namespace}/{rx}'))
    observed, meta, trace = stream.process(tx[CONTEXT:], preceding_context=tx[:CONTEXT], return_traces=True)
    delay = meta['numerical_filter_latency_samples']
    if (meta['channel_equalization_applied'] or meta['boundary'] != 'provided_context'
            or delay != cfg.fractional_delay_half_length):
        raise ValueError('Expected the original native full practical operator')
    gain = 10 ** (meta['agc_gain_db'] / 20)
    reconstructed = ((trace['pre_agc_signal'] + trace['pre_agc_noise']) * gain).astype(observed.dtype)
    if not np.array_equal(observed, reconstructed):
        raise ValueError('Signal and noise must reproduce the quantized native observation')
    offset = delay - CONTEXT
    received_band = chirp_samples(observed, offset)
    noise = chirp_samples((trace['pre_agc_noise'] * gain).astype(observed.dtype), offset)
    scale = np.sqrt(np.mean(np.abs(received_band)**2))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Invalid native trace scale')
    adjacent = band_windows(observed, offset) if sidebands else None
    return chirp_windows(observed, offset), adjacent, np.asarray(
        np.stack((noise.real, noise.imag), axis=1) / scale, dtype=np.float32)


def bounded_components(noise, shape):
    noise, shape = no_tone(noise), no_tone(shape)
    length = (noise + shape).abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    scale = (CAP / length).clamp_max(1.)
    return noise * scale, shape * scale


@torch.no_grad()
def feature_pack(observed, base, adjacent, tracker):
    work, rot, *_ = tracker.track(base)
    scale = work.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    main = no_tone(work / scale)
    raw = no_tone(complex_iq(observed) * rot.conj() / scale)
    neighbor = torch.complex(adjacent[:, :, :, 0], adjacent[:, :, :, 1]) * (rot.conj() / scale)[:, :, None]
    aux = torch.stack((neighbor.real, neighbor.imag), dim=3).flatten(2, 3)
    return torch.cat((real_iq(main), real_iq(raw), aux), dim=2), (main, rot, scale)


@torch.no_grad()
def targets(clean, known_noise, coordinates, tracker):
    observed, rotation, rms = coordinates
    target, *_ = tracker.track(clean)
    ground_rms = target.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    total = no_tone(target / ground_rms) - observed
    noise = no_tone(-complex_iq(known_noise) * (rotation.conj() / rms))
    return bounded_components(noise, total - noise)


class NativeDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Core(incoming=10, outgoing=4)

    def forward(self, inputs):
        out = self.head(inputs.flatten(0, 1)).reshape(len(inputs), 6, 4, 256)
        return bounded_components(complex_iq(out[:, :, :2]), complex_iq(out[:, :, 2:]))


def total_risk(estimated, desired):
    delta = estimated[0] + estimated[1] - desired[0] - desired[1]
    power = sum((target.abs().square().mean(-1) for target in desired)) + .002
    signal = delta.abs().square().mean(-1) / power
    components = sum(((pred - target).abs().square().mean(-1) / power
                      for pred, target in zip(estimated, desired, strict=True)))
    return (signal + .25 * components).mean()


def restore(base, delta, coordinates):
    _, rotation, rms = coordinates
    return base + real_iq(delta * rotation * rms)
