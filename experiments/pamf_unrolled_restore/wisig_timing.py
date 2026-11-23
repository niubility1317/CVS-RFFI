"""Local Wi-Fi timing from the received L-STF/L-LTF transition."""

import numpy as np

from .wisig_short_training import PERIOD, _estimate, _remove_cfo


# Standard legacy long-training subcarriers k=-32,…,31. The OFDM symbol is
# evaluated directly at 25 MS/s (80 samples per symbol), after a 40-sample CP.
LONG = np.array([
    0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
])
NOMINAL = 10 * PERIOD
ANCHOR = PERIOD // 4
MATCH = 16
STARTS = np.arange(NOMINAL - 2 * PERIOD, NOMINAL + 2 * PERIOD + 1)
READOUTS = ('direct', 'timing_only', 'timing_calibrated', 'timing_protocol')
_EPS = 1e-8


def long_template(t):
    return np.exp(2j * np.pi * (np.asarray(t)[..., None] - 40) * np.arange(-32, 32) / 80) @ LONG / np.sqrt(52)


def _energy_ratio(y, start):
    residual = np.abs(y[:, PERIOD:] - y[:, :-PERIOD])**2
    energy = np.abs(y[:, PERIOD:])**2 + np.abs(y[:, :-PERIOD])**2
    before = start[:, None] - np.arange(PERIOD, 0, -1) - PERIOD
    after = start[:, None] + np.arange(MATCH) - PERIOD
    mean_before = residual[:, before].sum(axis=-1) / np.maximum(energy[:, before].sum(axis=-1), _EPS)
    mean_after = residual[:, after].sum(axis=-1) / np.maximum(energy[:, after].sum(axis=-1), _EPS)
    return np.log((mean_after + .05) / (mean_before + .05))


def _long_match(y, start):
    samples = y[:, start[:, None] + np.arange(MATCH)]
    template = long_template(np.arange(MATCH))
    correlation = np.sum(samples * template.conj(), axis=-1)
    power = np.sum(np.abs(samples)**2, axis=-1) * np.sum(np.abs(template)**2)
    return np.clip(np.abs(correlation)**2 / np.maximum(power, _EPS), 0, 1)


def estimate_timing(y):
    """Return the strongest delayed/advanced hypothesis and its margin over nominal."""
    estimate, _ = _estimate(y)
    synced = _remove_cfo(y, estimate.coarse_rad_per_sample)
    scores = _energy_ratio(synced, STARTS) + 2 * _long_match(synced, STARTS)
    nominal = np.abs(STARTS - NOMINAL) <= ANCHOR
    on_time = scores[:, nominal].max(axis=1)
    shifted = scores[:, ~nominal]
    best = shifted.argmax(axis=1)
    margin = shifted[np.arange(len(y)), best] - on_time
    return STARTS[~nominal][best] - NOMINAL, margin, estimate


def realign(z, delta, *, protocol=False, cfo=None):
    """Retain exact observed overlap. Fill missing boundaries explicitly."""
    size = z.shape[-1]
    coordinate = np.arange(size)[None, :] + delta[:, None]
    valid = (coordinate >= 0) & (coordinate < size)
    mirrored = np.where(coordinate < 0, -coordinate, np.minimum(coordinate, 2 * (size - 1) - coordinate))
    y = np.take_along_axis(z, mirrored, axis=-1)
    if protocol:
        # Only extrapolate a missing suffix using the known L-LTF waveform.
        # Current observed samples retain their absolute phase and amplitude.
        samples = np.maximum(0, NOMINAL + delta)[:, None] + np.arange(MATCH)
        observed = np.take_along_axis(z, samples, axis=-1)
        base = long_template(np.arange(MATCH)) * np.exp(1j * cfo[:, None] * samples)
        gain = np.sum(observed * base.conj(), axis=-1) / np.sum(np.abs(base)**2, axis=-1)
        prior = gain[:, None] * long_template(np.arange(size) - NOMINAL) * np.exp(1j * cfo[:, None] * coordinate)
        y = np.where(~valid & (coordinate >= size), prior, y)
    return y


def readouts(iq, threshold=0.):
    z = iq[:, 0] + 1j * iq[:, 1]
    delta, margin, estimates = estimate_timing(z)
    shifted = realign(z, np.where(margin > 0, delta, 0))
    safe_delta = np.where(margin > threshold, delta, 0)
    clean_guard = realign(z, safe_delta)
    extrapolated = realign(z, safe_delta, protocol=True, cfo=estimates.coarse_rad_per_sample)
    return {name: np.stack((value.real, value.imag), axis=1).astype(iq.dtype, copy=False)
            for name, value in zip(READOUTS, (z, shifted, clean_guard, extrapolated))}, estimates
