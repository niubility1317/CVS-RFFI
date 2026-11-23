"""Per-packet L-STF engineering baselines for the fixed WiSig Main window."""

from dataclasses import dataclass

import numpy as np

from experiments.cvs_rff_physics.known_excitation import SIGNS, TONES


SAMPLE_RATE_HZ = 25_000_000
PERIOD = 20
STF = slice(80, 160)
FFT_BINS = (TONES // 4) % PERIOD
PILOTS = SIGNS * (1 + 1j) / np.sqrt(24)
RIDGE = 0.05
READOUTS = ('direct', 'coarse_cfo', 'fine_cfo', 'ridge', 'wiener', 'wiener_restore_cfo')
_TINY = np.finfo(np.float32).tiny


@dataclass(frozen=True)
class Estimate:
    coarse_rad_per_sample: np.ndarray
    fine_rad_per_sample: np.ndarray
    coarse_coherence: np.ndarray
    long_coherence: np.ndarray
    clock_delay_per_sample: np.ndarray
    clock_phase_rmse: np.ndarray
    response: np.ndarray
    effective_variance: np.ndarray
    signal_power: np.ndarray


def _lag(a, b):
    cross = (b * a.conj()).sum(axis=(1, 2))
    energy = (np.abs(a)**2).sum(axis=(1, 2)) * (np.abs(b)**2).sum(axis=(1, 2))
    confidence = np.abs(cross) / np.sqrt(np.maximum(energy, _TINY))
    return np.angle(cross), np.clip(confidence, 0, 1)


def _remove_cfo(iq, rad_per_sample):
    index = np.arange(iq.shape[-1], dtype=iq.real.dtype)
    return iq * np.exp(-1j * rad_per_sample[:, None] * index)


def _clock_slope(tones):
    # Compare the centers of cycles (0, 1) and (2, 3), separated by 40 samples.
    early = tones[:, :2].mean(axis=1)
    late = tones[:, 2:].mean(axis=1)
    cross = late * early.conj()
    common = np.angle(cross.sum(axis=1))
    phase = np.angle(cross * np.exp(-1j * common[:, None]))
    w = np.abs(cross)
    total = np.maximum(w.sum(axis=1), _TINY)
    k = TONES / 4.0
    k_centered = k - (w * k).sum(axis=1)[:, None] / total[:, None]
    phase_centered = phase - (w * phase).sum(axis=1)[:, None] / total[:, None]
    slope = (w * k_centered * phase_centered).sum(axis=1) / np.maximum(
        (w * k_centered**2).sum(axis=1), _TINY)
    rmse = np.sqrt((w * (phase_centered - slope[:, None] * k_centered)**2).sum(axis=1) / total)
    return -slope / (4 * np.pi), rmse


def _estimate(iq):
    symbols = iq[:, STF].reshape(-1, 4, PERIOD)
    coarse, short_coherence = _lag(symbols[:, :-1], symbols[:, 1:])
    coarse /= PERIOD
    long_phase, long_coherence = _lag(symbols[:, :1], symbols[:, -1:])
    long_lag = 3 * PERIOD
    fine = (long_phase + 2 * np.pi * np.rint((long_lag * coarse - long_phase) / (2 * np.pi))) / long_lag
    fine_iq = _remove_cfo(iq, fine)
    aligned = fine_iq[:, STF].reshape(-1, 4, PERIOD)
    tone_samples = (np.fft.fft(aligned, axis=-1) / PERIOD)[:, :, FFT_BINS]
    mean = tone_samples.mean(axis=1)
    clock, clock_rmse = _clock_slope(tone_samples)

    # Include inter-repeat drift, residual CFO and interference in the measured
    # residual power. CFO is estimated from the same packet, so AWGN bias is possible.
    difference = aligned[:, 1::2] - aligned[:, ::2]
    variance = (np.abs(difference)**2).mean(axis=(1, 2)) * 0.5
    received_power = (np.abs(aligned)**2).mean(axis=(1, 2))
    signal = np.maximum(received_power - variance, 0.0)
    estimate = Estimate(coarse, fine, short_coherence, long_coherence, clock, clock_rmse,
                        mean / PILOTS, variance, signal)
    return estimate, fine_iq


def _equalize(sync, estimate, *, measured_variance):
    response = estimate.response
    magnitude = np.abs(response)**2
    reference_power = magnitude.mean(axis=1, keepdims=True)
    regularizer = RIDGE * reference_power
    if measured_variance:
        # For a single normalized DFT, each tone has variance sigma² / N.
        regularizer = regularizer + estimate.effective_variance[:, None] / (PERIOD * np.abs(PILOTS)**2)
    # Preserve the average gain scale and leave unsounded frequencies untouched.
    weights = np.sqrt(reference_power) * response.conj() / np.maximum(magnitude + regularizer, _TINY)
    symbols = sync[:, STF].reshape(-1, 4, PERIOD)
    spectral = np.fft.fft(symbols, axis=-1)
    spectral[:, :, FFT_BINS] *= weights[:, None, :]
    equalized = np.fft.ifft(spectral, axis=-1).reshape(len(sync), -1)
    taper = np.ones(equalized.shape[-1], dtype=sync.real.dtype)
    taper[:PERIOD] = np.linspace(0.0, 1.0, PERIOD)
    taper[-PERIOD:] = taper[:PERIOD][::-1]
    confidence = np.minimum(estimate.coarse_coherence, estimate.long_coherence)
    result = sync.copy()
    result[:, STF] += (equalized - sync[:, STF]) * taper * confidence[:, None]
    return result


def readouts(iq):
    """Run frozen engineering controls with current normalized received IQ only."""
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[1:] != (2, 256) or iq.dtype not in (np.float32, np.float64):
        raise ValueError('Expected real floating B×2×256 WiSig IQ')
    if len(iq) == 0 or not np.isfinite(iq).all():
        raise ValueError('Expected a nonempty, finite WiSig IQ batch')
    z = iq[:, 0] + 1j * iq[:, 1]
    estimate, fine = _estimate(z)
    wiener = _equalize(fine, estimate, measured_variance=True)
    variants = (
        z,
        _remove_cfo(z, estimate.coarse_rad_per_sample),
        fine,
        _equalize(fine, estimate, measured_variance=False),
        wiener,
        _remove_cfo(wiener, -estimate.fine_rad_per_sample),
    )
    return {mode: np.stack((y.real, y.imag), axis=1).astype(iq.dtype, copy=False)
            for mode, y in zip(READOUTS, variants)}, estimate
