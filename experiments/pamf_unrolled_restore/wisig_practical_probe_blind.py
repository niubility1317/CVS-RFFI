"""Receive-only causal channel estimates with a public Wi-Fi preamble prior."""

from dataclasses import dataclass

import numpy as np

from . import wisig_practical_received as receive
from . import wisig_timing as timing
from .wisig_joint_preamble import waveform


TIME = np.arange(receive.SAMPLES)
KERNEL_LENGTH = 64
PILOT_POINTS = np.arange(72, receive.SAMPLES)
ARMS = ("source_delay", "guard_shrink", "pilot_phase", "pilot_echo")


@dataclass(frozen=True)
class Prior:
    anchor: int
    guard: np.ndarray
    decoder: np.ndarray
    calibration: np.ndarray
    phase_mean: complex
    noise_knots: np.ndarray
    coherence: np.ndarray


def unit(x):
    return x / abs(x).clip(1e-12)


def pilot_sync(samples):
    blocks = samples[:, 100:200].reshape(-1, 5, timing.PERIOD)
    cross = np.sum(blocks[:, 1:] * blocks[:, :-1].conj(), axis=(1, 2))
    return samples * np.exp(-1j * np.angle(cross[:, None]) / timing.PERIOD * (TIME - 150))


def phase_statistics(noise, coefficients, prior):
    coherence = np.interp(np.log(noise), prior.noise_knots, prior.coherence)
    axis = unit(np.asarray(prior.phase_mean)).item()
    observed = unit(coefficients[:, 0]) * np.conj(axis)
    phase = coherence + (1. - coherence) * abs(prior.phase_mean) * observed
    return coherence, phase, axis


def fit(received):
    samples = np.asarray(received, dtype=np.complex128)
    guard = receive.guard_basis()
    noise = receive.guard_power(samples, guard)
    sync = pilot_sync(samples)
    bright = noise <= np.quantile(noise, .20)
    starts = np.arange(timing.NOMINAL, min(241, timing.NOMINAL + KERNEL_LENGTH))
    good = sync[bright]
    evidence = timing._energy_ratio(good, starts) + 2 * timing._long_match(good, starts)
    anchor = int(starts[np.bincount(evidence.argmax(axis=-1), minlength=len(starts)).argmax()] - timing.NOMINAL)
    offsets = np.arange(anchor, min(anchor + 32, KERNEL_LENGTH))
    dictionary = waveform(PILOT_POINTS[:, None] - timing.NOMINAL - offsets)
    gram = dictionary.conj().T @ dictionary
    track = np.linspace(.04, .16, len(offsets))
    gram.flat[::len(offsets) + 1] += track * np.diag(gram).real
    decoder = np.linalg.solve(gram, dictionary.conj().T)
    coefficients = sync[:, PILOT_POINTS] @ decoder.T
    phase = unit(coefficients[:, 0])
    phase_mean = complex(phase[bright].mean())
    phase_axis = unit(np.asarray(phase_mean)).item()
    ratios = coefficients[bright] / (coefficients[bright, :1] + 1e-12)
    calibration = np.median(ratios.real, axis=0) + 1j * np.median(ratios.imag, axis=0)
    group = np.array_split(np.argsort(noise), 8)
    knots = np.log([np.median(noise[k]) for k in group])
    moment = np.array([(phase[k] * np.conj(phase_axis)).real.mean() for k in group])
    coherence = np.minimum.accumulate((moment / max(abs(phase_mean), .05)).clip(0., 1.))
    return Prior(anchor, guard, decoder, calibration, phase_mean, knots, coherence)


def estimate(received, prior):
    """Predict fixed FIRs using one observed WiSig packet per row."""
    samples = np.asarray(received, dtype=np.complex128)
    if samples.ndim != 2 or samples.shape[1] != len(TIME) or not np.isfinite(samples).all():
        raise ValueError("Only finite post-channel 256-point receive IQ is accepted")
    noise = receive.guard_power(samples, prior.guard)
    amplitude = np.sqrt((np.mean(abs(samples)**2, axis=-1) - noise).clip(0., 1.5))
    coefficients = pilot_sync(samples)[:, PILOT_POINTS] @ prior.decoder.T
    phase_condition, corrected_phase, axis = phase_statistics(noise, coefficients, prior)
    kernels = np.zeros((len(ARMS), len(samples), KERNEL_LENGTH), np.complex128)
    kernels[0, :, prior.anchor] = 1.
    kernels[1, :, prior.anchor] = amplitude * phase_condition
    kernels[2:, :, prior.anchor] = amplitude * corrected_phase
    response = coefficients[:, 1:] - coefficients[:, :1] * prior.calibration[None, 1:]
    phase_scale = amplitude * np.conj(axis) / np.maximum(abs(coefficients[:, 0]), .25)
    echo = response * phase_scale[:, None]
    cap = np.minimum(1., .35 / np.linalg.norm(echo, axis=1).clip(1e-12))
    kernels[3, :, prior.anchor + 1:prior.anchor + len(prior.calibration)] = echo * cap[:, None]
    return kernels, noise
