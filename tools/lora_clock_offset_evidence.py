"""Measure sample-clock observability on the six registered LoRa upchirps."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates

from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import BANDWIDTH_HZ, FS_HZ, SOURCE, WINDOW_START
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.packet_data import CHIRPS, PACKET_LEN
from experiments.pamf_lora_main.packet_physics import NATIVE_RATE, chirp_samples


NATIVE_CHIRP = 1024
CORE = slice(32, 224)
LAG = 48
DOSES = (0, -40, 40, -120, 120, -400, 400)
PREFIXES = (3, 4, 6)
SEED = 2026112498
CENTER = (PACKET_LEN - 1) / 2


def warp(packet, ppm):
    index = np.arange(len(packet), dtype=np.float64)
    positions = index + (index - CENTER) * (ppm * 1e-6)
    return map_coordinates(packet, positions[None], order=3, mode='nearest')


def windows(packets, ppm):
    samples = np.stack([chirp_samples(warp(packet, ppm) if ppm else packet) for packet in packets])
    scale = np.sqrt(np.mean(abs(samples)**2, axis=(1, 2), keepdims=True))
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError('Expected nonzero finite source LoRa packets')
    return samples / scale


def apparent_clock_ppm(samples):
    first, second = np.triu_indices(samples.shape[1], k=2)
    gap = np.asarray(CHIRPS)[second] - np.asarray(CHIRPS)[first]
    cross = samples[:, second, CORE] * samples[:, first, CORE].conj()
    autocorrelation = np.sum(cross[..., LAG:] * cross[..., :-LAG].conj(), axis=-1)
    slope_hz = np.angle(autocorrelation) * (FS_HZ / (2 * np.pi * LAG))
    return np.median(1e6 * slope_hz / (BANDWIDTH_HZ * gap), axis=-1)


def correct_core(samples, ppm):
    # Evaluate only the original coordinates whose inverse remains in the crop.
    nominal = np.arange(samples.shape[-1], dtype=np.float64)[CORE]
    starts = np.asarray(CHIRPS) * NATIVE_CHIRP + WINDOW_START
    center = (CENTER - starts) * FS_HZ / NATIVE_RATE
    fractional_rate = np.clip(ppm * 1e-6, -.001, .001)[:, None, None]
    positions = nominal + (center[:, None] - nominal) * fractional_rate / (1 + fractional_rate)
    left = np.floor(positions).astype(int)
    t = positions - left
    weights = (-t*(t-1)*(t-2)/6, (t+1)*(t-1)*(t-2)/2,
               -(t+1)*t*(t-2)/2, (t+1)*t*(t-1)/6)
    return sum(weight * np.take_along_axis(samples, left+offset, axis=-1)
               for offset, weight in zip((-1, 0, 1, 2), weights, strict=True))


def nmse(samples, reference):
    return np.sum(abs(samples - reference)**2, axis=(1, 2)) / np.sum(abs(reference)**2, axis=(1, 2))


def analytic_check():
    length = NATIVE_CHIRP / NATIVE_RATE
    time = (np.arange(PACKET_LEN) % NATIVE_CHIRP) / NATIVE_RATE
    chirp = np.exp(2j*np.pi*BANDWIDTH_HZ*(-time/2 + time**2/(2*length)))
    base = windows([chirp], 0)[:, :, CORE]
    error = []
    for ppm in DOSES:
        sample = windows([chirp], ppm)
        estimate = apparent_clock_ppm(sample)
        error.append(float(abs(estimate[0] - ppm)))
        if ppm and not np.all(nmse(correct_core(sample, estimate), base) < nmse(sample[:, :, CORE], base)):
            raise ValueError('Clock correction did not reduce the ideal chirp error')
    if max(error) >= 1.:
        raise ValueError('Known LoRa chirp slope differs from the expected clock stress')
    return max(error)


def measure(source):
    started = perf_counter()
    ideal_error = analytic_check()
    packets, rx, _, _ = source_role(source, source_contract(source), 'U_s')
    _, _, held = partition(rx, fit=4000)
    rng = np.random.default_rng(SEED)
    wave = {dose: windows(packets[held], dose) for dose in DOSES}
    reference = wave[0][:, :, CORE]
    jitter = (rng.standard_normal(wave[0].shape) + 1j*rng.standard_normal(wave[0].shape)) / np.sqrt(2)
    local_time = (np.arange(wave[0].shape[-1]) - 127.5) / FS_HZ
    offsets = rng.normal(scale=20., size=(*wave[0].shape[:2], 1))
    local_phase = np.exp(2j*np.pi*offsets*local_time)
    conditions = {
        'clock_only': (1., 0.),
        'noise_rms_0.08': (1., .08*jitter),
        'noise_rms_0.08_and_chirp_offsets_rms_20_hz': (local_phase, .08*jitter),
    }
    groups = {name: rx[held] == name for name in SOURCE}
    groups['all'] = np.ones(len(held), dtype=bool)
    result = {}
    for name, (phase, noise) in conditions.items():
        nominal = apparent_clock_ppm(wave[0] * phase + noise)
        result[name] = {}
        for ppm, values in wave.items():
            observed = values * phase + noise
            prefix_estimates = {n: apparent_clock_ppm(observed[:, :n]) for n in PREFIXES}
            estimate = prefix_estimates[len(CHIRPS)]
            errors = [nmse(data, reference) for data in (observed[:, :, CORE], correct_core(observed, estimate),
                                                         correct_core(observed, np.full(len(held), ppm)))]
            result[name][str(ppm)] = {receiver: dict(
                n=int(select.sum()), apparent_ppm_median=float(np.median(estimate[select])),
                error_relative_to_injected_ppm={n: np.quantile(abs(prediction[select]-ppm), (.5, .9)).tolist()
                                               for n, prediction in prefix_estimates.items()},
                increment_error_ppm=np.quantile(abs(estimate[select]-nominal[select]-ppm), (.5, .9)).tolist(),
                nmse_median=[float(np.median(error[select])) for error in errors],
                packets_improved=int((errors[1][select] < errors[0][select]).sum())) for receiver, select in groups.items()}
    return dict(status='VERIFIED', input_role='U_s', held=len(held), fitted_parameters=0, TX_reads=0, F_reads=0,
                V_reads=0, target_reads=0, trace_reads=0, analytic_max_error_ppm=ideal_error,
                nmse_order=('direct', 'receive_estimate', 'known_injected_clock_diagnostic'),
                variant='six_upchirps_cross_product_phase_slope', seconds=perf_counter()-started, metrics=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    result = measure(args.source) if args.source is not None else dict(analytic_max_error_ppm=analytic_check())
    print(json.dumps(result, allow_nan=False), flush=True)
