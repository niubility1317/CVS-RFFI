"""Measure the shared W24/256 kS/s content on held source U packets."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import tensor, write
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.dura_w24 import NARROW, WIDE, apertures, lift
from experiments.pamf_lora_main.dura_w24_model import WideTracker
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates


BANDS = ((1, 8), (9, 32), (33, 63), (72, 112), (129, 192))
LEVELS = (0., .12, .32)


def quantiles(value):
    return np.quantile(value, [.1, .5, .9]).tolist()


def guard(spectrum, band):
    return (spectrum[..., band].abs().square().median(-1).values / .69314718056).sqrt().median(1).values


def shared_guard_coefficients(value):
    n = value.shape[-1]
    window = torch.hann_window(n, periodic=False)
    density = torch.fft.fft(value*window, norm='ortho') * (WIDE/(n*window.square().mean())).sqrt()
    frequency = (n*torch.fft.fftfreq(n)).abs()
    return density[..., (frequency >= 72) & (frequency <= 112)]


def shared_guard(value):
    coefficients = shared_guard_coefficients(value)
    return (coefficients.abs().square().median(-1).values / .69314718056).sqrt().median(1).values


def compare(narrow, wide, tracker, narrow_tracker):
    narrow, wide = tensor(narrow[:, 1:7]), tensor(wide[:, 1:7, :, :WIDE])
    lifted = lift(narrow)
    reference, phase, _, scale = coordinates(tracker, wide)
    projected = complex_iq(lifted) / (phase * scale)
    tracked, *_ = coordinates(tracker, lifted)
    # Use each input's own receive-derived tracker for the guard statistic.
    spec, limited = (torch.fft.fft(x, norm='ortho') for x in (reference, tracked))
    frequency = (WIDE * torch.fft.fftfreq(WIDE)).abs()
    scores = torch.stack([guard(x, select) for select in (frequency > 64, (frequency >= 72) & (frequency <= 112))
                          for x in (spec, limited)], dim=1)

    # Attribute the spectrum in one common frame without a second alignment.
    full, probe = (torch.fft.fft(x-x.mean(1, keepdim=True), norm='ortho') for x in (reference, projected))
    power = full.abs().square()
    distortion = (probe-full).abs().square()
    errors, fractions = [], []
    total_power = power.sum((1, 2)).clamp_min(1e-12)
    for lo, hi in BANDS:
        band = (frequency >= lo) & (frequency <= hi)
        energy = power[..., band].sum((1, 2)).clamp_min(1e-12)
        errors.append(distortion[..., band].sum((1, 2)) / energy)
        fractions.append(energy / total_power)
    raw = complex_iq(wide)
    raw_power = raw.abs().square().sum((1, 2))
    raw_error = (complex_iq(lifted)-raw).abs().square().sum((1, 2)) / raw_power
    undisclosed = torch.fft.fft(raw, norm='ortho')[..., frequency > 128].abs().square().sum((1, 2)) / raw_power
    narrow_on_clock = coordinates(narrow_tracker, narrow)[0]
    without_lift = torch.stack((shared_guard(reference), shared_guard(narrow_on_clock)), 1)
    matrix = torch.cat((scores, torch.stack(errors, 1), torch.stack(fractions, 1),
                        raw_error[:, None], undisclosed[:, None], without_lift), 1)
    return np.asarray(matrix.tolist(), dtype=np.float32)


def summarize(samples):
    guard_db = 20 * np.log10(np.maximum(samples[:, :4], 1e-12))
    tapered_db = 20 * np.log10(np.maximum(samples[:, -2:], 1e-12))
    return dict(n=len(samples),
                native_guard_rms=quantiles(samples[:, 0]),
                native_common_guard_rms=quantiles(samples[:, 2]),
                legacy_guard_bias_db=quantiles(guard_db[:, 1]-guard_db[:, 0]),
                common_guard_bias_db=quantiles(guard_db[:, 3]-guard_db[:, 2]),
                native_tapered_common_guard_rms=quantiles(samples[:, -2]),
                tapered_common_direct_clock_bias_db=quantiles(tapered_db[:, 1]-tapered_db[:, 0]),
                received_iq_lift_nmse=quantiles(samples[:, -4]),
                received_energy_above_128_fraction=quantiles(samples[:, -3]),
                private_nmse={f'{lo}-{hi}': quantiles(samples[:, 4+i]) for i, (lo, hi) in enumerate(BANDS)},
                private_energy_fraction={f'{lo}-{hi}': quantiles(samples[:, 4+len(BANDS)+i])
                                         for i, (lo, hi) in enumerate(BANDS)})


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    native, rx, _, _ = source_role(source, contract, 'U_s')
    _, _, held = partition(rx, fit=4000)
    iq = np.asarray(native[held])
    selection = {'all': np.ones(len(iq), dtype=bool), **{name: rx[held] == name for name in SOURCE}}
    tracker = WideTracker().eval()
    narrow_tracker = PacketRestorer('consensus', samples=NARROW).eval()
    rng = np.random.default_rng(2026112582)
    noise = rng.standard_normal(iq.shape, dtype='float32') + 1j*rng.standard_normal(iq.shape, dtype='float32')
    scale = (np.mean(np.abs(iq)**2, axis=1, keepdims=True) / np.mean(np.abs(noise)**2, axis=1, keepdims=True))**.5
    noise *= scale
    result = {}
    for level in LEVELS:
        observed = iq + np.float32(level)*noise
        samples = np.concatenate([compare(*apertures(observed[i:i+32]), tracker, narrow_tracker)
                                  for i in range(0, len(iq), 32)])
        if not np.isfinite(samples).all():
            raise ValueError('Nonfinite W24 observation measurement')
        result[str(level)] = {name: summarize(samples[mask]) for name, mask in selection.items()}
    return dict(status='VERIFIED', held=len(iq), protocol=contract['protocol'], role='U_s',
                target_reads=0, trace_reads=0, F_reads=0, TX_reads=0, V_reads=0,
                quantiles=[.1, .5, .9], native_guard_khz='>64', common_guard_khz=[72, 112], results=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
