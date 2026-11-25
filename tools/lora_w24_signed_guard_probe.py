"""Stress LoRa receive-only gate statistics with one-sided generic IQ perturbations."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_apertures import shared_guard_coefficients
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from tools.lora_w24_guard_attribution import rms_floor
from tools.lora_w24_shared_guard_probe import RATES, complex_tensor, extract, generate_noise


SCORES = ('direct', 'balanced_sides', 'stronger_side', 'signed_minor_modes', 'dual_or_signed_minor')
LEVELS = (.04, .08, .16)
SEED = 2026112588


def signed_noise(iq, sign, *, repeated):
    packet = iq.reshape(len(iq), 8, -1)
    n = packet.shape[-1]
    frequency = np.fft.fftfreq(n, d=1e-6) * sign
    select = (frequency >= 82e3) & (frequency <= 104e3)
    spectrum = np.zeros((len(iq), 1 if repeated else 8, n), dtype=np.complex64)
    rng = np.random.default_rng(SEED)
    shape = (*spectrum.shape[:-1], int(select.sum()))
    spectrum[..., select] = rng.standard_normal(shape, dtype=np.float32) + 1j*rng.standard_normal(shape, dtype=np.float32)
    disturbance = packet * np.fft.ifft(spectrum, axis=-1).astype(np.complex64)
    noise_rms = np.maximum(np.mean(np.abs(disturbance)**2, axis=(1, 2), keepdims=True), 1e-20)**.5
    signal_rms = np.mean(np.abs(packet)**2, axis=(1, 2), keepdims=True)**.5
    return np.asarray(disturbance * (signal_rms/noise_rms), dtype=np.complex64).reshape(iq.shape)


def guard_scores(coefficients):
    sides = coefficients.chunk(2, dim=-1)
    power = coefficients.abs().square()
    left, right = (rms_floor(s.abs().square()) for s in sides)
    minor = []
    for side in sides:
        gram = side @ side.mH / side.shape[-1]
        modes = torch.linalg.eigvalsh(gram).clamp_min(0)
        minor.append(modes[:, :modes.shape[-1]//2].mean(1).sqrt())
    return torch.stack((rms_floor(power).median(1).values,
                        torch.minimum(left, right).median(1).values,
                        torch.maximum(left, right).median(1).values,
                        torch.maximum(*minor)), dim=1)


def panel(iq, trackers, disturbance=None, *, sign=0):
    rows, intended = [], []
    for start in range(0, len(iq), 32):
        raw = iq[start:start+32]
        added = None if disturbance is None else disturbance[start:start+32]
        received = raw if added is None else raw+added
        measures, shares = [], []
        for rate, tracker in trackers.items():
            real = real_iq(complex_tensor(extract(received, rate)))
            aligned, phase, _, rms = coordinates(tracker, real)
            measures.append(guard_scores(shared_guard_coefficients(aligned)))
            if sign:
                # Only the diagnostic knows the injected component. The gate above
                # has already been computed exclusively from the mixed receive IQ.
                change = complex_tensor(extract(added, rate)) / (phase*rms)
                sides = shared_guard_coefficients(change).abs().square().chunk(2, dim=-1)
                total = [s.sum((1, 2)) for s in sides]
                intended_index = 0 if sign > 0 else 1
                shares.append(total[intended_index] / (total[0]+total[1]).clamp_min(1e-12))
        rows.extend(torch.stack(measures, dim=-1).tolist())
        if shares:
            intended.extend(torch.stack(shares, dim=-1).tolist())
    return np.asarray(rows), np.asarray(intended)


def active(scores, thresholds):
    flags = scores > thresholds[None, :, None]
    return np.concatenate((flags, (flags[:, 1:2] | flags[:, 3:4])), axis=1)


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    _, calibration, held = partition(rx, fit=4000, held=300)
    held = held.reshape(len(SOURCE), 300)[:, 200:].ravel()
    trackers = {rate: PacketRestorer('consensus', samples=rate).eval() for rate in RATES}
    cal, _ = panel(iq[calibration], trackers)
    pooled = 1.05 * cal.max((0, 2))
    leave = {name: 1.05*cal[rx[calibration] != name].max((0, 2)) for name in SOURCE}
    raw = iq[held]
    by_rx = {name: rx[held] == name for name in SOURCE}

    def summarize(scores, fraction):
        if not np.isfinite(scores).all() or not np.isfinite(fraction).all():
            raise FloatingPointError('Nonfinite LoRa signed guard observation')
        hit = active(scores, pooled)
        result = dict(pooled=hit.sum(0).tolist(), rate_disagreement=(hit[..., 0] != hit[..., 1]).sum(0).tolist(),
                      leave_rx={name: active(scores[mask], leave[name]).sum(0).tolist()
                                for name, mask in by_rx.items()})
        if len(fraction):
            result['added_guard_energy_on_intended_side'] = np.quantile(fraction, [.1, .5, .9], axis=0).tolist()
        return result

    results = {'clean': summarize(*panel(raw, trackers))}
    white = generate_noise(raw, 'white', seed=SEED)
    results['white_0.04'] = summarize(*panel(raw, trackers, .04*white))
    for repeated in (False, True):
        family = 'repeat' if repeated else 'independent'
        for sign, side in ((1, 'positive'), (-1, 'negative')):
            disturbance = signed_noise(raw, sign, repeated=repeated)
            for rms in LEVELS:
                results[f'{family}_{side}_{rms:.2f}'] = summarize(*panel(raw, trackers, rms*disturbance, sign=sign))
    return dict(status='VERIFIED', protocol=contract['protocol'], source_role='U_s', scores=SCORES,
                rates=RATES, held_offset_per_rx=4700, held=len(raw), calibration=len(calibration),
                seed=SEED, pooled_thresholds=pooled.tolist(), results=results,
                noise_known_only_to_diagnostic=True, TX_reads=0, F_reads=0, V_reads=0, target_reads=0,
                trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
