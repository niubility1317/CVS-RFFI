"""Audit one LoRa gate on 256/384 kS/s with clean U calibration and held U."""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import tensor, write
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_apertures import shared_guard_coefficients
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.dura_w24 import CROP, NARROW, WIDE
from experiments.pamf_lora_main.dura_w24_model import WideTracker
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from tools.lora_w24_observation_overlap import quantiles


RATES = (NARROW, WIDE)
METHODS = ('direct', 'remove_repeat_mean')
LEVELS = (.04, .12, .32)
FAMILIES = ('white', 'colored', 'intermittent')
SEED = 2026112586


def extract(packet, rate):
    boundary = rate // 8
    return resample_poly(packet[..., CROP[1:7]], boundary, 125, axis=-1)[..., boundary:boundary+rate]


def complex_tensor(samples):
    return torch.complex(tensor(samples.real), tensor(samples.imag))


def repeat_components(coefficients):
    chirps = coefficients.shape[1]
    private = (coefficients - coefficients.mean(1, keepdim=True)) * (chirps/(chirps-1))**.5
    return coefficients, private


def measure_views(packet, trackers, disturbance=None):
    estimates, references = [], []
    for rate in RATES:
        observed = complex_tensor(extract(packet, rate))
        y, phase, _, rms = coordinates(trackers[rate], real_iq(observed))
        observed_bands = repeat_components(shared_guard_coefficients(y))
        estimates.append([(x.abs().square().median(-1).values/.69314718056).sqrt().median(1).values.tolist()
                          for x in observed_bands])
        if disturbance is not None:
            added = complex_tensor(extract(disturbance, rate)) / (phase*rms)
            reference_bands = repeat_components(shared_guard_coefficients(added))
            references.append([x.abs().square().mean(-1).sqrt().median(1).values.tolist()
                               for x in reference_bands])
    return np.asarray(estimates).T, np.asarray(references).T if references else None


def generate_noise(iq, kind):
    rng = np.random.default_rng(SEED)
    white = rng.standard_normal(iq.shape, dtype='float32') + 1j*rng.standard_normal(iq.shape, dtype='float32')
    if kind == 'colored':
        noise = white + .85 * np.roll(white, 1, axis=-1)
    elif kind == 'intermittent':
        pulses = rng.random((len(iq), iq.shape[-1]//16)) < .07
        noise = white * (.25 + 4.5*np.repeat(pulses, 16, axis=-1))
    else:
        noise = white
    scale = (np.mean(np.abs(iq)**2, axis=-1, keepdims=True) / np.mean(np.abs(noise)**2, axis=-1, keepdims=True))**.5
    return np.asarray(noise*scale, dtype=np.complex64)


def panel(iq, trackers, perturbation=None):
    results, reference = [], []
    for start in range(0, len(iq), 32):
        raw = iq[start:start+32]
        noise = None if perturbation is None else perturbation[start:start+32]
        estimated, exact = measure_views(raw if noise is None else raw+noise, trackers, noise)
        results.append(estimated)
        if exact is not None:
            reference.append(exact)
    return np.concatenate(results), np.concatenate(reference) if reference else None


def summarize(scores, threshold, reference):
    gate = scores > threshold
    db = 20 * np.log10(np.maximum(scores, 1e-12))
    result = dict(n=len(scores), active=gate.sum(0).tolist(), disagree=int((gate[:, 0] != gate[:, 1]).sum()),
                  score_rms=[quantiles(x) for x in scores.T], shared_guard_narrow_wide_db=quantiles(db[:, 0]-db[:, 1]))
    if reference is not None:
        exact_db = 20*np.log10(np.maximum(reference, 1e-12))
        result['estimate_vs_added_rms_db'] = [quantiles(x) for x in (db-exact_db).T]
        result['added_narrow_wide_db'] = quantiles(exact_db[:, 0]-exact_db[:, 1])
    return result


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    native, rx, _, _ = source_role(source, contract, 'U_s')
    _, calibration, held = partition(rx, fit=4000)
    trackers = {NARROW: PacketRestorer('consensus', samples=NARROW).eval(), WIDE: WideTracker().eval()}
    calibration_scores, _ = panel(native[calibration], trackers)
    threshold = 1.05 * calibration_scores.max((0, 2))
    leave_out = {name: 1.05 * calibration_scores[rx[calibration] != name].max((0, 2)) for name in SOURCE}
    iq = native[held]
    held_rx = rx[held]
    result = {}

    def record(name, scores, reference):
        if not np.isfinite(scores).all() or (reference is not None and not np.isfinite(reference).all()):
            raise FloatingPointError('Nonfinite source gate diagnostic')
        results = {}
        for index, method in enumerate(METHODS):
            exact = None if reference is None else reference[:, index]
            chosen = scores[:, index]
            receivers = {}
            for receiver in SOURCE:
                mask = held_rx == receiver
                ref = None if exact is None else exact[mask]
                receivers[receiver] = dict(pooled=summarize(chosen[mask], threshold[index], ref),
                                           leave_receiver_out=summarize(chosen[mask], leave_out[receiver][index], ref))
            results[method] = dict(all=summarize(chosen, threshold[index], exact), by_receiver=receivers)
        result[name] = results

    clean, _ = panel(iq, trackers)
    record('clean', clean, None)
    for family in FAMILIES:
        noise = generate_noise(iq, family)
        for level in LEVELS:
            scores, reference = panel(iq, trackers, np.float32(level)*noise)
            record(f'{family}_{level:.2f}', scores, reference)
    return dict(status='VERIFIED', role='U_s', protocol=contract['protocol'], rates_ksps=RATES, methods=METHODS,
                calibration=len(calibration), held=len(held), threshold=threshold.tolist(),
                leave_receiver_out_thresholds={name: value.tolist() for name, value in leave_out.items()},
                calibration_max_by_rate=calibration_scores.max(0).tolist(), results=result,
                exact_added_noise='isolated measurement after receive-only gate evaluation',
                F_reads=0, TX_reads=0, V_reads=0, target_reads=0, satellite_trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
