"""Isolate the effect of transporting raw-IQ channel action through LoRa dechirping."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import tensor, write
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_apertures import ApertureTrackers
from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.dura_w24 import NARROW, WIDE
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.receive_frame_action import DELAY_US, estimate, invert_half, opposite_fold
from experiments.pamf_lora_main.relative_echo import shift


CASES = ('clean', 'relative_weak', 'relative_strong', 'shared_weak', 'shared_strong')
ARMS = ('unit', 'dechirped', 'raw_iq_frame')
SEED = 2026112592


def perturb(raw, case):
    if case == 'clean':
        return raw, raw
    rng = torch.Generator().manual_seed(SEED)
    b = len(raw)
    rows = raw.unflatten(-1, (8, 1024))
    strength = .12 if case.endswith('weak') else .28
    gain, image, echo = (strength*torch.randn((b, 3, 8, 1), dtype=raw.dtype, generator=rng)).unbind(1)
    if case.startswith('shared'):
        gain, image, echo = (x[:, :1] for x in (gain, image, echo))
    delayed = shift(raw, DELAY_US).unflatten(-1, (8, 1024))
    signal = ((1.+gain)*rows + image*rows.conj() + echo*delayed).flatten(-2)
    noise_ratio = .035 if case.endswith('weak') else .12
    noise = torch.randn(raw.shape, dtype=raw.dtype, generator=rng)
    rms = raw.abs().square().mean(-1, keepdim=True).sqrt()
    return signal, signal+noise_ratio*rms*noise


def fidelity(x, clean, rms):
    absolute = ((x-clean)*rms).abs().square().mean((1, 2))/(clean*rms).abs().square().mean((1, 2)).clamp_min(1e-9)
    freq = (x.shape[-1]*torch.fft.fftfreq(x.shape[-1])).abs()
    select = (freq >= 9) & (freq <= 32)
    a, b = (torch.fft.fft(z-z.mean(1, keepdim=True), norm='ortho')[..., select] for z in (x, clean))
    error = (a-b).abs().square().sum((1, 2))/b.abs().square().sum((1, 2)).clamp_min(1e-9)
    return absolute, error


def evaluate(received, tracker, clean, signal):
    y, phase, _, rms = coordinates(tracker, real_iq(received))
    ref, folds, actual = opposite_fold(y)
    fitted = [estimate(y, phase, ref, y, folds, relative=True, raw_frame=flag) for flag in (False, True)]
    recovered = [invert_half(y, phase, coeff, raw_frame=flag) for (coeff, _), flag in zip(fitted, (False, True), strict=True)]

    clean = clean/(phase*rms)
    signal = signal/(phase*rms)
    indices = torch.cat(folds)
    oracle = (clean-signal)[..., indices].abs().square().mean((1, 2))
    ceiling = [estimate(y, phase, clean, signal, folds, relative=False, raw_frame=flag)[1] for flag in (False, True)]
    outputs = ((actual, *(m for _, m in fitted)), (oracle, *ceiling), *zip(
        *(fidelity(x, clean, rms) for x in (y, *recovered)), strict=True))
    return torch.stack([torch.stack(fields, -1) for fields in outputs], 1)


def evidence(unit, held):
    return np.maximum(0., (unit-held)/np.maximum(unit, 1e-9))


def view(tracker, raw, samples):
    wave = tracker.aperture(raw)[..., :samples]
    return wave if samples == NARROW else wave[:, 1:7]


def calibration_scores(raw, tracker, samples):
    values = []
    for chunk in raw.split(32):
        y, phase, *_ = coordinates(tracker, real_iq(view(tracker, chunk, samples)))
        ref, folds, unit = opposite_fold(y)
        _, held = estimate(y, phase, ref, y, folds, relative=True, raw_frame=True)
        values.append(torch.stack((unit, held), -1))
    values = np.asarray(torch.cat(values).tolist())
    return evidence(values[:, 0], values[:, 1])


def gate_reduction(rows, cutoff, left_out):
    score = evidence(rows[:, 0, 0], rows[:, 0, 2])
    mask = score > cutoff
    controls = dict(active=int(mask.sum()), leave_rx_out_active=int((score > left_out).sum()))
    for index, key in ((2, 'recovery_nmse'), (3, 'private_9_32_nmse')):
        direct, _, corrected = rows[:, index].T
        values = np.where(mask, corrected, direct)
        controls[key] = dict(median=float(np.median(values)), improved=int((values<direct).sum()),
                             harmed=int((values>direct).sum()))
    return controls


def reduce(rows):
    result = {}
    for field, metric in zip(('blind_forward', 'known_clean_forward', 'recovery_nmse', 'private_9_32_nmse'),
                             rows.transpose(1, 0, 2), strict=True):
        result[field] = dict(median=np.median(metric, axis=0).tolist(),
                             raw_iq_vs_dechirped_median_change_db=float(np.median(
                                 10*np.log10((metric[:, 2]+1e-9)/(metric[:, 1]+1e-9)))),
                             improved=int((metric[:, 2]<metric[:, 1]).sum()))
    return result


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    _, calibration, held = partition(rx, fit=4000)
    value = tensor(iq[held].view('float32')).view(torch.complex64)
    trackers = ApertureTrackers().eval()
    interfaces = ((NARROW, trackers.narrow), (WIDE, trackers.wide))
    raw_calibration = tensor(iq[calibration].view('float32')).view(torch.complex64)
    calibration_evidence = np.stack([calibration_scores(raw_calibration, tracker, samples)
                                     for samples, tracker in interfaces], 1)
    threshold = 1.05 * float(calibration_evidence.max())
    leave_rx_out = {rx_name: 1.05 * float(calibration_evidence[rx[calibration]!=rx_name].max())
                    for rx_name in SOURCE}
    result = {}
    for case in CASES:
        unnoised, observed = perturb(value, case)
        result[case] = {}
        for samples, tracker in interfaces:
            panels = []
            for clean, signal, received in zip(value.split(16), unnoised.split(16), observed.split(16), strict=True):
                waves = [view(tracker, x, samples) for x in (received, clean, signal)]
                panels.append(evaluate(waves[0], tracker, waves[1], waves[2]))
            samples_rows = np.asarray(torch.cat(panels).tolist())
            if not np.isfinite(samples_rows).all():
                raise FloatingPointError('Nonfinite source action diagnostic')
            result[case][str(samples)] = dict(all=reduce(samples_rows),
                                              by_rx={name: reduce(samples_rows[rx[held]==name]) for name in SOURCE},
                                              gate=gate_reduction(samples_rows, threshold, threshold),
                                              gate_by_rx={name: gate_reduction(samples_rows[rx[held]==name],
                                                                               threshold, leave_rx_out[name])
                                                          for name in SOURCE})
    return dict(status='VERIFIED', role='U_s', held=len(held), arms=ARMS, seed=SEED,
                delay_us=DELAY_US, artifact_scope='source U perturbation diagnostic',
                calibration=len(calibration), threshold=threshold, leave_rx_out_thresholds=leave_rx_out,
                candidate_inputs=['current received IQ'], known_clean_control='isolated after receive-only output',
                target_reads=0, F_reads=0, V_reads=0, TX_reads=0, satellite_trace_reads=0, results=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
