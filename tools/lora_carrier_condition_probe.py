"""Measure whether LoRa carrier context predicts genuine weak source sidebands."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.pamf_lora_main.band_allocation import band_masks, fft
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import chirp_windows
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
from experiments.pamf_lora_main.repeat_product import independent_noise


SEED = 2026112407
FIT_PER_RX = 4_000
HELD_PER_RX = 100
RIDGE = .05
BANDS = ((1, 8), (9, 32))
CONTEXTS = ('relative_amplitude', 'amplitude_frequency', 'amplitude_frequency_phase')
REGIMES = ('ground', 'independent_noise', 'generic_burst')
REFERENCES = ('observed_frame', 'tone_removed_observed_frame', 'ground_frame')


def carrier_features(tracker, tracked, phase, rms):
    # Use the tracked carrier before per-window scale normalization, then
    # remove packet-level gain and absolute carrier phase.
    amplitude = (tracked.mean(-1).abs() * rms.squeeze(-1)).clamp_min(1e-8).log()
    amplitude = amplitude - amplitude.mean(-1, keepdim=True)
    carrier = phase * tracker.reference.chirp.conj()
    radians = (carrier[..., 1:] * carrier[..., :-1].conj()).mean(-1).angle()
    frequency = radians * (256. / (2.*torch.pi))
    average = frequency.mean(-1, keepdim=True)
    relative_phase = carrier[:, 1:, 0] * carrier[:, :-1, 0].conj()
    relative_phase = relative_phase * torch.polar(torch.ones_like(radians[:, 1:]), -256. * radians[:, :-1])
    return torch.cat((amplitude, frequency - average, average, relative_phase.real, relative_phase.imag), -1)


def paired_context(tracker, clean, observed, mask):
    tracked, phase, _, rms = coordinates(tracker, real_iq(observed))
    ground, ground_phase, _, ground_rms = coordinates(tracker, real_iq(clean))
    ground_tone = ground.mean(-1, keepdim=True) * ground_phase * ground_rms
    observed_frame = phase * rms
    references = torch.stack((clean / observed_frame,
                              (clean - ground_tone) / observed_frame, ground), 1)
    reference = fft(references)[..., mask]
    return carrier_features(tracker, tracked, phase, rms), reference


def source_rows(data, raw_ids, tracker, regime, mask, *, seed, device):
    sampler = {'ground': None, 'independent_noise': independent_noise, 'generic_burst': burst_observation}[regime]
    rng = torch.Generator(device=device).manual_seed(seed)
    features, targets = [], []
    for indices in np.array_split(raw_ids, max(1, (len(raw_ids)+95)//96)):
        iq = np.stack([chirp_windows(data[index]) for index in indices])
        clean = complex_iq(torch.from_numpy(iq).to(device))
        observed = clean if sampler is None else sampler(clean, rng)
        inputs, target = paired_context(tracker, clean, observed, mask)
        features.append(inputs)
        targets.append(target)
    return torch.cat(features), torch.cat(targets)


def fit_conditional(inputs, targets):
    """Match the unconditional and conditional fit on the same source rows."""
    inputs, targets = inputs.double(), targets.to(torch.complex128)
    mean = inputs.mean(0)
    scale = inputs.std(0, unbiased=False).clamp_min(1e-4)
    normalized = ((inputs-mean)/scale).clamp(-5., 5.)
    intercept = targets.mean(0)
    target = torch.view_as_real(targets - intercept).flatten(1)
    clean_context = normalized.mean(0)
    normalized = normalized - clean_context
    covariance = normalized.T @ normalized / len(inputs)
    cross = normalized.T @ target / len(inputs)
    eye = torch.eye(covariance.shape[-1], device=inputs.device, dtype=inputs.dtype)
    coefficient = torch.linalg.solve(covariance + RIDGE*eye, cross)
    return mean, scale, clean_context, intercept, coefficient


def predict(fitted, inputs):
    mean, scale, center, unconditional, coefficient = fitted
    centered = ((inputs.double()-mean)/scale).clamp(-5., 5.) - center
    shape = (len(inputs), *unconditional.shape, 2)
    inferred = torch.view_as_complex((centered @ coefficient).reshape(shape).contiguous())
    return unconditional, inferred


def centered_components(value):
    common = value.mean(-2, keepdim=True)
    return common, value - common


def comparison(unconditional, estimate, reference, selectors):
    baseline = reference - unconditional
    masks = [slice(None), *selectors]
    result = []
    for full, direction in zip(centered_components(baseline), centered_components(estimate), strict=True):
        fraction = []
        for mask in masks:
            base = full[..., mask]
            delta = direction[..., mask]
            energy = base.abs().square().sum().clamp_min(1e-12)
            baseline_rmse = (base - delta).abs().square().sum()
            reversed_rmse = (base - delta.roll(1, 0)).abs().square().sum()
            fraction.append([float(1.-baseline_rmse/energy), float(1.-reversed_rmse/energy)])
        result.append(fraction)
    return result


@torch.no_grad()
def probe(source, device, n_fit=FIT_PER_RX, n_held=HELD_PER_RX):
    began = perf_counter()
    tracker = PacketRestorer('consensus').to(device).eval()
    contract = source_contract(source)
    raw, receivers, _, _ = source_role(source, contract, 'U_s')
    fit, _, held = partition(receivers, fit=n_fit, held=n_held)
    bins = band_masks(device)[:2]
    masks = tuple(bins[:, bins.any(0)])
    features = (slice(0, 6), slice(0, 13), slice(0, 23))
    rows = {}
    for regime_index, regime in enumerate(REGIMES):
        fit_arrays, held_arrays = [], []
        for i, (train, test) in enumerate(zip(np.split(fit,len(SOURCE)), np.split(held,len(SOURCE)), strict=True)):
            fit_arrays.append(source_rows(raw, train, tracker, regime, bins.any(0),
                                          seed=SEED+300*regime_index+i, device=device))
            held_arrays.append(source_rows(raw, test, tracker, regime, bins.any(0),
                                           seed=SEED+300*regime_index+100+i, device=device))
        results = {}
        for leave, (receiver, (inputs, reference)) in enumerate(zip(SOURCE, held_arrays, strict=True)):
            train_inputs, train_targets = (torch.cat(sequence) for sequence in
                zip(*(value for i,value in enumerate(fit_arrays) if i!=leave), strict=True))
            estimates = {}
            for name, feature_set in zip(CONTEXTS, features, strict=True):
                fitted = fit_conditional(train_inputs[:, feature_set], train_targets)
                unconditional, estimated = predict(fitted, inputs[:, feature_set])
                estimates[name] = {
                    basis: comparison(unconditional[i], estimated[:, i], reference[:, i], masks)
                    for i, basis in enumerate(REFERENCES)
                }
            results[receiver] = estimates
        rows[regime] = results
    return dict(status='SOURCE_U_ONLY', training_receivers_per_fold=len(SOURCE)-1,
                total_fit_packets=len(fit), total_held_packets=len(held), by_regime=rows,
                bands=['1–32 kHz', *(f'{start}–{end} kHz' for start,end in BANDS)],
                groups=['common', 'private'], fields=['same_packet_improvement', 'rolled_context_improvement'],
                ridge=RIDGE, seed=SEED, device=str(device), elapsed_seconds=perf_counter()-began,
                F_reads=0, TX_reads=0, V_reads=0, target_reads=0, exact_reads=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = probe(args.source, args.device)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(args.output)


if __name__ == '__main__':
    main()
