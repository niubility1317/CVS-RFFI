"""Attribute held-receiver clean activations using only LoRa source U IQ."""

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
from tools.lora_w24_shared_guard_probe import FAMILIES, LEVELS, RATES, complex_tensor, extract, generate_noise


SCORES = ('direct', 'power_q20', 'balanced_sides', 'all_four_subbands', 'minor_chirp_modes')
PROFILE = ('coherent_power_share', 'leading_chirp_mode_share', 'positive_to_negative_db')


def rms_floor(power):
    return (power.median(-1).values / np.log(2)).sqrt()


def metrics(coefficients):
    chirps = coefficients.shape[1]
    power = coefficients.abs().square()
    positive, negative = power.chunk(2, dim=-1)
    # Both signs start their inner subband at 72 kHz. The existing FFT mask
    # can omit ±112 at 384 kS/s because of the float32 reference grid.
    inner_pos, outer_pos = positive[..., :20], positive[..., 20:]
    outer_neg, inner_neg = negative[..., :-20], negative[..., -20:]
    energy = power.mean((1, 2)).clamp_min(1e-12)
    gram = coefficients @ coefficients.mH / coefficients.shape[-1]
    modes = torch.linalg.eigvalsh(gram).clamp_min(0)
    scores = torch.stack((
        rms_floor(power).median(1).values,
        (torch.quantile(power, .2, dim=-1) / -np.log(.8)).sqrt().median(1).values,
        torch.minimum(rms_floor(positive), rms_floor(negative)).median(1).values,
        torch.stack([rms_floor(band) for band in (inner_pos, outer_pos, inner_neg, outer_neg)]).amin(0).median(1).values,
        modes[:, :chirps//2].mean(1).sqrt(),
    ), dim=1)
    profile = torch.stack((
        coefficients.mean(1).abs().square().mean(1) / energy,
        modes[:, -1] / (chirps * energy),
        10 * (positive.mean((1, 2)).clamp_min(1e-12) / negative.mean((1, 2)).clamp_min(1e-12)).log10(),
    ), dim=1)
    return torch.cat((scores, profile), dim=1)


def panel(iq, trackers):
    batches = []
    for batch in np.array_split(iq, (len(iq)+31)//32):
        rows = []
        for rate, tracker in trackers.items():
            received = real_iq(complex_tensor(extract(batch, rate)))
            coefficients = shared_guard_coefficients(coordinates(tracker, received)[0])
            rows.append(metrics(coefficients))
        batches.append(torch.stack(rows, dim=-1).tolist())
    return np.concatenate(batches)


def overlap(quiet, disturbed):
    """Fraction of quiet/disturbed pairs where the disturbed score is higher, with half credit for ties."""
    ordered = np.sort(quiet, axis=0)
    ranks = [(np.searchsorted(x, y, 'left') + np.searchsorted(x, y, 'right')).sum()
             / (2 * len(x) * len(y)) for x, y in zip(ordered.T, disturbed.T, strict=True)]
    return ranks


@torch.no_grad()
def measure(source):
    numerical_policy()
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    _, calibration, held = partition(rx, fit=4000)
    trackers = {rate: PacketRestorer('consensus', samples=rate).eval() for rate in RATES}
    cal_scores = panel(iq[calibration], trackers)[:, :len(SCORES)]
    pooled = 1.05 * cal_scores.max((0, 2))
    by_rx = {name: 1.05 * cal_scores[rx[calibration] != name].max((0, 2)) for name in SOURCE}
    observed, receivers = iq[held], rx[held]
    quiet = panel(observed, trackers)
    groups = {name: receivers == name for name in SOURCE}
    rtl = groups['rtl_1']

    def summarize(matrix):
        if not np.isfinite(matrix).all():
            raise FloatingPointError('Nonfinite source gate attribution')
        scores = matrix[:, :len(SCORES)]
        paired = [overlap(quiet[rtl, :len(SCORES), j], scores[~rtl, :, j]) for j in range(len(RATES))]
        return dict(active_pooled=(scores > pooled[None, :, None]).sum(0).tolist(),
                    rtl_clean_vs_non_rtl_pairwise_auc=np.asarray(paired).T.tolist(),
                    by_rx={name: dict(n=int(mask.sum()),
                                      active_held_rx=(scores[mask] > by_rx[name][None, :, None]).sum(0).tolist(),
                                      quantiles=np.quantile(matrix[mask], [.1, .5, .9], axis=0).tolist())
                           for name, mask in groups.items()})

    results = {'clean': summarize(quiet)}
    for family in FAMILIES:
        noise = generate_noise(observed, family)
        for level in LEVELS:
            results[f'{family}_{level:.2f}'] = summarize(panel(observed + np.float32(level)*noise, trackers))
    return dict(status='VERIFIED', role='U_s', protocol=contract['protocol'], rates_ksps=RATES,
                scores=SCORES, profiles=PROFILE, quantiles=[.1, .5, .9], results=results,
                guard_bins_by_rate=[shared_guard_coefficients(torch.zeros(n, dtype=torch.complex64)).numel()
                                    for n in RATES],
                held=len(held), calibration=len(calibration), pooled_thresholds=pooled.tolist(),
                leave_receiver_out_thresholds={name: row.tolist() for name, row in by_rx.items()},
                F_reads=0, TX_reads=0, V_reads=0, target_reads=0, satellite_trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, measure(args.source))
