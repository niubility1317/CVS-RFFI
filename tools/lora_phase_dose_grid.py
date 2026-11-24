"""Measure source-U angular dose preferences without fitting or reading a classifier."""

import json
from pathlib import Path
import sys

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_phase import project
from experiments.pamf_lora_main.phase_observation_diagnostic import repeat_angle
from experiments.pamf_lora_main.reference_cancel import generic_observation


def preferences(relative_target, direction, weight, fractions):
    """Find the angular-error optimum for each block and for the entire packet."""
    residual = relative_target - direction * fractions[:, None, None, None]
    loss = (1. - residual.cos()) * weight
    positions = loss.unflatten(-1, (8, 32)).mean(-1)
    best_pos = positions.argmin(0)
    best_pack = positions.mean((-1, -2)).argmin(0)
    local_risk = positions.min(0).values.mean((-1, -2))
    pooled_risk = positions.mean((-1, -2)).min(0).values
    excitation = (weight * direction.square()).unflatten(-1, (8, 32)).mean(-1)
    return best_pos, best_pack, excitation, local_risk, pooled_risk


@torch.no_grad()
def summarize(rt, *, perturb=generic_observation):
    torch.set_num_threads(3)
    (_, _, held), _ = setup(rt, 'cpu', fit_per_rx=4000)
    tracker = PacketRestorer('consensus').eval()
    fractions = torch.linspace(0., 1., 9)
    rng = torch.Generator().manual_seed(2026112488)
    time = torch.linspace(-1., 1., 256)
    evidence = {}
    for receiver, group in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        rows = []
        for iq in group.split(24):
            clean = complex_iq(iq)
            distorted = perturb(clean, rng)
            y, rot, q, rms = coordinates(tracker, real_iq(distorted))
            target = clean / (rot * rms)
            signal = y * (rms / rms.square().mean(1, keepdim=True).sqrt())
            basis = project(repeat_angle(signal, q), distorted.abs().square(),
                            torch.ones_like(time), time, angle_cap=.65)
            actual_phase = (target * y.conj()).angle()
            target_power, observed_power = target.abs().square(), y.abs().square()
            reliability = (target_power * observed_power /
                           ((.05 + target_power) * (.05 + observed_power))).sqrt()
            rows.append(preferences(actual_phase, basis, reliability, fractions))
        block, packet, power, local, pooled = (torch.cat(items) for items in zip(*rows, strict=True))
        positional = torch.stack([((block == k) * power).sum() for k in range(len(fractions))])
        count = torch.bincount(packet, minlength=len(fractions))
        evidence[receiver] = dict(packets=len(group), optimal_packet_dose_counts=count.tolist(),
                                  optimal_position_dose_power=(positional / positional.sum().clamp_min(1e-8)).tolist(),
                                  local_risk=float(local.mean()), pooled_risk=float(pooled.mean()))
    return dict(scope='500 frozen source-U held packets, generic perturbation, no classifier or target',
                dose_grid=fractions.tolist(), results=evidence,
                projection_scope='per-block grid is before the final joint low-order projection')


if __name__ == '__main__':
    rt = read(Path(sys.argv[1]))
    perturb = generic_observation
    if 'bursts-source' in rt['run_id']:
        from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
        perturb = burst_observation
    print(json.dumps(summarize(rt, perturb=perturb), allow_nan=False))
