"""Measure available covariance-direction gain on fixed unlabeled source U."""

import argparse
import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.cross_frequency_dose import CrossFrequencyDose, initialize
from experiments.pamf_lora_main.cross_frequency_dose_source import RUN
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation


FIELDS = ('raw_error', 'analytic_error', 'best_tied_error', 'best_separate_error',
          'best_tied_gain', 'best_common_gain', 'best_private_gain', 'separate_edit_rms')


def projection(observed, source, directions):
    """Score the same capped two-direction space with a fixed 21×21 grid."""
    error = source - observed
    x, y = directions
    raw_energy = observed.abs().square().mean((1, 2)).clamp_min(1e-8)
    def norm(value):
        return value.abs().square().mean((1, 2)) / raw_energy

    def overlap(left, right):
        return (left * right.conj()).real.mean((1, 2)) / raw_energy

    # Build each physical squared error from the small Gram matrix. This is
    # exactly the error after capping the physical IQ change along each pair.
    values = torch.linspace(0., 1.25, 21, dtype=raw_energy.dtype, device=raw_energy.device)
    box = torch.cartesian_prod(values, values)
    pairs = torch.cat((box, observed.real.new_tensor([[1., 1.]])), dim=0)
    first, second = pairs.T
    movement = (norm(x)[:, None] * first.square() + norm(y)[:, None] * second.square()
                + 2. * overlap(x, y)[:, None] * first * second).clamp_min(0.)
    cap = (.30 / movement.clamp_min(1e-12).sqrt()).clamp_max(1.)
    raw = norm(error)
    inner = overlap(error, x)[:, None] * first + overlap(error, y)[:, None] * second
    residual = raw[:, None] - 2. * inner * cap + movement * cap.square()
    tied = torch.where(first == second, residual, torch.inf)
    index = residual.argmin(-1)
    selected = index[:, None]
    tied_gain = pairs[tied.argmin(-1), 0]
    best_dose = (movement.sqrt() * cap).gather(-1, selected)[:, 0]
    return torch.stack((raw, residual[:, -1], tied.amin(-1), residual.amin(-1),
                        tied_gain, pairs[index, 0], pairs[index, 1], best_dose), dim=-1)


@torch.no_grad()
def diagnose(rt):
    if rt.get('run_id') != RUN:
        raise ValueError('Expected the registered cross-frequency dose source run')
    torch.set_num_threads(3)
    (fit, _, held), _ = setup(rt, 'cpu', fit_per_rx=4000)
    tracker = PacketRestorer('consensus').eval()
    model = CrossFrequencyDose(separate=True).eval()
    initialize(model, tracker, fit, None)
    rng = torch.Generator().manual_seed(2026112477)
    by_rx = {}
    for rx, packets in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        checks = []
        for sample in packets.split(24):
            source = complex_iq(sample)
            raw = burst_observation(source, rng)
            y, phase, q, rms = coordinates(tracker, real_iq(raw))
            observed = model.observe(y, q, rms)
            directions = [edit * phase * rms for edit in observed[1]]
            checks.append(projection(raw, source, directions))
        rows = torch.cat(checks, dim=0)
        by_rx[rx] = dict(n=len(rows), median=rows.median(0).values.tolist(),
                         median_extra_reduction=(rows[:, 2] - rows[:, 3]).median().item(),
                         separate_improved_fraction=(rows[:, 2] - rows[:, 3] > 1e-5).float().mean().item())
    return dict(fields=FIELDS, by_rx=by_rx, source_u_fit=len(fit), source_u_held=len(held),
                classifier_reads=0, target_reads=0, source_V_reads=0, trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(diagnose(read(arguments.runtime)), allow_nan=False))
