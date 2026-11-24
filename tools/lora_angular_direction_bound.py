"""Compare matched-dose correction spaces on legal, held-out LoRa source waveforms."""

import argparse
import json
from math import atan, pi
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.angular_transport import AngularTransport, initialize, inner
from experiments.pamf_lora_main.angular_transport_source import RUN
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
from experiments.pamf_lora_main.split_shrink import spectrum, wave


FIELDS = ('analytic_error', 'projected_parallel_error', 'projected_complex_error',
          'parallel_required_angle_degrees', 'complex_required_angle_degrees',
          'parallel_gain', 'complex_gain', 'complex_minus_parallel_gain')


def best_plane_edit(origin, target, rms):
    norm = inner(origin, origin, rms).clamp_min(1e-12).sqrt()
    axis = origin / norm
    along = inner(target, axis, rms)
    across = target - along * axis
    radius = inner(across, across, rms).clamp_min(0.).sqrt()
    needed = torch.atan2(radius, along)
    angle = needed.clamp(max=atan(1.5))
    lateral = across / radius.clamp_min(1e-12)
    choice = norm * (angle.cos() * axis + angle.sin() * lateral)
    usable = (radius > 1e-8) & (norm > 1e-6)
    return torch.where(usable, choice, origin), needed


def projected_losses(anchor, axes, target, received, rms):
    unit = axes / axes.abs().clamp_min(1e-8)
    complex_ref = target - target.mean(-1, keepdim=True)
    parallel = wave((spectrum(complex_ref) * unit.conj()).real * unit)
    restricted, angle_parallel = best_plane_edit(anchor, parallel, rms)
    complex_edit, angle_complex = best_plane_edit(anchor, complex_ref, rms)
    normalization = inner(received, received, rms).clamp_min(1e-8)

    def error(edit):
        delta = edit - target
        return inner(delta, delta, rms) / normalization

    baseline, restricted_loss, complex_loss = (error(edit) for edit in (anchor, restricted, complex_edit))
    values = (baseline, restricted_loss, complex_loss, angle_parallel * (180 / pi),
              angle_complex * (180 / pi), baseline - restricted_loss, baseline - complex_loss,
              restricted_loss - complex_loss)
    return torch.cat(values, dim=1).squeeze(-1)


@torch.no_grad()
def diagnose(rt):
    if rt.get('run_id') != RUN:
        raise ValueError('Expected the registered source-only angular transport study')
    torch.set_num_threads(3)
    (fit, _, held), _ = setup(rt, 'cpu', fit_per_rx=4000)
    model = AngularTransport(angular=True).eval()
    tracker = PacketRestorer('consensus').eval()
    initialize(model, tracker, fit, None)
    rng = torch.Generator().manual_seed(2026112477)
    by_receiver = {}
    for receiver, batch in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        results = []
        for iq in batch.split(24):
            clean = complex_iq(iq)
            raw = burst_observation(clean, rng)
            y, phase, quality, rms = coordinates(tracker, real_iq(raw))
            anchor, axes, *_ = model.observe(y, quality, rms)
            desired = clean / (phase * rms) - y
            results.append(projected_losses(anchor, axes, desired, y, rms))
        values = torch.cat(results)
        if (values[:, -1] < -2e-6).any():
            raise ValueError('The full non-carrier correction space must include the parallel projection')
        by_receiver[receiver] = dict(n=len(values), median=values.median(0).values.tolist(),
                                     complex_gain_fraction=(values[:, -1] > 1e-5).float().mean().item())
    return dict(fields=FIELDS, by_receiver=by_receiver, n=len(held), source_u_fit=len(fit),
                target_reads=0, V_reads=0, F_reads=0, exact_trace_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(diagnose(read(arguments.runtime)), allow_nan=False))
