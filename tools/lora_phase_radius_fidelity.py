"""Measure the frozen LoRa radial corrections against isolated source traces."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.local_wiener_source import read_frozen
from experiments.pamf_lora_main.phase_radius_diagnostic import (
    DEVICE, MODES, original, radial_ratio, radius_path, radius_proposals, replay_trace,
)
from experiments.pamf_lora_main.runner import activate


FIELDS = ('estimated_noise', 'exact_added_noise', 'noise_ratio', 'fold_coherence',
          'power_direction_cosine', 'fold_direction_cosine',
          'power_radius_squared_error_improvement', 'fold_radius_squared_error_improvement')


def radial_evidence(iq, epsilon, tracker):
    y, rotation, quality, level = coordinates(tracker, iq)
    basis = rotation * level
    raw = complex_iq(iq)
    normalized_noise = complex_iq(epsilon) / basis
    clean = raw - (normalized_noise - normalized_noise.mean(-1, keepdim=True)) * basis
    amplitude = raw.abs()
    exact = radius_path(amplitude, clean.abs(), 2.) - amplitude
    power, folded, estimate, coherent = radius_proposals(y, quality, level)
    radius = [radius_path(amplitude, p, .06) for p in (power, folded)]
    budget = torch.minimum(*(radial_ratio(amplitude, d) for d in radius))[:, None, None]
    directions = [radius_path(amplitude, p, budget) - amplitude for p in (power, folded)]
    energy = amplitude.square().sum((-2, -1)).clamp_min(1e-10)
    exact_energy = exact.square().sum((-2, -1))
    dot = [(d * exact).sum((-2, -1)) for d in directions]
    dose = [d.square().sum((-2, -1)) for d in directions]
    cosine = [m / (p * exact_energy).sqrt().clamp_min(1e-10) for m, p in zip(dot, dose, strict=True)]
    benefit = [(2 * m - p) / energy for m, p in zip(dot, dose, strict=True)]
    variance = estimate.mean((1, 2))
    truth = normalized_noise.abs().square().mean((1, 2))
    values = (variance, truth, variance / truth.clamp_min(1e-10), coherent.mean((1, 2)), *cosine, *benefit)
    return torch.stack(values, -1), corruption_score(y)


@torch.no_grad()
def measure(rt, route):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Require truth-last completion before isolated waveform measurement')
    contract = source_contract(source)
    _, _, threshold = original(rt, contract)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = panel(rx)
    frozen, _, marks = read_frozen(root, route, MODES)
    if frozen != ids[chosen].tolist():
        raise ValueError('Waveform diagnosis must reuse the frozen physical IDs')
    replay = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    tracker = PacketRestorer('consensus').to(DEVICE).eval()
    result = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene in (VIEWS[1], VIEWS[-1]):
            frames, noise = (np.stack(v) for v in zip(
                *pool.map(partial(replay_trace, route, scene), replay, chunksize=8), strict=True))
            rows, guards = [], []
            for iq, epsilon in zip(tensor(frames, DEVICE).split(18), tensor(noise, DEVICE).split(18), strict=True):
                values, score = radial_evidence(iq, epsilon, tracker)
                rows.extend(values.tolist())
                guards.extend((score > threshold).tolist())
            rows, guards = np.asarray(rows), np.asarray(guards)
            if int(guards.sum()) != marks[scene]['active']:
                raise ValueError('Audit guard changed for the same waveform')
            result[scene] = {}
            for key in (*SOURCE, 'all'):
                selected = guards if key == 'all' else guards & (rx[chosen] == key)
                stats = np.median(rows[selected], axis=0).tolist() if selected.any() else []
                result[scene][key] = dict(active=int(selected.sum()), median=stats)
    return dict(route=route, n=len(frozen), fields=FIELDS, scenes=result, target_reads=0)


if __name__ == '__main__':
    runtime = read(Path(sys.argv[1]))
    activate(runtime)
    print(json.dumps(measure(runtime, sys.argv[2]), allow_nan=False))
