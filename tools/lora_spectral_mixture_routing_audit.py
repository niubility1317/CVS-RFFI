"""Check frozen LoRa mixture routing on clean and generically disturbed source U."""

import argparse
import json
from math import sqrt
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.phase_dose_bursts_source import burst_observation
from experiments.pamf_lora_main.spectral_mixture import COMPONENTS, ROUTING_DIMENSIONS
from experiments.pamf_lora_main.spectral_mixture_source import RUN, load, load_gate


def readout(model, tracked):
    probability = model.routing(tracked)[0][-1]
    certainty, component = probability.max(dim=-1)
    spectrum = model.spectrum(tracked)
    point = model.coefficients(spectrum)[:, 0, -ROUTING_DIMENSIONS:] / sqrt(6.)
    variance = model.routing_variance[component] + model.noise(spectrum)[:, 0, None] / 6.
    displacement = (point - model.routing_center[component]).square() / variance.clamp_min(1e-8)
    return component, certainty, displacement.mean(dim=-1)


@torch.no_grad()
def audit(rt):
    root = Path(rt['run_root'])
    if rt.get('run_id') != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the complete registered source-only study')
    torch.set_num_threads(3)
    (_, _, held), contract = setup(rt, 'cpu', fit_per_rx=4000)
    model = load(root, contract, load_gate(root), 'cpu')
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(2026112477)
    report = {}
    for rx, rows in zip(SOURCE, held.chunk(len(SOURCE)), strict=True):
        clean, disturbed = [], []
        for group in rows.split(32):
            clean.append(readout(model, coordinates(tracker, group)[0]))
            perturbed = real_iq(burst_observation(complex_iq(group), rng))
            disturbed.append(readout(model, coordinates(tracker, perturbed)[0]))
        grouped = [tuple(torch.cat(items) for items in zip(*branch, strict=True)) for branch in (clean, disturbed)]
        summaries = []
        for mode, certainty, displacement in grouped:
            summaries.append(dict(counts=mode.bincount(minlength=COMPONENTS).tolist(),
                                  median_max_posterior=certainty.median().item(),
                                  median_normalized_displacement=displacement.median().item()))
        report[rx] = dict(n=len(rows), clean=summaries[0], disturbed=summaries[1],
                          same_component=int((grouped[0][0] == grouped[1][0]).sum()))
    return dict(n=len(held), by_receiver=report, state_weights=model.mixture_weight.tolist(),
                target_reads=0, source_V_reads=0, classifier_reads=0, exact_noise_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(read(args.runtime)), allow_nan=False))
