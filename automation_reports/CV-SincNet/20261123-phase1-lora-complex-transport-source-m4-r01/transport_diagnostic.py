"""Replay weak-difference transport on the fixed unlabeled LoRa holdout."""
import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind import corruption_score, generic_observation
from experiments.pamf_lora_main.channel_blind_source import setup, proof
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.natural_detail import detail_bands, paired_echo_inputs, weak_difference
from experiments.pamf_lora_main.phase_transport_source import RECIPES, RUN

rt = read(Path('runtime.json'))
assert rt['run_id'] == RUN
root = Path(rt['run_root'])
torch.set_num_threads(2)
(_, _, held), contract = setup(rt, 'cpu')
models = {}
for name, recipe in RECIPES.items():
    checkpoint = torch.load(root/f'{name}-model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['proof'] == proof(contract, recipe)
    model = recipe.model().eval()
    model.load_state_dict(checkpoint['weights'], strict=True)
    models[name] = model

tracker = PacketRestorer('consensus').eval()
rng = torch.Generator().manual_seed(2026112315)
targets, measures, outputs = [], [], {'direct': [], **{key: [] for key in models}}
with torch.no_grad():
    for iq in (batch for receiver in held.chunk(len(SOURCE)) for batch in receiver.split(16)):
        clean = complex_iq(iq)
        clean_ref, rot, _, rms = coordinates(tracker, iq)
        clean_basis = rot * rms
        delta = weak_difference(clean_ref)
        targets.append(torch.fft.fft(delta, norm='ortho'))
        received = paired_echo_inputs(clean, generic_observation, rng, delta*clean_basis,
                                      with_identity=False)[:, (0, 2)]
        measured, rot, quality, rms = coordinates(tracker, real_iq(received.flatten(0, 1)))
        basis = rot * rms
        measures.append(corruption_score(measured[::2]))
        raw_delta = (received[:, 1] - received[:, 0]) / clean_basis
        outputs['direct'].append(torch.fft.fft(raw_delta, norm='ortho'))
        for arm, model in models.items():
            _, restored = model(measured, quality)
            restored = (restored * basis).unflatten(0, received.shape[:2])
            restored_delta = (restored[:, 1] - restored[:, 0]) / clean_basis
            outputs[arm].append(torch.fft.fft(restored_delta, norm='ortho'))

power_proxy = torch.cat(measures)
median = power_proxy.median()
target = torch.cat(targets)
bands = detail_bands(target)
results = {}
for arm, values in outputs.items():
    outputs_arm = torch.cat(values)
    results[arm] = {}
    for strength, rows in (('weaker', power_proxy <= median), ('stronger', power_proxy > median)):
        results[arm][strength] = {}
        for band_name, band in zip(('1-8 kHz','9-32 kHz'), bands, strict=True):
            pred, reference = outputs_arm[rows][..., band], target[rows][..., band]
            energy = reference.abs().square().sum()
            gain = (pred * reference.conj()).sum() / energy
            error = (pred - reference).abs().square().sum() / energy
            results[arm][strength][band_name] = dict(in_phase=gain.real.item(), quadrature=gain.imag.item(),
                                                     complex_nmse=error.item(), n=int(rows.sum()))
print(json.dumps(dict(source_U_holdout=len(held), seed=2026112315, median_power_proxy=median.item(),
                      source_truth_reads=0, target_reads=0, recognizer_reads=0, ungated=True,
                      metrics=results), indent=2, allow_nan=False))
