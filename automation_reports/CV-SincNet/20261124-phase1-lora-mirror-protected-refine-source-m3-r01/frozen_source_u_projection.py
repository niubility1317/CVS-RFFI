
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.mirror_refine import BANDS, spectrum
from experiments.pamf_lora_main.mirror_refine_source import frozen_mirror
from experiments.pamf_lora_main.pilot_joint import apply_echo, echo_taps
from experiments.pamf_lora_main.rf_noise_transport import colored_observation

torch.set_num_threads(2)
base = Path('/home/szu2070436088/2510044040/CV-SincNet')
run = '20261124-phase1-lora-mirror-protected-refine-source-m3-r01'
release = base/'releases'/f'{run}-release'
rt = read(release/'runtime.json')
source = Path(rt['source_root'])
contract = source_contract(source)
_, rx, _, _ = source_role(source, contract, 'U_s')
_, _, held = partition(rx)
data = np.load(base/'runs'/DATA_RUN/'source_views'/'U_s.npy', mmap_mode='r', allow_pickle=False)
samples = tensor(data[0, held], 'cpu')
tracker = PacketRestorer('consensus').eval()
model, threshold = frozen_mirror(Path(rt['run_root']), contract, 'cpu')
rng = torch.Generator().manual_seed(2026112418)
arms = ('coarse', 'ordinary', 'protected', 'equal_dose')
risk = {name: [] for name in arms}
confidences, perturbations = [], []
with torch.no_grad():
    for sampleset in samples.split(15):
        clean = complex_iq(sampleset)
        received = real_iq(colored_observation(apply_echo(clean, echo_taps(clean, rng)), rng))
        observed, rotation, q, rms = coordinates(tracker, received)
        active = corruption_score(observed) > threshold
        target = spectrum((clean / (rotation * rms))[active])
        estimates = model(observed[active], q[active], rotation[active], controls=True)
        *outputs, confidence = estimates
        if not outputs[0].numel():
            continue
        masks = model.band_masks
        energy = (target.abs().square().unsqueeze(-2) * masks).sum(-1) / masks.sum(-1)
        for name, restored in zip(arms, outputs, strict=True):
            error = (spectrum(restored) - target).abs().square()
            band_error = (error.unsqueeze(-2) * masks).sum(-1) / masks.sum(-1)
            risk[name].append((band_error / (.002 + energy)).log1p().flatten(0, 1))
        confidences.append(confidence.flatten(0, 1))
        perturbations.append(corruption_score(observed[active]).repeat_interleave(observed.shape[1]))
risks = {name: torch.cat(values) for name, values in risk.items()}
confidence = torch.cat(confidences)
power = torch.cat(perturbations)
threshold90 = power.quantile(.9)
reports = {}
for label, selected in (('all_active', torch.ones_like(power, dtype=torch.bool)),
                        ('top_observed_10percent', power >= threshold90)):
    ratio = {name: [float(x) for x in values[selected].mean(0)] for name, values in risks.items()}
    hits = (risks['protected'][selected] < risks['ordinary'][selected])
    difference = risks['ordinary'][selected]-risks['protected'][selected]
    dose_gap = risks['equal_dose'][selected]-risks['protected'][selected]
    report = dict(chirps=int(selected.sum()), mean_log1p_floored_complex_error=ratio,
                  mean_confidence=confidence[selected].mean(0).tolist(),
                  protected_better_fraction=hits.float().mean(0).tolist(),
                  mean_ordinary_minus_protected=difference.mean(0).tolist(),
                  mean_equal_dose_minus_protected=dose_gap.mean(0).tolist())
    reports[label]=report
print(json.dumps(dict(source_U_holdout=len(held), held_draw_seed=2026112418, bands_khz=BANDS,
                      selected_on_observed_corruption=True, source_V_reads=0, TX_reads=0,
                      target_reads=0, recognition_reads=0, checkpoint='frozen mirror parent',
                      risk_definition='log1p(band complex error power / (0.002 + reference power))',
                      statistics=reports), indent=2), flush=True)
