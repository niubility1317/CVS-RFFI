"""Check spectral edit and measured U waveform fidelity after both V routes close."""
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
release = root.parent.parent / 'releases' / (root.name + '-release')
sys.path.insert(0, str(release))

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition, corruption_score
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.repeat_covariance import RepeatCovariance, radial_at_same_dose, repeated_noise

torch.set_num_threads(2)
assert read(root / 'state.json')['status'] == 'SOURCE_SCORED'
cfg = read(release / 'runtime.json')
source = Path(cfg['source_root'])
contract = source_contract(source)
_, rx, _, _ = source_role(source, contract, 'U_s')
_, _, held = partition(rx)
sample = held[::10]
iq = np.load(root.parent / DATA_RUN / 'source_views' / 'U_s.npy', mmap_mode='r', allow_pickle=False)
clean = complex_iq(tensor(iq[0, sample], 'cpu'))
generator = torch.Generator().manual_seed(2026112417)
disturbed = real_iq(repeated_noise(clean, generator))
received, rotation, quality, rms = coordinates(PacketRestorer('consensus').eval(), disturbed)
ref = clean / (rotation * rms)

def fft(x):
    return torch.fft.fft(x, norm='ortho')

def power(x):
    return x.abs().square()

Y, T = fft(received), fft(ref)
frequency = torch.fft.fftfreq(Y.shape[-1], d=1/256).abs()
disturbance = corruption_score(received)
low_disturbance = disturbance <= disturbance.median()
bands = ((1,8),(9,32),(33,64),(65,128))
grid = torch.tensor([.10,.50,.90])

@torch.no_grad()
def collect(variant):
    name = 'protected_covariance' if variant else 'protected_diagonal'
    model = RepeatCovariance(variant, protected=True).eval()
    saved = torch.load(root / f'{name}-model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['weights'], strict=True)
    return model(received, quality)

a, b = collect(False), collect(True)
radial = radial_at_same_dose(received, b[1])
out = dict(direct=received, diagonal_coarse=a[0], diagonal=a[1],
           covariance_coarse=b[0], covariance=b[1], covariance_radial=radial)
metrics = {}
for name, wave in out.items():
    X = fft(wave)
    without_dc = (power(X[...,1:]).sum((1,2)) / power(Y[...,1:]).sum((1,2))).quantile(grid)
    valid = Y.abs() > 1e-5
    dose = (X - Y).abs()[valid] / Y.abs()[valid]
    angle = (Y.conj() * X).angle().abs()[valid]
    item = dict(noncarrier_retention=without_dc.tolist(), maximum_observed_edit=dose.max().item(),
                maximum_angle_deg=math.degrees(angle.max().item()))
    item['by_observed_disturbance'] = {}
    for level, keep in [('lower_half',low_disturbance),('upper_half',~low_disturbance)]:
        spectral = {}
        for first,last in bands:
            mask = (frequency >= first) & (frequency <= last)
            sel = X[keep][...,mask]
            ref_band = T[keep][...,mask]
            target_energy = power(ref_band).sum((1,2)).clamp_min(1e-10)
            recovery_norm = power(sel).sum((1,2))
            nmse = power(sel-ref_band).sum((1,2)) / target_energy
            magnitude_error = (sel.abs()-ref_band.abs()).square().sum((1,2)) / target_energy
            correlation = (sel*ref_band.conj()).real.sum((1,2)) / (target_energy * recovery_norm).clamp_min(1e-20).sqrt()
            spectral[f'{first}-{last}'] = dict(complex_nmse=nmse.median().item(), magnitude_nmse=magnitude_error.median().item(),
                                              real_correlation=correlation.median().item())
        item['by_observed_disturbance'][level] = spectral
    metrics[name] = item

difference = power(fft(b[1]-a[1])).sum((1,2)) / power(Y[...,1:]).sum((1,2))
matched_dose_delta = ((fft(radial)-Y).abs() - (fft(b[1])-Y).abs()).abs().amax().item()
predictions = {}
previous = root.parent / '20261124-phase1-lora-repeat-covariance-source-m4-r01'
for route in ('full','residual'):
    d = read(root / f'{route}-predictions.json')
    parent = read(previous / f'{route}-predictions.json')
    assert d['ids'] == parent['ids'] and d['rows'].keys() == parent['rows'].keys()
    assert all(np.array_equal(d['rows'][key][:2], parent['rows'][key][:2]) for key in d['rows'])
    x = np.asarray(list(d['rows'].values()))
    predictions[route] = {f'{arm}_vs_{control}': int(
        (x[:, d['modes'].index(arm)] != x[:, d['modes'].index(control)]).sum())
        for arm, control in [('covariance_coarse','diagonal_coarse'),('covariance','diagonal'),
                             ('covariance_half','diagonal_half'),('covariance','covariance_radial')]}

result = dict(status='VERIFIED', held_U=len(sample), seed=2026112417, target_reads=0, TX_reads=0,
              metrics=metrics, frozen_decision_differences=predictions,
              max_equal_dose_absolute_error=matched_dose_delta,
              relative_final_iq_distance=difference.quantile(grid).tolist())
print(json.dumps(result,indent=2))
