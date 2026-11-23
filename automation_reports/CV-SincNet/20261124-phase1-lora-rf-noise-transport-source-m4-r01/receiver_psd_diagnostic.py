"""Isolated source-U RF noise-profile comparison, after the two V routes close."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
release = root.parent.parent / 'releases' / (root.name + '-release')
sys.path.insert(0, str(release))

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import RFNoiseTransport, colored_observation

torch.set_num_threads(2)
assert read(root / 'state.json')['status'] == 'SOURCE_SCORED'
rt = read(release / 'runtime.json')
source = Path(rt['source_root'])
contract = source_contract(source)
_, rx, _, _ = source_role(source, contract, 'U_s')
_, _, held = partition(rx)
picked = held[::10]
source_iq = np.load(root.parent / DATA_RUN / 'source_views' / 'U_s.npy',
                    mmap_mode='r', allow_pickle=False)
clean = complex_iq(tensor(source_iq[0, picked], 'cpu'))
rng = torch.Generator().manual_seed(2026112421)
corrupted = colored_observation(clean, rng)
framed, rotation, quality, rms = coordinates(PacketRestorer('consensus').eval(), real_iq(corrupted))
truth = clean / (rotation * rms)
q = torch.tensor((.10,.50,.90))
regions = ((1,8),(9,32),(33,64),(65,128))
frequency = torch.fft.fftfreq(clean.shape[-1],d=1/256).abs()

def power(x):
    return x.abs().square()

def fft(x):
    return torch.fft.fft(x,norm='ortho')

def smooth(x):
    # Short blur before scoring a PSD against a single stochastic realization.
    return F.avg_pool2d(x.flatten(0,1).unsqueeze(1), (3,3),stride=1,padding=1,
                        count_include_pad=False).squeeze(1).unflatten(0,x.shape[:2])

def ratio(x):
    return x / x.mean(dim=-1,keepdim=True).clamp_min(1e-8)

with torch.no_grad():
    reference = RFNoiseTransport(False).eval()
    noise_evidence = smooth(power(reference.analyze(framed-truth)))
    high = corruption_score(framed)
    high = high > high.median()
    waves = {'direct': framed}
    stats={}
    for name,aligned in [('flat',False),('transport',True)]:
        model = RFNoiseTransport(aligned).eval()
        saved = torch.load(root / f'{name}-model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['weights'],strict=True)
        estimate = model.estimate_noise(framed,rotation)
        error = (ratio(estimate)-ratio(noise_evidence)).square().mean(dim=(1,2,3))
        bias = estimate.mean(dim=(1,2,3))/noise_evidence.mean(dim=(1,2,3))
        stats[name] = dict(relative_psd_shape_mse=error.quantile(q).tolist(),
                           estimated_to_measured_power=bias.quantile(q).tolist(),
                           frequency_relative_std=ratio(estimate).std(dim=-1).mean(dim=(1,2)).quantile(q).tolist())
        coarse, refined = model(framed,quality,rotation)
        waves[name+'_coarse'] = coarse
        waves[name] = refined
        waves[name+'_half'] = torch.lerp(framed,refined,.5)
    T,Y=fft(truth),fft(framed)
    evidence={}
    for mode,value in waves.items():
        X=fft(value)
        item=dict(relative_edit_power=(power(X-Y).sum(dim=(1,2))/power(Y).sum(dim=(1,2))).quantile(q).tolist(),
                  noncarrier_retention=(power(X[...,1:]).sum(dim=(1,2))/power(Y[...,1:]).sum(dim=(1,2))).quantile(q).tolist(),
                  max_dc_difference=(X[...,0]-Y[...,0]).abs().max().item())
        for name,selector in [('lower_observed_disturbance',~high),('higher_observed_disturbance',high)]:
            item[name]={}
            for first,last in regions:
                b=(frequency>=first)&(frequency<=last)
                target=T[selector][...,b]
                candidate=X[selector][...,b]
                e=power(target).sum(dim=(1,2)).clamp_min(1e-10)
                n=power(candidate).sum(dim=(1,2)).clamp_min(1e-10)
                complex_err=power(candidate-target).sum(dim=(1,2))/e
                amp_err=(candidate.abs()-target.abs()).square().sum(dim=(1,2))/e
                agreement=(candidate*target.conj()).real.sum(dim=(1,2))/(e*n).sqrt()
                item[name][f'{first}-{last}']=dict(complex_nmse=complex_err.median().item(),
                                                    magnitude_nmse=amp_err.median().item(),
                                                    real_correlation=agreement.median().item())
        evidence[mode]=item

comparisons={}
parent=root.parent/'20261124-phase1-lora-repeat-protected-spectrum-source-m4-r01'
for route in ('full','residual'):
    p=read(root/f'{route}-predictions.json')
    b=read(parent/f'{route}-predictions.json')
    assert p['ids']==b['ids']
    difference={}
    for view,rows in p['rows'].items():
        assert np.array_equal(rows[:2],b['rows'][view][:2])
        matrix=np.asarray(rows)
        difference[view]={name: int((matrix[p['modes'].index('transport'+name)]!=
                                     matrix[p['modes'].index('flat'+name)]).sum())
                          for name in ('_coarse','','_half')}
    comparisons[route]=difference

print(json.dumps(dict(status='VERIFIED',hold=len(picked),TX_reads=0,target_reads=0,
                      estimates=stats,waves=evidence,frozen_decision_differences=comparisons),indent=2))
