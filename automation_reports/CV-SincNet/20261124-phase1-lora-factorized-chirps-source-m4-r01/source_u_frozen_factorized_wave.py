import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import apply_transfer, generic_taps
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.factorized_chirps_source import RUN, RECIPES
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation

torch.set_num_threads(2)
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
(_, _, held), contract = setup(rt,'cpu')
tracker=PacketRestorer('consensus').eval()
root=Path(rt['run_root'])
models,cutoffs=zip(*(load_model(root,recipe,contract,'cpu') for recipe in RECIPES.values()),strict=True)
assert len(set(cutoffs))==1
independent,joint=models
bands=joint.bands.bool()
bands[0,0]=False
freq_names=('1_to_8','9_to_32','33_to_64','above_64')
rng=torch.Generator().manual_seed(2026112458)

def as_spectrum(y):
    return torch.fft.fft(y,norm='ortho')

def score(estimate, target):
    def components(z):
        common=z.mean(1,keepdim=True)
        return common,z-common
    for name,guess,actual in zip(('common','private'),components(estimate),components(target)):
        for idx,mask in enumerate(bands):
            x,y=guess[...,mask],actual[...,mask]
            power=y.abs().square().sum((1,2))
            error=(x-y).abs().square().sum((1,2))
            product=(x*y.conj()).sum((1,2))
            yield (name,freq_names[idx]),torch.stack((error,power,product.real,product.imag),dim=1)

results={}
activations={}
with torch.no_grad():
    for iq in held.split(20):
        clean=complex_iq(iq)
        echo=apply_transfer(clean,generic_taps(clean,rng))
        base_noise=colored_observation(clean,rng)-clean
        echo_noise=colored_observation(echo,rng)-echo
        views={'clean':clean,'light_noise':clean+.25*base_noise,'broad_noise':clean+.85*base_noise,
               'light_echo':echo+.25*echo_noise,'broad_echo':echo+.85*echo_noise}
        for condition,wave in views.items():
            y,p,q,r=coordinates(tracker,real_iq(wave))
            active=corruption_score(y)>cutoffs[0]
            activations[condition]=activations.get(condition,0)+int(active.sum())
            target=as_spectrum(clean/(p*r))
            estimates={'received':y,'independent_half':y.clone(),'joint_half':y.clone(),
                       'joint_unlinked_half':y.clone(),'joint_common_only':y.clone(),'joint_private_only':y.clone()}
            if active.any():
                packed=(y[active],q[active],p[active])
                observed=packed[0]
                _, guess=independent(*packed)
                _, corrected, _, _, unlinked=joint(*packed,controls=True)
                common=(corrected-observed).mean(1,keepdim=True)
                estimates['independent_half'][active]=torch.lerp(observed,guess,.5)
                estimates['joint_half'][active]=torch.lerp(observed,corrected,.5)
                estimates['joint_unlinked_half'][active]=torch.lerp(observed,unlinked,.5)
                estimates['joint_common_only'][active]=observed+.5*common
                estimates['joint_private_only'][active]=torch.lerp(observed,corrected,.5)-.5*common
            for name,estimate in estimates.items():
                for (component,band),affect in score(as_spectrum(estimate),target):
                    key=(condition,component,band,name)
                    results.setdefault(key,[]).append(affect)
summary={}
for (condition,component,band,name),rows in results.items():
    e,p,re,im=torch.cat(rows).T
    gain=re/p.clamp_min(1e-10)
    phase=torch.atan2(im,re).abs()
    summary.setdefault(condition,{}).setdefault(component,{}).setdefault(band,{})[name]={
        'relative_error':float(e.sum()/p.sum()), 'median_real_gain':float(gain.median()),
        'median_absolute_phase_rad':float(phase.median()),
    }
print(json.dumps(dict(packets=len(held),activations=activations,tx_reads=0,F_reads=0,V_reads=0,
                      targets_read=0,threshold=cutoffs[0],by_disturbance=summary)),flush=True)
