import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import LAGS, apply_transfer, generic_taps
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.factorized_chirps import ChirpPrior
from experiments.pamf_lora_main.coupled_echo_source import RUN
from experiments.pamf_lora_main.coupled_echo import CoupledEcho, echo_system, integrated_echo, short_echo_columns
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.variance_bound import spectral_contrast, variance_interval
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
(fit, _, held), _ = setup(rt,'cpu')
tracker=PacketRestorer('consensus').eval()
model=CoupledEcho((8,8,8,8),echo=True)
bands=model.bands.bool()
bands[0,0]=False
freq_names=('1_to_8','9_to_32','33_to_64','above_64')
rng=torch.Generator().manual_seed(2026112460)
weights_sample={}

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

results = {}
prior_summary = {}
with torch.no_grad():
    bank=[]
    for group in fit.chunk(len(SOURCE)):
        bank.append(torch.cat([as_spectrum(coordinates(tracker,q)[0]) for q in group[:100].split(20)]))
    for held_idx,group in enumerate(held.chunk(len(SOURCE))):
        prior=ChirpPrior()
        prior.fit(torch.cat([v for i,v in enumerate(bank) if i!=held_idx]))
        prior_summary[SOURCE[held_idx]]={name:float(prior.shared[m].sum() /
            (prior.shared[m]+prior.individual[m]).sum()) for name,m in zip(freq_names,bands)}
        for iq in group.split(20):
            clean=complex_iq(iq)
            shape=clean.shape
            flattened=clean.flatten(0,1).unsqueeze(1)
            independent_taps=generic_taps(flattened,rng)
            shared_taps=generic_taps(clean,rng)
            private=apply_transfer(flattened,independent_taps).reshape(shape)
            shared=apply_transfer(clean,shared_taps)
            mixed=clean+.70710678*((private-clean)+(shared-clean))
            def disturb(x):
                return x+.85*(colored_observation(x,rng)-x)
            views={'clean':clean,'noise_only':disturb(clean),'shared_echo':disturb(shared),
                   'per_chirp_echo':disturb(private),'mixed_echo':disturb(mixed)}
            for condition,wave in views.items():
                y,p,q,r=coordinates(tracker,real_iq(wave))
                measured=as_spectrum(y)
                moment,weights=spectral_contrast(measured,q)
                v,_=variance_interval(observed_noise_rms(y).square(),moment,weights,model.selectors)
                v=v@model.bands
                target=as_spectrum(clean/(p*r))
                bins=model.echo_bins
                center=prior.mean
                subset=(measured-center).index_select(2,bins)
                columns=short_echo_columns(y,p,bins,model.echo_scale)
                identity=torch.eye(y.shape[1],dtype=columns.dtype)
                private_matrix=(columns[...,None,:]*identity[None,:,None,:,None]).flatten(-2)
                parameters=(v[...,bins],prior.shared[bins],prior.individual[bins])
                a_shared,l_shared=integrated_echo(subset,echo_system(columns,*parameters))
                a_local,l_local=integrated_echo(subset,echo_system(private_matrix,*parameters))
                combo_columns=torch.cat((columns,private_matrix),dim=-1)*.70710678
                a_combo,l_combo=integrated_echo(subset,echo_system(combo_columns,*parameters))
                only_shared=(.25*l_shared).sigmoid()[:,None,None]
                options=torch.stack((l_shared,l_local,l_combo),dim=-1)*.25 - torch.tensor(3.).log()
                odds=torch.cat((torch.zeros_like(options[:,:1]),options),dim=-1).softmax(-1)
                weights_sample.setdefault(condition,[]).append(odds)
                mixture=odds[:,1,None,None]*a_shared+odds[:,2,None,None]*a_local+odds[:,3,None,None]*a_combo
                model_shared=measured.index_copy(2,bins,measured[...,bins]-only_shared*a_shared)
                model_mix=measured.index_copy(2,bins,measured[...,bins]-mixture)
                for name,estimate in {
                    'received':measured,
                    'linked':prior.posterior(measured,center,v,coupled=True),
                    'shared_hypothesis':prior.posterior(model_shared,center,v,coupled=True),
                    'mixture_hypotheses':prior.posterior(model_mix,center,v,coupled=True),
                }.items():
                    for (component,band),affect in score(estimate,target):
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
print(json.dumps(dict(packets=len(held),fit_per_fold=400,tx_reads=0,F_reads=0,V_reads=0,
                      targets_read=0,source_clean_shared_fraction=prior_summary,by_disturbance=summary,
                      hypothesis_order=['none','shared','per_chirp','mixed'],
                      average_posterior_probability={s:torch.cat(v).mean(0).tolist() for s,v in weights_sample.items()})),flush=True)
