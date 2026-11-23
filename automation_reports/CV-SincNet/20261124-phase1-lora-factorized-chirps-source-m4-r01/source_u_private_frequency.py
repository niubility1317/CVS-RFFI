import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import apply_transfer, generic_taps
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.factorized_chirps import FactorizedChirps, ChirpPrior
from experiments.pamf_lora_main.factorized_chirps_source import RUN
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.variance_bound import spectral_contrast, variance_interval
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
(fit, _, held), _ = setup(rt,'cpu')
tracker=PacketRestorer('consensus').eval()
model=FactorizedChirps((8,8,8,8),coupled=True)
bands=model.bands.bool()
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

results = {}
prior_summary = {}
correlation = {}
with torch.no_grad():
    bank=[]
    for group in fit.chunk(len(SOURCE)):
        bank.append(torch.cat([as_spectrum(coordinates(tracker,q)[0]) for q in group[:100].split(20)]))
    for held_idx,group in enumerate(held.chunk(len(SOURCE))):
        prior=ChirpPrior()
        snapshots=torch.cat([v for i,v in enumerate(bank) if i!=held_idx])
        prior.fit(snapshots)
        mask=bands[0] | bands[1]
        private=(snapshots-snapshots.mean(1,keepdim=True))[...,mask]
        center=private.mean(0)
        fitted=(private-center).flatten(0,1)
        covariance=fitted.T @ fitted.conj()/len(fitted)
        diag=torch.diag(covariance.diagonal().real.clamp_min(1e-8)).to(covariance.dtype)
        shrunken=.85 * covariance + .15 * diag
        values=torch.linalg.eigvalsh(covariance).clamp_min(0.)
        correlation[SOURCE[held_idx]]=float(values[-8:].sum()/values.sum())
        def private_update(observed,noise,joint,source_cov):
            chirps=observed.shape[1]
            private=observed-observed.mean(1,keepdim=True)
            residual=private[...,mask]-center
            v=(1-2/chirps)*noise + noise.sum(1,keepdim=True)/chirps**2
            total=source_cov+torch.diag_embed(v[...,mask])
            solved=torch.linalg.solve(total,residual[...,None])
            revised=center+(source_cov@solved).squeeze(-1)
            revised=revised-revised.mean(1,keepdim=True)
            # Isolate spectral communication within the private component.
            out=joint.clone()
            out[...,mask]=joint[...,mask].mean(1,keepdim=True)+revised
            return out
        prior_summary[SOURCE[held_idx]]={name:float(prior.shared[m].sum() /
            (prior.shared[m]+prior.individual[m]).sum()) for name,m in zip(freq_names,bands)}
        for iq in group.split(20):
            clean=complex_iq(iq)
            echo=apply_transfer(clean,generic_taps(clean,rng))
            base_noise=colored_observation(clean,rng)-clean
            echo_noise=colored_observation(echo,rng)-echo
            views={'clean':clean,'light_noise':clean+.25*base_noise,'broad_noise':clean+.85*base_noise,
                   'light_echo':echo+.25*echo_noise,'broad_echo':echo+.85*echo_noise}
            for condition,wave in views.items():
                y,p,q,r=coordinates(tracker,real_iq(wave))
                measured=as_spectrum(y)
                moment,weights=spectral_contrast(measured,q)
                v,_=variance_interval(observed_noise_rms(y).square(),moment,weights,model.selectors)
                v=v@model.bands
                target=as_spectrum(clean/(p*r))
                original_joint=prior.posterior(measured,prior.mean,v,coupled=True)
                for name,estimate in {
                    'received':measured,
                    'independent':prior.posterior(measured,prior.mean,v,coupled=False),
                    'joint':original_joint,
                    'private_independent_freq':private_update(measured,v,original_joint,diag),
                    'private_correlated_freq':private_update(measured,v,original_joint,shrunken),
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
                      targets_read=0,source_clean_shared_fraction=prior_summary,private_top8_frequency_share=correlation,by_disturbance=summary)),flush=True)
