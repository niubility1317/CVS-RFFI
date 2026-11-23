import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import (
    LAGS, apply_transfer, estimate_shared, generic_taps, regularized_inverse,
)
from experiments.pamf_lora_main.crossfit_pilot_source import RUN
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
(fit,_,held),_=setup(rt,'cpu')
calib=torch.cat([v[:100] for v in fit.chunk(len(SOURCE))])
tracker=PacketRestorer('consensus').eval()
rng=torch.Generator().manual_seed(2026112451)
freq=torch.fft.fftfreq(256)
band=freq.abs()<=.25
angle=-2*torch.pi*torch.tensor(LAGS)[:,None]*freq
delays=torch.polar(torch.ones_like(angle),angle)

def normalize(response):
    anchor=response[...,band].mean(-1,keepdim=True)
    return response*anchor.conj()/anchor.abs().square().clamp_min(1e-8)

def estimator(iq):
    y,rot,q,rms=coordinates(tracker,iq)
    v=observed_noise_rms(y).square()
    estimate,_=estimate_shared(y,rot,q,v)
    return y,rot,rms,v,normalize(estimate.squeeze(1))

def vector(h):
    return torch.cat((h.real,h.imag),dim=-1)

def transfer(v):
    re,im=v.chunk(2,dim=-1)
    diff=torch.complex(re,im)
    diff=diff*(.45/diff.abs().clamp_min(1e-8)).clamp_max(1.)
    return 1+diff

def center(wave):
    return wave-wave.mean(-1,keepdim=True)

with torch.no_grad():
    clean_vec=torch.cat([vector(estimator(iq)[-1]-1) for iq in calib.split(20)])
    clean_mean=clean_vec.mean(0)
    centered=clean_vec-clean_mean
    cov=centered.T@centered/(len(calib)-1)
    values,basis=torch.linalg.eigh(cov)
    tau=(.04*values[-1]).clamp_min(1e-6)
    shrink=tau/(tau+values.clamp_min(0))
    folds = []
    for receiver in range(len(SOURCE)):
        others = torch.cat([values for i,values in enumerate(clean_vec.chunk(len(SOURCE))) if i != receiver])
        mean = others.mean(0)
        difference = others - mean
        eig, axes = torch.linalg.eigh(difference.T @ difference / (len(others)-1))
        scale = (.04*eig[-1]).clamp_min(1e-6)
        folds.append((mean,axes,scale/(scale+eig.clamp_min(0))))
    modes=('identity','original','pooled_covariance','held_rx_covariance')
    outputs={stage:{mode:[] for mode in modes} for stage in ('clean','light','broad')}
    for step,iq in enumerate(held.split(20)):
        mean, axes, factors = folds[step//5]
        source=complex_iq(iq)
        taps=generic_taps(source,rng)
        exact=normalize(1+taps@delays)
        disturbed=apply_transfer(source,taps)
        epsilon=colored_observation(disturbed,rng)-disturbed
        for stage in outputs:
            if stage=='clean':
                wave=iq
                reference=torch.ones_like(exact)
            else:
                wave=real_iq(disturbed+(.25 if stage=='light' else .85)*epsilon)
                reference=exact
            y,rotation,rms,variance,H=estimator(wave)
            residual=vector(H-1)-clean_mean
            candidate=(residual@basis*shrink)@basis.T
            other=(vector(H-1)-mean)@axes
            other=(other*factors)@axes.T
            transfers=(torch.ones_like(H),H,transfer(candidate),transfer(other))
            known=center(source/(rotation*rms))
            for mode,guess in zip(modes,transfers,strict=True):
                relative=(guess[...,band]-reference[...,band]).abs().square().mean(-1)
                restored=regularized_inverse(y,y,rotation,variance,guess[:,None])
                waveform=(center(restored)-known).abs().square().mean((1,2))
                extras=(restored-y).abs().square().mean((1,2)).sqrt()
                outputs[stage][mode].append(torch.stack((relative,waveform,extras),dim=1))
results={}
for stage,items in outputs.items():
    results[stage]={}
    for name,chunks in items.items():
        all_rx=torch.cat(chunks)
        grouped={'pooled': all_rx} | dict(zip(SOURCE,all_rx.chunk(len(SOURCE)),strict=True))
        results[stage][name]={key:data.quantile(torch.tensor([.5,.9]),dim=0).tolist() for key,data in grouped.items()}
print(json.dumps(dict(held_packets=len(held),fit_packets=len(calib),target_reads=0,V_reads=0,TX_reads=0,F_reads=0,
                      views=['clean','light','broad'],quantiles=[.5,.9],fit_per_fold=400,held_per_rx=100,
                      vector_order=['response_mse','noncarrier_waveform_mse','edit_rms'],
                      eigenvalues_top5=values[-5:].tolist(),tau=float(tau),measurements=results)),flush=True)
