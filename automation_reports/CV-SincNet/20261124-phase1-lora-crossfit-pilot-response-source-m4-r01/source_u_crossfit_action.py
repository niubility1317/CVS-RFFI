import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import setup
from experiments.pamf_lora_main.crossfit_pilot import (
    LAGS, apply_transfer, estimate_shared, generic_taps, regularized_inverse,
)
from experiments.pamf_lora_main.crossfit_pilot_source import RUN
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.pilot_joint import estimate_pilot
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

torch.set_num_threads(2)
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
(_,_,held),_=setup(rt,'cpu')
tracker=PacketRestorer('consensus').eval()
g=torch.Generator().manual_seed(2026112451)
freq=torch.fft.fftfreq(256)
mask=freq.abs()<=.25
angle=-2*torch.pi*torch.tensor(LAGS)[:,None]*freq
delay=torch.polar(torch.ones_like(angle),angle)
data={k:[] for k in ('clean','light','broad')}

def normshape(response):
    signal=response[...,mask]
    mean=signal.mean(-1,keepdim=True)
    return signal*mean.conj()/mean.abs().square().clamp_min(1e-8)

def measure(iq):
    y,rot,q,rms=coordinates(tracker,iq)
    v=observed_noise_rms(y).square()
    cross,s=estimate_shared(y,rot,q,v)
    local=estimate_pilot(y,rot,v)
    return (y,rot,rms,v,cross,local,s)

def centered(value):
    return value - value.mean(-1,keepdim=True)

def correction(y,rot,v,transfer):
    return regularized_inverse(y,y,rot,v,transfer)

with torch.no_grad():
    for iq in held.split(20):
        source=complex_iq(iq)
        taps=generic_taps(source,g)
        true=normshape(1 + taps @ delay)
        distorted=apply_transfer(source,taps)
        noise=colored_observation(distorted,g)-distorted
        yc,rc,sc,vc,old_shared,old_local,support=measure(iq)
        out=correction(yc,rc,vc,old_shared)
        edit=(out-yc).abs().square().mean(dim=(1,2)).sqrt()
        data['clean'].append(torch.stack((support,edit),dim=1))
        for stage,amount in (('light',.25),('broad',.85)):
            mixed=real_iq(distorted + amount*noise)
            y,rot,rms,v,shared,local,support=measure(mixed)
            tested=normshape((shared / old_shared).squeeze(1))
            previous=normshape(local / old_local)
            direct_error=(1-true).abs().square().mean(-1)
            shared_error=(tested-true).abs().square().mean(-1)
            local_error=(previous-true[:,None]).abs().square().mean((1,2))
            true_frame=centered(source/(rot*rms))
            raw_error=(centered(y)-true_frame).abs().square().mean(dim=(1,2))
            inverse_error=(centered(correction(y,rot,v,shared))-true_frame).abs().square().mean(dim=(1,2))
            data[stage].append(torch.stack((support,corruption_score(y),direct_error,local_error,
                                           shared_error,raw_error,inverse_error),dim=1))
metrics={'clean':('support','would_edit_noncarrier_rms'),
         'light':('support','observed_disturbance','identity_relative_H_error','old_three_tap_relative_H_error',
                  'crossfit_relative_H_error','raw_noncarrier_error','equalized_noncarrier_error')}
out={}
for stage,series in data.items():
    values=torch.cat(series)
    columns=metrics['clean' if stage=='clean' else 'light']
    out[stage]=dict(n=len(values),positive_support=int((values[:,0]>0).sum()),mean_support=float(values[:,0].mean()),
                    quantiles={key:values[:,pos].quantile(torch.tensor([.5,.9])).tolist()
                               for pos,key in enumerate(columns)})
print(json.dumps(dict(seed=2026112451,packets=len(held),target_reads=0,V_reads=0,F_reads=0,TX_reads=0,
                      scope='Source U, generic known transfer only for evaluation; relative shape removes per-packet scalar',
                      measurements=out)),flush=True)
