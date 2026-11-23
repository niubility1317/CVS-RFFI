from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.echo_marginal import common_weights
from experiments.pamf_lora_main.variance_bound_source import RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.native_detail import native_pair
from experiments.pamf_lora_main.packet_physics import residual_view
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
rt = read(project/'releases'/f'{RUN}-release'/'runtime.json')
activate(rt)
torch.set_num_threads(2)
source = Path(rt['source_root'])
raw,rx,ids,_ = source_role(source,source_contract(source),'U_s')
_,_,held = partition(rx)
chosen = np.concatenate([rows[:30] for rows in np.split(held,len(SOURCE))])
scenes = (VIEWS[3], VIEWS[4], VIEWS[-1])
routes = ('full','residual')

def trace_row(payload):
    route,scene,raw,r,id = payload
    if route=='full':
        observed,_,noise = native_pair((raw,r,id,scene,'source-U',None),sidebands=False)
    else:
        observed,signal,noise_complex = residual_view(raw,scene,r,id,'source-U',components=True)
        reconstruction = signal+noise_complex
        if not np.allclose(observed[:,0]+1j*observed[:,1],reconstruction,atol=8e-7,rtol=2e-6):
            raise ValueError('Native residual trace does not reproduce the registered observation')
        noise = np.stack((noise_complex.real,noise_complex.imag),axis=1).astype(np.float32)
    return observed,noise

tracker = PacketRestorer('consensus').eval()
prob = torch.tensor([.1,.5,.9])

frequency = (256*torch.fft.fftfreq(256)).round().abs()
bands = {'1_to_8':(frequency>0) & (frequency<=8), '9_to_32':(frequency>8)&(frequency<=32),
         '33_to_64':(frequency>32)&(frequency<=64), 'above_64':frequency>64}

def gather(obs, known_noise):
    y, rotation, quality, rms = coordinates(tracker, obs)
    variance = observed_noise_rms(y).square()
    s = torch.fft.fft(y,norm='ortho')
    noise = torch.fft.fft(complex_iq(known_noise)/(rotation*rms),norm='ortho')
    views = {'coherence':common_weights(quality),
             'precision':common_weights(quality/(variance+.002).sqrt())}
    results = {}
    for name, weights in views.items():
        complement = (1-weights).clamp_min(1e-5)
        diff = (s-(weights*s).sum(1,keepdim=True))/complement
        diff_noise = (noise-(weights*noise).sum(1,keepdim=True))/complement
        model_variance = (weights.square()*variance).sum(1,keepdim=True)
        strength = variance / (variance + (model_variance-weights.square()*variance) /
                               complement.square()).clamp_min(1e-9)
        for band, selected in bands.items():
            a, n = diff[...,selected], noise[...,selected]
            error = a-diff_noise[...,selected]
            inner = (a*n.conj()).real.sum(-1)
            ideal = (inner/a.abs().square().sum(-1).clamp_min(1e-12)).clamp(0,1)[...,None]
            nenergy=n.abs().square().sum(-1)
            rows = {
                'noise':nenergy, 'signal_in_direction':error.abs().square().sum(-1),
                'observed_fit_residual':(n-strength*a).abs().square().sum(-1),
                'oracle_dose_residual':(n-ideal*a).abs().square().sum(-1),
            }
            results.setdefault(band,{})[name] = {
                k: v.flatten() for k,v in rows.items()
            }
    return results

out={}
with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
    for route in routes:
        for scene in scenes:
            items=((route,scene,raw[i],rx[i],ids[i]) for i in chosen)
            values=list(pool.map(trace_row,items,chunksize=6))
            y,n=map(lambda rows:tensor(np.stack(rows),'cpu'),zip(*values,strict=True))
            arrays=gather(y,n)
            out.setdefault(route,{})[scene]={}
            for band, row in arrays.items():
                out[route][scene][band]={}
                for name, values in row.items():
                    energy=values['noise']
                    out[route][scene][band][name]={
                        stat: dict(energy_ratio=float(v.sum()/energy.sum()),
                                   p50=float((v/energy.clamp_min(1e-12)).median()))
                        for stat,v in values.items() if stat!='noise'
                    }
print(json.dumps(dict(packets=len(chosen), source_role='held_U_s', trace_is_diagnostic_only=True,
                      target_reads=0,V_reads=0,F_reads=0,TX_reads=0,
                      quantiles=[.1,.5,.9],by_route=out)),flush=True)
