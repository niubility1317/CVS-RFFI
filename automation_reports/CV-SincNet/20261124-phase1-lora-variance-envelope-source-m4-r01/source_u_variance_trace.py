from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition, setup
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.echo_marginal import common_weights
from experiments.pamf_lora_main.echo_marginal_source import RUN
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

frequency = (256 * torch.fft.fftfreq(256)).round().abs()
bands = {'1_to_8':(frequency>0) & (frequency<=8),
         '9_to_32':(frequency>8)&(frequency<=32), 'above_64':frequency>64}
tracker = PacketRestorer('consensus').eval()
prob = torch.tensor([.1,.5,.9])
bands = {'1_to_8':(frequency>0) & (frequency<=8),
         '9_to_32':(frequency>8)&(frequency<=32),
         '33_to_64':(frequency>32)&(frequency<=64),
         'above_64':frequency>64}

def observed_moments(y, w):
    spectrum = torch.fft.fft(y,norm='ortho')
    average = (spectrum*w).sum(1,keepdim=True)
    return (w*(spectrum-average).abs().square()).sum(1)

(fit,_,_),_ = setup(rt,'cpu')
ancestors = torch.cat([packets[:100] for packets in fit.chunk(len(SOURCE))])
floor_pool = []
with torch.no_grad():
    for iq in ancestors.split(20):
        signal,_,q,_ = coordinates(tracker,iq)
        w = common_weights(q)
        observed_differences = observed_moments(signal,w)
        predicted = (w*(1-w)*observed_noise_rms(signal).square()).sum(1)
        floor_pool.append(torch.stack([(observed_differences[...,band].mean(-1)-predicted[:,0]).clamp_min(0.)
                                      for band in bands.values()],dim=-1))
floor=torch.cat(floor_pool).median(dim=0).values

def gather(obs, known_noise):
    y, rotation, quality, rms = coordinates(tracker, obs)
    weight = common_weights(quality)
    channel_noise = complex_iq(known_noise)/(rotation*rms)
    spectrum = torch.fft.fft(channel_noise, norm='ortho')
    observed_variance = observed_noise_rms(y).square()
    estimate = (observed_variance * weight.square()).sum(1).squeeze(-1)
    expected_contrast = (weight*(1-weight)*observed_variance).sum(1).squeeze(-1)
    moment = observed_moments(y,weight)
    results = {}
    for i,(name,select) in enumerate(bands.items()):
        spe = spectrum[...,select]
        actual = (spe*weight).sum(1).abs().square().mean(1)
        diag = (spe.abs().square()*weight.square()).sum(1).mean(1)
        ratio = (moment[:,select].mean(-1)-floor[i]).clamp_min(0.) / expected_contrast.clamp_min(1e-9)
        shrink = ratio.clamp(.25,4.).sqrt()
        results[name] = torch.stack((estimate,actual,diag,estimate*shrink,ratio),dim=1)
    return results

out={}
with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
    for route in routes:
        for scene in scenes:
            items=((route,scene,raw[i],rx[i],ids[i]) for i in chosen)
            values=list(pool.map(trace_row,items,chunksize=6))
            y,n=map(lambda rows:tensor(np.stack(rows),'cpu'),zip(*values,strict=True))
            arrays=gather(y,n)
            data={}
            for band,measurements in arrays.items():
                e,a,d,m,r=measurements.T
                data[band]=dict(proxy_over_actual_mean=float(e.mean()/a.mean()),
                    contrast_over_actual_mean=float(m.mean()/a.mean()),
                    actual_over_independent_mean=float(a.mean()/d.mean()),
                    proxy_abs_log_error_q=(e/a.clamp_min(1e-12)).clamp_min(1e-8).log().abs().quantile(prob).tolist(),
                    contrast_abs_log_error_q=(m/a.clamp_min(1e-12)).clamp_min(1e-8).log().abs().quantile(prob).tolist(),
                    measured_contrast_multiplier=r.quantile(prob).tolist())
            out.setdefault(route,{})[scene]=data
print(json.dumps(dict(packets=len(chosen), source_role='held_U_s', trace_is_diagnostic_only=True,
                      target_reads=0,V_reads=0,F_reads=0,TX_reads=0,
                      quantiles=[.1,.5,.9],calibration_packets=500,band_floor=floor.tolist(),by_route=out)),flush=True)
