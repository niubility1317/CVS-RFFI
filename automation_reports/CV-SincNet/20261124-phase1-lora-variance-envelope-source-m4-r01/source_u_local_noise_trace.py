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

def gather(obs, known_noise):
    y, rotation, quality, rms = coordinates(tracker, obs)
    weights = common_weights(quality)
    variance = observed_noise_rms(y).square()
    # Center each chirp independently before measuring the leave-one-out residual.
    s = y - y.mean(-1,keepdim=True)
    weighted = (weights*s).sum(1,keepdim=True)
    complement = (1-weights).clamp_min(1e-5)
    peer = (weighted - weights*s)/complement
    pool_variance = (weights.square()*variance).sum(1,keepdim=True)
    expected = variance + (pool_variance-weights.square()*variance)/complement.square()
    contrast = (s-peer).abs().square().unflatten(-1,(8,32)).mean(-1)
    proposed = variance * (contrast/expected.clamp_min(1e-9)).clamp(.25,4.).sqrt()
    noise = complex_iq(known_noise)/(rotation*rms)
    actual = noise.abs().square().unflatten(-1,(8,32)).mean(-1)
    actual_fraction = actual/actual.sum(-1,keepdim=True).clamp_min(1e-9)
    oracle = actual.topk(2,dim=-1).indices
    picked = proposed.topk(2,dim=-1).indices
    actual_log = actual.clamp_min(1e-10).log()
    proposal_log = proposed.clamp_min(1e-10).log()
    a=actual_log - actual_log.mean(-1,keepdim=True)
    b=proposal_log - proposal_log.mean(-1,keepdim=True)
    correlation = (a*b).sum(-1) / (a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-7)
    mixed_error = (proposal_log-actual_log).abs()
    original_error = (variance.clamp_min(1e-10).log()-actual_log).abs()
    return dict(window_energy_picked=actual_fraction.gather(-1,picked).sum(-1).flatten(),
                window_energy_oracle=actual_fraction.gather(-1,oracle).sum(-1).flatten(),
                within_chirp_log_power_correlation=correlation.flatten(),
                local_abs_log_error=mixed_error.flatten(),
                outer_abs_log_error=original_error.flatten())

out={}
with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
    for route in routes:
        for scene in scenes:
            items=((route,scene,raw[i],rx[i],ids[i]) for i in chosen)
            values=list(pool.map(trace_row,items,chunksize=6))
            y,n=map(lambda rows:tensor(np.stack(rows),'cpu'),zip(*values,strict=True))
            arrays=gather(y,n)
            out.setdefault(route,{})[scene] = {
                name: dict(mean=float(values.mean()),quantiles=values.quantile(prob).tolist())
                for name,values in arrays.items()
            }
print(json.dumps(dict(packets=len(chosen), source_role='held_U_s', trace_is_diagnostic_only=True,
                      target_reads=0,V_reads=0,F_reads=0,TX_reads=0,
                      quantiles=[.1,.5,.9],by_route=out)),flush=True)
