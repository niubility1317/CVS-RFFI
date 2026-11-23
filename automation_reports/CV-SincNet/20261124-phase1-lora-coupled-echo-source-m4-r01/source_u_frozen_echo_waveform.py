from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.coupled_echo_source import RUN, RECIPES
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import aligned_view, chirp_windows
from experiments.pamf_lora_main.runner import activate

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
rt = read(project/'releases'/f'{RUN}-release'/'runtime.json')
activate(rt)
torch.set_num_threads(2)
source = Path(rt['source_root'])
raw,rx,ids,_ = source_role(source,source_contract(source),'U_s')
_,_,held = partition(rx)
chosen = np.concatenate([rows[:30] for rows in np.split(held,len(SOURCE))])
scenes = ('clean', VIEWS[3], VIEWS[4], VIEWS[-1])
routes = ('full','residual')

root=Path(rt['run_root'])
models,cutoffs=zip(*(load_model(root,recipe,source_contract(source),'cpu')
                    for recipe in RECIPES.values()),strict=True)
assert len(set(cutoffs))==1
linked,echo=models
tracker=PacketRestorer('consensus').eval()
frequency=(256*torch.fft.fftfreq(256)).round().abs()
bands={'1_to_8':(frequency>0)&(frequency<=8),'9_to_32':(frequency>8)&(frequency<=32),
       '33_to_64':(frequency>32)&(frequency<=64),'above_64':frequency>64}

def wave_row(payload):
    route,scene,raw,receiver,key=payload
    clean=chirp_windows(raw)
    seen=clean if scene=='clean' else aligned_view(raw,scene,receiver,key,'source-U',processing_route=route)
    return seen,clean

def spectrum(values):
    return torch.fft.fft(values,norm='ortho')

def components(values):
    common=values.mean(1,keepdim=True)
    return {'common':common,'private':values-common}

def iq_pair(seen,clean):
    y,p,q,r=coordinates(tracker,seen)
    base=complex_iq(clean)/(p*r)
    carrier=base.mean(-1,keepdim=True)
    reliable=carrier.abs()>=.1
    aligned=base*y.mean(-1,keepdim=True)*carrier.conj()/carrier.abs().square().clamp_min(.01)
    active=corruption_score(y)>cutoffs[0]
    estimates={name:y.clone() for name in ('received','linked_half','echo_half','echo_off_half')}
    if active.any():
        chunk=(y[active],q[active],p[active])
        _,a=linked(*chunk)
        _,b,_,_,c,_=echo(*chunk,controls=True)
        for name,output in zip(('linked_half','echo_half','echo_off_half'),(a,b,c),strict=True):
            estimates[name][active]=torch.lerp(y[active],output,.5)
    return estimates,aligned,reliable,active

out={}
with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
    for route in routes:
        for scene in scenes:
            fields=((route,scene,raw[i],rx[i],ids[i]) for i in chosen)
            values=list(pool.map(wave_row,fields,chunksize=6))
            observed,clean=map(lambda x:tensor(np.stack(x),'cpu'),zip(*values,strict=True))
            arrays=[iq_pair(x,z) for x,z in zip(observed.split(15),clean.split(15),strict=True)]
            cut=torch.cat([a[3] for a in arrays])
            # The common/private decomposition needs the same six-chirp support in each variant.
            keep=torch.cat([a[2].flatten(1).all(1) for a in arrays])
            assert int(keep.sum())>100
            target=components(spectrum(torch.cat([a[1] for a in arrays])[keep]))
            details={}
            for name in arrays[0][0]:
                restore=components(spectrum(torch.cat([a[0][name] for a in arrays])[keep]))
                for part in target:
                    for band,mask in bands.items():
                        x,y=restore[part][...,mask],target[part][...,mask]
                        denom=y.abs().square().sum((1,2))
                        cross=(x*y.conj()).sum((1,2))
                        err=(x-y).abs().square().sum((1,2))
                        details.setdefault(part,{}).setdefault(band,{})[name]=dict(
                            relative_error=float(err.sum()/denom.sum()),
                            median_real_gain=float((cross.real/denom.clamp_min(1e-10)).median()),
                            median_absolute_phase_rad=float(torch.atan2(cross.imag,cross.real).abs().median()))
            out.setdefault(route,{})[scene]=dict(active=int(cut.sum()),packets_complete_support=int(keep.sum()),
                                                 detail=details)
print(json.dumps(dict(packets=len(chosen),threshold=cutoffs[0],source_role='held_U_s',
                      F_reads=0,TX_reads=0,V_reads=0,target_reads=0,trace_reads=0,
                      comparison_coordinate='received carrier per chirp',by_route=out)),flush=True)
