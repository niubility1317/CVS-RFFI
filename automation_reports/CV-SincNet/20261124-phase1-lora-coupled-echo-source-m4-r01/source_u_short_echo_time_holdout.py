from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.coupled_echo_source import RUN
from experiments.pamf_lora_main.coupled_echo import short_echo_columns
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.native_detail import native_pair
from experiments.pamf_lora_main.packet_physics import residual_view, chirp_windows
from experiments.pamf_lora_main.runner import activate

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
    return observed,noise,chirp_windows(raw)

frequency = (256 * torch.fft.fftfreq(256)).round().abs()
bins=torch.nonzero((frequency>0)&(frequency<=32)).flatten()
low=frequency[bins] <= 8
bands={'1_to_8':low,'9_to_32':~low}
tracker=PacketRestorer('consensus').eval()

def fit_columns(columns, target, evaluate=None):
    matrix = columns.flatten(1,-2)
    y = target.flatten(1)
    m = matrix.mH @ matrix
    diag = m.diagonal(dim1=-2,dim2=-1).real.mean(-1)
    safe = 1e-6 * diag.clamp_min(1e-5)
    eye = torch.eye(m.shape[-1],dtype=m.dtype)
    solved=torch.linalg.solve_ex(m+safe[:,None,None]*eye,matrix.mH@y[...,None]).result
    if evaluate is None:
        evaluate = columns
    prediction=evaluate.flatten(1,-2)@solved
    return prediction.squeeze(-1).reshape(evaluate.shape[:-1])

def gather(obs, known_noise, raw_clean):
    y, template, quality, rms=coordinates(tracker,obs)
    basis=template*rms
    signal=y-complex_iq(known_noise)/basis
    clean=complex_iq(raw_clean)/basis
    template_carrier=clean.mean(-1,keepdim=True)
    support=template_carrier.abs() >= .1
    ratio=signal.mean(-1,keepdim=True)*template_carrier.conj()/template_carrier.abs().square().clamp_min(.01)
    compressed=torch.fft.fft(signal-ratio*clean,norm='ortho').index_select(-1,bins)
    compressed=compressed*support
    columns=short_echo_columns(y,template,bins,y.real.new_ones(8))*support[...,None]
    answers={}
    for name,mask in bands.items():
        x=torch.zeros_like(signal)
        x.index_copy_(-1,bins[mask],compressed[...,mask])
        y=torch.fft.ifft(x,norm='ortho')
        design=columns.new_zeros((len(signal),signal.shape[1],signal.shape[-1],columns.shape[-1]))
        design.index_copy_(2,bins[mask],columns[...,mask,:])
        design=torch.fft.ifft(design,dim=2,norm='ortho')
        parity=(torch.arange(y.shape[-1])//32)%2
        methods={}
        for mode in ('shared','per_chirp'):
            predicted=torch.zeros_like(y)
            for fold in (0,1):
                fit=parity==fold
                observe=~fit
                source=y[...,fit]
                train=design[...,fit,:]
                test=design[...,observe,:]
                if mode=='shared':
                    output=fit_columns(train,source,evaluate=test)
                else:
                    output=fit_columns(train.flatten(0,1).unsqueeze(1),
                                       source.flatten(0,1).unsqueeze(1),
                                       evaluate=test.flatten(0,1).unsqueeze(1))
                    output=output.reshape_as(y[...,observe])
                predicted[...,observe]=output
            methods[mode]=((y-predicted).abs().square().sum((1,2)),y.abs().square().sum((1,2)))
        answers[name]=methods
    return answers,int(support.sum())

out={}
with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
    for route in routes:
        for scene in scenes:
            items=((route,scene,raw[i],rx[i],ids[i]) for i in chosen)
            values=list(pool.map(trace_row,items,chunksize=6))
            y,n,x=map(lambda rows:tensor(np.stack(rows),'cpu'),zip(*values,strict=True))
            arrays,supported=gather(y,n,x)
            data={}
            for band,measurements in arrays.items():
                data[band]={name:dict(unexplained_energy_fraction=float(rem.sum()/base.sum()),
                                      packet_p50=float((rem/base.clamp_min(1e-12)).median()))
                            for name,(rem,base) in measurements.items()}
            out.setdefault(route,{})[scene]=dict(supported_chirps=supported,bands=data)
print(json.dumps(dict(packets=len(chosen), source_role='held_U_s', trace_is_diagnostic_only=True,
                      target_reads=0,V_reads=0,F_reads=0,TX_reads=0,
                      min_clean_carrier=.1,validation='block parity in band-projected time coordinates',by_route=out)),flush=True)
