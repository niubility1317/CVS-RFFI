"""Read out the original LoRa DURA soft solve with isolated source-only oracle swaps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import Action, soft_map
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from experiments.pamf_lora_main.full_dura_action_probe import FIT_PROBES, fit_action, frozen_action, source_probes
from experiments.pamf_lora_main.full_packet_views import source_role
from experiments.pamf_lora_main.full_supervised import construct, handles
from experiments.pamf_lora_main.runner import activate


PER_RECEIVER=8
MODES=('direct','learned','known_action','known_action_and_variance')
SEED=2026092702


def complex_tensor(value):
    return torch.complex(tensor(value.real.astype(np.float32)),tensor(value.imag.astype(np.float32)))


def as_iq(source):
    return torch.stack((source.real,source.imag),dim=1)


@torch.no_grad()
def solve(model,actual,probes):
    x,response,clean,noise_free,noise=probes
    signal_scale=np.sqrt(np.mean(np.abs(clean)**2))
    observed_scale=np.sqrt(np.mean(np.abs(noise_free+noise)**2))
    coeff=fit_action(x[:FIT_PROBES],response[:FIT_PROBES],mirror=False)[...,0]
    main=complex_tensor(coeff*signal_scale/observed_scale)
    kernel=torch.zeros((len(actual),64),dtype=main.dtype)
    kernel[:,0]=1.
    known=Action(kernel,main,torch.zeros_like(main))
    observed=tensor(actual)
    y=torch.complex(observed[:,0],observed[:,1])
    prior,estimated,estimated_variance=model.estimate(observed)
    prior=torch.complex(prior[:,0],prior[:,1])
    measured_variance=.5*np.mean(np.abs(noise/observed_scale)**2,axis=-1)
    measured_variance=tensor(np.clip(measured_variance,1e-8,1.).astype(np.float32))
    modes=((estimated,estimated_variance),(known,estimated_variance),(known,measured_variance))
    values=[as_iq(soft_map(operator,y,prior,var,SCALE**2,6)) for operator,var in modes]
    outputs=torch.stack((observed,*values))
    if not torch.isfinite(outputs).all():
        raise FloatingPointError('Nonfinite oracle source solve')
    return outputs


@torch.no_grad()
def audit(rt):
    source,cache,contract=handles(rt)
    wave,receivers,ids,_=source_role(source,contract,'V')
    data=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    g=frozen_action(rt,contract)
    f=construct('cpu').eval().requires_grad_(False)
    f_path=complete_f(Path(rt['run_root']).parent/SET_RUN,contract,case(SEED,SET_METHOD))/'model.pt'
    f.load_state_dict(torch.load(f_path,map_location='cpu',weights_only=True),strict=True)
    rng=np.random.default_rng(2026112249)
    selected={rx:rng.choice(np.flatnonzero(receivers==rx),PER_RECEIVER,replace=False) for rx in SOURCE}
    predicted={}
    for view,scene in enumerate(VIEWS[1:],1):
        outcomes=[]
        for rx,indices in selected.items():
            packets=[]
            for i in indices:
                actual=data[view,i]
                probes=source_probes(wave[i],rx,ids[i],scene,actual)
                packets.append(solve(g,actual,probes))
            batch=torch.stack(packets).flatten(0,1)
            logits=f(batch)[0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite frozen F output on source probes')
            outcomes.extend(logits.argmax(1).reshape(PER_RECEIVER,len(MODES)).tolist())
        predicted[scene]=np.asarray(outcomes)

    # Access only visible source V labels, after the complete readout is fixed.
    with np.load(source/'V_metadata.npz',allow_pickle=False) as meta:
        truth=meta['y'][np.concatenate(list(selected.values()))]
    rows={}
    for scene,p in predicted.items():
        correct=p==truth[:,None]
        before=correct[:,0]
        rows[scene]={
            mode:dict(correct=int(c.sum()),gained=int((c&~before).sum()),lost=int((before&~c).sum()))
            for mode,c in zip(MODES,correct.T,strict=True)}
    return dict(status='SOURCE_SOFT_SOLVE_ORACLE_COMPLETE',stage='action',seed=SEED,source_packets=len(truth),
                frozen_G_updates=0,frozen_F_updates=0,target_reads=0,
                exact_noisy_input_replay=True,oracle_fitted_from_source_simulator_only=True,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source_config',type=Path)
    args=p.parse_args()
    runtime=read(args.source_config)
    if set(runtime)&{'truth','target_views','target_truth','F_checkpoint'}:
        raise ValueError('Expected source-only worker configuration')
    activate(runtime)
    print(json.dumps(audit(runtime),allow_nan=False),flush=True)
