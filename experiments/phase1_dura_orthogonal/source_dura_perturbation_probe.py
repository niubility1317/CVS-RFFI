"""Source-only finite differences through estimated physical action and DURA."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs

EPSILON=.01
DIRECTIONS=('conjugate','amplitude_cubic','phase_slope','local_difference')


def perturbations(clean):
    if clean.ndim!=3 or clean.shape[1:]!=(2,256):
        raise ValueError('Source IQ windows required')
    z=torch.complex(clean[:,0],clean[:,1])
    power=z.abs().square()
    meanpower=power.mean(1,keepdim=True).clamp_min(1e-12)
    t=torch.linspace(-1,1,256,device=z.device,dtype=z.real.dtype)[None]
    directions=(z.conj(),z*(power/meanpower-1),1j*z*t,torch.roll(z,1,-1)-z)
    normalized=[]
    for d in directions:
        norm=d.abs().square().mean(1,keepdim=True).sqrt()
        if (norm<1e-10).any() or not torch.isfinite(norm).all():
            raise ValueError('Nonzero finite source perturbation direction required')
        normalized.append(EPSILON*meanpower.sqrt()*d/norm)
    return normalized


def response_metrics(delta, response):
    if delta.shape!=response.shape or not torch.is_complex(delta) or not torch.is_complex(response):
        raise ValueError('Matching complex perturbations required')
    energy=delta.abs().square().sum(-1).clamp_min(1e-20)
    gain=(delta.conj()*response).real.sum(-1)/energy
    error=(response-delta).abs().square().sum(-1)/energy
    relative_norm=(response.abs().square().sum(-1)/energy).sqrt()
    return torch.stack((gain,error,relative_norm),1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source-only output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    dura=frozen_dura(args.root,'cpu')
    chunks=[]
    started=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),16):
            selected=ids[start:start+16]
            x,y=core.tensor(received[selected],'cpu'),core.tensor(clean[selected],'cpu')
            context=dura.estimate_context(x)
            anchor=dura.stage_outputs(x,context)[0][-1]
            if start==0:
                torch.testing.assert_close(anchor,dura(x),rtol=1e-5,atol=3e-6)
                args.out.mkdir(parents=True)
                print('DURA_PERTURBATION_SOURCE_SMOKE_PASS',flush=True)
            a=context[1]
            values=[]
            for delta in perturbations(y):
                observed=a(delta)
                changed=x+torch.stack((observed.real,observed.imag),1)
                output=dura(changed)
                response=torch.complex(output[:,0]-anchor[:,0],output[:,1]-anchor[:,1])
                values.append(torch.stack((response_metrics(delta,observed),response_metrics(delta,response)),1))
            result=torch.stack(values,1)
            if not torch.isfinite(result).all():
                raise FloatingPointError('Finite perturbation response required')
            chunks.append(result.cpu())
    values=torch.cat(chunks)
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=np.isin(rx,receivers)&(scene==view)
            if mask.sum()!=48:
                raise ValueError('Complete 48-record source groups required')
            cell=values[torch.tensor(mask.tolist(),dtype=torch.bool)]
            groups[name+'/'+view]=dict(count=len(cell),mean_metrics=cell.mean(0).tolist(),
                negative_dura_gain=(cell[:,:,1,0]<0).sum(0).tolist())
    core.write(args.out/'result.json',dict(status='DURA_PERTURBATION_SOURCE_COMPLETE',commit=args.commit,
        records=len(ids),directions=list(DIRECTIONS),epsilon_relative_rms=EPSILON,
        route_order=['estimated_action_only','estimated_action_then_reestimated_dura'],
        metric_order=['real_projection_gain','relative_squared_response_error','response_norm_ratio'],
        groups=groups,elapsed_seconds=time.monotonic()-started,TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Artificial clean-IQ perturbations and DURA-estimated action are source-only proxies, not actual transmitter identity, target performance, or a universal bound.'))
    print('DURA_PERTURBATION_SOURCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
