"""Held-out source tradeoff for waveform and perturbation-response corrections."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS, perturbations

WEIGHTS=(0., .05, .5)
RIDGE=.001


def features(z):
    spectrum=torch.fft.fft(z,dim=-1,norm='ortho')
    reverse=(-torch.arange(z.shape[-1],device=z.device))%z.shape[-1]
    return torch.stack((spectrum,spectrum[:,reverse].conj()),-1)


def statistics(x,target,scale):
    f=(features(x)/scale[:,None,None]).permute(1,0,2)
    t=(torch.fft.fft(target,dim=-1,norm='ortho')/scale[:,None]).permute(1,0)
    return f.conj().transpose(1,2)@f, f.conj().transpose(1,2)@t.unsqueeze(-1)


def fit_weights(anchor,clean,delta,response,weight):
    if anchor.shape!=clean.shape or delta.shape!=response.shape or delta.shape[:1]!=anchor.shape[:1] or delta.shape[1]!=4:
        raise ValueError('Matched source waveform and four perturbation responses required')
    energy=clean.abs().square().mean(1).clamp_min(1e-12).sqrt()
    gram,rhs=statistics(anchor,clean-anchor,energy)
    if weight:
        for direction in range(4):
            scale=delta[:,direction].abs().square().mean(1).clamp_min(1e-12).sqrt()
            g,h=statistics(response[:,direction],delta[:,direction]-response[:,direction],scale)
            gram+=weight*g/4
            rhs+=weight*h/4
    scale=gram.diagonal(dim1=-2,dim2=-1).real.mean(-1).clamp_min(1e-12)
    eye=torch.eye(2,dtype=gram.dtype,device=gram.device)
    return torch.linalg.solve(gram+RIDGE*scale[:,None,None]*eye,rhs).squeeze(-1)


def apply(z,weights):
    correction=(features(z)*weights[None]).sum(-1)
    return z+torch.fft.ifft(correction,dim=-1,norm='ortho')


def measurements(anchor,clean,delta,response,weights=None):
    output=anchor if weights is None else apply(anchor,weights)
    response_out=response if weights is None else torch.stack([apply(response[:,j],weights) for j in range(4)],1)
    energy=clean.abs().square().mean(1).clamp_min(1e-12)
    nmse=(output-clean).abs().square().mean(1)/energy
    direction_energy=delta.abs().square().sum(-1).clamp_min(1e-20)
    response_error=(response_out-delta).abs().square().sum(-1)/direction_energy
    gain=(delta.conj()*response_out).real.sum(-1)/direction_energy
    return torch.cat((nmse[:,None],response_error,gain),1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    dura=frozen_dura(args.root,'cpu')
    values=[]
    started=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),16):
            selected=ids[start:start+16]
            x,y=core.tensor(received[selected],'cpu'),core.tensor(clean[selected],'cpu')
            context=dura.estimate_context(x)
            base=dura.stage_outputs(x,context)[0][-1]
            if start==0:
                torch.testing.assert_close(base,dura(x),rtol=1e-5,atol=3e-6)
                args.out.mkdir(parents=True)
                print('RESPONSE_TRADEOFF_SOURCE_SMOKE_PASS',flush=True)
            deltas=perturbations(y)
            responses=[]
            for delta in deltas:
                observed=context[1](delta)
                changed=x+torch.stack((observed.real,observed.imag),1)
                out=dura(changed)
                responses.append(torch.complex(out[:,0]-base[:,0],out[:,1]-base[:,1]))
            values.append((torch.complex(base[:,0],base[:,1]),torch.complex(y[:,0],y[:,1]),
                           torch.stack(deltas,1),torch.stack(responses,1)))
        anchor,clean_iq,delta,response=(torch.cat(v,0) for v in zip(*values))
        train=np.zeros(len(ids),dtype=bool)
        for receivers in ((4,6),(8,)):
            for view in core.VIEWS:
                pos=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
                if len(pos)!=48:
                    raise ValueError('Complete source group required')
                train[pos[::2]]=True
        if train.sum()!=336:
            raise ValueError('336 disjoint fit and holdout source records required')
        ix=torch.tensor(train.tolist(),dtype=torch.bool)
        models=[fit_weights(anchor[ix],clean_iq[ix],delta[ix],response[ix],w) for w in WEIGHTS]
        if any(not torch.isfinite(m).all() for m in models):
            raise FloatingPointError('Finite source correction required')
        measured=torch.stack([measurements(anchor,clean_iq,delta,response,w) for w in (None,*models)],1)
        if not torch.isfinite(measured).all():
            raise FloatingPointError('Finite held-out source metrics required')
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=(~train)&np.isin(rx,receivers)&(scene==view)
            if mask.sum()!=24:
                raise ValueError('24 held-out records per source group required')
            cell=measured[torch.tensor(mask.tolist(),dtype=torch.bool)]
            groups[name+'/'+view]=dict(count=len(cell),mean_metrics=cell.mean(0).tolist())
    core.write(args.out/'result.json',dict(status='RESPONSE_TRADEOFF_SOURCE_COMPLETE',commit=args.commit,
        records=672,fit_records=336,heldout_records=336,groups=groups,
        methods=['dura_baseline','wave_only','wave_plus_response_0.05','wave_plus_response_0.5'],
        directions=list(DIRECTIONS),metric_order=['wave_nmse','response_error_conjugate','response_error_amplitude_cubic',
            'response_error_phase_slope','response_error_local_difference','gain_conjugate','gain_amplitude_cubic',
            'gain_phase_slope','gain_local_difference'],ridge=RIDGE,weights=list(WEIGHTS),
        elapsed_seconds=time.monotonic()-started,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        V_reads=0,target_reads=0,model_updates=0,
        caveat='Source pilot held out only for these per-frequency diagnostic fits; DURA may have seen related source data. Synthetic perturbations use source clean and estimated action. No weights saved; not identity accuracy or deployment evidence.'))
    print('RESPONSE_TRADEOFF_SOURCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
