"""Source-only initial output-head gradients of cross-expert response objectives."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.counterfactual_response import response_loss
from experiments.phase1_dura_orthogonal.signal_confidence_gate import source_gate_loss
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS,EPSILON,perturbations
from experiments.phase1_dura_orthogonal.train_source import extract_context

WEIGHT=.01
SEED=2026110830
VIEWS=('practical_high','practical_low_urban')


def geometry(base,auxiliary,weight=WEIGHT):
    b=base.norm().clamp_min(1e-20);a=auxiliary.norm().clamp_min(1e-20)
    cosine=(base@auxiliary)/(b*a);ratio=weight*a/b
    return dict(cosine=float(cosine),weighted_norm_ratio=float(ratio),
        base_dot_combined=float(1+ratio*cosine),aux_dot_combined=float(cosine+ratio))


def gradient(loss,parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=False)
    return torch.cat([v.detach().reshape(-1) for v in values])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    positions=[];labels=[]
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in VIEWS:
            found=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(found)!=48:raise ValueError('Complete source group required')
            positions.extend(found[:8].tolist());labels.append(name+'/'+view)
    positions=np.asarray(positions,dtype=np.int64)
    if len(positions)!=32 or len(np.unique(positions))!=32:raise ValueError('32 unique source observations required')
    direct=load_direct(args.root,ids)
    dura=frozen_dura(args.root,'cpu')
    torch.manual_seed(SEED)
    model=CrossExpertRefiner().train()
    params=tuple(model.time_out.parameters())+tuple(model.freq_out.parameters())
    base_sum=None;aux_sums=None;rows=[]
    for start in range(0,len(positions),8):
        sel=ids[positions[start:start+8]]
        x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
        with torch.no_grad():
            action=dura.estimate_context(x)[1]
            a,d,v=extract_context(dura,x)
            proposal=direct(x,validate=False)
            pairs=[]
            for delta in perturbations(y):
                observed=action(delta)
                changed=x+torch.stack((observed.real,observed.imag),1)
                ctx=extract_context(dura,changed)
                changed_proposal=direct(changed,validate=False)
                pairs.append((changed,ctx,changed_proposal,torch.stack((delta.real,delta.imag),1)))
        original=model(x,a,d,proposal,v)
        if start==0:
            torch.testing.assert_close(a,dura(x),rtol=1e-5,atol=3e-6)
            torch.testing.assert_close(original,(a+proposal)*.5,rtol=1e-5,atol=3e-6)
            print('CROSS_RESPONSE_GRADIENT_SMOKE_PASS',len(x),flush=True)
        base=source_gate_loss(original,a,y)
        bg=gradient(base,params)
        if base_sum is None:
            base_sum=torch.zeros_like(bg);aux_sums=[torch.zeros_like(bg) for _ in DIRECTIONS]
        base_sum+=bg/4
        losses=[]
        for i,(changed,(ca,cd,cv),cp,delta_iq) in enumerate(pairs):
            changed_output=model(changed,ca,cd,cp,cv)
            loss=response_loss(original,changed_output,delta_iq)
            aux_sums[i]+=gradient(loss,params)/4
            losses.append(float(loss.detach()))
        rows.append(dict(group=labels[start//8],source_count=len(x),wave_loss=float(base.detach()),response_losses=losses))
        print('CROSS_RESPONSE_GRADIENT_GROUP',labels[start//8],flush=True)
    if not torch.isfinite(base_sum).all() or any(not torch.isfinite(g).all() for g in aux_sums):raise FloatingPointError('Finite source gradients required')
    if any(p.grad is not None for obj in (dura,direct) for p in obj.parameters()):raise ValueError('Frozen experts must remain unchanged')
    args.out.mkdir(parents=True)
    payload=dict(status='CROSS_RESPONSE_GRADIENT_COMPLETE',commit=args.commit,source_records=32,
        model_seed=SEED,output_head_parameters=sum(p.numel() for p in params),
        directions=list(DIRECTIONS),epsilon_relative_rms=EPSILON,reference_weight=WEIGHT,
        base_gradient_norm=float(base_sum.norm()),directions_geometry={n:geometry(base_sum,g) for n,g in zip(DIRECTIONS,aux_sums)},
        batches=rows,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Output-head gradients at fresh initialization only, four groups of eight explored source records. Synthetic response is not TX identity or target benefit; no training checkpoint tested.')
    (args.out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('CROSS_RESPONSE_GRADIENT_COMPLETE',len(positions),flush=True)


if __name__=='__main__':main()
