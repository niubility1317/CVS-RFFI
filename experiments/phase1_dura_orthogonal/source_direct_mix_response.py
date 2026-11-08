"""Frozen source-IQ perturbation response for DURA/direct mixtures."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.signal_mix_gate_ground import load_gate as load_adaptive
from experiments.phase1_dura_orthogonal.signal_balanced_mix_ground import load_gate as load_balanced
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS, EPSILON, perturbations, response_metrics
from experiments.phase1_dura_orthogonal.train_source import extract_context

ADAPTIVE='20261108-phase1-dura-direct-adaptive-mix-source-s2026092701-r01'
ADAPTIVE_COMMIT='708a3d8a9b1ce57907de5a26f9b6bbf6ba9553ab'
BALANCED='20261108-phase1-dura-direct-balanced-mix-source-s2026092701-r02'
BALANCED_COMMIT='597b621aa76558c4ba6fdea3487c7f89eb904639'
METHODS=('dura','direct','half','adaptive','balanced')


def outputs(x,context,direct,gates):
    anchor,direction,variance=context(x)
    proposal=direct(x)
    return (anchor,proposal,(anchor+proposal)*.5,*(g(x,anchor,direction,proposal,variance) for g in gates))


def evaluate(dura,direct,gates,x,clean):
    context=lambda v:extract_context(dura,v)
    proposal=lambda v:direct(v,validate=False)
    action=dura.estimate_context(x)[1]
    originals=outputs(x,context,proposal,gates)
    energy=clean.square().mean((1,2)).clamp_min(1e-8)
    nmse=torch.stack([(z-clean).square().mean((1,2))/energy for z in originals],1)
    metrics=[]
    for delta in perturbations(clean):
        observed=action(delta)
        changed=x+torch.stack((observed.real,observed.imag),1)
        revised=outputs(changed,context,proposal,gates)
        row=[]
        for a,b in zip(originals,revised):
            response=b-a
            row.append(response_metrics(delta,torch.complex(response[:,0],response[:,1])))
        metrics.append(torch.stack(row,1))
    result=torch.stack(metrics,1)
    if not torch.isfinite(nmse).all() or not torch.isfinite(result).all():raise FloatingPointError('Finite response metrics required')
    return result,nmse


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
    for source in (ADAPTIVE,BALANCED):
        state=torch.load(args.root/'runs'/source/'final.pt',map_location='cpu',weights_only=False)
        if np.intersect1d(ids,np.asarray(state['fit_ids'])).size:raise ValueError('Source pilot must be gate-disjoint')
    direct=load_direct(args.root,ids)
    gates=(load_adaptive(args.root,ADAPTIVE,ADAPTIVE_COMMIT),load_balanced(args.root,BALANCED,BALANCED_COMMIT))
    dura=frozen_dura(args.root,'cpu')
    positions=[]
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS:
            found=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(found)!=48:raise ValueError('Complete source group required')
            positions.extend(found[:8].tolist())
    positions=np.asarray(positions,dtype=np.int64)
    if len(positions)!=112 or len(np.unique(positions))!=112:raise ValueError('112 unique source observations required')
    chunks=[];errors=[]
    with torch.no_grad():
        for start in range(0,len(positions),8):
            sel=ids[positions[start:start+8]]
            x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
            result,nmse=evaluate(dura,direct,gates,x,y)
            if start==0:
                torch.testing.assert_close(extract_context(dura,x)[0],dura(x),rtol=1e-5,atol=3e-6)
                if result.shape!=(8,4,5,3) or nmse.shape!=(8,5):raise ValueError('Complete source response smoke required')
                print('DIRECT_MIX_RESPONSE_SMOKE_PASS',len(x),flush=True)
            chunks.append(result.cpu());errors.append(nmse.cpu())
    values=torch.cat(chunks);errors=torch.cat(errors)
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=np.isin(rx[positions],receivers)&(scene[positions]==view)
            if mask.sum()!=8:raise ValueError('Eight source records per group required')
            ix=torch.tensor(mask.tolist(),dtype=torch.bool)
            groups[name+'/'+view]=dict(count=8,mean_metrics=values[ix].mean(0).tolist(),mean_nmse=errors[ix].mean(0).tolist())
    args.out.mkdir(parents=True)
    payload=dict(status='DIRECT_MIX_RESPONSE_COMPLETE',commit=args.commit,records=len(positions),
        directions=list(DIRECTIONS),epsilon_relative_rms=EPSILON,methods=list(METHODS),
        metric_order=['real_projection_gain','relative_squared_response_error','response_norm_ratio'],groups=groups,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='112 previously explored source records, eight per group; perturbations and estimated operator are proxies, not identity performance or target benefit.')
    (args.out/'result.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('DIRECT_MIX_RESPONSE_COMPLETE',len(positions),flush=True)


if __name__=='__main__':main()
