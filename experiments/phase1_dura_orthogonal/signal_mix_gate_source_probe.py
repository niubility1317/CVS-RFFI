"""Inspect frozen source-only confidence gate and paired source waveform error."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.signal_mix_gate_ground import load_gate
from experiments.phase1_dura_orthogonal.train_source import extract_context

SOURCE='20261108-phase1-dura-direct-adaptive-mix-source-s2026092701-r01'
SOURCE_COMMIT='708a3d8a9b1ce57907de5a26f9b6bbf6ba9553ab'


def validate_pilot_disjoint(fit,pilot):
    if np.intersect1d(np.asarray(fit),np.asarray(pilot)).size:
        raise ValueError('Source pilot must be disjoint from gate fit')


def alpha_and_active(anchor,proposal,output):
    delta=proposal-anchor
    q=delta.square().sum((1,2))
    energy=anchor.square().sum((1,2)).clamp_min(1e-12)
    active=q/energy > 1e-10
    alpha=((output-anchor)*delta).sum((1,2))/q.clamp_min(1e-20)
    return alpha,active


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source output required')
    torch.set_num_threads(2)
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    state=torch.load(args.root/'runs'/SOURCE/'final.pt',map_location='cpu',weights_only=False)
    validate_pilot_disjoint(state['fit_ids'],ids)
    del state
    parent=load_direct(args.root,ids)
    gate=load_gate(args.root,SOURCE,SOURCE_COMMIT)
    dura=frozen_dura(args.root,'cpu')
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            sel=ids[start:start+8]
            x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
            a,d,v=extract_context(dura,x)
            proposal=parent(x,validate=False)
            output=gate(x,a,d,proposal,v)
            alpha,active=alpha_and_active(a,proposal,output)
            energy=y.square().sum((1,2)).clamp_min(1e-12)
            errors=[(z-y).square().sum((1,2))/energy for z in (a,proposal,output)]
            values=torch.stack((*errors,alpha,active.to(alpha.dtype)),1)
            if not torch.isfinite(values).all():raise FloatingPointError('Finite source gate metrics required')
            if start==0:
                torch.testing.assert_close(a,dura(x),rtol=1e-5,atol=3e-6)
                print('SIGNAL_MIX_GATE_PROBE_SMOKE_PASS',len(x),flush=True)
            chunks.append(values.cpu())
    values=np.from_dlpack(torch.cat(chunks).contiguous()).copy()
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=np.isin(rx,receivers)&(scene==view)
            cell=values[mask]
            if len(cell)!=48:raise ValueError('48 records per source group required')
            active=cell[:,4]>0
            alphas=cell[active,3]
            groups[name+'/'+view]=dict(count=len(cell),mean_nmse=cell[:,:3].mean(0).tolist(),active=int(active.sum()),
                alpha_mean=float(alphas.mean()) if len(alphas) else None,
                alpha_ge_0999=int((alphas>=.999).sum()),alpha_ge_099=int((alphas>=.99).sum()),
                gate_better_than_parent=int((cell[:,2]<cell[:,1]).sum()),
                gate_worse_than_parent=int((cell[:,2]>cell[:,1]).sum()),
                gate_better_than_dura=int((cell[:,2]<cell[:,0]).sum()),
                gate_worse_than_dura=int((cell[:,2]>cell[:,0]).sum()))
    args.out.mkdir(parents=True)
    result=dict(status='SIGNAL_MIX_GATE_PROBE_COMPLETE',commit=args.commit,source=SOURCE,
        source_commit=SOURCE_COMMIT,records=len(ids),groups=groups,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Previously explored source pilot disjoint from gate and direct fits; recovered alpha excludes near-zero proposal changes. Waveform NMSE is not identity accuracy.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('SIGNAL_MIX_GATE_PROBE_COMPLETE',len(ids),flush=True)


if __name__=='__main__':main()
