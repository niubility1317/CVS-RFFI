"""Source-only waveform complement of frozen DURA and a direct wide model."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.large_direct_wide_residual import WideResidualDirectTimeFreq
from experiments.phase1_dura_orthogonal.physical_line_source_probe import bounded_step

SOURCE='20261108-phase1-direct-wide-physics-source-s2026092701-r01'
SOURCE_COMMIT='1a10d53c0c35d1613300f0c0e413d96ccdd800e2'


def load_direct(root,pilot):
    state=torch.load(root/'runs'/SOURCE/'final.pt',map_location='cpu',weights_only=False)
    expected=dict(schema='phase1_dura_orthogonal_source_v1',commit=SOURCE_COMMIT,
        architecture='large_direct_timefreq_stable',objective='wide_large_direct_timefreq_stable_wide_physics',
        epochs=20,direct_timefreq_scale=8.,source_forward_physics_weight=.1,
        simulator_action_inference_reads=0,inference_received_iq_only=True,DURA_training_reads=0,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0)
    fit=np.asarray(state.get('fit_ids',[]))
    if (any(state.get(k)!=v for k,v in expected.items()) or len(fit)!=32768
            or np.intersect1d(fit,pilot).size):
        raise ValueError('Matching source-only direct final20 and disjoint pilot required')
    model=WideResidualDirectTimeFreq(width=state['direct_timefreq_width'],layers=state['direct_timefreq_layers'])
    model.load_state_dict(state['model'],strict=True)
    return model.eval().requires_grad_(False)


def mixture_metrics(received,anchor,direct,clean,action):
    def c(z):return torch.complex(z[:,0],z[:,1])
    x,a,p,y=map(c,(received,anchor,direct,clean))
    delta=p-a
    physical=bounded_step(x-action(a),action(delta))
    oracle=bounded_step(y-a,delta)
    energy=y.abs().square().sum(-1).clamp_min(1e-12)
    estimates=(a,p,a+.5*delta,a+physical[:,None]*delta,a+oracle[:,None]*delta)
    errors=[(v-y).abs().square().sum(-1)/energy for v in estimates]
    return torch.stack((*errors,physical,oracle),1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Unique source output required')
    torch.set_num_threads(2)
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    direct=load_direct(args.root,ids)
    dura=frozen_dura(args.root,'cpu')
    rows=[]
    with torch.no_grad():
        for start in range(0,len(ids),8):
            sel=ids[start:start+8]
            x,y=core.tensor(received[sel],'cpu'),core.tensor(clean[sel],'cpu')
            context=dura.estimate_context(x)
            stages,_,action,_=dura.stage_outputs(x,context)
            anchor=stages[-1]
            proposal=direct(x,validate=False)
            values=mixture_metrics(x,anchor,proposal,y,action)
            if not torch.isfinite(values).all():raise FloatingPointError('Finite source mixture metrics required')
            if start==0:
                torch.testing.assert_close(anchor,dura(x),rtol=1e-5,atol=3e-6)
                print('DIRECT_DURA_MIX_SMOKE_PASS',len(x),flush=True)
            rows.append(values.cpu())
    values=np.from_dlpack(torch.cat(rows).contiguous()).copy()
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=np.isin(rx,receivers)&(scene==view)
            cell=values[mask]
            if len(cell)!=48:raise ValueError('48 source records per group required')
            groups[name+'/'+view]=dict(count=len(cell),mean_metrics=cell.mean(0).tolist(),
                oracle_better_than_both=int(((cell[:,4]<cell[:,0]) & (cell[:,4]<cell[:,1])).sum()),
                physical_better_than_dura=int((cell[:,3]<cell[:,0]).sum()),
                physical_worse_than_dura=int((cell[:,3]>cell[:,0]).sum()))
    args.out.mkdir(parents=True)
    with (args.out/'values.npz').open('xb') as f:np.savez(f,ids=ids,rx=rx,scene=scene,values=values)
    result=dict(status='DIRECT_DURA_MIX_SOURCE_COMPLETE',commit=args.commit,source=SOURCE,source_commit=SOURCE_COMMIT,
        records=len(ids),columns=['dura_nmse','direct_nmse','half_mix_nmse','physical_line_nmse','clean_oracle_nmse','physical_alpha','oracle_alpha'],groups=groups,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Previously explored source pilot; clean oracle is unavailable at deployment. Waveform NMSE is not TX accuracy.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('DIRECT_DURA_MIX_SOURCE_COMPLETE',len(ids),flush=True)


if __name__=='__main__':main()
