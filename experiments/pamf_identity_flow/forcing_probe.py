"""Compare oracle interpolation and free-running source paths, no TX/target."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import IdentityConditionalFlow,flow_path
from .training import FixedAntitheticFlow
from .trajectory_probe import source_inputs,PARENT,PARENT_CODE
from experiments.pamf_dura_direct_train import runner as base

RUN='20261101-diagnostic-identity-flow-forcing-s2026092701-r01'
TIMES=(0.,.01,.05,.1,.25,.5,.75,.9,1.)
SCALES=(0.,.005,.02,.04)


@torch.no_grad()
def teacher_forced_metrics(model,clean,received,noise,time):
    t=clean.new_full((len(clean),),time);state,target=flow_path(clean,received,t,noise,.02)
    velocity=model.velocity(state,received,t);energy=clean.square().mean((1,2)).clamp_min(1e-4)
    loss=(velocity-target).square().mean((1,2))/energy
    endpoint=state+(1-time)*velocity;endpoint_loss=(endpoint-clean).square().mean((1,2))/energy
    pred_power=velocity.square().mean((1,2))/energy;target_power=target.square().mean((1,2))/energy
    if any(not torch.isfinite(x).all() for x in (loss,endpoint_loss,pred_power,target_power)):raise ValueError('Finite oracle-path diagnostic required')
    return dict(count=len(clean),velocity_NMSE_mean=float(loss.mean()),projected_endpoint_NMSE_mean=float(endpoint_loss.mean()),
        predicted_velocity_power=float(pred_power.mean()),target_velocity_power=float(target_power.mean()))


@torch.no_grad()
def noise_metrics(model,clean,received,template,scale):
    noise=template.expand_as(received)
    a=model.integrate(received,noise,noise_scale=scale,steps=4)
    b=model.integrate(received,-noise,noise_scale=scale,steps=4)
    out=.5*(a+b);energy=clean.square().mean((1,2)).clamp_min(1e-4)
    nmse=(out-clean).square().mean((1,2))/energy
    difference=(a-b).square().mean()
    if not torch.isfinite(nmse).all() or not torch.isfinite(difference):raise ValueError('Finite source rollout required')
    return dict(count=len(clean),mean_path_NMSE=float(nmse.mean()),opposite_path_difference_MSE=float(difference))


def run(root,runtime,commit,output,device):
    if output.exists():raise FileExistsError('New immutable source diagnostic required')
    parent=root/'runs'/PARENT;done=base.read_json(parent/'complete.json');summary=base.read_json(parent/'identity_flow/training_complete.json')
    if done.get('status')!='ANALYZED' or (summary.get('status'),summary.get('epoch'),summary.get('updates'),summary.get('F0_frozen'))!=('SOURCE_TRAINED',20,9860,True):
        raise ValueError('Fixed completed source E20 model required')
    if any(summary.get(k)!=0 for k in ('TX_labels_read','V_IQ_reads','OE_IQ_reads','target_IQ_reads','F0_updates')):
        raise ValueError('Source-only parent provenance required')
    checkpoint=parent/'identity_flow/final.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in dict(schema='identity_flow_source_v1',run_id=PARENT,code_commit=PARENT_CODE,epoch=20,updates=9860).items()):
        raise ValueError('Exact parent checkpoint required')
    model=IdentityConditionalFlow();model.load_state_dict(payload['model'],strict=True);model=model.to(device).eval().requires_grad_(False)
    p,clean,roles,g_clean,g_received,teacher,identity=source_inputs(root,runtime,device)
    if identity!=summary['F0']:raise ValueError('Exact frozen Phase3 F0 required')
    received=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    if received.shape!=(63000,2,256) or received.dtype!=np.float32:raise ValueError('Fixed source view required')
    select={role:np.flatnonzero(roles==role)[:128] for role in ('L_s','U_s')}
    if any(len(v)!=128 for v in select.values()):raise ValueError('Fixed source groups required')
    groups={'R07_L_s':(clean[select['L_s']],received[select['L_s']]),'R07_U_s':(clean[select['U_s']],received[select['U_s']]),
        'G10_L_s':(g_clean,g_received),'clean_L_s':(clean[select['L_s']],clean[select['L_s']])}
    template=FixedAntitheticFlow(IdentityConditionalFlow()).noise_template.to(device)
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (model,teacher)]
    output.mkdir(parents=True)
    a=base.tensor(g_clean[:2],device);b=base.tensor(g_received[:2],device);n=template.expand_as(b)
    smoke=teacher_forced_metrics(model,a,b,n,0.)
    with torch.no_grad():features=teacher.features(a)
    if len(features)!=2 or not torch.isfinite(features).all():raise ValueError('Actual F0 source checkpoint smoke failed')
    base.write_json(output/'smoke.json',dict(status='PASS',actual_E20=True,actual_F0=True,source_G10_L_s=2,
        oracle_path_smoke=smoke,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0))
    records={}
    for name,(x,y) in groups.items():
        x=base.tensor(x,device);y=base.tensor(y,device);noise=template.expand_as(y)
        records[name]=dict(oracle_interpolation={str(t):teacher_forced_metrics(model,x,y,noise,t) for t in TIMES},
            free_rollout={str(scale):noise_metrics(model,x,y,template,scale) for scale in SCALES})
        print('SOURCE_FORCING_COMPLETE',name,flush=True)
    for model_,snapshot in zip((model,teacher),snapshots):
        if any(not torch.equal(v,snapshot[k]) for k,v in model_.state_dict().items()):raise ValueError('Frozen state changed')
    base.write_json(output/'complete.json',dict(status='SOURCE_FORCING_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=commit,parent_run=PARENT,
        parent_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),F0=identity,groups=records,times=list(TIMES),noise_scales=list(SCALES),
        sample_count_per_group=128,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0,optimizer_updates=0,parameters_unchanged=True,
        scope='Training-source only. Oracle interpolation contains paired clean and is not available at deployment; source timing/noise sensitivity is not target model selection.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reference-runtime',type=Path,required=True)
    parser.add_argument('--code-commit',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(args.root,args.reference_runtime,args.code_commit,args.root/'runs'/RUN,args.device)


if __name__=='__main__':main()
