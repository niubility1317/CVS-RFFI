"""Frozen E20 source-train path diagnostic; no labels, V or target IQ."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import IdentityConditionalFlow
from .training import FixedAntitheticFlow, align_g10
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_dura_direct_train.f_guided import source_ids
from experiments.pamf_phase3_open_world import star_relative_source as original_f0

RUN='20261101-diagnostic-identity-flow-source-paths-s2026092701-r01'
PARENT='20261101-phase3-identity-flow-source-E20-s2026092701-r01'
PARENT_CODE='77b93f1b1713880f1cb1901be34cc6dce474105c'
STEPS=(1,2,4,8,16)


def source_inputs(root,runtime,device):
    native=base.read_json(runtime)['eval_config']['reference_config']['source_control_config']['reference_config']['train_config']['native_config']
    spec=dict(reader_config={k:native[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=native['source_contract'])
    loaded=original_f0._backend(spec,device,with_dura=False)
    teacher=loaded['reader'].eval().requires_grad_(False);identity=loaded['reader_identity']
    if (identity['parameters'],identity['epoch'],identity['updates'])!=(232993,50,11100):raise ValueError('Original F0 required')
    p=base.paths(root);clean,roles=base.source_training(p);ids=source_ids(p,roles);contract=original_f0._contract(spec)
    for role in ('L_s','U_s'):
        if set(ids[roles==role].tolist())!=set(contract['role_ids'][role]):raise ValueError('Source roles differ from F0 contract')
    cache=root/'runs/20261029-phase3-star-representation-s2026092701-r01/cache'
    done=base.read_json(cache/'complete.json');manifest=base.read_json(cache/'L/manifest.json')
    if (done.get('status')!='STAR_REPRESENTATION_CACHE_COMPLETE' or not done.get('fixed_G10_reconstruction')
            or done.get('parent_F0_identity')!=identity or manifest.get('status')!='STAR_REPRESENTATION_ROLE_COMPLETE'
            or manifest.get('role')!='L' or manifest.get('count')!=6300 or manifest.get('parent_F0_identity')!=identity):
        raise ValueError('Matching complete G10 source L_s required')
    indices=align_g10(ids,roles,np.load(cache/'L/ids.npy',allow_pickle=False))
    received=np.load(cache/'L/received.npy',mmap_mode='r',allow_pickle=False)
    if received.shape!=(6300,2,256) or received.dtype!=np.float32:raise ValueError('Complete G10 source L_s required')
    return p,clean,roles,clean[indices[:128]],received[:128],teacher,identity


@torch.no_grad()
def path_metrics(model,teacher,clean,received,template,steps):
    noise=template.expand_as(received)
    a=model.integrate(received,noise,noise_scale=.02,steps=steps)
    b=model.integrate(received,-noise,noise_scale=.02,steps=steps)
    out=.5*(a+b);energy=clean.square().mean((1,2)).clamp_min(1e-4)
    nmse=(out-clean).square().mean((1,2))/energy
    ref=teacher.features(clean);restored=teacher.features(out)
    cosine=(F.normalize(ref,dim=1)*F.normalize(restored,dim=1)).sum(1)
    c0=teacher.classify_features(ref).argmax(1);c1=teacher.classify_features(restored).argmax(1)
    if not all(torch.isfinite(v).all() for v in (out,nmse,cosine)):raise ValueError('Finite source path values required')
    return dict(count=len(clean),IQ_NMSE_mean=float(nmse.mean()),IQ_NMSE_median=float(nmse.median()),
        feature_cosine_mean=float(cosine.mean()),F0_argmax_agreement=float((c0==c1).float().mean()),
        output_to_clean_power_ratio=float((out.square().mean((1,2))/energy).mean()),
        antithetic_IQ_difference_MSE=float((a-b).square().mean()))


def run(root,runtime,commit,output,device):
    if output.exists():raise FileExistsError('Immutable diagnostic output required')
    parent=root/'runs'/PARENT
    if base.read_json(parent/'complete.json').get('status')!='ANALYZED':raise ValueError('Fully scored E20 source parent required')
    summary=base.read_json(parent/'identity_flow/training_complete.json')
    if (summary.get('status'),summary.get('epoch'),summary.get('updates'),summary.get('F0_frozen'))!=('SOURCE_TRAINED',20,9860,True):
        raise ValueError('Fixed source-only E20 parent required')
    if any(summary.get(k)!=0 for k in ('F0_updates','TX_labels_read','V_IQ_reads','OE_IQ_reads','target_IQ_reads')):
        raise ValueError('Parent source training contract differs')
    path=parent/'identity_flow/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in dict(schema='identity_flow_source_v1',run_id=PARENT,code_commit=PARENT_CODE,epoch=20,updates=9860,
            TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0).items()):
        raise ValueError('Exact original E20 candidate checkpoint required')
    flow=IdentityConditionalFlow();flow.load_state_dict(payload['model'],strict=True);flow=flow.to(device).eval().requires_grad_(False)
    p,clean,roles,g_clean,g_received,teacher,identity=source_inputs(root,runtime,device)
    if identity!=summary['F0']:raise ValueError('Source teacher must equal parent F0')
    template=FixedAntitheticFlow(IdentityConditionalFlow()).noise_template.to(device)
    output.mkdir(parents=True)
    initial=path_metrics(flow,teacher,base.tensor(g_clean[:2],device),base.tensor(g_received[:2],device),template,4)
    base.write_json(output/'smoke.json',dict(status='PASS',actual_E20_G=True,actual_F0=True,
        G10_source_L_s_count=2,metrics=initial,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0))
    original=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    if original.shape!=(63000,2,256) or original.dtype!=np.float32:raise ValueError('Fixed source view required')
    select={role:np.flatnonzero(roles==role)[:128] for role in ('L_s','U_s')}
    if any(len(v)!=128 for v in select.values()):raise ValueError('Fixed source roles required')
    groups={'R07_L_s':(clean[select['L_s']],original[select['L_s']]),
            'R07_U_s':(clean[select['U_s']],original[select['U_s']]),
            'G10_L_s':(g_clean[:128],g_received[:128]),
            'clean_L_s':(clean[select['L_s']],clean[select['L_s']])}
    before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (flow,teacher)]
    records={}
    for name,(x,y) in groups.items():
        x=base.tensor(x,device);y=base.tensor(y,device)
        records[name]={str(n):path_metrics(flow,teacher,x,y,template,n) for n in STEPS}
        print('SOURCE_TRAJECTORY_COMPLETE',name,flush=True)
    for model,snapshot in zip((flow,teacher),before):
        if any(not torch.equal(v,snapshot[k]) for k,v in model.state_dict().items()):raise ValueError('Frozen state changed')
    base.write_json(output/'complete.json',dict(status='SOURCE_PATH_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=commit,
        parent_run=PARENT,parent_code_commit=PARENT_CODE,parent_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        F0=identity,steps=list(STEPS),groups=records,source_roles=['L_s','U_s'],sample_count_per_group=128,
        repeated_views_are_new_physical_records=False,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0,
        optimizer_updates=0,parameters_unchanged=True,
        scope='Fixed source training records only; teacher argmax agreement is not TX accuracy; step sensitivity is not target model selection.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reference-runtime',type=Path,required=True)
    parser.add_argument('--code-commit',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(args.root,args.reference_runtime,args.code_commit,args.root/'runs'/RUN,args.device)


if __name__=='__main__':main()
