"""Frozen training-source trajectory comparison; never a target performance bound."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import multiview_trial as base
from . import direct_bridge_trial as direct
from . import state_bridge_trial as state
from .source_inverse import panel
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import load_scale,recovery_gpu_choice
from .cost_probe import write

RUN='20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.bridge_trajectory'
METHODS=('DirectBridge','StateBridge')
COMMITS=('84a063c9ba1f9974cfe361e293a9e11fa93920b4','c50a9159b1fb891c6f1a6f8ad4bfa2d7eb66222e')
METRICS=('iq_nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement',
    'latent_cosine_distance','latent_standardized_mse','latent_clean_decision_agreement',
    'raw_complex_power','output_power_abs_error','change_from_first_nmse')


def trajectory(model,received):
    value=received;items=[]
    for remaining in (4,3,2,1):
        t=torch.full((len(received),),remaining/4,device=received.device,dtype=received.dtype)
        restored,aux=model.estimate(value,received,t,return_aux=True)
        items.append((remaining/4,restored,aux))
        value=restored if remaining==1 else value+(restored-value)/remaining
    return items


def summarize(a):
    if len(a['method'])!=81920 or set(a['method'])!=set(METHODS):raise ValueError('Both fixed methods x4steps x10240 source observations required')
    result=[]
    for method in METHODS:
        first={k:v[(a['method']==method)&(a['step']==1)] for k,v in a.items()}
        if len(first['epoch'])!=10240 or set(first['epoch'])!=set(range(1,21)):raise ValueError('All20 original source draws required')
        selected=first['source_index'][:512]
        if len(set(selected.tolist()))!=512:raise ValueError('Unique fixed512 source packets required')
        for epoch in range(1,21):
            if not np.array_equal(first['source_index'][first['epoch']==epoch],selected):raise ValueError('Same512 source panel order in every draw required')
        for step in range(1,5):
            q={k:v[(a['method']==method)&(a['step']==step)] for k,v in a.items()}
            if any(not np.array_equal(q[k],first[k]) for k in ['epoch','source_index','scene','lock']):raise ValueError('Every step must refer to exactly the same source cases')
            if any(q[k].shape!=(10240,) or not np.isfinite(q[k]).all() for k in METRICS):raise ValueError('Complete finite trajectory metrics required')
            if not np.isin(q['clean_decision_agreement'],[0,1]).all():raise ValueError('Binary Fclean agreement required')
            masks=[('overall','ALL',np.ones(10240,bool))]
            for key in ['scene','lock']:masks.extend((key,str(v),q[key]==v) for v in np.unique(q[key]))
            before=first['clean_decision_agreement'].astype(bool);after=q['clean_decision_agreement'].astype(bool)
            for dimension,stratum,mask in masks:
                repairs=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
                if repairs-damage!=int(after[mask].sum())-int(before[mask].sum()):raise ValueError('Paired disagreement counts must reconcile')
                result.append(dict(method=method,step=step,time=1-(step-1)/4,dimension=dimension,stratum=stratum,count=int(mask.sum()),
                    metrics={k:float(q[k][mask].mean()) for k in METRICS},paired_against_first=dict(repairs=repairs,damage=damage,net=repairs-damage)))
    return result


def load_models(root,device):
    models={}
    for name,trial,commit,parameters in zip(METHODS,(direct,state),COMMITS,(4677538,4676898)):
        folder=root/'runs'/trial.RUN/'rows'/trial.ARM/'training';done=read(folder/'complete.json')
        if (done['code_commit'],done['epoch'],done['updates'],done['G_parameters'])!=(commit,30,7410,parameters):raise ValueError('Both exact fixedfinal30 source models required')
        if any(done[k] for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):raise ValueError('Matching source-only scratch ancestry required')
        p=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        if set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=trial.configuration(trial.ARM) or (p['code_commit'],p['epoch'],p['updates'])!=(commit,30,7410):raise ValueError('Exact model identity/config required')
        G=trial.factory(trial.ARM);expected=G.state_dict()
        if set(p['state'])!=set(expected) or any(v.dtype!=torch.float32 or v.shape!=expected[k].shape or not torch.isfinite(v).all() for k,v in p['state'].items()):raise ValueError('Exact finite model tensors required')
        G.load_state_dict(p['state'],strict=True);models[name]=G.to(device).eval().requires_grad_(False)
    return models


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old,roles=base.source(args);selected=panel(roles);models=load_models(args.root,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    all_models=[*models.values(),F];snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in all_models]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a[selected],device)
    finally:a._mmap.close()
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[selected[:2]],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        for model in models.values():
            path=trajectory(model,source2)
            if not torch.equal(path[-1][1],model(source2)):raise ValueError('Recorded trajectory must exactly match frozen deployment')
            z=F.classify_features(F.features(path[-1][1]))
            if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual finalG/F0 source smoke required')
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,actual_both_finalG=True,fourth_equals_deployment=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
    print('BRIDGE_TRAJECTORY_SOURCE_SMOKE_PASS',flush=True)
    scale=load_scale(args.root).to(device)
    with torch.inference_mode():target=F.features(clean);target_prediction=F.classify_features(target).argmax(1)
    keys=['method','step','epoch','source_index','scene','lock','Fclean_prediction','output_prediction','latent_prediction',*METRICS]
    records={k:[] for k in keys}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Original complete source draw required')
            with np.load(folder/'groups.npz',allow_pickle=False) as f:groups={k:f[k][selected] for k in ['scene','lock']}
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            try:
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];n=len(take);received=tensor(a[take],device)
                    reference=clean[offset:offset+n];features=target[offset:offset+n];pred=target_prediction[offset:offset+n]
                    energy=reference.square().mean((1,2))
                    for name,model in models.items():
                        path=trajectory(model,received);first=path[0][1]
                        for step,(t,value,aux) in enumerate(path,1):
                            z=F.features(value);p=F.classify_features(z).argmax(1);latent=aux['predicted_features'];lp=F.classify_features(latent).argmax(1)
                            metrics=dict(iq_nmse=(value-reference).square().mean((1,2))/energy,
                                feature_cosine_distance=1-torch.nn.functional.cosine_similarity(z,features,dim=1),
                                feature_standardized_mse=((z-features).square()/scale).mean(1),clean_decision_agreement=p==pred,
                                latent_cosine_distance=1-torch.nn.functional.cosine_similarity(latent,features,dim=1),
                                latent_standardized_mse=((latent-features).square()/scale).mean(1),latent_clean_decision_agreement=lp==pred,
                                raw_complex_power=aux['raw_restored'].square().sum(1).mean(1),
                                output_power_abs_error=(value.square().sum(1).mean(1)-1).abs(),change_from_first_nmse=(value-first).square().mean((1,2))/energy)
                            if any(not torch.isfinite(v).all() for v in metrics.values()):raise ValueError('Nonfinite source step;never omit cases')
                            for k,v in metrics.items():records[k].extend(v.cpu().tolist())
                            for k,v in [('method',[name]*n),('step',[step]*n),('epoch',[epoch]*n),('source_index',take.tolist()),
                                    ('scene',groups['scene'][offset:offset+n].tolist()),('lock',groups['lock'][offset:offset+n].tolist()),
                                    ('Fclean_prediction',pred.cpu().tolist()),('output_prediction',p.cpu().tolist()),('latent_prediction',lp.cpu().tolist())]:records[k].extend(v)
            finally:a._mmap.close()
            print('BRIDGE_TRAJECTORY_SOURCE_DRAW_COMPLETE',epoch,512,flush=True)
    for model,before in zip(all_models,snapshots):check_F0_state(model,before)
    records={k:np.asarray(v) for k,v in records.items()};summary=summarize(records)
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='BRIDGE_TRAJECTORY_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        methods=list(METHODS),parent_code_commits=list(COMMITS),source_packets=512,source_draws=20,source_cases=10240,trajectory_rows=81920,
        summary=summary,G_core_rows=81920,F0_output_feature_rows=81920,F0_clean_feature_rows=512,latent_defaulthead_rows=81920,
        smoke_extra_G_core_rows=32,smoke_extra_F0_feature_rows=4,TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,
        optimizer_updates=0,model_state_updates=0,checkpoint_selection=False,source_in_sample=True,agreement_is_TX_accuracy=False,
        theoretical_ceiling=False,default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty diagnostic owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    print('WAITING_FOR_BOTH_FIXED_PARENT_MAIN_COMPLETION',flush=True)
    while True:
        ready=True
        for trial in (direct,state):
            folder=args.root/'runs'/trial.RUN
            if any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):raise RuntimeError('Parent technical failure;preserve diagnostic without retry')
            p=folder/'complete.json'
            if not p.is_file() or read(p)['status']!='ARTIFACTS_COMPLETE':ready=False
        if ready:break
        time.sleep(5)
    choices=recovery_gpu_choice()
    while not choices:time.sleep(5);choices=recovery_gpu_choice()
    a=choices[0];cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=a['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=a))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source trajectory failed;preserve/no automatic retry')
    if read(out/'results.json')['status']!='BRIDGE_TRAJECTORY_COMPLETE':raise ValueError('Complete source trajectory results required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    if a.mode=='run':controller(a)
    else:worker(a)
