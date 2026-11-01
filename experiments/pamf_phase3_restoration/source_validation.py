"""Fixed original source-V validation after all parent Main tests finish."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import multiview_trial as base
from . import variance_trial as varf
from . import physical_trial as phy
from . import full_feature_trial as full
from .multiscale import scratch_multiscale
from .physical import scratch_physical
from .physical_diagnostic import true_trace,restore_with
from .source_inverse import R07,SCENES
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .main_split_eval import dura,valid_identity
from .cost_probe import write

RUN='20261101-diagnostic-Restore-fixed-sourceV-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.source_validation'
SPECS={
 'MSAT_M20':('20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01','8720a8584803366d42d2879fb99502b8d220ac26',base.configuration,scratch_multiscale),
 'MSAT_VarF':(varf.RUN,'9fdb1f5b164484f2bc438cb1e45d1ee805d2473a',varf.configuration,scratch_multiscale),
 'PHY_MSat':(phy.RUN,'8198a2e70c655f24ebaf321f232a676cfb46d916',phy.configuration,scratch_physical),
 'MSAT_FullFGrad':(full.RUN,'940fa78ba450b686b1b03c7530c47af4360faafc',full.configuration,scratch_multiscale)}
METHODS=('received','DURA_v1','MSAT_M20','MSAT_VarF','PHY_MSat','MSAT_FullFGrad','PHY_true_H')
METRICS=('nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement')


def validation_roles(ids,roles,rx,day,train_ids):
    if (ids.shape!=(1800,) or roles.shape!=(1800,) or rx.shape!=(1800,) or day.shape!=(1800,)
            or len(set(ids.tolist()))!=1800 or set(roles.tolist())!={'V'} or set(ids.tolist())&set(train_ids.tolist())
            or set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3}
            or any(int(((rx==r)&(day==d)).sum())!=120 for r in [1,3,4,6,8] for d in [1,2,3])):
        raise ValueError('Original disjoint fixed1800 source-V/5RX3day handles required')


def load_frozen(path,arm,device):
    run,commit,configuration,factory=SPECS[arm]
    p=torch.load(path,map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=configuration(arm)
            or (p['code_commit'],p['epoch'],p['updates'])!=(commit,30,7410)):
        raise ValueError('Exact source-only fixedfinal30 parent required')
    G=factory(2026092701);state=G.state_dict()
    if set(p['state'])!=set(state) or any(v.dtype!=torch.float32 or v.shape!=state[k].shape or not torch.isfinite(v).all() for k,v in p['state'].items()):
        raise ValueError('Exact finite source-only G state required')
    G.load_state_dict(p['state'],strict=True)
    return G.to(device).eval().requires_grad_(False)


def summarize(records):
    if len(records['scene'])!=12600 or set(records['scene'])!=set(SCENES):raise ValueError('All1800sourceV x7views required')
    result={}
    for scene in SCENES:
        mask=records['scene']==scene
        if int(mask.sum())!=1800:raise ValueError('Every sourceV view requires all1800 packets')
        group=dict(count=1800,methods={})
        for method in METHODS:
            group['methods'][method]={key:float(records[method+'__'+key][mask].mean()) for key in METRICS}
        result[scene]=group
    return result


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old=args.root/'runs'/R07/'cache';meta=read(old/'complete.json')
    if (meta['status'],meta['train_count'],meta['source_V_count'],meta['epochs'],meta['TX_labels_read'],meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,1800,20,0,0):
        raise ValueError('Existing validated original R07 source family required')
    folder=old/'validation';manifest=read(folder/'manifest.json')
    if manifest['count']!=1800 or manifest['roles']!={'V':1800} or manifest['label_reads']!=0:raise ValueError('Original fixed source V only')
    ids,roles,rx,day=[np.load(folder/(k+'.npy'),allow_pickle=False) for k in ['physical_ids','roles','rx','day']]
    validation_roles(ids,roles,rx,day,np.load(old/'train/physical_ids.npy',allow_pickle=False))
    models={}
    for arm,(run,commit,_,_) in SPECS.items():
        parent=args.root/'runs'/run
        if read(parent/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Every parent Main run must finish first')
        done=read(parent/'rows'/arm/'training/complete.json')
        if (done['code_commit'],done['epoch'],done['updates'])!=(commit,30,7410) or any(done[k] for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
            raise ValueError('Original source-only scratch parent provenance required')
        models[arm]=load_frozen(parent/'rows'/arm/'training/final.pt',arm,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    D,engine=dura(c,device);allmodels=[F,D,*models.values()]
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in allmodels]
    # No validation NN is used before the real source two-L_s checkpoint smoke.
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:smoke=tensor(a[:2],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        for value in [engine(smoke)[0],*[G(smoke) for G in models.values()]]:
            logits=F.classify_features(F.features(value))
            if value.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(value).all() or not torch.isfinite(logits).all():
                raise ValueError('Actual source checkpoint no-query smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,all_four_G_and_default_DURA=True,F0_identity=identity,optimizer_updates=0,target_inputs=0))
    print('SOURCE_V_REAL_SOURCE_SMOKE_PASS',flush=True)
    a=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(1800,2,256) or a.dtype!=np.float32:raise ValueError('Original sourceV FP32 clean required')
        clean=tensor(a,device)
    finally:a._mmap.close()
    if not torch.isfinite(clean).all():raise ValueError('Finite validation IQ required')
    scale=varf.load_scale(args.root).to(device)
    with torch.inference_mode():
        target=torch.cat([F.features(clean[i:i+64]) for i in range(0,1800,64)])
        clean_prediction=F.classify_features(target).argmax(1)
    records={k:[] for k in ['scene','source_index','rx','day','Fclean_prediction',
        *[m+'__'+k for m in METHODS for k in (*METRICS,'prediction')]]}
    with torch.inference_mode():
        for scene in SCENES:
            path=old/'validation_actions'/scene;done=read(path/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(1800,0,0):raise ValueError('Existing fixed sourceV action trace required')
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            try:
                shapes={'received':(1800,2,256),'kernel':(1800,2,64),'coefficients':(1800,4,256),'variance':(1800,)}
                if any(q.shape!=shapes[k] or q.dtype!=np.float32 for k,q in arrays.items()):raise ValueError('SourceV trace shapes differ')
                for offset in range(0,1800,64):
                    n=min(64,1800-offset);take=slice(offset,offset+n);x=clean[take];y=tensor(arrays['received'][take],device)
                    h,unused=true_trace(tensor(arrays['kernel'][take],device),tensor(arrays['coefficients'][take],device),tensor(arrays['variance'][take],device))
                    values={'received':y,'DURA_v1':engine(y)[0],**{arm:G(y) for arm,G in models.items()}}
                    estimated_noise=models['PHY_MSat'].estimator(y)[1]
                    values['PHY_true_H']=restore_with(models['PHY_MSat'],y,h,estimated_noise)
                    for method,value in values.items():
                        z=F.features(value);prediction=F.classify_features(z).argmax(1)
                        metrics=dict(nmse=(value-x).square().mean((1,2))/x.square().mean((1,2)),
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(z,target[take],dim=1),
                            feature_standardized_mse=((z-target[take]).square()/scale).mean(1),
                            clean_decision_agreement=prediction==clean_prediction[take],prediction=prediction)
                        if not torch.isfinite(value).all() or any(not torch.isfinite(q).all() for q in metrics.values()):raise ValueError('Nonfinite fixed validation result; omit no cases')
                        for key,q in metrics.items():records[method+'__'+key].extend(q.cpu().tolist())
                    records['scene'].extend([scene]*n);records['source_index'].extend(range(offset,offset+n))
                    for key,q in [('rx',rx[take]),('day',day[take])]:records[key].extend(q.tolist())
                    records['Fclean_prediction'].extend(clean_prediction[take].cpu().tolist())
            finally:
                for a in arrays.values():a._mmap.close()
            print('SOURCE_V_VIEW_COMPLETE',scene,1800,flush=True)
    for model,before in zip(allmodels,snapshots):check_F0_state(model,before)
    records={k:np.asarray(q) for k,q in records.items()}
    with (out/'validation_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='SOURCE_V_VALIDATION_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        source_V_packets=1800,source_V_views=7,count=12600,methods=list(METHODS),summary=summarize(records),
        source_V_validation_only=True,V_training_rows=0,TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,
        optimizer_updates=0,model_state_updates=0,checkpoint_selection=False,agreement_is_TX_accuracy=False,
        oracle_deployable=False,theoretical_ceiling=False,source_receiver_seed=2026101902,training_receiver_seed=2026101911,
        V_historically_exposed=True,default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();parent=args.root/'runs'/full.RUN
    write(out/'waiting.json',dict(status='WAIT_PARENT_COMPLETE',parent=full.RUN,depends_on_technical_completion_only=True))
    while not (parent/'complete.json').is_file():
        for p in parent.glob('terminal-*.json'):
            if read(p)['exit_code']!=0:raise RuntimeError('Parent technical failure; preserve/no automatic retry')
        time.sleep(5)
    if read(parent/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete parent Main artifacts required')
    choices=varf.recovery_gpu_choice()
    while not choices:time.sleep(5);choices=varf.recovery_gpu_choice()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Fixed sourceV validation failed; preserve/no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_V_VALIDATION_COMPLETE':raise ValueError('Missing complete sourceV validation')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual Git code identity required')
    if args.mode=='run':controller(args)
    else:worker(args)


if __name__=='__main__':main()
