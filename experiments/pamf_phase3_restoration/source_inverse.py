"""Frozen source-only blind/known-action diagnostic, never a deployable oracle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .cache import tensor,load_F0,check_F0_state,load_cache,close_cache
from .source_pair import read,validate_source
from .cost_probe import gpu_choice,write
from .main_split_eval import dura,valid_identity

RUN='20261101-diagnostic-Restore-source-inverse-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.source_inverse'
R07='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
METHODS=('received','DURA_v1','source_mean','oracle_white64','oracle_mean64','oracle_mean128')
SCENES=('clean','practical_high','practical_mid','practical_low_suburban',
        'practical_high_urban','practical_mid_urban','practical_low_urban')


def align_source(old_ids,roles,new_ids,new_roles):
    if (len(old_ids)!=63000 or len(new_ids)!=63000 or len(set(old_ids.tolist()))!=63000
            or set(old_ids.tolist())!=set(new_ids.tolist())):
        raise ValueError('Exact original L_s/U_s physical ID set required')
    lookup={str(v):i for i,v in enumerate(new_ids)}
    order=np.asarray([lookup[str(v)] for v in old_ids],np.int64)
    if (not np.array_equal(roles,new_roles[order]) or (roles=='L_s').sum()!=6300
            or (roles=='U_s').sum()!=56700):
        raise ValueError('Original hidden-label L_s/U_s roles must match by ID')
    return order


def panel(roles):
    # Fixed before source results: 128 L_s plus 384 U_s, all 20 existing draws.
    return np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)]
        for role,count,n in [('L_s',6300,128),('U_s',56700,384)]])


def action_values(kernel,coefficients):
    from experiments.pamf_action_restore.operators import Action
    import torch
    return Action(torch.complex(kernel[:,0],kernel[:,1]),
        torch.complex(coefficients[:,0],coefficients[:,1]),
        torch.complex(coefficients[:,2],coefficients[:,3]))


def worker(args):
    import torch
    from experiments.pamf_action_restore.operators import soft_map
    started=time.perf_counter();device='cuda:0';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    D,engine=dura(c,device)
    snapshots=[(m,{k:v.detach().clone() for k,v in m.state_dict().items()}) for m in (F,D)]
    out=args.root/'runs'/RUN;old=args.root/'runs'/R07/'cache'
    done=read(old/'complete.json')
    if (done.get('status'),done.get('train_count'),done.get('epochs'),done.get('TX_labels_read'),done.get('target_reads'))!=(
            'ACTION_SOURCE_CACHE_COMPLETE',63000,20,0,0):raise ValueError('Completed original 20-draw source cache required')
    ids=np.load(old/'train/physical_ids.npy',allow_pickle=False);roles=np.load(old/'train/roles.npy',allow_pickle=False)
    bundle=load_cache(args.cache,guided=False)
    try:align_source(ids,roles,bundle['ids'],bundle['roles'])
    finally:close_cache(bundle)
    selected=panel(roles);clean=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256) or clean.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
    # A single unlabeled source mean and its empirical residual variance; no per-TX fitting.
    mean=np.asarray(clean,dtype=np.float64).mean(0);residual_power=0.;clean_power=0.
    for offset in range(0,63000,1024):
        a=np.asarray(clean[offset:offset+1024],dtype=np.float64)
        residual_power+=float(((a-mean)**2).sum());clean_power+=float((a*a).sum())
    prior_variance=residual_power/(63000*2*256)
    if not 0<prior_variance<1:raise ValueError('Invalid source-only mean residual variance')
    mu=tensor(mean[None],device);targets=tensor(clean[selected],device)
    with torch.inference_mode():
        smoke=engine(targets[:2])[0];scores=F.classify_features(F.features(smoke))
        if not torch.isfinite(smoke).all() or scores.shape!=(2,6) or not torch.isfinite(scores).all():
            raise ValueError('Actual original F0/default DURA source smoke failed')
    write(out/'smoke.json',dict(status='PASS',source_L_s=2,actual_TriSIGTIME_SSL=identity,
        actual_default_DURA=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('RESTORE_SOURCE_INVERSE_SMOKE_PASS',flush=True)
    with torch.inference_mode():
        target_features=F.features(targets);target_decisions=F.classify_features(target_features).argmax(1)
    allrows={key:[] for key in ['epoch','source_index','scene','state','lock','actual_variance','native_noise_nmse',
        *[method+'__'+metric for method in METHODS for metric in ('nmse','feature_cosine_distance','clean_decision_agreement')]]}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';complete=read(folder/'complete.json')
            if complete.get('count')!=63000 or complete.get('TX_labels_read')!=0 or complete.get('target_reads')!=0:
                raise ValueError('Complete original source action draw required')
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)
                for k in ('received','kernel','coefficients','variance')}
            with np.load(folder/'groups.npz',allow_pickle=False) as group:groups={k:group[k][selected] for k in ('scene','state','lock')}
            try:
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];x=targets[offset:offset+64];n=len(x)
                    vals={k:tensor(a[take],device) for k,a in arrays.items()}
                    action=action_values(vals['kernel'],vals['coefficients'])
                    observed=torch.complex(vals['received'][:,0],vals['received'][:,1])
                    tx=torch.complex(x[:,0],x[:,1]);prior=torch.complex(mu[:,0],mu[:,1]).expand(n,-1)
                    energy=x.square().mean((1,2));noise=(observed-action(tx)).abs().square().mean(1)/(2*energy)
                    values={'received':vals['received'],'DURA_v1':engine(vals['received'])[0],
                        'source_mean':mu.expand(n,-1,-1)}
                    for name,base,variance,steps in [('oracle_white64',torch.zeros_like(prior),.5,64),
                            ('oracle_mean64',prior,prior_variance,64),('oracle_mean128',prior,prior_variance,128)]:
                        z=soft_map(action,observed,base,vals['variance'],variance,steps=steps)
                        values[name]=torch.stack((z.real,z.imag),1)
                    for method,value in values.items():
                        features=F.features(value)
                        metrics=dict(nmse=(value-x).square().mean((1,2))/energy,
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(features,target_features[offset:offset+n],dim=1),
                            clean_decision_agreement=F.classify_features(features).argmax(1)==target_decisions[offset:offset+n])
                        if not torch.isfinite(value).all() or any(not torch.isfinite(v).all() for v in metrics.values()):
                            raise ValueError('Nonfinite fixed source inverse result; never omit a case')
                        for key,value in metrics.items():allrows[method+'__'+key].extend(value.cpu().tolist())
                    allrows['epoch'].extend([epoch]*n);allrows['source_index'].extend(take.tolist())
                    allrows['actual_variance'].extend(vals['variance'].cpu().tolist());allrows['native_noise_nmse'].extend(noise.cpu().tolist())
                    for key in groups:allrows[key].extend(groups[key][offset:offset+n].tolist())
            finally:
                for a in arrays.values():a._mmap.close()
            print('RESTORE_SOURCE_INVERSE_DRAW_COMPLETE',epoch,512,flush=True)
    clean._mmap.close()
    for m,before in snapshots:check_F0_state(m,before)
    records={k:np.asarray(v) for k,v in allrows.items()}
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    summary=summarize(records)
    write(out/'results.json',dict(status='SOURCE_INVERSE_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=args.commit,
        F0_identity=identity,count=10240,unique_source_packets=512,L_s=128,U_s=384,existing_draws=20,
        summary=summary,source_mean_energy_fraction=float((mean*mean).sum()/(clean_power/63000)),
        source_mean_residual_NMSE=residual_power/clean_power,source_mean_real_prior_variance=prior_variance,
        source_ID_role_alignment='same63000set/orderrealigned',labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        optimizer_updates=0,model_state_updates=0,default_DURA_changed=False,
        physical_oracle_deployable=False,oracle='True source simulator FIR/main/image/variance, Gaussian linear MAP;no clean per-record prior',
        agreement_is_TX_accuracy=False,theoretical_limit_established=False,seconds=time.perf_counter()-started))
    print('RESTORE_SOURCE_INVERSE_COMPLETE',flush=True)


def summarize(records):
    scenes=np.asarray(records['scene']);result={}
    if len(scenes)!=10240 or set(scenes)!=set(SCENES):raise ValueError('Full fixed seven-scene source diagnostic required')
    for scene in ('ALL',*SCENES):
        mask=np.ones(len(scenes),bool) if scene=='ALL' else scenes==scene
        result[scene]=dict(count=int(mask.sum()),methods={})
        for method in METHODS:
            result[scene]['methods'][method]={metric:float(np.asarray(records[method+'__'+metric])[mask].mean())
                for metric in ('nmse','feature_cosine_distance','clean_decision_agreement')}
    return result


def controller(args):
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    choices=gpu_choice()
    while not choices:time.sleep(5);choices=gpu_choice()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(admission['GPU'])),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source-only inverse diagnostic failed; preserve outputs')
    if read(out/'results.json').get('status')!='SOURCE_INVERSE_DIAGNOSTIC_COMPLETE':raise ValueError('No complete artifacts')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',training_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual code Git identity required')
    if a.mode=='run':controller(a)
    else:worker(a)


if __name__=='__main__':main()
