"""Synthetic source-IQ response proxies; no TX truth, fitting, or target inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .cache import check_F0_state,load_F0,tensor
from .cost_probe import write
from .covariance_family_source_validation import METHODS,SPECS,load_models
from .covariance_inverse import dense_action
from .low_rank_source_probe import RECEIVERS,SCENES,select_rows
from .main_split_eval import valid_identity
from .phase_mixture_source import read_trace
from .source_inverse import R07
from .source_pair import read,validate_source
from .structured_noise_source import known_parameters

RUN='20261107-diagnostic-Restore-CovarianceResponse-sourceLU-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.source_fingerprint_response'
DIRECTIONS=('conjugate','amplitude_cubic','phase_slope','local_difference')
EPSILON=.01


def perturbations(clean):
    if clean.ndim!=3 or clean.shape[1:]!=(2,256) or not torch.isfinite(clean).all():
        raise ValueError('Finite original source IQ2x256 required')
    z=torch.complex(clean[:,0],clean[:,1]);power=z.abs().square()
    meanpower=power.mean(1,keepdim=True)
    if (meanpower<=0).any():raise ValueError('Positive source signal energy required')
    t=torch.linspace(-1,1,256,device=clean.device,dtype=clean.dtype)[None]
    directions=(z.conj(),z*(power/meanpower-1),1j*z*t,torch.roll(z,1,-1)-z)
    result=[]
    for direction in directions:
        norm=direction.abs().square().mean(1,keepdim=True).sqrt()
        if (norm<1e-10).any() or not torch.isfinite(norm).all():raise ValueError('Nonzero finite synthetic direction required')
        delta=EPSILON*meanpower.sqrt()*direction/norm
        result.append(torch.stack((delta.real,delta.imag),1))
    return torch.stack(result,1)


def inject(received,delta,action):
    if received.shape!=delta.shape or received.ndim!=3 or received.shape[1:]!=(2,256) or action.shape!=(len(received),512,512):
        raise ValueError('Matched source IQ and saved full real action required')
    return received+(action.double()@delta.flatten(1).double()[:,:,None]).squeeze(-1).float().reshape_as(received)


def response_statistics(expected,actual):
    if expected.shape!=actual.shape or expected.ndim<2 or not all(torch.isfinite(x).all() for x in (expected,actual)):
        raise ValueError('Matched finite source responses required')
    a,b=expected.flatten(1).double(),actual.flatten(1).double()
    return torch.stack((a.square().sum(1),b.square().sum(1),(a*b).sum(1),(b-a).square().sum(1)),1)


def summarize(values):
    a=np.asarray(values,np.float64)
    if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all() or (a[:,[0,1,3]]<0).any():
        raise ValueError('Finite per-record response sufficient statistics required')
    ref,actual,dot,error=a.sum(0)
    return dict(count=len(a),zero_reference_rows=int((a[:,0]==0).sum()),reference_energy_sum=float(ref),
        response_energy_sum=float(actual),projection_sum=float(dot),squared_error_sum=float(error),
        projection_gain=None if ref==0 else float(dot/ref),relative_squared_error=None if ref==0 else float(error/ref),
        response_norm_ratio=None if ref==0 else float(np.sqrt(actual/ref)))


def predict(args,out):
    started=time.perf_counter();models,parent_identity=load_models(args.root)
    reader,identity=load_F0(validate_source(read(args.inputs)),'cpu');valid_identity(identity)
    if identity!=parent_identity:raise ValueError('Original shared frozen F0 required')
    all_models=[*models.values(),reader]
    states=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in all_models]
    cache=args.root/'runs'/R07/'cache';meta=read(cache/'complete.json')
    if (meta['status'],meta['train_count'],meta['epochs'],meta['TX_labels_read'],meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,20,0,0):
        raise ValueError('Original label-hidden source training cache required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False);rx=np.load(cache/'train/rx.npy',allow_pickle=False)
    ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False)
    with np.load(cache/'epochs/01/groups.npz',allow_pickle=False) as f:take=select_rows(f['scene'],roles,rx)
    if len(set(take.tolist()))!=224 or len(set(ids[take].tolist()))!=224 or (roles[take]=='L_s').sum()!=35 or (roles[take]=='U_s').sum()!=189:
        raise ValueError('Fixed224 source rows with35L_s/189U_s required')
    array=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(array[take].copy(),'cpu')
    finally:array._mmap.close()
    trace=read_trace(cache/'epochs/01',take,63000)
    with torch.inference_mode():
        # First two selected rows are one L_s and one U_s; neither exposes TX.
        sample=trace['received'][:2]
        for model in models.values():
            y=model(sample);z=reader.features(y)
            if y.shape!=(2,2,256) or z.shape!=(2,160) or not torch.isfinite(y).all() or not torch.isfinite(z).all():raise ValueError('Actual source G/F0 feature smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_source_rows=2,roles=roles[take[:2]].tolist(),actual_three_final30_G=True,
        actual_original_F0_features=True,classification_head_calls=0,TX_labels_read=0,V_reads=0,target_inputs=0,optimizer_updates=0))
    print('SOURCE_FINGERPRINT_RESPONSE_SMOKE_PASS',flush=True)
    with (out/'reference.npz').open('xb') as file:np.savez(file,source_index=take,ids=ids[take],roles=roles[take],rx=rx[take])
    results=[]
    with torch.inference_mode():
        for scene_index,scene in enumerate(SCENES):
            take_scene=slice(scene_index*32,(scene_index+1)*32);x=clean[take_scene]
            t={k:v[take_scene] for k,v in trace.items()};y=t['received']
            h,main,image,_=known_parameters(t);action=dense_action(h,main,image)
            delta=perturbations(x);feature=reader.features(x)
            target_feature=torch.stack([reader.features(x+delta[:,j])-feature for j in range(4)],1)
            arrays=dict(ids=ids[take][take_scene],delta_IQ=np.asarray(delta.tolist()),delta_F0=np.asarray(target_feature.tolist()))
            for method,model in models.items():
                base,aux=model(y,return_aux=True);base_feature=reader.features(base)
                if aux['relative_solve_residual'].max()>1e-8:raise ValueError('Original posterior residual failed')
                iq_responses=[];feature_responses=[];iq_stats=[];feature_stats=[]
                for j,direction in enumerate(DIRECTIONS):
                    changed=inject(y,delta[:,j],action)
                    restored,aux=model(changed,return_aux=True);z=reader.features(restored)
                    if aux['relative_solve_residual'].max()>1e-8:raise ValueError('Perturbed posterior residual failed')
                    iq=restored-base;feat=z-base_feature
                    stat_iq=response_statistics(delta[:,j],iq);stat_feature=response_statistics(target_feature[:,j],feat)
                    iq_responses.append(iq);feature_responses.append(feat);iq_stats.append(stat_iq);feature_stats.append(stat_feature)
                    results.append(dict(scene=scene,method=method,direction=direction,IQ=summarize(stat_iq.tolist()),F0=summarize(stat_feature.tolist())))
                arrays.update({method+'__IQ_response':np.asarray(torch.stack(iq_responses,1).tolist()),
                    method+'__F0_response':np.asarray(torch.stack(feature_responses,1).tolist()),
                    method+'__IQ_statistics':np.asarray(torch.stack(iq_stats,1).tolist()),
                    method+'__F0_statistics':np.asarray(torch.stack(feature_stats,1).tolist()),
                    method+'__base_IQ_NMSE':np.asarray(((base-x).square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4)).tolist())})
                print('SOURCE_FINGERPRINT_RESPONSE_MODEL_COMPLETE',scene,method,32,flush=True)
            with (out/(scene+'.npz')).open('xb') as file:np.savez(file,**arrays)
    for model,before in zip(all_models,states):check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_DIRECTIONAL_RESPONSE_COMPLETE',run_id=RUN,code_commit=args.commit,
        parents={name:dict(run=spec[0].RUN,commit=spec[1],epoch=30,updates=7410) for name,spec in SPECS.items()},F0_identity=identity,
        methods=list(METHODS),scenes=list(SCENES),directions=list(DIRECTIONS),epsilon_relative_rms=EPSILON,
        source_records=224,source_L_s=35,source_U_s=189,source_RX=list(RECEIVERS),source_draw=1,cells=84,
        source_G_rows=3360,F0_clean_reference_rows=1120,F0_output_rows=3360,source_smoke_G_rows=6,source_smoke_F0_rows=6,
        original_noise_reused=True,source_action='Recorded simulator action, not estimated or a target/oracle deployment input.',
        classification_head_calls=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        results=results,seconds=time.perf_counter()-started,actual_TX_identity_claim=False,Main_performance_claim=False,
        caveat='Synthetic1% source perturbation sensitivity, not actual transmitter identity recovery. Noise is held fixed, not redrawn; strong response can also amplify nuisance.'))


def controller(args,out):
    out.mkdir();(out/'logs').mkdir();write(out/'launch.json',dict(PID=os.getpid(),CWD=os.getcwd(),code_commit=args.commit,CUDA_empty=True,CPU_threads=2))
    command=[sys.executable,'-u','-m',MODULE,'predict','--root',str(args.root),'--inputs',str(args.inputs),'--commit',args.commit]
    with (out/'logs/predict.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(out/'submitted-predict.json',dict(PID=child.pid,command=command,CWD=os.getcwd()));rc=child.wait()
    write(out/'terminal-predict.json',dict(PID=child.pid,exit_code=rc))
    if rc:raise RuntimeError('Source response probe failed; preserve, no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_DIRECTIONAL_RESPONSE_COMPLETE':raise ValueError('All response cells required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,model_state_updates=0,target_inputs=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','predict'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('CPU source diagnostic and actual code identity required')
    os.nice(max(0,10-os.nice(0)));torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'run':controller,'predict':predict}[args.mode](args,args.root/'runs'/RUN)


if __name__=='__main__':main()
