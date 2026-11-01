"""Frozen source paired-transform stability, never a native-channel claim."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import multiview_trial as base
from . import variance_trial as varf
from .source_validation import load_frozen,SPECS
from .source_inverse import panel,SCENES
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .main_split_eval import dura,valid_identity
from .cost_probe import write

RUN='20261101-diagnostic-Restore-source-style-equivariance-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.source_style'
STYLES=('identity','phase90','gain070','gain140','reflect_fir025')
METHODS=('DURA_v1','PHY_MSat','MSAT_FullFGrad')
METRICS=('nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement')


def transform(value,style):
    if value.ndim!=3 or value.shape[1:]!=(2,256) or value.dtype!=torch.float32:
        raise ValueError('Per-record FP32 IQ2x256 required')
    if style=='identity':return value
    if style=='phase90':return torch.stack((-value[:,1],value[:,0]),1)
    if style=='gain070':return .7*value
    if style=='gain140':return 1.4*value
    if style=='reflect_fir025':
        z=torch.complex(value[:,0],value[:,1]);lag=torch.cat((z[:,1:2],z[:,:-1]),1)
        beta=.25*complex(math.cos(math.pi/3),math.sin(math.pi/3))
        z=(z+beta*lag)/math.sqrt(1+.25**2)
        return torch.stack((z.real,z.imag),1)
    raise ValueError('Only five fixed paired transforms allowed')


def metrics(reader,value,target,target_features,target_prediction,scale):
    energy=target.square().mean((1,2))
    if not (energy>0).all():raise ValueError('Nonzero clean reference energy required')
    features=reader.features(value);prediction=reader.classify_features(features).argmax(1)
    result=dict(nmse=(value-target).square().mean((1,2))/energy,
        feature_cosine_distance=1-torch.nn.functional.cosine_similarity(features,target_features,dim=1),
        feature_standardized_mse=((features-target_features).square()/scale).mean(1),
        clean_decision_agreement=prediction==target_prediction,prediction=prediction)
    if not torch.isfinite(value).all() or any(not torch.isfinite(q).all() for q in result.values()):
        raise ValueError('Nonfinite source case; never omit cases')
    return result


def summarize(records):
    if len(records['scene'])!=51200 or set(records['style'])!=set(STYLES):raise ValueError('All512source x20draw x5styles required')
    result={}
    for style in STYLES:
        selected=records['style']==style
        if int(selected.sum())!=10240 or set(records['scene'][selected])!=set(SCENES):raise ValueError('Full source cases for every style required')
        result[style]={}
        for scene in ('ALL',*SCENES):
            mask=selected if scene=='ALL' else selected&(records['scene']==scene)
            group=dict(count=int(mask.sum()),clean_feature_shift=float(records['clean_feature_shift'][mask].mean()),
                clean_decision_preservation=float(records['clean_decision_preservation'][mask].mean()),methods={})
            for method in METHODS:
                q=dict(equivariance_nmse=float(records[method+'__equivariance_nmse'][mask].mean()),paths={})
                before=records[method+'__reference__clean_decision_agreement'].astype(bool)
                after=records[method+'__actual__clean_decision_agreement'].astype(bool)
                repair=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
                q['actual_against_transformed_baseline']=dict(repairs=repair,damage=damage,net=repair-damage)
                for path in ('actual','reference'):
                    q['paths'][path]={k:float(records[method+'__'+path+'__'+k][mask].mean()) for k in METRICS}
                group['methods'][method]=q
            result[style][scene]=group
    return result


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old,roles=base.source(args);selected=panel(roles)
    models={}
    for arm in METHODS[1:]:
        run,commit,_,_=SPECS[arm];parent=args.root/'runs'/run
        if read(parent/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Completed frozen parent required')
        done=read(parent/'rows'/arm/'training/complete.json')
        if (done['code_commit'],done['epoch'],done['updates'])!=(commit,30,7410) or any(done[k] for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
            raise ValueError('Matching original source-only parent required')
        models[arm]=load_frozen(parent/'rows'/arm/'training/final.pt',arm,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    D,engine=dura(c,device);allmodels=[F,D,*models.values()]
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in allmodels]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
        clean=tensor(a[selected],device)
    finally:a._mmap.close()
    # First NN work is the actual two-L_s checkpoint smoke; no query or updates.
    with torch.inference_mode():
        for value in [engine(clean[:2])[0],*[G(clean[:2]) for G in models.values()]]:
            z=F.classify_features(F.features(value))
            if value.shape!=(2,2,256) or z.shape!=(2,6) or not torch.isfinite(value).all() or not torch.isfinite(z).all():
                raise ValueError('Actual frozen source checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_source_Ls=2,all_three_G=True,F0_identity=identity,optimizer_updates=0,target_inputs=0))
    print('SOURCE_STYLE_REAL_CHECKPOINT_SMOKE_PASS',flush=True)
    scale=varf.load_scale(args.root).to(device)
    with torch.inference_mode():
        targets={style:transform(clean,style) for style in STYLES}
        target_features={style:F.features(x) for style,x in targets.items()}
        target_predictions={style:F.classify_features(z).argmax(1) for style,z in target_features.items()}
        clean_shift={style:1-torch.nn.functional.cosine_similarity(z,target_features['identity'],dim=1) for style,z in target_features.items()}
    keys=['epoch','source_index','scene','style','Fclean_prediction','Fclean_original_prediction','clean_feature_shift','clean_decision_preservation',
        *[m+'__equivariance_nmse' for m in METHODS],*[m+'__'+p+'__'+k for m in METHODS for p in ('actual','reference') for k in (*METRICS,'prediction')]]
    records={k:[] for k in keys}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete original source training draw required')
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received FP32 required')
            with np.load(folder/'groups.npz',allow_pickle=False) as f:scenes=f['scene'][selected]
            try:
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];part=slice(offset,offset+64);n=len(take);received=tensor(a[take],device)
                    baseline={'DURA_v1':engine(received)[0],**{arm:G(received) for arm,G in models.items()}}
                    for style in STYLES:
                        observed=transform(received,style);target=targets[style][part];z=target_features[style][part];p=target_predictions[style][part]
                        actual=baseline if style=='identity' else {'DURA_v1':engine(observed)[0],**{arm:G(observed) for arm,G in models.items()}}
                        for method,value in actual.items():
                            reference=transform(baseline[method],style)
                            actual_metrics=metrics(F,value,target,z,p,scale)
                            ref_metrics=actual_metrics if style=='identity' else metrics(F,reference,target,z,p,scale)
                            for path,values in [('actual',actual_metrics),('reference',ref_metrics)]:
                                for key,q in values.items():records[method+'__'+path+'__'+key].extend(q.cpu().tolist())
                            eq=(value-reference).square().mean((1,2))/target.square().mean((1,2))
                            if not torch.isfinite(eq).all():raise ValueError('Nonfinite source stability metric')
                            records[method+'__equivariance_nmse'].extend(eq.cpu().tolist())
                        records['epoch'].extend([epoch]*n);records['source_index'].extend(take.tolist())
                        records['scene'].extend(scenes[part].tolist());records['style'].extend([style]*n)
                        records['Fclean_prediction'].extend(p.cpu().tolist());records['Fclean_original_prediction'].extend(target_predictions['identity'][part].cpu().tolist())
                        records['clean_feature_shift'].extend(clean_shift[style][part].cpu().tolist())
                        records['clean_decision_preservation'].extend((p==target_predictions['identity'][part]).cpu().tolist())
            finally:a._mmap.close()
            print('SOURCE_STYLE_DRAW_COMPLETE',epoch,512,5,flush=True)
    for model,before in zip(allmodels,snapshots):check_F0_state(model,before)
    records={k:np.asarray(q) for k,q in records.items()}
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='SOURCE_STYLE_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        source_packets=512,source_Ls=128,source_Us=384,source_draws=20,styles=list(STYLES),methods=list(METHODS),count=51200,
        summary=summarize(records),G_request_rows=153600,F0_readout_rows=276480,F0_clean_reference_rows=2560,
        F0_total_source_rows=279040,source2_smoke_F0_rows=6,counts_are_request_rows_not_internal_calls=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        source_only=True,paired_transform_is_native_channel=False,agreement_is_TX_accuracy=False,
        claim_Main_causal_explanation=False,claim_theoretical_ceiling=False,default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    choices=varf.recovery_gpu_choice()
    while not choices:time.sleep(5);choices=varf.recovery_gpu_choice()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source paired-style diagnostic failed; preserve/no auto retry')
    if read(out/'results.json')['status']!='SOURCE_STYLE_COMPLETE':raise ValueError('Complete source-style results required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual Git code identity required')
    if args.mode=='run':controller(args)
    else:worker(args)


if __name__=='__main__':main()
