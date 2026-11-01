"""Frozen training-source path counterfactuals, never a deployable clean-feature oracle."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as functional
from . import feature_conditioned_trial as parent
from . import multiview_trial as base
from . import variance_trial as varf
from .feature_conditioned import scratch_feature_conditioned,unit_power
from .source_inverse import panel,SCENES
from .source_pair import read,validate_source
from .source_style import metrics
from .cache import tensor,load_F0,check_F0_state
from .main_split_eval import valid_identity
from .cost_probe import write

RUN='20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.feature_condition_diagnostic'
PARENT_COMMIT='c8659b15bb3d316325867b154eede51f1a61e247'
REFERENCE_RUN='20261101-diagnostic-Restore-source-style-equivariance-s2026092701-r01'
PATHS=('actual','zero_feature','no_FiLM','clean_feature_oracle','raw_preprojection')
METRICS=('nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement','complex_power')
LATENT_METRICS=('feature_cosine_distance','feature_standardized_mse','clean_decision_agreement')


def source_paths(model,received,clean_features):
    """Reuse one encoder; clean teacher enters only this explicit source diagnostic."""
    if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
        raise ValueError('Source FP32 IQ2x256 required')
    if clean_features.shape!=(len(received),160) or clean_features.dtype!=torch.float32:
        raise ValueError('Source clean continuous fused160 reference required')
    value=model.stem(torch.cat((received,model.coordinates.expand(len(received),-1,-1)),1));skips=[]
    for encoder,down in zip(model.encoders,model.down):
        value=encoder(value);skips.append(value);value=down(value)
    tokens=model.bottleneck(value.transpose(1,2)+model.position)
    predicted=model.feature_regressor(tokens.mean(1))
    outputs={}
    for path,feature in [('actual',predicted),('zero_feature',torch.zeros_like(predicted)),
            ('no_FiLM',None),('clean_feature_oracle',clean_features)]:
        value=tokens.transpose(1,2)
        if feature is not None:value=model.condition(value,feature,model.conditions[0])
        for up,decoder,skip,layer in zip(model.up,model.decoders,reversed(skips),model.conditions[1:]):
            value=functional.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False)
            value=up(torch.cat((value,skip),1))
            if feature is not None:value=model.condition(value,feature,layer)
            value=decoder(value)
        raw=received+model.output(value)
        outputs[path]=unit_power(raw)
        if path=='actual':outputs['raw_preprojection']=raw
    return {k:outputs[k] for k in PATHS},predicted


def validate_records(records):
    if len(records['scene'])!=10240 or set(records['scene'])!=set(SCENES):
        raise ValueError('Complete fixed512 source packets x20 original draws required')
    selected=None
    for epoch in range(1,21):
        indices=np.asarray(records['source_index'])[records['epoch']==epoch]
        if len(indices)!=512 or len(set(indices.tolist()))!=512:
            raise ValueError('Unique full512 packets in every source draw required')
        if selected is None:selected=indices
        elif not np.array_equal(indices,selected):raise ValueError('Same source panel order in every draw required')
    if set(records['epoch'])!=set(range(1,21)):raise ValueError('Exactly20 stored source draws required')


def summarize(records):
    validate_records(records);result={}
    for scene in ('ALL',*SCENES):
        mask=np.ones(10240,bool) if scene=='ALL' else records['scene']==scene
        group=dict(count=int(mask.sum()),paths={},latent={},paired_against_actual={})
        for path in PATHS:
            group['paths'][path]={k:float(records[path+'__'+k][mask].mean()) for k in METRICS}
        group['latent']={k:float(records['latent__'+k][mask].mean()) for k in LATENT_METRICS}
        before=records['actual__clean_decision_agreement'].astype(bool)
        for path in (*PATHS[1:],'latent'):
            after=records[path+'__clean_decision_agreement'].astype(bool)
            repair=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
            group['paired_against_actual'][path]=dict(repairs=repair,damage=damage,net=repair-damage)
        result[scene]=group
    return result


def load_parent(root,device):
    folder=root/'runs'/parent.RUN
    if read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Parent full Main run must already finish')
    done=read(folder/'rows'/parent.ARM/'training/complete.json')
    if (done['code_commit'],done['epoch'],done['updates'])!=(PARENT_COMMIT,30,7410) or any(done[k] for k in
            ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
        raise ValueError('Matching original source-only scratch parent required')
    p=torch.load(folder/'rows'/parent.ARM/'training/final.pt',map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=parent.configuration(parent.ARM)
            or (p['code_commit'],p['epoch'],p['updates'])!=(PARENT_COMMIT,30,7410)):
        raise ValueError('Exact original FeatureCond fixedfinal30 checkpoint required')
    model=scratch_feature_conditioned(2026092701);state=model.state_dict()
    if set(p['state'])!=set(state) or any(v.dtype!=torch.float32 or v.shape!=state[k].shape or not torch.isfinite(v).all() for k,v in p['state'].items()):
        raise ValueError('Exact finite original parent state required')
    model.load_state_dict(p['state'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old,roles=base.source(args);selected=panel(roles)
    model=load_parent(args.root,device)
    c=validate_source(read(args.inputs));reader,identity=load_F0(c,device);valid_identity(identity)
    models=[model,reader];snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
        clean=tensor(a[selected],device)
    finally:a._mmap.close()
    # First neural work is two actual L_s records; no query or updates.
    with torch.inference_mode():
        target=reader.features(clean[:2]);values,latent=source_paths(model,clean[:2],target)
        actual,aux=model(clean[:2],return_aux=True)
        if not torch.equal(values['actual'],actual) or not torch.equal(latent,aux['predicted_features']) or not torch.equal(values['raw_preprojection'],aux['raw_restored']):
            raise ValueError('Diagnostic actual path must exactly reproduce deployed parent')
        for value in values.values():
            logits=reader.classify_features(reader.features(value))
            if value.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(value).all() or not torch.isfinite(logits).all():
                raise ValueError('Actual frozen source checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_source_Ls=2,actual_and_latent_and_raw_exact=True,F0_identity=identity,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('FEATURE_PATH_SOURCE_SMOKE_PASS',flush=True)
    scale=varf.load_scale(args.root).to(device)
    with torch.inference_mode():
        target=reader.features(clean);target_prediction=reader.classify_features(target).argmax(1)
    records={k:[] for k in ['epoch','source_index','scene','Fclean_prediction',
        *[p+'__'+k for p in PATHS for k in (*METRICS,'prediction')],
        *['latent__'+k for k in (*LATENT_METRICS,'prediction')]]}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete original source draw required')
            with np.load(folder/'groups.npz',allow_pickle=False) as group:scenes=group['scene'][selected]
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original received source FP32 required')
                for offset in range(0,512,64):
                    take=selected[offset:offset+64];n=len(take);reference=target[offset:offset+n];prediction=target_prediction[offset:offset+n]
                    values,latent=source_paths(model,tensor(a[take],device),reference)
                    for path,value in values.items():
                        q=metrics(reader,value,clean[offset:offset+n],reference,prediction,scale)
                        q['complex_power']=value.square().sum(1).mean(1)
                        for key,v in q.items():records[path+'__'+key].extend(v.cpu().tolist())
                    p=reader.classify_features(latent).argmax(1)
                    q=dict(feature_cosine_distance=1-functional.cosine_similarity(latent,reference,dim=1),
                        feature_standardized_mse=((latent-reference).square()/scale).mean(1),clean_decision_agreement=p==prediction,prediction=p)
                    if any(not torch.isfinite(v).all() for v in q.values()):raise ValueError('Nonfinite source latent; never omit cases')
                    for key,v in q.items():records['latent__'+key].extend(v.cpu().tolist())
                    records['epoch'].extend([epoch]*n);records['source_index'].extend(take.tolist())
                    records['scene'].extend(scenes[offset:offset+n].tolist());records['Fclean_prediction'].extend(prediction.cpu().tolist())
            finally:a._mmap.close()
            print('FEATURE_PATH_SOURCE_DRAW_COMPLETE',epoch,512,flush=True)
    for m,before in zip(models,snapshots):check_F0_state(m,before)
    records={k:np.asarray(v) for k,v in records.items()};summary=summarize(records)
    # Reuse fixed identity-path source controls; no repeated DURA/FullFGrad forward.
    ref=args.root/'runs'/REFERENCE_RUN
    if read(ref/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete exact source reference required')
    reference_meta=read(ref/'results.json');valid_identity(reference_meta['F0_identity'])
    if reference_meta['code_commit']!='ac8015d9dcdcd9c5d1f05d7c09882a0591825d7d':raise ValueError('Exact existing source control execution required')
    with np.load(ref/'source_records.npz',allow_pickle=False) as f:
        keep=f['style']=='identity'
        for key in ['epoch','source_index','scene']:
            if not np.array_equal(f[key][keep],records[key]):raise ValueError('Exact original source reference order required')
        reference_prediction_differences=int(np.count_nonzero(f['Fclean_prediction'][keep]!=records['Fclean_prediction']))
        controls={}
        for method in ['DURA_v1','MSAT_FullFGrad']:
            controls[method]={k:f[method+'__actual__'+k][keep].copy() for k in (*METRICS[:4],'prediction')}
    if reference_prediction_differences:raise ValueError('Frozen Fclean reference changed; preserve and investigate')
    for scene,item in summary.items():
        mask=np.ones(10240,bool) if scene=='ALL' else records['scene']==scene
        item['frozen_source_controls']={m:{k:float(v[mask].mean()) for k,v in q.items() if k!='prediction'} for m,q in controls.items()}
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='FEATURE_PATH_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        parent_run=parent.RUN,parent_commit=PARENT_COMMIT,count=10240,source_Ls=128,source_Us=384,source_draws=20,
        paths=list(PATHS),latent_readout='same frozen F0 defaulthead on predicted continuous latent;diagnostic only',summary=summary,
        source_reference=REFERENCE_RUN,reference_Fclean_prediction_differences=reference_prediction_differences,
        neural_F0_feature_rows=51712,latent_defaulthead_rows=10240,encoder_rows=10240,decoder_rows=40960,smoke_rows_extra=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        checkpoint_selection=False,source_in_sample=True,agreement_is_TX_accuracy=False,clean_oracle_deployable=False,
        oracle_off_support_not_performance_bound=True,theoretical_ceiling=False,default_DURA_changed=False,
        seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    choices=varf.recovery_gpu_choice()
    while not choices:time.sleep(5);choices=varf.recovery_gpu_choice()
    admission=choices[0]
    command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source path diagnostic failed; preserve/no automatic retry')
    if read(out/'results.json')['status']!='FEATURE_PATH_DIAGNOSTIC_COMPLETE':raise ValueError('Missing complete source results')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git code identity required')
    if a.mode=='run':controller(a)
    else:worker(a)


if __name__=='__main__':main()
