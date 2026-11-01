"""Frozen source parameter attribution; no deployed variant or target feedback."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import covariance_trial as parent
from . import multiview_trial as base
from .covariance_inverse import dense_action,gaussian_posterior
from .source_inverse import panel
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import recovery_gpu_choice
from .cost_probe import write

RUN='20261102-diagnostic-Restore-covariance-parameter-source-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_parameter_source'
G_COMMIT='bf29554a03e28a012eaaa134f5c7b3f7856f2268'
METHODS=('blind','true_noise','true_action','true_both')
METRICS=('iq_nmse','feature_cosine_distance','clean_decision_agreement','relative_solve_residual')


def parameter_pairs(estimated_action,estimated_noise,true_action,true_noise):
    return {'blind':(estimated_action,estimated_noise),'true_noise':(estimated_action,true_noise),
        'true_action':(true_action,estimated_noise),'true_both':(true_action,true_noise)}


def summarize(a):
    if len(a['epoch'])!=10240 or set(a['epoch'])!=set(range(1,21)):
        raise ValueError('All fixed source512x20 cases required')
    masks=[('overall','ALL',np.ones(10240,bool))]
    for key in ('scene','lock'):masks.extend((key,str(v),a[key]==v) for v in np.unique(a[key]))
    result=[]
    for method in METHODS:
        after=a[method+'__clean_decision_agreement'].astype(bool)
        for dimension,stratum,mask in masks:
            pairs={}
            for name in ('blind','DURA_v1','received'):
                before=a[name+'__clean_decision_agreement'].astype(bool)
                repairs=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
                pairs[name]=dict(repairs=repairs,damage=damage,net=repairs-damage)
            result.append(dict(method=method,dimension=dimension,stratum=stratum,count=int(mask.sum()),
                metrics={k:float(a[method+'__'+k][mask].mean()) for k in METRICS},
                parameter_errors={k:float(a[k][mask].mean()) for k in ['action_matrix_relative_mse','noise_log_mse','predicted_noise','true_noise']},pairs=pairs))
    return result


def load_model(root,device):
    folder=root/'runs'/parent.RUN/'rows'/parent.ARM/'training';done=read(folder/'complete.json')
    if (done['status'],done['code_commit'],done['epoch'],done['updates'],done['G_parameters'])!=(
            'SOURCE_TRAINED',G_COMMIT,30,7410,1230341):raise ValueError('Exact source-only CovarianceMAP final30 required')
    if any(done[k] for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
        raise ValueError('Original source-only scratch30 ancestry required')
    parent.install()
    return base.load(folder/'final.pt',parent.ARM,G_COMMIT,device)


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old,roles=base.source(args);selected=panel(roles);G=load_model(args.root,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    control=args.root/'runs/20261101-diagnostic-Restore-source-inverse-s2026092701-r01'
    previous=read(control/'results.json')
    if previous['F0_identity']!=identity or previous['count']!=10240:raise ValueError('Exact original source scalar control required')
    with np.load(control/'source_records.npz',allow_pickle=False) as f:cached={k:f[k].copy() for k in f.files}
    models=[G,F];snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a[selected],device)
    finally:a._mmap.close()
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[selected[:2]],device)
    finally:a._mmap.close()
    mean=G.prior_mean.double();cov=G.prior_covariance.double()
    with torch.inference_mode():
        aux=G.estimate(source2);expected,residual=gaussian_posterior(aux['action_matrix'],source2.flatten(1).double(),mean,cov,aux['variance'].double())
        expected=expected.float().reshape(2,2,256)
        if not torch.equal(expected,G(source2)):raise ValueError('Blind source readout must equal actual deployed G')
        prediction=F.classify_features(F.features(expected))
        if prediction.shape!=(2,6) or not torch.isfinite(prediction).all() or residual.max()>1e-8:
            raise ValueError('Actual final30/source2 posterior and F0 smoke failed')
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_rows=2,blind_equals_actual_deployment=True,actual_original_F0=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('COVARIANCE_PARAMETER_REAL_SOURCE_SMOKE_PASS',flush=True)
    with torch.inference_mode():target=F.features(clean);target_prediction=F.classify_features(target).argmax(1)
    records={key:[] for key in ['epoch','source_index','scene','lock','Fclean_prediction','action_matrix_relative_mse','noise_log_mse','predicted_noise','true_noise',
        *[m+'__'+key for m in METHODS for key in (*METRICS,'output_prediction')]]}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Original complete source draw required')
            with np.load(folder/'groups.npz',allow_pickle=False) as f:groups={key:f[key][selected] for key in ['scene','lock']}
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ['received','kernel','coefficients','variance']}
            try:
                for offset in range(0,512,32):
                    take=selected[offset:offset+32];n=len(take);v={key:tensor(a[take],device) for key,a in arrays.items()}
                    aux=G.estimate(v['received']);reference=clean[offset:offset+n];energy=reference.square().mean((1,2))
                    true=dense_action(torch.complex(v['kernel'][:,0],v['kernel'][:,1]).to(torch.complex128),
                        torch.complex(v['coefficients'][:,0],v['coefficients'][:,1]).to(torch.complex128),
                        torch.complex(v['coefficients'][:,2],v['coefficients'][:,3]).to(torch.complex128))
                    parameters=dict(action_matrix_relative_mse=(aux['action_matrix']-true).square().sum((1,2))/true.square().sum((1,2)).clamp_min(1e-8),
                        noise_log_mse=(aux['variance'].log()-v['variance'].clamp_min(1e-7).log()).square(),predicted_noise=aux['variance'],true_noise=v['variance'])
                    for method,(action,noise) in parameter_pairs(aux['action_matrix'],aux['variance'].double(),true,v['variance'].double()).items():
                        restored,residual=gaussian_posterior(action,v['received'].flatten(1).double(),mean,cov,noise)
                        if residual.max()>1e-8:raise ValueError('Posterior residual too large;preserve without fallback')
                        restored=restored.float().reshape(n,2,256);features=F.features(restored);p=F.classify_features(features).argmax(1)
                        metrics=dict(iq_nmse=(restored-reference).square().mean((1,2))/energy,
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(features,target[offset:offset+n],dim=1),
                            clean_decision_agreement=p==target_prediction[offset:offset+n],relative_solve_residual=residual)
                        if any(not torch.isfinite(value).all() for value in metrics.values()):raise ValueError('Nonfinite source parameter path')
                        for key,value in metrics.items():records[method+'__'+key].extend(value.cpu().tolist())
                        records[method+'__output_prediction'].extend(p.cpu().tolist())
                    for key,value in parameters.items():
                        if not torch.isfinite(value).all():raise ValueError('Nonfinite source parameter error')
                        records[key].extend(value.cpu().tolist())
                    for key,value in [('epoch',[epoch]*n),('source_index',take.tolist()),('scene',groups['scene'][offset:offset+n].tolist()),
                            ('lock',groups['lock'][offset:offset+n].tolist()),('Fclean_prediction',target_prediction[offset:offset+n].cpu().tolist())]:records[key].extend(value)
            finally:
                for a in arrays.values():a._mmap.close()
            print('COVARIANCE_PARAMETER_SOURCE_DRAW_COMPLETE',epoch,512,flush=True)
    records={key:np.asarray(value) for key,value in records.items()}
    for key in ['epoch','source_index','scene','lock']:
        if not np.array_equal(records[key],cached[key]):raise ValueError('Exact old source case alignment required')
    for method in ['DURA_v1','received']:
        records[method+'__clean_decision_agreement']=cached[method+'__clean_decision_agreement']
        records[method+'__iq_nmse']=cached[method+'__nmse'];records[method+'__feature_cosine_distance']=cached[method+'__feature_cosine_distance']
    for model,before in zip(models,snapshots):check_F0_state(model,before)
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='COVARIANCE_PARAMETER_SOURCE_COMPLETE',run_id=RUN,code_commit=args.commit,parent_G_commit=G_COMMIT,F0_identity=identity,
        source_packets=512,source_draws=20,source_cases=10240,methods=list(METHODS),summary=summarize(records),
        G_estimator_rows=10240,posterior_rows=40960,F0_output_feature_rows=40960,F0_clean_feature_rows=512,
        smoke_extra_G_estimator_rows=4,smoke_extra_posterior_rows=4,smoke_extra_F0_feature_rows=2,
        TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        source_in_sample=True,physical_oracle_deployable=False,agreement_is_TX_accuracy=False,theoretical_ceiling=False,checkpoint_selection=False,
        default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def spare_card(root):
    folder=root/'runs'/parent.RUN
    if any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):
        raise RuntimeError('Parent technical failure before diagnostic;preserve without retry')
    if (folder/('terminal-'+parent.ARM+'-cross_both.json')).is_file():return (4,6)
    path=folder/('submitted-'+parent.ARM+'-cross_both.json')
    if not path.is_file():return ()
    job=read(path);gpu=job['admission']['GPU'];process=Path('/proc')/str(job['pid'])
    try:
        command=(process/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        if gpu in (4,6) and parent.MODULE+' predict ' in command and parent.RUN in command and '--task cross_both' in command:
            return (6,) if gpu==4 else (4,)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return ()


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty diagnostic controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    print('WAITING_FOR_PARENT_FINAL_MAIN_GPU_STAGE_OR_EXIT0',flush=True)
    choices=[]
    while not choices:
        allowed=spare_card(args.root)
        if allowed:choices=[g for g in recovery_gpu_choice() if g['GPU'] in allowed]
        if not choices:time.sleep(5)
    admission=choices[0];command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Frozen source parameter diagnostic failed;preserve/no automatic retry')
    if read(out/'results.json')['status']!='COVARIANCE_PARAMETER_SOURCE_COMPLETE':raise ValueError('All source cases required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    if a.mode=='run':controller(a)
    else:worker(a)
