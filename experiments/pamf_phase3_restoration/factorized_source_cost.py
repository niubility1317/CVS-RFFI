"""Zero-update source forward/backward cost for exact physical-loss routes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json
import numpy as np
import torch
from . import covariance_feature_trial as parent
from . import multiview_trial as base
from .covariance_inverse import dense_action
from .covariance_model import source_action_loss
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .variance_trial import load_scale,recovery_gpu_choice
from .cost_probe import write

RUN='20261102-diagnostic-Restore-factorized-source-cost-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.factorized_source_cost'
G_COMMIT='5359a1b82b1fa5d640a394c3e3a77ce3b945fe1e'
ROUTES=('dense','factor')


def validate_parent_metadata(done):
    expected=dict(status='SOURCE_TRAINED',code_commit=G_COMMIT,arm=parent.ARM,epoch=30,updates=7410,
        G_parameters=1230341,training_stages=1,source_actual_IQ_F0_guidance=True,feature_weight=.01,
        online_F0_feature_rows=236880,F0_clean_target_rows=63000,training_posterior_calls=7410)
    if any(done.get(k)!=v for k,v in expected.items()) or any(done.get(k)!=0 for k in [
            'G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
        raise ValueError('Exact original source-only posterior-feature final30 required')
    if done.get('F0_identity',{}).get('source_code_commit')!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':
        raise ValueError('Original frozen ground Tri-SIG-TIME SSL required')


def logical_step(G,F,values,clean,target,scale,route,*,capture=False):
    if route not in ROUTES:raise ValueError('Only preregistered dense/factor source routes allowed')
    n=len(clean);parent.feature_window(n);G.zero_grad(set_to_none=True)
    total=clean.new_zeros(())
    for offset in range(0,n,32):
        part=slice(offset,min(offset+32,n));v={k:x[part] for k,x in values.items()}
        aux=G.estimate(v['received'],dense=(route=='dense'))
        physical,_=source_action_loss(aux,v['kernel'],v['coefficients'],v['variance']);guide=None
        if offset==0:
            if route=='factor':
                aux['action_matrix']=dense_action(aux['kernel'].to(torch.complex128),aux['main'].to(torch.complex128),aux['image'].to(torch.complex128))
            guide,info=parent.posterior_feature_guide(F,aux,v['received'],clean[part],target[part],scale,G)
            if info['relative_solve_residual'].max()>1e-8:raise ValueError('Source posterior residual too large')
        loss=parent.micro_objective(physical,guide,n)
        if not torch.isfinite(loss):raise ValueError('Nonfinite source benchmark objective')
        loss.backward();total+=loss.detach()
    gradients={k:p.grad.detach().clone() for k,p in G.named_parameters()} if capture else None
    norm=torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
    return dict(loss=total,grad_norm=norm,gradients=gradients)


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise ValueError('One admitted physical GPU required')
    device='cuda:0';old,roles=base.source(args)
    folder=args.root/'runs'/parent.RUN/'rows'/parent.ARM/'training'
    done=read(folder/'complete.json');validate_parent_metadata(done);parent.install()
    G=base.load(folder/'final.pt',parent.ARM,G_COMMIT,device).requires_grad_(True)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    if identity!=done['F0_identity'] or any(p.requires_grad for p in F.parameters()):raise ValueError('Same frozen source F0 required')
    with np.load(folder/'data_schedule.npz',allow_pickle=False) as f:
        selected=f['orders'][29,:256].copy();draw=int(f['draws'][29])
    if draw!=10 or selected.shape!=(256,) or len(set(selected.tolist()))!=256:raise ValueError('Fixed final-epoch first logical source batch required')
    source=old/'epochs'/f'{draw:02d}';source_done=read(source/'complete.json')
    if (source_done['count'],source_done['TX_labels_read'],source_done['target_reads'])!=(63000,0,0):raise ValueError('Original R07 source draw required')
    def subset(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return tensor(a[selected],device)
        finally:a._mmap.close()
    clean=subset(old/'train/clean.npy')
    values={key:subset(source/(key+'.npy')) for key in ['received','kernel','coefficients','variance']}
    if clean.shape!=(256,2,256) or any(not torch.isfinite(x).all() for x in [clean,*values.values()]):raise ValueError('Finite fixed source IQ/parameters required')
    scale=load_scale(args.root).to(device)
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in [G,F]]
    out=args.root/'runs'/RUN;take=np.flatnonzero(roles[selected]=='L_s')[:2]
    if len(take)!=2:raise ValueError('Two legal Ls source smoke records required')
    index=torch.tensor(take.tolist(),device=device,dtype=torch.long)
    with torch.no_grad():
        restored=G(values['received'][index]);logits=F.classify_features(F.features(restored))
        if logits.shape!=(2,6) or not torch.isfinite(logits).all() or not torch.isfinite(restored).all():raise ValueError('Actual frozen final G/F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',real_Ls=2,actual_G_commit=G_COMMIT,actual_original_F0=True,optimizer_updates=0,TX_labels_read=0,target_inputs=0))
    print('FACTORIZED_COST_REAL_SOURCE_SMOKE_PASS',flush=True)
    tick=time.perf_counter()
    with torch.no_grad():target=F.features(clean).detach()
    torch.cuda.synchronize();target_seconds=time.perf_counter()-tick;G.train()
    reference=logical_step(G,F,values,clean,target,scale,'dense',capture=True)
    factor=logical_step(G,F,values,clean,target,scale,'factor',capture=True)
    torch.testing.assert_close(factor['loss'],reference['loss'],rtol=2e-6,atol=2e-7)
    differences=[];norms=[];max_abs=0.
    for key,expected in reference['gradients'].items():
        actual=factor['gradients'][key];torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
        delta=(actual-expected).double();differences.append(delta.square().sum());norms.append(expected.double().square().sum())
        max_abs=max(max_abs,float(delta.abs().max()))
    relative=float(torch.stack(differences).sum().sqrt()/torch.stack(norms).sum().sqrt().clamp_min(1e-30))
    parity=dict(status='PASS',source_rows=256,all_G_parameter_gradients_checked=True,gradient_max_abs=max_abs,gradient_relative_L2=relative,
        dense_loss=float(reference['loss']),factor_loss=float(factor['loss']),loss_rtol=2e-6,loss_atol=2e-7,gradient_rtol=2e-4,gradient_atol=2e-6,
        before_clip_comparison=True,optimizer_updates=0)
    write(out/'parity.json',parity);del reference,factor,differences,norms
    records=[]
    schedule=[('warmup',route) for _ in range(2) for route in ROUTES]
    schedule += [('timed',route) for pair in range(10) for route in (ROUTES if pair%2==0 else ROUTES[::-1])]
    with (out/'steps.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for sequence,(kind,route) in enumerate(schedule):
            G.zero_grad(set_to_none=True);torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();begin.record()
            result=logical_step(G,F,values,clean,target,scale,route)
            end.record();torch.cuda.synchronize();wall_ms=1000*(time.perf_counter()-wall)
            record=dict(sequence=sequence,kind=kind,route=route,source_rows=256,microbatch=32,feature_rows=32,
                wall_ms=wall_ms,CUDA_event_ms=begin.elapsed_time(end),baseline_allocated_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_incremental_bytes=torch.cuda.max_memory_allocated()-baseline,
                loss=float(result['loss']),preclip_gradient_norm=float(result['grad_norm']))
            log.write(json.dumps(record,allow_nan=False)+'\n');records.append(record)
            print('FACTORIZED_SOURCE_COST_STEP',json.dumps(record,allow_nan=False),flush=True)
    G.zero_grad(set_to_none=True)
    for m,before in zip([G,F],snapshots):check_F0_state(m,before)
    summary={}
    for route in ROUTES:
        selected_records=[x for x in records if x['kind']=='timed' and x['route']==route]
        summary[route]=dict(timed_steps=len(selected_records),median_wall_ms=float(np.median([x['wall_ms'] for x in selected_records])),
            p90_wall_ms=float(np.quantile([x['wall_ms'] for x in selected_records],.9)),median_CUDA_event_ms=float(np.median([x['CUDA_event_ms'] for x in selected_records])),
            max_allocated_bytes=max(x['peak_allocated_bytes'] for x in selected_records),max_incremental_bytes=max(x['peak_incremental_bytes'] for x in selected_records))
    write(out/'results.json',dict(status='SOURCE_FORWARD_BACKWARD_COST_COMPLETE',run_id=RUN,code_commit=args.commit,parent_G_commit=G_COMMIT,F0_identity=identity,
        source_draw=draw,source_selection='E30 saved first logical256/sameorder/noTX-qualityselection',source_rows=256,
        source_L_s=int((roles[selected]=='L_s').sum()),source_U_s=int((roles[selected]=='U_s').sum()),
        logical_batch=256,microbatch=32,feature_rows_per_step=32,warmups_per_route=2,timed_steps_per_route=10,parity_steps_per_route=1,
        total_logical_backward_steps=26,G_estimator_visits=6656,posterior_F0_output_visits=832,F0_clean_target_visits=256,
        smoke_extra_G_posterior_F0_output_visits=2,defaulthead_smoke_rows=2,source_target_cache_seconds=target_seconds,
        summary=summary,parity=parity,isolated_wall_ratio_factor_over_dense=summary['factor']['median_wall_ms']/summary['dense']['median_wall_ms'],
        timing_scope='Forward/backward/finite-and-solve checks/clip only;no optimizer step/data IO/epoch metric logging. Cleared gradients before region;GPU synchronous wall/event.',
        model_state_unchanged=True,F0_state_unchanged=True,training_epochs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,
        performance_evaluation=False,same_quality_full_training_speedup_proven=False,default_DURA_changed=False,goal_complete=False,seconds=time.perf_counter()-started))


def spare_card(root):
    folder=root/'runs'/parent.RUN
    if any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):raise RuntimeError('Parent technical failure;preserve without retry')
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
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty cost controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    print('WAITING_FOR_PARENT_LAST_MAIN_GPU_STAGE_OR_EXIT0',flush=True);choices=[]
    while not choices:
        allowed=spare_card(args.root)
        if allowed:choices=[g for g in recovery_gpu_choice() if g['GPU'] in allowed]
        if not choices:time.sleep(5)
    admission=choices[0];command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source cost worker failed;preserve without automatic retry')
    if read(out/'results.json')['status']!='SOURCE_FORWARD_BACKWARD_COST_COMPLETE':raise ValueError('Complete source cost matrix required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    if a.mode=='run':controller(a)
    else:worker(a)
