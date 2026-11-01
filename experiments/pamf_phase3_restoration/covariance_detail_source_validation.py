"""Frozen final30 source-V IQ/feature/TX diagnosis; reuse previous source controls."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from . import covariance_guidance_source as base
from . import covariance_detail_trial as parent
from . import covariance_detail_latency as latency
from .covariance_inverse import dense_action
from .source_inverse import R07,SCENES
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import recovery_gpu_choice
from .main_split_eval import valid_identity
from .cost_probe import write

RUN='20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_detail_source_validation'
BASE_RUN=base.RUN
BASE_CODE='4178f020951d4799e912a941240bea36f45bedf1'
METHODS=('received','physical','guided','detail')


def dependency_ready(root):
    out=root/'runs'/latency.RUN
    if (out/'terminal.json').is_file() and read(out/'terminal.json')['exit_code']!=0:raise RuntimeError('Latency dependency technical failure; preserve/no retry')
    if not (out/'complete.json').is_file():return False
    if read(out/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete preceding source cost task required')
    return True


def baseline_reference(folder,identity):
    m=read(folder/'predictions_complete.json')
    expected=dict(status='SOURCE_V_PREDICTIONS_COMPLETE',code_commit=BASE_CODE,methods=['received','physical','guided'],views=list(SCENES),source_V_packets=1800,count=12600,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('F0_identity')!=identity:raise ValueError('Exact original frozen source V predictions/F0 required')
    if any(not (folder/(v+'.npz')).is_file() for v in SCENES):raise ValueError('All existing baseline source V views required')
    with np.load(folder/'references.npz',allow_pickle=False) as f:ref={k:f[k].copy() for k in f.files}
    if ref['ids'].shape!=(1800,) or len(set(ref['ids'].tolist()))!=1800:raise ValueError('Original source V reference IDs required')
    for k,shape in [('rx',(1800,)),('day',(1800,)),('clean',(1800,2,256)),('features',(1800,160)),('logits',(1800,6)),('scale',(160,))]:
        if ref[k].shape!=shape or not np.isfinite(ref[k]).all():raise ValueError('Finite original V reference shape required')
    if (ref['scale']<=0).any():raise ValueError('Positive existing source feature variance required')
    return ref


def load_final(root,device):
    folder=root/'runs'/parent.RUN/'rows'/parent.ARM/'training';done=read(folder/'complete.json');latency.validate_parent(done)
    p=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=parent.configuration(parent.ARM) or (p['code_commit'],p['epoch'],p['updates'])!=(latency.G_COMMIT,30,7410)):raise ValueError('Exact frozen final source G configuration required')
    G=parent.factory(parent.ARM);expected=G.state_dict()
    if set(p['state'])!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() for k,v in p['state'].items()):raise ValueError('Exact finite FP32 G state required')
    G.load_state_dict(p['state'],strict=True)
    return G.to(device).eval().requires_grad_(False),done['F0_identity']


def worker(args):
    if not dependency_ready(args.root) or not latency.parent_ready(args.root):raise ValueError('Parent Main and preceding cost task must finish first')
    start=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old=args.root/'runs'/R07/'cache';previous=args.root/'runs'/BASE_RUN
    G,parent_identity=load_final(args.root,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    if parent_identity!=identity:raise ValueError('Same original frozen F0 required')
    ref=baseline_reference(previous,identity);manifest=read(old/'validation/manifest.json')
    if manifest['count']!=1800 or manifest['roles']!={'V':1800} or manifest['label_reads']!=0 or not np.array_equal(ref['ids'],np.load(old/'validation/physical_ids.npy',allow_pickle=False)):raise ValueError('Original fixed V handles required')
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in [G,F]]
    roles=np.load(old/'train/roles.npy',allow_pickle=False);indices=np.flatnonzero(roles=='L_s')[:2]
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:x=tensor(a[indices],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        y=G(x);logits=F.classify_features(F.features(y))
        if y.shape!=(2,2,256) or logits.shape!=(2,6) or not torch.isfinite(y).all() or not torch.isfinite(logits).all():raise ValueError('Actual finalG/F0 source checkpoint smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_final30=True,actual_original_F0=True,optimizer_updates=0,TX_labels_read=0,target_inputs=0))
    print('COVARIANCE_DETAIL_SOURCE_V_REAL_SOURCE_SMOKE_PASS',flush=True)
    shutil.copyfile(previous/'references.npz',out/'references.npz')
    clean,target,clean_logits,scale=[tensor(ref[k],device) for k in ['clean','features','logits','scale']]
    with torch.inference_mode():
        for scene in SCENES:
            with np.load(previous/(scene+'.npz'),allow_pickle=False) as f:records={k:f[k].copy() for k in f.files}
            if not np.array_equal(records['ids'],ref['ids']):raise ValueError('Exact baseline source V order required')
            path=old/'validation_actions'/scene;m=read(path/'complete.json')
            if (m['count'],m['TX_labels_read'],m['target_reads'])!=(1800,0,0):raise ValueError('Existing source V action trace required')
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            values={k:[] for k in [*base.METRICS,'iq','features','logits','action_matrix_relative_mse','noise_log_mse','predicted_noise','true_noise','relative_solve_residual']}
            try:
                for offset in range(0,1800,32):
                    take=slice(offset,min(1800,offset+32));v={k:tensor(a[take],device) for k,a in arrays.items()}
                    y,aux=G(v['received'],return_aux=True);z=F.features(y);logits=F.classify_features(z)
                    action=dense_action(torch.complex(v['kernel'][:,0],v['kernel'][:,1]).to(torch.complex128),torch.complex(v['coefficients'][:,0],v['coefficients'][:,1]).to(torch.complex128),torch.complex(v['coefficients'][:,2],v['coefficients'][:,3]).to(torch.complex128))
                    metrics=base.source_metrics(y,clean[take],z,logits,target[take],clean_logits[take],scale)
                    metrics.update(iq=y,features=z,logits=logits,action_matrix_relative_mse=(aux['action_matrix']-action).square().sum((1,2))/action.square().sum((1,2)).clamp_min(1e-8),
                        noise_log_mse=(aux['variance'].log()-v['variance'].clamp_min(1e-7).log()).square(),predicted_noise=aux['variance'],true_noise=v['variance'],relative_solve_residual=aux['relative_solve_residual'])
                    if aux['relative_solve_residual'].max()>1e-8:raise ValueError('Posterior residual too large')
                    for k,q in metrics.items():
                        if not torch.isfinite(q).all():raise ValueError('Nonfinite source V output; no case omission')
                        values[k].extend(q.cpu().tolist())
            finally:
                for a in arrays.values():a._mmap.close()
            records.update({'detail__'+k:np.asarray(v) for k,v in values.items()})
            with (out/(scene+'.npz')).open('xb') as f:np.savez(f,**records)
            print('COVARIANCE_DETAIL_SOURCE_V_VIEW_COMPLETE',scene,1800,flush=True)
    for model,before in zip([G,F],snapshots):check_F0_state(model,before)
    write(out/'predictions_complete.json',dict(status='SOURCE_V_PREDICTIONS_COMPLETE',code_commit=args.commit,F0_identity=identity,methods=list(METHODS),views=list(SCENES),source_V_packets=1800,count=12600,
        TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,new_G_rows=12600,F0_new_output_rows=12600,F0_new_clean_rows=0,
        cached_clean_F0_rows_reused=1800,baseline_G_rows_recomputed=0,baseline_reused_from=BASE_RUN,baseline_code_commit=BASE_CODE,extra_source_smoke_G_rows=2,extra_source_smoke_F0_rows=2,seconds=time.perf_counter()-start))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    while not dependency_ready(args.root):time.sleep(5)
    choices=recovery_gpu_choice()
    while not choices:time.sleep(5);choices=recovery_gpu_choice()
    admission=choices[0];common=['--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs)]
    for mode,env in [('worker',dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID'])),('score',dict(os.environ,CUDA_VISIBLE_DEVICES=''))]:
        command=[sys.executable,'-u','-m',MODULE,mode,*common]
        with (out/'logs'/(mode+'.log')).open('xb') as log:p=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+mode+'.json'),dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission if mode=='worker' else None))
        observed=False
        while mode=='worker' and p.poll() is None:
            if not observed and (out/'smoke.json').is_file():
                proc=Path('/proc')/str(p.pid)
                try:
                    actual=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();cwd=os.readlink(proc/'cwd')
                    cuda=[x.decode() for x in (proc/'environ').read_bytes().split(bytes([0])) if x.startswith(b'CUDA_VISIBLE_DEVICES=')]
                    query=subprocess.check_output(['nvidia-smi','-i',admission['CUDA_ID'],'--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                    if MODULE+' worker ' in actual and args.commit in actual and cwd==os.getcwd() and cuda==['CUDA_VISIBLE_DEVICES='+admission['CUDA_ID']] and any(s.split(',')[0].strip()==admission['CUDA_ID'] and int(s.split(',')[1])==p.pid for s in query.splitlines()):
                        write(out/'runtime_worker_observation.json',dict(status='VERIFIED',pid=p.pid,CWD=cwd,command=actual,CUDA=cuda,GPU=query,source_smoke=read(out/'smoke.json'),log_bytes=(out/'logs/worker.log').stat().st_size,log_growth_from_zero=True,observer_PID=os.getpid()));observed=True
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
            time.sleep(.2)
        code=p.wait();write(out/('terminal-'+mode+'.json'),dict(pid=p.pid,exit_code=code))
        if code:raise RuntimeError('Frozen source V diagnostic failed; preserve/no retry')
    if read(out/'results.json')['status']!='SOURCE_V_GUIDANCE_ANALYZED':raise ValueError('Complete source V truth-last score required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','score']);p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual code identity required')
    if args.mode=='run':controller(args)
    elif args.mode=='worker':worker(args)
    else:base.RUN=RUN;base.METHODS=METHODS;base.score(args,comparison='detail')
