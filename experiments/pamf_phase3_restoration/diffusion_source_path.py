"""Frozen source-only internal sampling path; no alternative deployment selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import multiview_trial as base
from . import residual_diffusion_trial as parent
from .source_inverse import panel
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import load_scale,recovery_gpu_choice
from .cost_probe import write

RUN='20261102-diagnostic-Restore-diffusion-source-path-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.diffusion_source_path'
G_COMMIT='5e5921c975678367a5d0a00ee72d4619e5f6927e'
METRICS=('iq_nmse','feature_cosine_distance','feature_standardized_mse','clean_decision_agreement',
    'complex_power','output_power_abs_error','change_from_first_nmse')


def trajectory(model,received):
    state=model.fixed_latent.expand_as(received)*model.sigma_steps[0];items=[]
    for index in range(8):
        sigma,next_sigma=model.sigma_steps[index:index+2]
        denoised=model.denoise(state,received,sigma.expand(len(received)))
        derivative=(state-denoised)/sigma;proposal=state+(next_sigma-sigma)*derivative
        if index<7:
            correction=model.denoise(proposal,received,next_sigma.expand(len(received)))
            next_derivative=(proposal-correction)/next_sigma
            state=state+(next_sigma-sigma)*(derivative+next_derivative)/2
        else:state=proposal
        # The final Euler arithmetic matches the actual FP32 sampler exactly.
        items.append(received+(state if index==7 else denoised))
    return items


def summarize(a):
    if len(a['step'])!=92160 or set(a['step'])!=set(range(9)):raise ValueError('Received plus8steps x10240 source cases required')
    first={k:v[a['step']==1] for k,v in a.items()};result=[]
    if len(first['epoch'])!=10240 or set(first['epoch'])!=set(range(1,21)):raise ValueError('All20 fixed source draws required')
    selected=first['source_index'][:512]
    if len(set(selected.tolist()))!=512:raise ValueError('Unique fixed512 source packets required')
    for epoch in range(1,21):
        if not np.array_equal(first['source_index'][first['epoch']==epoch],selected):raise ValueError('Same512 source order required')
    received=a['clean_decision_agreement'][a['step']==0].astype(bool)
    for step in range(9):
        q={k:v[a['step']==step] for k,v in a.items()}
        if any(not np.array_equal(q[k],first[k]) for k in ['epoch','source_index','scene','lock']):raise ValueError('Every step must refer to the same cases')
        if any(q[k].shape!=(10240,) or not np.isfinite(q[k]).all() for k in METRICS):raise ValueError('Finite complete source path required')
        if not np.isin(q['clean_decision_agreement'],[0,1]).all():raise ValueError('Binary Fclean agreement required')
        masks=[('overall','ALL',np.ones(10240,bool))]
        for key in ['scene','lock']:masks.extend((key,str(v),q[key]==v) for v in np.unique(q[key]))
        after=q['clean_decision_agreement'].astype(bool)
        for dimension,stratum,mask in masks:
            pairs={}
            for name,before in [('first',first['clean_decision_agreement'].astype(bool)),('received',received)]:
                repairs=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
                if repairs-damage!=int(after[mask].sum())-int(before[mask].sum()):raise ValueError('Paired counts must reconcile')
                pairs['paired_against_'+name]=dict(repairs=repairs,damage=damage,net=repairs-damage)
            result.append(dict(step=step,dimension=dimension,stratum=stratum,count=int(mask.sum()),metrics={k:float(q[k][mask].mean()) for k in METRICS},**pairs))
    return result


def load_model(root,device):
    folder=root/'runs'/parent.RUN/'rows'/parent.ARM/'training';done=read(folder/'complete.json')
    if (done['code_commit'],done['epoch'],done['updates'],done['G_parameters'])!=(G_COMMIT,30,7410,1614608):raise ValueError('Exact frozen residual diffusion final30 required')
    if any(done[k] for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):raise ValueError('Source-only scratch G ancestry required')
    parent.install(load_scale(root))
    return base.load(folder/'final.pt',parent.ARM,G_COMMIT,device)


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old,roles=base.source(args);selected=panel(roles);G=load_model(args.root,device)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    models=[G,F];snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a[selected],device)
    finally:a._mmap.close()
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[selected[:2]],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        path=trajectory(G,source2)
        if not torch.equal(path[-1],G(source2)):raise ValueError('Source path final must equal actual deployed G')
        z=F.classify_features(F.features(path[-1]))
        if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual source2 G/F0 smoke required')
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,actual_final30=True,eighth_equals_deployment=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
    print('DIFFUSION_PATH_SOURCE_SMOKE_PASS',flush=True)
    scale=load_scale(args.root).to(device)
    with torch.inference_mode():target=F.features(clean);target_prediction=F.classify_features(target).argmax(1)
    records={k:[] for k in ['step','epoch','source_index','scene','lock','Fclean_prediction','output_prediction',*METRICS]}
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
                    energy=reference.square().mean((1,2));path=trajectory(G,received);first=path[0]
                    for step,value in enumerate([received,*path]):
                        z=F.features(value);p=F.classify_features(z).argmax(1);power=value.square().sum(1).mean(1)
                        metrics=dict(iq_nmse=(value-reference).square().mean((1,2))/energy,
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(z,features,dim=1),
                            feature_standardized_mse=((z-features).square()/scale).mean(1),clean_decision_agreement=p==pred,
                            complex_power=power,output_power_abs_error=(power-1).abs(),change_from_first_nmse=(value-first).square().mean((1,2))/energy)
                        if any(not torch.isfinite(v).all() for v in metrics.values()):raise ValueError('Nonfinite source path;never omit cases')
                        for k,v in metrics.items():records[k].extend(v.cpu().tolist())
                        for k,v in [('step',[step]*n),('epoch',[epoch]*n),('source_index',take.tolist()),('scene',groups['scene'][offset:offset+n].tolist()),
                            ('lock',groups['lock'][offset:offset+n].tolist()),('Fclean_prediction',pred.cpu().tolist()),('output_prediction',p.cpu().tolist())]:records[k].extend(v)
            finally:a._mmap.close()
            print('DIFFUSION_PATH_SOURCE_DRAW_COMPLETE',epoch,512,flush=True)
    for model,before in zip(models,snapshots):check_F0_state(model,before)
    records={k:np.asarray(v) for k,v in records.items()};summary=summarize(records)
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    write(out/'results.json',dict(status='DIFFUSION_SOURCE_PATH_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        parent_G_commit=G_COMMIT,source_packets=512,source_draws=20,source_cases=10240,path_rows=92160,summary=summary,
        G_core_rows=153600,F0_output_feature_rows=92160,F0_clean_feature_rows=512,smoke_extra_G_core_rows=60,smoke_extra_F0_feature_rows=2,
        TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        checkpoint_selection=False,source_in_sample=True,agreement_is_TX_accuracy=False,theoretical_ceiling=False,default_DURA_changed=False,
        seconds=time.perf_counter()-started,goal_complete=False))


def parent_ready(root):
    folder=root/'runs'/parent.RUN
    if any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):
        raise RuntimeError('Parent technical failure before diagnostic;preserve without retry')
    if (folder/('terminal-'+parent.ARM+'-cross_both.json')).is_file():return True
    p=folder/('submitted-'+parent.ARM+'-cross_both.json')
    if not p.is_file():return False
    job=read(p)
    if job['admission']['GPU']!=4:return False
    process=Path('/proc')/str(job['pid'])
    try:
        command=(process/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        return parent.MODULE+' predict ' in command and parent.RUN in command and '--task cross_both' in command
    except (FileNotFoundError,ProcessLookupError,PermissionError):return False


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty source diagnostic controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    # Wait for its last GPU stage on GPU4 or its end, so it cannot later choose GPU6.
    print('WAITING_FOR_PARENT_LAST_GPU_STAGE_ON4_OR_COMPLETION',flush=True)
    while not parent_ready(args.root):time.sleep(5)
    # Parent remains untouched; only the spare preregistered GPU6 is admitted.
    choices=[x for x in recovery_gpu_choice() if x['GPU']==6]
    while not choices:time.sleep(5);choices=[x for x in recovery_gpu_choice() if x['GPU']==6]
    a=choices[0];cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=a['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=a))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source diagnostic failed;preserve/no automatic retry')
    if read(out/'results.json')['status']!='DIFFUSION_SOURCE_PATH_COMPLETE':raise ValueError('Complete fixed source path required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    if a.mode=='run':controller(a)
    else:worker(a)
