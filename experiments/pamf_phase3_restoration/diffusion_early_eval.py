"""Frozen source-selected fourth denoised readout; no new G training or target selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from . import multiview_trial as base
from . import residual_diffusion_trial as parent
from .diffusion_source_fixes import early_denoised4
from .diffusion_source_path import trajectory,load_model
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import recovery_gpu_choice
from .cost_probe import write

RUN='20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.diffusion_early_eval'
ARM='TF_DiffusionEarly4'
STREAMS=('raw','DURA_v1','full8','candidate')


class FrozenEarly4(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-record FP32 IQ2x256 required')
        return early_denoised4(self.model,received)


def predict(args,out):
    c=validate_source(read(args.inputs));old,_=base.source(args);model=load_model(args.root,'cuda:0')
    G=FrozenEarly4(model).eval().requires_grad_(False)
    F,identity=load_F0(c,'cuda:0');base.main_eval.valid_identity(identity)
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in [G,F]]
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[:2],'cuda:0')
    finally:a._mmap.close()
    out.mkdir()
    with torch.inference_mode():
        restored=G(source2);expected=trajectory(model,source2)[3]
        if not torch.equal(restored,expected):raise ValueError('Exact source-selected fourth readout required')
        logits=F.classify_features(F.features(restored))
        if not torch.isfinite(restored).all() or logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual frozen parentG/early4/F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',source_rows=2,selected_readout=4,core_calls=7,
        exact_source_internal_point=True,TX_labels_read=0,target_inputs=0,model_updates=0,new_G_training_updates=0))
    from experiments.cric_dura_comparison.evaluation import handle
    o=base.main_eval.official(args.root);manifest,ids,positions=handle(o['tests'][args.task]['views_root'],o['classes'])
    if len(ids)!=base.main_eval.COUNTS[args.task]:raise ValueError('Full original Main population required')
    previous=args.root/'runs'/parent.RUN/'rows'/parent.ARM/'prediction'/args.task
    done=read(previous/'complete.json');base.main_eval.valid_identity(done['F0_identity'])
    if (done['status'],done['count'],done['model_state_updates'],done['code_commit'])!=('PREDICTIONS_COMPLETE',len(ids),0,'5e5921c975678367a5d0a00ee72d4619e5f6927e'):raise ValueError('Exact frozen full8 comparison required')
    with np.load(previous/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(f['ids'],ids):raise ValueError('Fixed Main physical order differs')
        values={s+'__'+v:f[('candidate' if s=='full8' else s)+'__'+v].copy() for s in STREAMS[:-1] for v in base.main_eval.VIEWS}
    for view in base.main_eval.VIEWS:
        path=manifest['clean_ref'] if view=='clean' else str(Path(manifest['observations_root'])/(view+'.npy'))
        a=np.load(path,mmap_mode='r',allow_pickle=False);decisions=[]
        if a.ndim!=3 or a.shape[1:]!=(2,256) or a.dtype!=np.float32:raise ValueError('Original Main FP32 IQ required')
        try:
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    take=slice(start,start+128) if positions is None else positions[start:start+128]
                    restored=G(tensor(a[take],'cuda:0'));logits=F.classify_features(F.features(restored))
                    if not torch.isfinite(restored).all() or not torch.isfinite(logits).all():raise ValueError('Invalid frozen early4 IQ/logits')
                    decisions.extend(logits.argmax(1).cpu().tolist())
        finally:a._mmap.close()
        values['candidate__'+view]=np.asarray(decisions,np.int64)
        print('DIFFUSION_EARLY4_MAIN_VIEW_COMPLETE',args.task,view,len(ids),flush=True)
    for m,snapshot in zip([G,F],snapshots):check_F0_state(m,snapshot)
    with (out/'predictions.npz').open('xb') as f:np.savez(f,ids=ids,**values)
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',run_id=RUN,arm=ARM,task=args.task,count=len(ids),
        code_commit=args.commit,F0_identity=identity,readouts=list(STREAMS),views=list(base.main_eval.VIEWS),
        baseline_reused_from=parent.RUN,truth_read=False,query_fit=False,model_state_updates=0,new_G_training_updates=0,
        frozen_parent_training_epochs=30,frozen_parent_training_updates=7410,selected_readout=4,deployed_core_calls=7))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty evaluation controller required')
    folder=args.root/'runs'/parent.RUN
    if read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE' or any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):
        raise ValueError('Fixed complete original full8 comparison required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();(out/'rows').mkdir();row=out/'rows'/ARM;row.mkdir();(row/'prediction').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit,new_G_training_updates=0))
    for stage in ['cross_rx','cross_both','score']:
        if stage=='score':admission=None;cmd=base.command(args,'score',ARM)
        else:
            choices=[x for x in recovery_gpu_choice() if x['GPU']==4]
            while not choices:time.sleep(5);choices=[x for x in recovery_gpu_choice() if x['GPU']==4]
            admission=choices[0];cmd=base.command(args,'predict',ARM,stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+ARM+'-'+stage+'.json'),dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=admission))
        code=p.wait();write(out/('terminal-'+ARM+'-'+stage+'.json'),dict(pid=p.pid,exit_code=code))
        if code:raise RuntimeError('Frozen early4 evaluation failed;preserve/no automatic retry')
    if read(row/'results.json')['status']!='ANALYZED':raise ValueError('Full original Main results required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',new_G_training_updates=0,all_Main_two_populations_seven_scored=True,default_DURA_changed=False,goal_complete=False))


def install():
    base.RUN=RUN;base.MODULE=MODULE;base.ARMS=(ARM,);base.STREAMS=STREAMS


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','predict','score']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--arm',choices=[ARM]);p.add_argument('--task',choices=tuple(base.main_eval.COUNTS));a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git code/CWD required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    install()
    if a.mode=='run':controller(a)
    else:
        if a.arm!=ARM:p.error('--arm required')
        row=a.root/'runs'/RUN/'rows'/ARM
        if a.mode=='predict':
            if not a.task:p.error('--task required')
            predict(a,row/'prediction'/a.task)
        else:base.score(a,row/'score')
