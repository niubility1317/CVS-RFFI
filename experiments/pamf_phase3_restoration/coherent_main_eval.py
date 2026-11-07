"""Full frozen Main evaluation of the source-verified coherent main phase readout."""

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
from .cache import check_F0_state, load_F0, tensor
from .coherent_main_phase import full_phase_direction, phase_basis, restore_main_direction
from .cost_probe import write
from .covariance_inverse import dense_action
from .covariance_parameter_source import G_COMMIT, load_model
from .covariance_trial import RUN as BF_RUN
from .phase_only_eval import STREAMS, frozen_controls
from .phase_ramp_posterior import fixed_grid
from .source_coherent_main_phase import readouts, received_only
from .source_pair import read, validate_source

RUN = '20261107-evaluation-Restore-CoherentMainPhase-CPU-main-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.coherent_main_eval'
ARM = 'TF_CovarianceCoherentMainFull'
BATCH = 32


class FrozenCoherentMain(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        phase,_,prior = fixed_grid('cpu')
        self.register_buffer('phase',phase)
        self.register_buffer('log_prior',prior[4])

    def forward(self,received):
        est = self.model.estimate(received,dense=False)
        h,main,image = (est[k].to(torch.complex128) for k in ('kernel','main','image'))
        action = dense_action(h,main,image)
        main_action = dense_action(h,main,torch.zeros_like(image))
        observed,variance = received.flatten(1).double(),est['variance'].double()
        mean,covariance = self.model.prior_mean.double(),self.model.prior_covariance.double()
        normal,basis = phase_basis(action,main_action,observed,mean,covariance,variance)
        direction = full_phase_direction(normal,basis,self.phase,self.log_prior)
        restored,_ = restore_main_direction(main_action,action-main_action,observed,mean,covariance,variance,direction)
        return restored.float().reshape_as(received)


def collect_task_controls(root,task,ids,identity):
    if len(ids) != base.main_eval.COUNTS[task]:
        raise ValueError('Full original Main population required')
    return frozen_controls(root,task,ids,identity)


def logits_for(G,F,received):
    restored = G(received)
    logits = F.classify_features(F.features(restored))
    if logits.shape!=(len(received),6) or not torch.isfinite(restored).all() or not torch.isfinite(logits).all():
        raise ValueError('Invalid frozen coherent-main IQ or original F0 logits')
    return logits


def source_smoke_and_time(out,cache,roles,G,F):
    take = np.flatnonzero(roles=='L_s')[:128]
    if len(take) != 128:
        raise ValueError('Original source-only L_s readout required')
    source = received_only(cache/'epochs/01',take,63000)
    phase,drift,prior = fixed_grid('cpu')
    with torch.inference_mode():
        expected,_ = readouts(G.model,source[:2],(phase,drift[4:5],prior[4:5]))
        torch.testing.assert_close(G(source[:2]),expected['main_mean_full']['iq'],rtol=0,atol=0)
        logits_for(G,F,source[:2])
        write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_original_BF=True,actual_original_F0=True,
            equals_source_full_covariance=True,TX_labels_read=0,target_inputs=0,new_G_training_updates=0))
        print('COHERENT_MAIN_CPU_REAL_SOURCE_SMOKE_PASS',flush=True)
        start = time.perf_counter()
        for batch in source.split(BATCH):
            logits_for(G,F,batch)
        seconds = time.perf_counter()-start
    projected = seconds/len(source)*sum(base.main_eval.COUNTS.values())*len(base.main_eval.VIEWS)
    write(out/'source_throughput.json',dict(records=len(source),batch=BATCH,CPU_threads=2,seconds=seconds,
        records_per_second=len(source)/seconds,projected_Main_compute_seconds=projected,
        target_inputs=0,TX_labels_read=0,excludes_disk_IO_and_score=True))
    print('COHERENT_MAIN_CPU_SOURCE_TIMING',len(source),seconds,flush=True)


def predict_task(args,out,official,task,G,F,identity):
    from experiments.cric_dura_comparison.evaluation import handle

    manifest,ids,positions = handle(official['tests'][task]['views_root'],official['classes'])
    values = collect_task_controls(args.root,task,ids,identity)
    out.mkdir()
    for view in base.main_eval.VIEWS:
        path = manifest['clean_ref'] if view=='clean' else str(Path(manifest['observations_root'])/(view+'.npy'))
        source = np.load(path,mmap_mode='r',allow_pickle=False)
        decisions = np.empty(len(ids),dtype=np.int64)
        try:
            if source.ndim!=3 or source.shape[1:]!=(2,256) or source.dtype!=np.float32:
                raise ValueError('Original Main FP32 IQ required')
            with torch.inference_mode():
                for start in range(0,len(ids),BATCH):
                    stop = min(start+BATCH,len(ids))
                    take = slice(start,stop) if positions is None else positions[start:stop]
                    logits = logits_for(G,F,tensor(source[take],'cpu'))
                    decisions[start:stop] = logits.argmax(1).tolist()
                    if start==0 or stop%8192==0:
                        print('COHERENT_MAIN_CPU_PROGRESS',task,view,stop,len(ids),flush=True)
        finally:
            source._mmap.close()
        with (out/(view+'.npy')).open('xb') as file:
            np.save(file,decisions)
        values['candidate__'+view] = decisions
        print('COHERENT_MAIN_CPU_VIEW_COMPLETE',task,view,len(ids),flush=True)
    with (out/'predictions.npz').open('xb') as file:
        np.savez(file,ids=ids,**values)
    return out,dict(status='PREDICTIONS_COMPLETE',run_id=RUN,arm=ARM,task=task,
        count=len(ids),code_commit=args.commit,F0_identity=identity,readouts=list(STREAMS),views=list(base.main_eval.VIEWS),
        baseline_reused_from=BF_RUN,truth_read=False,query_fit=False,model_state_updates=0,
        new_G_training_updates=0,CPU_threads=2,phase_grid_points=256,drift_points=1)


def predict(args,out):
    os.nice(10)
    c = validate_source(read(args.inputs))
    cache,roles = base.source(args)
    model = load_model(args.root,'cpu')
    install()
    G = FrozenCoherentMain(model).eval().requires_grad_(False)
    F,identity = load_F0(c,'cpu')
    base.main_eval.valid_identity(identity)
    models = G,F
    snapshots = [{key:value.detach().clone() for key,value in m.state_dict().items()} for m in models]
    source_smoke_and_time(out,cache,roles,G,F)
    official = base.main_eval.official(args.root)
    finished = []
    for task in base.main_eval.COUNTS:
        finished.append(predict_task(args,out/task,official,task,G,F,identity))
    for m,snapshot in zip(models,snapshots):
        check_F0_state(m,snapshot)
    for folder,record in finished:
        write(folder/'complete.json',record)


def controller(args):
    if read(args.root/'runs'/BF_RUN/'complete.json')['status']!='ARTIFACTS_COMPLETE':
        raise ValueError('Original frozen BF Main controls must be complete')
    out = args.root/'runs'/RUN
    out.mkdir()
    (out/'logs').mkdir()
    row = out/'rows'/ARM
    (row/'prediction').mkdir(parents=True)
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit,
        CPU_threads=2,GPU_workers=0,new_G_training_updates=0))
    for stage in ('predict','score'):
        command = [sys.executable,'-u','-m',MODULE,stage,'--root',str(args.root),'--commit',args.commit,
            '--inputs',str(args.inputs),'--cache',str(args.cache)]
        with (out/'logs'/(stage+'.log')).open('xb') as log:
            p = subprocess.Popen(command,cwd=Path.cwd(),env=os.environ.copy(),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+stage+'.json'),dict(pid=p.pid,command=command,CWD=os.getcwd(),CPU_threads=2))
        code = p.wait()
        write(out/('terminal-'+stage+'.json'),dict(pid=p.pid,exit_code=code))
        if code:
            raise RuntimeError('CPU coherent-main stage failed; preserve artifacts')
    if read(row/'results.json')['status']!='ANALYZED':
        raise ValueError('All Main views and both populations must be scored')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_Main_two_populations_seven_scored=True,
        frozen_BF_commit=G_COMMIT,new_G_training_updates=0,default_DURA_changed=False,goal_complete=False))


def install():
    base.RUN,base.MODULE,base.ARMS,base.STREAMS = RUN,MODULE,(ARM,),STREAMS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=['run','predict','score'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only coherent Main evaluation with exact release identity required')
    torch.set_num_threads(2)
    install()
    row = args.root/'runs'/RUN/'rows'/ARM
    if args.mode=='run':
        controller(args)
    elif args.mode=='predict':
        predict(args,row/'prediction')
    else:
        args.arm = ARM
        base.score(args,row/'score')


if __name__=='__main__':
    main()
