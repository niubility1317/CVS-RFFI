"""Render the original frozen R17 final IQ on the twenty legal source views."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_dura_direct_train.coarse_hint import load_teacher
from .r17_teacher_paths import TEACHER_RUN,TEACHER_COMMIT

RUN='20261031-source-DURA-R17-nativeIQ-cache-s2026092701-r01'
MODULE='experiments.pamf_dura_efficiency.r17_native_cache'
VIEWS=tuple(range(1,21))
COUNT=63000
BATCH=128


def native_iq(teacher,received):
    stages,_,_,_=teacher.stage_outputs(received)
    if len(stages)!=3:raise ValueError('Original R17 final third stage required')
    result=stages[-1]
    if result.shape!=received.shape or not torch.isfinite(result).all():
        raise FloatingPointError('Finite original final source IQ required')
    return result


def configuration():
    return dict(run=RUN,kind='source_IQ_cache_preparation',source_contract='original7/63/30',
        source_action_run=base.ACTION,source_count=COUNT,L_count=6300,U_count=56700,
        source_corruption_views=list(VIEWS),batch=BATCH,teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,
        teacher_component='native_map_prior.stage3',teacher_parameters=4665073,
        teacher_coarse_epochs=20,teacher_action_epochs=20,teacher_prior_epochs=20,teacher_bundle_component_epochs=60,
        teacher_source_selection_before_target=True,historical_source_V_label_selection=True,
        cache_inputs=COUNT*len(VIEWS),cache_L_visits=126000,cache_U_visits=1134000,
        cache_numeric_bytes=COUNT*len(VIEWS)*2*256*4,model_optimizer_updates=0,
        new_training_epochs=0,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        allowed_GPUs=[4],max_total_visible_GPU_processes=3,max_goal_GPU_workers=2,
        max_own_workers=1,minimum_free_MB=12000,FP32=True,TF32=False,CPU_threads=2,
        stop_rule='Badteacher/source/roles/count/nonfinite/collision/statechange:preserve/noautoretry/nootherownercontrol',
        scope='OriginalsourcefinalIQcacheonly;notnewGtraining/notformalprediction/notgoalcompletion')


@torch.no_grad()
def smoke(args,p):
    clean,roles=base.source_training(p);take=np.flatnonzero(roles=='L_s')[:2]
    received=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    teacher=load_teacher(args.root,'cpu')
    before={k:v.clone() for k,v in teacher.state_dict().items()}
    output=native_iq(teacher,base.tensor(received[take],'cpu'))
    if any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()) or any(v.grad is not None for v in teacher.parameters()):
        raise ValueError('Frozen source teacher changed during smoke')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,
        teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,teacher_parameters=sum(v.numel() for v in teacher.parameters()),
        teacher_component='native_map_prior.stage3',output_shape=list(output.shape),teacher_unchanged=True,
        G_smoke_inputs=2,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))
    print('{"event":"REAL_SOURCE_R17_NATIVE_CACHE_NOQUERY_SMOKE_PASS","Ls":2,"F_reads":0}',flush=True)
    del teacher,before,output;gc.collect()


@torch.no_grad()
def worker(args,p):
    row=p['output']/args.arm
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='4':raise ValueError('Registered GPU4 worker required')
        row.mkdir();folder=row/'native_teacher_cache';folder.mkdir()
        base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
            physical_GPU=os.environ['CUDA_VISIBLE_DEVICES']))
        clean,roles=base.source_training(p)
        ids=np.load(p['cache']/'train/physical_ids.npy',allow_pickle=False)
        if ids.shape!=(COUNT,):raise ValueError('Original full source physical population required')
        np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
        np.save(folder/'roles.npy',roles,allow_pickle=False)
        teacher=load_teacher(args.root,'cpu');before={k:v.clone() for k,v in teacher.state_dict().items()}
        teacher.to('cuda:0')
        if sum(v.numel() for v in teacher.parameters())!=4665073:raise ValueError('Original full R17 required')
        started=time.perf_counter();inputs=0;timings={};torch.cuda.reset_peak_memory_stats()
        for view in VIEWS:
            received=np.load(p['cache']/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=(COUNT,2,256) or received.dtype!=np.float32:raise ValueError('Original full source FP32 view required')
            target=np.lib.format.open_memmap(folder/f'epoch{view:02d}.npy',mode='w+',dtype=np.float32,shape=received.shape)
            tick=time.perf_counter()
            for offset in range(0,COUNT,BATCH):
                end=min(offset+BATCH,COUNT)
                output=native_iq(teacher,base.tensor(received[offset:end],'cuda:0'))
                target[offset:end]=base.numpy_values(output);inputs+=end-offset
            target.flush();timings[str(view)]=time.perf_counter()-tick
            base.write_json(folder/f'epoch{view:02d}.json',dict(status='SOURCE_NATIVE_IQ_VIEW_COMPLETE',source_view=view,
                count=COUNT,teacher_inputs_cumulative=inputs,seconds=timings[str(view)],F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0))
            print(json.dumps(dict(event='SOURCE_NATIVE_IQ_CACHE_VIEW_COMPLETE',source_view=view,count=COUNT,
                teacher_inputs=inputs,seconds=time.perf_counter()-started)),flush=True)
        if inputs!=1260000:raise ValueError('All twenty source views required')
        if any(not torch.equal(v.detach().cpu(),before[k]) for k,v in teacher.state_dict().items()) or any(v.grad is not None for v in teacher.parameters()):
            raise ValueError('Frozen source teacher changed during cache production')
        record=dict(status='SOURCE_NATIVE_IQ_CACHE_COMPLETE',source_contract='original7/63/30',code_commit=args.commit,
            source_action_run=base.ACTION,teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,
            teacher_component='native_map_prior.stage3',teacher_used_parameters=4665073,
            teacher_coarse_epochs=20,teacher_action_epochs=20,teacher_prior_epochs=20,teacher_bundle_component_epochs=60,
            teacher_source_selection_before_target=True,teacher_unchanged=True,
            views=list(VIEWS),count_per_view=COUNT,teacher_inputs=inputs,teacher_smoke_inputs=2,
            L_visits=126000,U_visits=1134000,numeric_bytes=inputs*2*256*4,teacher_updates=0,
            F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,model_optimizer_updates=0,
            seconds=time.perf_counter()-started,per_view_seconds=timings,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            timing_scope='Teacher GPU residency/cache-loop/transfer/output flush;teacher loading and CPU smoke separate',
            scope='Reusable source-only original R17 finalIQ;not formal target predictions')
        base.write_json(folder/'complete.json',record)
        base.write_json(row/'complete.json',dict(status='ANALYZED',scope='SOURCE_CACHE_ONLY',cache_complete=True,
            model_optimizer_updates=0,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,goal_complete=False))
    except BaseException as error:
        if row.exists():base.write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
            traceback=traceback.format_exc(),outputs_preserved=True))
        raise


def command(args,mode,arm=None,role=None):
    if mode not in ('run','worker') or role is not None:raise ValueError('Source cache has no scoring or target mode')
    result=[sys.executable,'-u','-m',MODULE,mode,'--root',str(args.root),'--commit',args.commit]
    if arm is not None:result+=['--arm',arm]
    return result


def configure():
    base.RUN=RUN;base.configuration=configuration;base.smoke=smoke;base.command=command;base.worker=worker


def parse_arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--commit',required=True);parser.add_argument('--arm',choices=base.ARMS)
    args=parser.parse_args(argv)
    if args.mode=='worker' and args.arm is None:parser.error('Fixed source-cache arm required')
    args.root=args.root.resolve()
    return args


def main():
    args=parse_arguments();configure()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=base.paths(args.root)
    if args.mode=='run':base.dispatch(args,p)
    else:worker(args,p)


if __name__=='__main__':main()
