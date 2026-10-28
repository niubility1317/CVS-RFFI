"""Immutable source-only development dispatcher; no target stage exists."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .launch import choose_gpu,read,write

FIELDS={'run_id','run_root','release_root','python','code_commit','launch_owner','allowed_gpus',
        'max_added_workers_per_gpu','minimum_free_mb','train'}


def validate_runtime(rt):
    if not isinstance(rt,dict) or set(rt)!=FIELDS:
        raise ValueError('Source dispatcher runtime whitelist differs')
    if (rt['max_added_workers_per_gpu']!=2 or not rt['allowed_gpus']
            or any(type(g) is not int or g<0 for g in rt['allowed_gpus'])
            or len(rt['code_commit'])!=40 or not isinstance(rt['train'],dict)):
        raise ValueError('Explicit source code identity and GPU capacity required')
    root=Path(rt['run_root'])
    if not root.is_absolute() or root.name!=rt['run_id'] or not rt['launch_owner']:
        raise ValueError('New run-owned source output root required')


def dispatch(rt):
    validate_runtime(rt)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();(root/'configs').mkdir()
    train=dict(rt['train'],run_id=rt['run_id'],row_id='source-holdout-fold0-s2026092701',
               code_commit=rt['code_commit'],output_root=str(root/'train'))
    config=root/'configs/train.json';write(config,train)
    state=dict(status='PREPARING_SOURCE_ONLY',source_only=True,target_reads=0,pid=os.getpid(),cwd=os.getcwd(),
        argv=sys.argv,owner=rt['launch_owner'],code_commit=rt['code_commit'],started_at=time.time(),jobs={},
        foreign_processes_stopped=0,max_added_workers_per_gpu=2)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record()
    commands=dict(train=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_train',
                         '--config',str(config),'--device','cuda:0'],
                  rules=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.source_geometry',
                         '--features',str(root/'train/source_features.npz'),'--output',str(root/'rules'),
                         '--heldout-receiver',str(train['fold']['heldout_source_rx'])])
    try:
        for kind in ('train','rules'):
            gpu=choose_gpu(rt) if kind=='train' else None
            environment=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=root/'logs'/(kind+'.log')
            with log.open('x',encoding='utf-8') as stream:
                child=subprocess.Popen(commands[kind],cwd=rt['release_root'],env=environment,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
                state['status']='RUNNING_SOURCE_ONLY'
                state['jobs'][kind]=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=rt['release_root'],argv=commands[kind],
                    log=str(log),started_at=time.time())
                record();status=child.wait()
            if status:raise RuntimeError(kind+' failed with exit='+str(status)+'; preserve '+str(log))
            marker=read(root/kind/'complete.json')
            if marker.get('source_only') is not True or marker.get('target_reads')!=0:
                raise ValueError('Source-only completion boundary differs')
            if kind=='train':
                if (marker.get('status')!='SOURCE_DEVELOPMENT_FROZEN' or marker.get('updates')!=540
                        or marker.get('train_count')!=3360 or marker.get('validation_count')!=27000):
                    raise ValueError('Fixed source training budget/coverage did not complete')
            elif marker.get('status')!='SOURCE_RULES_EVALUATED':
                raise ValueError('Source rule evaluation did not complete')
            state['jobs'][kind].update(status='COMPLETE',ended_at=time.time());record()
        state.update(status='ANALYZED_SOURCE_ONLY',ended_at=time.time(),next_step=marker['next_step'],
                     eligible_methods=marker['eligible_methods'],target_performance_claim=False)
        record();print('ANALYZED_SOURCE_ONLY',rt['run_id'],state['next_step'],flush=True)
    except Exception as exc:
        state.update(status='STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE',failure=str(exc),ended_at=time.time())
        record();traceback.print_exc();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',required=True)
    args=parser.parse_args();dispatch(read(args.runtime))


if __name__=='__main__':main()
