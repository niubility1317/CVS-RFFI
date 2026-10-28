"""One immutable source-only branch export and rule-evaluation owner."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .launch import choose_gpu,read,write

FIELDS={'run_id','run_root','release_root','python','code_commit','launch_owner','allowed_gpus',
        'max_added_workers_per_gpu','minimum_free_mb','export'}


def validate_runtime(rt):
    if not isinstance(rt,dict) or set(rt)!=FIELDS:raise ValueError('Branch runtime whitelist differs')
    if (rt['max_added_workers_per_gpu']!=2 or not rt['allowed_gpus']
            or any(type(g) is not int or g<0 for g in rt['allowed_gpus'])
            or not isinstance(rt['code_commit'],str) or len(rt['code_commit'])!=40
            or not isinstance(rt['export'],dict)):
        raise ValueError('Explicit source code identity and bounded GPU capacity required')
    path=Path(rt['run_root'])
    if not path.is_absolute() or path.name!=rt['run_id'] or not rt['launch_owner']:
        raise ValueError('New run-owned branch output root required')


def dispatch(rt):
    validate_runtime(rt)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();(root/'configs').mkdir()
    config=dict(rt['export'],run_id=rt['run_id'],row_id='source-branches-fold0-s2026092701',
                code_commit=rt['code_commit'],output_root=str(root/'export'))
    config_path=root/'configs/export.json';write(config_path,config)
    commands=dict(export=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.branch_export',
        '--config',str(config_path),'--device','cuda:0'],
        rules=[rt['python'],'-B','-u','-m','experiments.pamf_phase3_open_world.branch_rules',
        '--features',str(root/'export/source_branches.npz'),'--output',str(root/'rules'),
        '--heldout-receiver',str(config['fold']['heldout_source_rx'])])
    state=dict(status='PREPARING_SOURCE_ONLY',source_only=True,target_reads=0,optimizer_updates=0,
        pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=rt['launch_owner'],code_commit=rt['code_commit'],
        started_at=time.time(),jobs={},foreign_processes_stopped=0,max_added_workers_per_gpu=2)
    def record():write(root/'pipeline_state.json',state,exclusive=False)
    record()
    try:
        for kind in ('export','rules'):
            gpu=choose_gpu(rt) if kind=='export' else None
            env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=root/'logs'/(kind+'.log')
            with log.open('x',encoding='utf-8') as stream:
                child=subprocess.Popen(commands[kind],cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
                state['status']='RUNNING_SOURCE_ONLY'
                state['jobs'][kind]=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=rt['release_root'],argv=commands[kind],
                    log=str(log),started_at=time.time())
                record();status=child.wait()
            if status:raise RuntimeError(kind+' failed with exit='+str(status)+'; preserve '+str(log))
            marker=read(root/kind/'complete.json')
            if marker.get('source_only') is not True or marker.get('target_reads')!=0:
                raise ValueError('Source-only branch boundary differs')
            if kind=='export':
                if (marker.get('status')!='SOURCE_BRANCHES_EXPORTED' or marker.get('optimizer_updates')!=0
                        or marker.get('head_changed') is not False or marker.get('train_count')!=3360
                        or marker.get('validation_count')!=27000 or marker.get('parent_epochs')!=20
                        or marker.get('parent_updates')!=540):raise ValueError('Frozen own-source export incomplete')
            elif marker.get('status')!='SOURCE_BRANCH_RULES_EVALUATED':raise ValueError('Branch rule evaluation incomplete')
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
