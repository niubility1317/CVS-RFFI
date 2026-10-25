"""Bounded three-GPU source diagnostic owner. Does not launch any target work."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.cache import atomic
from .plan import SOURCE_RUN,rows,validate_context,GPU_BY_SEED


def worker(config):
    from experiments.pamf_repl_completion.runner import activate
    activate(config)
    from .source_quality import run
    return run(config['case'],config['output_root'])


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes
    if rt['run_id']!=SOURCE_RUN or rt['allowed_gpus']!=[5,6,7]:raise ValueError('Fixed source run/resources required')
    context=read(rt['context']);cases=validate_context(context)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    jobs={}
    for row in rows():
        key=row['row_id'];c=dict(frozen_sources=rt['frozen_sources'],
            case=cases[row['protocol'],row['model_seed']],output_root=str(root/'rows'/key))
        path=root/'configs'/(key+'.json');write(path,c)
        jobs[key]=dict(row,config=str(path),status='QUEUED')
    state=dict(status='SOURCE_DIAGNOSTIC_RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],
        launch_owner=rt['launch_owner'],started_at=time.time(),jobs=jobs)
    active={};stop=False
    while True:
        for key,child in list(active.items()):
            status=child.poll()
            if status is None:continue
            item=jobs[key];path=root/'rows'/key/'complete.json'
            if status==0 and path.exists() and read(path)['status']=='SOURCE_DIAGNOSTIC_COMPLETE':
                result=read(path);item.update(status='COMPLETE',scientific_pass=result['scientific_rule']['pass'],
                    result=str(path),ended_at=time.time())
            else:
                item.update(status='FAILED',exit_code=status,ended_at=time.time())
                # A technical failure stops new admissions; healthy children
                # finish and retain outputs. There is no automatic retry.
                stop=True
            del active[key]
        pending=processes()
        for key,item in jobs.items():
            if stop or item['status']!='QUEUED':continue
            gpu=GPU_BY_SEED[item['model_seed']]
            if sum(jobs[k]['gpu']==gpu for k in active)>=2:continue
            pids,free=capacity(gpu)
            if len(pids|{p for p,g in pending.items() if g==gpu})>=3 or free<10000:continue
            log=root/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
            cmd=[rt['python'],'-u','-m','experiments.pamf_channel_pl.source_runner','worker','--config',item['config']]
            env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=child;pending[child.pid]=gpu
            item.update(status='RUNNING',pid=child.pid,log=str(log),cwd=rt['release_root'],argv=cmd,started_at=time.time())
        state['updated_at']=time.time();atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in jobs.values())):break
        time.sleep(15)
    complete=all(v['status']=='COMPLETE' for v in jobs.values())
    state.update(status='SOURCE_DIAGNOSTIC_COMPLETE' if complete else 'STOPPED_FOR_TECHNICAL_FAILURE',ended_at=time.time())
    atomic(root/'pipeline_state.json',state)
    if complete:
        results=[read(v['result']) for v in jobs.values()]
        write(root/'complete.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=results,
            passed=sum(v['scientific_rule']['pass'] for v in results),total=len(results),
            target_reads=0,model_updates=0,U_truth_read=False,code_commit=rt['code_commit']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker'));p.add_argument('--config',required=True)
    args=p.parse_args();value=read(args.config)
    (dispatch if args.action=='dispatch' else worker)(value)
