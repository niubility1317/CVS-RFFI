"""Bounded CPU-only data workers; no identity training/prediction/scoring."""
import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,atomic,require


def run(runtime):
    rt=read(runtime);release=Path.cwd();root=Path(rt['run_root']);started=time.time()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    status=release/'preparation_status.json'
    base=[sys.executable,'-u','-m','experiments.pamf_phase2_practical.episodic_data']
    with (release/'pool.log').open('x') as f:
        command=base+['pool','--runtime',runtime]
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        atomic(status,dict(status='RUNNING',stage='pool',controller_pid=os.getpid(),pid=child.pid,argv=command,code_commit=rt['code_commit']))
        result=child.wait()
    if result:
        atomic(status,dict(status='FAILED',stage='pool',exit_code=result,run_root=rt['run_root']))
        raise SystemExit(result)
    jobs=deque((e['id'],r) for e in rt['episodes'] for r in rt['target_receivers']);total=len(jobs)
    active={};done=[];logs=release/'worker_logs';logs.mkdir()
    try:
        while jobs or active:
            while jobs and len(active)<rt['max_cpu_workers']:
                e,r=jobs.popleft();path=logs/(e+'-'+r+'.log');f=path.open('x')
                command=base+['worker','--runtime',runtime,'--receiver',r,'--episode',e]
                p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                active[p.pid]=dict(process=p,file=f,episode=e,receiver=r,argv=command,log=str(path))
            for pid,item in list(active.items()):
                rc=item['process'].poll()
                if rc is None:continue
                item['file'].close()
                require(rc==0,f'Worker failed: {item["episode"]}/{item["receiver"]}, exit={rc}')
                report=read(root/'worker_reports'/(item['episode']+'-'+item['receiver']+'.json'))
                require(report['status']=='VALIDATED_ONCE','Incomplete bank output')
                done.append(report);del active[pid]
            atomic(status,dict(status='RUNNING',stage='episodes',controller_pid=os.getpid(),run_root=rt['run_root'],
                                code_commit=rt['code_commit'],completed_banks=len(done),total_banks=total,
                                completed_observations=sum(x['observations'] for x in done),
                                workers=[{k:v for k,v in x.items() if k not in ('process','file')}|{'pid':pid} for pid,x in active.items()],
                                elapsed_seconds=time.time()-started))
            time.sleep(2)
    except BaseException as exc:
        # Only child Popen objects created by this controller are terminated.
        for item in active.values():
            if item['process'].poll() is None:item['process'].terminate()
        for item in active.values():
            try:item['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait()
            item['file'].close()
        atomic(status,dict(status='FAILED',stage='episodes',error=repr(exc),run_root=rt['run_root'],completed_banks=len(done)))
        raise
    pool=read(root/'builder_only/pool/manifest.json')
    observations=sum(x['observations'] for x in done);splits=sum(x['splits'] for x in done)
    require(observations==pool['physical_records']*len(rt['episodes']) and len(done)==total,'Wrong episode totals')
    catalog=[dict(path=x['capsule'],episode=x['episode'],receiver=x['receiver'],observations=x['observations'],splits=x['splits']) for x in done]
    write(root/'catalog.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsules=catalog))
    write(root/'score_only/label_map.json',dict(classes=rt['old_classes']+rt['new_classes'],dates=rt['dates'],
                                              scenes=[s['id'] for s in read(rt['mixture_config'])['scenarios']]))
    report=dict(status='VALIDATED_ONCE',code_commit=rt['code_commit'],unique_physical_pool_records=pool['physical_records'],
                reoverlaid_observations=observations,episodes=len(rt['episodes']),capsules=len(done),splits=splits,
                query_rule='entire_current_registered_pool_minus_K_support_per_class',raw_roles_are_task_relative=True,
                fresh_channels_per_episode=True,same_channel_across_K_within_episode=True,
                source_roles_unchanged=True,model_reads=0,identity_training=0,identity_predictions=0,scoring=0,
                elapsed_seconds=time.time()-started)
    write(root/'builder_report.json',report)
    atomic(status,dict(status='DATA_COMPLETE',**{k:v for k,v in report.items() if k!='status'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True)
    run(p.parse_args().runtime)
