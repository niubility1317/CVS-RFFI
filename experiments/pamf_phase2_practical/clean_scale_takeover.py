"""Drain one clean queue and dispatch only its unstarted work on five GPUs."""
import argparse
import json
import os
from pathlib import Path
import time

from .clean_scheduler import completion_matches, dispatch
from .io import read, require, write


def partition(root, entries):
    completed=[]; pending=[]
    for entry in entries:
        output=Path(entry['completion']).parent
        log=root/'scheduler'/'logs'/(entry['id']+'.log')
        if Path(entry['completion']).is_file():
            require(completion_matches(entry), 'Invalid parent completion')
            completed.append(entry)
        else:
            require(not output.exists() and not log.exists(), 'Parent partial or launch log requires reconciliation')
            pending.append(entry)
    return completed,pending


def process_live(pid):
    proc=Path('/proc')/str(pid)
    try:
        return (proc/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        return False


def run(parent_file, plan_root, run_root, release_root):
    parent_file=Path(parent_file).resolve(); parent=read(parent_file)
    old=Path(parent['run_root']).resolve();control=old/'scheduler'
    plan=Path(plan_root).resolve();run_root=Path(run_root).resolve();release=Path(release_root).resolve()
    require(not plan.exists() and not run_root.exists() and release.is_dir(), 'Unsafe successor paths')
    owner=read(control/'owner.json');pid=int(owner['pid'])
    require(process_live(pid), 'Parent controller is not live')
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    require('clean_scheduler' in cmd and str(parent_file) in cmd, 'Parent owner does not match')
    state=read(control/'state.json')
    require(state['status']=='RUNNING' and not state['failed'], 'Parent not healthy')
    reservation=control/'scale7_successor';reservation.mkdir()
    write(reservation/'owner.json',dict(pid=os.getpid(),plan_root=str(plan),run_root=str(run_root)))
    with (control/'USER_STOP.json').open('x',encoding='utf-8') as f:
        json.dump(dict(reason='user_requested_five_gpu_seven_workers',successor=str(run_root)),f)
    while process_live(pid):
        state=read(control/'state.json')
        if state['failed']:raise RuntimeError('Parent worker failed during drain')
        time.sleep(5)
    state=read(control/'state.json')
    require(state['status']=='STOPPED' and not state['active'] and not state['failed'] and state['pending']==0,
            'Parent did not drain cleanly')
    completed,pending=partition(old,parent['jobs'])
    require(len(completed)==state['completed'] and len(completed)+len(pending)==len(parent['jobs']) and pending,
            'Parent completion count mismatch')
    plan.mkdir(parents=True);(plan/'jobs').mkdir();(plan/'runtime').mkdir()
    jobs=[];native_cache={}
    for entry in pending:
        job=read(entry['job']);runtime=read(entry['runtime']);name=entry['id'];out=run_root/'tasks'/name
        if entry['group']=='pamf':
            rtf=plan/'runtime'/(name+'.json')
            runtime.update(run_root=str(out),release_root=str(release),runtime_file=str(rtf))
            write(rtf,runtime)
            completion=out/'predictions'/job['capsule'].split('/')[-1]/'complete.json'
        else:
            key=entry['runtime']
            if key not in native_cache:
                rtf=plan/'runtime'/('shared-'+str(len(native_cache))+'.json')
                runtime.update(release_root=str(release),runtime_file=str(rtf));write(rtf,runtime)
                native_cache[key]=str(rtf)
            rtf=Path(native_cache[key]);job=dict(job,output=str(out));completion=out/'complete.json'
        jf=plan/'jobs'/(name+'.json');write(jf,job)
        jobs.append(dict(entry,job=str(jf),runtime=str(rtf),completion=str(completion)))
    config=dict(parent,run_root=str(run_root),release_root=str(release),jobs=jobs,
                allowed_gpus=[0,1,2,3,4],max_per_gpu=7,minimum_free_MiB=6000,
                worker_reserve_MiB=3000,one_per_snapshot=True)
    config.pop('auto_score',None)
    write(plan/'scheduler.json',config)
    write(plan/'reuse.json',dict(parent=str(old),completed=len(completed),pending=len(pending),
          completed_ids=[entry['id'] for entry in completed],truth_read=False))
    dispatch(config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-config','plan-root','run-root','release-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.parent_config,a.plan_root,a.run_root,a.release_root)


if __name__=='__main__':main()
