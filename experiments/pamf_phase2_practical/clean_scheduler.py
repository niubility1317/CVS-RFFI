"""Single-owner clean prediction queue. Checking never starts an experiment."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .clean_external_worker import validate as validate_external
from .clean_pairing import EPISODES,POOLS,RECEIVERS
from .io import read, require, write


def validate_plan(config):
    root=Path(config['run_root']).resolve();jobs=config['jobs'];ids=set();outputs=set();capsules=set();seeds=set()
    require(jobs and config.get('max_per_gpu')==2 and config.get('minimum_free_MiB',0)>=6000,'Wrong resource limits')
    require(config.get('input_condition')=='clean_ground_pre_satellite_normalized','Wrong input condition')
    require(config.get('phase')=='pilot','Full matrix requires preregistered scientific thresholds and is not yet enabled')
    require(Path(config['release_root']).is_dir(),'Missing immutable release root')
    gpus=config.get('allowed_gpus',[])
    require(gpus and len(gpus)==len(set(gpus)) and all(type(g) is int and 0<=g<=7 for g in gpus),'Wrong GPU list')
    allowed_capsules={f'capsules/{p}/{e}/{r}' for p in POOLS for e in EPISODES for r in RECEIVERS}
    for entry in jobs:
        key=entry['id'];require(re.fullmatch(r'[A-Za-z0-9_-]+',key) is not None and key not in ids,'Invalid or duplicate job id');ids.add(key)
        group=entry['group'];rt=read(entry['runtime']);job=read(entry['job'])
        require(job['capsule'] in allowed_capsules and job['seed'] in (2026092701,2026092702,2026092703,2026092704),
                'Job outside fixed capsule or seed matrix')
        capsules.add(job['capsule']);seeds.add(job['seed'])
        require(job.get('points')==[[5,20]],'Pilot must use the fixed K5/N20 point')
        require(group in ('pamf','native_da','native_cil','hybrid'),'Unknown group')
        require(rt.get('clean_root') and rt.get('data_root'),'Missing clean input binding')
        require(Path(rt['release_root']).resolve()==Path(config['release_root']).resolve(),'Worker release mismatch')
        data=Path(rt['data_root']).resolve();clean=Path(rt['clean_root']).resolve()
        complete=read(clean/'completion.json')
        require(data.is_dir() and complete.get('status')=='CLEAN_PAIRED_COMPLETE' and
                Path(complete['parent_task_root']).resolve()==data,'Clean data not bound to parent')
        require(Path(rt['source_contract']).is_file() and Path(rt['source_cache']).is_dir() and
                Path(rt['phase1_release']).is_dir(),'Missing source inputs')
        if group=='pamf':
            require(rt.get('arm')=='R42' and rt.get('kernel_method')=='adaptive_residual_ensemble'
                    and rt.get('model_seeds')==[job['seed']] and not job.get('reuse_file'), 'Wrong SCALR runtime')
            if rt.get('restored_root'):
                restored=read(Path(rt['restored_root'])/'completion.json')
                require(restored.get('status')=='CLEAN_DURA_COMPLETE' and restored.get('restorer_id')=='dura_v1'
                        and Path(restored['clean_root']).resolve()==clean,'Restored cache is missing or mismatched')
            source_root=Path(rt.get('scalr_source_root',rt['run_root']))
            require((Path(rt['PAMF_sources'][str(job['seed'])])/'final_ssdg.pth').is_file() and
                    Path(rt['DURA']['checkpoint']).is_file() and
                    (source_root/'source_selection.json').is_file() and
                    (source_root/'source_orbit_cca.npz').is_file(),'Missing SCALR source state')
            out=Path(rt['run_root'])/'predictions'/job['capsule'].split('/')[-1]
        else:
            validate_external(rt,job,group);profile=job['profile'];seed=str(job['seed'])
            if profile in ('tweak_m1','issl_fixed_v1'):
                source=Path(rt['additional_sources'][profile][seed])/'source.pt'
            else:
                source_root=rt['source_run_root'] if group=='hybrid' else rt['run_root']
                source=Path(source_root)/'source'/f'{profile}-s{seed}'/'source.pt'
            require(source.is_file(),'Missing selected source checkpoint')
            out=Path(job['output'])
        out=out.resolve()
        require(out.is_relative_to(root) and out!=root and str(out) not in outputs,'Unsafe or duplicate output')
        require(Path(entry['completion']).resolve()==out/'complete.json','Wrong completion path')
        outputs.add(str(out))
        require(not out.exists(),'Output already exists; reconcile before any launch')
        require(entry.get('priority',0)>=0,'Invalid priority')
    require(len(seeds)==1 and len(capsules)<=5,'Pilot must use one seed and at most five target tasks')
    return dict(jobs=len(jobs),groups=sorted({x['group'] for x in jobs}),prediction_only=True,truth_opened=False)


def choose(jobs,active,devices,allowed,minimum_free):
    """Return safe (job,gpu) admissions for one resource snapshot."""
    pending=sorted(jobs,key=lambda j:(j.get('priority',0),-j.get('measured_cost_seconds',0),j['id']))
    own_pids={v['pid'] for v in active.values()};chosen=[]
    for gpu in allowed:
        if not pending:break
        device=devices[gpu];others=set(device['pids'])-own_pids
        if device.get('blocked') or others:continue
        own=sum(v['gpu']==gpu for v in active.values())
        free=device['free']
        while pending and own<2 and free>=minimum_free:
            job=pending.pop(0);chosen.append((job,gpu));own+=1;free-=minimum_free
    return chosen


def completion_matches(entry):
    done=read(entry['completion']);rt=read(entry['runtime']);job=read(entry['job'])
    manifest=read(Path(rt['data_root'])/job['capsule']/'manifest.json')
    return (done.get('status')=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False
            and done.get('query_fit') is False and done.get('capsule_id')==manifest['capsule_id']
            and done.get('model_seed')==job['seed'] and int(done.get('rows',0))>0)


def dispatch(config):
    validate_plan(config)
    root=Path(config['run_root']).resolve();root.mkdir(parents=True,exist_ok=True)
    control=root/'scheduler';control.mkdir(exist_ok=False);(control/'logs').mkdir()
    write(control/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    pending=list(config['jobs']);active={};done=[];failed=[];stopped=False
    from .runner import occupancy
    while pending or active:
        if (control/'USER_STOP.json').is_file():
            stopped=True;pending.clear()
        for key,entry in list(active.items()):
            process=entry['process'];code=process.poll()
            if code is None:continue
            entry['stream'].close();job=entry['job'];marker=Path(job['completion'])
            try:valid=code==0 and completion_matches(job)
            except (FileNotFoundError,ValueError,KeyError,json.JSONDecodeError):valid=False
            (done if valid else failed).append(dict(id=key,exit_code=code,marker=str(marker)))
            del active[key]
        if failed:pending.clear()
        if not failed and not stopped:
            for job,gpu in choose(pending,{k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in active.items()},
                                  occupancy(),config['allowed_gpus'],config['minimum_free_MiB']):
                module='clean_scalr_worker' if job['group']=='pamf' else 'clean_external_worker'
                cmd=[sys.executable,'-u','-m',__package__+'.'+module,'--runtime',job['runtime'],'--job',job['job']]
                if job['group']!='pamf':cmd.extend(['--group',job['group']])
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                log=(control/'logs'/(job['id']+'.log')).open('x',encoding='utf-8')
                try:process=subprocess.Popen(cmd,cwd=config['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except Exception:log.close();raise
                active[job['id']]=dict(job=job,gpu=gpu,process=process,stream=log);pending.remove(job)
        status='FAILED' if failed else 'STOPPED' if stopped else 'RUNNING'
        write(control/'state.json',dict(status=status,completed=len(done),failed=failed,pending=len(pending),
             active=[dict(id=k,pid=v['process'].pid,gpu=v['gpu']) for k,v in active.items()],truth_opened=False),exclusive=False)
        if pending or active:time.sleep(config.get('poll_seconds',5))
    if failed:raise RuntimeError('Technical failure retained; no further jobs dispatched')
    if stopped:return
    write(control/'predictions_complete.json',dict(status='SELECTED_PREDICTIONS_FIXED',jobs=len(done),truth_read=False,
          global_prediction_barrier=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','dispatch'));p.add_argument('--config',required=True)
    args=p.parse_args();config=read(args.config)
    if args.action=='check':print(json.dumps(validate_plan(config)))
    else:dispatch(config)


if __name__=='__main__':main()
