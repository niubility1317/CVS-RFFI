"""Adopt Group1 workers and accelerate only not-yet-started external jobs."""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import time
from .cuda_takeover import Adopted, _check_children
from .external_takeover import process
from .group1_dispatch import completed,jobs
from .io import atomic,read,require,write


def resolve_jobs(items,done,active,new_by,rt):
    resolved=[]
    for original in items:
        j=dict(original);key=j['id']
        use_new=(j['origin']=='legacy' and j['kind']!='source' and key not in done and key not in active)
        if use_new and j['kind']=='score':
            predecessor=j['dependencies'][0]
            use_new=predecessor not in done and predecessor not in active
        if use_new:
            new=new_by[j['original_id']]
            for field in ('output','completion','prediction'):
                if field in new:j[field]=new[field]
            j['runtime_file']=rt['runtime_file'];j['release_root']=rt['release_root']
            j['execution_version']='external_cuda'
        resolved.append(j)
    return resolved


def _parent(rt):
    state=read(Path(rt['run_root'])/'pipeline_state.json');pid=state['controller_pid'];item=process(pid)
    require(state['status']=='RUNNING' and not state['counts']['FAILED'] and not state['counts']['BLOCKED'],
            'Existing controller is not healthy')
    require(item and item['uid']==os.getuid() and item['cwd']==rt['release_root']
            and rt['runtime_file'] in item['argv']
            and 'experiments.pamf_phase2_practical.cuda_takeover' in item['argv']
            and '--job' not in item['argv'],'Existing controller identity mismatch')
    return pid


def _workers(configs,by):
    lookup={(j['runtime_file'],j['original_id']):j['id'] for j in by.values()}
    require(len(lookup)==len(by),'Duplicate runtime/job bindings')
    found={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        item=process(int(path.name))
        if not item or item['uid']!=os.getuid() or '--job' not in item['argv']:continue
        matches=[name for name in configs if name in item['argv']]
        if not matches:continue
        require(len(matches)==1 and 'experiments.pamf_phase2_practical.group1_dispatch' in item['argv'],
                'Unexpected process using Group1 runtime')
        cfg=configs[matches[0]];require(item['cwd']==cfg['release_root'],'Worker CWD mismatch')
        original=item['argv'][item['argv'].index('--job')+1]
        require((matches[0],original) in lookup,'Unknown active worker')
        key=lookup[matches[0],original];require(key not in found,'Duplicate active worker')
        try:env=dict(v.split('=',1) for v in (path/'environ').read_bytes().decode().split('\0') if '=' in v)
        except (FileNotFoundError,ProcessLookupError):continue
        gpu=env.get('CUDA_VISIBLE_DEVICES','');require(gpu=='' or gpu.isdigit(),'Unexpected GPU setting')
        found[key]=dict(pid=item['pid'],gpu=int(gpu) if gpu else None,kind=by[key]['kind'],
            profile=by[key]['profile'],command=item['argv'],cwd=item['cwd'],runtime_file=matches[0],
            origin=by[key]['origin'],adopted=True,log=str(Path(cfg['run_root'])/'logs'/(original+'.log')),
            started_at=None,output=by[key]['output'],completion=by[key]['completion'],prediction=by[key].get('prediction'))
    require(all(v<=2 for v in Counter(x['gpu'] for x in found.values() if x['gpu'] is not None).values()),
            'Existing workers exceed GPU limit')
    require(sum(x['kind']=='score' for x in found.values())<=1,'Multiple existing scorers')
    return found


def _check_pending(items,configs):
    for j in items:
        cfg=configs[j['runtime_file']];log=Path(cfg['run_root'])/'logs'/(j['original_id']+'.log')
        require(not Path(j['output']).exists() and not log.exists(),
                'Unowned partial output or launched-job log: '+j['id'])


def dispatch(rt):
    parent_rt=read(rt['takeover_runtime']);previous=read(parent_rt['takeover_runtime'])
    legacy=read(previous['recovery_runtime'])
    require(parent_rt['run_id']=='20261030-phase2-group1-cuda-takeover-m4-r05','Unexpected current run')
    require(rt['external_source_root']==legacy['run_root'],'External source root mismatch')
    identity={'run_id','release_root','run_root','runtime_file','code_commit','launch_owner',
              'authorization','method_profiles'}
    require(set(legacy).issubset(rt),'New runtime omitted existing fields')
    for name in set(legacy)-identity:
        require(rt[name]==legacy[name],'External scientific input changed: '+name)
    root=Path(rt['run_root']);old_root=Path(parent_rt['run_root'])
    configs={c['runtime_file']:c for c in (parent_rt,previous,legacy)}
    items=read(old_root/'jobs.json');by={j['id']:j for j in items}
    require(len(by)==len(items),'Duplicate existing job ID')
    new_by={j['id']:j for j in jobs(rt)}
    require(all(j['original_id'] in new_by for j in items if j['origin']=='legacy'),
            'External matrix does not match existing jobs')
    require(all(completed(j) for j in items if j['kind']=='source'),'Source jobs not all complete')
    producer=read(Path(legacy['run_root'])/'augmentation_owner.json')
    require(process(producer['pid']) is not None,'Source augmentation producer missing')
    old_roots=[Path(c['run_root']) for c in configs.values()]
    require(not any((p/'USER_STOP.json').exists() for p in old_roots),'Existing stop request')
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();(root/'job_results').mkdir()
    write(root/'original_jobs.json',items)
    from .external_accel_smoke import run as smoke
    smoke(rt,root/'smoke.json')
    parent=_parent(parent_rt);os.kill(parent,signal.SIGSTOP);time.sleep(.3);terminated=False
    try:
        state=read(old_root/'pipeline_state.json')
        require(state['controller_pid']==parent and not state['counts']['FAILED']
                and not state['counts']['BLOCKED'],'Owner changed during handoff')
        observed=_workers(configs,by);_check_children(parent,observed)
        done={j['id'] for j in items if completed(j)}
        for key in state['active']:
            require(key in observed or key in done,'Worker exited without completion: '+key)
        require(not set(observed).intersection(done),'Active job already has completion')
        remaining=set(by)-set(observed)-done
        _check_pending((by[key] for key in remaining),configs)
        resolved=resolve_jobs(items,done,observed,new_by,rt);write(root/'jobs.json',resolved)
        by={j['id']:j for j in resolved};configs[rt['runtime_file']]=rt
        write(root/'owner.json',dict(pid=os.getpid(),runtime=rt['runtime_file'],owner=rt['launch_owner'],
            replacement_for=parent_rt['run_id']))
        write(root/'handoff_prepared.json',dict(old_controller_pid=parent,adopted_workers=list(observed.values()),
            completed=len(done),pending=len(remaining)))
        os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT);deadline=time.monotonic()+5
        while process(parent) and time.monotonic()<deadline:time.sleep(.1)
        require(process(parent) is None,'Old controller did not exit');terminated=True
    finally:
        if not terminated and process(parent):os.kill(parent,signal.SIGCONT)
    write(root/'handoff.json',dict(status='VERIFIED',old_controller_pid=parent,old_controller_exited=True,
        model_workers_restarted=0,adopted_workers=list(observed.values()),scientific_config_unchanged=True))
    active={k:(Adopted(v['pid']),v) for k,v in observed.items()};failed={};blocked=set();profile_errors=set()
    while True:
        stop=any((p/'USER_STOP.json').exists() for p in [root,*old_roots])
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and completed(by[key]);result=dict(item,status='COMPLETE' if good else 'FAILED',
                exit_code=code,ended_at=time.time(),output=by[key]['output'],completion=by[key]['completion'],
                prediction=by[key].get('prediction'))
            write(root/'job_results'/(key+'.json'),result)
            if good:done.add(key)
            else:failed[key]=result;profile_errors.add((by[key]['origin'],by[key]['profile']))
            del active[key]
        for key in list(remaining):
            if ((by[key]['origin'],by[key]['profile']) in profile_errors
                    or any(dep in failed or dep in blocked for dep in by[key]['dependencies'])):
                blocked.add(key);remaining.remove(key)
        available=sorted((by[k] for k in remaining if all(d in done for d in by[k]['dependencies'])),
            key=lambda j:(j['priority'],j['origin']!='replacement',j['id']))
        def launch(job,gpu):
            cfg=configs[job['runtime_file']]
            if job['kind']=='score':
                predecessor=by[job['dependencies'][0]]
                require(completed(predecessor) and job['prediction']==predecessor['output'],
                        'Scorer prediction root mismatch')
            args=[cfg['python'],'-u','-m','experiments.pamf_phase2_practical.group1_dispatch',
                  '--runtime',cfg['runtime_file'],'--job',job['original_id']]
            env=dict(os.environ,PYTHONPATH=cfg['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=Path(cfg['run_root'])/'logs'/(job['original_id']+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(args,cwd=cfg['release_root'],env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
            item=dict(pid=child.pid,gpu=gpu,kind=job['kind'],profile=job['profile'],command=args,
                cwd=cfg['release_root'],runtime_file=cfg['runtime_file'],log=str(log),started_at=time.time(),
                origin=job['origin'],adopted=False,output=job['output'],completion=job['completion'],
                prediction=job.get('prediction'))
            active[job['id']]=(child,item);remaining.remove(job['id']);available.remove(job)
            print('EXTERNAL_ACCEL_LAUNCH',job['id'],child.pid,gpu,cfg['run_id'],flush=True)
        if not stop:
            if not any(v['kind']=='score' for _,v in active.values()):
                scorer=next((j for j in available if not j['gpu']),None)
                if scorer:launch(scorer,None)
            from .runner import occupancy
            devices=occupancy()
            for _ in (0,1):
                for gpu in rt['allowed_gpus']:
                    own=sum(v['gpu']==gpu for _,v in active.values())
                    if own>=2 or devices[gpu]['free']<rt['minimum_free_MiB']:continue
                    job=next((j for j in available if j['gpu']),None)
                    if job is None:continue
                    launch(job,gpu);devices[gpu]['free']-=rt['minimum_free_MiB']
        counts=dict(COMPLETE=len(done),FAILED=len(failed),BLOCKED=len(blocked),RUNNING=len(active),QUEUED=len(remaining))
        status='STOP_REQUESTED' if stop else 'RUNNING'
        if not active and not remaining:status='COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json',dict(status=status,run_id=rt['run_id'],controller_pid=os.getpid(),
            counts=counts,active={k:v for k,(_,v) in active.items()},updated_at=time.time(),
            failed_jobs=list(failed),previous_run=parent_rt['run_id']))
        if (not active and not remaining) or (stop and not active):break
        time.sleep(rt.get('poll_seconds',10))
    if not stop:Path(rt['augmentation_cache'],'stop').write_text('Group1 external acceleration completed\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);args=parser.parse_args()
    dispatch(read(args.runtime))
