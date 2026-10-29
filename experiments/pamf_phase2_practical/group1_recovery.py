"""PAMF replacement plus unchanged, adopted Group1 external workers.

Only the identified CPU dispatcher is retired. Existing GPU children retain
their PID, release, runtime, scientific configuration and output directory.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import time
from .io import read,write,atomic,require
from .external_takeover import process
from .group1_dispatch import jobs,completed
from .group1 import row


def matrix(rt,old):
    result=[]
    for origin,cfg in (('legacy',old),('replacement',rt)):
        for job in jobs(cfg):
            if origin=='legacy' and row(job['profile'])['method']=='pamf':continue
            key=origin+'-'+job['id']
            result.append(dict(job,id=key,original_id=job['id'],origin=origin,
                runtime_file=cfg['runtime_file'],release_root=cfg['release_root'],
                dependencies=[origin+'-'+v for v in job['dependencies']]))
    return result


def suspend_parent(old):
    root=Path(old['run_root']);owner=read(root/'owner.json');pid=owner['pid'];item=process(pid)
    require(item and item['uid']==os.getuid() and item['cwd']==old['release_root']
        and old['runtime_file'] in item['argv']
        and 'experiments.pamf_phase2_practical.group1_dispatch' in item['argv']
        and '--job' not in item['argv'],'Legacy CPU dispatcher identity mismatch')
    os.kill(pid,signal.SIGSTOP)
    time.sleep(.2)
    return pid


def running_workers(old,by):
    result={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        item=process(int(path.name))
        if not item or item['uid']!=os.getuid() or item['cwd']!=old['release_root']:continue
        argv=item['argv']
        if old['runtime_file'] not in argv or '--job' not in argv:continue
        require('experiments.pamf_phase2_practical.group1_dispatch' in argv,'Unexpected legacy worker entry')
        original=argv[argv.index('--job')+1];key='legacy-'+original
        require(key in by,'Unexpected legacy job during replacement')
        env=dict(v.split('=',1) for v in (path/'environ').read_bytes().decode().split('\0') if '=' in v)
        gpu=env.get('CUDA_VISIBLE_DEVICES','')
        result[key]=dict(item,gpu=int(gpu) if gpu.isdigit() else None,
            kind=by[key]['kind'],profile=by[key]['profile'],adopted=True,
            command=argv,log=str(Path(old['run_root'])/'logs'/(original+'.log')),
            started_at=None,origin='legacy',runtime_file=old['runtime_file'])
    return result


class Adopted:
    def __init__(self,pid):self.pid=pid
    def poll(self):return None if process(self.pid) else 0


def dispatch(rt):
    old=read(rt['recovery_runtime']);root=Path(rt['run_root'])
    require(rt['method_profiles'] in (['pamf_pl_milkr'],['pamf_tri_sig_time_ssl_milkr']),
            'Replacement must contain only PAMF')
    require(rt['augmentation_cache']==old['augmentation_cache'],'Source simulation cache must remain shared')
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();(root/'job_results').mkdir()
    items=matrix(rt,old);by={j['id']:j for j in items}
    write(root/'jobs.json',items)
    write(root/'owner.json',dict(pid=os.getpid(),owner=rt['launch_owner'],runtime=rt['runtime_file'],
        replacement_for=old['run_id'],legacy_queues_resumed=False))
    producer=read(Path(old['run_root'])/'augmentation_owner.json')
    require(process(producer['pid']) is not None,'Existing source augmentation producer missing')
    correction=read(rt['method_correction_stop']) if rt.get('method_correction_stop') else None
    if correction:
        require(correction['controller_exited'] and process(correction['stopped_controller']['pid']) is None
            and all(process(v['pid']) is None for v in correction['stopped_PAMF_workers']),
            'Wrong-recipe controller/workers still alive')
        parent=None
    else:parent=suspend_parent(old)
    try:
        observed=running_workers(old,by)
        occupancy=Counter(v['gpu'] for v in observed.values() if v['gpu'] is not None)
        require(all(v<=2 for v in occupancy.values()),'Legacy own workers exceed two/card')
        active={key:(Adopted(v['pid']),v) for key,v in observed.items()}
        done={j['id'] for j in items if j['id'] not in active and completed(j)}
        failed={};profile_errors=set()
        for j in items:
            file=Path(old['run_root'])/'job_results'/(j['original_id']+'.json')
            if j['origin']=='legacy' and file.is_file() and read(file)['status']=='FAILED':
                failed[j['id']]=read(file);profile_errors.add((j['origin'],j['profile']))
        remaining=set(by)-set(active)-done-set(failed);blocked=set()
        # Terminate only the independently verified CPU parent, never its children.
        if parent is not None:
            os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
            deadline=time.monotonic()+5
            while process(parent) and time.monotonic()<deadline:time.sleep(.1)
            require(process(parent) is None,'Legacy CPU parent did not exit')
        handoff=dict(status='VERIFIED',legacy_parent_pid=parent,legacy_parent_exited=True,
            model_workers_restarted=0,adopted_workers=list(observed.values()),
            scientific_config_unchanged=True,output_paths_unchanged=True,
            controller_pid=os.getpid(),controller_runtime=rt['runtime_file'])
        write(root/'handoff.json',handoff)
        old_handoff='group1_controller_'+rt['run_id']+'.json' if correction else 'group1_replacement_handoff.json'
        write(Path(old['run_root'])/old_handoff,handoff)
    finally:
        if parent is not None and process(parent):os.kill(parent,signal.SIGCONT)
    while True:
        stop=(root/'USER_STOP.json').is_file() or (Path(old['run_root'])/'USER_STOP.json').is_file()
        if any(by[k]['kind']=='source' for k in remaining) and process(producer['pid']) is None:
            raise RuntimeError('Source cache producer failed; preserve active workers')
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and completed(by[key])
            result=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code,ended_at=time.time())
            write(root/'job_results'/(key+'.json'),result)
            if good:done.add(key)
            else:failed[key]=result;profile_errors.add((by[key]['origin'],by[key]['profile']))
            del active[key]
        for key in list(remaining):
            j=by[key]
            if (j['origin'],j['profile']) in profile_errors or any(d in failed or d in blocked for d in j['dependencies']):
                remaining.remove(key);blocked.add(key)
        available=sorted((by[k] for k in remaining if all(d in done for d in by[k]['dependencies'])),
            key=lambda j:(j['priority'],j['origin']!='replacement',j['id']))
        def launch(j,gpu):
            cfg=old if j['origin']=='legacy' else rt
            args=[cfg['python'],'-u','-m','experiments.pamf_phase2_practical.group1_dispatch',
                '--runtime',cfg['runtime_file'],'--job',j['original_id']]
            env=dict(os.environ,PYTHONPATH=cfg['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=Path(cfg['run_root'])/'logs'/(j['original_id']+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(args,cwd=cfg['release_root'],env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
            item=dict(pid=child.pid,gpu=gpu,kind=j['kind'],profile=j['profile'],command=args,
                cwd=cfg['release_root'],runtime_file=cfg['runtime_file'],log=str(log),
                started_at=time.time(),origin=j['origin'],adopted=False)
            active[j['id']]=(child,item);remaining.remove(j['id']);available.remove(j)
            print('GROUP1_RECOVERY_LAUNCH',j['id'],child.pid,gpu,flush=True)
        if not stop:
            if not any(v['kind']=='score' for _,v in active.values()):
                j=next((j for j in available if not j['gpu']),None)
                if j:launch(j,None)
            from .runner import occupancy as devices_now
            devices=devices_now()
            for slot in (0,1):
                for gpu in rt['allowed_gpus']:
                    own=sum(v['gpu']==gpu for _,v in active.values())
                    if own>=2 or devices[gpu]['free']<rt['minimum_free_MiB']:continue
                    j=next((j for j in available if j['gpu']),None)
                    if j:launch(j,gpu);devices[gpu]['free']-=rt['minimum_free_MiB']
        counts=dict(COMPLETE=len(done),FAILED=len(failed),BLOCKED=len(blocked),RUNNING=len(active),QUEUED=len(remaining))
        status='STOP_REQUESTED' if stop else 'RUNNING'
        if not remaining and not active:status='COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json',dict(status=status,run_id=rt['run_id'],controller_pid=os.getpid(),
            counts=counts,active={k:v for k,(_,v) in active.items()},updated_at=time.time(),
            superseded_PAMF_failure_preserved=True,external_run_id=old['run_id'],legacy_queues_resumed=False))
        if (not active and not remaining) or (stop and not active):break
        time.sleep(rt.get('poll_seconds',10))
    Path(rt['augmentation_cache'],'stop').write_text('group1 unified controller completed\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);a=p.parse_args()
    dispatch(read(a.runtime))
