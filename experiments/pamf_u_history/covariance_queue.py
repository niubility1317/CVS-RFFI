"""Continue the unchanged three-seed matrix with bounded shared GPU admission."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import time
from experiments.cric_dura_comparison.io import read,write
from .covariance_three_seed import RUN as SCIENCE_RUN,METHODS,NEW_SEEDS,SEEDS,OLD_CONFIRM

RUN='20261027-control-PAMF-Covariance-threeSeed-queue-r01'
RELEASE='20261027-control-PAMF-Covariance-threeSeed-queue-release-r01'
MODULE='experiments.pamf_u_history.covariance_three_seed'
POLICY=dict(max_own_GPU_workers=4,max_total_GPU_workers_per_card=2,max_own_GPU_workers_per_card=1,min_free_MiB=10000,CPU_scorers=1)


def validate_context(c):
    if set(c)!={'project','python','frozen_sources','original_runtime'} or c['frozen_sources'] or c['original_runtime']!=c['project']+'/releases/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-release-r01/runtime.json':raise ValueError('One original scientific runtime;controller has no data/model handles')
    return c


def admissible(gpu,pids,pending,active,free):
    reserved={pid for pid,g in pending.items() if g==gpu}
    own=[v for v in active.values() if v.get('gpu') is not None]
    return free>=10000 and len(set(pids)|reserved)<2 and len(own)<4 and not any(v['gpu']==gpu for v in own)


def controller_identity(expected,pid,cwd,args,uid,owner_uid):
    return uid==owner_uid and pid==expected['pid'] and cwd==expected['cwd'] and args==expected['argv']


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        live=p.joinpath('stat').read_text().split()[2]!='Z'
        return dict(pid=pid,live=live,cwd=os.readlink(p/'cwd') if live else None,
            argv=[s.decode() for s in p.joinpath('cmdline').read_bytes().split(b'\0') if s],uid=p.stat().st_uid)
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,live=False)


def expected_jobs():
    names={'calibrate-s'+str(s) for s in NEW_SEEDS}
    names|={'train__'+m+'-s'+str(s) for m in METHODS for s in NEW_SEEDS}
    names|={kind+'__F0-'+m+'-s'+str(s) for kind in ('predict','score') for m in METHODS for s in SEEDS if s!=SEEDS[0] or m=='CovFull-U'}
    return names


def workers(old,jobs):
    expected={tuple([j['execution_python'],'-u','-m',MODULE,'worker','--kind',j['kind'],'--config',j['config']]):j for j in jobs};found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=tuple(s.decode() for s in p.joinpath('cmdline').read_bytes().split(b'\0') if s)
            if args not in expected:continue
            info=process(int(p.name))
            if not info['live']:continue
            if info['cwd']!=old['release_root']:raise ValueError('Unexpected own-worker checkout')
            job=expected[args];key=job['id']
            if key in found:raise ValueError('Duplicate own workers;preserve without launching')
            env=dict(v.split(b'=',1) for v in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in v);g=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
            if job['kind']!='score' and g not in map(str,range(8)):raise ValueError('Actual worker GPU must be explicit')
            found[key]=dict(status='RUNNING',pid=info['pid'],cwd=info['cwd'],argv=list(args),gpu=None if job['kind']=='score' else int(g),
                log=str(Path(old['run_root'])/'logs'/(key+'.log')),adopted=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return found


def reconcile(jobs,previous,observed,done,log_exists):
    result={};active={};stop=False
    for job in jobs:
        key=job['id'];item=deepcopy(previous[key])
        if key in observed:result[key]=active[key]=observed[key];continue
        if done(job):result[key]=dict(item,status='COMPLETE');continue
        if item['status'] in ('RUNNING','FAILED','COMPLETE'):
            result[key]=dict(item,status='FAILED');stop=True;continue
        if item['status']!='QUEUED' or log_exists(job):raise ValueError('Unknown partial job output;do not overwrite')
        result[key]=dict(status='QUEUED')
    return result,active,stop


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    if rt['run_id']!=RUN or not rt.get('cpu_only') or rt.get('queue_policy')!=POLICY:raise ValueError('One bounded queue handoff only')
    c=validate_context(read(rt['context']));old=read(c['original_runtime']);science=Path(old['run_root']);root=Path(rt['run_root'])
    if old['run_id']!=SCIENCE_RUN or old['entrypoint']!=MODULE:raise ValueError('Original scientific run required')
    if (science/'complete.json').exists():raise ValueError('Scientific run already complete;do not replace controller')
    jobs=read(science/'jobs.json');by={j['id']:j for j in jobs}
    if set(by)!=expected_jobs() or len(jobs)!=28:raise ValueError('Original complete28job matrix required')
    for j in jobs:
        if j['execution_release']!=old['release_root'] or j['execution_python']!=old['python'] or j['entrypoint']!=MODULE or not Path(j['config']).is_relative_to(science/'configs'):raise ValueError('Every worker must use original frozen code/config')
    root.mkdir(parents=True,exist_ok=False);write(root/'process.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=__import__('sys').argv))
    expected=read(Path(old['release_root'])/'submit.json');before=process(expected['pid']);write(root/'original_controller_before.json',before)
    if before['live']:
        if not controller_identity(expected,before['pid'],before['cwd'],before['argv'],before['uid'],os.getuid()):raise ValueError('Only exact own original controller may be stopped')
        os.kill(expected['pid'],signal.SIGTERM)
        for _ in range(50):
            if not process(expected['pid'])['live']:break
            time.sleep(.1)
        else:raise RuntimeError('Original controller still live;no new launches')
    after=process(expected['pid']);write(root/'original_controller_after.json',after)
    if after['live']:raise ValueError('One live controller required')
    previous=read(science/'pipeline_state.json');write(root/'original_pipeline_state.json',previous)
    observed=workers(old,jobs);items,active,stop=reconcile(jobs,previous['jobs'],observed,done,lambda j:(science/'logs'/(j['id']+'.log')).exists())
    write(root/'handoff.json',dict(status='CONTROLLER_HANDOFF_VERIFIED',original_controller_stopped=not after['live'],adopted_workers=observed,
        queue_policy=POLICY,scientific_commit=old['code_commit'],queue_commit=rt['code_commit'],source_target_model_config_changes=0,new_scientific_rows=0))
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=old['code_commit'],queue_code_commit=rt['code_commit'],queue_control_run=RUN,queue_policy=POLICY,jobs=items)
    children={}
    while True:
        for key,item in list(active.items()):
            code=children[key].poll() if key in children else None
            live=code is None if key in children else process(item['pid'])['live']
            if live:continue
            good=(code is None or code==0) and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
            stop|=not good;del active[key];children.pop(key,None)
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(score=0,predict=1,calibrate=2,train=3)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v['pid'] for v in active.values()}):continue
            else:
                choices=[]
                for card in old['allowed_gpus']:
                    pids,memory=capacity(card)
                    if admissible(card,pids,pending,active,memory):choices.append((-memory,card))
                if not choices:continue
                gpu=min(choices)[1]
            child,item=launch(old,job,gpu);active[key]=item;children[key]=child;state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state);atomic(science/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'THREE_SEED_COVARIANCE_COMPLETE';atomic(root/'pipeline_state.json',state);atomic(science/'pipeline_state.json',state)
    if not stop:
        result=dict(status='THREE_SEED_COVARIANCE_COMPLETE',model_seeds=list(SEEDS),methods=list(METHODS),logical_models=12,new_training=8,reused_seed01_models=4,
            reused_seed01_readouts=3,new_prediction_rows=9,raw_only=True,truth_last=True,goal_complete=False,queue_control_run=RUN)
        write(science/'complete.json',result);write(root/'complete.json',dict(status='CONTROL_COMPLETE',science_complete=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();dispatch(read(a.config))
