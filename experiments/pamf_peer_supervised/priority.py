"""User-authorized queue priority handoff; detached healthy scientific workers survive."""
from copy import deepcopy
import os
from pathlib import Path
import signal
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.pipeline import done
from experiments.pamf_repl_completion.recovery import process_info,AdoptedProcess

OLD_COMMIT='1fa92a0b8b61fba7ddca532a12078a8ddae2bdf7'
OLD_RUN='20261025-phase1-PAMF-ChannelPL-m33-r01'
OLD_OWNER='codex-channel-pl-training-20261025'


def owner_matches(info,rt):
    expected=[rt['python'],'-u','-m','experiments.pamf_channel_pl.training_runner','dispatch','--config',rt['runtime_file']]
    return bool(info and info['cwd']==rt['release_root'] and info['argv']==expected)


def worker_item(job,oldrt,pid):
    info=process_info(pid)
    if info is None:return None
    expected=[oldrt['python'],'-u','-m','experiments.pamf_channel_pl.training_runner','worker',
        '--kind',job['kind'],'--config',job['config']]
    gpu=None if job['kind']=='score' else job['gpu']
    if (info['cwd']!=oldrt['release_root'] or info['argv']!=expected
        or info['gpu']!=('' if gpu is None else str(gpu))):
        raise ValueError('Exact old worker PID/CWD/argv/GPU mismatch; no duplicate adoption')
    return dict(status='RUNNING',pid=pid,cwd=info['cwd'],argv=info['argv'],gpu=gpu,
        log=str(Path(oldrt['run_root'])/'logs'/(job['id']+'.log')),adopted=True)


def handoff(rt):
    if rt.get('queue_priority_authorization')!='human_defer_unstarted_21_preserve_healthy_and_all_GPU':
        raise ValueError('Explicit human queue priority authorization required')
    oldrt=read(rt['previous_runtime']);oldroot=Path(oldrt['run_root']);root=Path(rt['run_root'])
    state=read(oldroot/'pipeline_state.json');jobs=read(oldroot/'jobs.json')
    if (oldrt['run_id']!=OLD_RUN or oldrt['code_commit']!=OLD_COMMIT or state['code_commit']!=OLD_COMMIT
        or oldrt['launch_owner']!=OLD_OWNER or state['launch_owner']!=OLD_OWNER or state['status']!='RUNNING'
        or len(jobs)!=99 or {j['id'] for j in jobs}!=set(state['jobs'])):
        raise ValueError('Only the declared live ChannelPL owner may be handed off')
    pid=state['pid']
    if not owner_matches(process_info(pid),oldrt):raise ValueError('Old controller identity differs; nothing stopped')
    # Preserve the exact pre-handoff snapshot before stopping only this PID.
    write(root/'previous_pipeline_state.json',state)
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if process_info(pid) is None:break
        time.sleep(.1)
    if process_info(pid) is not None:raise RuntimeError('Old controller remains alive; refuse a second launch owner')
    state=read(oldroot/'pipeline_state.json');write(root/'previous_pipeline_state_after_stop.json',state)
    byconfig={j['config']:j for j in jobs};live={}
    # Reconcile a detached launch that preceded the old controller's last state write.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            info=process_info(int(p.name))
            if not info or '--config' not in info['argv']:continue
            cfg=info['argv'][info['argv'].index('--config')+1]
            if cfg not in byconfig:continue
            job=byconfig[cfg];item=worker_item(job,oldrt,int(p.name))
            if item:
                if job['id'] in live:raise ValueError('Duplicate existing scientific worker; no new launches')
                live[job['id']]=item
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    converted=[];states={};active={};children={};started_rows=set();deferred=[]
    for old in jobs:
        job=deepcopy(old);oldid=old['id'];key='old::'+oldid
        job.update(id=key,original_id=oldid,lane='old',execution_release=oldrt['release_root'],
            entrypoint='experiments.pamf_channel_pl.training_runner',execution_python=oldrt['python'],
            log_root=str(oldroot/'logs'))
        prior=state['jobs'][oldid]
        if oldid in live:
            item=live[oldid];states[key]=item;active[key]=item;children[key]=AdoptedProcess(item['pid']);started_rows.add(job['row_id'])
        elif done(old):
            states[key]=dict(prior,status='COMPLETE');started_rows.add(job['row_id'])
        elif prior['status']=='QUEUED':
            # Existing partial output without an owner must never be overwritten.
            c=read(job['config'])
            if Path(c['output_root']).exists() or (oldroot/'logs'/(oldid+'.log')).exists():
                raise ValueError('Unowned partial output requires technical reconciliation')
            states[key]=dict(status='QUEUED')
        else:raise ValueError('Missing completion for previously active/failed scientific job')
        converted.append(job)
    for job in converted:
        if job['kind']=='train' and states[job['id']]['status']=='QUEUED' and job['row_id'] not in started_rows:
            deferred.append(job['row_id'])
    write(root/'priority_handoff.json',dict(status='HEALTHY_WORKERS_ADOPTED',previous_runtime=rt['previous_runtime'],
        stopped_controller_pid=pid,healthy_worker_PIDs=[v['pid'] for v in live.values()],
        scientific_execution_commit=OLD_COMMIT,scientific_jobs_unchanged=99,deferred_rows=deferred,
        authorization=rt['queue_priority_authorization'],new_owner_pid=os.getpid(),new_controller_commit=rt['code_commit'],
        worker_processes_stopped=0,config_changes=0,model_changes=0))
    write(oldroot/'continuation_owner.json',dict(pid=os.getpid(),runtime=rt['runtime_file'],control_root=str(root),
        status='PRIORITY_HANDOFF',previous_dispatcher_pid=pid,scientific_execution_commit=OLD_COMMIT))
    return converted,states,active,children,set(deferred),oldrt


def defer(job,completed_new,deferred_rows):
    return job['lane']=='old' and job['row_id'] in deferred_rows and not completed_new


def mirror_old(state,jobs,oldrt,rt):
    old=read(Path(oldrt['run_root'])/'pipeline_state.json')
    old.update(status='RUNNING_UNDER_PRIORITY_OWNER',pid=os.getpid(),controller_commit=rt['code_commit'],
        delegated_to=rt['run_root'],updated_at=time.time(),
        jobs={j['original_id']:state['jobs'][j['id']] for j in jobs if j['lane']=='old'})
    from experiments.practical_full_2x2.cache import atomic
    atomic(Path(oldrt['run_root'])/'pipeline_state.json',old)
