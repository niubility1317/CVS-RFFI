"""User-authorized handoff of a never-started queue; immutable workers reused.

New launches use wholly idle GPUs, not the last slot of a busy GPU. The old
launchers can see our named train/predict entries before CUDA initializes.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

WORKER_COMMIT='ae783c81cd4949f93a80dbd2e80c4b2568eb99ea'


def visible_reservations(proc_root=Path('/proc')):
    result={}
    for p in proc_root.iterdir():
        if not p.name.isdigit():continue
        try:
            state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state in ('Z','X','x'):continue
            argv=(p/'cmdline').read_bytes()
            if not argv:continue
            env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
            device=env.get('CUDA_VISIBLE_DEVICES','')
            if device.isdigit():result.setdefault(int(device),set()).add(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeError):continue
    return result


def idle_only(caps,reservations):
    result={}
    for gpu,c in caps.items():
        pids=set(c['pids'])|set(reservations.get(gpu,()))
        result[gpu]=dict(pids=pids,free_mb=0 if pids else c['free_mb'])
    return result


def check_pending(d):
    configs=[d.config(r) for r in d.rows()]
    if (d.ROOT/'release_commit.txt').read_text().strip()!=WORKER_COMMIT:raise ValueError('Worker release changed')
    q=d.read(d.BASE/'queue_state.json')
    if q['phase']!='WAITING_PRIOR_OWNERS' or q['active'] or set(q['pending'])!={c['row_id'] for c in configs}:
        raise ValueError('Queue is no longer wholly pending; do not replace owner')
    if list(d.BASE.glob('launch_*.json')) or any((d.BASE/c['row_id']).exists() for c in configs):
        raise ValueError('Worker artifacts exist; no duplicate launch')
    if (d.BASE/'failure.json').exists() or (d.BASE/'completion.json').exists():raise ValueError('Terminal run')
    for c in configs:
        if d.read(d.BASE/'configs'/(c['row_id']+'.json'))!=c:raise ValueError('Registered config changed')
    return configs


def run(worker_root):
    sys.path[:0]=[str(worker_root),str(worker_root/'code')]
    from experiments.cvs_phase1_repair import dispatch as controller
    from experiments.cvs_phase1_stack.capacity16 import proc
    d=controller.d;configs=check_pending(d)
    handoff=d.read(Path(__file__).parent/'handoff.json')
    old=handoff['previous_owner']
    if proc(old['pid']):raise ValueError('Previous repair owner still alive')
    if handoff['run_id']!=d.RUN or handoff['worker_commit']!=WORKER_COMMIT:raise ValueError('Handoff mismatch')
    with (d.BASE/'direct_owner.json').open('x') as f:
        __import__('json').dump(dict(pid=os.getpid(),commit=(Path(__file__).parent/'release_commit.txt').read_text().strip()),f)
    sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'))
    from scripts import dispatch_xuc_full as capacity
    original=capacity.occupancy
    capacity.occupancy=lambda active:idle_only(original(active),visible_reservations())
    controller.competing_owners=lambda:[]
    original_popen=subprocess.Popen
    def guarded_popen(cmd,*args,**kwargs):
        entries={str(d.ROOT/'experiments/cvs_phase1_repair'/n) for n in ('train_repair.py','predict_repair.py')}
        if isinstance(cmd,list) and len(cmd)>2 and cmd[2] in entries:
            if '--config' in cmd:
                rid=Path(cmd[cmd.index('--config')+1]).stem;kind='source'
            else:rid=cmd[cmd.index('--row')+1];kind='predict'
            cmd=[sys.executable,'-u',str(Path(__file__).parent/'train_direct.py'),
                '--worker-root',str(d.ROOT),'--lease',str(d.BASE/'capacity_leases'/(kind+'-'+rid+'.json')),
                '--entry',cmd[2],'--',*cmd[3:]]
        return original_popen(cmd,*args,**kwargs)
    controller.subprocess.Popen=guarded_popen
    d.write(d.BASE/'dispatcher_active.json',dict(**proc(os.getpid()),worker_commit=WORKER_COMMIT,
        control_commit=(Path(__file__).parent/'release_commit.txt').read_text().strip(),
        policy='START_ON_IDLE_GPU; max4 workers; all-process pre-CUDA reservation; preserve other owners',handoff=handoff))
    try:
        controller.queue(configs,'source')
        records=[controller.source_complete(c) for c in configs]
        d.write(d.BASE/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=configs,
            source_metrics=[r['final_source_metrics'] for r in records],target_access=False,
            selection='ALL_PREREGISTERED_FIXED_CONTROLS',frozen_at=time.time()))
        controller.queue(configs,'predict')
        subprocess.run([sys.executable,'-m','experiments.cvs_phase1_repair.evaluate','--mode','score'],cwd=d.ROOT,check=True)
        if d.read(d.BASE/'scoring_complete.json')['status']!='SCORED_COMPLETE':raise ValueError('Scoring incomplete')
        from experiments.cvs_phase1_repair.evaluate import VIEWS
        d.write(d.BASE/'completion.json',dict(status='ANALYZED',rows=len(configs),views=list(VIEWS),
            independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),no_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True);run(p.parse_args().worker_root)
