"""Continue only unstarted Group2 jobs while preserving one known failure.

No failed job is retried, no closed prediction is recalculated, and the global
truth-last barrier stays closed until the deferred job is resolved separately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, write, require
from .group2_schedule import tasks, closed_prediction, validate_runtime, _queue, _smoke_gpu


def require_same_method(previous, current):
    # Only run identity, recovery bookkeeping, and execution capacity may
    # differ. Unlisted configuration (including any training overrides) must
    # remain identical; this compares semantic configuration, not file hashes.
    allowed={'run_id','run_root','release_root','runtime_file','code_commit','authorization','stop_rule',
             'launch_owner','previous_run_root','previous_runtime_file','deferred_failure_keys',
             'gpu_limits','minimum_free_MiB','score_processes','poll_seconds','priority_peer'}
    old={key:value for key,value in previous.items() if key not in allowed}
    new={key:value for key,value in current.items() if key not in allowed}
    require(old==new,'Previous experiment identity differs')


def verify_reuse_binding(rt, job):
    cap=Path(rt['data_root'])/job['capsule']
    manifest=read(cap/'manifest.json')
    index=read(Path(job['output'])/'predictions.json')
    require(index['capsule_id']==manifest['capsule_id'], 'Reused capsule identity differs')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    for item in index['rows']:
        spec=specs[item['k'],item['new_count']]
        require(item['split_id']==spec['split_id'] and item['registered_classes']==spec['registered_classes'],
                'Reused prediction split or classes differ')


def classify(rt, previous_root, deferred_keys):
    reused=[]; deferred=[]; pending=[]
    for job in tasks(rt):
        old=Path(previous_root)/'predictions'/job['key']
        previous=dict(job,output=str(old))
        if (old/'complete.json').is_file():
            closed_prediction(previous,len(job['points'])*len(job['methods']))
            verify_reuse_binding(rt,previous)
            reused.append(previous)
        elif job['key'] in deferred_keys:
            require(old.is_dir(), 'Known failure output is missing')
            deferred.append(previous)
        elif old.exists():
            raise ValueError('Unreconciled partial prediction directory: '+str(old))
        else:
            pending.append(job)
    require({job['key'] for job in deferred}==set(deferred_keys), 'Deferred job key mismatch')
    return reused,deferred,pending


def classify_chain(rt, previous_roots, deferred_keys):
    """Inventory closed and quarantined outputs across the actual ancestors."""
    jobs = tasks(rt)
    by_key = {job['key']: job for job in jobs}
    require(len(by_key) == len(jobs), 'Duplicate task keys')
    found = {}
    for root in previous_roots:
        directory = Path(root)/'predictions'
        require(directory.is_dir(), 'Ancestor predictions directory missing')
        for output in directory.iterdir():
            require(output.is_dir() and output.name in by_key, 'Unexpected ancestor output')
            require(output.name not in found, 'Multiple ancestor outputs for '+output.name)
            found[output.name] = output
    reused = []; deferred = []; pending = []
    for job in jobs:
        output = found.get(job['key'])
        if output is None:
            require(job['key'] not in deferred_keys, 'Known failure output is missing')
            pending.append(job)
            continue
        previous = dict(job, output=str(output))
        if job['key'] in deferred_keys:
            require(not (output/'complete.json').exists(), 'Known failure is already closed')
            deferred.append(previous)
        elif (output/'complete.json').is_file():
            closed_prediction(previous, len(job['points'])*len(job['methods']))
            verify_reuse_binding(rt, previous)
            reused.append(previous)
        else:
            raise ValueError('Unreconciled partial prediction directory: '+str(output))
    require({job['key'] for job in deferred} == set(deferred_keys), 'Deferred job key mismatch')
    return reused, deferred, pending


def dispatch(rt):
    validate_runtime(rt)
    previous=Path(rt['previous_run_root'])
    previous_rt=read(rt['previous_runtime_file'])
    require(str(previous)==previous_rt['run_root'] and rt['previous_runtime_file']==previous_rt['runtime_file'],
            'Previous owner or runtime path differs')
    require_same_method(previous_rt,rt)
    chain=[]; seen=set(); ancestor=previous_rt
    while True:
        root=Path(ancestor['run_root'])
        require(str(root) not in seen, 'Repeated ancestor root')
        seen.add(str(root))
        state=read(root/'worker_state.json')
        require(state['status']=='FAILED' and not state['active'], 'Previous owner has not drained')
        require_same_method(ancestor,rt)
        chain.append((root,ancestor,state))
        if not ancestor.get('previous_run_root'):
            break
        parent=read(ancestor['previous_runtime_file'])
        require(parent['run_root']==ancestor['previous_run_root'] and
                parent['runtime_file']==ancestor['previous_runtime_file'], 'Ancestor runtime path differs')
        ancestor=parent
    chain.reverse()
    failures=set()
    for _,ancestor,state in chain:
        require(set(ancestor.get('deferred_failure_keys', []))==failures,
                'Ancestor deferred failures differ')
        current_failures={entry['key'] for entry in state['failed']}
        require(current_failures and not (failures & current_failures), 'Ancestor failures overlap or are missing')
        failures.update(current_failures)
    # State files can be stale; reject a still-live predecessor controller or
    # worker only when PID, cwd and exact prior runtime all identify this run.
    for root,ancestor,_ in chain:
        launches=[root/'owner.json']+list((root/'launches').glob('*.json'))
        for path in launches:
            launch=read(path); pid=launch.get('pid'); proc=Path('/proc')/str(pid)
            if not pid or not proc.exists(): continue
            try:
                cwd=os.readlink(proc/'cwd'); argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:
                continue
            require(not (cwd==ancestor['release_root'] and ancestor['runtime_file'] in argv),
                    'Previous run still has an owned controller or worker')
    deferred_keys=set(rt['deferred_failure_keys'])
    require(deferred_keys==failures, 'Deferred set differs from recorded failures')
    reused,deferred,pending=classify_chain(rt,[root for root,_,_ in chain],deferred_keys)
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    if rt.get('priority_peer'):
        # All ancestor/configuration/prediction checks passed and the successor
        # now has a real immutable root. Failed preparation leaves no claim.
        write(previous.parent/'.resume_successors'/(previous.name+'.json'),
              dict(successor_runtime=rt['runtime_file'],successor_root=str(root)))
    for name in ('logs','launches','jobs'): (root/name).mkdir()
    for index,job in enumerate(pending): write(root/'jobs'/f'{index:05d}.json',job)
    write(root/'resume_manifest.json',dict(reused=reused,deferred=deferred,pending_count=len(pending),
        previous_run_root=str(previous),truth_read=False,query_fit=False))
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    gpu=_smoke_gpu(rt)
    while gpu is None:
        time.sleep(rt['poll_seconds']);gpu=_smoke_gpu(rt)
    command=[sys.executable,'-u','-m',__package__+'.group2_schedule','smoke','--runtime',rt['runtime_file']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'logs'/'smoke.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(root/'launches'/'smoke.json',dict(pid=proc.pid,gpu=gpu,cwd=str(Path.cwd()),argv=command))
        require(proc.wait()==0, 'Real-source no-query smoke failed')
    _queue(rt,pending,'worker')
    for job in pending: closed_prediction(job,len(job['points'])*len(job['methods']))
    write(root/'pending_predictions_complete.json',dict(status='INCOMPLETE_DEFERRED_FAILURES',
        completed_pending=len(pending),reused=len(reused),deferred=[j['key'] for j in deferred],
        truth_read=False,query_fit=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True)
    args=parser.parse_args();dispatch(read(args.runtime))
