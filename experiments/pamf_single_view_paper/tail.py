"""Append the paused paper matrix after both view-shuffle groups, without overwrite."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
from experiments.cric_dura_comparison.io import read,write
from . import runner
from .plan import RUN,make_plan
from .source import saved_segments

TAIL='20261027-phase1-pamf-single-view-paper-tail-r01'
PREDECESSOR='20261027-phase1-pamf-repl-shuffled-views-m16-r02'

def validate_runtime(rt):
    project=Path(rt['project'])
    if (rt['continuation_run_id']!=TAIL or rt['run_id']!=RUN
        or Path(rt['original_root'])!=project/'runs'/RUN
        or Path(rt['queue_root'])!=project/'runs'/TAIL
        or Path(rt['run_root'])!=project/'runs'/TAIL/'pipeline'
        or Path(rt['wait_for'])!=project/'runs'/PREDECESSOR/'analysis/complete.json'
        or rt.get('reuse_completed_jobs') is not True
        or rt.get('max_own_per_GPU')!=1 or rt.get('max_total_per_GPU')!=2):
        raise ValueError('Exact paused matrix, predecessor and immutable roots required')

def predecessor_ready(rt):
    marker=Path(rt['wait_for'])
    if not marker.exists():return False
    d=read(marker)
    if d.get('status')!='ANALYZED' or d.get('models')!=16:
        raise ValueError('Both predecessor groups must complete independent scoring')
    return True

def materialize(rt,persist=True):
    validate_runtime(rt)
    old=Path(rt['original_root']);root=Path(rt['run_root'])
    state=read(old/'pipeline_state.json')
    if state['status']!='PAUSED_BY_USER_PRIORITY' or Path('/proc/'+str(state['pid'])).exists():
        raise ValueError('Original paper controller must remain paused and exited')
    for v in state['jobs'].values():
        if v.get('status')=='PAUSED_BY_USER_PRIORITY' and Path('/proc/'+str(v.get('pid'))).exists():
            raise ValueError('Original paused worker still exists')
    original=read(old/'jobs.json')
    if len(original)!=348 or sum(j['kind']=='train' for j in original)!=108:
        raise ValueError('Original frozen 120-row matrix required')
    jobs=deepcopy(original);configs={j['id']:read(j['config']) for j in jobs};mapping={};reused=[]
    from experiments.cric_dura_comparison.pipeline import done
    for j in jobs:
        if done(j):reused.append(j['id']);continue
        if state['jobs'][j['id']]['status']=='COMPLETE':
            raise ValueError('Previously completed artifact is missing: '+j['id'])
        c=configs[j['id']];prior=Path(c['output_root'])
        dest=root/('F' if j['kind']=='train' else 'readouts')/c['row_id']/('source' if j['kind']=='train' else 'predictions' if j['kind']=='predict' else 'scores')
        mapping[str(prior)]=str(dest);mapping[str(Path(j['completion']))]=str(dest/'complete.json')
    def remap(path):return mapping.get(str(Path(path)),path)
    segment_rows={};pending=Counter()
    for j in jobs:
        if j['id'] in reused:continue
        c=configs[j['id']];pending[j['kind']]+=1
        prior=c['output_root'];c['output_root']=remap(prior)
        j['completion']=remap(j['completion'])
        j['dependencies']=[remap(p) for p in j.get('dependencies',[])]
        if j['kind']=='train':
            c['code_commit']=rt['code_commit'];c['continuation_run_id']=TAIL
            if (Path(prior)/'resolved_config.json').exists():
                c['resume_source_root']=prior
                segment_rows[c['row_id']]=[x['updates'] for x in saved_segments(c)]
        for key in ('source_output','prediction_output'):
            if key in c:c[key]=remap(c[key])
        path=root/'configs'/(j['id']+'.json');j['config']=str(path)
        if persist:write(path,c)
    sources=read(old/'sources.json')
    for row in sources.values():
        for key in ('source_output','predictions','scores'):row[key]=remap(row[key])
    summary=dict(continuation_run_id=TAIL,original_run_id=RUN,logical_models=120,
                 reused_stages=dict(Counter(j['kind'] for j in jobs if j['id'] in reused)),
                 remaining_stages=dict(pending),saved_segments=segment_rows,
                 predecessor=PREDECESSOR,partial_in_memory_work_not_resumed=True)
    if persist:
        write(root/'jobs.json',jobs);write(root/'sources.json',sources);write(root/'recovery_plan.json',summary)
    return jobs,summary

def dispatch(rt):
    validate_runtime(rt)
    from experiments.practical_full_2x2.cache import atomic
    queue=Path(rt['queue_root']);queue.mkdir(parents=True,exist_ok=False)
    _,summary=materialize(rt,persist=False)
    state=dict(status='WAITING_FOR_PREDECESSOR',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
               code_commit=rt['code_commit'],wait_for=rt['wait_for'],GPU_workers=0,plan=summary)
    while not predecessor_ready(rt):
        state['updated_at']=time.time();atomic(queue/'queue_state.json',state)
        print('WAITING_FOR_PREDECESSOR '+PREDECESSOR,flush=True) if 'announced' not in state else None
        state['announced']=True
        time.sleep(30)
    state.update(status='STARTING_PAPER_REMAINDER',updated_at=time.time());atomic(queue/'queue_state.json',state)
    original_prepare=runner.prepare
    runner.prepare=lambda config:materialize(config)[0]
    try:runner.dispatch(rt)
    finally:runner.prepare=original_prepare
    final=read(Path(rt['run_root'])/'pipeline_state.json')
    state.update(status=final['status'],updated_at=time.time(),pipeline_state=str(Path(rt['run_root'])/'pipeline_state.json'))
    atomic(queue/'queue_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','audit']);p.add_argument('--config',required=True);a=p.parse_args();rt=read(a.config)
    if a.action=='audit':print(__import__('json').dumps(materialize(rt,persist=False)[1]))
    else:dispatch(rt)
