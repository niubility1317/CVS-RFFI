"""Score each fixed clean prediction task without waiting for other tasks."""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import json
import os
from pathlib import Path
import re
import time

from .clean_auto_score import EXCLUDED_PLANS, expected_units, unit, validate_prediction_rows
from .clean_comparison_plan import tasks
from .clean_score import expected_rows, score_job, verify_all
from .io import atomic, require, write


def unique_entries(pairs):
    result=[];seen=set()
    for plan,entry in pairs:
        name=Path(plan).parent.name;key=entry['id']
        require(re.fullmatch(r'[A-Za-z0-9_-]+',name) is not None and
                re.fullmatch(r'[A-Za-z0-9_-]+',key) is not None,'Unsafe score identity')
        identity=(name,key)
        if identity in seen:continue
        seen.add(identity);result.append(dict(entry=entry,score_key=name+'/'+key))
    return result


def score_entry(payload):
    entry,output=payload;output=Path(output)
    if output.exists():
        return dict(status='ALREADY_SCORED' if (output/'complete.json').is_file() else 'PRESERVED_INCOMPLETE_OUTPUT')
    item=verify_all({'jobs':[entry]},{'jobs':1})[0]
    item['payload']['rows']=[dict(row,prediction=str((item['folder']/row['prediction']).resolve()))
                            for row in item['payload']['rows']]
    validate_prediction_rows(item)
    score_job(item,output)
    return dict(status='SCORED_COMPLETE',rows=len(item['payload']['rows']))


def discover(preparations,expected):
    pairs=[];errors=[]
    plans=set(Path(preparations).glob('*phase2*clean*/scheduler.json'))
    plans.update(Path(preparations).glob('mopc-stable-*/scheduler.json'))
    for plan in sorted(plans):
        if plan.parent.name in EXCLUDED_PLANS:continue
        try:
            config=json.loads(plan.read_text(encoding='utf-8'))
            require(config.get('input_condition')=='clean_ground_pre_satellite_normalized' and
                    isinstance(config.get('jobs'),list),'Invalid clean plan')
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            errors.append(dict(plan=plan.parent.name,error=f'{type(exc).__name__}: {exc}'));continue
        for entry in config['jobs']:
            try:
                require(re.fullmatch(r'[A-Za-z0-9_-]+',plan.parent.name) is not None and
                        re.fullmatch(r'[A-Za-z0-9_-]+',entry['id']) is not None,'Unsafe score identity')
                marker=Path(entry['completion'])
                if not marker.is_file():continue
                complete=json.loads(marker.read_text(encoding='utf-8'))
                require(complete.get('status')=='PREDICTIONS_COMPLETE','Prediction completion is not fixed')
                job=json.loads(Path(entry['job']).read_text(encoding='utf-8'))
                group=entry['group'];u=unit(group,job);version=job.get('mopc_stabilization','none')
                require(version in expected and u in expected[version],'Job outside retained matrix')
                canonical=job
                if group=='hybrid' and job.get('profile')=='protonet_cda' and \
                   job.get('mopc_bridge','original')=='original' and 'mopc_hr' in job.get('methods',[]):
                    # Historical original bridge is scored under its own arm, never merged with affine.
                    canonical=dict(job,mopc_bridge='proto_affine_v1')
                rows=expected_rows(group,canonical)
                require(rows and rows.issubset(expected[version][u]),'Job differs from retained K/N matrix')
                pairs.append((plan,entry))
            except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                errors.append(dict(plan=plan.parent.name,id=entry.get('id','UNKNOWN') if isinstance(entry,dict) else 'UNKNOWN',
                                   error=f'{type(exc).__name__}: {exc}'))
    return unique_entries(pairs),errors


def next_pending(backlog,recent,prefer_recent):
    if recent and (prefer_recent or not backlog):return recent.popleft(),False
    if backlog:return backlog.popleft(),True
    return recent.popleft(),False


def run(preparations,output,workers,interval):
    require(1<=workers<=4 and interval>=60,'Wrong incremental scoring limits')
    root=Path(output).resolve();require(not root.exists(),'Scoring root exists')
    root.mkdir(parents=True);(root/'jobs').mkdir();(root/'failures').mkdir()
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),workers=workers,
        preparations=str(Path(preparations).resolve()),scope='individually_fixed_jobs',truth_last=True))
    expected={v:expected_units(tasks(),v) for v in ('none','clip1000_v1')}
    backlog=deque();recent=deque();seen=set();active={};counts=Counter();last_scan=0;discovery_errors=[]
    scans=0;prefer_recent=True
    effective_workers=workers;pool_restarts=0;pool=ProcessPoolExecutor(max_workers=effective_workers)
    try:
        while True:
            stopping=(root/'USER_STOP.json').is_file()
            if not stopping and time.monotonic()-last_scan>=interval:
                try:
                    found,discovery_errors=discover(preparations,expected)
                    buckets={}
                    for item in found:
                        key=item['score_key']
                        if key in seen:continue
                        try:
                            job=json.loads(Path(item['entry']['job']).read_text(encoding='utf-8'))
                            bucket=(key.split('/')[0],item['entry']['group'],job.get('profile','R42'))
                            buckets.setdefault(bucket,deque()).append(item);seen.add(key)
                        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                            discovery_errors.append(dict(score_key=key,error=f'{type(exc).__name__}: {exc}'))
                    while buckets:
                        for bucket in list(buckets):
                            (backlog if scans==0 else recent).append(buckets[bucket].popleft())
                            if not buckets[bucket]:del buckets[bucket]
                    scans+=1
                except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                    discovery_errors=[dict(error=f'{type(exc).__name__}: {exc}')]
                last_scan=time.monotonic()
            broken=[];pool_broken=False
            while (backlog or recent) and len(active)<effective_workers and not stopping:
                item,prefer_recent=next_pending(backlog,recent,prefer_recent);target=root/'jobs'/item['score_key']
                try:
                    future=pool.submit(score_entry,(item['entry'],str(target)));active[future]=item
                except BrokenProcessPool:
                    recent.appendleft(item);pool_broken=True;break
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED) if active else (set(),set())
            for future in done:
                item=active.pop(future);key=item['score_key']
                try:
                    result=future.result();counts[result['status']]+=1
                    if result['status']=='PRESERVED_INCOMPLETE_OUTPUT':
                        write(root/'failures'/(key.replace('/','--')+'.json'),dict(score_key=key,**result))
                except BrokenProcessPool:
                    broken.append(item);pool_broken=True
                except Exception as exc:
                    counts['FAILED']+=1
                    write(root/'failures'/(key.replace('/','--')+'.json'),dict(score_key=key,status='FAILED',
                        error=f'{type(exc).__name__}: {exc}'))
            if pool_broken:
                pool.shutdown(wait=True,cancel_futures=True)
                broken.extend(active.values());active.clear();pool_restarts+=1
                if effective_workers>1:
                    effective_workers=1
                    recent.extendleft(reversed(broken))
                else:
                    for item in broken:
                        key=item['score_key'];counts['FAILED']+=1
                        write(root/'failures'/(key.replace('/','--')+'.json'),dict(score_key=key,
                            status='FAILED',error='Scoring worker exited unexpectedly; output retained'))
                pool=ProcessPoolExecutor(max_workers=effective_workers)
            atomic(root/'state.json',dict(status='DRAINING' if stopping and active else 'STOPPED' if stopping else 'RUNNING',
                discovered=len(seen),queued=len(backlog)+len(recent),recent_queued=len(recent),active=len(active),counts=dict(counts),
                discovery_error_count=len(discovery_errors),discovery_errors=discovery_errors[:20],
                workers=effective_workers,pool_restarts=pool_restarts,scope='individually_fixed_jobs',truth_last=True))
            if stopping and not active:return
            if not active and not backlog and not recent:time.sleep(5)
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparations',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--interval',type=int,default=120)
    a=p.parse_args();run(a.preparations,a.output,a.workers,a.interval)


if __name__=='__main__':main()
