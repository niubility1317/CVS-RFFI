import argparse
import gzip
from itertools import zip_longest
import os
from pathlib import Path
import subprocess
import sys
import time
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def worker(rid):
    c=d.config(next(r for r in d.rows() if r['row_id']==rid))
    with lock(d.BASE/(rid+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
        from .source import train
        train(c)

def dispatch():
    with lock(d.ROOT/'raw_anchor_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve/reconcile existing immutable raw-anchor root')
        spec=d.read(d.ROOT/'experiments/cvs_raw_anchor20/registration.json')
        if [v['config'] for v in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Raw-anchor source matrix differs')
        d.prerequisite();d.BASE.mkdir();(d.BASE/'logs').mkdir();start=time.time()
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        pending=d.rows();active={};launched=[];done=[];failed=[]
        while pending or active:
            for rid,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                if code or not (d.BASE/rid/'source/raw_anchor_proof.json').is_file():failed.append(dict(row_id=rid,exit_code=code))
                else:done.append(rid)
                del active[rid]
            if pending and not failed:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_raw_anchor20/runner.py'),'--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as f:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child;launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=done,failed=failed))
            if failed and not active:break
            if pending or active:time.sleep(3)
        if failed:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,pending_preserved=pending,no_automatic_retry=True));raise RuntimeError('Own failure retained; healthy rows were not stopped')
        try:
            paths=[d.BASE/r['row_id']/'source/training_sequence.jsonl.gz' for r in d.rows()];count=0
            with gzip.open(paths[0],'rt') as x,gzip.open(paths[1],'rt') as y,gzip.open(paths[2],'rt') as z:
                for a,b,c in zip_longest(x,y,z):
                    if a!=b or a!=c:raise ValueError('Actual three-arm full L/U sample sequences differ')
                    count+=1
            d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',actual_L_U_batch_records=count,arms=list(d.ARMS)))
            results={r['arm']:d.read(d.BASE/r['row_id']/'source/result.json') for r in d.rows()};q=d.qualify(results);d.write(d.BASE/'source_decision.json',q)
            d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',source_decision=q['status'],source_only=True,target_access=False,
                automatic_extension=False,new_models=3,new_model_epochs=60,new_updates=13320,wall_seconds=time.time()-start,goal_completed=False))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),no_automatic_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--row');a=p.parse_args()
    if a.row:worker(a.row)
    else:dispatch()
