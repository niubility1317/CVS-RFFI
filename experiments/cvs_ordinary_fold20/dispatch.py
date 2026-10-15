"""One owner, current all-owner GPU protection, fixed two source rows."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def worker(row_id):
    c=d.config(next(r for r in d.rows() if r['row_id']==row_id))
    with lock(d.BASE/(row_id+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
        from .source import train
        train(c)


def compare_sequences():
    paths=[d.BASE/r['row_id']/'source/training_sequence.jsonl.gz' for r in d.rows()]
    count=0
    with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b:
        from itertools import zip_longest
        for left,right in zip_longest(a,b):
            if left!=right:raise ValueError('Matched real L/U physical sample order differs')
            count+=1
    return dict(status='VERIFIED',batch_records=count,all_L_U_physical_sample_sequences_equal=True)


def dispatch():
    with lock(d.ROOT/'ordinary_RX1_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve/reconcile existing immutable run, never resubmit')
        spec=d.read(d.ROOT/'experiments/cvs_ordinary_fold20/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered matrix differs from native design')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip();d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        start=time.time();pending=d.rows();active={};launched=[];completed=[];failed=[]
        while pending or active:
            for row,child in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                path=d.BASE/row/'source/result.json'
                if rc or not path.is_file():failed.append(dict(row=row,exit_code=rc))
                else:completed.append(row)
                del active[row]
            if pending and not failed:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));r=pending.pop(0)
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_ordinary_fold20/runner.py'),'--row',r['row_id']]
                        log=d.BASE/'logs'/(r['row_id']+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[r['row_id']]=child
                        launched.append(dict(row_id=r['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),launched_at=time.time()))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failed=failed,updated_at=time.time()))
            if failed and not active:break
            if pending or active:time.sleep(3)
        if failed:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Own technical failure preserved; healthy rows not stopped')
        try:
            sequence=compare_sequences();d.write(d.BASE/'matched_sequence.json',sequence)
            result=d.qualify({r['arm']:d.read(d.BASE/r['row_id']/'source/result.json') for r in d.rows()})
            d.write(d.BASE/'source_decision.json',result)
            d.write(d.BASE/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_decision=result['status'],rows=completed,
                source_only=True,target_access=False,automatic_extension=False,goal_completed=False,wall_seconds=time.time()-start))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),no_automatic_retry=True));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--row');args=parser.parse_args()
    if args.row:worker(args.row)
    else:dispatch()
