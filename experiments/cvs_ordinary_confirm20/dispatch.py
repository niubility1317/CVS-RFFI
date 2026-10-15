"""Fixed eight new source rows; reuse RX1 results, never its model states."""
import argparse
import gzip
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


def compare_pair(held):
    from itertools import zip_longest
    paths=[d.BASE/f'{arm}-RX{held}-s{d.SEED}'/'source/training_sequence.jsonl.gz' for arm in d.ARMS]
    count=0
    with gzip.open(paths[0],'rt') as left,gzip.open(paths[1],'rt') as right:
        for a,b in zip_longest(left,right):
            if a!=b:raise ValueError(f'Actual L/U physical sample sequence differs in heldRX{held}')
            count+=1
    return dict(held_RX=held,status='VERIFIED',batch_records=count,actual_L_U_sequences_equal=True)


def dispatch():
    with lock(d.ROOT/'ordinary_source_confirm_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve/reconcile existing immutable run')
        spec=d.read(d.ROOT/'experiments/cvs_ordinary_confirm20/registration.json')
        if [v['config'] for v in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered confirmation matrix differs')
        first=d.prerequisite()
        d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'reused_RX1.json',first)
        spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip();d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        start=time.time();pending=d.rows();active={};launched=[];completed=[];failed=[]
        while pending or active:
            for rid,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                path=d.BASE/rid/'source/result.json'
                if code or not path.is_file():failed.append(dict(row_id=rid,exit_code=code))
                else:completed.append(rid)
                del active[rid]
            if pending and not failed:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_ordinary_confirm20/runner.py'),'--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],held_RX=row['held_RX'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),launched_at=time.time()))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failed=failed,updated_at=time.time()))
            if failed and not active:break
            if pending or active:time.sleep(3)
        if failed:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failed,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Own technical failure preserved; healthy rows were not stopped')
        try:
            sequence=[compare_pair(rx) for rx in d.HELD];d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',folds=sequence))
            folds={1:first['results']}
            for rx in d.HELD:folds[rx]={arm:d.read(d.BASE/f'{arm}-RX{rx}-s{d.SEED}'/'source/result.json') for arm in d.ARMS}
            decision=d.qualify(folds);d.write(d.BASE/'source_decision.json',decision)
            d.write(d.BASE/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_decision=decision['status'],new_rows=completed,
                source_only=True,target_access=False,automatic_extension=False,goal_completed=False,wall_seconds=time.time()-start,
                new_models=8,new_model_epochs=160,new_updates=35520,reused_models=2,reused_model_epochs=40,reused_updates=8880))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),no_automatic_retry=True));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--row');args=parser.parse_args()
    if args.row:worker(args.row)
    else:dispatch()
