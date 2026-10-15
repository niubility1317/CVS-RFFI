"""Fixed five-row E20, source freeze, per-row seven views then fresh CPU truth."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from . import train_design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    env=dict(os.environ)
    if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
    os.execve(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_mixed_iq20/train_runner.py'),'--mode',mode,'--row',row],env)


def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU process limit exceeded')
        import torch
        torch.cuda.init()
        from .train_source import train
        train(c)
    execute('predict',row)


def dispatch():
    with lock(d.ROOT/'mixed_IQ_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run: reconcile, never overwrite or resubmit')
        spec=d.read(d.ROOT/'experiments/cvs_mixed_iq20/train_registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered fixed E20 matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip()
        d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:
            from .train_panel import prepare
            prepare()
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source smoke/panel',error=str(exc),no_automatic_retry=True))
            raise
        pending=d.rows();active={};launched=[];complete=[];failures=[];qualified=False
        while pending or active:
            for rid,child in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                done=d.read(d.BASE/rid/'complete.json') if (d.BASE/rid/'complete.json').is_file() else {}
                if rc or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=rid,exit_code=rc,completion=done))
                else:complete.append(rid)
                del active[rid]
            ready=all((d.BASE/r['row_id']/'source/source_panel_results.json').is_file() for r in d.rows())
            if ready and not qualified:
                from .train_panel import qualification
                qualification();qualified=True
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_mixed_iq20/train_runner.py'),'--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),launched_at=time.time()))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=complete,failures=failures,
                source_qualified=qualified,per_GPU_limit=3,all_owners=True,updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Owned failure preserved; healthy rows unchanged')
        if not qualified:raise ValueError('No source-only qualification was frozen')
        from .train_report import collect
        result=collect()
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',actual_model_count=5,seed_count=1,
            rows=complete,truth_last=True,performance_status=result['performance_status'],goal_completed=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
    if a.mode=='worker':worker(a.row)
    elif a.mode=='predict':
        from .train_evaluate import predict
        predict(a.row);execute('score',a.row)
    elif a.mode=='score':
        from .train_evaluate import score
        score(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
    else:dispatch()
