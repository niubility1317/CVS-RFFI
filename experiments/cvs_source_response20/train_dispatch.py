"""Scratch E20 Ag pair, same-row source decision then automatic original tests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from . import train_design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def source_decision(c):
    root=Path(c['output_root']);done=d.read(root/'completion.json');control=d.control_source()
    if done['config']!=c or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access']:raise ValueError('Final source budget differs')
    a=done['final_source_metrics'];b=control['source_metrics']
    checks=dict(clean_gain=a['source_val_accuracy']>=b['source_val_accuracy']+.005,
        worst_RX_health=a['source_val_worst_rx']>=b['source_val_worst_rx']-.01)
    value=dict(status='SOURCE_DECISION_FROZEN',config=c,passed=all(checks.values()),checks=checks,
        source_metrics=a,control=control,clean_gain_pp=100*(a['source_val_accuracy']-b['source_val_accuracy']),
        target_metrics_read=False,before_own_target_prediction=True,recorded_at=time.time(),automatic_joint_training=False,automatic_seed_expansion=False)
    d.write(root/'source_decision.json',value);return value


def execute(mode,row):
    env=dict(os.environ)
    if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
    os.execve(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_source_response20/train_runner.py'),'--mode',mode,'--row',row],env)


def worker(row):
    c=d.config(next(v for v in d.rows() if v['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
        import torch
        torch.cuda.init()
        from experiments.cvs_mixed_iq20.train_source import train
        train(c);source_decision(c)
    execute('predict',row)


def dispatch():
    with lock(d.ROOT/'source_response_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve and reconcile existing run')
        spec=d.read(d.ROOT/'experiments/cvs_source_response20/train_registration.json')
        if [v['config'] for v in spec['rows']]!=[d.config(v) for v in d.rows()]:raise ValueError('Registered Ag matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip();d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:
            from experiments.cvs_mixed_iq20.train_panel import prepare
            prepare();d.write(d.BASE/'control_source_reuse.json',d.control_source())
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source smoke/control',error=str(exc),no_automatic_retry=True));raise
        pending=d.rows();active={};launched=[];complete=[];failures=[]
        while pending or active:
            for rid,child in list(active.items()):
                rc=child.poll()
                if rc is None:continue
                path=d.BASE/rid/'complete.json';done=d.read(path) if path.is_file() else {}
                if rc or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=rid,exit_code=rc,completion=done))
                else:complete.append(rid)
                del active[rid]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_source_response20/train_runner.py'),'--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child;launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),launched_at=time.time()))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=complete,failures=failures,updated_at=time.time(),per_GPU_limit=3,all_owners=True))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True));raise RuntimeError('Owned failure preserved')
        decisions={v['arm']:d.read(Path(d.config(v)['output_root'])/'source_decision.json') for v in d.rows()}
        gain=decisions['EMP']['source_metrics']['source_val_accuracy']-decisions['ROT']['source_metrics']['source_val_accuracy']
        d.write(d.BASE/'source_qualification.json',dict(status='SOURCE_QUALIFICATION_FROZEN',passed=decisions['EMP']['passed'],rows=decisions,
            EMP_minus_ROT_source_clean_pp=100*gain,physical_direction_advantage=gain>0,target_scores_read=False,automatic_joint_training=False,automatic_seed_expansion=False))
        d.write(d.BASE/'complete.json',dict(status='ARTIFACTS_COMPLETE',actual_model_count=2,seed_count=1,rows=complete,
            source_qualified=decisions['EMP']['passed'],truth_last=True,goal_completed=False,analysis_pending=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
    if a.mode=='worker':worker(a.row)
    elif a.mode=='predict':
        c=d.config(next(v for v in d.rows() if v['row_id']==a.row))
        decision=d.read(Path(c['output_root'])/'source_decision.json')
        if decision['config']!=c or not decision['before_own_target_prediction'] or decision['target_metrics_read']:raise ValueError('Source decision must precede own prediction')
        from experiments.cvs_mixed_iq20.train_evaluate import predict
        predict(a.row);execute('score',a.row)
    elif a.mode=='score':
        from experiments.cvs_mixed_iq20.train_evaluate import score
        score(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
    else:dispatch()
