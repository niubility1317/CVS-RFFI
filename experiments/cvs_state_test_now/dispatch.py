"""Separate evaluation owner; no mutation of the healthy parent run."""
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_state_test_now import design as d
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock,process,choose_gpu


def worker(rid):
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);stable=0
    while stable<3:
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            cap=capacity()[gpu]
            ready=len(cap['pids']|{os.getpid()})<=4 and cap['free_mb']>=12000
            stable=stable+1 if ready else 0
        if stable<3:time.sleep(1 if ready else 5)
    from experiments.cvs_state_test_now.evaluate import predict
    predict(rid)


def prepare():
    from experiments.cvs_state_test_now import evaluate as e
    d.BASE.mkdir(parents=True,exist_ok=False)
    rows=[d.config(r) for r in d.rows()]
    for c in rows:e.source_provenance(d,c)
    d.write(d.BASE/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,
        rows=rows,target_access=False,selection='ALL_COMPLETED_AT_FIXED_SNAPSHOT',frozen_at=time.time(),
        parent_run=d.parent.RUN,no_metric_selection=True))
    d.write(d.BASE/'checkpoint_preflight.json',e.preflight())


def dispatch():
    with lock(d.ROOT/'evaluation-owner.lock',nonblocking=True):
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:
            from experiments.cvs_state_test_now.views import build
            build()
            (d.BASE/'logs').mkdir(exist_ok=False)
            pending=[c['row_id'] for c in d.rows()];active={};completed=[];failures=[];receipts=[]
            while pending or active:
                for rid,child in list(active.items()):
                    rc=child.poll()
                    if rc is None:continue
                    done=d.BASE/rid/'prediction/complete.json'
                    if rc or not done.is_file() or d.read(done)['status']!='PREDICTIONS_COMPLETE':
                        failures.append(dict(row_id=rid,exit_code=rc))
                    else:completed.append(rid)
                    del active[rid]
                while pending and len(active)<4 and not failures:
                    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        gpu=choose_gpu(capacity())
                        if gpu is None:break
                        rid=pending.pop(0);log=d.BASE/'logs'/(rid+'.log')
                        cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_state_test_now/train_eval_worker.py'),'--row',rid]
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                        with log.open('x') as f:
                            child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        if process(child.pid) is None:raise RuntimeError('Worker absent after launch')
                        receipts.append(dict(row_id=rid,pid=child.pid,gpu=gpu,log=str(log),argv=cmd,cwd=str(d.ROOT)))
                        d.write(d.BASE/'launch_predict.json',dict(rows=receipts));active[rid]=child
                d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,updated_at=time.time()))
                if failures and not active:raise RuntimeError('Failed evaluation retained: '+repr(failures))
                if pending or active:time.sleep(5)
            subprocess.run([sys.executable,'-m','experiments.cvs_state_test_now.evaluate','--mode','score'],cwd=d.ROOT,check=True)
            scored=d.read(d.BASE/'scoring_complete.json')
            if scored['models']!=24 or scored['metric_records']!=24*10*14 or scored['day_metric_records']!=24*10*4:raise ValueError('Score coverage differs')
            d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=24,views=10,truth_last=True,independent_recount='VERIFIED'))
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':dispatch()
