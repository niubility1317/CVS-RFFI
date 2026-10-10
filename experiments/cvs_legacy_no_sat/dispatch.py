"""Each seed owns its train -> freeze -> predict -> independent score chain."""
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_legacy_no_sat import design as d
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock,process

def worker(rid):
    c=d.config(next(r for r in d.rows() if r['row_id']==rid))
    if d.read(d.BASE/'configs'/(rid+'.json'))!=c:raise ValueError('Config differs')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);stable=0
    while stable<3:
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            cap=capacity()[gpu];ready=len(cap['pids']|{os.getpid()})<=2 and cap['free_mb']>=12000
            stable=stable+1 if ready else 0
        if stable<3:time.sleep(1 if ready else 5)
    try:
        from experiments.cvs_legacy_no_sat.source import train
        train(c)
        from experiments.cvs_legacy_no_sat import evaluate as e
        e.source_provenance(d,c)
        d.write(d.BASE/rid/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False,
            selection='FIXED_OWN_E200',frozen_at=time.time()))
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
        e.EVAL_ROW=rid;e.predict(rid)
        # Separate process is the first stage to open truth; other trainers never read it.
        subprocess.run([sys.executable,'-m','experiments.cvs_legacy_no_sat.evaluate','--mode','score','--row',rid],
            cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        result=d.read(d.BASE/'row_scoring'/rid/'scoring_complete.json')
        if result['models']!=1 or result['views']!=7 or result['metric_records']!=98 or result['day_metric_records']!=28:raise ValueError('Incomplete row score')
        d.write(d.BASE/rid/'completion.json',dict(status='ANALYZED',views=7,independent_recount='VERIFIED'))
    except Exception as exc:
        d.write(d.BASE/rid/'failure.json',dict(status='FAILED',error=repr(exc)));raise

def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
        configs=[d.config(r) for r in d.rows()]
        for c in configs:d.write(d.BASE/'configs'/(c['row_id']+'.json'),c)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),owner=d.OWNER))
        pending=list(configs);active={};receipts=[];done=[];failures=[]
        try:
            while pending or active:
                for rid,child in list(active.items()):
                    rc=child.poll()
                    if rc is None:continue
                    path=d.BASE/rid/'completion.json'
                    if rc or not path.is_file() or d.read(path)['status']!='ANALYZED':failures.append(dict(row_id=rid,exit_code=rc))
                    else:done.append(rid)
                    del active[rid]
                while pending and len(active)<3 and not failures:
                    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        candidates=[(len(x['pids']),-x['free_mb'],gpu) for gpu,x in capacity().items() if len(x['pids'])<2 and x['free_mb']>=12000]
                        if not candidates:break
                        gpu=min(candidates)[2];c=pending.pop(0);rid=c['row_id'];log=d.BASE/'logs'/(rid+'.log')
                        cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_legacy_no_sat/train_worker.py'),'--row',rid]
                        with log.open('x') as f:
                            child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        if process(child.pid) is None:raise RuntimeError('Absent worker after launch')
                        receipts.append(dict(row_id=rid,pid=child.pid,gpu=gpu,log=str(log),argv=cmd,cwd=str(d.ROOT)))
                        d.write(d.BASE/'launch.json',dict(rows=receipts));active[rid]=child
                d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=[c['row_id'] for c in pending],completed=done,failures=failures,per_gpu_limit=2,updated_at=time.time()))
                if failures and not active:raise RuntimeError('Failed rows retained, no retry '+repr(failures))
                if pending or active:time.sleep(10)
            subprocess.run([sys.executable,'-m','experiments.cvs_legacy_no_sat.evaluate','--mode','score'],cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
            result=d.read(d.BASE/'scoring_complete.json')
            if result['models']!=3 or result['views']!=7 or result['metric_records']!=294 or result['day_metric_records']!=84:raise ValueError('Incomplete matrix score')
            d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=3,views=7,independent_recount='VERIFIED',all_rows_scored_automatically=True))
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc)));raise

if __name__=='__main__':dispatch()
