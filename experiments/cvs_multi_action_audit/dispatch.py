"""Fixed four-row source diagnostic queue, one launch owner, no target stage."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_multi_action_audit import design as d
from experiments.cvs_multi_disentangle.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock,process,choose_gpu


def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False)
        logs=Path(d.PROJECT)/'logs'/d.RUN;logs.mkdir(parents=True,exist_ok=False)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        pending=d.rows();active={};complete=[];failures=[];launched=[]
        try:
            while pending or active:
                for rid,item in list(active.items()):
                    rc=item['process'].poll()
                    if rc is None:continue
                    path=d.BASE/rid/'completion.json'
                    valid=path.is_file() and d.read(path)['status']=='SOURCE_AUDIT_COMPLETE'
                    (complete if rc==0 and valid else failures).append(rid)
                    del active[rid]
                while pending and not failures:
                    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        gpu=choose_gpu(capacity())
                        if gpu is None:break
                        row=pending.pop(0);rid=row['row_id'];c=d.config(row)
                        d.write(d.BASE/'configs'/(rid+'.json'),c)
                        cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_multi_action_audit/train_worker.py'),'--row',rid]
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'),
                            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                        log=logs/(rid+'.log')
                        with log.open('x') as f:
                            child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        actual=process(child.pid)
                        if actual is None:raise RuntimeError('Worker disappeared')
                        launched.append(dict(row_id=rid,pid=child.pid,start_ticks=actual['start_ticks'],gpu=gpu,
                            cwd=str(d.ROOT),argv=cmd,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
                        active[rid]=dict(process=child,gpu=gpu)
                        print('LAUNCH '+json.dumps(launched[-1]),flush=True)
                d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=[r['row_id'] for r in pending],
                    completed=complete,failures=failures,updated_at=time.time(),per_gpu_limit=4))
                if failures and not active:raise RuntimeError('Rows failed; artifacts retained, no automatic retry: '+repr(failures))
                if pending or active:time.sleep(10)
            d.write(d.BASE/'completion.json',dict(status='SOURCE_AUDIT_COMPLETE',rows=len(complete),
                identity_training=False,target_access=False,new_test_score=None,
                next_stage='Review source evidence before new identity training; no target feedback'))
        except Exception as e:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),no_automatic_retry=True));raise

if __name__=='__main__':dispatch()
