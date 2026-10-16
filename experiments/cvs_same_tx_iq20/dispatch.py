"""One source launch owner, shared capacity, no retries or healthy-job edits."""
import json
import os
import subprocess
import sys
import time
from . import design as d
from experiments.cvs_rdrop_fold20.resources import capacity_lock,capacity,choose
from experiments.cvs_rdrop_sinc_units.startup import launch_once


def run():
    if d.BASE.exists():raise FileExistsError('Immutable source root exists; reconcile rather than relaunch')
    d.BASE.mkdir();(d.BASE/'logs').mkdir()
    from .preflight import smoke
    smoke()
    d.write(d.BASE/'request.json',dict(rows=[d.config(r) for r in d.rows()],source_only=True))
    pending=d.rows().copy();live={};launched=[];complete=[];failure=None
    def record(receipt):
        index=next((i for i,v in enumerate(launched) if v['row_id']==receipt['row_id']),None)
        if index is None:launched.append(receipt.copy())
        else:launched[index]=receipt.copy()
        d.write(d.BASE/'launch.json',dict(rows=launched))
    while pending or live:
        for pid,(child,row) in list(live.items()):
            code=child.poll()
            if code is None:continue
            del live[pid]
            result=d.BASE/row['row_id']/'source/satmix_result.json'
            if code!=0 or not result.is_file():
                failure=dict(row_id=row['row_id'],pid=pid,returncode=code,reason='Source worker failed; preserve all artifacts and healthy workers; no automatic retry')
                d.write(d.BASE/'failure.json',failure);pending=[]
            else:complete.append(row['row_id'])
        if pending and not failure:
            with capacity_lock():
                caps=capacity();gpu=choose(caps)
                if gpu is not None:
                    row=pending.pop(0);path=d.BASE/'logs'/(row['row_id']+'.stdout.log')
                    command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_same_tx_iq20/runner.py'),'--worker',row['row_id']]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    stream=path.open('x')
                    try:
                        create=lambda:subprocess.Popen(command,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        base=dict(row_id=row['row_id'],cwd=str(d.ROOT),argv=command,gpu=gpu,owner=d.OWNER,log=str(path),capacity_before=dict(process_count=len(caps[gpu]['pids']),free_MiB=caps[gpu]['free_MiB']),per_GPU_limit=3)
                        child,receipt=launch_once(create,base,record,timeout=2.)
                        live[child.pid]=(child,row)
                    finally:stream.close()
        d.write(d.BASE/'queue_state.json',dict(pending=[r['row_id'] for r in pending],live=[r['row_id'] for _,r in live.values()],completed=complete,failure=failure,source_only=True))
        if live or pending:time.sleep(3.)
    if failure:raise RuntimeError('Preserved systemic source failure')
    results={r['arm']:d.read(d.BASE/r['row_id']/'source/satmix_result.json') for r in d.rows()}
    qualified=d.qualify(results)
    d.write(d.BASE/'result.json',dict(status='ANALYZED',results=results,qualification=qualified,target_access=False))
    d.write(d.BASE/'completion.json',dict(status='ANALYZED',rows=3,new_epochs=60,new_optimizer_steps=13320,qualified=qualified['qualified'],target_access=False,head_fit_updates=0))
    print(json.dumps(qualified),flush=True)
