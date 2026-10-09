"""Freeze all 32 sources; predict all fixed rows; then independent scorers."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_receiver_residual_test_recovery import design as plan
from experiments.cvs_receiver_residual_test_recovery import evaluate as ev
from experiments.cvs_receiver_residual_capacity4.control import lock, process, choose_gpu, write, read


def freeze(run):
    ev.configure(run['run_id']);d=ev.d;base=Path(plan.PROJECT)/'runs'/run['run_id']
    metrics=[];cost={}
    for c in d.rows():
        source,done,initial,resolved,contract=ev.source_provenance(c)
        if resolved['commit']!=run['source_commit']:raise ValueError('Source runtime commit differs')
        metrics.append(dict(arm=c['arm'],model_seed=c['model_seed'],**done['final_source_metrics']))
        cost[c['arm']]=resolved['total_parameters']
        if not (source/'last.pt').is_file():raise FileNotFoundError('Frozen checkpoint missing')
    scores={a:sum(.5*r['source_val_accuracy']+.5*r['source_val_worst_rx'] for r in metrics if r['arm']==a)/4 for a in d.ARMS}
    for r in metrics:
        if r['source_val_count']!=27000 or set(r['source_val_rx_accuracy'])!=set(map(str,d.SOURCE_RXS)):
            raise ValueError('Source validation coverage differs')
    preferred=max(d.ARMS,key=lambda a:(scores[a],-cost[a],-d.ARMS.index(a)))
    write(base/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=run['run_id'],
        parent_run_id=run['parent_run_id'],rows=d.rows(),source_metrics=metrics,arm_scores=scores,
        source_preferred_arm=preferred,source_preference='mean4(.5V+.5worstRX), exact ties fewer parameters',
        target_access=False,selection='ALL_PREREGISTERED_FIXED_CONTROLS',frozen_at=time.time(),
        checkpoint_provenance='All original contract fields and ordered class map exact; each own scratch E200,10000 steps'))
    return [dict(run=run,row_id=d.row_id(c['arm'],c['model_seed'])) for c in d.rows()]


def capacity():
    # Uses the already validated CUDA + pre-CUDA accounting implementation.
    return importlib.import_module('experiments.'+ev.d.METHOD+'.dispatch').capacity()


def worker(run_id,rid):
    ev.configure(run_id);gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);pid=os.getpid();stable=0
    base=Path(plan.PROJECT)/'runs'/run_id
    while stable<3:
        with lock(Path(plan.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            c=capacity()[gpu];pids=c['pids']|{pid};ready=len(pids)<=4 and c['free_mb']>=12000
            stable=stable+1 if ready else 0
            write(base/'capacity_leases'/(rid+'.json'),dict(pid=pid,gpu=gpu,observed_pids=sorted(pids),
                stable_checks=stable,per_gpu_limit=4,status='CAPACITY_READY' if stable==3 else 'WAITING_CAPACITY'))
        if stable<3:time.sleep(1 if ready else 5)
    print('CAPACITY_READY '+json.dumps(dict(pid=pid,gpu=gpu)),flush=True)
    ev.predict(rid)


def dispatch():
    root=plan.ROOT;owner_file=root/'owner.json'
    with lock(root/'owner.lock',nonblocking=True):
        write(owner_file,dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,launch_owner=plan.OWNER,
            commit=(root/'release_commit.txt').read_text().strip(),started=time.time()))
        rows=[];receipts={r['run_id']:[] for r in plan.RUNS};active={};pending=[];failures=[];complete=[]
        try:
            # All32 checkpoints/source rules fixed before the first predictor starts.
            for run in plan.RUNS:
                base=Path(plan.PROJECT)/'runs'/run['run_id'];base.mkdir(parents=True,exist_ok=False)
                (Path(plan.PROJECT)/'logs'/run['run_id']).mkdir(parents=True,exist_ok=False)
                write(base/'dispatcher.json',dict(read(owner_file),run_id=run['run_id']))
                rows.extend(freeze(run))
            pending=rows.copy()
            while pending or active:
                for key,item in list(active.items()):
                    rc=item['process'].poll()
                    if rc is None:continue
                    base=Path(plan.PROJECT)/'runs'/item['run']['run_id']
                    artifact=base/item['row_id']/'prediction/complete.json'
                    valid=artifact.is_file() and read(artifact)['status']=='PREDICTIONS_COMPLETE'
                    if rc or not valid:failures.append(dict(run_id=item['run']['run_id'],row_id=item['row_id'],exit_code=rc,artifact_valid=valid))
                    else:complete.append(key)
                    del active[key]
                while pending and not failures:
                    with lock(Path(plan.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        gpu=choose_gpu(capacity())
                        if gpu is None:break
                        item=pending.pop(0);run=item['run'];rid=item['row_id'];key=run['run_id']+'/'+rid
                        cmd=[sys.executable,'-u','-m','experiments.cvs_receiver_residual_test_recovery.dispatch',
                            '--worker','--run',run['run_id'],'--row',rid]
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root),
                            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                        log=Path(plan.PROJECT)/'logs'/run['run_id']/(rid+'.log')
                        with log.open('x') as f:
                            child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,
                                stderr=subprocess.STDOUT,start_new_session=True)
                        actual=process(child.pid)
                        receipt=dict(row_id=rid,pid=child.pid,start_ticks=actual['start_ticks'],gpu=gpu,cwd=str(root),argv=cmd,log=str(log))
                        receipts[run['run_id']].append(receipt)
                        write(Path(plan.PROJECT)/'runs'/run['run_id']/'launch_predict.json',dict(rows=receipts[run['run_id']]))
                        active[key]=dict(item,process=child,gpu=gpu)
                        print('LAUNCH '+json.dumps(dict(run_id=run['run_id'],**receipt)),flush=True)
                for run in plan.RUNS:
                    prefix=run['run_id']+'/'
                    write(Path(plan.PROJECT)/'runs'/run['run_id']/'queue_state.json',dict(kind='predict',
                        active=[k[len(prefix):] for k in active if k.startswith(prefix)],
                        pending=[i['row_id'] for i in pending if i['run']==run],
                        complete=[k[len(prefix):] for k in complete if k.startswith(prefix)],
                        failures=[f for f in failures if f['run_id']==run['run_id']],per_gpu_limit=4,updated_at=time.time()))
                if failures and not active:raise RuntimeError('Prediction failures retained; no automatic retry '+repr(failures))
                if pending or active:time.sleep(5)
            # All32x7 predictions fixed before either scorer can open truth.
            for run in plan.RUNS:
                ev.configure(run['run_id']);ev.validate_predictions()
            write(root/'all_predictions_complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',rows=32,views=7,at=time.time()))
            for run in plan.RUNS:
                subprocess.run([sys.executable,'-m','experiments.cvs_receiver_residual_test_recovery.evaluate',
                    '--mode','score','--run',run['run_id']],cwd=root,check=True)
                base=Path(plan.PROJECT)/'runs'/run['run_id'];score=read(base/'scoring_complete.json')
                if score['result_rows']!=1568 or score['independent_recount']!='VERIFIED':raise ValueError('Score coverage differs')
                write(base/'completion.json',dict(status='ANALYZED',rows=16,views=7,parent_run_id=run['parent_run_id'],
                    disposition=run['disposition'],independent_recount='VERIFIED',target_feedback_forbidden=True))
            write(root/'completion.json',dict(status='ANALYZED',rows=32,views=7,all_predictions_before_truth=True))
        except Exception as e:
            write(root/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--run');p.add_argument('--row');a=p.parse_args()
    worker(a.run,a.row) if a.worker else dispatch()
