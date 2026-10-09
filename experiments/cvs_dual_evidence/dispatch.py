"""One owner; 16 scratch rows, source freeze, all predictions, truth-last score."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_dual_evidence import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock,process,choose_gpu


def capacity():
    # Account all CUDA owners and pre-CUDA reservations, including foreign runs.
    result={};uuids={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
        g,u,free=[x.strip() for x in line.split(',')];g=int(g)
        result[g]=dict(pids=set(),free_mb=int(free));uuids[u]=g
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        if not line.strip():continue
        pid,u=[x.strip() for x in line.split(',')]
        if u in uuids:result[uuids[u]]['pids'].add(int(pid))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            info=process(int(p.name))
            if info and str(info['gpu']).isdigit() and int(info['gpu']) in result:
                result[int(info['gpu'])]['pids'].add(info['pid'])
        except (PermissionError,UnicodeError):continue
    return result


def worker(kind,rid):
    c=d.config(next(r for r in d.rows() if r['row_id']==rid))
    if d.read(d.BASE/'configs'/(rid+'.json'))!=c:raise ValueError('Worker config differs')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);stable=0
    while stable<3:
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            cap=capacity()[gpu];pids=cap['pids']|{os.getpid()};ready=len(pids)<=4 and cap['free_mb']>=12000
            stable=stable+1 if ready else 0
            d.write(d.BASE/'capacity_leases'/(kind+'-'+rid+'.json'),dict(pid=os.getpid(),gpu=gpu,
                observed_pids=sorted(pids),stable_checks=stable,per_gpu_limit=4,
                status='CAPACITY_READY' if stable==3 else 'WAITING_CAPACITY'))
        if stable<3:time.sleep(1 if ready else 5)
    print('CAPACITY_READY '+json.dumps(dict(pid=os.getpid(),gpu=gpu)),flush=True)
    if kind=='source':
        from experiments.cvs_dual_evidence.source import train
        train(c)
    else:
        from experiments.cvs_dual_evidence.evaluate import predict
        predict(rid)


def queue(configs,kind):
    pending=list(configs);active={};receipts=[];complete=[];failures=[]
    while pending or active:
        for rid,item in list(active.items()):
            rc=item['process'].poll()
            if rc is None:continue
            artifact=d.BASE/rid/('source/completion.json' if kind=='source' else 'prediction/complete.json')
            valid=artifact.is_file() and d.read(artifact)['status']==('SOURCE_TRAINED' if kind=='source' else 'PREDICTIONS_COMPLETE')
            if rc or not valid:failures.append(dict(row_id=rid,exit_code=rc,artifact_valid=valid))
            else:complete.append(rid)
            del active[rid]
        while pending and not failures:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                gpu=choose_gpu(capacity())
                if gpu is None:break
                c=pending.pop(0);rid=c['row_id']
                # train_ filename makes pre-CUDA reservation visible to legacy owners.
                cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_dual_evidence/train_worker.py'),'--kind',kind,'--row',rid]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'),
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                log=Path(d.PROJECT)/'logs'/d.RUN/(kind+'-'+rid+'.log')
                with log.open('x') as f:
                    child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,
                        stderr=subprocess.STDOUT,start_new_session=True)
                actual=process(child.pid)
                if actual is None:raise RuntimeError('Worker absent immediately after launch')
                receipts.append(dict(row_id=rid,kind=kind,pid=child.pid,start_ticks=actual['start_ticks'],
                    gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
                d.write(d.BASE/('launch_'+kind+'.json'),dict(rows=receipts))
                active[rid]=dict(process=child,config=c,gpu=gpu)
                print('LAUNCH '+json.dumps(receipts[-1]),flush=True)
        d.write(d.BASE/'queue_state.json',dict(kind=kind,active=list(active),pending=[r['row_id'] for r in pending],
            completed=complete,failures=failures,per_gpu_limit=4,updated_at=time.time()))
        if failures and not active:raise RuntimeError('Failed rows retained without automatic retry '+repr(failures))
        if pending or active:time.sleep(10)


def freeze(configs):
    from experiments.cvs_dual_evidence.evaluate import source_provenance
    metrics=[];cost={}
    for c in configs:
        source=source_provenance(d,c);done=d.read(source/'completion.json');resolved=d.read(source/'resolved_config.json')
        if resolved['commit']!=(d.ROOT/'release_commit.txt').read_text().strip():raise ValueError('Actual source commit differs')
        r=done['final_source_metrics']
        if r['source_val_count']!=27000 or set(r['source_val_rx_accuracy'])!=set(map(str,[1,3,4,6,8])):raise ValueError('Source V coverage differs')
        metrics.append(dict(arm=c['arm'],model_seed=c['model_seed'],**r));cost[c['arm']]=resolved['total_parameters']
    scores={a:sum(.5*r['source_val_accuracy']+.5*r['source_val_worst_rx'] for r in metrics if r['arm']==a)/4 for a in d.ARMS}
    preferred=max(d.ARMS,key=lambda a:(scores[a],-cost[a],-d.ARMS.index(a)))
    d.write(d.BASE/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=configs,
        source_metrics=metrics,arm_scores=scores,source_preferred_arm=preferred,
        source_preference='mean4(.5V+.5worstRX); exact ties fewer parameters; all fixed controls tested',
        target_access=False,selection='ALL_PREREGISTERED_FIXED_CONTROLS',frozen_at=time.time()))


def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False);(Path(d.PROJECT)/'logs'/d.RUN).mkdir(parents=True,exist_ok=False)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,launch_owner=d.OWNER,
            commit=(d.ROOT/'release_commit.txt').read_text().strip()))
        configs=[d.config(r) for r in d.rows()]
        try:
            for c in configs:d.write(d.BASE/'configs'/(c['row_id']+'.json'),c)
            queue(configs,'source');freeze(configs)
            from experiments.cvs_dual_evidence.evaluate import preflight
            d.write(d.BASE/'checkpoint_preflight.json',preflight())
            queue(configs,'predict')
            subprocess.run([sys.executable,'-m','experiments.cvs_dual_evidence.evaluate','--mode','score'],cwd=d.ROOT,check=True)
            scored=d.read(d.BASE/'scoring_complete.json')
            if scored['status']!='SCORED_COMPLETE' or scored['metric_records']!=1568 or scored['day_metric_records']!=448:raise ValueError('Score coverage differs')
            d.write(d.BASE/'completion.json',dict(status='ANALYZED',rows=16,views=7,independent_recount='VERIFIED',target_feedback_forbidden=True))
        except Exception as e:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':dispatch()
