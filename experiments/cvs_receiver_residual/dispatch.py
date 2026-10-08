"""One launch owner: source matrix, freeze, predictions, independent scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_receiver_residual import design as d


def reservations(proc_root=Path('/proc')):
    result={}
    for p in proc_root.iterdir():
        if not p.name.isdigit(): continue
        try:
            if (p/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X','x'): continue
            env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
            gpu=env.get('CUDA_VISIBLE_DEVICES','')
            if gpu.isdigit(): result.setdefault(int(gpu),set()).add(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeError): pass
    return result


def capacity():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    caps={};uuid={}
    for line in output.strip().splitlines():
        i,u,free=[v.strip() for v in line.split(',')];i=int(i)
        caps[i]=dict(pids=set(),free_mb=int(free));uuid[u]=i
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    for line in apps.strip().splitlines():
        if not line.strip(): continue
        pid,u=[v.strip() for v in line.split(',')]
        if u in uuid: caps[uuid[u]]['pids'].add(int(pid))
    for gpu,pids in reservations().items():
        if gpu in caps: caps[gpu]['pids'].update(pids)
    return caps


def validate_matrix(spec):
    expected=[dict(row_id=d.row_id(c['arm'],c['model_seed']),config=c) for c in d.rows()]
    if (spec.get('run_id')!=d.RUN or spec.get('rows')!=expected
        or spec.get('launch_owner')!='codex/root/receiver-residual-20261008'
        or spec.get('max_active')!=4 or spec.get('per_gpu_limit')!=2):
        raise ValueError('Registered launch matrix differs')
    return expected


def queue(spec,kind):
    base=Path(d.PROJECT)/'runs'/d.RUN;logs=Path(d.PROJECT)/'logs'/d.RUN
    pending=list(spec['rows']);active={};receipts=[];failures=[]
    while pending or active:
        for rid,item in list(active.items()):
            rc=item['process'].poll()
            if rc is None: continue
            row=item['row'];source=Path(row['config']['output_root'])
            artifact=source/'completion.json' if kind=='source' else source.parent/'prediction/complete.json'
            valid=artifact.exists() and d.read(artifact)['status']==('SOURCE_TRAINED' if kind=='source' else 'PREDICTIONS_COMPLETE')
            if rc or not valid: failures.append(dict(row_id=rid,kind=kind,exit_code=rc,artifact_valid=valid))
            del active[rid]
        while pending and len(active)<4 and not failures:
            caps=capacity()
            idle=[gpu for gpu,c in caps.items() if not c['pids'] and c['free_mb']>=12000
                and gpu not in [j['gpu'] for j in active.values()]]
            if not idle: break
            gpu=min(idle);row=pending.pop(0);rid=row['row_id']
            cfg=d.PROJECT+'/releases/'+d.RELEASE+'/experiments/cvs_receiver_residual/configs/'+rid+'.json'
            command=[sys.executable,'-u','-m','experiments.cvs_receiver_residual.worker',
                '--kind',kind,'--config',cfg,'--row',rid]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'))
            logpath=logs/(kind+'-'+rid+'.log')
            with logpath.open('x') as log:
                child=subprocess.Popen(command,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[rid]=dict(process=child,gpu=gpu,row=row)
            receipts.append(dict(row_id=rid,kind=kind,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(logpath)))
            d.write(base/('launch_'+kind+'.json'),dict(rows=receipts))
            print('LAUNCH '+json.dumps(receipts[-1]),flush=True)
        d.write(base/'queue_state.json',dict(kind=kind,active=list(active),pending=[r['row_id'] for r in pending],
            failures=failures,updated_at=time.time(),max_active=4,per_gpu_limit=2,policy='start only on wholly idle GPU'))
        if failures and not active: raise RuntimeError('Failed rows retained; no automatic retry '+repr(failures))
        if active or pending: time.sleep(10)


def freeze():
    from experiments.cvs_receiver_residual.evaluate import source_provenance
    metrics=[];cost={}
    for c in d.rows():
        source,done,*_=source_provenance(c)
        metrics.append(dict(arm=c['arm'],model_seed=c['model_seed'],**done['final_source_metrics']))
        cost[c['arm']]=d.read(source/'resolved_config.json')['total_parameters']
    scores={a:sum(.5*r['source_val_accuracy']+.5*r['source_val_worst_rx'] for r in metrics if r['arm']==a)/4 for a in d.ARMS}
    for r in metrics:
        if r['source_val_count']!=27000 or set(r['source_val_rx_accuracy'])!=set(map(str,d.SOURCE_RXS)):
            raise ValueError('Source metrics coverage differs')
    best=max(d.ARMS,key=lambda a:(scores[a],-cost[a],-d.ARMS.index(a)))
    d.write(Path(d.PROJECT)/'runs'/d.RUN/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',
        run_id=d.RUN,rows=d.rows(),source_metrics=metrics,arm_scores=scores,source_preferred_arm=best,
        source_preference='mean4(.5V+.5worstRX), exact ties fewer parameters; all fixed controls tested',
        target_access=False,selection='ALL_PREREGISTERED_FIXED_CONTROLS',frozen_at=time.time()))


def dispatch(path):
    spec=d.read(path);validate_matrix(spec)
    base=Path(d.PROJECT)/'runs'/d.RUN;logs=Path(d.PROJECT)/'logs'/d.RUN
    base.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    d.write(base/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        run_id=d.RUN,launch_owner=spec['launch_owner'],commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    try:
        queue(spec,'source');freeze();queue(spec,'predict')
        subprocess.run([sys.executable,'-m','experiments.cvs_receiver_residual.evaluate','--mode','score'],cwd=d.ROOT,check=True)
        if d.read(base/'scoring_complete.json')['status']!='SCORED_COMPLETE': raise ValueError('Scoring incomplete')
        d.write(base/'completion.json',dict(status='ANALYZED',rows=16,views=7,
            independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as e:
        d.write(base/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);dispatch(p.parse_args().spec)
