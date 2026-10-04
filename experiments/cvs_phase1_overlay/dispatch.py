"""Exclusive queue: complete source matrix -> freeze -> predictions -> scorer."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_phase1_overlay.contract import *


def validate_matrix(spec):
    if spec.get('run_id')!=RUN or spec.get('runtime_root')!=PROJECT+'/runs/'+RUN or spec.get('log_root')!=PROJECT+'/logs/'+RUN or spec.get('p1_truth')!=TRUTH:
        raise ValueError('Unregistered runtime')
    rows=spec['rows']
    expected={(a,s) for a in ARMS for s in SEEDS}
    if len(rows)!=16 or {(r['arm'],r['model_seed']) for r in rows}!=expected:raise ValueError('Incomplete or duplicate matrix')
    for r in rows:
        rid=r['arm']+'-s'+str(r['model_seed'])
        base=PROJECT+'/runs/'+RUN+'/'+rid
        cfg=PROJECT+'/releases/'+RELEASE+'/experiments/cvs_phase1_overlay/configs/'
        if (r['row_id']!=rid or r['method']!=r['arm'] or r['stage']!='phase12' or r['source_output']!=base+'/source' or r['output_root']!=base+'/prediction' or r['source_config']!=cfg+'source-'+rid+'.json' or r['config']!=cfg+'predict-'+rid+'.json'):
            raise ValueError('Row contract mismatch')


def ranking(rows):
    scores={a:sum(.5*r['source_val_accuracy']+.5*r['source_val_worst_rx'] for r in rows if r['arm']==a)/4 for a in ARMS}
    return scores,max(ARMS,key=lambda a:scores[a])


def validate_freeze(f):
    rows=f['rows']
    if f.get('status')!='SOURCE_MATRIX_FROZEN' or f.get('run_id')!=RUN or f.get('target_access') is not False or len(rows)!=16 or {(r['arm'],r['model_seed']) for r in rows}!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError('Incomplete/contaminated source freeze')
    for r in rows:
        if r['source_output']!=PROJECT+'/runs/'+RUN+'/'+r['arm']+'-s'+str(r['model_seed'])+'/source':raise ValueError('Freeze row path mismatch')
        rates=r['source_val_rx_accuracy']
        if r['source_val_count']!=27000 or set(rates)!=set(map(str,[1,3,4,6,8])) or any(not math.isfinite(v) or not 0<=v<=1 for v in [r['source_val_accuracy'],*rates.values()]) or r['source_val_worst_rx']!=min(rates.values()):raise ValueError('Invalid source metrics')
    scores,winner=ranking(rows)
    if f['arm_scores']!=scores or f['continuation_arm']!=winner:raise ValueError('Source ranking mismatch')


def freeze_sources(spec):
    rows=[]
    for r in spec['rows']:
        p=Path(r['source_output']);done=read(p/'completion.json');cfg=read(p/'resolved_config.json');initial=read(p/'initialization.json')
        validate_config(cfg)
        if done['status']!='SOURCE_TRAINED' or done['epoch']!=200 or done['steps']!=10000 or done['target_access'] or done['target_evaluated'] or initial['status']!='SCRATCH' or initial['checkpoint_sources'] or initial['ancestors'] or initial['target_access'] or initial['target_contact']:raise ValueError('Source completion/provenance mismatch')
        rows.append(dict(arm=r['arm'],model_seed=r['model_seed'],source_output=r['source_output'],**done['final_source_metrics']))
    scores,winner=ranking(rows)
    f=dict(status='SOURCE_MATRIX_FROZEN',run_id=RUN,rows=rows,arm_scores=scores,continuation_arm=winner,target_access=False,
        rule='mean4(0.5V+0.5worstRX), exact tie CE/LEO/MixStyle/both order; all16 fixed comparisons tested',frozen_at=time.time())
    validate_freeze(f);write(Path(spec['runtime_root'])/'source_matrix_frozen.json',f)
    return f


def queue(spec,stage):
    sys.path.insert(0,str(ROOT/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import available_gpu,occupancy
    run=Path(spec['runtime_root']);logs=Path(spec['log_root'])
    active={};pending=list(spec['rows']);receipts=[];failures=[]
    while pending or active:
        for rid,item in list(active.items()):
            code=item['process'].poll()
            if code is not None:
                if code:failures.append(dict(row_id=rid,stage=stage,exit_code=code))
                del active[rid]
        while pending and len(active)<8 and not failures:
            gpu=available_gpu(active)
            if gpu is None:break
            slots=occupancy(active)
            if len(slots[gpu]['pids'])>=2 or slots[gpu]['free_mb']<12000:break
            row=pending.pop(0);rid=row['row_id']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            command=[sys.executable,'-u','-m','experiments.cvs_phase1_overlay.'+stage,'--config',row['source_config' if stage=='source' else 'config']]
            logpath=logs/(stage+'-'+rid+'.log')
            with logpath.open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[rid]=dict(process=child,gpu=gpu)
            receipts.append(dict(row_id=rid,stage=stage,pid=child.pid,gpu=gpu,cwd=str(ROOT),argv=command,log=str(logpath)))
            write(run/('launch_'+stage+'.json'),dict(run_id=RUN,rows=receipts))
            print('LAUNCHED '+json.dumps(receipts[-1]),flush=True)
        write(run/'queue_state.json',dict(stage=stage,active=list(active),pending=[r['row_id'] for r in pending],failures=failures,updated_at=time.time()))
        if failures and not active:raise RuntimeError('Failed rows, no automatic retry: '+repr(failures))
        if pending or active:time.sleep(5)


def dispatch(path):
    spec=read(path);validate_matrix(spec)
    run=Path(spec['runtime_root']);logs=Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    write(run/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,run_id=RUN,launch_owner=spec['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip()))
    try:
        queue(spec,'source');freeze_sources(spec);queue(spec,'predict')
        subprocess.run([sys.executable,'-m','comparison_suite.score','--spec',str(path),'--stage','p1'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','experiments.cvs_phase1_overlay.analyze','--spec',str(path)],cwd=ROOT,check=True)
        write(run/'completion.json',dict(status='ANALYZED',rows=16,target_feedback_forbidden=True,independent_recount='VERIFIED'))
    except Exception as e:
        write(run/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();dispatch(a.spec)
