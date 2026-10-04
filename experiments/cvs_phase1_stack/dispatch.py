import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_phase1_stack.design import *
from experiments.cvs_phase1_stack.source import require_budget


def validate_freeze(f,stage):
    expected={r['row_id'] for r in rows() if r['stage']==stage}
    if f.get('status')!='SOURCE_FROZEN' or f.get('run_id')!=RUN or f.get('stage')!=stage or f.get('target_access') is not False or len(f['rows'])!=len(expected) or {r['row_id'] for r in f['rows']}!=expected:raise ValueError('Incomplete source stage freeze')
    for r in f['rows']:
        require_budget(r['logged_steps'],r['optimizer_steps'])
        validate(r['config']);m=r['metrics']
        if m['source_val_count']!=27000 or set(m['source_val_rx_accuracy'])!=set(map(str,[1,3,4,6,8])):raise ValueError('Invalid source V coverage')
        vals=[m['source_val_accuracy'],m['source_val_worst_rx'],*m['source_val_rx_accuracy'].values()]
        if not all(math.isfinite(v) and 0<=v<=1 for v in vals) or min(m['source_val_rx_accuracy'].values())!=m['source_val_worst_rx']:raise ValueError('Invalid source metric')
    scores={arm:sum(.5*r['metrics']['source_val_accuracy']+.5*r['metrics']['source_val_worst_rx'] for r in f['rows'] if r['config']['arm']==arm)/4 for arm in STAGES[stage]}
    winner=max(STAGES[stage],key=lambda arm:scores[arm])
    winning=next(r['config']['features'] for r in f['rows'] if r['config']['arm']==winner)
    if f['scores']!=scores or f['winner']!=winner or f['features']!=winning:raise ValueError('Source ranking changed')


def freeze(stage,configs):
    records=[]
    for c in configs:
        done=read(Path(c['output_root'])/'completion.json')
        if done.get('status')!='SOURCE_TRAINED' or done['epoch']!=200 or done['target_access'] or done['target_evaluated'] or done['config']!=c:raise ValueError('Invalid source checkpoint completion')
        require_budget(done['logged_steps'],done['optimizer_steps'])
        records.append(dict(row_id=c['row_id'],config=c,metrics=done['final_source_metrics'],logged_steps=done['logged_steps'],optimizer_steps=done['optimizer_steps']))
    scores={a:sum(.5*r['metrics']['source_val_accuracy']+.5*r['metrics']['source_val_worst_rx'] for r in records if r['config']['arm']==a)/4 for a in STAGES[stage]}
    winner=max(STAGES[stage],key=lambda a:scores[a]);f=dict(status='SOURCE_FROZEN',run_id=RUN,stage=stage,rows=records,scores=scores,winner=winner,
        features=next(c['features'] for c in configs if c['arm']==winner),target_access=False,frozen_at=time.time())
    validate_freeze(f,stage);write(Path(BASE)/(stage+'_source_frozen.json'),f);return f


def queue(jobs,kind,phase):
    sys.path.insert(0,str(ROOT/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import available_gpu,occupancy
    pending=list(jobs);active={};receipts=[];failures=[]
    while pending or active:
        for rid,job in list(active.items()):
            rc=job['process'].poll()
            if rc is not None:
                if rc:failures.append(dict(row_id=rid,exit_code=rc))
                del active[rid]
        while pending and len(active)<8 and not failures:
            gpu=available_gpu(active)
            if gpu is None:break
            caps=occupancy(active)
            if len(caps[gpu]['pids'])>=2 or caps[gpu]['free_mb']<12000:break
            row=pending.pop(0);rid=row['row_id'];cmd=[sys.executable,'-u','-m','experiments.cvs_phase1_stack.'+kind,'--config',row['config']]
            logfile=Path(PROJECT)/'logs'/RUN/(kind+'-'+rid+'.log')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with logfile.open('x') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[rid]=dict(process=p,gpu=gpu)
            receipts.append(dict(row_id=rid,pid=p.pid,gpu=gpu,cwd=str(ROOT),argv=cmd,log=str(logfile)))
            write(Path(BASE)/('launch_'+phase+'.json'),dict(rows=receipts));print('LAUNCH '+json.dumps(receipts[-1]),flush=True)
        write(Path(BASE)/'queue_state.json',dict(phase=phase,kind=kind,active=list(active),pending=[r['row_id'] for r in pending],failures=failures,updated_at=time.time()))
        if failures and not active:raise RuntimeError('Failed rows preserved; no retry '+repr(failures))
        if active or pending:time.sleep(10)


def dispatch():
    root=Path(BASE);root.mkdir(parents=True,exist_ok=False);(Path(PROJECT)/'logs'/RUN).mkdir(parents=True,exist_ok=False)
    write(root/'dispatcher.json',dict(run_id=RUN,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=OWNER,commit=(ROOT/'release_commit.txt').read_text().strip()))
    try:
        parent=read(Path(PROJECT)/'runs'/PARENT_RUN/'source_matrix_frozen.json')
        from experiments.cvs_phase1_overlay.dispatch import validate_freeze as parent_valid
        parent_valid(parent)
        if parent['continuation_arm']!='leo':raise ValueError('Previously observed frozen source parent changed')
        write(root/'parent_source_freeze.json',parent)
        selected=['leo'];prediction_rows=[]
        for stage in STAGES:
            configs=[config(r,selected) for r in rows() if r['stage']==stage]
            jobs=[]
            for c in configs:
                path=root/'configs'/('source-'+c['row_id']+'.json');write(path,c);jobs.append(dict(row_id=c['row_id'],config=str(path)))
            write(root/(stage+'_resolved_matrix.json'),dict(stage=stage,parent_features=selected,rows=configs,target_access=False))
            queue(jobs,'source',stage);f=freeze(stage,configs);selected=f['features']
            for c in configs:
                path=root/'configs'/('predict-'+c['row_id']+'.json')
                pred=dict(row_id=c['row_id'],source_config=str(root/'configs'/('source-'+c['row_id']+'.json')),
                    source_output=c['output_root'],output_root=BASE+'/'+c['row_id']+'/prediction',p1_capsule=CAPSULE)
                write(path,pred)
                prediction_rows.append(dict(row_id=c['row_id'],config=str(path),output_root=pred['output_root'],model_seed=c['model_seed'],method=c['stage']+'-'+c['arm'],stage='phase12'))
        write(root/'all_sources_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=RUN,rows=[r['row_id'] for r in rows()],stages=list(STAGES),target_access=False))
        evaluation=dict(run_id=RUN,runtime_root=BASE,p1_truth=TRUTH,rows=prediction_rows)
        write(root/'evaluation_spec.json',evaluation)
        queue(prediction_rows,'predict','predict')
        subprocess.run([sys.executable,'-m','comparison_suite.score','--spec',str(root/'evaluation_spec.json'),'--stage','p1'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','experiments.cvs_phase1_stack.analyze','--spec',str(root/'evaluation_spec.json')],cwd=ROOT,check=True)
        write(root/'completion.json',dict(status='ANALYZED',rows=136,independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as e:
        write(root/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':dispatch()
