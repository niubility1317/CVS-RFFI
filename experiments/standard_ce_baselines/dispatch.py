"""One owner; per-row train -> freeze -> clean prediction -> separate scorer."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from experiments.standard_ce_baselines.common import ROOT,read,write,SEEDS,OWNER,RUN,EXTENSION_RUN,EXTENSION_OWNER,EXTENSION_SEEDS,EXTENSION_CONFIG
from experiments.standard_ce_baselines.model import VARIANTS
from experiments.standard_ce_baselines.common import SIG_RUN,SIG_SEEDS,SIG_OWNER,SIG_CONFIG,PROJECT
from experiments.cvs_clean_design.dispatch import choose_gpu


@contextmanager
def launch_guard(spec):
    if spec['run_id']!=SIG_RUN:
        yield
        return
    import fcntl
    with (Path(PROJECT)/'runs/receiver_residual_capacity4.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def matrix(spec):
    if spec['run_id']==SIG_RUN:return ('sig_time',),SIG_SEEDS,SIG_OWNER
    if spec['run_id']==EXTENSION_RUN:return tuple(v for v in VARIANTS if v!='native'),EXTENSION_SEEDS,EXTENSION_OWNER
    if spec['run_id']==RUN:return VARIANTS,SEEDS,OWNER
    raise ValueError('Unregistered run')

def validate_spec(spec):
    variants,seeds,owner=matrix(spec)
    expected={(v,s) for v in variants for s in seeds}
    if len(spec['rows'])!=len(expected) or {(r['variant'],r['model_seed']) for r in spec['rows']}!=expected or spec['launch_owner']!=owner:raise ValueError('Wrong fixed matrix')
    paths=[r[k] for r in spec['rows'] for k in ('source_output','prediction_output','score_output')]
    root=Path(spec['runtime_root']).resolve()
    if len(set(paths))!=len(paths) or any(not Path(p).resolve().is_relative_to(root) for p in paths):raise ValueError('Output collision/path escape')


def summary(spec,state):
    variants,seeds,_=matrix(spec)
    groups={v:[] for v in variants};paired={}
    for row in spec['rows']:
        if state['rows'][row['row_id']]['status']!='SCORED':continue
        values=read(Path(row['score_output'])/'scores.json')['results']
        overall=next(v for v in values if v['dimension']=='overall')
        item=dict(seed=row['model_seed'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],
                  worst_receiver_accuracy=min(v['accuracy'] for v in values if v['dimension']=='receiver'))
        groups[row['variant']].append(item)
    result={}
    for variant,rows in groups.items():
        result[variant]=dict(actual_seed_count=len(rows),expected_seed_count=len(seeds),rows=rows,
            **{metric:dict(mean=statistics.mean(r[metric] for r in rows) if rows else None,
                sample_sd=statistics.stdev(r[metric] for r in rows) if len(rows)>1 else None) for metric in ('accuracy','macro_f1','worst_receiver_accuracy')})
        paired[variant]={}
        for control in (v for v in ('native','residual_fusion') if v in groups):
            reference={r['seed']:r for r in groups[control]}
            diffs=[r['accuracy']-reference[r['seed']]['accuracy'] for r in rows if r['seed'] in reference]
            paired[variant][control]=dict(actual_pairs=len(diffs),mean_accuracy_difference=statistics.mean(diffs) if diffs else None,per_seed_differences=diffs)
    write(Path(spec['runtime_root'])/'summary.json',dict(groups=result,paired=paired,full_matrix=all(len(r)==len(seeds) for r in groups.values()),target_feedback_forbidden=True))
    lines=['# Fixed full-width CE comparison','', 'State: '+state['status'], '', '|Architecture|Completed seeds|Clean accuracy mean|Sample SD|','|---|---:|---:|---:|']
    for variant,value in result.items():
        m=value['accuracy'];lines.append('|'+variant+'|'+str(value['actual_seed_count'])+'/'+str(len(seeds))+'|'+str(m['mean'])+'|'+str(m['sample_sd'])+'|')
    (Path(spec['runtime_root'])/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def finish(spec,state,metadata):
    run=Path(spec['runtime_root'])
    complete=all(r['status']=='SCORED' for r in state['rows'].values())
    state['status']='MERGING_EIGHT_SEEDS' if complete and spec['run_id']==EXTENSION_RUN else ('ANALYZED' if complete else 'FAILED_PARTIAL')
    write(run/'pipeline_state.json',state)
    if state['status']=='MERGING_EIGHT_SEEDS':
        from experiments.standard_ce_baselines.combine_seeds import combine
        try:
            combine(spec);state['status']='ANALYZED'
        except Exception as error:
            state['status']='FAILED_COMBINE';state['combine_error']=str(error)
    write(run/'pipeline_state.json',state)
    metadata['status']=state['status'];write(run/'experiment.json',metadata);summary(spec,state)
    if state['status']=='ANALYZED' and spec['run_id']==SIG_RUN:
        from experiments.standard_ce_baselines.compare_sigtime import compare
        compare(spec)
    return 0 if state['status']=='ANALYZED' else 1


def dispatch(spec_path):
    import fcntl
    spec=read(spec_path);validate_spec(spec)
    run,logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    lock=(run/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    _,_,owner=matrix(spec)
    state=dict(status='RUNNING',pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,run_id=spec['run_id'],launch_owner=owner,
        commit=(ROOT/'release_commit.txt').read_text().strip(),rows={r['row_id']:dict(status='QUEUED',variant=r['variant'],model_seed=r['model_seed']) for r in spec['rows']})
    folder=SIG_CONFIG if spec['run_id']==SIG_RUN else EXTENSION_CONFIG if spec['run_id']==EXTENSION_RUN else 'experiments/standard_ce_baselines/configs'
    metadata=read(ROOT/(folder+'/experiment_spec.json'));metadata['code']['commit']=state['commit'];write(run/'experiment.json',metadata)
    pending=[(r,'source') for r in spec['rows']];active={};technical_failures=0;write(run/'pipeline_state.json',state)
    def event(row,status):
        state['rows'][row['row_id']]['status']=status
        with (run/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),row_id=row['row_id'],status=status))+'\n')
        write(run/'pipeline_state.json',state)
        metadata['status']=state['status']
        for item in metadata['rows']:item['status']=state['rows'][item['row_id']]['status']
        write(run/'experiment.json',metadata)
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];row=job['row'];stage=job['stage']
            if code:
                state['rows'][rid]['exit_code']=code;event(row,'FAILED_'+stage.upper());technical_failures+=1
                if spec['run_id'] in (EXTENSION_RUN,SIG_RUN) and technical_failures>=2:
                    for queued,queued_stage in pending:
                        if queued_stage=='source':event(queued,'NOT_STARTED_SYSTEMIC_TECHNICAL_FAILURE')
                    pending=[job for job in pending if job[1]!='source']
                    state['systemic_technical_stop']=True
                continue
            try:
                if stage=='source':
                    out=Path(row['source_output']);done=read(out/'completion.json')
                    if done.get('status')!='SOURCE_TRAINED' or done.get('epoch')!=200 or done.get('steps')!=10000:raise ValueError('Incomplete E200')
                    write(out/'freeze.json',dict(status='FROZEN',epoch=200,variant=row['variant'],model_seed=row['model_seed'],checkpoint=str(out/'last.pt'),target_access=False,target_score_used=False))
                    event(row,'FROZEN_QUEUED_FOR_CLEAN');pending.insert(0,(row,'predict'))
                else:
                    cfg=dict(row,capsule=spec['capsule'],truth=spec['truth'],output_root=row['score_output'])
                    config=run/(rid+'.score.json');write(config,cfg)
                    with (logs/(rid+'.score.log')).open('x') as output:
                        subprocess.run([sys.executable,'-u','-m','experiments.standard_ce_baselines.score','--config',str(config)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
                    if read(Path(row['score_output'])/'complete.json').get('status')!='SCORED_COMPLETE':raise ValueError('Missing scoring artifact')
                    event(row,'SCORED');summary(spec,state)
            except Exception as error:
                state['rows'][rid]['error']=str(error);event(row,'FAILED_'+stage.upper())
                technical_failures+=1
                if spec['run_id'] in (EXTENSION_RUN,SIG_RUN) and technical_failures>=2:
                    for queued,queued_stage in pending:
                        if queued_stage=='source':event(queued,'NOT_STARTED_SYSTEMIC_TECHNICAL_FAILURE')
                    pending=[job for job in pending if job[1]!='source']
                    state['systemic_technical_stop']=True
        with launch_guard(spec):
            if pending:
                gpu=choose_gpu(active)
                if gpu is not None:
                    row,stage=pending.pop(0);rid=row['row_id']
                    if stage=='source':config=row['source_config']
                    else:
                        config=str(run/(rid+'.predict.json'))
                        write(config,dict(variant=row['variant'],model_seed=row['model_seed'],source_output=row['source_output'],
                            source_contract=row['source_contract'],output_root=row['prediction_output'],capsule=spec['capsule']))
                    command=[sys.executable,'-u','-m','experiments.standard_ce_baselines.'+stage,'--config',config]
                    logpath=logs/(rid+'.'+stage+'.log');log=logpath.open('x')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[rid]=dict(process=child,gpu=gpu,log=log,row=row,stage=stage)
                    state['rows'][rid].update(pid=child.pid,gpu=gpu,argv=command,log=str(logpath),source_output=row['source_output'],started=time.time())
                    event(row,'TRAINING' if stage=='source' else 'PREDICTING_CLEAN');print('LAUNCHED '+json.dumps(state['rows'][rid]),flush=True)
        if pending or active:time.sleep(5)
    return finish(spec,state,metadata)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
