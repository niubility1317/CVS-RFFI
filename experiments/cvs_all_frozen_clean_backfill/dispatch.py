"""One owner predicts the registered clean rows, then invokes a separate scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.cvs_clean_design.dispatch import occupancy
from experiments.cvs_all_frozen_clean_backfill.contracts import read,validate_spec

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def choose_gpu(active):
    values=occupancy(active)
    # One evaluator per GPU; no training is launched here.
    eligible=[(-v['free_mb'],g) for g,v in values.items() if not v['pids'] and v['free_mb']>=12000]
    return min(eligible)[1] if eligible else None


def dispatch(path):
    spec=validate_spec(read(path));run,logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='CLEAN_PREDICTING',pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,run_id=spec['run_id'],launch_owner=spec['launch_owner'],
        commit=(ROOT/'release_commit.txt').read_text().strip(),truth_read=False,view='clean',
        rows={r['row_id']:dict(status='PREDICTIONS_COMPLETE' if r['reuse'] else 'QUEUED',variant=r['variant'],model_seed=r['model_seed'],
            output_root=r['output_root'],reuse=r['reuse']) for r in spec['rows']})
    write(run/'pipeline_state.json',state);pending=[r for r in spec['rows'] if not r['reuse']];active={}
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];entry=state['rows'][rid]
            entry.update(exit_code=code,status='FAILED' if code else 'PREDICTIONS_COMPLETE',finished=time.time())
            if not code:
                marker=Path(entry['output_root'])/'clean_complete.json'
                if not marker.exists() or read(marker)['status']!='PREDICTIONS_COMPLETE':entry['status']='FAILED'
            write(run/'pipeline_state.json',state)
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                row=pending.pop(0);rid=row['row_id'];logpath=logs/(rid+'.log');log=logpath.open('x')
                command=[sys.executable,'-u','-m','experiments.cvs_all_frozen_clean_backfill.predict','--config',row['config']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=log)
                state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,log=str(logpath),output_root=row['output_root'],started=time.time())
                write(run/'pipeline_state.json',state)
        if pending or active:time.sleep(2)
    if any(r['status']!='PREDICTIONS_COMPLETE' for r in state['rows'].values()):
        state['status']='FAILED_PARTIAL_TRUTH_CLOSED';write(run/'pipeline_state.json',state);return 1
    state['status']='ALL_CLEAN_PREDICTIONS_COMPLETE';write(run/'pipeline_state.json',state)
    # The independent scorer is the first process allowed to connect target truth.
    subprocess.run([sys.executable,'-m','experiments.cvs_all_frozen_clean_backfill.score','--spec',str(path)],cwd=ROOT,check=True)
    marker=read(run/'scoring_clean_complete.json')
    if marker['status']!='SCORED_COMPLETE':raise ValueError('Scoring did not complete')
    state['status']='ANALYZED';state['independent_scoring_complete']=True;write(run/'pipeline_state.json',state)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
