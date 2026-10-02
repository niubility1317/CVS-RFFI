"""Unique-owner source -> prediction -> independent scoring, without retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def worker(source_config,predict_config):
    for module, config in [('source',source_config),('predict',predict_config)]:
        subprocess.run([sys.executable,'-u','-m','experiments.cvs_selected_concat.'+module,
            '--config',str(config)],cwd=ROOT,check=True)


def dispatch(spec_path):
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    run, logs=Path(spec['runtime_root']),Path(spec['log_root'])
    run.mkdir(parents=True,exist_ok=False); logs.mkdir(parents=True,exist_ok=False)
    # Exclusive run directory is the remote launch lock; no same-root resubmit.
    write(run/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        run_id=spec['run_id'],launch_owner=spec['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip()))
    sys.path.insert(0,str(ROOT/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import available_gpu,occupancy
    active={}; receipts=[]
    for row in spec['rows']:
        gpu=available_gpu(active)
        if gpu is None: raise RuntimeError('No legal GPU slot; preserve partial run')
        slots=occupancy(active)
        if len(slots[gpu]['pids'])>=2 or slots[gpu]['free_mb']<12000:
            raise RuntimeError('GPU capacity changed')
        rowid=row['row_id']; env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u','-m','experiments.cvs_selected_concat.dispatch',
            '--source-config',row['source_config'],'--predict-config',row['config']]
        logpath=logs/(rowid+'.log')
        with logpath.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[rowid]=dict(process=child,gpu=gpu)
        receipts.append(dict(row_id=rowid,pid=child.pid,gpu=gpu,cwd=str(ROOT),argv=command,log=str(logpath)))
        write(run/'launch.json',dict(run_id=spec['run_id'],rows=receipts))
        print('LAUNCHED '+json.dumps(receipts[-1]),flush=True)
    failures=[]
    for rowid,item in active.items():
        code=item['process'].wait()
        if code: failures.append(dict(row_id=rowid,exit_code=code))
    if failures:
        write(run/'failure.json',dict(status='FAILED',rows=failures,no_retry=True))
        raise RuntimeError('Failed rows; no incomplete prediction scoring')
    # Separate scorer process sees truth only after all4 prediction markers.
    subprocess.run([sys.executable,'-m','comparison_suite.score','--spec',str(spec_path),
        '--stage','p1'],cwd=ROOT,check=True)
    write(run/'completion.json',dict(status='ANALYZED',rows=len(receipts),target_feedback_forbidden=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec');p.add_argument('--source-config');p.add_argument('--predict-config')
    a=p.parse_args()
    if a.spec: dispatch(a.spec)
    else: worker(a.source_config,a.predict_config)
