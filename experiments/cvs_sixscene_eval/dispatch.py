"""Unique launch owner; build once, predict all rows, independently score once."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/adv3b02_xuc/code')]
from experiments.cvs_sixscene_eval.common import read,write

def dispatch(path):
    s=read(path);root=Path(s['runtime_root']);root.mkdir(parents=True,exist_ok=False);logs=Path(s['log_root']);logs.mkdir(parents=True,exist_ok=False)
    write(root/'dispatcher.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,launch_owner=s['launch_owner'],commit=(ROOT/'release_commit.txt').read_text().strip()))
    try:
        subprocess.run([sys.executable,'-u','-m','experiments.cvs_sixscene_eval.views','--spec',path],cwd=ROOT,check=True)
        from scripts.dispatch_xuc_full import available_gpu,occupancy
        active={};receipts=[]
        for row in s['rows']:
            gpu=available_gpu(active)
            if gpu is None or occupancy(active)[gpu]['free_mb']<12000:raise RuntimeError('GPU capacity changed')
            command=[sys.executable,'-u','-m','experiments.cvs_sixscene_eval.predict','--config',row['config']]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            logpath=logs/(row['row_id']+'.log')
            with logpath.open('x') as log:child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[row['row_id']]=dict(process=child,gpu=gpu);receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(ROOT),argv=command,log=str(logpath)))
            write(root/('launch_'+str(len(receipts))+'.json'),dict(rows=receipts))
        write(root/'launch.json',dict(rows=receipts))
        failures=[r for r,i in active.items() if i['process'].wait()!=0]
        if failures:raise RuntimeError('Prediction failures: '+str(failures))
        subprocess.run([sys.executable,'-m','experiments.cvs_sixscene_eval.score','--spec',path],cwd=ROOT,check=True)
        write(root/'completion.json',dict(status='ANALYZED',rows=8,views=7,target_feedback_forbidden=True))
    except Exception as e:
        write(root/'failure.json',dict(status='FAILED',error=str(e),no_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();dispatch(a.spec)
