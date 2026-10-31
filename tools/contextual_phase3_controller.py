"""One isolated Phase3 prediction and independent scorer; no model training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.pamf_dura_direct_train import contextual_recovery as resources
from experiments.pamf_dura_direct_train import runner as base

RUN = '20261101-phase3-contextual-inverse-rx3-2-s2026092701-r01'
ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
G_RUN = '20261101-phase1-contextual-inverse-E30-s2026092701-r01'
REF_RELEASE = '20261031-phase3-fresh-dura-s2026092701-r01-release'


def write(path, obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-commit',required=True)
    args=parser.parse_args()
    release=Path.cwd()
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit:
        raise ValueError('Pinned Phase3 evaluation release required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU-only controller required')
    root=ROOT/'runs'/RUN
    if root.exists():
        raise FileExistsError('Existing Phase3 run must be reconciled, not replaced')
    g=ROOT/'runs'/G_RUN/'contextual_inverse/final.pt'
    c=dict(run_id=RUN,code_commit=args.code_commit,
        reference_runtime=str(ROOT/'releases'/REF_RELEASE/'runtime.json'),
        candidate_checkpoint=str(g),output_root=str(root/'predict'))
    root.mkdir(); (root/'logs').mkdir()
    write(root/'config.json',c)
    entry=release/'tools/contextual_phase3_entry.py'
    command=[sys.executable,'-u',str(entry),'predict','--config',str(root/'config.json'),'--device','cuda:0']
    child=None
    while child is None:
        with (ROOT/'runs/receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=base.capacity()
            commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
            tokens=resources.goal_worker_tokens()+('contextual_phase3_entry.py predict',)
            count,used=resources.active_goal_workers(commands,tokens)
            free=base.eligible_gpus(state,used,3,[4,6,7])
            if count >= 2:
                free=[]
            if free:
                gpu=free[0]
                with (root/'logs/predict.log').open('xb') as log:
                    child=subprocess.Popen(command,cwd=release,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(root/'submitted.json',dict(pid=child.pid,GPU=gpu,command=command,
                    preexisting_compute_PIDs=len(state[gpu]['pids']),free_memory_MB=state[gpu]['free_mb']))
        if child is None:
            print('WAITING_SAFE_GPU_CAPACITY',flush=True)
            time.sleep(10)
    result=child.wait()
    if result != 0:
        write(root/'complete.json',dict(status='FAILED_PREDICTION_PRESERVED',exit_code=result,truth_opened=False))
        return
    metadata=json.loads((root/'predict/complete.json').read_text(encoding='utf-8'))
    if metadata.get('status')!='CONTEXTUAL_PHASE3_PREDICTIONS_COMPLETE' or metadata.get('private_truth_reads')!=0:
        raise ValueError('All frozen predictions must complete before scorer')
    score=[sys.executable,'-u',str(entry),'score','--prediction-root',str(root/'predict'),'--output',str(root/'score')]
    with (root/'logs/score.log').open('xb') as log:
        result=subprocess.run(score,cwd=release,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
    status='ANALYZED' if result==0 and (root/'score/results.json').is_file() else 'FAILED_SCORE_PRESERVED'
    write(root/'complete.json',dict(status=status,score_exit_code=result,full_goal_complete=False,
        target_feedback=False,phase3_registration_complete=False))


if __name__=='__main__':
    main()
