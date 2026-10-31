"""One source-only frozen trajectory diagnostic on safely admitted GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.pamf_dura_direct_train import contextual_recovery as resources
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_identity_flow import trajectory_probe as probe

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--code-commit',required=True);args=parser.parse_args()
    release=Path.cwd();logs=ROOT/'logs'/probe.RUN
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Pinned CPU controller required')
    if logs.exists() or (ROOT/'runs'/probe.RUN).exists():raise FileExistsError('Unique diagnostic paths required')
    logs.mkdir();runtime=ROOT/'releases/20261031-phase3-fresh-dura-s2026092701-r01-release/runtime.json'
    cmd=[sys.executable,'-u','-m','experiments.pamf_identity_flow.trajectory_probe','--root',str(ROOT),'--reference-runtime',str(runtime),'--code-commit',args.code_commit]
    child=None
    while child is None:
        with (ROOT/'runs/receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=base.capacity();commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
            tokens=resources.goal_worker_tokens()+('contextual_phase3_entry.py predict','identity_flow_worker.py --root','experiments.pamf_identity_flow.phase3_profile --root','experiments.pamf_identity_flow.trajectory_probe --root')
            count,used=resources.active_goal_workers(commands,tokens);free=base.eligible_gpus(state,used,3,[4,6,7])
            if count>=2:free=[]
            if free:
                gpu=free[0]
                with (logs/'probe.log').open('xb') as log:
                    child=subprocess.Popen(cmd,cwd=release,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with (logs/'submitted.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=child.pid,GPU=gpu,free_memory_MB=state[gpu]['free_mb']),f)
        if child is None:print('WAITING_SAFE_GPU_CAPACITY',flush=True);time.sleep(10)
    code=child.wait();path=ROOT/'runs'/probe.RUN/'complete.json'
    if code or not path.exists() or base.read_json(path).get('status')!='SOURCE_PATH_DIAGNOSTIC_COMPLETE':
        raise RuntimeError('Source trajectory diagnostic failed; outputs preserved')
    print('SOURCE_PATH_DIAGNOSTIC_VERIFIED_BY_CONTROLLER',flush=True)


if __name__=='__main__':main()
