"""Wait for E30 scoring, then run one matched synthetic Phase3 timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.pamf_dura_direct_train import contextual_recovery as resources
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_identity_flow import rollout_train as candidate
from experiments.pamf_identity_flow.resource_policy import is_restoration_gpu_worker

RUN='20261101-diagnostic-identity-rollout-phase3-profile-s2026092701-r02'
ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
CANDIDATE_COMMIT='54e74d7f7f8c01e5755909f3a417a2279cc4c28b'


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--profile-commit',required=True);args=parser.parse_args()
    release=Path.cwd();out=ROOT/'runs'/RUN;parent=ROOT/'runs'/candidate.RUN
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=args.profile_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Pinned CPU-only profile controller required')
    if out.exists():raise FileExistsError('Independent profile output already exists')
    out.mkdir();(out/'logs').mkdir()
    try:
        partial_since=None
        while True:
            if (parent/'complete.json').exists():
                try:
                    parent_done=base.read_json(parent/'complete.json')
                    break
                except json.JSONDecodeError:
                    if partial_since is None:partial_since=time.monotonic()
                    if time.monotonic()-partial_since>60:raise
            print('WAITING_FOR_FIXED_E30_COMPLETE',flush=True);time.sleep(20)
        if parent_done.get('status')!='ANALYZED':raise RuntimeError('Parent did not complete fixed evaluations')
        runtime=ROOT/'releases/20261031-phase3-fresh-dura-s2026092701-r01-release/runtime.json'
        command=[sys.executable,'-u','-m','experiments.pamf_identity_flow.rollout_phase3_profile','--root',str(ROOT),
            '--reference-runtime',str(runtime),'--candidate-commit',CANDIDATE_COMMIT,'--profile-commit',args.profile_commit,'--output',str(out/'profile')]
        child=None
        while child is None:
            with (ROOT/'runs/receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                state=base.capacity();commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
                tokens=resources.goal_worker_tokens()+('contextual_phase3_entry.py predict','identity_flow_worker.py --root','identity_rollout_worker.py --root','experiments.pamf_identity_flow.phase3_profile --root','experiments.pamf_identity_flow.rollout_phase3_profile --root','experiments.pamf_identity_flow.trajectory_probe --root','experiments.pamf_identity_flow.forcing_probe --root')
                count,used=resources.active_goal_workers(commands,tokens)
                external=[line for line in commands if is_restoration_gpu_worker(line)]
                other_count,other_used=resources.active_goal_workers(external,('experiments.pamf_phase3_restoration.','/pamf_phase3_restoration/'))
                count+=other_count;used.update(other_used)
                free=[gpu for gpu in base.eligible_gpus(state,used,3,[4,6,7]) if not state[gpu]['pids']]
                if count:free=[]
                if free:
                    gpu=free[0]
                    with (out/'logs/profile.log').open('xb') as log:
                        child=subprocess.Popen(command,cwd=release,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write(out/'submitted.json',dict(pid=child.pid,GPU=gpu,free_memory_MB=state[gpu]['free_mb'],existing_compute_PIDs=len(state[gpu]['pids'])))
            if child is None:print('WAITING_SAFE_GPU_CAPACITY',flush=True);time.sleep(10)
        code=child.wait();result=out/'profile/complete.json'
        if code or not result.is_file() or base.read_json(result).get('status')!='MATCHED_PHASE3_PROFILE_COMPLETE':
            raise RuntimeError('Matched Phase3 profile failed; partial output preserved')
        write(out/'complete.json',dict(status='ANALYZED',candidate_run=candidate.RUN,profile_ref=str(result),real_IQ_reads=0,truth_reads=0))
    except BaseException as error:
        if not (out/'complete.json').exists():write(out/'complete.json',dict(status='FAILED_PRESERVED',error=repr(error)))
        raise


if __name__=='__main__':main()
