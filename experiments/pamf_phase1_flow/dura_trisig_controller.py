"""Single-owner Phase1 training, frozen prediction and CPU scoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .dura_trisig_baseline import RUN, read, write


def capacity():
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in gpus.splitlines():
        index,uuid,free=(v.strip() for v in line.split(','));rows[uuid]=dict(index=int(index),free=int(free),pids=[])
    for line in procs.splitlines():
        if not line.strip():continue
        uuid,pid=(v.strip() for v in line.split(','))
        if uuid in rows:rows[uuid]['pids'].append(int(pid))
    return sorted(rows.values(),key=lambda v:(len(v['pids']),-v['free'],v['index']))


def owned(pid, worker):
    if pid==worker:return True
    try:
        status=(Path('/proc')/str(pid)/'status').read_text()
        return int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))==worker
    except (FileNotFoundError,ProcessLookupError,StopIteration):return False


def same_user(pid):
    try:return (Path('/proc')/str(pid)).stat().st_uid==os.getuid()
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    root=a.root;out=root/'runs'/RUN;release=Path.cwd()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (release/'release_commit.txt').read_text().strip()!=a.commit or out.exists():
        raise ValueError('Unique registered output, CPU controller and pinned code required')
    out.mkdir();(out/'logs').mkdir()
    common=['--root',str(root),'--commit',a.commit]
    def run(mode,env):
        cmd=[sys.executable,'-u','-m','experiments.pamf_phase1_flow.dura_trisig_baseline',mode,*common]
        with (out/'logs'/(mode+'.log')).open('xb') as log:
            rc=subprocess.run(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError(mode+' failed; outputs preserved')
    try:
        cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        run('smoke',cpu)
        if read(out/'smoke.json').get('status')!='PASS':raise ValueError('Real source checkpoint smoke required')
        # Preserve the earlier candidate's GPU capacity. This gate consumes
        # only its terminal process status, never its test metrics.
        prior=root/'runs/20261101-phase1-trisigtime-source-rollout-E30-s2026092701-r02/complete.json'
        prior_pid=2176487
        while True:
            proc=Path('/proc')/str(prior_pid)
            try:
                state=proc.joinpath('stat').read_text().split(') ')[1][0]
                cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ')
            except FileNotFoundError:state,cmd=None,b''
            if state in (None,'Z') or b'experiments.pamf_phase1_flow.controller' not in cmd:
                if not prior.is_file():raise RuntimeError('Prior controller vanished without terminal artifact; reconcile prior state')
                break
            print('WAITING_FOR_PRIOR_RUN_TERMINATION',flush=True);time.sleep(20)
        if read(prior).get('status') not in ('ANALYZED','FAILED_TECHNICAL_PRESERVED'):raise ValueError('Prior run terminal state unclear')
        submitted=prior.parent/'submitted.json'
        if submitted.is_file():
            oldpid=read(submitted)['pid']
            while True:
                proc=Path('/proc')/str(oldpid)
                try:
                    state=proc.joinpath('stat').read_text().split(') ')[1][0]
                    cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ')
                except FileNotFoundError:break
                if state=='Z' or b'experiments.pamf_phase1_flow.worker' not in cmd:break
                print('WAITING_FOR_PRIOR_GPU_WORKER',flush=True);time.sleep(20)
        selected=None
        while selected is None:
            with (root/'runs/receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                for gpu in capacity():
                    # At most two compute processes; never share with a foreign user.
                    if gpu['free']>=12000 and len(gpu['pids'])<2 and all(same_user(pid) for pid in gpu['pids']):
                        selected=gpu;break
                if selected is not None:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(selected['index']))
                    cmd=[sys.executable,'-u','-m','experiments.pamf_phase1_flow.dura_trisig_worker',*common]
                    with (out/'logs/worker.log').open('xb') as log:
                        child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    # Keep the shared admission lock until the child actually
                    # owns a CUDA context, so another scheduler cannot observe
                    # this card as idle during model loading.
                    try:
                        write(out/'submitted.json',dict(pid=child.pid,gpu=selected['index'],free_MB=selected['free'],existing_compute_pids=selected['pids']))
                        deadline=time.monotonic()+120
                        while time.monotonic()<deadline and child.poll() is None:
                            current=next(x for x in capacity() if x['index']==selected['index'])
                            if any(owned(pid,child.pid) for pid in current['pids']):break
                            time.sleep(1)
                        else:raise RuntimeError('Worker did not establish a CUDA context')
                    except BaseException:
                        # Worker is a single process. Do not release the
                        # reservation while an unconfirmed child remains live,
                        # including when resource inspection itself fails.
                        if child.poll() is None:
                            child.terminate()
                            try:child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                child.kill();child.wait()
                        raise
            if selected is None:print('WAITING_FOR_FREE_GPU',flush=True);time.sleep(20)
        if child.wait():raise RuntimeError('GPU training or prediction failed; outputs preserved')
        done=read(out/'flow/test_prediction/complete.json')
        if done.get('status')!='PREDICTIONS_COMPLETE' or done.get('TX_truth_reads')!=0:raise ValueError('Complete frozen predictions required before scoring')
        run('score',cpu)
        summary=read(out/'flow/test_summary.json')
        if summary['status']!='ANALYZED':raise ValueError('CPU scoring did not complete')
        write(out/'complete.json',dict(status='ANALYZED',six_scene_gap_at_most_3pp=summary['six_scene_gap_at_most_3pp'],
              clean_damage_at_most_1pp=summary['clean_damage_at_most_1pp'],all_predictions_before_truth=True))
    except BaseException as e:
        if not (out/'complete.json').exists():write(out/'complete.json',dict(status='FAILED_TECHNICAL_PRESERVED',error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
