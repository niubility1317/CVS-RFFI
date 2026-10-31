"""One fixed source-flow training and mandatory frozen evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.pamf_dura_direct_train import contextual_recovery as resources
from experiments.pamf_dura_direct_train import runner as base
from experiments.pamf_identity_flow import fullg10_rollout_train as candidate
from experiments.pamf_identity_flow.resource_policy import is_restoration_gpu_worker

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
REFERENCE='20261031-phase3-fresh-dura-s2026092701-r01-release/runtime.json'


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--code-commit',required=True);args=parser.parse_args()
    commit=args.code_commit;release=Path.cwd();runtime=ROOT/'releases'/REFERENCE;out=ROOT/'runs'/candidate.RUN
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!=commit:raise ValueError('Pinned source-flow release required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or out.exists():raise ValueError('CPU-only controller and new immutable run required')
    previous_profile=ROOT/'runs/20261101-diagnostic-identity-rollout-phase3-profile-s2026092701-r01/complete.json'
    partial_since=None
    while True:
        if previous_profile.is_file():
            try:
                previous=base.read_json(previous_profile)
                break
            except json.JSONDecodeError:
                if partial_since is None:partial_since=time.monotonic()
                if time.monotonic()-partial_since>60:raise
        print('WAITING_FOR_PRECEDING_ROLLOUT_PROFILE',flush=True);time.sleep(20)
    if previous.get('status')!='ANALYZED':raise ValueError('Preceding rollout profile did not complete successfully')
    out.mkdir();(out/'logs').mkdir();write(out/'plan.json',candidate.config())
    common=['--root',str(ROOT),'--reference-runtime',str(runtime),'--code-commit',commit]
    gpucommon=['--root',str(ROOT),'--commit',commit]
    def run(command,name,env):
        with (out/'logs'/(name+'.log')).open('xb') as log:
            return subprocess.run(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
    cpu=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    phase_config=dict(run_id=candidate.RUN,code_commit=commit,candidate_code_commit=commit,
        reference_runtime=str(runtime),candidate_checkpoint=str(out/candidate.ARM/'final.pt'),output_root=str(out/'phase3_predict'))
    write(out/'phase3_config.json',phase_config)
    try:
        smoke=[sys.executable,'-u','-m','experiments.pamf_identity_flow.fullg10_rollout_train','smoke',*common]
        if run(smoke,'smoke',cpu)!=0 or base.read_json(out/'smoke.json').get('status')!='PASS':
            raise RuntimeError('Actual G10/F0 checkpoint smoke failed')
        # A single GPU worker owns the entire fixed training and prediction pipeline.
        child=None
        while child is None:
            with (ROOT/'runs/receiver_residual_capacity4.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                state=base.capacity();commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
                tokens=resources.goal_worker_tokens()+('contextual_phase3_entry.py predict','identity_flow_worker.py --root','identity_rollout_worker.py --root','identity_fullg10_rollout_worker.py --root','experiments.pamf_identity_flow.phase3_profile --root','experiments.pamf_identity_flow.rollout_phase3_profile --root','experiments.pamf_identity_flow.trajectory_probe --root','experiments.pamf_identity_flow.forcing_probe --root')
                count,used=resources.active_goal_workers(commands,tokens)
                external=[line for line in commands if is_restoration_gpu_worker(line)]
                other_count,other_used=resources.active_goal_workers(external,('experiments.pamf_phase3_restoration.','/pamf_phase3_restoration/'))
                count+=other_count;used.update(other_used)
                free=base.eligible_gpus(state,used,3,[4,6,7])
                if count>=2:free=[]
                if free:
                    gpu=free[0]
                    command=[sys.executable,'-u',str(release/'tools/identity_fullg10_rollout_worker.py'),'--root',str(ROOT),'--commit',commit]
                    with (out/'logs/worker.log').open('xb') as log:
                        child=subprocess.Popen(command,cwd=release,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write(out/'submitted.json',dict(pid=child.pid,GPU=gpu,free_memory_MB=state[gpu]['free_mb'],existing_compute_PIDs=len(state[gpu]['pids'])))
            if child is None:print('WAITING_SAFE_GPU_CAPACITY',flush=True);time.sleep(10)
        if child.wait()!=0:raise RuntimeError('Source training or fixed prediction failed; logs and artifacts preserved')
        # Do not open any truth until both seven-view populations and all four Phase3 predictions complete.
        row=out/candidate.ARM
        for role in ('source','target'):
            done=base.read_json(row/(role+'_prediction')/'complete.json')
            if done.get('status')!='PREDICTIONS_COMPLETE' or done.get('TX_truth_reads')!=0:raise ValueError('All historical predictions required')
        phase=base.read_json(out/'phase3_predict/complete.json')
        if phase.get('status')!='IDENTITY_FULLG10_ROLLOUT_PHASE3_PREDICTIONS_COMPLETE' or phase.get('private_truth_reads')!=0:
            raise ValueError('All Phase3 predictions required')
        for role in ('source','target'):
            command=[sys.executable,'-u','-m','experiments.pamf_identity_flow.fullg10_rollout_historical_eval','score',*gpucommon,'--role',role]
            if run(command,'score-'+role,cpu)!=0:raise RuntimeError('Historical CPU truth scorer failed')
        command=[sys.executable,'-u','-m','experiments.pamf_identity_flow.fullg10_rollout_phase3_eval','score','--prediction-root',str(out/'phase3_predict'),'--output',str(out/'phase3_score')]
        if run(command,'score-phase3',cpu)!=0:raise RuntimeError('Phase3 CPU truth scorer failed')
        if not all((row/(role+'_summary.json')).is_file() for role in ('source','target')) or not (out/'phase3_score/results.json').is_file():
            raise ValueError('All independent score artifacts required')
        write(out/'complete.json',dict(status='ANALYZED',all_predictions_before_any_truth=True,target_feedback=False,
            final_phase3_complete=False,profile_available=(row/'matched_profile.json').is_file()))
    except BaseException as error:
        if not (out/'complete.json').exists():
            write(out/'complete.json',dict(status='FAILED_TECHNICAL_PRESERVED',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
