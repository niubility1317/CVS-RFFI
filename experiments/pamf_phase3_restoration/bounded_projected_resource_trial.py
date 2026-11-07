"""Same bounded candidate with conservative shared-GPU admission."""
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bounded_projected_prior_trial as method
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .cost_probe import write
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r02'
MODULE='experiments.pamf_phase3_restoration.bounded_projected_resource_trial'
ARM=method.ARM


def configuration(arm):
    cfg=method.configuration(arm)
    cfg.update(run_id=RUN,resource_policy='GPU2/3/4; at most one existing compute PID before admission, monitor total cap3.',
        replaces_resource_stopped_run=method.RUN,scientific_recipe_unchanged=True)
    return cfg


def factory(arm):
    configuration(arm)
    return method.factory(arm)


def train(args,out):
    training.train(args,out,build=factory,config_for=configuration,training_readout=method.training_readout)


def admitted_gpus():
    # Leave one slot for an unrelated worker establishing its GPU context.
    return recovery_gpu_choice(allowed_gpus=(2,3,4),maximum_processes=2)


def compute_pids(gpu,uuid,runner=subprocess.run):
    result=runner(['nvidia-smi','-i',str(gpu),'--query-compute-apps=gpu_uuid,pid',
                   '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Unable to verify shared GPU process count: '+result.stderr[-400:])
    pids=set()
    for line in result.stdout.splitlines():
        if not line.strip():continue
        identity,pid=[value.strip() for value in line.split(',')]
        if identity!=uuid:raise ValueError('GPU-scoped process query returned another UUID')
        pids.add(int(pid))
    return pids


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();row=out/'rows'/ARM;row.mkdir(parents=True);(row/'prediction').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    for stage in ('train','cross_rx','cross_both','score'):
        admission=None
        if stage!='score':
            while not admission:
                choices=admitted_gpus()
                if choices:admission=choices[0]
                else:print('WAITING_FOR_ADMITTED_GPU_WITH_HEADROOM',stage,flush=True);time.sleep(30)
        command=trial.command(args,'score' if stage=='score' else ('train' if stage=='train' else 'predict'),ARM,
                              stage if stage in ('cross_rx','cross_both') else None)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,command=command,CWD=os.getcwd(),admission=admission))
        failures=0;reason=None;observed=None
        while child.poll() is None:
            if admission is not None:
                try:
                    observed=compute_pids(admission['GPU'],admission['CUDA_ID']);failures=0
                    if child.pid in observed and len(observed)>3:reason='TOTAL_COMPUTE_PID_CAP_EXCEEDED'
                except (RuntimeError,ValueError) as error:
                    failures+=1;print('GPU_COUNT_OBSERVATION_FAILED',str(error),flush=True)
                    if failures>=3:reason='GPU_COUNT_UNVERIFIABLE'
                if reason:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    write(out/'resource_stop.json',dict(stage=stage,pid=child.pid,GPU=admission['GPU'],reason=reason,
                        observed_compute_PIDs=sorted(observed) if observed is not None else None,cap=3,
                        only_own_child_signaled=True,other_owner_processes_signaled=False))
                    break
            time.sleep(2)
        code=child.wait()
        write(out/('terminal-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,GPU=None if admission is None else admission['GPU'],exit_code=code))
        if code or reason:raise RuntimeError('Preserved stopped or failed stage: '+stage+' '+str(reason))
    if read(row/'results.json')['status']!='ANALYZED':raise ValueError('Missing complete original Main result')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_rows_trained30_and_main_seven_scored=True,
        default_DURA_changed=False,goal_complete=False))


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller=configuration,factory,train,controller


if __name__=='__main__':install();trial.main()
