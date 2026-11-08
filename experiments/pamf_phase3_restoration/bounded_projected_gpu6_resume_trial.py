"""Resume the existing E26 experiment on the authorized shared GPU 6."""
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bounded_projected_resume_trial as base
from . import multiview_trial as trial
from .cost_probe import write
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r04'
MODULE = 'experiments.pamf_phase3_restoration.bounded_projected_gpu6_resume_trial'
REPLACED_RUN = base.RUN
REPLACED_MODULE = base.MODULE
ARM = base.ARM
CAP = 5
PARENT_CONFIGURATION = base.configuration


def configuration(arm):
    cfg = PARENT_CONFIGURATION(arm)
    cfg.update(run_id=RUN,
               resource_policy='GPU6; at most three existing compute PIDs at admission; monitor total cap5.',
               replaces_unstarted_queue=REPLACED_RUN)
    return cfg


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(6,), maximum_processes=4)
            if item['free_memory_MB'] >= 12288]


def ensure_replaced_queue_inactive(root):
    old = root / 'runs' / REPLACED_RUN
    if any(old.glob('submitted-*.json')):
        raise ValueError('Replaced queue already submitted work; do not duplicate')
    launch = read(old / 'launch.json')
    path = Path('/proc') / str(launch['pid']) / 'cmdline'
    if path.exists() and REPLACED_MODULE in path.read_bytes().replace(b'\0', b' ').decode(errors='replace'):
        raise ValueError('Replaced queue owner still live; do not duplicate')


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA-empty controller required')
    ensure_replaced_queue_inactive(args.root)
    out = args.root / 'runs' / RUN
    out.mkdir(); (out / 'logs').mkdir(); row = out / 'rows' / ARM
    row.mkdir(parents=True); (row / 'prediction').mkdir()
    write(out / 'launch.json', dict(pid=os.getpid(), command=sys.argv, CWD=os.getcwd(), code_commit=args.commit))
    for stage in ('train', 'cross_rx', 'cross_both', 'score'):
        admission = None
        if stage != 'score':
            while not admission:
                choices = admitted_gpus()
                if choices: admission = choices[0]
                else:
                    print('WAITING_FOR_ADMITTED_GPU_WITH_HEADROOM', stage, flush=True)
                    time.sleep(30)
        command = trial.command(args, 'score' if stage == 'score' else ('train' if stage == 'train' else 'predict'), ARM,
                                stage if stage in ('cross_rx', 'cross_both') else None)
        with (out / 'logs' / (ARM + '-' + stage + '.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=Path.cwd(), env=dict(os.environ,
                CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(out / ('submitted-' + ARM + '-' + stage + '.json'),
              dict(pid=child.pid, command=command, CWD=os.getcwd(), admission=admission))
        failures = 0; reason = None; observed = None
        while child.poll() is None:
            if admission is not None:
                try:
                    observed = base.original.compute_pids(admission['GPU'], admission['CUDA_ID']); failures = 0
                    if child.pid in observed and len(observed) > CAP: reason = 'TOTAL_COMPUTE_PID_CAP_EXCEEDED'
                except (RuntimeError, ValueError) as error:
                    failures += 1
                    print('GPU_COUNT_OBSERVATION_FAILED', str(error), flush=True)
                    if failures >= 3: reason = 'GPU_COUNT_UNVERIFIABLE'
                if reason:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    write(out / 'resource_stop.json', dict(stage=stage, pid=child.pid, GPU=admission['GPU'],
                        reason=reason, observed_compute_PIDs=sorted(observed) if observed is not None else None,
                        cap=CAP, only_own_child_signaled=True, other_owner_processes_signaled=False))
                    break
            time.sleep(2)
        code = child.wait()
        write(out / ('terminal-' + ARM + '-' + stage + '.json'),
              dict(pid=child.pid, GPU=None if admission is None else admission['GPU'], exit_code=code))
        if code or reason: raise RuntimeError('Preserved stopped or failed stage: ' + stage + ' ' + str(reason))
    if read(row / 'results.json')['status'] != 'ANALYZED':
        raise ValueError('Missing complete original Main result')
    write(out / 'complete.json', dict(status='ARTIFACTS_COMPLETE', all_rows_trained30_and_main_seven_scored=True,
                                     default_DURA_changed=False, goal_complete=False))


def install():
    base.RUN = RUN
    base.MODULE = MODULE
    base.configuration = configuration
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train, trial.controller = configuration, base.factory, base.train, controller


if __name__ == '__main__':
    install()
    trial.main()
