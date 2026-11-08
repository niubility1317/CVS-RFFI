"""Adopt the exact unstarted mixture run under the authorized GPU-3 cap."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .phase_only_resource_dispatch import legacy_environment, read, write_new
from .bounded_projected_resource_trial import compute_pids
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-MixtureFusion-E30-s2026092701-r01'
ARM = 'TF_MixtureFusion'
SCIENCE_MODULE = 'experiments.pamf_phase3_restoration.mixture_gate_trial'
SCIENCE_COMMIT = 'fa22a4663b3fe6c5b7ff8b756ea151faab19e2a3'
STAGES = ('train', 'cross_rx', 'cross_both', 'score')


def verify_unstarted_retired(out, release, proc_root=Path('/proc')):
    if (release/'release_commit.txt').read_text().strip() != SCIENCE_COMMIT:
        raise ValueError('Exact original scientific release required')
    old = read(out/'launch.json')
    if (old['code_commit'], old['CWD']) != (SCIENCE_COMMIT, str(release)):
        raise ValueError('Original queue identity mismatch')
    cmdline = proc_root/str(old['pid'])/'cmdline'
    if cmdline.is_file() and cmdline.read_bytes():
        raise ValueError('Original queue still live')
    row = out/'rows'/ARM
    if (list(out.glob('submitted-*.json')) or list(out.glob('terminal-*.json'))
            or (row/'training').exists() or (out/'resource_controller.json').exists()
            or any((row/'prediction'/task).exists() for task in ('cross_rx', 'cross_both'))):
        raise ValueError('Only original unstarted output skeleton can be adopted')


def stage_command(args, stage):
    if stage not in STAGES:
        raise ValueError('Only original fixed training and evaluation stages are allowed')
    cmd = [sys.executable, '-u', '-m', SCIENCE_MODULE,
           'predict' if stage in ('cross_rx', 'cross_both') else stage,
           '--root', str(args.root), '--commit', SCIENCE_COMMIT,
           '--inputs', str(args.legacy_release/'inputs.json'), '--cache', str(args.cache), '--arm', ARM]
    return cmd + (['--task', stage] if stage in ('cross_rx', 'cross_both') else [])


def cap_exceeded(child_pid, observed):
    return child_pid in observed and len(observed) > 3


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.legacy_release != args.root/'releases'/(RUN+'-release'):
        raise ValueError('CUDA-empty controller and original release required')
    out = args.root/'runs'/RUN
    verify_unstarted_retired(out, args.legacy_release)
    write_new(out/'resource_controller.json', dict(pid=os.getpid(), command=sys.argv, CWD=os.getcwd(),
        dispatch_code_commit=args.commit, scientific_code_commit=SCIENCE_COMMIT,
        GPU=3, admission_total_process_cap=3, original_launch_preserved=True,
        resource_change='admit at most two existing processes, minimum 12288 MiB; monitor total cap3',
        source_epochs=30, source_updates=7410))
    for stage in STAGES:
        admission = None
        if stage != 'score':
            while not admission:
                choices = [g for g in recovery_gpu_choice(allowed_gpus=(3,), maximum_processes=3)
                           if g['free_memory_MB'] >= 12288]
                if choices:
                    admission = choices[0]
                else:
                    print('WAITING_FOR_GPU3_AUTHORIZED_CAP', stage, flush=True)
                    time.sleep(30)
        cmd = stage_command(args, stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            child = subprocess.Popen(cmd, cwd=args.legacy_release,
                env=legacy_environment(args.root, args.legacy_release, '' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_new(out/('submitted-'+ARM+'-'+stage+'.json'), dict(pid=child.pid, command=cmd,
            CWD=str(args.legacy_release), admission=admission, dispatch_code_commit=args.commit,
            scientific_code_commit=SCIENCE_COMMIT))
        failures = 0
        reason = None
        observed = None
        while child.poll() is None:
            if admission is not None:
                try:
                    observed = compute_pids(admission['GPU'], admission['CUDA_ID'])
                    failures = 0
                    if cap_exceeded(child.pid, observed):
                        reason = 'TOTAL_COMPUTE_PID_CAP_EXCEEDED'
                except (RuntimeError, ValueError) as error:
                    failures += 1
                    print('GPU_COUNT_OBSERVATION_FAILED', str(error), flush=True)
                    if failures >= 3:
                        reason = 'GPU_COUNT_UNVERIFIABLE'
                if reason:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    write_new(out/'resource_stop.json', dict(stage=stage, pid=child.pid, GPU=3,
                        reason=reason, observed_compute_PIDs=sorted(observed) if observed is not None else None,
                        cap=3, only_own_child_signaled=True, other_owner_processes_signaled=False))
                    break
            time.sleep(2)
        code = child.wait()
        write_new(out/('terminal-'+ARM+'-'+stage+'.json'), dict(pid=child.pid, GPU=3 if admission else None,
            exit_code=code, dispatch_code_commit=args.commit, scientific_code_commit=SCIENCE_COMMIT))
        if code or reason:
            raise RuntimeError('Own stage failed or stopped: '+stage+' '+str(reason))
    row = out/'rows'/ARM
    trained, result = read(row/'training/complete.json'), read(row/'results.json')
    if (trained['status'], trained['code_commit'], trained['epoch'], trained['updates']) != ('SOURCE_TRAINED', SCIENCE_COMMIT, 30, 7410):
        raise ValueError('Original fixed final30 training required')
    if (result['status'], result['run_id'], result['arm'], result['code_commit']) != ('ANALYZED', RUN, ARM, SCIENCE_COMMIT):
        raise ValueError('Original complete truth-last result required')
    write_new(out/'complete.json', dict(status='ARTIFACTS_COMPLETE', all_rows_trained30_and_main_seven_scored=True,
        scientific_code_commit=SCIENCE_COMMIT, dispatch_code_commit=args.commit, default_DURA_changed=False, goal_complete=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'legacy-release', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual dispatcher code identity required')
    controller(args)


if __name__ == '__main__':
    main()
