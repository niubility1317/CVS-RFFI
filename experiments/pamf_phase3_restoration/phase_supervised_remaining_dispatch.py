"""Run only untouched predictions and scoring for the trained phase restorer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .phase_only_resource_dispatch import legacy_environment, read, write_new
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-PhaseSupervisedCovariance-E30-s2026092701-r01'
ARM = 'TF_PhaseSupervisedCovariance'
SCIENCE_MODULE = 'experiments.pamf_phase3_restoration.phase_supervised_covariance_trial'
SCIENCE_COMMIT = '36b13adad5b091bf1f8f4aa7934c403e3b50d8eb'
STAGES = ('cross_rx', 'cross_both', 'score')


def verify_remaining(out):
    row = out/'rows'/ARM
    if read(out/('terminal-'+ARM+'-train.json'))['exit_code'] != 0:
        raise ValueError('Original source training must have succeeded')
    trained = read(row/'training/complete.json')
    if (trained['status'], trained['code_commit'], trained['epoch'], trained['updates']) != ('SOURCE_TRAINED', SCIENCE_COMMIT, 30, 7410):
        raise ValueError('Original fixed final30 checkpoint required')
    if not (row/'training/final.pt').is_file():
        raise ValueError('Original frozen final checkpoint required')
    forbidden = [row/'prediction/cross_rx', row/'prediction/cross_both', row/'score',
                 row/'prediction/complete.json', row/'results.json', out/'complete.json', out/'resource_controller.json']
    for stage in STAGES:
        forbidden += [out/('submitted-'+ARM+'-'+stage+'.json'), out/('terminal-'+ARM+'-'+stage+'.json'), out/'logs'/(ARM+'-'+stage+'.log')]
    if any(path.exists() for path in forbidden):
        raise ValueError('Prediction or scoring already started; preserve existing work')


def stage_command(args, stage):
    if stage not in STAGES:
        raise ValueError('Completed source training cannot be rerun')
    command = [sys.executable, '-u', '-m', SCIENCE_MODULE, 'score' if stage == 'score' else 'predict',
               '--root', str(args.root), '--commit', SCIENCE_COMMIT, '--inputs', str(args.legacy_release/'inputs.json'),
               '--cache', str(args.cache), '--arm', ARM]
    return command if stage == 'score' else command+['--task', stage]


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.legacy_release != args.root/'releases'/(RUN+'-release'):
        raise ValueError('CUDA-empty owner and exact original scientific release required')
    if (args.legacy_release/'release_commit.txt').read_text().strip() != SCIENCE_COMMIT:
        raise ValueError('Original scientific commit required')
    out = args.root/'runs'/RUN
    verify_remaining(out)
    old = read(out/'launch.json')
    try:
        command = (Path('/proc')/str(old['pid'])/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
    except FileNotFoundError:
        command = ''
    if SCIENCE_MODULE+' run ' in command and SCIENCE_COMMIT in command:
        raise ValueError('Old owner still live; no duplicate')
    write_new(out/'resource_controller.json', dict(pid=os.getpid(), command=sys.argv, CWD=os.getcwd(),
        dispatch_code_commit=args.commit, scientific_code_commit=SCIENCE_COMMIT, completed_stages_preserved=['train'],
        remaining_stages=list(STAGES), GPU=3, total_process_cap=3))
    for stage in STAGES:
        admission = None
        if stage != 'score':
            choices = recovery_gpu_choice(allowed_gpus=(3,), maximum_processes=3)
            while not choices:
                print('WAITING_FOR_NORMAL_GPU3_CAP3_PREDICTION', flush=True)
                time.sleep(5)
                choices = recovery_gpu_choice(allowed_gpus=(3,), maximum_processes=3)
            admission = choices[0]
        command = stage_command(args, stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=args.legacy_release,
                env=legacy_environment(args.root,args.legacy_release,'' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_new(out/('submitted-'+ARM+'-'+stage+'.json'), dict(pid=child.pid, command=command,
                CWD=str(args.legacy_release), admission=admission, dispatch_code_commit=args.commit,
                scientific_code_commit=SCIENCE_COMMIT))
            rc = child.wait()
        write_new(out/('terminal-'+ARM+'-'+stage+'.json'), dict(pid=child.pid, exit_code=rc, dispatch_code_commit=args.commit))
        if rc:
            raise RuntimeError('Original remaining stage failed; preserve, no restart')
    result = read(out/'rows'/ARM/'results.json')
    if (result['status'], result['run_id'], result['arm'], result['code_commit']) != ('ANALYZED', RUN, ARM, SCIENCE_COMMIT):
        raise ValueError('Original complete Main result required')
    write_new(out/'complete.json', dict(status='ARTIFACTS_COMPLETE', scientific_code_commit=SCIENCE_COMMIT,
        dispatch_code_commit=args.commit, all_rows_trained30_and_main_seven_scored=True, default_DURA_changed=False, goal_complete=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('root','legacy-release','cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual dispatcher commit required')
    controller(args)


if __name__ == '__main__':
    main()
