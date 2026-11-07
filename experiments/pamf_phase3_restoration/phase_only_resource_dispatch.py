"""Move an unstarted frozen comparison to normal GPU3 capacity, same run/artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .variance_trial import recovery_gpu_choice

RUN = '20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01'
ARM = 'TF_CovariancePhaseOnly'
LEGACY_MODULE = 'experiments.pamf_phase3_restoration.phase_only_eval'
LEGACY_COMMIT = 'd17c50dceaba5aed58c9eeaafc433b0a5fda5a20'
MODULE = 'experiments.pamf_phase3_restoration.phase_only_resource_dispatch'
STAGES = ('cross_rx','cross_both','score')


def read(path):
    return json.loads(path.read_text())


def write_new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,allow_nan=False)


def verify_unstarted_retired(out,legacy_release,proc_root=Path('/proc')):
    if (legacy_release/'release_commit.txt').read_text().strip()!=LEGACY_COMMIT:
        raise ValueError('Unchanged original phase-only prediction code required')
    original=read(out/'launch.json')
    if original['code_commit']!=LEGACY_COMMIT or original['CWD']!=str(legacy_release):
        raise ValueError('Original controller identity required')
    command=proc_root/str(original['pid'])/'cmdline'
    if command.is_file() and command.read_bytes():
        raise ValueError('Original controller is still live; no second owner permitted')
    if list(out.glob('submitted-*.json')) or list((out/'rows'/ARM/'prediction').glob('*/complete.json')):
        raise ValueError('Only the exact unstarted comparison can be handed over')


def stage_command(args,stage):
    command=[sys.executable,'-u','-m',LEGACY_MODULE,'score' if stage=='score' else 'predict',
        '--root',str(args.root),'--commit',LEGACY_COMMIT,
        '--inputs',str(args.legacy_release/'inputs.json'),'--cache',str(args.cache),'--arm',ARM]
    return command+(['--task',stage] if stage!='score' else [])


def legacy_environment(root,legacy_release,cuda):
    fallback=root/'releases/20261031-phase3-source-us-replay-s2026092701-r01-release'
    f0=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
    dura=root/'releases/pamf_stage2_inference_20261021_r01'
    return dict(os.environ,CUDA_VISIBLE_DEVICES=cuda,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                PYTHONPATH=os.pathsep.join(map(str,[legacy_release,fallback,f0,f0/'code',dura,dura/'code'])))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CUDA-empty resource controller required')
    out=args.root/'runs'/RUN
    verify_unstarted_retired(out,args.legacy_release)
    write_new(out/'resource_controller.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),
        dispatch_code_commit=args.commit,prediction_code_commit=LEGACY_COMMIT,
        legacy_release=str(args.legacy_release),GPU=3,admission_total_process_cap=3,
        no_new_G_training=True,removed_resource_dependency='C4 last GPU stage',
        original_launch_preserved=True))
    for stage in STAGES:
        admission=None
        if stage!='score':
            choices=recovery_gpu_choice(allowed_gpus=(3,),maximum_processes=3)
            while not choices:
                time.sleep(5)
                choices=recovery_gpu_choice(allowed_gpus=(3,),maximum_processes=3)
            admission=choices[0]
        command=stage_command(args,stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=args.legacy_release,
                env=legacy_environment(args.root,args.legacy_release,'' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(out/('submitted-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,command=command,
            CWD=str(args.legacy_release),admission=admission,dispatch_code_commit=args.commit,prediction_code_commit=LEGACY_COMMIT))
        code=child.wait()
        write_new(out/('terminal-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,exit_code=code,dispatch_code_commit=args.commit))
        if code:
            raise RuntimeError('Original frozen phase-only stage failed; preserve all artifacts')
    result=read(out/'rows'/ARM/'results.json')
    if any(result.get(k)!=v for k,v in dict(status='ANALYZED',run_id=RUN,arm=ARM,code_commit=LEGACY_COMMIT).items()):
        raise ValueError('Original full Main prediction/truth-last result required')
    write_new(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_Main_two_populations_seven_scored=True,
        prediction_code_commit=LEGACY_COMMIT,dispatch_code_commit=args.commit,new_G_training_updates=0,
        default_DURA_changed=False,goal_complete=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--legacy-release',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual dispatcher Git execution identity required')
    if args.legacy_release!=args.root/'releases'/(RUN+'-release'):
        raise ValueError('Exact original frozen prediction release required')
    controller(args)


if __name__=='__main__':
    main()
