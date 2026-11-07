"""Run the unchanged registered packet filter on normal GPU5 capacity."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .phase_only_resource_dispatch import legacy_environment,read,write_new
from .variance_trial import recovery_gpu_choice

RUN='20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01'
ARM='TF_PacketInverseFilter'
LEGACY_MODULE='experiments.pamf_phase3_restoration.packet_inverse_filter_trial'
LEGACY_COMMIT='92d0dcbe0f25927d5a1cf4d98669ded45a410a5b'
MODULE='experiments.pamf_phase3_restoration.packet_filter_resource_dispatch'
STAGES=('train','cross_rx','cross_both','score')


def stage_command(args,stage):
    if stage not in STAGES:
        raise ValueError('Only the registered single30/Main7 stages allowed')
    command=[sys.executable,'-u','-m',LEGACY_MODULE,stage if stage in ('train','score') else 'predict',
        '--root',str(args.root),'--commit',LEGACY_COMMIT,'--inputs',str(args.legacy_release/'inputs.json'),
        '--cache',str(args.cache),'--arm',ARM]
    return command+(['--task',stage] if stage in ('cross_rx','cross_both') else [])


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CUDA-empty resource controller required')
    if (args.legacy_release/'release_commit.txt').read_text().strip()!=LEGACY_COMMIT:
        raise ValueError('Original registered scientific code required')
    out=args.root/'runs'/RUN
    # Publication retires only the exact unstarted owner before this exclusive claim.
    out.mkdir()
    (out/'logs').mkdir()
    row=out/'rows'/ARM
    (row/'prediction').mkdir(parents=True)
    write_new(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit,
        scientific_code_commit=LEGACY_COMMIT,legacy_release=str(args.legacy_release),
        GPU=5,admission_total_process_cap=3,removed_resource_dependency='ProjectedPrior last GPU stage',
        fixed_scratch_epochs=30,fixed_scratch_updates=7410))
    for stage in STAGES:
        admission=None
        if stage!='score':
            choices=recovery_gpu_choice(allowed_gpus=(5,),maximum_processes=3)
            while not choices:
                time.sleep(5)
                choices=recovery_gpu_choice(allowed_gpus=(5,),maximum_processes=3)
            admission=choices[0]
        command=stage_command(args,stage)
        with (out/'logs'/(ARM+'-'+stage+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=args.legacy_release,
                env=legacy_environment(args.root,args.legacy_release,'' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(out/('submitted-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,command=command,CWD=str(args.legacy_release),
            admission=admission,dispatch_code_commit=args.commit,scientific_code_commit=LEGACY_COMMIT))
        code=child.wait()
        write_new(out/('terminal-'+ARM+'-'+stage+'.json'),dict(pid=child.pid,exit_code=code,dispatch_code_commit=args.commit))
        if code:
            raise RuntimeError('Original packet filter stage failed; preserve all artifacts')
    trained=read(row/'training/complete.json');result=read(row/'results.json')
    expected=dict(status='SOURCE_TRAINED',code_commit=LEGACY_COMMIT,arm=ARM,epoch=30,updates=7410)
    if any(trained.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete original scratch30/7410 source training required')
    expected=dict(status='ANALYZED',run_id=RUN,arm=ARM,code_commit=LEGACY_COMMIT)
    if any(result.get(k)!=v for k,v in expected.items()):
        raise ValueError('Original full Main and truth-last result required')
    write_new(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_rows_trained30_and_main_seven_scored=True,
        scientific_code_commit=LEGACY_COMMIT,dispatch_code_commit=args.commit,default_DURA_changed=False,goal_complete=False))


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
        raise ValueError('Exact original packet filter release required')
    controller(args)


if __name__=='__main__':main()
