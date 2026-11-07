"""Unchanged source probe in the already authorized GPU6 sharing slot."""
import os
from pathlib import Path
import subprocess
import sys
import time
from . import low_rank_source_probe as method
from .source_pair import read

RUN='20261107-diagnostic-Restore-CoupledLowRank-sourceLU-s2026092701-r03'
MODULE='experiments.pamf_phase3_restoration.low_rank_source_probe_shared'
PREVIOUS='20261107-diagnostic-Restore-CoupledLowRank-sourceLU-s2026092701-r02'
OLD_MODULE='experiments.pamf_phase3_restoration.low_rank_source_probe'


def previous_unstarted(root,proc_root=Path('/proc')):
    old=root/'runs'/PREVIOUS
    if (old/'submitted.json').exists():raise ValueError('Previous probe already started; no duplicate')
    pid=read(old/'launch.json')['pid'];path=proc_root/str(pid)/'cmdline'
    try:command=path.read_bytes().replace(bytes([0]),b' ').decode()
    except FileNotFoundError:return
    if OLD_MODULE+' run ' in command and PREVIOUS in command:
        raise ValueError('Previous queue owner still active; no duplicate')


def resource_choice():
    return method.recovery_gpu_choice(allowed_gpus=(6,),maximum_processes=5)


def controller(args):
    previous_unstarted(args.root)
    out=args.root/'runs'/RUN;out.mkdir()
    method.write(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit,CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while True:
        available=resource_choice()
        if available:break
        print('WAITING_FOR_APPROVED_GPU6_CAP5',flush=True);time.sleep(30)
    admission=available[0]
    command=[sys.executable,'-u','-m',MODULE,'probe','--root',str(args.root),'--inputs',str(args.inputs),'--commit',args.commit]
    with (out/'probe.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        method.write(out/'submitted.json',dict(pid=child.pid,command=command,admission=admission,code_commit=args.commit));rc=child.wait()
    method.write(out/'terminal.json',dict(pid=child.pid,exit_code=rc))
    if rc:raise SystemExit(rc)
    if read(out/'results.json')['status']!='SOURCE_LOW_RANK_PROBE_COMPLETE':raise ValueError('Missing complete source numerical probe')
    method.write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,source_cells=14,target_reads=0))


if __name__=='__main__':
    method.RUN,method.MODULE,method.controller=RUN,MODULE,controller
    method.main()
