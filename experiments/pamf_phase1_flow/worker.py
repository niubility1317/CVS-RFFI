"""Sequential GPU work; never opens labels."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .experiment import RUN


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Assigned GPU required')
    out=a.root/'runs'/RUN
    for mode in ('train','predict'):
        cmd=[sys.executable,'-u','-m','experiments.pamf_phase1_flow.experiment',mode,'--root',str(a.root),'--commit',a.commit]
        with (out/'logs'/(mode+'.log')).open('xb') as log:
            rc=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError(mode+' failed; no subsequent stage')


if __name__=='__main__':main()
