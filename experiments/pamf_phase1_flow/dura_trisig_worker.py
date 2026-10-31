"""Sequential GPU work; never opens labels."""
import argparse
import contextlib
import os
from pathlib import Path
import sys

import torch
from . import dura_trisig_baseline as experiment


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:
        raise ValueError('Assigned GPU and pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'runs'/experiment.RUN
    for mode in ('predict',):
        with (out/'logs'/(mode+'.log')).open('x',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            getattr(experiment,mode)(a.root,out,a.commit)


if __name__=='__main__':main()
