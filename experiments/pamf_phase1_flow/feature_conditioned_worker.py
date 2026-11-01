"""One persistent GPU context for source smoke, training and prediction."""
import argparse
import contextlib
import os
from pathlib import Path

import torch
from . import feature_conditioned_experiment as experiment

MARKER='pamf_dura_direct_train.f_guided worker'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--capacity-marker',required=True);p.add_argument('--resume-checkpoint',type=Path);a=p.parse_args()
    if a.capacity_marker!=MARKER or not os.environ.get('CUDA_VISIBLE_DEVICES') or Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:
        raise ValueError('Assigned GPU, resource-accounting marker and pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'runs'/experiment.RUN
    for mode in ('gpu_smoke','train','predict'):
        with (out/'logs'/(mode+'.log')).open('x',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if mode=='train':experiment.train(a.root,out,a.commit,resume_checkpoint=a.resume_checkpoint)
            else:getattr(experiment,mode)(a.root,out,a.commit)
        if mode=='gpu_smoke' and experiment.read(out/'gpu_smoke.json').get('status')!='PASS':raise ValueError('Full GPU batch smoke required')


if __name__=='__main__':main()
