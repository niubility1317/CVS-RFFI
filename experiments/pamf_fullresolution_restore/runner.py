import argparse
import os
from pathlib import Path
import subprocess
import torch
from . import design as d
from .runtime import ROOT
from .training import train
from .evaluation import predict,score
from .controller import PATH_ARGS,command,run
from experiments.pamf_restoration.runner import code_commit
from experiments.pamf_restoration.data import write_json


def main():
    parser=argparse.ArgumentParser(description='Fullresolution TCN prior vsU-Net,scratch20,fullLU,zeroTX/Ftraining')
    parser.add_argument('mode',choices=('run','train','target','score_source','score_target'))
    parser.add_argument('--row',choices=d.ROWS,default='unet_control')
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned fullresolution run/code required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args);return
    if args.mode in ('score_source','score_target'):score(args,'source' if args.mode=='score_source' else 'target');return
    if args.mode=='train':train(args)
    role='target' if args.mode=='target' else 'source';torch.cuda.empty_cache();predict(args,role)
    subprocess.run(command(args,'score_target' if role=='target' else 'score_source',args.row),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    write_json(args.output/args.row/('target_complete.json' if role=='target' else 'complete.json'),dict(status='ANALYZED',candidate=args.row,role=role,all_predictions_before_truth=True,training_TX_F_access=False))


if __name__=='__main__':main()
