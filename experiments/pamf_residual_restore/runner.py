import argparse
from pathlib import Path
import torch
from . import design as d
from .models import NAMES
from .runtime import ROOT,calibrate,load_checkpoint,score
from .training import train
from experiments.pamf_restoration.runner import code_commit,read_json,predict_row
from experiments.pamf_restoration.data import write_json


def main():
    parser=argparse.ArgumentParser(description='Frozen coarse restoration plus source-only residual correction')
    parser.add_argument('mode',choices=('calibrate','worker','confirm'))
    parser.add_argument('--candidate',choices=NAMES)
    for key in ('cache','output','parent-restoration','source-cache','frozen-bundle'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('target-reference','target-truth'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in ('cache','output','parent_restoration','source_cache','frozen_bundle'):
        setattr(args,key,getattr(args,key).resolve())
    if code_commit()!=args.expected_commit:raise ValueError('Pinned residual code differs')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='calibrate':
        calibrate(args.cache,args.output,args.parent_restoration,'cuda:0');return
    if args.candidate is None:raise ValueError('Registered corrector required')
    if args.mode=='worker':
        train(args.candidate,args.cache,args.output,args.parent_restoration,args.source_cache,args.frozen_bundle);return
    decision=read_json(args.output/'source_decision.json')
    if decision['status']!='SOURCE_SELECTION_FROZEN' or args.candidate not in decision['target_candidates']:
        raise ValueError('Frozen source choice required before target prediction')
    if args.target_reference is None or args.target_truth is None:raise ValueError('Fixed target reference/truth required')
    row=args.output/'target_confirmation'/args.candidate;row.mkdir(parents=True)
    (row/'final.pt').symlink_to((args.output/args.candidate/'final.pt').resolve())
    predict_row(args.candidate,args.target_reference,row,'cuda:0',args.frozen_bundle,args.source_cache,checkpoint_loader=load_checkpoint)
    score(args.target_reference,row,target_truth=args.target_truth)
    write_json(row/'complete.json',dict(status='ANALYZED',source_selection_frozen=True,
        original_target_inputs_reused=True,parameter_updates=0,training_TX_labels_read=0))


if __name__=='__main__':main()
