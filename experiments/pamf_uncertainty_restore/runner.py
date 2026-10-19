import argparse
import os
from pathlib import Path
import subprocess
import sys
import torch
from . import design as d
from .runtime import ROOT, calibrate
from .training import train
from .evaluation import predict, score, row_folder
from .controller import run, command
from experiments.pamf_restoration.runner import code_commit
from experiments.pamf_restoration.data import write_json


def main():
    parser = argparse.ArgumentParser(description='Source-only uncertainty-aware reception restoration with fixed conditional formal confirmation')
    parser.add_argument('mode', choices=('run', 'calibrate', 'reference', 'train', 'target', 'score_source', 'score_target'))
    parser.add_argument('--kind', choices=(*d.NAMES, 'reference_controls', 'calibration'), default='calibration')
    for key in ('source-cache', 'parent-coarse', 'parent-action', 'parent-residual', 'reader-folder', 'formal-reference', 'formal-f50', 'target-truth', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    for key in ('source_cache', 'parent_coarse', 'parent_action', 'parent_residual', 'reader_folder', 'formal_reference', 'formal_f50', 'target_truth', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.name != d.RUN or code_commit() != args.expected_commit:
        raise ValueError('Pinned run/code differs')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode == 'run':
        run(args); return
    if args.mode == 'calibrate':
        calibrate(args); return
    if args.mode in ('score_source', 'score_target'):
        score(args, 'source' if args.mode == 'score_source' else 'target'); return
    if args.mode == 'train':
        train(args)
    elif args.mode == 'reference':
        if args.kind != 'reference_controls':
            raise ValueError('Fixed reference control group required')
        (args.output/args.kind).mkdir()
    role = 'target' if args.mode == 'target' else 'source'
    if args.kind not in (*d.NAMES, 'reference_controls'):
        raise ValueError('Registered inference kind required')
    if role == 'target' and args.kind not in ('direct_control', 'uncertain_map_prior'):
        raise ValueError('Only preregistered matched confirmation pair allowed')
    torch.cuda.empty_cache()
    predict(args, role)
    subprocess.run(command(args, 'score_target' if role == 'target' else 'score_source', args.kind), cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    write_json(row_folder(args, role)/'complete.json', dict(status='ANALYZED', kind=args.kind, role=role,
        all_predictions_before_truth=True, training_TX_F_access=False))


if __name__ == '__main__':
    main()
