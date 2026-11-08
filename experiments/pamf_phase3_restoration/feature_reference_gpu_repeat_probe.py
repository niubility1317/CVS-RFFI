"""Repeat each original/reference-cache path on the same source GPU batches."""
import argparse
from pathlib import Path

from . import feature_reference_gpu_probe as base

RUN = '20261108-diagnostic-Restore-FeatureReferenceGPURepeat-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.feature_reference_gpu_repeat_probe'


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Release identity mismatch')
    base.RUN=RUN
    if args.mode=='run':
        base.controller.RUN,base.controller.MODULE=RUN,MODULE
        base.controller.controller(args)
    else:base.worker(args,repeat_controls=True)


if __name__=='__main__':main()
