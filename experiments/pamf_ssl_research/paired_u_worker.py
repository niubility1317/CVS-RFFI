"""Paired-U continuation; frozen source diagnostics then both seven-view readouts."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('smoke','row','score'))
    p.add_argument('--config',required=True)
    a=p.parse_args()
    c=read(a.config)
    if a.action=='smoke':
        from .view_anchor_worker import smoke
        smoke(c,paired=True)
    elif a.action=='score':
        from experiments.cric_dura_comparison.evaluation import score
        score(c)
    else:
        train=read(c['train_config'])
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        activate(train);numerical_policy()
        from .paired_u_train import train as fit
        fit(train)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run as source_diagnostics
        source_diagnostics(train,Path(train['output_root'])/'source_V_diagnostic.json')
        gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(c['prediction_config']))


if __name__=='__main__':main()
