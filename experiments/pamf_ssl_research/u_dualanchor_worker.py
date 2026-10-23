"""Source dual-anchor smoke, original-parent training, raw/DURA and truth-last."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','row','score'));parser.add_argument('--config',required=True)
    args=parser.parse_args();value=read(args.config)
    if args.action=='smoke':
        from .view_anchor_worker import smoke
        smoke(value,u_dualanchor=True)
    elif args.action=='score':
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
    else:
        config=read(value['train_config'])
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        activate(config);numerical_policy()
        from .u_dualanchor_train import train
        train(config);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run as source_diagnostics
        source_diagnostics(config,Path(config['output_root'])/'source_V_diagnostic.json')
        gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))


if __name__=='__main__':main()
