"""Same default head and frozen DURA; independent scoring after all fourteen views."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('smoke', 'row', 'score'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = read(args.config)
    if args.action == 'smoke':
        from .view_anchor_worker import smoke
        smoke(c, u_cric=True)
    elif args.action == 'score':
        from experiments.cric_dura_comparison.evaluation import score
        score(c)
    else:
        train = read(c['train_config'])
        from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
        activate(train); numerical_policy()
        from .u_cric_train import train as fit
        fit(train)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run as source_diagnostics
        source_diagnostics(train, Path(train['output_root'])/'source_V_diagnostic.json')
        gc.collect(); torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(c['prediction_config']))


if __name__ == '__main__': main()
