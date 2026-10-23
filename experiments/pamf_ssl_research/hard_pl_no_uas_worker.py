"""Original F21 source parent; U raw-As ablation, frozen dual-readout evaluation."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read
from .hard_pl_no_uas import validate


def train(c,device='cuda:0'):
    from .hard_pl_train import train as original
    return original(c,device=device,validate_config=validate)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','row','score'));parser.add_argument('--config',required=True)
    args=parser.parse_args();value=read(args.config)
    if args.action=='smoke':
        from .hard_pl_worker import smoke
        smoke(value,validate_config=validate,max_rows=1)
    elif args.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        cfg=read(value['train_config']);activate(cfg);numerical_policy();train(cfg);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(cfg,Path(cfg['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
