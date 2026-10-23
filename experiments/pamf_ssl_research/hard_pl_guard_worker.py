"""Stronger guarded PL;same original E20 parents and immutable final-student readouts."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_guard import validate,METHODS
from .hard_pl_early_train import source_parents,train
from .hard_pl_guard_objective import objective


def smoke(rt):
    from .hard_pl_worker import smoke as original
    original(rt,validate_config=validate,max_rows=2,objective_cls=objective,
        source_parents_fn=source_parents,parent_epoch=20)
    # The parent,threshold,acceptance and style recipe are identical to E20 arms;
    # the prior source screen made no parameter/normalization updates.
    source=read(Path(rt['early_GPU_owner_state']).parent/'source_style_identity.json')
    if (source['status']!='PASS' or source['source_parent_epoch']!=20 or len(source['rows'])!=3
        or not all(r['scientific_rule_pass'] for r in source['rows'])
        or any(source[k] for k in ('model_updates','normalization_updates','EMA_updates','head_fit_updates','U_truth_reads','target_reads'))):
        raise ValueError('Existing matching frozen E20 source identity screen required')
    for row in rt['rows']:
        c=read(row['train_config']);source_parents(c,validate_config=validate)
    write(Path(rt['run_root'])/'source_style_identity.json',dict(source,
        reused_from_same_original_E20=True,gradient_guard_does_not_change_frozen_pair_screen=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','row','score'));p.add_argument('--config',required=True)
    a=p.parse_args();value=read(a.config)
    if a.action=='smoke':smoke(value)
    elif a.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        c=read(value['train_config']);activate(c);numerical_policy()
        train(c,validate_config=validate,objective_cls=objective);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(c,Path(c['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
