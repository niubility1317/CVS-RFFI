"""Reuse unchanged source pairing diagnostic;real six-pass joint gradient first."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_no_as_mixstyle import validate
from .hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective


def train(c,device='cuda:0'):
    from .hard_pl_train import train as original
    return original(c,device=device,validate_config=validate,objective_cls=NoAsMixStyleObjective)


def smoke(rt):
    from .hard_pl_worker import smoke as original
    original(rt,validate_config=validate,max_rows=1,objective_cls=NoAsMixStyleObjective)
    source=read(rt['source_identity_reference'])
    if (source['status']!='PASS' or len(source['rows'])!=3
        or {v['model_seed'] for v in source['rows']}!={2026092701,2026092702,2026092703}
        or not all(v['scientific_rule_pass'] for v in source['rows'])
        or any(source[k] for k in ('model_updates','normalization_updates','EMA_updates','head_fit_updates','U_truth_reads','target_reads'))):
        raise ValueError('Unchanged frozen original-F21 source identity diagnostic required')
    from .hard_pl_train import source_parents
    for row in rt['rows']:
        c=read(row['train_config']);source_parents(c,validate_config=validate)
        previous=next(v for v in rt['source_identity_rows'] if v['model_seed']==row['model_seed'])
        before=read(previous['train_config'])
        if (c['fork_parent_source']!=before['fork_parent_source'] or c['hard_pl_mixstyle']!=before['hard_pl_mixstyle']
            or c['source_contract']!=before['source_contract'] or c['source_cache']!=before['source_cache']):
            raise ValueError('Reused source pairing diagnostic belongs to another parent/recipe/data')
    write(Path(rt['run_root'])/'source_style_identity.json',dict(source,reused_unchanged_parent_pair_diagnostic=True,
        reference_run=rt['source_identity_run'],new_V_model_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','row','score'));p.add_argument('--config',required=True)
    a=p.parse_args();value=read(a.config)
    if a.action=='smoke':smoke(value)
    elif a.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        c=read(value['train_config']);activate(c);numerical_policy();train(c);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(c,Path(c['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
