"""Original E20+30 reliability-filtered source As,defaulthead and both readouts."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_reliable_as import validate
from .hard_pl_early_train import source_parents,train
from .hard_pl_reliable_as_objective import GuardedReliableAs


def smoke(rt):
    from .hard_pl_worker import smoke as original
    original(rt,validate_config=validate,max_rows=1,objective_cls=GuardedReliableAs,source_parents_fn=source_parents,parent_epoch=20)
    source=read(Path(rt['guard_GPU_owner_state']).parent/'source_style_identity.json')
    if (source['status']!='PASS' or source['source_parent_epoch']!=20 or len(source['rows'])!=3
        or not all(r['scientific_rule_pass'] for r in source['rows'])
        or any(source[k] for k in ('model_updates','normalization_updates','EMA_updates','head_fit_updates','U_truth_reads','target_reads'))):
        raise ValueError('Matching unchanged originalE20 no-update raw style identity record required')
    for row in rt['rows']:source_parents(read(row['train_config']),validate_config=validate)
    write(Path(rt['run_root'])/'source_style_identity.json',dict(source,reused_for_same_E20_same_style=True,
        actual_reliable_As_and_raw_fallback_diagnosed_in_smoke=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','row','score'));p.add_argument('--config',required=True)
    a=p.parse_args();value=read(a.config)
    if a.action=='smoke':smoke(value)
    elif a.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        c=read(value['train_config']);activate(c);numerical_policy()
        train(c,validate_config=validate,objective_cls=GuardedReliableAs);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(c,Path(c['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
