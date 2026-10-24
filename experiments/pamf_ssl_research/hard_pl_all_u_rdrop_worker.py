"""Uncapped raw-U magnitude;all original step safeguards and frozen target tests retained."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_all_u_rdrop import validate
from .hard_pl_early_train import source_parents,train
from .hard_pl_all_u_rdrop_objective import AllURDrop
from .optimizer_step_guard import installed
from .hard_pl_step_no_cap_worker import optimizer_probe as original_probe


def optimizer_probe(model,objective,c):
    state=original_probe(model,objective,c)
    q=objective.state_dict()['U_all_RDrop']
    if (q['calls']!=1 or q['seen']!=256 or q['rejected']<=0 or q['extra_KL_evaluations']!=3
            or q['hard_label_or_CE_on_rejected'] or q['raw_acceptance_changed'] or q['physical_As_KL_weight']
            or not objective.smoke_allrd_extra_requires_grad or float(objective.smoke_allrd_extra)<=0):
        raise ValueError('Actual rejected-U label-free RDrop must contribute before the original raw/actual-step guards')
    return state


def smoke(rt):
    from .hard_pl_worker import smoke as original
    with installed():
        original(rt,validate_config=validate,max_rows=1,objective_cls=AllURDrop,
            source_parents_fn=source_parents,parent_epoch=20,optimizer_probe=optimizer_probe)
    source=read(Path(rt['guard_GPU_owner_state']).parent/'source_style_identity.json')
    if (source['status']!='PASS' or source['source_parent_epoch']!=20 or len(source['rows'])!=3
            or not all(r['scientific_rule_pass'] for r in source['rows'])
            or any(source[k] for k in ('model_updates','normalization_updates','EMA_updates','head_fit_updates','U_truth_reads','target_reads'))):
        raise ValueError('Matching original E20 source style evidence required')
    for row in rt['rows']:source_parents(read(row['train_config']),validate_config=validate)
    write(Path(rt['run_root'])/'source_style_identity.json',dict(source,reused_for_same_E20_same_style=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','row','score'));p.add_argument('--config',required=True)
    a=p.parse_args();value=read(a.config)
    if a.action=='smoke':smoke(value)
    elif a.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        c=read(value['train_config']);activate(c);numerical_policy()
        with installed():train(c,validate_config=validate,objective_cls=AllURDrop)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(c,Path(c['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
