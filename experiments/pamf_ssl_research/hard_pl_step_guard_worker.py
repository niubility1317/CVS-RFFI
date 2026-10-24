"""One original-E20 actual-step smoke,source-only continuation,and automatic frozen tests."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_step_guard import validate
from .hard_pl_early_train import source_parents,train
from .hard_pl_step_guard_objective import StepGuardedNativeAs
from .optimizer_step_guard import installed


def optimizer_probe(model,objective,c):
    import math
    import torch
    from experiments.cric_dura_comparison.f_optimizer import build
    from experiments.cric_dura_comparison import f_optimizer
    from experiments.cric_dura_comparison.f_train import arguments
    from experiments.cvs_phase1_repair.design import cosine_lr
    args=arguments(dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440),'cuda:0')
    lr=float(cosine_lr(float(args.lr),1,args.epochs,float(args.phase1_lr_min)))
    optimizer=build(model,model.parameters(),'CENTER_UNITS',lr=lr,weight_decay=args.weight_decay)
    f_optimizer.clip_physical(optimizer,model.parameters(),5.,clip_impl=torch.nn.utils.clip_grad_norm_)
    optimizer.step()
    state=objective.parameter_guard.state_dict();details=objective.parameter_guard.last_details
    if (state['calls']!=1 or state['anchor_captures']!=1 or state['pending_anchors']
            or state['reference_optimizer_steps']!=1 or optimizer._sinc_unit_steps!=1
            or state['minimum_whole_increment_retention']<.5 or not math.isfinite(state['seconds'])
            or any(not d['actual_parameter_check'] or not d['norm_nonexpansion'] for d in details.values())):
        raise ValueError('Real original E20 protected FP32 optimizer step/source retention rule failed')
    return dict(state,details=details,probe_updates_discarded=True,formal_original_parent_reloaded=True)


def smoke(rt):
    from .hard_pl_worker import smoke as original
    with installed():
        original(rt,validate_config=validate,max_rows=1,objective_cls=StepGuardedNativeAs,
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
        with installed():train(c,validate_config=validate,objective_cls=StepGuardedNativeAs)
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
