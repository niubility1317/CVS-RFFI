"""Bounded source-only CORE90 negative-control replay, never a formal run.

Usage: python code/scripts/accept_core90_v2_source.py --wisig-pkl PATH --output DIR
No checkpoint loading, no target access/evaluation, and no validation fitting.
The stage schedule is an explicitly synthetic time jump over real source batches.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.data import build_source, audit_indices_v2
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.game_tracking.source_audit import SourceAuditor, AuditConfig
from cvsrffi.game_tracking.budget import ComputeBudget, BudgetConfig
from cvsrffi.game_tracking.controller import GameControllerV2, ControllerConfig
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.step_context import prepare_context, Core90Objective
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights, _update_ema_model
from cvsrffi.schedule import (build_stage_state, build_aug_base_cfg, make_augmentor,
                              configure_augmentor_for_epoch, configure_mixstyle_for_epoch)
from cvsrffi.tensors import set_seed


def acceptance_args(*, synthetic=False, wisig_pkl=None, device='cpu', probe_config=None):
    argv = ['--output_dir','unused-functional-replay','--device',device,'--batch_size','18',
            '--num_workers','0','--game_audit_interval','1',
            '--game_audit_samples_per_capture','2','--game_skip_final_eval',
            '--game_evidence_version','2']
    if probe_config is None:argv.extend(['--game_probe_steps','2'])
    else:argv.extend(['--game_config_json',str(probe_config)])
    if synthetic:
        argv.append('--game_synthetic')
    else:
        if wisig_pkl is None or not Path(wisig_pkl).is_file():
            raise FileNotFoundError('local WiSig source file required; no download attempted')
        argv.extend(['--wisig_pkl',str(wisig_pkl)])
    return parse_args(argv)


def first_difference(left,right,path='state'):
    """Return the first exact mismatch with quantitative tensor diagnostics."""
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype!=right.dtype or left.shape!=right.shape:
            return dict(path=path,reason='tensor_type_dtype_or_shape')
        if torch.equal(left,right):
            return None
        delta=(left.detach().cpu().double()-right.detach().cpu().double()).abs()
        return dict(path=path,reason='tensor_values',max_abs_difference=float(delta.max()),
                    unequal_count=int(torch.count_nonzero(left!=right)))
    if isinstance(left,np.ndarray):
        return None if isinstance(right,np.ndarray) and np.array_equal(left,right) else dict(path=path,reason='numpy_values')
    if isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():
            return dict(path=path,reason='dictionary_keys')
        for key in left:
            found=first_difference(left[key],right[key],f'{path}.{key}')
            if found:return found
        return None
    if isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):
            return dict(path=path,reason='sequence_type_or_length')
        for i,(a,b) in enumerate(zip(left,right)):
            found=first_difference(a,b,f'{path}[{i}]')
            if found:return found
        return None
    if isinstance(left,float) and np.isnan(left):
        return None if isinstance(right,float) and np.isnan(right) else dict(path=path,reason='nan_mismatch')
    return None if left==right else dict(path=path,reason='scalar_value',left=left,right=right)


def _snapshot(model,ema,optimizer,scaler,proto,solver,gen,ctx,epoch,step):
    # Include grads, flags, RNG and transient context caches omitted by standard checkpoints.
    rng=RNGState.capture()
    return deepcopy(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),prototype=vars(proto),solver=solver.state_dict(),
        gradients={n:p.grad for n,p in model.named_parameters()},
        module_training={n:m.training for n,m in model.named_modules()},
        rng=vars(rng),satellite_generator=gen.get_state(),context=vars(ctx),
        data_position=dict(epoch=epoch,step=step,sample_ids=list(ctx.sample_ids))))


def replay(args,source,epochs,*,audited=False,controlled=False,mask_schedule=None):
    """Scratch, paired source replay, one actual training update per epoch label.

    ``mask_schedule`` overrides legitimate pseudo masks for a declared functional
    boundary fixture only; it never accesses hidden U labels. No model is selected.
    """
    set_seed(args.seed)
    device=torch.device(args.device)
    model=runtime.build_model(args,len(source.domains),device).train()
    ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    head_ids={id(p) for p in model.adv_head.parameters()}
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in head_ids],lr=args.lr),
        dict(params=list(model.adv_head.parameters()),lr=args.lr*args.game_head_lr_ratio)],
        lr=args.lr,weight_decay=args.weight_decay)
    scaler=runtime.make_grad_scaler(device,args.amp)
    solver=GameSolver(model,optimizer,'simultaneous',max_grad_norm=args.game_max_grad_norm,scaler=scaler)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains),momentum=args.proto_momentum,
        margin=args.proto_margin,domain_align_weight=args.proto_domain_align_weight,
        push_weight=args.proto_push_weight,min_count=args.proto_min_count)
    proto._lazy_init(160,device,torch.float32)
    objective=Core90Objective(model,args,proto)
    gen=torch.Generator(device=device).manual_seed(args.seed+991)
    aug_cfg=build_aug_base_cfg(args) if args.use_aug else None
    augmentor=make_augmentor(aug_cfg) if aug_cfg else None
    if augmentor is not None and hasattr(augmentor,'to'):augmentor=augmentor.to(device)
    budget=ComputeBudget(BudgetConfig(max_head_steps=0,max_field_evaluations=0,
        max_correction_fraction=0.,max_audit_seconds=3600.))
    controller=GameControllerV2(ControllerConfig(confirmation_windows=1,catchup_steps=1))
    coordinator=None
    if audited:
        from cvsrffi.game_tracking.runtime_control import V2Coordinator
        indexes=audit_indices_v2(source,args.game_audit_samples_per_capture,args.seed)
        auditor=SourceAuditor(AuditConfig(steps=args.game_probe_steps))
        coordinator=V2Coordinator(model,source,args,proto,auditor,budget,indexes)
    snapshots=[];previous_weights=None
    for step,epoch in enumerate(epochs):
        model.train()
        configure_mixstyle_for_epoch(model,args,epoch)
        if augmentor is not None:
            configure_augmentor_for_epoch(augmentor,aug_cfg,min(epoch,args.label_epochs),args)
        weights=_loss_weights(args,build_stage_state(epoch,args))
        if previous_weights is not None and weights!=previous_weights:
            solver.reset_history('objective_weights_change')
        previous_weights=dict(weights)
        if epoch in (41,91,args.label_epochs+1,args.sat_cons_start_epoch):
            solver.reset_history('scheduled_problem_change')
        loader=source.loader('train',args.batch_size,seed=args.seed+epoch,shuffle=True,workers=0,drop_last=True)
        batch=next(iter(loader))
        ubatch=None
        if epoch>args.label_epochs and args.use_unlabeled:
            ubatch=next(iter(source.unlabeled_epoch_loader(args.batch_size,
                epoch_index=max(0,epoch-args.label_epochs-1),steps=len(loader),seed=args.seed,workers=0)))
        ctx=prepare_context(batch,ubatch,model,ema,args,epoch,1,weights,gen,augmentor)
        if mask_schedule is not None:
            if ctx.base_mask is None:raise ValueError('mask fixture requires active U context')
            ctx.base_mask.fill_(bool(mask_schedule[step]))
            ctx.strong_mask=ctx.base_mask.clone()
        decision=dict(action='NORMAL',reason='ordinary_baseline');metrics={}
        if audited:
            metrics=coordinator.observe(ctx,step=step,version=step,epoch=epoch,augmentor=augmentor)
            if controlled:
                decision=controller.decide(metrics,step=step,encoder_version=step,budget=budget)
                if decision['action']!='NORMAL':
                    raise AssertionError('zero-action budget failed: '+str(decision))
        runtime.update_pseudo_problem(solver,ctx,args.num_classes,args.game_optimistic_pseudo_change,1)
        result=solver.step(lambda:objective(ctx))
        if not result.accepted:raise AssertionError('finite main update was not accepted')
        _update_ema_model(ema,model,args.ema_decay)
        proto.update(ctx.origin_features,ctx.y,ctx.domain)
        snapshots.append(dict(training=_snapshot(model,ema,optimizer,scaler,proto,solver,gen,ctx,epoch,step),
            audit_schema=metrics.get('schema'),decision=decision,loss=result.loss))
    return snapshots


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wisig-pkl',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--epochs',default='1,40,41,79,80,130,131')
    parser.add_argument('--probe-config',help='Explicit frozen source probe configuration JSON.')
    cli=parser.parse_args(argv)
    output=Path(cli.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing existing nonempty acceptance output')
    args=acceptance_args(wisig_pkl=cli.wisig_pkl,device=cli.device,probe_config=cli.probe_config)
    torch.set_num_threads(cli.threads)
    source=build_source(args)
    epochs=[int(e) for e in cli.epochs.split(',')]
    output.mkdir(parents=True,exist_ok=True)
    baseline=replay(args,source,epochs)
    report=dict(kind='source_functional_stage_jump_replay_not_formal_training',
                initialization='scratch_only',epochs=epochs,target_access=False,validation_fit=False,
                comparison='exact_all_tensors_rng_integer_masks_no_tolerance',comparisons=[],
                probe_config=cli.probe_config,probe_steps=args.game_probe_steps,probe_lr=args.game_probe_lr,
                audit_samples_per_capture=args.game_audit_samples_per_capture,
                scope_note='This replay budget is independent of the separately selected source probe budget.')
    for name,controlled in [('audit_only',False),('no_action_controller',True)]:
        candidate=replay(args,source,epochs,audited=True,controlled=controlled)
        for i,(left,right) in enumerate(zip(baseline,candidate)):
            difference=first_difference(left['training'],right['training'])
            report['comparisons'].append(dict(branch=name,step=i,epoch=epochs[i],first_difference=difference))
            if difference:
                torch.save(dict(left=left,right=right),output/f'first_divergence_{name}_step{i}.pth')
                break
    report['status']='FAILED' if any(r['first_difference'] for r in report['comparisons']) else 'VERIFIED'
    runtime.json_write(output/'acceptance.json',report)
    return int(report['status']!='VERIFIED')


if __name__=='__main__':
    raise SystemExit(main())
