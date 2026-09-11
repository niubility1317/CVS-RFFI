"""Fixed source-only probe candidates on one scratch E131 functional context.

This is a bounded diagnostic, not E131 trained weights or target confirmation.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.accept_core90_v2_source import acceptance_args
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.data import build_source, audit_indices_v2
from cvsrffi.game_tracking.runtime_audit_v2 import (_lag, _gradient_reference,
                                                  _freeze_actual_u, reference_coverage)
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import (build_stage_state, build_aug_base_cfg, make_augmentor,
                              configure_augmentor_for_epoch, configure_mixstyle_for_epoch)
from cvsrffi.tensors import set_seed


CANDIDATES=((2e-4,40),(2e-4,120),(2e-3,40),(2e-3,120),(.02,40))
SELECTION_RULE='complete_reliable_control_ready_then_minimum_same_target_fit_endpoint_CE; old_lr_reference_excluded'


def choose_candidate(candidates):
    eligible=[row for row in candidates if not row.get('instability_reference',False) and
        row['lag'].get('quality_pass') and row['lag'].get('control_ready') and
        row['lag'].get('status') in ('RELIABLE_HIGH_GAP','RELIABLE_LOW_GAP') and
        row['lag'].get('budget',{}).get('actual_steps')==row['lag'].get('budget',{}).get('requested_steps') and
        isinstance(row['lag'].get('fit_after'),(int,float))]
    return min(eligible,key=lambda row:row['lag']['fit_after']) if eligible else None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wisig-pkl',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--replay-selected-from',help='Re-evaluate the previously selected fixed candidate only; do not reselect.')
    cli=parser.parse_args(argv)
    output=Path(cli.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('refusing nonempty development output')
    args=acceptance_args(wisig_pkl=cli.wisig_pkl,device=cli.device)
    args.game_audit_samples_per_capture=4
    torch.set_num_threads(cli.threads);set_seed(args.seed)
    print('[source-probe] loading local source builder',flush=True)
    source=build_source(args);device=torch.device(args.device)
    indexes=audit_indices_v2(source,4,args.seed)
    coverage=reference_coverage(source,args,indexes['lag_reference'])
    if not coverage['valid']:raise ValueError('source contract coverage invalid: '+str(coverage))
    output.mkdir(parents=True,exist_ok=True)
    runtime.json_write(output/'source_coverage.json',coverage)
    prior=None;candidates=CANDIDATES
    if cli.replay_selected_from:
        prior=json.loads(Path(cli.replay_selected_from).read_text(encoding='utf-8'))
        previous=next(r for r in prior['candidates'] if r['name']==prior['selected'])
        candidates=((previous['lr'],previous['steps']),)
    runtime.json_write(output/'selection_rule.json',dict(rule=SELECTION_RULE,candidates=candidates,
                       source_only=True,checkpoint_loaded=False,target_access=False))
    print('[source-probe] verified source counts '+str(source.info['counts']),flush=True)
    model=runtime.build_model(args,len(source.domains),device).train();epoch=131
    configure_mixstyle_for_epoch(model,args,epoch)
    ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains),momentum=args.proto_momentum,
        margin=args.proto_margin,domain_align_weight=args.proto_domain_align_weight,
        push_weight=args.proto_push_weight,min_count=args.proto_min_count)
    proto._lazy_init(160,device,torch.float32)
    aug_cfg=build_aug_base_cfg(args) if args.use_aug else None
    augmentor=make_augmentor(aug_cfg) if aug_cfg else None
    if augmentor is not None:
        if hasattr(augmentor,'to'):augmentor=augmentor.to(device)
        configure_augmentor_for_epoch(augmentor,aug_cfg,min(epoch,args.label_epochs),args)
    loader=source.loader('train',args.batch_size,seed=args.seed+epoch,shuffle=True,workers=0,drop_last=True)
    batch=next(iter(loader))
    ubatch=next(iter(source.unlabeled_epoch_loader(args.batch_size,epoch_index=0,steps=len(loader),seed=args.seed,workers=0)))
    ctx=prepare_context(batch,ubatch,model,ema,args,epoch,1,_loss_weights(args,build_stage_state(epoch,args)),
                         torch.Generator(device=device).manual_seed(args.seed+991),augmentor)
    training_rng=RNGState.capture();seed=1729
    with isolated_rng(seed):frozen_u,preview_count=_freeze_actual_u(model,args,proto,ctx,training_rng)
    report=dict(kind='bounded_source_probe_development_scratch_stage_context_not_trained_E131',
        initialization='scratch_only',epoch_context=epoch,target_access=False,validation_fit=False,
        selection_rule=SELECTION_RULE,coverage=coverage,preview_forwards=preview_count,
        candidates=[],selected=None,gradient=None)
    heads={}
    for lr,steps in candidates:
        name=f'lr{lr:g}_steps{steps}'
        local=deepcopy(args);local.game_probe_lr=lr;local.game_probe_steps=steps
        started=time.perf_counter()
        with isolated_rng(seed):
            result=_lag(model,source,indexes['lag_reference'],local,ctx,epoch,seed,augmentor,frozen_u)
        row=dict(name=name,lr=lr,steps=steps,instability_reference=lr==.02,
                 lag=result.metrics,elapsed_seconds=time.perf_counter()-started)
        report['candidates'].append(row);heads[name]=result.recovered_head
        runtime.json_write(output/(name+'.json'),row)
        print('[source-probe] '+name+' '+row['lag']['status']+' fit='+str(row['lag']['fit_after']),flush=True)
    chosen=choose_candidate(report['candidates'])
    baselines=[r['lag']['fit_before'] for r in report['candidates'] if r['lag']['fit_before'] is not None]
    report['baseline_ce_range']=max(baselines)-min(baselines) if baselines else None
    if baselines and report['baseline_ce_range']>1e-6:
        raise AssertionError('candidate objectives did not share the same original-head CE')
    if chosen is not None:
        report['selected']=chosen['name']
        if prior is not None:
            if chosen['name']!=prior['selected'] or abs(chosen['lag']['fit_after']-previous['lag']['fit_after'])>1e-6:
                raise AssertionError('fixed candidate replay changed; selection cannot be silently replaced')
            report['prior_selection_artifact']=str(Path(cli.replay_selected_from).resolve())
        runtime.json_write(output/'selected_probe_config.json',dict(game_evidence_version=2,
                           game_probe_lr=chosen['lr'],game_probe_steps=chosen['steps']))
        runtime.json_write(output/'selected_probe_metadata.json',dict(selection_rule=SELECTION_RULE,
            evidence='development.json',selected=chosen['name'],source_only=True,target_used_for_selection=False,
            scope='frozen_source_development_configuration_not_default_parser_change'))
        with isolated_rng(seed+1):
            report['gradient']=_gradient_reference(model,heads[chosen['name']],source,indexes['gradient_reference'],
                args,ctx,epoch,seed+1,augmentor,frozen_u,proto)
        print('[source-probe] selected '+chosen['name']+' gradient='+report['gradient']['status'],flush=True)
    report['status']='VERIFIED' if chosen is not None and report['gradient']['valid'] else 'UNKNOWN'
    report['scope_note']='Engineering selection among fixed source candidates only; no population-optimal or performance claim.'
    runtime.json_write(output/'development.json',report)
    return 0


if __name__=='__main__':raise SystemExit(main())
