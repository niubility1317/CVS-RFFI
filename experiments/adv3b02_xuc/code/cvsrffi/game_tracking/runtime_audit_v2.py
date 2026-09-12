"""Owned-copy source audits against fixed training-mode CORE90 objectives.

The bounded reference is a training-pool observation, not an independent
validation split or a claim of population/global-head optimality.
"""
from collections import Counter
from copy import deepcopy
from itertools import product
import math
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .audit_evidence import make_evidence_v2, unavailable
from .data import labeled_source_records
from .gradient_audit import gradient_diagnostics, compare_gradients
from .source_audit import ProbeConfigV2, fit_empirical_lag, cross_tx_readout, isolated_rng
from .state import RNGState, clone_buffers, restore_buffers
from .step_context import prepare_context, Core90Objective
from .step_context import slice_batch
from .legacy.objective import labeled_terms


def reference_coverage(source,args,indices):
    """Compare selected groups with the declared full source contract."""
    records=labeled_source_records(source)
    by_index={r['index']:r for r in records}
    if source.info.get('synthetic'):
        # The synthetic fixture explicitly has 3 RX, unlike the physical 5-RX contract.
        rxs=sorted({r for r,d in source.domains});days=sorted({d for r,d in source.domains})
    else:
        rxs=[int(v) for v in args.wisig_train_rxs.split(',')]
        days=[int(v) for v in args.wisig_train_days.split(',')]
    txs=list(range(args.num_classes))
    expected=set(product(txs,rxs,days))
    selected=[by_index[i] for i in indices if i in by_index]
    actual={(r['tx'],r['rx'],r['day']) for r in selected}
    available={(r['tx'],r['rx'],r['day']) for r in records}
    data_valid=(len(selected)==len(indices) and len(set(indices))==len(indices) and
                bool(selected) and all(r['role']=='train' for r in selected))
    return dict(valid=bool(data_valid and actual==expected and available==expected),
                data_valid=data_valid,expected_groups=len(expected),observed_groups=len(actual),
                missing_groups=[list(v) for v in sorted(expected-actual)],
                unexpected_groups=[list(v) for v in sorted(actual-expected)],
                missing_available_groups=[list(v) for v in sorted(expected-available)],
                group_counts=dict(Counter(f"{r['tx']}:{r['rx']}:{r['day']}" for r in selected)),
                expected_txs=txs,expected_rxs=rxs,expected_days=days,
                synthetic=bool(source.info.get('synthetic')),samples=len(indices),
                grouping_kind='whole_tx_rx_day_container',independent_repeat_claim=False)


def _batch(source,indices,seed):
    if not indices:raise ValueError('empty source reference')
    return next(iter(DataLoader(Subset(source.train,indices),batch_size=len(indices),
                               shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed))))


def _source_weights(source,indices,device,dtype):
    records=labeled_source_records(source)
    by_index={r['index']:r for r in records}
    all_counts=Counter((r['tx'],r['rx'],r['day']) for r in records)
    selected_counts=Counter((by_index[i]['tx'],by_index[i]['rx'],by_index[i]['day']) for i in indices)
    values=[all_counts[(by_index[i]['tx'],by_index[i]['rx'],by_index[i]['day'])]/
            (len(records)*selected_counts[(by_index[i]['tx'],by_index[i]['rx'],by_index[i]['day'])]) for i in indices]
    return torch.tensor(values,device=device,dtype=dtype)


def _reference_context(model,source,indices,args,ctx,epoch,seed,augmentor,frozen_u):
    batch=_batch(source,indices,seed)
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed+991)
    reference=prepare_context(batch,None,model,None,args,epoch,ctx.batch_index,ctx.weights,
                              generator,deepcopy(augmentor),getattr(ctx,'capability_level',None))
    for name in ('strong','pseudo','base_mask','strong_mask'):
        value=getattr(frozen_u,name)
        setattr(reference,name,None if value is None else value.detach().clone())
    return reference


def _freeze_actual_u(model,args,proto,ctx,training_rng):
    """Compute the actual upcoming mask on a COPY, before reference RNG/views."""
    frozen=deepcopy(ctx)
    if frozen.strong is None or frozen.strong_mask is not None:
        return frozen,0
    training_rng.restore()
    preview=deepcopy(model)
    Core90Objective(preview,args,deepcopy(proto))(frozen)
    return frozen,frozen.forward_calls-ctx.forward_calls


def _lag(model,source,indices,args,ctx,epoch,seed,augmentor,frozen_u):
    owned=deepcopy(model)
    reference=_reference_context(owned,source,indices,args,ctx,epoch,seed,augmentor,frozen_u)
    captured={}
    def pre_head(module,inputs):
        captured['head']=deepcopy(module)
        captured['state']=dict(cpu_rng=torch.get_rng_state().clone(),
            module_training={name:m.training for name,m in module.named_modules()})
        if inputs[0].is_cuda:captured['state']['cuda_rng']=torch.cuda.get_rng_state_all()
    hook=owned.adv_head.register_forward_pre_hook(pre_head)
    try:
        n=len(reference.y)
        grl=reference.weights['adv'] if args.game_head_scale=='separate_head_scale' else 1.
        with torch.no_grad():
            out=owned(torch.cat((reference.x,reference.satellite)),
                y_tx=torch.cat((reference.y,reference.y)),grl_lambda=grl,return_aux=True,
                domain_labels=torch.cat((reference.domain,reference.domain)))
    finally:hook.remove()
    # The copied pre-head includes the temporary hook; remove it before any replay.
    captured['head']._forward_pre_hooks.clear()
    weights=_source_weights(source,indices,out['z_id'].device,out['z_id'].dtype)
    weights=torch.cat((weights,torch.zeros_like(weights)))
    labels=torch.cat((reference.domain,reference.domain))
    scale=(1. if reference.weights['adv']>0 else 0.) if args.game_head_scale=='separate_head_scale' else reference.weights['adv']
    config=ProbeConfigV2(steps=args.game_probe_steps,lr=args.game_probe_lr,objective_scale=max(0.,scale),seed=seed)
    result=fit_empirical_lag(captured['head'],out['z_id'],labels,sample_weights=weights,
        objective_scope='current_training_head_objective',fixed_head_state=captured['state'],config=config)
    # Replay the original head, not the recovered endpoint, against captured logits.
    head=captured['head']
    torch.set_rng_state(captured['state']['cpu_rng'])
    if 'cuda_rng' in captured['state']:torch.cuda.set_rng_state_all(captured['state']['cuda_rng'])
    with torch.no_grad():logits=head(out['z_id'].detach())
    error=float((logits-out['adv_dom_logits']).abs().max())
    result.metrics['head_forward_parity']=dict(max_logit_error=error,valid=error<=1e-6)
    if not math.isfinite(error) or error>1e-6:
        result.metrics.update(status='OPTIMIZATION_FAILURE',quality_pass=False,control_ready=False,
                              gap_raw=None,gap_normalized=None)
        result.metrics['reason_codes'].append('head_forward_replay_mismatch')
    result.metrics['sampling']=dict(kind='bounded_labeled_source_training_pool',main_samples=n,
        head_batch_samples=2*n,weighting='source_container_frequency_divided_by_selected_count',
        satellite_direct_loss_weight=0.,main_view='current_training_augmentation',
        capability_level=getattr(ctx,'capability_level',None),epoch=epoch)
    result.metrics['effective_adversarial_weight']=float(reference.weights['adv'])
    return result


def _gradient_reference(model,recovered,source,indices,args,ctx,epoch,seed,augmentor,frozen_u,proto):
    """Full current objective on one representative source/U fixed context.

    The model's adv_dom_logits already contain GRL. Multiplying that loss by
    its actual head coefficient gives the correct signed encoder contribution.
    No extra sign reversal or additional multiplication by lambda is applied.
    """
    owned=deepcopy(model)
    reference=_reference_context(owned,source,indices,args,ctx,epoch,seed,augmentor,frozen_u)
    params=[p for p in owned.id_backbone.parameters() if p.requires_grad]
    original_buffers=clone_buffers(owned)
    rng=RNGState.capture()
    def derivatives():
        local=deepcopy(reference);captured=[]
        hook=owned.register_forward_hook(lambda module,inputs,out:captured.append(out) if not captured else None)
        try:loss=Core90Objective(owned,args,deepcopy(proto))(local)
        finally:hook.remove()
        n=len(local.y);out=captured[0]
        tx=F.cross_entropy(out['tx_logits'][:n],local.y,label_smoothing=float(args.label_smoothing))
        labeled_nonadv,_=labeled_terms(slice_batch(out,0,n,2*n),local.y,local.domain,args,
            epoch,local.batch_index,dict(local.weights,adv=0.),deepcopy(proto))
        scale=(1. if local.weights['adv']>0 else 0.) if args.game_head_scale=='separate_head_scale' else local.weights['adv']
        adv_ce=F.cross_entropy(out['adv_dom_logits'][:n],local.domain)
        adv=scale*adv_ce
        def grad(value,retain_graph):
            gradients=torch.autograd.grad(value,params,retain_graph=retain_graph,allow_unused=True)
            return [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gradients)]
        gi=grad(tx,True);gl=grad(labeled_nonadv,True);ga=grad(adv,True);full=grad(loss,False)
        return gi,gl,ga,full,float(loss.detach()),local.forward_calls-reference.forward_calls,float(adv_ce.detach())
    gi,gl,ga,full,loss,forwards,online_adv_ce=derivatives()
    # Only learned head parameters change between the paired field evaluations.
    with torch.no_grad():
        for target,value in zip(owned.adv_head.parameters(),recovered.parameters()):target.copy_(value)
    restore_buffers(owned,original_buffers);rng.restore()
    _,_,gr,recovered_full,recovered_loss,recovered_forwards,recovered_adv_ce=derivatives()
    nonadv=[g-a for g,a in zip(full,ga)]
    recovered_nonadv=[g-a for g,a in zip(recovered_full,gr)]
    metric=gradient_diagnostics(gi,ga,gr,g_nonadv=nonadv)
    metric['labeled_nonadversarial']=gradient_diagnostics(gl,ga,gr)
    metric['labeled_nonadversarial_scope']='all_current_labeled_main_view_terms_excluding_adversarial_satellite_and_U'
    metric['full_nonadversarial_scope']='all_current_labeled_satellite_and_U_terms_excluding_adversarial'
    metric['full_online_recovered']=compare_gradients(full,recovered_full)
    residual=max(float((a-b).abs().max()) for a,b in zip(nonadv,recovered_nonadv))
    metric['nonadversarial_replay_max_abs_error']=residual
    metric.update(status='MEASURED' if metric['valid'] else 'UNAVAILABLE',
        scope='full_current_training_objective',representative=True,
        identity_scope='current_labeled_TX_CE',full_loss=loss,recovered_full_loss=recovered_loss,
        reference_samples=len(reference.y),unlabeled_samples=0 if reference.strong is None else len(reference.strong),
        pseudo_mask_source='owned_current_training_context_preview',
        pseudo_selected=0 if reference.strong_mask is None else int(reference.strong_mask.sum()),
        effective_adversarial_weight=float(ctx.weights['adv']),
        sign_convention='adv_dom_logits_already_has_GRL; actual_head_coefficient_only',
        decomposition='full_current_training_objective_minus_signed_adversarial_contribution',
        field_evaluations=2,backward_evaluations=8,model_forwards=forwards+recovered_forwards)
    metric.update(reference_online_adv_ce=online_adv_ce,reference_recovered_adv_ce=recovered_adv_ce,
                  reference_transfer_tolerance=1e-6,
                  reference_transfer_threshold_source='fixed_numerical_absolute_tolerance')
    cosine=metric['online_recovered']['cosine']
    metric['direction_imbalance']=None if cosine is None else 1.-cosine
    if residual>1e-5 or not math.isfinite(residual):
        metric.update(valid=False,status='UNAVAILABLE',reason='paired_nonadversarial_replay_mismatch')
    if recovered_adv_ce>online_adv_ce+1e-6:
        metric.update(valid=False,status='TRANSFER_FAILURE',reason='reference_head_objective_worsened')
    return metric


def _eval_sets(model,source,indexes,args,seed):
    owned=deepcopy(model).eval();unique=sorted(set(i for values in indexes.values() for i in values))
    if not unique:return {},0
    batch=_batch(source,unique,seed)
    x,y,d,meta=batch;device=next(model.parameters()).device
    x=x.to(device);zs=[];forwards=0
    with torch.no_grad():
        for block in x.split(args.eval_batch_size):
            zs.append(owned(block,return_aux=True)['z_id'].float().detach());forwards+=1
    z=torch.cat(zs);positions={i:p for p,i in enumerate(unique)}
    sets={}
    for name,values in indexes.items():
        selected=torch.tensor([positions[i] for i in values],device=device,dtype=torch.long)
        sets[name]=dict(z=z[selected],x=x[selected],y=y.to(device)[selected],d=d.to(device)[selected],
            rx=meta['rx_i'].to(device)[selected],groups=[meta['capture_group'][positions[i]] for i in values])
    return sets,forwards


def game_audit_v2(model,source,indexes,auditor,args,ctx,step,version,epoch,proto=None,augmentor=None):
    """Independent empirical lag/readout/gradient; caller merges capability.

    Only auditor.calls advances. Model, prototype, ctx caches, augmentation state,
    caller gradients and all global RNG streams remain unchanged.
    """
    started=time.perf_counter();training_rng=RNGState.capture()
    lag_cov=reference_coverage(source,args,indexes.get('lag_reference',[]))
    grad_cov=reference_coverage(source,args,indexes.get('gradient_reference',[]))
    coverage_valid=lag_cov['valid'] and grad_cov['valid']
    metrics=make_evidence_v2(observation_id=f'game:{step}:{version}:{auditor.calls}',step=step,
        encoder_version=version,data_valid=lag_cov['data_valid'] and grad_cov['data_valid'],coverage_valid=coverage_valid)
    metrics['coverage']=dict(lag=lag_cov,gradient=grad_cov)
    metrics.update(model_forwards=0,gradient_backward_evaluations=0,probe_steps=0)
    sets={};seed=auditor.config.seed+auditor.calls
    with isolated_rng(seed):
        if coverage_valid:
            frozen_u,preview_forwards=_freeze_actual_u(model,args,proto,ctx,training_rng)
            metrics['model_forwards']+=preview_forwards
            # Preview used the live upcoming RNG; fixed reference generation is independent.
            torch.manual_seed(seed)
            result=_lag(model,source,indexes['lag_reference'],args,ctx,epoch,seed,augmentor,frozen_u)
            metrics['lag']=result.metrics;metrics['model_forwards']+=1
            metrics['probe_steps']+=result.metrics['budget']['actual_steps']
            if result.metrics['quality_pass'] and result.metrics['control_ready']:
                grad=_gradient_reference(model,result.recovered_head,source,indexes['gradient_reference'],
                    args,ctx,epoch,seed+1,augmentor,frozen_u,proto)
                metrics['gradient']=grad
                metrics['model_forwards']+=grad['model_forwards']
                metrics['gradient_backward_evaluations']+=grad['backward_evaluations']
            else:
                metrics['gradient']=dict(unavailable('untrusted_recovered_head'),valid=False,representative=False,
                                         scope='full_current_training_objective')
            due=auditor.calls % max(1,args.game_independent_probe_every)==0
            if due or args.game_response_tracking:
                sets,count=_eval_sets(model,source,indexes,args,seed)
                metrics['model_forwards']+=count
            if due:
                ref=sets['lag_reference']
                records=labeled_source_records(source);by_index={r['index']:r for r in records}
                tx=torch.tensor([by_index[i]['tx'] for i in indexes['lag_reference']],device=ref['z'].device)
                unique_rx=sorted(set(ref['rx'].tolist()));rx_map={r:i for i,r in enumerate(unique_rx)}
                rx=torch.tensor([rx_map[r] for r in ref['rx'].tolist()],device=ref['z'].device)
                readout=cross_tx_readout(ref['z'],ref['d'],rx,tx,ref['groups'],
                    config=ProbeConfigV2(steps=args.game_probe_steps,lr=args.game_probe_lr,seed=seed),seed=seed)
                metrics['cross_tx_readout']=readout
                metrics['probe_steps']+=sum(p.get('budget',{}).get('actual_steps',0)
                    for fold in readout['folds'] for task in fold['readouts'].values() for p in task.values())
            else:metrics['cross_tx_readout']=unavailable('readout_not_due')
        else:
            metrics['lag']=unavailable('incomplete_source_contract_coverage')
            metrics['gradient']=dict(unavailable('incomplete_source_contract_coverage'),valid=False,representative=False)
    auditor.calls+=1
    metrics['elapsed_seconds']=time.perf_counter()-started
    return metrics,sets
