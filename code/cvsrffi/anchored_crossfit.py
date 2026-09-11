"""Head-only RX cross-fitting, including nested held-out gate evaluation."""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import torch
from .anchored_fit import FitConfig,fit_expert,paired_weights
from .anchored_fusion import ALPHAS,realize_actions,action_utilities,utility_features,UtilityGate,protection_quantile


@dataclass(frozen=True)
class RXFold:
    heldout_rx:int
    train_rx:tuple
    train_physical_ids:tuple
    predict_physical_ids:tuple


def build_rx_folds(cache):
    cache.validate()
    if cache.identity.role!='L_s':raise ValueError('crossfit requires L_s')
    receivers=sorted(set(cache.receiver.tolist()))
    if len(receivers)<3:raise ValueError('RX crossfit needs at least three receivers')
    folds=[]
    for held in receivers:
        train_ids=tuple(sorted({pid for pid,rx in zip(cache.physical_ids,cache.receiver.tolist()) if rx!=held}))
        predict_ids=tuple(sorted({pid for pid,rx in zip(cache.physical_ids,cache.receiver.tolist()) if rx==held}))
        if set(train_ids)&set(predict_ids):raise ValueError('crossfit physical overlap')
        folds.append(RXFold(held,tuple(r for r in receivers if r!=held),train_ids,predict_ids))
    return folds


@dataclass
class OOFArtifact:
    scores:torch.Tensor
    folds:list
    experts:dict
    final_expert:object
    cache_identity:dict
    binding:dict


def oof_binding(cache,config,seed,w0,tau0):
    from .anchored_pipeline import state_identity
    return dict(row_keys=list(zip(cache.physical_ids,cache.view_ids)),config=asdict(FitConfig.parse(config)),
                head_seed=int(seed),anchor=state_identity(dict(w0=w0,tau0=float(tau0))))


def validate_oof(cache,oof,config,seed,w0,tau0):
    if asdict(cache.identity)!=oof.cache_identity or oof.binding!=oof_binding(cache,config,seed,w0,tau0):
        raise ValueError('OOF row/config/seed/anchor identity mismatch')
    if oof.folds!=build_rx_folds(cache):raise ValueError('OOF fold mismatch')
    for train_rx,expert in oof.experts.items():
        if tuple(expert.train_rx)!=tuple(train_rx):raise ValueError('OOF expert training RX mismatch')
        if hasattr(expert,'config') and expert.config!=asdict(FitConfig.parse(config)):raise ValueError('OOF expert configuration mismatch')
        if hasattr(expert,'seed') and expert.seed!=seed:raise ValueError('OOF expert seed mismatch')
    if oof.scores.shape!=cache.baseline_inference_logits.shape or not torch.isfinite(oof.scores).all():raise ValueError('invalid OOF scores')


def run_expert_oof(cache,config,seed,output,*,w0,tau0,device='cpu',fit_fn=fit_expert):
    config=FitConfig.parse(config);folds=build_rx_folds(cache)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scores=torch.empty_like(cache.baseline_inference_logits,device='cpu');seen=torch.zeros(len(cache),dtype=torch.bool)
    experts={}
    for fold in folds:
        expert=fit_fn(cache,fold.train_rx,config,seed,output/f'rx_{fold.heldout_rx}',w0=w0,tau0=tau0,device=device,validation_rx=(fold.heldout_rx,))
        indices=torch.where(cache.receiver.cpu()==fold.heldout_rx)[0];pred=expert.predict(cache.take(indices))
        if seen[indices].any():raise ValueError('duplicate OOF predictions')
        scores[indices]=pred;seen[indices]=True;experts[fold.train_rx]=expert
    if not seen.all() or not torch.isfinite(scores).all():raise ValueError('incomplete OOF predictions')
    all_rx=tuple(sorted(set(cache.receiver.tolist())))
    final=fit_fn(cache,all_rx,config,seed,output/'all_source',w0=w0,tau0=tau0,device=device)
    experts[all_rx]=final
    binding=oof_binding(cache,config,seed,w0,tau0)
    torch.save(dict(schema='head_only_oof_v1',scores=scores,s0=cache.baseline_inference_logits.cpu(),binding=binding,
                    physical_ids=list(cache.physical_ids),view_ids=list(cache.view_ids),folds=[asdict(f) for f in folds],
                    cache_identity=asdict(cache.identity),candidate=config.candidate,head_seed=seed),output/'oof_predictions.pt')
    return OOFArtifact(scores,folds,experts,final,asdict(cache.identity),binding)


def _gate_training(rows,sg,lambda_h,ridge,utility_margin,quantile):
    s0=rows.baseline_inference_logits.cpu();sg=sg.cpu()
    threshold=protection_quantile(s0,quantile)
    actions=realize_actions(s0,sg,protection_threshold=threshold,valid=rows.valid.cpu())
    phi=utility_features(s0,sg,rows.quality.cpu(),rows.observed.float().mean(-1).cpu())
    u=action_utilities(actions['log_probabilities'],rows.labels.cpu(),s0.argmax(-1),lambda_h,predictions=actions['predictions'])
    weights=paired_weights(rows,True).cpu()
    gate=UtilityGate(ridge,utility_margin).fit(phi,u,weights=weights)
    # Physical-weighted actual utility, tie breaks toward smaller action.
    means=(weights[:,None]*u).sum(0);fixed=int(means.argmax())
    return gate,threshold,fixed,means,actions


def outer_fixed_actions(cache,oof,protection_q=.9):
    """Each fixed arm's guard uses only other RX H0 margins.

    The guard depends on H0 alone, so inner G fits cannot change this threshold.
    This gives the same guard-training rows as nested OOF without extra fits.
    """
    n,c=oof.scores.shape
    result=dict(log_probabilities=torch.empty(n,5,c,dtype=torch.double),alpha=torch.empty(n,5,dtype=torch.double),
                reason=torch.empty(n,5,dtype=torch.long),predictions=torch.empty(n,5,dtype=torch.long),protected=torch.empty(n,dtype=torch.bool),requested_alpha=torch.tensor(ALPHAS,dtype=torch.double))
    audit=[]
    for fold in oof.folds:
        train=torch.where(cache.receiver!=fold.heldout_rx)[0];held=torch.where(cache.receiver==fold.heldout_rx)[0]
        threshold=protection_quantile(cache.baseline_inference_logits[train],protection_q)
        actual=realize_actions(cache.baseline_inference_logits[held],oof.scores[held],protection_threshold=threshold,valid=cache.valid[held])
        for key in ('log_probabilities','alpha','reason','protected','predictions'):result[key][held]=actual[key]
        audit.append(dict(heldout_rx=fold.heldout_rx,protection_train_rx=list(fold.train_rx),threshold=threshold))
    return result,audit


def run_nested_fusion_audit(cache,oof,config,seed,output,*,w0,tau0,lambda_h=2.,ridge=.001,utility_margin=0.,
                            protection_q=.9,device='cpu',fit_fn=fit_expert):
    config=FitConfig.parse(config);cache.validate()
    if config.candidate!='A4':raise ValueError('A6 uses the preregistered A4 expert')
    validate_oof(cache,oof,config,seed,w0,tau0)
    if len(oof.folds)<4:raise ValueError('nested gate audit requires at least four source RX')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fitted=dict(oof.experts);selected=torch.zeros(len(cache),dtype=torch.long)
    mixture=torch.empty((len(cache),cache.baseline_inference_logits.shape[1]),dtype=torch.double)
    fixed_mixture=torch.empty_like(mixture);decisions=torch.empty(len(cache),dtype=torch.long);fixed_decisions=torch.empty_like(decisions)
    audits=[];all_rx=tuple(sorted(set(cache.receiver.tolist())))
    for fold in oof.folds:
        inner_indices=torch.where(cache.receiver.cpu()!=fold.heldout_rx)[0];inner=cache.take(inner_indices)
        inner_scores=torch.empty_like(inner.baseline_inference_logits);seen=torch.zeros(len(inner),dtype=torch.bool)
        for held in fold.train_rx:
            training=tuple(rx for rx in all_rx if rx not in {fold.heldout_rx,held})
            if training not in fitted:
                fitted[training]=fit_fn(cache,training,config,seed,output/('train_'+'_'.join(map(str,training))),w0=w0,tau0=tau0,device=device)
            index=torch.where(inner.receiver.cpu()==held)[0]
            inner_scores[index]=fitted[training].predict(inner.take(index));seen[index]=True
        if not seen.all():raise ValueError('missing inner OOF predictions')
        gate,threshold,fixed,means,_=_gate_training(inner,inner_scores,lambda_h,ridge,utility_margin,protection_q)
        held_indices=torch.where(cache.receiver.cpu()==fold.heldout_rx)[0];held_rows=cache.take(held_indices)
        s0=held_rows.baseline_inference_logits.cpu();sg=oof.scores[held_indices]
        phi=utility_features(s0,sg,held_rows.quality.cpu(),held_rows.observed.float().mean(-1).cpu())
        actions=realize_actions(s0,sg,protection_threshold=threshold,valid=held_rows.valid.cpu())
        choice=gate.choose(phi)['action'];selected[held_indices]=choice
        mixture[held_indices]=actions['log_probabilities'][torch.arange(len(choice)),choice]
        fixed_mixture[held_indices]=actions['log_probabilities'][:,fixed]
        decisions[held_indices]=actions['predictions'][torch.arange(len(choice)),choice]
        fixed_decisions[held_indices]=actions['predictions'][:,fixed]
        audits.append(dict(heldout_rx=fold.heldout_rx,expert_train_rx=list(fold.train_rx),gate_train_rx=list(fold.train_rx),
                           gate_train_physical_ids=list(fold.train_physical_ids),predict_physical_ids=list(fold.predict_physical_ids),
                           protection_threshold=threshold,inner_fixed_action=fixed,inner_action_utilities=means.tolist()))
    gate,threshold,fixed,means,final_actions=_gate_training(cache,oof.scores,lambda_h,ridge,utility_margin,protection_q)
    result=dict(schema='nested_head_only_gate_audit_v1',folds=audits,selected_actions=selected,
                nested_log_probabilities=mixture,nested_fixed_log_probabilities=fixed_mixture,
                nested_predictions=decisions,nested_fixed_predictions=fixed_decisions,
                final_gate_state=gate.state_dict(),final_protection_threshold=threshold,final_fixed_action=fixed,
                final_oof_action_utilities=means,training_set_count=len(fitted),
                interpretation='Head-only RX holdout; frozen H0 backbone saw all source RX; final gate OOF is its training data.')
    torch.save(result,output/'nested_fusion.pt')
    (output/'fold_manifest.json').write_text(json.dumps(dict(folds=audits,training_set_count=len(fitted),
        cache_identity=asdict(cache.identity),candidate=config.candidate,head_seed=seed),indent=2)+'\n',encoding='utf-8')
    return result
