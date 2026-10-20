import torch
from . import design as d
from experiments.pamf_r46_anchor_refine.models import CompactSuccessor as Existing
from experiments.pamf_r46_anchor_refine.models import compact_from_parent as weight_only_prune
from experiments.pamf_r46_anchor_refine.models import objective as iq_objective

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered feature-hint scope required')
    model=Existing('prune_decay');model.arm=arm
    if sum(p.numel() for p in model.parameters())!=273765:raise ValueError('Unchanged complete compressed graph required')
    return model

@torch.no_grad()
def compact_from_parent(parent,arm):
    if arm not in d.ARMS:raise ValueError('Registered feature-hint scope required')
    model=weight_only_prune(parent,'prune_decay');model.arm=arm
    model.initialization_report.update(mode='R46_weight_only_prune_feature_hint_'+arm,feature_hint_scope=arm,feature_hint_weight=d.HINT_WEIGHT,
        new_trainable_parameters=0,new_calibration_IQ=0)
    return model

def objective(student,teacher,clean,epoch,arm,features):
    if arm not in d.ARMS or set(features)!={'teacher_context_feature_nmse','teacher_prior_feature_nmse'}:
        raise ValueError('Registered feature-hint components required')
    context,prior=features['teacher_context_feature_nmse'],features['teacher_prior_feature_nmse']
    if any(v.shape!=(len(clean),) for v in (context,prior)) or (arm=='context_hint' and bool(torch.count_nonzero(prior))):
        raise ValueError('Per-IQ registered scope required')
    base,terms=iq_objective(student,teacher,clean,epoch,'prune_decay')
    return base+d.HINT_WEIGHT*(context+prior),dict(terms,**features)
