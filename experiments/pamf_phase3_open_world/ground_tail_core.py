"""Equal RX/day mean or deterministic upper-quarter pairwise rank risk.

The tail loss is a source-only surrogate, not a tail-accuracy guarantee.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from . import logit_energy_core as base

ARMS=('DMEAN','DTAIL')
TAIL_FRACTION=.25


def compute(logits_L,y_L,teacher_logits_L,domains_L,logits_OE,domains_OE,arm):
    if arm not in ARMS:raise ValueError('Only equal-domain DMEAN/DTAIL source objectives allowed')
    common=base.compute(logits_L,y_L,teacher_logits_L,domains_L,logits_OE,domains_OE,'RXPAIR')
    sL=torch.logsumexp(logits_L,dim=1);sOE=torch.logsumexp(logits_OE,dim=1)
    pairs=F.softplus(sOE[None]-sL[:,None]);means=[];tails=[];counts=[0]*15;selected=[0]*15
    mean_values=[None]*15;tail_values=[None]*15
    for domain in range(15):
        mask=(domains_L[:,None]==domain)&(domains_OE[None,:]==domain)
        values=pairs[mask];n=values.numel()
        if not n:continue
        k=math.ceil(n*TAIL_FRACTION)
        # Row-major boolean selection plus stable descending order fixes ties.
        indices=torch.argsort(values.detach(),descending=True,stable=True)[:k]
        average=values.mean();tail=values[indices].mean()
        means.append(average);tails.append(tail);counts[domain]=n;selected[domain]=n if arm=='DMEAN' else k
        mean_values[domain]=float(average.detach().cpu());tail_values[domain]=float(tail.detach().cpu())
    if not means:raise ValueError('At least one matched public source domain required')
    mean=torch.stack(means).mean();tail=torch.stack(tails).mean();outside=mean if arm=='DMEAN' else tail
    return dict(total=common['CE']+common['KD']+outside,CE=common['CE'],KD=common['KD'],OE=outside,rank=outside,
        uniform=logits_L.new_zeros(()),s_L=sL.mean(),s_OE=sOE.mean(),pair_count=sum(counts),selected_pair_count=sum(selected),
        domain_count=len(means),tail_fraction=1. if arm=='DMEAN' else TAIL_FRACTION,domain_pair_counts=counts,domain_selected_counts=selected,
        domain_mean_losses=mean_values,domain_tail_losses=tail_values,equal_domain_mean=mean,equal_domain_tail=tail,global_pair_mean=common['rank'],
        loss_reduction='equal_public_RX_day',hard_pair_selection='largest_softplus_stable_row_major',source_only=True)
