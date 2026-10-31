"""Source-only equal-domain uniform OE; pair ranking is diagnostic only."""
from __future__ import annotations
import torch
from . import ground_centered_core as centered

ARMS=('UNIFORM',)


def compute(logitsL,y,teacher,dL,logitsOE,dOE,arm='UNIFORM'):
    if arm not in ARMS:raise ValueError('Only the ground UNIFORM source objective is allowed')
    base=centered.compute(logitsL,y,teacher,dL,logitsOE,dOE)
    per_record=centered.centered_evidence(logitsOE)
    counts=[0]*15;losses=[None]*15;terms=[]
    for domain in range(15):
        values=per_record[dOE==domain]
        if not values.numel():continue
        mean=values.mean();terms.append(mean);counts[domain]=values.numel();losses[domain]=float(mean.detach().cpu())
    uniform=torch.stack(terms).mean()
    return dict(base,total=base['CE']+base['KD']+uniform,OE=uniform,uniform=uniform,
        domain_uniform_mean=uniform,uniform_global_mean=per_record.mean(),domain_uniform_count=len(terms),
        domain_uniform_losses=losses,domain_uniform_counts=counts,OE_weight=1.,
        OE_objective='equal_domain_uniform_posterior',rank_is_diagnostic=True,pair_risk_in_objective=False,
        OE_true_labels_used=False)
