"""Fixed visible-L references for auxiliary cross-receiver identity relations."""
import math
import statistics
import torch
from torch.nn import functional as F

TEMPERATURE=.1
VARIANTS=('RawReference','RestoredReference','PairedReference')
BATCHES=12


def metadata_long(values):
    """The N607 Torch/NumPy combination cannot use the NumPy tensor bridge."""
    return torch.tensor(values.tolist(),dtype=torch.long)


def build_references(features,labels,receivers,source_rxs,classes,role):
    if role!='L_s':raise ValueError('Identity references may only use visible source L_s')
    rxs=tuple(source_rxs)
    if len(rxs)<2 or len(set(rxs))!=len(rxs):raise ValueError('Distinct source receiver references required')
    if (features.ndim!=2 or features.shape[1]!=160 or len(features)!=len(labels) or labels.shape!=receivers.shape
            or labels.ndim!=1 or not bool(torch.isfinite(features).all())
            or labels.min()<0 or labels.max()>=classes or set(receivers.tolist())-set(rxs)):
        raise ValueError('Aligned finite visible-L features/classes/source RX required')
    z=F.normalize(features.detach(),dim=1);means=[];counts=[]
    for label in range(classes):
        row=[];sizes=[]
        for rx in rxs:
            selected=(labels==label)&(receivers==rx);n=int(selected.sum())
            if n==0:raise ValueError('Every source class/RX requires visible L references')
            mean=z[selected].double().mean(0).to(z.dtype)
            if not bool(mean.norm()>1e-12):raise ValueError('Nonzero source reference required')
            row.append(F.normalize(mean,dim=0));sizes.append(n)
        means.append(torch.stack(row));counts.append(sizes)
    return torch.stack(means).detach(),torch.tensor(counts,dtype=torch.long)


def relation_loss(features,references,receivers,source_rxs,target,accepted):
    rxs=tuple(source_rxs);n=len(features)
    if (features.ndim!=2 or features.shape[1]!=160 or references.ndim!=3 or references.shape[1:]!=(len(rxs),160)
            or target.shape!=(n,len(references)) or accepted.shape!=(n,) or receivers.shape!=(n,)
            or len(rxs)<2 or len(set(rxs))!=len(rxs) or set(receivers.tolist())-set(rxs)
            or references.requires_grad or target.requires_grad or accepted.dtype!=torch.bool
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(references).all())
            or not bool(torch.isfinite(target).all()) or bool((target<0).any())
            or not torch.allclose(target.sum(1),torch.ones(n,device=target.device),atol=1e-5,rtol=1e-5)):
        raise ValueError('Detached aligned source references/soft targets and hidden-U acceptance required')
    keep=receivers[:,None]!=torch.tensor(rxs,device=receivers.device)[None,:]
    similarity=torch.einsum('nd,crd->ncr',F.normalize(features,dim=1),references)
    score=(similarity*keep[:,None,:]).sum(2)/keep.sum(1)[:,None]
    per_packet=-(target*F.log_softmax(score/TEMPERATURE,dim=1)).sum(1)
    return (per_packet*accepted).mean()


def decide_shared_dose(rows):
    if len(rows)!=BATCHES:raise ValueError('Twelve complete source gradient batches required')
    primary=[r['primary_norm'] for r in rows]
    extras={v:[r[v+'_norm'] for r in rows] for v in VARIANTS}
    cosine=[r[v+'_cosine'] for r in rows for v in VARIANTS]
    if any(not math.isfinite(x) for x in primary+sum(extras.values(),[])+cosine) or any(x<=0 for x in primary+sum(extras.values(),[])):
        raise ValueError('Finite nonzero source gradient norms required')
    medians={v:statistics.median(x) for v,x in extras.items()}
    weight=min(.25,.10*statistics.median(primary)/max(medians.values()))
    return dict(selected_weight=weight,median_primary_norm=statistics.median(primary),median_extra_norm=medians,
        median_cosine={v:statistics.median(r[v+'_cosine'] for r in rows) for v in VARIANTS},
        weighted_median_norm_ratio={v:weight*x/statistics.median(primary) for v,x in medians.items()},
        rule='min(.25,.10*median(primary_norm)/max_v median(extra_v_norm));one shared B/C/D dose;no target feedback',
        scope='Preclip CENTER_UNITS physical gradient proxy;not AdamW update guarantee;source transfer still untested')
