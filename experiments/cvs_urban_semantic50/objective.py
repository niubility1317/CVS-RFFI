"""Bounded same-reference-RX/day triplets; only anchors receive gradients."""
import math
import torch
from torch.nn import functional as F
from experiments.cvs_urban_identity50.features import auxiliary_joint

MARGIN=.1
EPS=1e-4


def validate(clean,urban,labels,receiver,day,quality,margin):
    n=len(labels)
    if (clean.shape!=(n,160) or urban.shape!=clean.shape or labels.ndim!=1 or labels.dtype!=torch.long
            or bool(((labels<0)|(labels>=6)).any())):
        raise ValueError('Actual joint160 with legal L labels required; U sentinels forbidden')
    if not math.isfinite(margin) or not 0<margin<=.5:raise ValueError('Predeclared bounded cosine margin required')
    if any(v.shape!=(n,) for v in (receiver,day,quality)) or quality.requires_grad:
        raise ValueError('Matching source metadata and detached observation quality required')
    if not torch.isfinite(clean).all() or not torch.isfinite(urban).all() or not torch.isfinite(quality).all():
        raise ValueError('Finite source identity/quality required')
    if bool(((quality<0)|(quality>1)).any()):raise ValueError('Continuous observation weights must be in [0,1]')
    if not set(receiver.detach().cpu().tolist())<=set((1,3,4,6,8)) or not set(day.detach().cpu().tolist())<={1,2,3}:
        raise ValueError('Only registered source RX/day metadata may define references')


def mask(labels,receiver,day):
    # [anchor, positive, negative]. Both references share one RX/day; their
    # RX differs from the anchor. Known same-TX packets are never negatives.
    positive=(labels[:,None]==labels[None,:])&(receiver[:,None]!=receiver[None,:])
    negative=labels[:,None]!=labels[None,:]
    reference=(receiver[:,None]==receiver[None,:])&(day[:,None]==day[None,:])
    return positive[:,:,None]&negative[:,None,:]&reference[None,:,:]


def rank_one(anchor,keys,valid,weights,margin):
    similarities=anchor@keys.detach().T
    values=F.relu(margin+similarities[:,None,:]-similarities[:,:,None])
    count=valid.sum((1,2))
    # Normalize every anchor first; missing reference groups contribute zero.
    # There is no confidence, student agreement, fitted head or hidden U label.
    each=(values*valid*weights).sum((1,2))/count.clamp_min(1)
    return each.mean(),dict(triplets=int(valid.sum()),anchors_with_references=int((count>0).sum()),
        active_triplets=int(((values.detach()>0)&valid).sum()))


def semantic(clean,urban,labels,receiver,day,quality,*,bidirectional=False,margin=MARGIN):
    validate(clean,urban,labels,receiver,day,quality,margin)
    c=F.normalize(clean,dim=1,eps=EPS);u=F.normalize(urban,dim=1,eps=EPS)
    valid=mask(labels,receiver,day)
    forward,stats=rank_one(u,c,valid,quality[:,None,None],margin)
    reverse_weight=(quality[None,:,None]*quality[None,None,:]).sqrt()
    reverse,reverse_stats=rank_one(c,u,valid,reverse_weight,margin)
    loss=.5*(forward+reverse) if bidirectional else forward
    return loss,dict(urban_to_clean=forward,clean_to_urban=reverse,**stats,
        reverse_active_triplets=reverse_stats['active_triplets'],
        bidirectional=bool(bidirectional),maximum_per_triplet=margin+2,
        normalization='per anchor triplet mean then full batch mean; bidirectional arithmetic half sum')
