"""Prototype losses for RX-aware L/U relations and label-free recovery views.

No launcher installs these yet. Ordinary PL and the existing labeled objective
are caller-owned and are never replaced by this module.
"""
import math
import torch
from torch.nn import functional as F


def gaussian_view(x,*,snr_db,generator):
    if (x.ndim!=3 or x.shape[1]!=2 or not x.is_floating_point() or not torch.isfinite(x).all()
            or not math.isfinite(snr_db) or snr_db<0):
        raise ValueError('Finite IQ and explicit nonnegative SNR required')
    norm=x.flatten(1).norm(dim=1)
    if bool((norm<=0).any()):raise ValueError('Nonzero IQ required')
    noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
    noise=noise-noise.mean((1,2),keepdim=True)
    scale=noise.flatten(1).norm(dim=1)
    if bool((scale<=0).any()):raise ValueError('Nonzero noise draw required')
    noise=noise*(norm/scale*10**(-snr_db/20))[:,None,None]
    return x+noise


def instance_alignment(student_views,raw_reference):
    """Every U can align its own views; other U are never assumed different TX."""
    if (student_views.ndim!=3 or raw_reference.ndim!=2 or student_views.shape[1:]!=raw_reference.shape
            or not torch.isfinite(student_views).all() or not torch.isfinite(raw_reference).all()):
        raise ValueError('Aligned [views,physical samples,features] and raw reference required')
    z=F.normalize(student_views,dim=-1);target=F.normalize(raw_reference.detach(),dim=-1)
    return (1-(z*target[None]).sum(-1)).mean()


def cross_receiver_relation(features,*,rx,source_rxs,labeled,visible_tx,pseudo,trust,temperature=.1,scope='LU'):
    """L truth and certified U pseudo targets; balance positive RX mass per anchor.

    scope U anchors only U and detaches labeled reference features. scope LU
    trains both; it does not add confidence gates to true labeled identities.
    """
    n=len(features);vectors=(rx,labeled,visible_tx,pseudo,trust)
    if (features.ndim!=2 or n<1 or any(v.shape!=(n,) or v.device!=features.device for v in vectors)
            or labeled.dtype!=torch.bool or rx.dtype!=torch.long or visible_tx.dtype!=torch.long or pseudo.dtype!=torch.long
            or scope not in ('U','LU') or temperature<=0 or not math.isfinite(temperature)
            or not torch.isfinite(features).all() or not torch.isfinite(trust).all()
            or bool(((trust<0)|(trust>1)).any()) or set(rx.tolist())-set(source_rxs)
            or bool((visible_tx[~labeled]!=-1).any())
            or bool(((visible_tx[labeled]<0)|(visible_tx[labeled]>=6)).any())
            or bool(((pseudo<0)|(pseudo>=6)).any())):
        raise ValueError('Source RX, L true labels, hidden U labels and bounded frozen trust required')
    h=torch.where(labeled[:,None],features.detach(),features) if scope=='U' else features
    z=F.normalize(h,dim=1);y=torch.where(labeled,visible_tx,pseudo).detach()
    w=torch.where(labeled,torch.ones_like(trust),trust).detach();valid=w>0
    cross=rx[:,None]!=rx[None,:]
    candidates=cross&valid[:,None]&valid[None,:]
    positive=candidates&(y[:,None]==y[None,:]);negative=candidates&~positive
    active=valid&positive.any(1)&negative.any(1)
    if scope=='U':active&=~labeled
    weights=positive*w[None,:]
    # Equal total positive weight per contributing receiver, not per packet count.
    for receiver in source_rxs:
        mask=(rx==receiver)[None,:];mass=(weights*mask).sum(1,keepdim=True)
        weights=torch.where(mask,weights/mass.clamp_min(1e-12),weights)
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    if not bool(active.any()):return features.sum()*0.,dict(anchors=0,positive_pairs=0,scope=scope)
    scores=z@z.T/temperature
    denominator=torch.logsumexp(scores[active].masked_fill(~candidates[active],-torch.inf),dim=1)
    per_anchor=denominator-(scores[active]*weights[active]).sum(1)
    # Complete physical batch denominator: no renormalization to the trusted subset.
    loss=(per_anchor*w[active]).sum()/n
    return loss,dict(anchors=int(active.sum()),positive_pairs=int(positive[active].sum()),scope=scope,
        trusted_U=int((valid&~labeled).sum()),true_L=int(labeled.sum()),U_truth_read=False)
