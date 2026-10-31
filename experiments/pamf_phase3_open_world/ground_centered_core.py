"""Shift-invariant six-logit evidence and equal-public-domain source ranking."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from . import logit_energy_core as base

ARMS=('CPAIR',)


def centered_evidence(logits):
    """LSE(z)-mean(z)-log6 = KL(Uniform6 || softmax(z)), per record.

    log_softmax removes the common peak before reduction, avoiding cancellation
    of a large common offset. Original classifier logits are never modified.
    """
    value=base._logits(logits)
    return -F.log_softmax(value,dim=1).mean(dim=1)-math.log(6.)


def centered_energy(logits):
    value=np.asarray(logits)
    if value.ndim!=2 or not len(value) or value.shape[1]!=6 or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('Finite complete-six FP32 native logits required')
    a=value.astype(np.float64);relative=a-a.max(axis=1,keepdims=True)
    result=np.log(np.exp(relative).sum(axis=1))-relative.mean(axis=1)-math.log(6.)
    if not np.isfinite(result).all():raise ValueError('Finite FP64 centered six-logit evidence required')
    return result


def compute(logits_L,y_L,teacher_logits_L,domains_L,logits_OE,domains_OE,arm='CPAIR'):
    if arm not in ARMS:raise ValueError('Only the single centered-pair source objective is permitted')
    L=base._logits(logits_L);OE=base._logits(logits_OE);teacher=base._logits(teacher_logits_L,len(L))
    if any(a.device!=L.device for a in (OE,teacher)):raise ValueError('Source roles and frozen teacher must share a device')
    y=base._codes(y_L,len(L),L.device,6,'known L labels')
    ld=base._codes(domains_L,len(L),L.device,15,'public L domains')
    od=base._codes(domains_OE,len(OE),L.device,15,'public OE domains')
    if not bool((ld[:,None]==od[None]).any(dim=1).all()):raise ValueError('Every labeled source domain needs matching public OE')
    ce=F.cross_entropy(L,y,label_smoothing=0.)
    kd=4*F.kl_div(F.log_softmax(L/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='batchmean')
    cL=centered_evidence(L);cOE=centered_evidence(OE)
    pairs=F.softplus(cOE[None]-cL[:,None]);counts=[0]*15;means=[None]*15;rawmeans=[None]*15;terms=[];raw_terms=[]
    sL=torch.logsumexp(L,dim=1);sOE=torch.logsumexp(OE,dim=1)
    with torch.no_grad():raw_pairs=F.softplus(sOE.detach()[None]-sL.detach()[:,None])
    for domain in range(15):
        mask=(ld[:,None]==domain)&(od[None,:]==domain);selected=pairs[mask]
        if not selected.numel():continue
        mean=selected.mean();raw=raw_pairs[mask].mean();terms.append(mean);raw_terms.append(raw)
        counts[domain]=selected.numel();means[domain]=float(mean.detach().cpu());rawmeans[domain]=float(raw.cpu())
    rank=torch.stack(terms).mean();raw_mean=torch.stack(raw_terms).mean()
    return dict(total=ce+kd+rank,CE=ce,KD=kd,OE=rank,rank=rank,uniform=L.new_zeros(()),
        domain_count=len(terms),pair_count=sum(counts),domain_pair_counts=counts,domain_mean_losses=means,
        domain_raw_mean_losses=rawmeans,raw_domain_mean=raw_mean,centered_domain_mean=rank,
        s_L=sL.mean(),s_OE=sOE.mean(),centered_s_L=cL.mean(),centered_s_OE=cOE.mean(),
        logit_mean_L=L.mean(),logit_mean_OE=OE.mean(),source_only=True,
        loss_reduction='equal_public_RX_day',evidence='logsumexp_minus_mean_minus_log6')
