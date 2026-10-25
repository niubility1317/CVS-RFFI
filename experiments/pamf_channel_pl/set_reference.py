"""Detached visible-L sample sets for cross-RX identity relation diagnostics."""
import math
import torch
from torch.nn import functional as F
from .receiver_reference import TEMPERATURE


def build_bank(features,labels,receivers,source_rxs,classes,role):
    rxs=tuple(source_rxs)
    if role!='L_s':raise ValueError('Only visible source L_s may build an identity bank')
    if (features.ndim!=2 or features.shape[1]!=160 or len(features)==0 or labels.shape!=(len(features),)
            or receivers.shape!=labels.shape or classes<2 or labels.min()<0 or labels.max()>=classes
            or len(rxs)<2 or len(set(rxs))!=len(rxs) or set(receivers.tolist())-set(rxs)
            or not bool(torch.isfinite(features).all())):
        raise ValueError('Aligned visible source features/classes/RX required')
    z=F.normalize(features.detach(),dim=1)
    groups=[[z[(labels==c)&(receivers==r)] for r in rxs] for c in range(classes)]
    if any(len(v)==0 for row in groups for v in row):raise ValueError('Each class/RX needs visible L support')
    size=max(len(v) for row in groups for v in row)
    bank=z.new_zeros(classes,len(rxs),size,160);valid=torch.zeros(bank.shape[:-1],device=z.device,dtype=torch.bool)
    for c,row in enumerate(groups):
        for j,v in enumerate(row):bank[c,j,:len(v)]=v;valid[c,j,:len(v)]=True
    return bank.detach(),valid


def mean_logits(features,references,receivers,source_rxs):
    rxs=tuple(source_rxs)
    if (features.ndim!=2 or features.shape[1]!=160 or references.ndim!=3 or references.shape[1:]!=(len(rxs),160)
            or len(rxs)<2 or len(set(rxs))!=len(rxs) or receivers.shape!=(len(features),)
            or set(receivers.tolist())-set(rxs) or references.requires_grad
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(references).all())):
        raise ValueError('Frozen mean references and aligned source features/RX required')
    keep=receivers[:,None]!=torch.tensor(rxs,device=receivers.device)[None,:]
    similarity=torch.einsum('nd,crd->ncr',F.normalize(features,dim=1),references)
    return (similarity*keep[:,None,:]).sum(2)/keep.sum(1)[:,None]/TEMPERATURE


def set_logits(features,bank,valid,receivers,source_rxs,inner_temperature):
    rxs=tuple(source_rxs)
    if (features.ndim!=2 or features.shape[1]!=160 or bank.ndim!=4 or bank.shape[1]!=len(rxs) or bank.shape[-1]!=160
            or bank.shape[0]<2 or bank.shape[2]<1 or valid.shape!=bank.shape[:-1] or valid.dtype!=torch.bool
            or not bool((valid.sum(-1)>0).all()) or receivers.shape!=(len(features),)
            or len(rxs)<2 or len(set(rxs))!=len(rxs) or set(receivers.tolist())-set(rxs)
            or bank.requires_grad or not math.isfinite(inner_temperature) or inner_temperature<=0
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(bank).all())):
        raise ValueError('Detached complete visible-L bank and aligned source features/RX required')
    sim=F.normalize(features,dim=1)@F.normalize(bank,dim=-1).reshape(-1,160).T
    sim=sim.reshape(len(features),*bank.shape[:-1])/inner_temperature
    log_average=torch.logsumexp(sim.masked_fill(~valid[None],-torch.inf),dim=-1)-valid.sum(-1).log()[None]
    score=inner_temperature*log_average
    keep=receivers[:,None]!=torch.tensor(rxs,device=receivers.device)[None,:]
    return (score*keep[:,None,:]).sum(2)/keep.sum(1)[:,None]/TEMPERATURE
