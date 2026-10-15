import torch
from torch.nn import functional as F

def paired_margin(clean,satellite,labels,q,*,rho=.5,m_max=1.,scale=30.):
    if not 0<rho<1 or scale<=0:raise ValueError('Invalid margin retention or cosine scale')
    if clean.shape!=satellite.shape or clean.ndim!=2 or clean.shape[1]<2:
        raise ValueError('Paired classifier shapes differ')
    clean=clean/scale;satellite=satellite/scale
    reference=(clean.gather(1,labels[:,None])-clean).detach().clamp(0,m_max)
    margin=satellite.gather(1,labels[:,None])-satellite
    competitors=~F.one_hot(labels,clean.shape[1]).bool()
    valid=(clean.detach().argmax(1)==labels).to(clean.dtype)
    hinges=F.relu(rho*reference-margin)*competitors
    # All B samples stay in denominator. q affects this constraint alone.
    return (valid*q*hinges.sum(1)/(clean.shape[1]-1)).mean()

def soft_urban_ce(logits,teacher_prob,weak_mask):
    if teacher_prob.requires_grad or weak_mask.dtype!=torch.bool:raise ValueError('Detached source teacher required')
    if teacher_prob.shape!=logits.shape or weak_mask.shape!=(len(logits),):raise ValueError('SSL pair mismatch')
    loss=-(teacher_prob*logits.log_softmax(1)).sum(1)
    return loss[weak_mask].mean() if bool(weak_mask.any()) else logits.sum()*0
