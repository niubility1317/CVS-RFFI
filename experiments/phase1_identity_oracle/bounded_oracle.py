"""Bounded waveform optimization; source-label oracle diagnostic only."""
import torch
from torch.nn import functional as F

def oracle_refine(classifier,base,labels,epsilon,steps=16):
    if base.ndim!=3 or labels.shape!=(len(base),) or epsilon<0 or steps<=0:raise ValueError('Matched IQ and labels, nonnegative radius, positive steps required')
    if epsilon==0:return base.detach().clone()
    radius=epsilon*base.flatten(1).norm(dim=1).clamp_min(1e-8);delta=torch.zeros_like(base)
    for _ in range(steps):
        delta=delta.detach().requires_grad_(True);loss=F.cross_entropy(classifier(base+delta),labels,reduction='sum');grad=torch.autograd.grad(loss,delta)[0];length=grad.flatten(1).norm(dim=1).clamp_min(1e-12);proposal=delta.detach()-.5*radius[:,None,None]*grad/length[:,None,None];size=proposal.flatten(1).norm(dim=1).clamp_min(1e-12);delta=proposal*torch.minimum(torch.ones_like(size),radius/size)[:,None,None]
    return (base+delta).detach()
