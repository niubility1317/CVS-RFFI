"""Exact frozen-F source guidance with one F reverse pass and a step-local cap."""
import math
import torch
from torch.nn import functional as F
from .soft_reference_tangent import ALPHA


def feature_phi(reader,iq,basis):
    features=reader.features(iq)
    return features,F.normalize(features,dim=1,eps=1e-4)@basis.detach()*4


def exact_guide(reader,restored,clean,target,basis,alpha=ALPHA):
    if (restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape
        or target.shape!=(len(restored),10) or basis.shape!=(160,10)
        or not math.isfinite(alpha) or alpha<=0 or reader.training
        or any(p.requires_grad for p in reader.parameters())):
        raise ValueError('Frozen eval F, source IQ, fixed source10 target/basis and positive alpha required')
    _,phi=feature_phi(reader,restored,basis)
    error=(phi-target.detach()).square().sum(1)
    raw_gradient=torch.autograd.grad((alpha*error).sum(),restored)[0].detach()
    if not torch.isfinite(raw_gradient).all():raise FloatingPointError('Nonfinite exact F input gradient')
    with torch.no_grad():
        norm=raw_gradient.flatten(1).norm(dim=1)
        IQnorm=2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*clean.detach().square().mean((1,2)).clamp_min(1e-4))
        weight=torch.clamp(IQnorm/norm.clamp_min(1e-12),max=1)
        gradient=raw_gradient*weight[:,None,None]
        value=alpha*error.detach()*weight
    injection=(restored*gradient).flatten(1).sum(1)
    # The F graph was consumed above. This zero-valued term injects exactly the
    # detached weighted IQ gradient into the still-live G graph, without a
    # second F backward or derivatives through this step's weight.
    surrogate=value+(injection-injection.detach())
    return surrogate,dict(feature_error=error.detach(),phi=phi.detach(),weight=weight,
        guide_value=value,raw_gradient=raw_gradient,weighted_gradient=gradient,IQ_gradient_norm=IQnorm,
        raw_pressure=norm/IQnorm.clamp_min(1e-12),pressure_defined=IQnorm>1e-12,
        bounded_pressure=gradient.flatten(1).norm(dim=1)/IQnorm.clamp_min(1e-12))
