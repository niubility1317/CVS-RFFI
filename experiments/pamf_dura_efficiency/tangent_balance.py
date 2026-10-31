"""Direction-preserving per-step source guidance balance, with detached weights.

The returned scalar is a step-local weighted surrogate. Its weight is held
constant in backward; it is not the total derivative of a globally fixed
energy whose weights are recomputed at perturbed IQ values.
"""
import math
import torch
from .clean_tangent import tangent_loss


def balanced_tangent_loss(restored,clean,jacobian,alpha):
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Positive finite fixed guidance coefficient required')
    raw=tangent_loss(restored,clean,jacobian)
    with torch.no_grad():
        delta=restored.detach()-clean.detach();jac=jacobian.detach()
        projected=(jac*delta[:,None]).sum((2,3));gradient=2*alpha*(jac*projected[:,:,None,None]).sum(1)
        energy=512*clean.detach().square().mean((1,2)).clamp_min(1e-4)
        IQ_norm=2*delta.flatten(1).norm(dim=1)/energy;guide_norm=gradient.flatten(1).norm(dim=1)
        weight=torch.minimum(torch.ones_like(IQ_norm),IQ_norm/guide_norm.clamp_min(1e-12))
        pressure=guide_norm/IQ_norm.clamp_min(1e-12)
    return raw*weight,dict(weight=weight,limited=guide_norm>IQ_norm,raw_pressure=pressure,bounded_pressure=pressure*weight)
