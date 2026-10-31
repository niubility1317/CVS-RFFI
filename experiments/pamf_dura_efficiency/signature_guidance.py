"""SOURCE-only original TripleSIG guidance with an IQ-gradient ceiling."""
import torch
from .signature_iq import signature_features

CAP=.5


def signature_guide(reader,restored,clean,target):
    if (restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape
        or target.shape!=(len(restored),160)):
        raise ValueError('Paired source IQ and actual 160D clean signature required')
    signature=signature_features(reader,restored)
    error=(signature-target.detach()).square().sum(1)/target.detach().square().sum(1).clamp_min(1e-4)
    raw=torch.autograd.grad(error.sum(),restored)[0].detach()
    if not torch.isfinite(raw).all() or not torch.isfinite(error).all():
        raise FloatingPointError('Nonfinite source SIG input gradient')
    with torch.no_grad():
        norm=raw.flatten(1).norm(dim=1)
        iqnorm=2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*clean.detach().square().mean((1,2)).clamp_min(1e-4))
        weight=(CAP*iqnorm/norm.clamp_min(1e-12)).clamp_max(1.)
        gradient=raw*weight[:,None,None];value=error.detach()*weight
    injection=(restored*gradient).flatten(1).sum(1)
    # Exactly one F input VJP. The normal G backward receives only this detached
    # input derivative; it does not differentiate the cap or traverse F again.
    surrogate=value+(injection-injection.detach())
    return surrogate,dict(feature_error=error.detach(),weight=weight,raw_gradient=raw,weighted_gradient=gradient,
        IQ_gradient_norm=iqnorm,raw_pressure=norm/iqnorm.clamp_min(1e-12),pressure_defined=iqnorm>1e-12,
        bounded_pressure=gradient.flatten(1).norm(dim=1)/iqnorm.clamp_min(1e-12))
