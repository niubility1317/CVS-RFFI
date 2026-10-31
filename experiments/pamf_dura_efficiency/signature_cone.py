"""Source-only common IQ/signature descent; no G update or inference oracle."""
import torch
from .signature_iq import signature_features,project


def common_step(value,clean,feature_loss,feature_gradient):
    """Bisect normalized derivatives and limit to the exact IQ line minimum.

    For a nonzero common direction, both current first derivatives are
    nonnegative. The finite step decreases the quadratic IQ error before
    projection; no analogous nonlinear signature guarantee is asserted.
    """
    if (value.ndim!=3 or value.shape!=clean.shape or feature_gradient.shape!=value.shape
            or feature_loss.shape!=(len(value),)):
        raise ValueError('Matched source IQ, scalar signature errors and input derivatives required')
    with torch.no_grad():
        error=value-clean;power=clean.square().mean((1,2)).clamp_min(1e-4)
        IQ=2*error/(value[0].numel()*power[:,None,None])
        a=IQ.flatten(1);b=feature_gradient.flatten(1);na=a.norm(dim=1);nb=b.norm(dim=1)
        defined=(na>1e-12)&(nb>1e-12)
        direction=a/na.clamp_min(1e-12)[:,None]+b/nb.clamp_min(1e-12)[:,None]
        length2=direction.square().sum(1);dotIQ=(a*direction).sum(1);dotF=(b*direction).sum(1)
        active=defined&(length2>1e-12)&(dotIQ>0)&(dotF>0)
        # Exact minimizer of ||value-s*direction-clean||^2, plus local F shrink.
        sIQ=(error.flatten(1)*direction).sum(1)/length2.clamp_min(1e-12)
        sF=feature_loss.detach()/dotF.clamp_min(1e-20)
        scale=torch.minimum(sIQ.clamp_min(0),sF.clamp_min(0))*active
        step=(-scale[:,None]*direction).reshape_as(value)
        cosine=(a*b).sum(1)/(na*nb).clamp_min(1e-20)
        weight=(.5*na/nb.clamp_min(1e-12)).clamp_max(1.)
        baseline=a+weight[:,None]*b
        return step,dict(cosine=cosine,defined=defined.float(),active=active.float(),
            baseline_feature_first_order=(b*baseline).sum(1),common_IQ_first_order=dotIQ,
            common_feature_first_order=dotF)


def common_signature_steps(reader,anchor,reference,clean):
    value=anchor.detach().clone();anchor_rms=anchor.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
    history=[];geometry=[]
    for _ in range(4):
        value.requires_grad_(True);signature=signature_features(reader,value)
        loss=(signature-reference.detach()).square().sum(1)/reference.detach().square().sum(1).clamp_min(1e-4)
        gradient=torch.autograd.grad(loss.sum(),value)[0]
        if not torch.isfinite(loss).all() or not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite source SIG gradient')
        step,info=common_step(value.detach(),clean.detach(),loss.detach(),gradient.detach())
        info.update(IQ_before=value.detach(),feature_gradient=gradient.detach())
        history.append(loss.detach());geometry.append(info)
        with torch.no_grad():
            rms=step.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-12)
            step=step*(.0025*anchor_rms/rms).clamp_max(1.)
            value=project(anchor,value.detach()+step-anchor)
            initial=(anchor-clean).square().sum((1,2));current=(value-clean).square().sum((1,2))
            if (current>initial*(1+3e-5)+1e-10).any():raise FloatingPointError('Source IQ anchor risk bound failed')
    return value.detach(),torch.stack(history),{key:torch.stack([g[key] for g in geometry]) for key in geometry[0]}
