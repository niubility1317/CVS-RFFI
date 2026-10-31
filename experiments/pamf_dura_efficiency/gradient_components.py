"""Separate source head/auxiliary IQ derivatives and compare fixed local steps."""
import torch
from .exact_reference import feature_phi
from .soft_reference_tangent import ALPHA


def split_gradients(reader,iq,target,basis,*,retain_graph=False):
    if (iq.ndim!=3 or iq.shape[1:]!=(2,256) or not iq.requires_grad
        or target.shape!=(len(iq),10) or basis.shape!=(160,10)
        or reader.training or any(p.requires_grad for p in reader.parameters())):
        raise ValueError('Frozen eval F, source IQ and fixed source10 reference required')
    features,phi=feature_phi(reader,iq,basis)
    error=(phi-target.detach()).square()
    head=error[:,:5].sum(1);aux=error[:,5:].sum(1)
    gh=torch.autograd.grad((ALPHA*head).sum(),iq,retain_graph=True)[0].detach()
    ga=torch.autograd.grad((ALPHA*aux).sum(),iq,retain_graph=retain_graph)[0].detach()
    if not torch.isfinite(gh).all() or not torch.isfinite(ga).all():
        raise FloatingPointError('Nonfinite source derivative component')
    return dict(features=features.detach(),phi=phi.detach(),head_error=head.detach(),aux_error=aux.detach(),head_gradient=gh,aux_gradient=ga,
        joint_error=(head+aux) if retain_graph else (head+aux).detach())


def local_steps(restored,clean,head,aux):
    """Equal per-record displacement budgets, with physical IQ+bounded guidance.

    These use paired source clean and are diagnostic IQ oracles, not deployment
    operations or G optimization. A norm cap does not imply NMSE decreases.
    """
    with torch.no_grad():
        delta=restored-clean;power=clean.square().mean((1,2)).clamp_min(1e-4)
        IQ=2*delta/(512*power[:,None,None]);normIQ=IQ.flatten(1).norm(dim=1)
        budget=torch.minimum(.01*clean.flatten(1).norm(dim=1),.1*delta.flatten(1).norm(dim=1))
        result=dict(IQ_gradient=IQ,step_budget=budget)
        for name,gradient in [('head',head),('joint',head+aux)]:
            norm=gradient.flatten(1).norm(dim=1)
            weight=torch.clamp(normIQ/norm.clamp_min(1e-12),max=1)
            bounded=gradient*weight[:,None,None];direction=IQ+bounded
            length=direction.flatten(1).norm(dim=1)
            step=-direction*(budget/length.clamp_min(1e-12))[:,None,None]
            result.update({name+'_weight':weight,name+'_bounded_gradient':bounded,name+'_step':step,name+'_IQ':restored+step})
        return result


def cosine(a,b):
    a=a.flatten(1);b=b.flatten(1);den=a.norm(dim=1)*b.norm(dim=1)
    return (a*b).sum(1)/den.clamp_min(1e-12),den>1e-12
