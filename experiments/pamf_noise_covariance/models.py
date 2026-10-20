import math
import torch
from . import design as d
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_restoration.models import complex_iq,pair


def relative_pseudo(main,image):
    if (main.ndim!=2 or main.shape!=image.shape or not main.is_complex() or not image.is_complex()
            or not torch.isfinite(main).all() or not torch.isfinite(image).all()):
        raise ValueError('Finite matching native complex main/image coefficients required')
    power=main.abs().square()+image.abs().square()
    if (power<=0).any():raise ValueError('Nonzero native noise transformation required')
    q=2*main*image/power
    if (q.abs()>=1).any():raise ValueError('Positive definite IQ covariance required')
    return q


def _weights(value,variance,q):
    if (value.ndim!=2 or not value.is_complex() or not torch.isfinite(value).all()
            or variance.shape!=(len(value),) or not variance.is_floating_point()
            or not torch.isfinite(variance).all() or (variance<=0).any()
            or q.shape!=value.shape or not q.is_complex() or not torch.isfinite(q).all() or (q.abs()>=1).any()):
        raise ValueError('Finite IQ with positive scalar variance and nonsingular matching IQ block required')
    return variance[:,None].reciprocal()/(1-q.abs().square())


def precision(value,variance,q):return _weights(value,variance,q)*(value-q*value.conj())


def block_map(action,observed,prior,variance,q,prior_variance=SCALE**2,steps=2,*,initial=None):
    weights=_weights(observed,variance,q)
    if (prior.shape!=observed.shape or not prior.is_complex() or not torch.isfinite(prior).all()
            or not math.isfinite(prior_variance) or prior_variance<=0 or type(steps)is not int or steps<1):
        raise ValueError('Finite matching prior and positive regularization/CG budget required')
    def apply(value):return weights*(value-q*value.conj())
    rhs=action.adjoint(apply(observed-action(prior)))
    if initial is None:delta=torch.zeros_like(prior);residual=rhs
    else:
        if initial.shape!=prior.shape or not initial.is_complex() or not torch.isfinite(initial).all():raise ValueError('Finite matched warm start required')
        delta=initial-prior;residual=rhs-action.adjoint(apply(action(delta)))-delta/prior_variance
    direction=residual;squared=real_dot(residual,residual)
    for _ in range(steps):
        normal=action.adjoint(apply(action(direction)))+direction/prior_variance
        alpha=squared/real_dot(direction,normal).clamp_min(1e-20)
        delta=delta+alpha[:,None]*direction;residual=residual-alpha[:,None]*normal
        following=real_dot(residual,residual);direction=residual+(following/squared.clamp_min(1e-20))[:,None]*direction;squared=following
    return prior+delta


def trajectory(model,received,base,action,variance,q):
    observed=complex_iq(received);current=base
    logvar=variance.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
    for core in model.prior:
        innovation=pair(action.adjoint(observed-action(complex_iq(current))))
        innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
        prior=base+SCALE*core(torch.cat((received,base,current,innovation,logvar),1))
        current=pair(block_map(action,observed,complex_iq(prior),variance,q,steps=2,initial=complex_iq(current)))
    return current


@torch.no_grad()
def counterfactuals(model,received,known_action,simulator_variance):
    if (received.ndim!=3 or received.shape[1:]!=(2,256) or not received.is_floating_point() or not torch.isfinite(received).all()
            or getattr(model,'candidate',None)!='unroll_unshared' or len(model.prior)!=3):
        raise ValueError('Finite source IQ and original three unshared priors required')
    base,estimated,_=model.frozen.estimate(received)
    q=relative_pseudo(known_action.main,known_action.image)
    values=dict(estimated_covariance=trajectory(model,received,base,estimated,simulator_variance,q),
        known_covariance=trajectory(model,received,base,known_action,simulator_variance,q))
    if tuple(values)!=d.ROWS or any(value.shape!=received.shape or not torch.isfinite(value).all() for value in values.values()):
        raise ValueError('Finite complete registered covariance outputs required')
    return values
