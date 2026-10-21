"""Same frozen modules and six CG updates; no tensor-to-host branches."""
import torch
from experiments.pamf_r46_cuda_graph import program as original
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_restoration.models import complex_iq, pair


def reused_map(action, prior, initial, variance, raw_innovation):
    # variance is Bx1: one real scalar per observation, never a time-varying
    # diagonal. It therefore commutes with the per-observation real-linear A*.
    if variance.shape != (len(prior),1) or variance.is_complex():
        raise ValueError('One real variance scalar per window required')
    weight=variance.reciprocal()
    delta=initial-prior
    residual=weight*raw_innovation-delta/(SCALE**2)
    direction=residual
    squared=real_dot(residual,residual)
    for step in range(2):
        normal=action.adjoint(weight*action(direction))+direction/(SCALE**2)
        alpha=squared/real_dot(direction,normal).clamp_min(1e-20)
        delta=delta+alpha[:,None]*direction
        if step == 0:
            residual=residual-alpha[:,None]*normal
            following=real_dot(residual,residual)
            direction=residual+(following/squared.clamp_min(1e-20))[:,None]*direction
            squared=following
    return prior+delta


def outputs(model, received, *, reuse=False):
    if received.ndim!=3 or received.shape[1:]!=(2,256) or len(model.prior)!=3:
        raise ValueError('Original R65 three-stage IQ graph required')
    if not reuse:
        current,variance,action,base=original.outputs(model,received)
        checks=[torch.isfinite(current).all()]
    else:
        base=model.frozen.coarse(received)
        action,native=original.estimate(model.frozen.estimator,received,base)
        variance=native[:,None].expand(-1,256)
        observed,current=complex_iq(received),base
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        checks=[]
        for core in model.prior:
            initial=complex_iq(current)
            raw=action.adjoint(observed-action(initial))
            innovation=pair(raw)
            innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
            prior=base+SCALE*core(torch.cat((received,base,current,innovation,logvar),1))
            current=pair(reused_map(action,complex_iq(prior),initial,native[:,None],raw))
            checks.append(torch.isfinite(current).all())
    # Preserve error detection while avoiding twelve host synchronizations.
    checks.extend(torch.isfinite(v).all() for v in (received,base,action.kernel,action.main,action.image,variance))
    checks.append((variance>0).all())
    return current,variance,action,base,torch.stack(checks).all()
