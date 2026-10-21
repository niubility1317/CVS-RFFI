"""Frozen two-stage arithmetic; four CG updates and no tensor-to-host branches."""
import torch
from experiments.pamf_r46_cuda_graph import program as original
from experiments.pamf_r65_inference.program import reused_map
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair


def outputs(model,received,*,reuse=True):
    if received.ndim!=3 or received.shape[1:]!=(2,256) or len(model.prior)!=2:
        raise ValueError('Original frozen two-stage IQ graph required')
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
        if reuse:
            current=pair(reused_map(action,complex_iq(prior),initial,native[:,None],raw))
        else:
            current=pair(original.weighted_map(action,observed,complex_iq(prior),variance,initial=initial))
        checks.append(torch.isfinite(current).all())
    checks.extend(torch.isfinite(v).all() for v in (received,base,action.kernel,action.main,action.image,variance))
    checks.append((variance>0).all())
    return current,variance,action,base,torch.stack(checks).all()
