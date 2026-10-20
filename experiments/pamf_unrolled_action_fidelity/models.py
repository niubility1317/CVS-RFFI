import torch
from .design import ROWS
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


def trajectory(model,received,base,action,variance):
    if (variance.shape!=(len(received),) or not torch.isfinite(variance).all() or (variance<=0).any()):
        raise ValueError('Positive current-window real variance proxy required')
    observed=complex_iq(received);current=base
    expanded=variance[:,None].expand(-1,256)
    logvar=variance.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
    for core in model.prior:
        innovation=pair(action.adjoint(observed-action(complex_iq(current))))
        innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
        prior=base+SCALE*core(torch.cat((received,base,current,innovation,logvar),1))
        current=pair(weighted_map(action,observed,complex_iq(prior),expanded,steps=2,initial=complex_iq(current)))
    return current,prior


@torch.no_grad()
def counterfactuals(model,received,known_action,simulator_variance):
    if (received.ndim!=3 or received.shape[1:]!=(2,256) or not received.is_floating_point()
            or not torch.isfinite(received).all()):raise ValueError('Finite current source IQ2x256 required')
    if getattr(model,'candidate',None)!='unroll_unshared' or len(model.prior)!=3:
        raise ValueError('Original unshared three-stage R17 model required')
    base,estimated,variance=model.frozen.estimate(received)
    native,prior=trajectory(model,received,base,estimated,variance)
    values=dict(native_estimated=native,
        known_action_estvar=trajectory(model,received,base,known_action,variance)[0],
        est_action_simvar=trajectory(model,received,base,estimated,simulator_variance)[0],
        known_action_simvar=trajectory(model,received,base,known_action,simulator_variance)[0],prior_only=prior)
    if tuple(values)!=ROWS or any(v.shape!=received.shape or not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Finite complete registered source counterfactuals required')
    return values
