import torch
from .design import ROWS
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_uncertainty_restore.models import weighted_map
from experiments.pamf_restoration.models import complex_iq,pair


def counterfactuals(model,received,known_action,simulator_variance):
    if received.ndim!=3 or received.shape[1:]!=(2,256):
        raise ValueError('Only current source observation IQ2x256 required')
    if (simulator_variance.shape!=(len(received),) or not torch.isfinite(simulator_variance).all()
            or (simulator_variance<=0).any()):
        raise ValueError('Positive matched simulator variance proxy required')
    base,estimated_action,estimated_variance=model.frozen.estimate(received)
    prior=base+SCALE*model.prior(torch.cat((received,base),1))
    observed=complex_iq(received);prior_c=complex_iq(prior)
    values=dict(prior_only=prior,
        known_action_estvar=pair(weighted_map(known_action,observed,prior_c,estimated_variance)),
        est_action_simvar=pair(weighted_map(estimated_action,observed,prior_c,simulator_variance)),
        known_action_simvar=pair(weighted_map(known_action,observed,prior_c,simulator_variance)),
        coarse_known_action_simvar=pair(weighted_map(known_action,observed,complex_iq(base),simulator_variance)))
    if tuple(values)!=ROWS or any(x.shape!=received.shape or not torch.isfinite(x).all() for x in values.values()):
        raise ValueError('All finite preregistered counterfactual outputs required')
    return values
