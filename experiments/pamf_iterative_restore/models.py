import torch
from . import design as d
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class IterativeRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:
            raise ValueError('Registered iterative control required')
        # Exactly the R17 unshared parameter construction and RNG draws.
        super().__init__('unroll_unshared',frozen)
        self.candidate=candidate

    def stage_outputs(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):
            raise ValueError('Only current received IQ2x256 required')
        spec=d.CANDIDATES[self.candidate]
        with torch.no_grad():
            base,action,native=self.frozen.estimate(received)
        variance=native[:,None].expand(-1,256)
        observed=complex_iq(received);current=base;stages=[]
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        for k,core in enumerate(self.prior):
            innovation=pair(action.adjoint(observed-action(complex_iq(current))))
            innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
            inputs=torch.cat((received,base,current,innovation,logvar),1)
            anchor=current if spec['anchor']=='current' else base
            prior=anchor+SCALE*core(inputs)
            steps=spec['CG_schedule'][k]
            current=(pair(weighted_map(action,observed,complex_iq(prior),variance,
                steps=steps,initial=complex_iq(current))) if steps else prior)
            stages.append(current)
        return stages,variance,action,base
