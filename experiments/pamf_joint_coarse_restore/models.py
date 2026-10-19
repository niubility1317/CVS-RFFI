import torch
from . import design as d
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class JointCoarseRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:raise ValueError('Registered coarse-update arm required')
        super().__init__('unroll_unshared',frozen)
        self.candidate=candidate
        self.frozen.coarse.requires_grad_(candidate=='joint_coarse')

    def train(self,mode=True):
        super().train(mode)
        self.frozen.coarse.train(mode and self.candidate=='joint_coarse')
        return self

    def stage_outputs(self,received):
        if (received.ndim!=3 or received.shape[1:]!=(2,256) or not received.is_floating_point()
                or not torch.isfinite(received).all()):
            raise ValueError('Finite current received real IQ2x256 required')
        if self.candidate=='joint_coarse':
            # ActionRestorer.estimate intentionally detaches coarse: bypass that
            # boundary only here. Frozen estimator weights still pass input gradients.
            base=self.frozen.coarse(received)
            action,native=self.frozen.estimator(received,base)
        else:
            with torch.no_grad():base,action,native=self.frozen.estimate(received)
        variance=native[:,None].expand(-1,256);observed=complex_iq(received);current=base;stages=[]
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        for core in self.prior:
            innovation=pair(action.adjoint(observed-action(complex_iq(current))))
            innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
            prior=base+SCALE*core(torch.cat((received,base,current,innovation,logvar),1))
            current=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=complex_iq(current)))
            stages.append(current)
        return stages,variance,action,base
