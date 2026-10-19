import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_reception_restore.models import Core,mirror_spectrum
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class Stage(nn.Module):
    def __init__(self,temporal,auxiliary):
        super().__init__();self.temporal=temporal;self.auxiliary=auxiliary


def normalized_innovation(action,observed,current):
    value=pair(action.adjoint(observed-action(complex_iq(current))))
    return value/(value.square().mean((1,2),keepdim=True)+1e-6).sqrt()


def auxiliary_inputs(signals,logvar,frequency):
    if len(signals)!=4 or any(s.shape!=signals[0].shape for s in signals):
        raise ValueError('Four matching current IQ signals required')
    values=[complex_iq(s) for s in signals]
    if frequency:values=[torch.fft.fft(v,norm='ortho') for v in values]
    mirrors=[mirror_spectrum(v) if frequency else v.conj() for v in values]
    return torch.cat([*[pair(v) for v in values],*[pair(v) for v in mirrors],logvar],1)


class AlternatingRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:raise ValueError('Registered alternating prior required')
        super().__init__('unroll_unshared',frozen)
        auxiliary=Core(incoming=17,widths=d.AUX_WIDTHS)
        self.prior=nn.ModuleList(Stage(temporal,copy.deepcopy(auxiliary)) for temporal in self.prior)
        self.candidate=candidate

    def stage_outputs(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):
            raise ValueError('Only current received IQ2x256 required')
        with torch.no_grad():base,action,native=self.frozen.estimate(received)
        variance=native[:,None].expand(-1,256);observed=complex_iq(received);current=base;stages=[]
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        frequency=self.candidate=='time_frequency'
        for stage in self.prior:
            innovation=normalized_innovation(action,observed,current)
            u=base+SCALE*stage.temporal(torch.cat((received,base,current,innovation,logvar),1))
            refreshed=normalized_innovation(action,observed,u)
            correction=stage.auxiliary(auxiliary_inputs((received,base,u,refreshed),logvar,frequency))
            if frequency:correction=pair(torch.fft.ifft(complex_iq(correction),norm='ortho'))
            prior=u+SCALE*correction
            current=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=complex_iq(current)))
            stages.append(current)
        return stages,variance,action,base
