import torch
from . import design as d
from experiments.pamf_flow_start_restore.models import FlowStartRestorer
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class EndpointRestorer(FlowStartRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:raise ValueError('Registered matched endpoint row required')
        super().__init__('zero_start',frozen);self.candidate=candidate

    def restore(self,received,noise=None):
        if noise is not None and (noise.shape!=received.shape or not torch.isfinite(noise).all()):
            raise ValueError('Explicit finite noise must match IQ even though sigma0=0')
        base,action,native,condition=self.context(received);point=torch.zeros_like(received)
        for k in range(d.SAMPLING_STEPS):
            time=received.new_full((len(received),),k/d.SAMPLING_STEPS)
            point=point+self.prior.field(point,condition,time)/d.SAMPLING_STEPS
        prior=base+SCALE*point;variance=native[:,None].expand(-1,256)
        output=pair(weighted_map(action,complex_iq(received),complex_iq(prior),variance,steps=6,initial=complex_iq(base)))
        return output,variance,action,base

    def training_loss(self,clean,received,generator=None):
        if clean.shape!=received.shape:raise ValueError('Matching legal source pair required')
        noise=torch.randn(clean.shape,dtype=clean.dtype,device=clean.device,generator=generator)
        time=torch.rand((len(clean),),dtype=clean.dtype,device=clean.device,generator=generator)
        power=clean.square().mean((1,2)).clamp_min(1e-4)
        shared=dict(flow_time=time,start_noise_power=torch.zeros_like(time))
        if self.candidate=='endpoint_four':
            output=self(received,noise)
            loss=(output-clean).square().mean((1,2))/power
            return loss,dict(shared,objective_nmse=loss,endpoint_iq_nmse=loss)
        base,_,_,condition=self.context(received);endpoint=(clean-base)/SCALE
        point=time[:,None,None]*endpoint;velocity=self.prior.field(point,condition,time)
        loss=SCALE**2*(velocity-endpoint).square().mean((1,2))/power
        estimate=base+SCALE*(point+(1-time[:,None,None])*velocity)
        return loss,dict(shared,objective_nmse=loss,teacher_path_endpoint_nmse=(estimate-clean).square().mean((1,2))/power)
