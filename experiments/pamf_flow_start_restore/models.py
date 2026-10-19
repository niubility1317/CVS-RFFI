import math
import torch
from torch import nn
from . import design as d
from experiments.pamf_reception_restore.models import Core,Encoder
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class VelocityField(nn.Module):
    def __init__(self):
        super().__init__();self.conditioner=Encoder(incoming=5);self.network=Core(conditional=True)
        self.time_encoder=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,64))

    def field(self,point,condition,time):
        frequency=torch.logspace(0,3,16,device=point.device,dtype=point.dtype)
        angle=time[:,None]*frequency[None]*math.pi
        embedded=self.time_encoder(torch.cat((angle.sin(),angle.cos()),-1))
        return self.network(point,condition,embedded)


class FlowStartRestorer(nn.Module):
    def __init__(self,candidate,frozen):
        super().__init__()
        if candidate not in d.ROWS:raise ValueError('Registered residual flow start required')
        self.candidate=candidate;self.frozen=frozen.requires_grad_(False).eval();self.prior=VelocityField()

    def train(self,mode=True):super().train(mode);self.frozen.eval();return self

    def context(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):raise ValueError('Current received IQ2x256 required')
        with torch.no_grad():base,action,native=self.frozen.estimate(received)
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        condition=self.prior.conditioner(torch.cat((received,base,logvar),1))
        return base,action,native,condition

    def training_loss(self,clean,received,generator=None):
        if clean.shape!=received.shape:raise ValueError('Matching source clean/received pair required')
        base,_,_,condition=self.context(received)
        # Consume both random streams in both rows, including sigma0=0.
        noise=torch.randn(clean.shape,dtype=clean.dtype,device=clean.device,generator=generator)
        time=torch.rand((len(clean),),dtype=clean.dtype,device=clean.device,generator=generator)
        start=d.SIGMA[self.candidate]*noise;endpoint=(clean-base)/SCALE
        point=(1-time[:,None,None])*start+time[:,None,None]*endpoint
        target=endpoint-start;velocity=self.prior.field(point,condition,time)
        power=clean.square().mean((1,2)).clamp_min(1e-4)
        error=SCALE**2*(velocity-target).square().mean((1,2))/power
        estimate=base+SCALE*(point+(1-time[:,None,None])*velocity)
        diagnostic=(estimate-clean).square().mean((1,2))/power
        return error,dict(velocity_nmse=error,teacher_path_endpoint_nmse=diagnostic,flow_time=time,start_noise_power=start.square().mean((1,2)))

    def restore(self,received,noise=None):
        if noise is None:
            if d.SIGMA[self.candidate]:raise ValueError('Stochastic inference requires explicit independent noise')
            noise=torch.zeros_like(received)
        if noise.shape!=received.shape or not torch.isfinite(noise).all():raise ValueError('Finite noise matching current IQ required')
        base,action,native,condition=self.context(received);point=d.SIGMA[self.candidate]*noise
        for k in range(d.SAMPLING_STEPS):
            time=received.new_full((len(received),),k/d.SAMPLING_STEPS)
            point=point+self.prior.field(point,condition,time)/d.SAMPLING_STEPS
        prior=base+SCALE*point;variance=native[:,None].expand(-1,256)
        output=pair(weighted_map(action,complex_iq(received),complex_iq(prior),variance,steps=6,initial=complex_iq(base)))
        return output,variance,action,base

    def forward(self,received,noise=None):return self.restore(received,noise)[0]
