import math
import torch
from torch import nn
from experiments.pamf_reception_restore.models import Core, Encoder, MultiscaleDiffusion, MultiscaleFlow

NAMES=('residual_deterministic','residual_diffusion','residual_flow')


class FrozenCoarse:
    def attach_coarse(self,coarse,scale):
        if not math.isfinite(float(scale)) or float(scale)<=0:
            raise ValueError('One finite positive source-training scale required')
        self.coarse=coarse.requires_grad_(False).eval()
        self.register_buffer('residual_scale',torch.tensor(float(scale),dtype=torch.float32))

    def train(self,mode=True):
        super().train(mode)
        self.coarse.eval()
        return self

    def context(self,received):
        with torch.no_grad():base=self.coarse(received)
        return base,torch.cat((received,base),1)


class ResidualDeterministic(FrozenCoarse,nn.Module):
    def __init__(self,coarse,scale):
        super().__init__()
        self.attach_coarse(coarse,scale)
        self.correction=Core(incoming=4)

    def forward(self,received):
        base,condition=self.context(received)
        return base+self.residual_scale*self.correction(condition)


class ResidualDiffusion(FrozenCoarse,MultiscaleDiffusion):
    def __init__(self,coarse,scale):
        super().__init__()
        self.conditioner=Encoder(incoming=4)
        self.attach_coarse(coarse,scale)

    def training_prediction(self,clean,received,generator=None,return_diagnostics=False):
        base,condition=self.context(received)
        residual=(clean-base)/self.residual_scale
        values=super().training_prediction(residual,condition,generator,return_diagnostics)
        return (base+self.residual_scale*values[0],*values[1:])

    def forward(self,received,noise):
        if noise is None or noise.shape!=received.shape:
            raise ValueError('Noise must match only the current observed IQ')
        base,context=self.context(received)
        condition=self.conditioner(context)
        point=noise
        indices=torch.linspace(self.steps-1,0,self.sampling_steps).round().long().tolist()
        for offset,index in enumerate(indices):
            time=torch.full((len(point),),index,dtype=torch.long,device=point.device)
            a,b=self.alpha_bar[index].sqrt(),(1-self.alpha_bar[index]).sqrt()
            velocity=self.field(point,condition,time)
            estimate,epsilon=a*point-b*velocity,b*point+a*velocity
            if offset==len(indices)-1:point=estimate
            else:
                following=self.alpha_bar[indices[offset+1]]
                point=following.sqrt()*estimate+(1-following).sqrt()*epsilon
        return base+self.residual_scale*point


class ResidualFlow(FrozenCoarse,MultiscaleFlow):
    def __init__(self,coarse,scale):
        super().__init__()
        self.network.conditioner=Encoder(incoming=4)
        self.attach_coarse(coarse,scale)

    def training_prediction(self,clean,received,generator=None,return_diagnostics=False):
        base,context=self.context(received)
        residual=(clean-base)/self.residual_scale
        noise=torch.randn(clean.shape,dtype=clean.dtype,device=clean.device,generator=generator)
        start=self.sigma*noise
        time=torch.rand((len(clean),),device=clean.device,generator=generator)
        point=(1-time[:,None,None])*start+time[:,None,None]*residual
        velocity=self.network.field(point,self.network.conditioner(context),time*self.network.steps)
        error=(velocity-(residual-start)).square().mean((1,2))
        estimate=base+self.residual_scale*(point+(1-time[:,None,None])*velocity)
        values=(estimate,error.mean())
        if return_diagnostics:
            return (*values,dict(time_fraction=time,velocity_error=error))
        return values

    def forward(self,received,noise):
        if noise is None or noise.shape!=received.shape:
            raise ValueError('Noise must match only the current observed IQ')
        base,context=self.context(received)
        condition=self.network.conditioner(context)
        point=self.sigma*noise
        for index in range(self.sampling_steps):
            time=torch.full((len(point),),index/self.sampling_steps*self.network.steps,
                            dtype=point.dtype,device=point.device)
            point=point+self.network.field(point,condition,time)/self.sampling_steps
        return base+self.residual_scale*point


def build(name,coarse,scale):
    classes=dict(zip(NAMES,(ResidualDeterministic,ResidualDiffusion,ResidualFlow)))
    if name not in classes:raise ValueError('Registered residual corrector required')
    return classes[name](coarse,scale)
