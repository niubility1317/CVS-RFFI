"""Conditional flow whose trajectories start at a frozen restoration anchor."""
import torch
from torch import nn
from torch.nn import functional as F


def integrate(field,received,anchor,noise,*,steps=4,noise_scale=.02):
    if received.ndim!=3 or received.shape[1:]!=(2,256) or anchor.shape!=received.shape or noise.shape!=received.shape:
        raise ValueError('Aligned single-window received, anchor and noise required')
    if type(steps) is not int or steps<1:raise ValueError('Positive fixed integration count required')
    state=anchor.detach()+float(noise_scale)*noise
    for i in range(steps):
        t=received.new_full((len(received),),i/steps)
        state=state+field.velocity(state,received,t)/steps
    return state


class FixedAnchoredFlow(nn.Module):
    """Two fixed antithetic trajectories; requires only received and anchor."""
    def __init__(self,field):
        super().__init__();self.field=field
        generator=torch.Generator(device='cpu').manual_seed(2026110103)
        self.register_buffer('noise_template',torch.randn((1,2,256),generator=generator))

    def forward(self,received,anchor):
        noise=self.noise_template.expand_as(received)
        return .5*(integrate(self.field,received,anchor,noise)+integrate(self.field,received,anchor,-noise))


def source_objective(field,teacher,received,anchor,clean,noise,*,kl_weight=.1,trust_weight=.05):
    """Source-paired loss, frozen F0 teacher; no labels or test adaptation."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):raise ValueError('Frozen teacher required')
    a=integrate(field,received,anchor,noise);b=integrate(field,received,anchor,-noise)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    nmse=.5*((a-clean.detach()).square().mean((1,2))+(b-clean.detach()).square().mean((1,2)))/energy
    deviation=.5*((a-anchor.detach()).square().mean((1,2))+(b-anchor.detach()).square().mean((1,2)))/energy
    with torch.no_grad():
        features=teacher.features(clean);target=F.normalize(features,dim=1)
        logp=F.log_softmax(teacher.classify_features(features),dim=1);p=logp.exp()
    fa=teacher.features(a);fb=teacher.features(b);na=F.normalize(fa,dim=1);nb=F.normalize(fb,dim=1)
    anchor_error=.5*((1-(na*target).sum(1))+(1-(nb*target).sum(1)))
    agreement=1-(na*nb).sum(1)
    la=F.log_softmax(teacher.classify_features(fa),dim=1);lb=F.log_softmax(teacher.classify_features(fb),dim=1)
    kl=.5*((p*(logp-la)).sum(1)+(p*(logp-lb)).sum(1))
    loss=nmse+anchor_error+.25*agreement+kl_weight*kl+trust_weight*deviation
    return loss,dict(iq=nmse,feature=anchor_error,agreement=agreement,semantic=kl,deviation=deviation,path_a=a,path_b=b)
