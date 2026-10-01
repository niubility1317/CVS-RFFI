"""Packet-local bounded attention pooling, with the original extractor geometry."""
import torch
from torch import nn
from experiments.cvs_residual_identity.model import build as residual_build
VARIANTS=('attentive_mean','attentive_moments')

class AttentivePool(nn.Module):
    def __init__(self,channels,moments=False):
        super().__init__()
        self.score=nn.Conv1d(channels,1,1,bias=False)
        nn.init.zeros_(self.score.weight)
        self.mix=nn.Parameter(torch.zeros(1,channels,1)) if moments else None

    def weights(self,x):
        # Statistics are strictly within each packet and feature channel.
        center=x-x.mean(-1,keepdim=True)
        normalized=center/(center.square().mean(-1,keepdim=True)+1e-5).sqrt()
        # Positive convex weights; max/min ratio <= exp(2), no hard selection.
        return torch.softmax(torch.tanh(self.score(normalized)),dim=-1)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=self.score.in_channels:raise ValueError('Native feature-map schema changed')
        w=self.weights(x);mean=(w*x).sum(-1,keepdim=True)
        if self.mix is None:return mean
        deviation=((w*(x-mean).square()).sum(-1,keepdim=True)+1e-6).sqrt()
        return mean+self.mix.tanh()*deviation

def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered attentive identity variant')
    model=residual_build('residual_fusion');b=model.id_backbone
    for pool,projection in (('t_pool','t_proj'),('f_pool','f_proj'),('pa_pool','pa_proj')):
        p=getattr(b,projection)
        linear=p if isinstance(p,nn.Linear) else next(m for m in p.modules() if isinstance(m,nn.Linear))
        setattr(b,pool,AttentivePool(linear.in_features,variant=='attentive_moments'))
    return model
