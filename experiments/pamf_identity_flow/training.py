"""Source-only conditional-flow training primitives.

G10 uses exact source L_s physical-ID pairing. F0 guidance transfers its
learned source identity information, without passing TX labels into G.
"""
import numpy as np
import torch
from torch import nn

from .model import source_flow_loss, frozen_identity_consistency


def align_g10(source_ids, roles, g10_ids):
    source_ids=np.asarray(source_ids);roles=np.asarray(roles);g10_ids=np.asarray(g10_ids)
    if source_ids.ndim!=1 or roles.shape!=source_ids.shape or g10_ids.ndim!=1:
        raise ValueError('One-dimensional aligned physical IDs and roles required')
    source=source_ids[roles=='L_s']
    if (len(set(source_ids.tolist()))!=len(source_ids) or len(set(g10_ids.tolist()))!=len(g10_ids)
            or set(source.tolist())!=set(g10_ids.tolist())):
        raise ValueError('G10 IDs must exactly equal the distinct allowed source L_s IDs')
    lookup={value:index for index,value in enumerate(source_ids)}
    return np.asarray([lookup[value] for value in g10_ids],dtype=np.int64)


def mix_epoch(clean, received, g10_clean, g10_received, order, g10_order):
    """Half R07, quarter G10 source L_s, quarter clean, with aligned pairs."""
    n=len(order)
    if n%4 or len(g10_order)!=n//4:
        raise ValueError('Fixed half/quarter/quarter source schedule required')
    half,quarter=n//2,n//4
    x=np.array(clean[order],dtype=np.float32,copy=True)
    y=np.array(received[order],dtype=np.float32,copy=True)
    x[half:half+quarter]=g10_clean[g10_order]
    y[half:half+quarter]=g10_received[g10_order]
    y[half+quarter:]=x[half+quarter:]
    return x,y,dict(r07=half,g10=quarter,clean=quarter)


def objective(model, teacher, clean, received, time, noise_a, noise_b):
    """Two matching paths with fixed teacher anchor/agreement coefficients."""
    loss_a,end_a=source_flow_loss(model,clean,received,time,noise_a)
    loss_b,end_b=source_flow_loss(model,clean,received,time,noise_b)
    anchor,agreement=frozen_identity_consistency(teacher,clean,end_a,end_b)
    flow=.5*(loss_a+loss_b)
    identity=anchor+.25*agreement
    return flow+identity,dict(flow=flow,identity=identity,anchor=anchor,agreement=agreement)


class FixedAntitheticFlow(nn.Module):
    """Fixed two-path mean; identical template for every observation, no selection."""
    def __init__(self, flow):
        super().__init__()
        self.flow=flow
        generator=torch.Generator(device='cpu').manual_seed(2026110103)
        self.register_buffer('noise_template',torch.randn((1,2,256),generator=generator))

    def forward(self, received):
        noise=self.noise_template.expand_as(received)
        a=self.flow.integrate(received,noise,noise_scale=.02,steps=4)
        b=self.flow.integrate(received,-noise,noise_scale=.02,steps=4)
        return .5*(a+b)
