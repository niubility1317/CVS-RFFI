"""Conditional IQ flow and frozen-identity source losses.

This is a research architecture, not a calibrated posterior or a deployment
claim. Inference uses only one received window and explicitly supplied noise;
several draws are computational variants of the same observation, not shots.
"""
import torch
from torch import nn
from torch.nn import functional as F


def flow_path(clean, received, time, noise, noise_scale):
    """Linear source pairing from noisy received to paired source clean IQ."""
    if clean.shape != received.shape or noise.shape != received.shape or received.ndim != 3 or received.shape[1:] != (2,256):
        raise ValueError('Aligned clean/received/noise IQ2x256 required')
    if time.shape != (len(received),) or not bool(((time >= 0) & (time <= 1)).all()):
        raise ValueError('One interpolation time in [0,1] per source window required')
    start = received + float(noise_scale) * noise
    fraction = time[:,None,None]
    return (1-fraction)*start + fraction*clean, clean-start


class IdentityConditionalFlow(nn.Module):
    """Full-window attention velocity field with fixed-step integration."""
    def __init__(self, width=64, layers=3, heads=4):
        super().__init__()
        self.input = nn.Conv1d(4,width,4,stride=4)
        self.position = nn.Parameter(torch.randn(1,64,width)*.01)
        self.time = nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,width))
        block = nn.TransformerEncoderLayer(width,heads,4*width,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True)
        self.field = nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.output = nn.Linear(width,8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def velocity(self, state, received, time):
        if state.shape != received.shape or received.ndim != 3 or received.shape[1:] != (2,256):
            raise ValueError('Current state and one received IQ2x256 required')
        if time.shape != (len(received),):
            raise ValueError('One flow time per received window required')
        angles = time[:,None]*time.new_tensor([1.,2.])[None,:]*torch.pi
        temporal = torch.cat((time[:,None], angles.sin(), angles.cos()),dim=1)
        h = self.input(torch.cat((state,received),dim=1)).transpose(1,2)
        h = h + self.position + self.time(temporal)[:,None,:]
        v = self.output(self.field(h))
        return v.reshape(len(v),64,2,4).permute(0,2,1,3).reshape_as(received)

    def integrate(self, received, noise, *, noise_scale=.02, steps=8):
        if noise.shape != received.shape or type(steps) is not int or steps < 1:
            raise ValueError('Aligned explicit noise and positive fixed step count required')
        state = received + float(noise_scale)*noise
        for i in range(steps):
            time = received.new_full((len(received),),i/steps)
            state = state + self.velocity(state,received,time)/steps
        return state

    def forward(self, received):
        """Fixed zero-noise path, without an identity or a target input."""
        return self.integrate(received,torch.zeros_like(received))

    def ensemble(self, received, noises, *, noise_scale=.02, steps=8):
        if noises.ndim != 4 or noises.shape[1:] != received.shape or len(noises) < 2:
            raise ValueError('At least two explicit noise draws for the same IQ batch are required')
        draws = torch.stack([self.integrate(received,noise,noise_scale=noise_scale,steps=steps)
                             for noise in noises],dim=0)
        return draws.mean(dim=0), draws.var(dim=0,unbiased=False)


def source_flow_loss(model, clean, received, time, noise, *, noise_scale=.02):
    """Source-only vector-field supervision; no transmitter labels required."""
    state,target = flow_path(clean,received,time,noise,noise_scale)
    velocity = model.velocity(state,received,time)
    energy = clean.detach().square().mean((1,2)).clamp_min(1e-4)
    loss = (velocity-target.detach()).square().mean((1,2))/energy
    endpoint = state + (1-time[:,None,None])*velocity
    return loss, endpoint


def frozen_identity_consistency(reader, clean, endpoint_a, endpoint_b):
    """Source-only frozen-F feature anchoring and two-trajectory agreement.

    This transfers the reader's learned identity information into G and is not
    F-free. It does not access a transmitter label or update reader weights.
    """
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Existing reader must be frozen and in eval mode')
    with torch.no_grad():
        reference = F.normalize(reader.features(clean).detach(),dim=1)
    a = F.normalize(reader.features(endpoint_a),dim=1)
    b = F.normalize(reader.features(endpoint_b),dim=1)
    anchor = .5*((1-(a*reference).sum(1))+(1-(b*reference).sum(1)))
    agreement = 1-(a*b).sum(1)
    return anchor, agreement
