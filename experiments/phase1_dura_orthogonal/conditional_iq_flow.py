"""Source-trained conditional IQ transport from an anchor to a clean waveform.

The network receives one observation and optional fixed physical context. It does
not require a classifier, transmitter label, target statistic or second view.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalIQFlow(nn.Module):
    def __init__(self, width=32, steps=4):
        super().__init__()
        if width < 1 or steps < 1:
            raise ValueError('Positive network width and integration steps required')
        self.steps = steps
        self.entry = nn.Conv1d(10, width, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 9, padding=4*d, dilation=d)
                                    for d in (1, 2, 4))
        self.out = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def velocity(self, state, received, anchor, innovation, variance, time):
        if (state.ndim != 3 or state.shape[1:] != (2, 256) or received.shape != state.shape
                or anchor.shape != state.shape or innovation.shape != state.shape
                or variance.shape != (len(state),) or time.shape != (len(state),)):
            raise ValueError('Matching IQ, variance and time batches required')
        scale = received.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        time_channel = time[:, None, None].expand(-1, 1, 256)
        noise = (variance[:, None, None] / scale.square()).clamp_min(0.)
        noise_channel = torch.log1p(noise).expand(-1, 1, 256)
        features = torch.cat((state/scale, received/scale, anchor/scale, innovation/scale,
                              time_channel, noise_channel), dim=1)
        h = F.silu(self.entry(features))
        for block in self.blocks:
            h = h + .2*F.silu(block(h))
        return self.out(h)*scale

    def forward(self, received, anchor, innovation, variance):
        state = anchor
        for i in range(self.steps):
            time = torch.full((len(state),), i/self.steps, device=state.device, dtype=state.dtype)
            state = state + self.velocity(state, received, anchor, innovation, variance, time)/self.steps
        return state


def flow_matching_loss(model, received, anchor, innovation, variance, clean, time):
    if clean.shape != anchor.shape or time.shape != (len(anchor),):
        raise ValueError('Matching clean IQ and time batches required')
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError('Finite flow time in [0, 1] required')
    alpha = time[:, None, None]
    state = (1-alpha)*anchor + alpha*clean
    target = clean-anchor
    predicted = model.velocity(state, received, anchor, innovation, variance, time)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return ((predicted-target).square().mean((1, 2))/energy).mean()
