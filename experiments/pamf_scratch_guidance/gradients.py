import torch
from . import design as d
from experiments.pamf_gradient_balance.gradients import merge_gradients


def output_gradient(estimate,paired,identity,arm):
    if arm not in d.ARMS or paired.shape!=(len(estimate),) or identity.shape!=paired.shape:
        raise ValueError('Registered waveform/identity loss vectors required')
    wave=torch.autograd.grad(paired.sum(),estimate,retain_graph=arm==d.ARMS[1])[0]
    semantic=torch.autograd.grad(identity.sum(),estimate)[0] if arm==d.ARMS[1] else torch.zeros_like(wave)
    merged,stats=merge_gradients(wave,semantic,'per_frame_cap')
    return merged.detach()/len(estimate),stats
