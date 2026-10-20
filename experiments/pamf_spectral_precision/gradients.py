import torch
from . import design as d
from experiments.pamf_gradient_balance.gradients import merge_gradients


def output_gradient(estimate,paired,identity,arm):
    if arm not in d.ARMS or paired.shape!=(len(estimate),) or identity.shape!=paired.shape:
        raise ValueError('Registered per-IQ wave/identity objectives required')
    wave=torch.autograd.grad(paired.sum(),estimate,retain_graph=True)[0]
    semantic=torch.autograd.grad(identity.sum(),estimate)[0]
    merged,stats=merge_gradients(wave,semantic,'per_frame_cap')
    return merged.detach()/len(estimate),stats
