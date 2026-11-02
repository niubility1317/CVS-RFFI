"""Hard-example supervision for a physical waveform mixture."""
import torch
from torch.nn import functional as F

def hard_objective(ground_logits,controller_logits,predicted,ground,target,hard):
 n=len(target)
 if ground_logits.shape!=(n,6) or controller_logits.shape!=(n,6) or predicted.shape!=(n,2,256) or ground.shape!=predicted.shape or hard.shape!=(n,) or hard.dtype!=torch.bool:raise ValueError('Aligned source targets, waveforms and hard mask required')
 ce=F.cross_entropy(ground_logits,target,reduction='none');direct=F.cross_entropy(controller_logits,target,reduction='none');den=ground.flatten(1).square().sum(1).clamp_min(1e-12);iq=(predicted-ground).flatten(1).square().sum(1)/den;weight=1.+4.*hard.float();return (weight*(ce+.5*direct+.2*iq)).sum()/weight.sum()
