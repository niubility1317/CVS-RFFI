"""Input-only choices among class-conditional waveform posterior hypotheses."""
import torch
from torch.nn import functional as F

def choose_hypotheses(log_weights,logits):
    if log_weights.ndim!=2 or logits.ndim!=3 or logits.shape[:2]!=log_weights.shape or logits.shape[2]!=log_weights.shape[1] or not torch.isfinite(log_weights).all() or not torch.isfinite(logits).all():raise ValueError('Finite [B,C] log weights and [B,C,C] classifier logits required')
    self_logp=F.log_softmax(logits,dim=2).diagonal(dim1=1,dim2=2);return {'map':log_weights.argmax(1),'self':self_logp.argmax(1),'product':(log_weights+self_logp).argmax(1)}
