import torch
from . import design as d
def build(m,c):
 d.validate(c);return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
def clip(m,c,opt):
 d.validate(c);return torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5.)
