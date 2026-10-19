import torch
from . import design as d
from .model import counts
from experiments.cvs_rdrop_sinc_units import optimizer as units

def build(m,c):
 d.validate(c);counts(m);params=[p for p in m.parameters() if p.requires_grad]
 options=dict(lr=.0002,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
 return units.build(m,params,'CENTER_UNITS',**options) if c['style']=='FULL' else torch.optim.AdamW(params,**options)

def clip(m,c,opt):
 params=[p for p in m.parameters() if p.requires_grad]
 return units.clip_physical(opt,params,5.) if c['style']=='FULL' else torch.nn.utils.clip_grad_norm_(params,5.)
