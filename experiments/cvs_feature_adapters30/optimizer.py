"""Frozen-head feature updates, retaining the existing physical Sinc algebra."""
import torch
from . import design as d
from .model import counts
from experiments.cvs_rdrop_sinc_units import optimizer as units

def build(m,c):
 d.validate(c);counts(m);params=[p for p in m.parameters() if p.requires_grad];lr=.0002;wd=.0001
 if c['style']!='FULL':return torch.optim.AdamW(params,lr=lr,weight_decay=wd,betas=(.9,.999),eps=1e-8)
 named=dict(m.named_parameters());sinc=m.get_submodule(units.PREFIX[:-1]);q=[named[k] for k in units.QNAMES]
 if sinc.mode!='CENTER_25K' or sinc.scale_hz!=25000. or sinc.sample_rate!=25000000. or any(not p.requires_grad or p.shape!=(24,1) for p in q):raise ValueError('Original fixed F physical Sinc units required')
 qids={id(p) for p in q};other=[p for p in params if id(p) not in qids]
 opt=torch.optim.AdamW([dict(params=other),dict(params=q,lr=lr*1000,eps=1e-11,weight_decay=0.)],lr=lr,weight_decay=wd,betas=(.9,.999),eps=1e-8)
 policy=dict(mode='CENTER_UNITS',multiplier=1000.,base_eps=1e-8,base_weight_decay=wd,sinc=q,names=units.QNAMES,scale=25000.,origins=[sinc.low_origin_hz,sinc.band_origin_hz])
 return units.attach(opt,policy)

def clip(m,c,opt):
 params=[p for p in m.parameters() if p.requires_grad]
 return units.clip_physical(opt,params,5.) if c['style']=='FULL' else torch.nn.utils.clip_grad_norm_(params,5.)
