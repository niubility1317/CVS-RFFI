from . import design as d
from .model import counts
from experiments.cvs_rdrop_sinc_units import optimizer as units
def build(m,c):
 d.validate(c);counts(m);params=[p for p in m.parameters() if p.requires_grad]
 return units.build(m,params,'CENTER_UNITS',lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
def clip(m,c,opt):
 d.validate(c);return units.clip_physical(opt,[p for p in m.parameters() if p.requires_grad],5.)
