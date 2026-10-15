"""Unchanged physical engine, nominal scene sampling and epoch/view streams."""
from . import design as d
from experiments.cvs_urban_sigtime50.channel import PairedChannel as Original,configurations,imports,audit,tensor_iq


class PairedChannel(Original):
    def __init__(self,c):d.validate(c);self.c=c;self.configs=configurations();self.ch,self.residual=imports()

    def apply(self,x,meta,epoch,step,*,view='L',eta=0.):
        if view!='L' or eta!=0. or self.c['arm']=='D0':raise ValueError('Only registered nominal L augmentation is active')
        return super().apply(x,meta,epoch,step,view=view,eta=0.)
