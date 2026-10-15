"""Original practical residual engine with nominal training draws only."""
from . import design as d
from experiments.cvs_urban_sigtime50.channel import (PairedChannel as Original,
    configurations,imports,audit,tensor_iq)


class PairedChannel(Original):
    def __init__(self,c):
        d.validate(c);self.c=c;self.configs=configurations();self.ch,self.residual=imports()

    def apply(self,x,meta,epoch,step,*,view='L',eta=0.):
        if eta!=0. or view!='L':raise ValueError('No curriculum or new U urban objective in position comparison')
        return super().apply(x,meta,epoch,step,view=view,eta=0.)
