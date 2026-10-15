"""Same physical engine and named draws as A; only configuration identity differs."""
from . import design as d
from experiments.cvs_urban_sigtime50.channel import (PairedChannel as Original,
    configurations,imports,audit,tensor_iq)


class PairedChannel(Original):
    def __init__(self,c):
        d.validate(c);self.c=c;self.configs=configurations();self.ch,self.residual=imports()
