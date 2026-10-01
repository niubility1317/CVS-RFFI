"""Restore structured time/frequency interactions through a direct identity path."""
import torch
from torch import nn
from experiments.cvs_balanced_identity.model import build as balanced_build,BalancedFusion
VARIANTS=('tf_lowrank32','tf_bilinear')

class InteractionFusion(BalancedFusion):
    def __init__(self,dimension,has_circularity,drop,kind):
        super().__init__(dimension,has_circularity,drop)
        self.kind=kind
        if kind=='tf_lowrank32':
            self.down=nn.Linear(2*dimension+int(has_circularity),32)
            self.up=nn.Linear(32,dimension,bias=False)
            nn.init.zeros_(self.up.weight)
        elif kind=='tf_bilinear':
            self.interaction_gain=nn.Parameter(torch.zeros(dimension))
        else:raise ValueError('Unregistered time/frequency interaction')

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=2*self.dimension+int(self.has_circularity):raise ValueError('Native branch schema changed')
        t=self.time_norm(x[:,:self.dimension]);f=self.frequency_norm(x[:,self.dimension:2*self.dimension])
        joint=t+self.frequency_gain.tanh()*f
        if self.has_circularity:joint=joint+x[:,-1:]*self.circularity_gain.tanh()
        if self.kind=='tf_lowrank32':
            parts=[t,f]+([x[:,-1:]] if self.has_circularity else [])
            joint=joint+0.25*self.up(torch.nn.functional.silu(self.down(torch.cat(parts,1))))
        else:
            joint=joint+self.interaction_gain.tanh()*torch.tanh(t*f)
        return self.drop(joint)

def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered interaction identity variant')
    model=balanced_build('balanced_fusion');b=model.id_backbone;old=b.fuse
    b.fuse=InteractionFusion(b.emb_dim,b.use_circularity,old.drop.p,variant)
    return model
