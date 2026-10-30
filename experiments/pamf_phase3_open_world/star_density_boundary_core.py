"""Differentiable fixed source density ratio; no fitted head or target state."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import star_density_core as density

DIMENSION=160
MODES_PER_CLASS=15


class FrozenDensity(nn.Module):
    def __init__(self,state,device='cpu'):
        super().__init__();density.validate_state(state,'density_ratio')
        if state['known_means'].shape!=(90,DIMENSION):raise ValueError('Only the selected original source D160 density is allowed')
        def constant(name,array):
            self.register_buffer(name,torch.tensor(np.asarray(array).tolist(),dtype=torch.float64,device=device))
        for name in ('origin','whitener','known_means','known_variances','background_means','background_variances'):
            constant(name,state[name])
        constant('known_log_constant',-math.log(MODES_PER_CLASS)-80*np.log(2*np.pi*state['known_variances']))
        constant('background_log_constant',np.log(state['background_weights'])-80*np.log(2*np.pi*state['background_variances']))
        self.eval()

    def forward(self,z,closed):
        if (not torch.is_tensor(z) or z.ndim!=2 or z.shape[1]!=DIMENSION or len(z)==0
                or z.dtype not in (torch.float32,torch.float64) or z.device!=self.origin.device
                or not bool(torch.isfinite(z).all()) or not torch.is_tensor(closed)
                or closed.dtype!=torch.long or closed.shape!=(len(z),) or closed.device!=z.device
                or not bool(((closed>=0)&(closed<6)).all())):
            raise ValueError('Finite live z160 and aligned frozen F0 predicted identities required')
        value=z.double();norm=torch.linalg.vector_norm(value,dim=1,keepdim=True)
        if not bool((norm>0).all()):raise ValueError('Every source feature must be nonzero')
        white=(value/norm-self.origin)@self.whitener
        indices=closed[:,None]*MODES_PER_CLASS+torch.arange(MODES_PER_CLASS,device=z.device)[None]
        difference=white[:,None]-self.known_means[indices]
        known=torch.logsumexp(self.known_log_constant[indices]-difference.square().sum(2)/(2*self.known_variances[indices]),dim=1)
        background_difference=white[:,None]-self.background_means[None]
        background=torch.logsumexp(self.background_log_constant[None]-background_difference.square().sum(2)/(2*self.background_variances[None]),dim=1)
        # Both log densities use the same whitening Jacobian, which cancels.
        ratio=known-background
        if not bool(torch.isfinite(ratio).all()):raise FloatingPointError('Nonfinite differentiable conditional density ratio')
        return ratio

    def loss(self,Lz,OEz,L_closed,OE_closed):
        left=self(Lz,L_closed);right=self(OEz,OE_closed)
        L_BCE=F.softplus(-left/DIMENSION).mean();OE_BCE=F.softplus(right/DIMENSION).mean()
        return dict(loss=.5*(L_BCE+OE_BCE),L_BCE=L_BCE,OE_BCE=OE_BCE,
                    L_mean_ratio=left.mean(),OE_mean_ratio=right.mean())
