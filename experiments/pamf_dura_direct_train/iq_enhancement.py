"""Bounded IQ shape correction; no classifier, labels, or data access."""
import torch
from torch import nn
from torch.nn import functional as F


def shape_correction(raw, base, limit=.05):
    """Remove complex gain/phase directions and bound relative RMS by limit."""
    b=base.detach()
    z=torch.complex(b[:,0],b[:,1])
    v=torch.complex(raw[:,0].tanh(),raw[:,1].tanh())
    energy=z.abs().square().sum(-1,keepdim=True)
    coefficient=(z.conj()*v).sum(-1,keepdim=True)/energy.clamp_min(1e-12)
    perpendicular=v-z*coefficient
    rms=b.square().mean((1,2),keepdim=True).sqrt()
    return limit*rms*torch.stack((perpendicular.real,perpendicular.imag),1)


class ShapeEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(4,16,7,padding=3),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv1d(16,16,5,padding=2,groups=16),nn.Conv1d(16,16,1),
            nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,received,base):
        # The source-F loss cannot change the physical restoration parameters.
        b=base.detach()
        return b+shape_correction(self.net(torch.cat((received.detach(),b),1)),b)


def head_basis(reader):
    weight=F.normalize(reader.core.id_backbone.cls_head.weight.detach().double(),dim=1)
    _,singular,vh=torch.linalg.svd(weight-weight.mean(0),full_matrices=False)
    if int((singular>singular.max()*1e-7).sum())!=5:
        raise ValueError('Original centered six-class head must have rank5')
    return vh[:5].T.to(dtype=torch.float32).detach()


def amplified_reference(features,basis):
    unit=F.normalize(features.detach(),dim=1,eps=1e-4)
    # Amplify centered-head-visible content by 2; no hard pseudo-TX assignment.
    visible=(unit@basis)@basis.T
    return F.normalize(unit+visible,dim=1,eps=1e-4)
