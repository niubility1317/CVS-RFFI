"""Source-only equiangular class geometry; ordinary cosine CE."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiments.cvs_coherence_identity.model import CoherenceCVS
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
VARIANTS=('simplex_learned','simplex_fixed')

def orthogonal_frame(frame):
    q,r=torch.linalg.qr(frame,mode='reduced')
    signs=torch.where(r.diagonal()>=0,1.,-1.)
    return q*signs.unsqueeze(0)

def helmert_simplex(classes,reference):
    h=reference.new_zeros(classes-1,classes)
    for j in range(classes-1):
        value=1/math.sqrt((j+1)*(j+2))
        h[j,:j+1]=value;h[j,j+1]=-(j+1)*value
    return h*math.sqrt(classes/(classes-1))

class SimplexHead(ResidualPhysicalHead):
    def __init__(self,original,learned):
        nn.Module.__init__(self)
        self.base_norm=original.base_norm;self.pa_norm=original.pa_norm
        self.gain=original.gain;self.scale=original.scale
        self.geometry_mode='learned_rotated_simplex' if learned else 'fixed_simplex'
        classes,dimension=original.weight.shape
        if dimension<classes-1:raise ValueError('Embedding too small for regular simplex')
        # Reuse scratch initialization only: no extra RNG draw or trained weights.
        frame=original.weight.detach()[:classes-1].T.clone()
        h=helmert_simplex(classes,frame)
        if learned:
            self.frame=nn.Parameter(frame);self.register_buffer('simplex_basis',h)
        else:
            self.register_buffer('fixed_weight',(orthogonal_frame(frame)@h).T.contiguous())

    def prototypes(self):
        if self.geometry_mode=='fixed_simplex':return self.fixed_weight
        return (orthogonal_frame(self.frame)@self.simplex_basis).T

    def classify(self,joint):
        return self.scale*F.linear(F.normalize(joint,dim=1,eps=1e-4),
                                  F.normalize(self.prototypes(),dim=1,eps=1e-4))

class SimplexCVS(CoherenceCVS):
    def __init__(self,variant):
        super().__init__('coherence_phase')
        self.id_backbone.cls_head=SimplexHead(self.id_backbone.cls_head,variant=='simplex_learned')

def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered simplex variant')
    return SimplexCVS(variant)
