"""Known identity CE/KD, OE relational preservation and default-head exposure."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

ARMS=('relation_control','relation_oe')
TEMPERATURE=2.
GRAM_WEIGHT=1.
OE_WEIGHT=1.


def compute(z_OE,logits_L,logits_OE,teacher_logits_L,y_L,teacher_unit_OE,arm):
    """Only the main arm sends a uniform-six OE target into the default head."""
    if arm not in ARMS:raise ValueError('Only the two fixed relation/OE arms are allowed')
    if not isinstance(z_OE,torch.Tensor) or z_OE.ndim!=2 or len(z_OE)<2 or z_OE.shape[1]!=160 or not z_OE.is_floating_point():
        raise ValueError('At least two source OE raw fused160 rows required for off-diagonal Gram')
    if not isinstance(logits_L,torch.Tensor) or logits_L.ndim!=2 or not len(logits_L) or logits_L.shape[1]!=6:
        raise ValueError('Nonempty complete default-six source L logits required')
    n,m=len(logits_L),len(z_OE)
    for value,shape in ((logits_L,(n,6)),(logits_OE,(m,6)),(teacher_logits_L,(n,6)),(teacher_unit_OE,(m,160))):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=z_OE.dtype or value.device!=z_OE.device
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError('Aligned finite source tensors with the same dtype/device required')
    if not bool(torch.isfinite(z_OE).all()):raise ValueError('Finite source OE features required')
    if (not isinstance(y_L,torch.Tensor) or y_L.shape!=(n,) or y_L.dtype!=torch.int64 or y_L.device!=z_OE.device
            or bool(((y_L<0)|(y_L>=6)).any())):raise ValueError('Aligned source L labels in the original six classes required')
    student=F.normalize(z_OE,dim=1,eps=1e-12);teacher=F.normalize(teacher_unit_OE.detach(),dim=1,eps=1e-12)
    offdiagonal=~torch.eye(m,device=z_OE.device,dtype=torch.bool)
    gram=(student@student.T-teacher@teacher.T)[offdiagonal].square().mean()
    ce=F.cross_entropy(logits_L,y_L)
    kd=F.kl_div(F.log_softmax(logits_L/TEMPERATURE,dim=1),F.softmax(teacher_logits_L.detach()/TEMPERATURE,dim=1),reduction='batchmean')*TEMPERATURE**2
    uniform=-F.log_softmax(logits_OE,dim=1).mean()-math.log(6.)
    if arm=='relation_control':uniform=uniform.detach()
    total=ce+kd+GRAM_WEIGHT*gram+(OE_WEIGHT*uniform if arm=='relation_oe' else 0.)
    values=dict(total=total,ce=ce,kd=kd,gram_OE=gram,OE_uniform=uniform)
    if any(not bool(torch.isfinite(v)) for v in values.values()):raise ValueError('Finite relation/OE loss components required')
    return values
