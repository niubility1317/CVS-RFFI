"""Source-only default-head CE/KD, unit-feature preservation and auxiliary OVA.

Both arms train the same six binary auxiliary logits. Only the auxiliary path
to the student encoder is detached in the control; CE, KD and feature
preservation retain their student gradients. The auxiliary head never replaces
the original six-class identity head. No fitting, IO or sampling occurs here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


RUN_ID='20261029-phase3-star-joint-ova-s2026092701-r02'
FEATURE_DIM=160
CLASS_COUNT=6
TEMPERATURE=2.
FEATURE_WEIGHT=5.
OVA_WEIGHT=1.
UNIT_EPS=1e-12


def _matrix(value,columns,name):
    if (not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[0]==0
            or value.shape[1]!=columns or not value.is_floating_point()):
        raise ValueError('Nonempty floating Torch matrix required: '+name)


def _inputs(z_L,z_OE,logits_L,teacher_logits_L,y_L,teacher_unit_L,teacher_unit_OE,aux_weight,aux_bias,joint):
    if type(joint) is not bool:raise ValueError('joint must be an explicit bool')
    _matrix(z_L,FEATURE_DIM,'source L raw fused160');_matrix(z_OE,FEATURE_DIM,'source OE raw fused160')
    n,m=len(z_L),len(z_OE)
    floating=((z_L,(n,FEATURE_DIM),'z_L'),(z_OE,(m,FEATURE_DIM),'z_OE'),
        (logits_L,(n,CLASS_COUNT),'original default-head L logits'),
        (teacher_logits_L,(n,CLASS_COUNT),'same-input original F0 teacher logits'),
        (teacher_unit_L,(n,FEATURE_DIM),'source L teacher unit features'),
        (teacher_unit_OE,(m,FEATURE_DIM),'source OE teacher unit features'),
        (aux_weight,(CLASS_COUNT,FEATURE_DIM),'auxiliary binary weight'),(aux_bias,(CLASS_COUNT,),'auxiliary binary bias'))
    for value,shape,name in floating:
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or not value.is_floating_point()
                or value.dtype!=z_L.dtype or value.device!=z_L.device):
            raise ValueError('Aligned floating dtype/device/shape required: '+name)
    if not bool(torch.stack([torch.isfinite(value).all() for value,_,_ in floating]).all()):
        raise ValueError('All source loss inputs must be finite')
    if (not isinstance(y_L,torch.Tensor) or y_L.shape!=(n,) or y_L.dtype!=torch.int64
            or y_L.device!=z_L.device or bool(((y_L<0)|(y_L>=CLASS_COUNT)).any())):
        raise ValueError('Only aligned source L int64 labels 0..5 are accepted')


def compute(z_L,z_OE,logits_L,teacher_logits_L,y_L,teacher_unit_L,teacher_unit_OE,aux_weight,aux_bias,joint:bool):
    """Return ten scalar tensors, preserving only student/auxiliary gradients.

    Feature and OVA component diagnostics are unweighted role means. The total
    is CE + KD(T=2, weight=1) + 5*(.5*feature_L+.5*feature_OE)
    + .5*(.5*OVA_L_positive+.5*OVA_L_hardnegative) + .5*OVA_OE.
    Source sampling is caller-owned; no OE transmitter labels are accepted.
    """
    _inputs(z_L,z_OE,logits_L,teacher_logits_L,y_L,teacher_unit_L,teacher_unit_OE,aux_weight,aux_bias,joint)
    unit_L=F.normalize(z_L,p=2,dim=1,eps=UNIT_EPS)
    unit_OE=F.normalize(z_OE,p=2,dim=1,eps=UNIT_EPS)
    ce=F.cross_entropy(logits_L,y_L,reduction='mean')
    kd=F.kl_div(F.log_softmax(logits_L/TEMPERATURE,dim=1),
        F.softmax(teacher_logits_L.detach()/TEMPERATURE,dim=1),reduction='batchmean')*TEMPERATURE**2
    feature_L=(1.-(unit_L*teacher_unit_L.detach()).sum(dim=1).clamp(-1.,1.)).mean()
    feature_OE=(1.-(unit_OE*teacher_unit_OE.detach()).sum(dim=1).clamp(-1.,1.)).mean()
    feature_total=.5*feature_L+.5*feature_OE
    g_L=F.linear(unit_L if joint else unit_L.detach(),aux_weight,aux_bias)
    g_OE=F.linear(unit_OE if joint else unit_OE.detach(),aux_weight,aux_bias)
    positive=F.softplus(-g_L.gather(1,y_L[:,None])[:,0]).mean()
    # Mask after softplus so the true class remains excluded even if every
    # wrong-class softplus underflows to zero; it cannot win a zero tie.
    wrong=F.softplus(g_L).masked_fill(F.one_hot(y_L,num_classes=CLASS_COUNT).bool(),-torch.inf)
    hardnegative=wrong.max(dim=1).values.mean()
    oe=F.softplus(g_OE).max(dim=1).values.mean()
    ova_total=.5*(.5*positive+.5*hardnegative)+.5*oe
    total=ce+kd+FEATURE_WEIGHT*feature_total+OVA_WEIGHT*ova_total
    result=dict(total=total,ce=ce,kd=kd,feature_L=feature_L,feature_OE=feature_OE,feature_total=feature_total,
                ova_L_positive=positive,ova_L_hardnegative=hardnegative,ova_OE=oe,ova_total=ova_total)
    if not bool(torch.stack([torch.isfinite(value) for value in result.values()]).all()):
        raise ValueError('Nonfinite source joint OVA loss')
    return result
