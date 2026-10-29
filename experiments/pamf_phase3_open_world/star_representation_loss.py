"""Differentiable CE/KD, uniform OE, raw-feature VICReg and phase/gain views."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def _matrix(value, width, name, minimum=1):
    if (not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != width
            or len(value) < minimum or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite tensor matrix required: '+name)
    return value


def labeled(logits0, logits1, y, teacher_logits):
    """Mean two-view CE0 plus T=2 raw-view teacher KL, weight one."""
    a,b,t=(_matrix(v,6,k) for v,k in ((logits0,'raw logits'),(logits1,'phase/gain logits'),(teacher_logits,'fixed teacher logits')))
    if (a.shape != b.shape or a.shape != t.shape or not torch.is_tensor(y) or y.shape != (len(a),)
            or y.dtype != torch.long or bool(((y<0)|(y>=6)).any())):
        raise ValueError('Aligned six-known-class labels/logits required')
    ce=.5*(F.cross_entropy(a,y,label_smoothing=0.)+F.cross_entropy(b,y,label_smoothing=0.))
    kd=4.*F.kl_div(F.log_softmax(a/2.,dim=1),F.softmax(t.detach()/2.,dim=1),reduction='batchmean')
    return dict(ce=ce,kd=kd,total=ce+kd)


def uniform_oe(logits0, logits1):
    """Average KL(uniform6 || softmax(full default head)) of both views."""
    a,b=_matrix(logits0,6,'OE raw logits'),_matrix(logits1,6,'OE phase/gain logits')
    if a.shape != b.shape: raise ValueError('Aligned OE views required')
    return -.5*(F.log_softmax(a,dim=1).mean()+F.log_softmax(b,dim=1).mean())-math.log(6.)


def vicreg(z0,z1):
    """Official 25/25/1 arithmetic on raw fused160, with sample variance."""
    a,b=_matrix(z0,160,'raw fused view0',2),_matrix(z1,160,'raw fused view1',2)
    if a.shape != b.shape: raise ValueError('Aligned raw fused160 views required')
    inv=F.mse_loss(a,b)
    std=.5*(F.relu(1.-torch.sqrt(a.var(dim=0,unbiased=True)+1e-4)).mean()
             +F.relu(1.-torch.sqrt(b.var(dim=0,unbiased=True)+1e-4)).mean())
    ac,bc=a-a.mean(dim=0),b-b.mean(dim=0)
    ca,cb=ac.T@ac/(len(a)-1),bc.T@bc/(len(b)-1)
    mask=~torch.eye(160,dtype=torch.bool,device=a.device)
    cov=(ca[mask].square().sum()+cb[mask].square().sum())/160.
    return dict(inv=inv,std=std,cov=cov,total=25.*inv+25.*std+cov)


def phase_gain_view(x,generator):
    """One global phase/gain per CPU packet; no RMS/crop/noise/channel step."""
    if (not torch.is_tensor(x) or x.device.type!='cpu' or x.dtype!=torch.float32
            or x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or not len(x) or not bool(torch.isfinite(x).all())
            or not isinstance(generator,torch.Generator) or generator.device.type!='cpu'):
        raise ValueError('Finite CPU FP32[B,2,256] and independent CPU generator required')
    draw=torch.rand((len(x),2),generator=generator,dtype=torch.float32)
    phase=(draw[:,0]*2.-1.)*math.pi;gain=.95+.1*draw[:,1]
    c,s=torch.cos(phase)[:,None],torch.sin(phase)[:,None]
    return gain[:,None,None]*torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)
