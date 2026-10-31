"""Sixteen bounded real parameters for a complex FIR plus conjugate image."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

PARAMETERS=16


class BoundedIQ(nn.Module):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(PARAMETERS,dtype=torch.float32))

    def _check(self):
        if self.theta.shape!=(PARAMETERS,) or self.theta.dtype!=torch.float32 or not bool(torch.isfinite(self.theta).all()):
            raise ValueError('Finite FP32 theta[16] required')

    def coefficients(self):
        self._check();bounded=.05*self.theta.tanh();center=torch.zeros((7,2),device=self.theta.device,dtype=torch.float32);center[3,0]=1.
        return bounded[:14].reshape(7,2)+center,bounded[14:]

    def components(self):
        fir,gamma=self.coefficients();return dict(fir=fir,gamma=gamma)

    def prior(self):
        self._check();return self.theta.tanh().square().mean()

    def forward(self,x):
        self._check()
        if (not torch.is_tensor(x) or x.ndim!=3 or not len(x) or tuple(x.shape[1:])!=(2,256) or x.dtype!=torch.float32
                or x.device!=self.theta.device or not bool(torch.isfinite(x).all())):raise ValueError('Finite aligned FP32[N,2,256] input required')
        with torch.autocast(device_type=x.device.type,enabled=False):
            fir,gamma=self.coefficients();windows=F.pad(x,(3,3),mode='reflect').unfold(2,7,1);real,imag=windows[:,0],windows[:,1]
            out_real=(real*fir[:,0]-imag*fir[:,1]).sum(-1)+gamma[0]*x[:,0]+gamma[1]*x[:,1]
            out_imag=(real*fir[:,1]+imag*fir[:,0]).sum(-1)+gamma[1]*x[:,0]-gamma[0]*x[:,1]
            result=torch.stack((out_real,out_imag),1)
        if not bool(torch.isfinite(result).all()):raise ValueError('Input adapter produced nonfinite IQ')
        return result


def _matrix(value,n,width,name):
    if (not torch.is_tensor(value) or value.shape!=(n,width) or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite FP32 matrix required: '+name)
    return value


def source_loss(logits,y,teacher_logits,adapter):
    n=len(logits);_matrix(logits,n,6,'source logits');_matrix(teacher_logits,n,6,'cached source teacher logits')
    if (not n or not torch.is_tensor(y) or y.shape!=(n,) or y.dtype!=torch.long or y.device!=logits.device
            or teacher_logits.device!=logits.device or bool(((y<0)|(y>=6)).any())):raise ValueError('Aligned legal source labels and same-input teacher required')
    ce=F.cross_entropy(logits,y);kd=4*F.kl_div(F.log_softmax(logits/2,dim=1),F.softmax(teacher_logits.detach()/2,dim=1),reduction='batchmean');prior=adapter.prior()
    return dict(total=ce+kd+.1*prior,CE=ce,KD=kd,prior=prior)


def target_loss(logits,pseudo,mask,z,teacher_z):
    n=len(logits);_matrix(logits,n,6,'target logits');_matrix(z,n,160,'target PCD');_matrix(teacher_z,n,160,'frozen target PCD teacher')
    if (not n or not torch.is_tensor(pseudo) or pseudo.shape!=(n,) or pseudo.dtype!=torch.long or bool(((pseudo<0)|(pseudo>=6)).any())
            or not torch.is_tensor(mask) or mask.shape!=(n,) or mask.dtype!=torch.bool or any(v.device!=logits.device for v in (pseudo,mask,z,teacher_z))):
        raise ValueError('Aligned frozen pseudo predictions/mask and target features required')
    pl=(F.cross_entropy(logits,pseudo.detach(),reduction='none')*mask.detach().to(logits.dtype)).mean()
    anchor=(z-teacher_z.detach()).square().sum(1).mean()
    return dict(total=pl+10*anchor,PL=pl,z_anchor=anchor)
