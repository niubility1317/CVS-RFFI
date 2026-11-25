"""Shared affine parameters, separate source clean/SG BN statistics."""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F

class ViewBatchNorm(nn.BatchNorm1d):
    def __init__(self, old):
        super().__init__(old.num_features,old.eps,old.momentum,old.affine,old.track_running_stats)
        if not old.track_running_stats:raise ValueError('Running BN statistics required')
        self.load_state_dict(old.state_dict())
        self.register_buffer('sg_running_mean',old.running_mean.clone())
        self.register_buffer('sg_running_var',old.running_var.clone())
        self.register_buffer('sg_num_batches_tracked',old.num_batches_tracked.clone())
        self.view='clean'
    def bank(self):
        if self.view=='clean':return self.running_mean,self.running_var,self.num_batches_tracked
        if self.view=='sg':return self.sg_running_mean,self.sg_running_var,self.sg_num_batches_tracked
        raise ValueError('Unknown BN view')
    def reset_bank(self):
        m,v,n=self.bank();m.zero_();v.fill_(1);n.zero_()
    def forward(self,x):
        self._check_input_dim(x)
        m,v,n=self.bank();factor=self.momentum if self.momentum is not None else 0.
        if self.training:
            n.add_(1)
            if self.momentum is None:factor=1./float(n)
        return F.batch_norm(x,m,v,self.weight,self.bias,self.training,factor,self.eps)

def install_view_bn(module):
    for name,child in list(module.named_children()):
        if isinstance(child,ViewBatchNorm):continue
        if isinstance(child,nn.BatchNorm1d):setattr(module,name,ViewBatchNorm(child))
        else:install_view_bn(child)

@contextmanager
def view_context(encoder,view):
    banks=[m for m in encoder.modules() if isinstance(m,ViewBatchNorm)]
    if not banks:raise RuntimeError('View BN not installed')
    old=[m.view for m in banks]
    try:
        for m in banks:m.view=view
        yield
    finally:
        for m,v in zip(banks,old):m.view=v

def encode_view(encoder,x,view):
    with view_context(encoder,view):return encoder(x)

def encode_pairs(encoder,x):
    if len(x)%2:raise ValueError('Interleaved complete pairs required')
    clean=encode_view(encoder,x[0::2],'clean')
    sg=encode_view(encoder,x[1::2],'sg')
    return torch.stack((clean,sg),1).flatten(0,1)
