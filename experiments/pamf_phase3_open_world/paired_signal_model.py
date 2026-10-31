"""Shared P0/P1 SignalOpen with a zero-initialized bounded pair residual."""
import math
import torch
from torch import nn
from torch.nn import functional as F

BASE_PARAMETERS=334721
ADAPTER_PARAMETERS=51360
PARAMETERS=386081


class PairedSignal(nn.Module):
    def __init__(self,base,paired:bool):
        super().__init__()
        if not isinstance(base,nn.Module) or type(paired) is not bool:raise ValueError('Shared base and boolean input mode required')
        self.base=base;self.paired=paired
        parameter=next(base.parameters())
        self.adapter=nn.Linear(320,160,bias=True).to(device=parameter.device,dtype=parameter.dtype)
        nn.init.zeros_(self.adapter.weight);nn.init.zeros_(self.adapter.bias);self.eval()

    def train(self,mode=True):return super().train(False)

    def features(self,received,restored):
        # A P1-only control does not even validate/read the received argument.
        first=received if self.paired else restored
        z0=self.base(first);z1=self.base(restored)
        if (z0.ndim!=2 or z0.shape!=z1.shape or z0.shape[1]!=160 or z0.dtype!=torch.float32
            or z1.dtype!=torch.float32 or not bool(torch.isfinite(z0).all()) or not bool(torch.isfinite(z1).all())):
            raise ValueError('Aligned finite shared-base unit160 features required')
        pair=torch.cat((z0-z1,z0*z1),dim=1)
        delta=(.1/math.sqrt(160))*torch.tanh(self.adapter(pair))
        return dict(z=F.normalize(z1+delta,dim=1,eps=1e-12),z0=z0,z1=z1,delta=delta)

    def forward(self,received,restored):return self.features(received,restored)['z']


def parameter_report(model):
    return dict(signal_parameters=sum(p.numel() for p in model.parameters()),
        base_signal_parameters=sum(p.numel() for p in model.base.parameters()),
        pair_adapter_parameters=sum(p.numel() for p in model.adapter.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
