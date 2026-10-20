"""Small trainable group-statistic correction inside existing GN locations."""
import math
import torch
from torch import nn

class StatisticGroupNorm(nn.GroupNorm):
    def __init__(self,num_groups,num_channels,eps=1e-5):
        if num_groups!=4 or num_channels%4 or not math.isfinite(eps) or eps<=0:raise ValueError('Existing GN4 shape and positive epsilon required')
        super().__init__(num_groups,num_channels,eps=eps,affine=True)
        # Rows: log mean slope, mean offset, log logvariance slope, logvariance offset.
        self.stat_parameters=nn.Parameter(torch.zeros(4,num_groups))

    def forward(self,value):
        if value.ndim!=3 or value.shape[1]!=self.num_channels:raise ValueError('Native[B,C,T] group-normalization input required')
        groups=value.reshape(value.shape[0],self.num_groups,-1)
        variance,mean=torch.var_mean(groups,dim=-1,unbiased=False)
        slopes=self.stat_parameters[[0,2]].clamp(math.log(.25),math.log(4.)).exp()
        center=slopes[0]*mean+self.stat_parameters[1]
        logvariance=slopes[1]*(variance+self.eps).log()+self.stat_parameters[3]
        logvariance=logvariance.clamp(math.log(self.eps),math.log(torch.finfo(value.dtype).max))
        normalized=((groups-center.unsqueeze(-1))*torch.exp(-.5*logvariance).unsqueeze(-1)).reshape_as(value)
        return normalized*self.weight[None,:,None]+self.bias[None,:,None]

    @torch.no_grad()
    def set_calibration(self,mean_fit,logvariance_fit):
        for offset,item in ((0,mean_fit),(2,logvariance_fit)):
            slope=torch.tensor(item['slope'],device=self.stat_parameters.device,dtype=self.stat_parameters.dtype)
            bias=torch.tensor(item['intercept'],device=self.stat_parameters.device,dtype=self.stat_parameters.dtype)
            if slope.shape!=(4,) or bias.shape!=(4,) or not torch.isfinite(slope).all() or not torch.isfinite(bias).all() or (slope<.25).any() or (slope>4).any():
                raise ValueError('Four finite fixed-range group response coefficients required')
            self.stat_parameters[offset].copy_(slope.log());self.stat_parameters[offset+1].copy_(bias)

@torch.no_grad()
def convert_normalization(model):
    count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,nn.GroupNorm):
                if type(child)!=nn.GroupNorm or not child.affine or child.num_groups!=4:raise ValueError('Original independent affine GN4 modules required')
                fresh=StatisticGroupNorm(4,child.num_channels,child.eps).to(device=child.weight.device,dtype=child.weight.dtype)
                fresh.weight.copy_(child.weight);fresh.bias.copy_(child.bias);fresh.train(child.training);setattr(module,name,fresh);count+=1
            else:replace(child)
    replace(model)
    if count!=168:raise ValueError('All168 original GN locations required')
    return model
