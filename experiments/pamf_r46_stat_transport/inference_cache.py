"""Optional weight-derived slopes for an already frozen R58 inference model.

The original training/prediction implementation remains the default. Build the
cache after strict checkpoint loading and final device/dtype placement. It is
nonpersistent tensor state, counted separately from learned parameters.
"""
import copy
import math
import torch
from torch import nn
from .normalization import StatisticGroupNorm


class CachedStatisticGroupNorm(StatisticGroupNorm):
    def __init__(self,num_groups,num_channels,eps=1e-5):
        super().__init__(num_groups,num_channels,eps)
        self.register_buffer('_inference_slopes',torch.empty(0),persistent=False)
        self._cache_version=None
        self._cache_device=None
        self._cache_dtype=None

    @torch.no_grad()
    def refresh_cache(self):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Eval mode and frozen normalization parameters required')
        try:version=self.stat_parameters._version
        except RuntimeError as error:
            raise ValueError('Use prepare_frozen_inference to build versioned cached layers after loading') from error
        self._inference_slopes=self.stat_parameters[[0,2]].clamp(math.log(.25),math.log(4.)).exp()
        self._cache_version=version
        self._cache_device=self.stat_parameters.device
        self._cache_dtype=self.stat_parameters.dtype

    def forward(self,value):
        if value.ndim!=3 or value.shape[1]!=self.num_channels:
            raise ValueError('Native[B,C,T] group-normalization input required')
        if self.training or self.weight.requires_grad or self.bias.requires_grad or self.stat_parameters.requires_grad:
            raise ValueError('Cached coefficients are only for frozen eval normalization')
        if (self._cache_version!=self.stat_parameters._version or self._cache_device!=self.stat_parameters.device
                or self._cache_dtype!=self.stat_parameters.dtype or self._inference_slopes.shape!=(2,self.num_groups)):
            raise ValueError('Refresh coefficient cache after loading/changing parameters or device/dtype')
        groups=value.reshape(value.shape[0],self.num_groups,-1)
        variance,mean=torch.var_mean(groups,dim=-1,unbiased=False)
        slopes=self._inference_slopes
        center=slopes[0]*mean+self.stat_parameters[1]
        logvariance=slopes[1]*(variance+self.eps).log()+self.stat_parameters[3]
        logvariance=logvariance.clamp(math.log(self.eps),math.log(torch.finfo(value.dtype).max))
        normalized=((groups-center.unsqueeze(-1))*torch.exp(-.5*logvariance).unsqueeze(-1)).reshape_as(value)
        return normalized*self.weight[None,:,None]+self.bias[None,:,None]


@torch.inference_mode(False)
@torch.no_grad()
def prepare_frozen_inference(model,*,clone=True):
    if type(clone) is not bool:raise ValueError('Explicit clone bool required')
    norms=[m for m in model.modules() if isinstance(m,nn.GroupNorm)]
    if (sum(p.numel() for p in model.parameters())!=276453 or len(norms)!=168
            or any(type(m) not in (StatisticGroupNorm,CachedStatisticGroupNorm) for m in norms)):
        raise ValueError('Complete original276453 R58 graph and all168 statistic locations required')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze complete model and enter eval mode before inference preparation')
    target=copy.deepcopy(model) if clone else model
    before_keys=tuple(target.state_dict())
    count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if type(child) in (StatisticGroupNorm,CachedStatisticGroupNorm):
                fresh=CachedStatisticGroupNorm(child.num_groups,child.num_channels,child.eps).to(device=child.weight.device,dtype=child.weight.dtype)
                fresh.load_state_dict(child.state_dict(),strict=True)
                fresh.eval().requires_grad_(False);fresh.refresh_cache();setattr(module,name,fresh);count+=1
            else:replace(child)
    replace(target)
    if count!=168 or tuple(target.state_dict())!=before_keys:raise ValueError('Cache must preserve the complete portable checkpoint state')
    cached=[m for m in target.modules() if type(m) is CachedStatisticGroupNorm]
    report=dict(modules=168,parameters=276453,additional_learnable_parameters=0,
        cache_tensor_scalars=sum(m._inference_slopes.numel() for m in cached),
        cache_tensor_bytes=sum(m._inference_slopes.numel()*m._inference_slopes.element_size() for m in cached),
        nonpersistent_cache_tensors=168,checkpoint_keys_unchanged=True,cloned=clone,
        cache_source='Frozen weight coefficients only;no IQ/source/target/F or fitting',
        measured_GPU_speedup=False,formal_prediction_parity_verified=False)
    return target,report
