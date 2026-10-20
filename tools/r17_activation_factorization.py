"""Activation-second-moment factor initialization; no dataset or label access."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_r17_compact.models import FactorConv


def factor_rank(conv,fraction):
    if (not isinstance(conv,nn.Conv1d) or conv.groups!=1 or conv.padding_mode!='zeros'
            or conv.kernel_size[0]<=1 or not math.isfinite(fraction) or not 0<fraction<=1):
        raise ValueError('Ordinary long Conv1d and positive rank fraction<=1 required')
    rank=max(1,int(min(conv.out_channels,conv.in_channels*conv.kernel_size[0])*fraction))
    if rank*(conv.in_channels*conv.kernel_size[0]+conv.out_channels)>=conv.weight.numel():
        raise ValueError('Factorization must save actual kernel parameters')
    return rank


@torch.no_grad()
def input_patches(conv,values,positions=8):
    if not isinstance(positions,int) or positions<1:raise ValueError('Positive deterministic position budget required')
    if values.ndim!=3 or values.shape[1]!=conv.in_channels or not torch.isfinite(values).all():
        raise ValueError('Finite current convolution inputs required')
    unfolded=F.unfold(values.unsqueeze(2),kernel_size=(1,conv.kernel_size[0]),
        dilation=(1,conv.dilation[0]),padding=(0,conv.padding[0]),stride=(1,conv.stride[0]))
    length=unfolded.shape[-1]
    chosen=torch.linspace(0,length-1,min(positions,length),device=values.device).round().long()
    return unfolded.index_select(-1,chosen).permute(1,0,2).reshape(unfolded.shape[1],-1).contiguous()


class InputSecondMoment:
    """Streaming uncentered moments; callers decide which permitted IQ to pass."""
    def __init__(self,conv,positions=8):
        if not isinstance(conv,nn.Conv1d) or conv.groups!=1 or conv.padding_mode!='zeros':
            raise ValueError('Ordinary dense zero-padded convolution required')
        self.conv=conv;self.positions=positions;self.total=None;self.count=0

    @torch.no_grad()
    def update(self,values):
        patches=input_patches(self.conv,values,self.positions).float()
        product=patches@patches.T
        if self.total is None:self.total=product
        else:self.total.add_(product)
        self.count+=patches.shape[1]

    def value(self):
        if not self.count:raise ValueError('At least one actual calibration input required')
        moment=(self.total/self.count).detach().cpu().double()
        return (moment+moment.T)/2


@torch.no_grad()
def calibrated_factor(conv,second_moment,fraction=.5,ridge_fraction=1e-3):
    """Minimize ridge-regularized E||Wx-W_rank x||² with unchanged conv bias."""
    rank=factor_rank(conv,fraction)
    inner=conv.in_channels*conv.kernel_size[0]
    if (not torch.is_tensor(second_moment) or second_moment.shape!=(inner,inner)
            or not torch.isfinite(second_moment).all() or not math.isfinite(ridge_fraction) or ridge_fraction<=0):
        raise ValueError('Finite input second moment and positive ridge required')
    moment=second_moment.detach().cpu().double()
    if not torch.allclose(moment,moment.T,rtol=0,atol=1e-10):raise ValueError('Symmetric second moment required')
    if (moment.diagonal()<0).any():raise ValueError('Nonnegative input energy required')
    ridge=max(1e-8,ridge_fraction*float(moment.diagonal().mean()))
    regularized=moment+ridge*torch.eye(inner,dtype=moment.dtype)
    lower=torch.linalg.cholesky(regularized)
    weight=conv.weight.detach().cpu().double().flatten(1)
    u,s,vh=torch.linalg.svd(weight@lower,full_matrices=False)
    root=s[:rank].sqrt();right=u[:,:rank]*root[None]
    weighted_left=root[:,None]*vh[:rank]
    left=torch.linalg.solve_triangular(lower.T,weighted_left.T,upper=True).T
    factor=FactorConv.from_conv(conv,fraction,initialize=False)
    factor.left.weight.copy_(left.reshape_as(factor.left.weight).to(conv.weight))
    factor.right.weight.copy_(right[:,:,None].to(conv.weight))
    if conv.bias is not None:factor.right.bias.copy_(conv.bias)
    reconstructed=right@left;difference=weight-reconstructed
    error=float(torch.sum((difference@regularized)*difference))
    original=float(torch.sum((weight@regularized)*weight))
    report=dict(rank=rank,original_kernel_parameters=weight.numel(),
        factor_kernel_parameters=rank*(inner+conv.out_channels),ridge=ridge,
        regularized_output_relative_squared_error=error/max(original,1e-30),
        uncentered_second_moment=True,teacher_weight_changed=False)
    return factor,report
