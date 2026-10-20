import copy
import math
import torch
from torch import nn
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_restoration.models import Block,zero_output


def validate(value):
    if (value.ndim!=3 or value.shape[1:]!=(9,256) or value.dtype!=torch.float32
            or not torch.isfinite(value).all()):
        raise ValueError('Finite FP32 current-packet prior9x256 required')


def haar_split(value):
    if value.ndim!=3 or value.shape[-1]%2 or not value.is_floating_point():
        raise ValueError('Real BxCxL with even L required for Haar analysis')
    even,odd=value[...,0::2],value[...,1::2]
    return torch.cat(((even+odd)/math.sqrt(2),(even-odd)/math.sqrt(2)),1)


def haar_merge(bands):
    if bands.ndim!=3 or bands.shape[1]%2 or not bands.is_floating_point():
        raise ValueError('Real Bx2CxL required for Haar synthesis')
    low,high=bands.chunk(2,1)
    even,odd=(low+high)/math.sqrt(2),(low-high)/math.sqrt(2)
    return torch.stack((even,odd),-1).flatten(-2)


def analysis_bands(value,keep_high):
    bands=haar_split(value)
    if keep_high:return bands
    low,high=bands.chunk(2,1)
    return torch.cat((low,torch.zeros_like(high)),1)


class WaveletCore(nn.Module):
    """Three lossless analysis operators; learned blocks are not invertible.

    The control zeros only analysis high bands, retaining the same parameters,
    full-resolution skips, learned synthesis and complete initial state.
    """
    def __init__(self,keep_high=True,widths=(32,64,96,128)):
        super().__init__();self.keep_high=keep_high
        self.encode=nn.ModuleList(nn.Sequential(Block(a,b),Block(b,b))
            for a,b in zip((9,*(2*w for w in widths[:-1])),widths))
        self.expand=nn.ModuleList(nn.Conv1d(a,2*b,1) for a,b in zip(widths[:0:-1],widths[-2::-1]))
        self.decode=nn.ModuleList(nn.Sequential(Block(2*w,w),Block(w,w)) for w in widths[-2::-1])
        self.out=zero_output(nn.Conv1d(widths[0],2,1))

    def forward(self,value):
        validate(value);features=[]
        for index,block in enumerate(self.encode):
            if index:value=analysis_bands(value,self.keep_high)
            value=block(value);features.append(value)
        for expand,block,skip in zip(self.expand,self.decode,features[-2::-1]):
            value=haar_merge(expand(value));value=block(torch.cat((value,skip),1))
        return self.out(value)


class WaveletRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in ('lowpass_control','haar_allbands'):
            raise ValueError('Registered matched wavelet prior required')
        nn.Module.__init__(self);self.candidate=candidate
        self.frozen=frozen.requires_grad_(False).eval()
        core=WaveletCore(keep_high=candidate=='haar_allbands')
        self.prior=nn.ModuleList(copy.deepcopy(core) for _ in range(3))
