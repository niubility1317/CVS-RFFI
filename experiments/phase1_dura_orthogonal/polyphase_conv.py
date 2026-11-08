"""Mathematical polyphase form of a fixed odd-kernel dilated Conv1d."""
import torch
from torch.nn import functional as F


def polyphase_conv1d(x, weight, bias, dilation):
    if (x.ndim!=3 or weight.ndim!=3 or x.shape[1]!=weight.shape[1]
            or weight.shape[2]%2!=1 or not isinstance(dilation,int) or dilation<1
            or x.shape[-1]%dilation!=0 or (bias is not None and bias.shape!=(weight.shape[0],))):
        raise ValueError('Matching channels, odd kernel and divisible signal length required')
    batch,channels,length=x.shape
    phase=x.reshape(batch,channels,length//dilation,dilation).permute(0,3,1,2).reshape(batch*dilation,channels,length//dilation)
    value=F.conv1d(phase,weight,bias,padding=weight.shape[-1]//2)
    return value.reshape(batch,dilation,weight.shape[0],length//dilation).permute(0,2,3,1).reshape(batch,weight.shape[0],length)
