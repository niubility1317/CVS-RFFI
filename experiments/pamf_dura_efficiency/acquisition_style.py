"""Bounded invertible acquisition perturbations with exact source-A pairing."""
from typing import NamedTuple
import torch
from experiments.pamf_restoration.models import complex_iq,pair


class Style(NamedTuple):
    taps:torch.Tensor
    image:torch.Tensor


def sample_style(count,generator,device='cpu',dtype=torch.float32):
    values=torch.randn(count,2,2,generator=generator,device=device,dtype=dtype)*.03
    taps=torch.complex(values[...,0],values[...,1])
    taps=taps*torch.clamp(.08/taps.abs().sum(1,keepdim=True).clamp_min(1e-12),max=1)
    values=torch.randn(count,2,generator=generator,device=device,dtype=dtype)*.005
    image=torch.complex(values[:,0],values[:,1])
    image=image*torch.clamp(.015/image.abs().clamp_min(1e-12),max=1)
    return Style(taps,image)


def linear_style(clean,style):
    if clean.ndim!=3 or clean.shape[1]!=2 or clean.shape[2]<3 or style.taps.shape!=(len(clean),2) or style.image.shape!=(len(clean),):
        raise ValueError('Paired IQ and two causal taps plus one image coefficient required')
    x=complex_iq(clean)
    one=torch.cat((torch.zeros_like(x[:,:1]),x[:,:-1]),1)
    two=torch.cat((torch.zeros_like(x[:,:2]),x[:,:-2]),1)
    value=x+style.taps[:,0,None]*(one-x)+style.taps[:,1,None]*(two-x)
    return pair(value+style.image[:,None]*value.conj())


def apply_style(clean,style):
    value=linear_style(clean,style)
    before=clean.square().mean((1,2),keepdim=True)
    after=value.square().mean((1,2),keepdim=True)
    return value*(before/after.clamp_min(1e-12)).sqrt()


def paired_style(received,clean,actual_action,style):
    """Keep the actual A and same noise: y'=y+A(S(x)-x), reference S(x)."""
    if received.shape!=clean.shape:raise ValueError('Matched original source IQ required')
    reference=apply_style(clean,style)
    observed=received+pair(actual_action(complex_iq(reference-clean)))
    return observed,reference
