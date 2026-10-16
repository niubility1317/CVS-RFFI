"""Finite, noncircular fractional timing view. No label or channel access."""
import torch
from torch.nn import functional as F

MAX_DELAY=.125
RADIUS=16
SEED=20261016071


def draw_delays(count,*,step):
    if count<1 or step<1:raise ValueError('Positive packet count and named step required')
    generator=torch.Generator(device='cpu').manual_seed(SEED+int(step))
    return (torch.rand(count,generator=generator)*2-1)*MAX_DELAY


def delayed(x,delays,*,boundary='reflect',restore_rms=True):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite floating IQ[B,2,256] required')
    delays=torch.as_tensor(delays,device=x.device,dtype=x.dtype)
    if delays.shape!=(len(x),) or not bool(torch.isfinite(delays).all()) or bool((delays.abs()>MAX_DELAY+1e-9).any()):
        raise ValueError('One bounded delay per physical packet required')
    if boundary not in ('reflect','replicate'):raise ValueError('Noncircular boundary required')
    if not bool(delays.any()):return x.clone(),dict(waveform_MSE=0.,max_relative_power_error=0.,max_abs_delay=0.)
    t=torch.arange(-RADIUS,RADIUS+1,device=x.device,dtype=x.dtype)
    window=torch.hann_window(2*RADIUS+1,periodic=False,device=x.device,dtype=x.dtype)
    kernel=torch.sinc(t[None]-delays[:,None])*window[None]
    kernel=kernel/kernel.sum(1,keepdim=True)
    # conv1d correlates: reverse the kernel for positive delay x[n-d].
    padded=F.pad(x,(RADIUS,RADIUS),mode=boundary)
    weights=kernel.flip(1).repeat_interleave(2,dim=0)[:,None]
    out=F.conv1d(padded.reshape(1,2*len(x),256+2*RADIUS),weights,groups=2*len(x)).reshape_as(x)
    before=x.square().mean((1,2),keepdim=True);after=out.square().mean((1,2),keepdim=True)
    if restore_rms:out=out*torch.sqrt(before.clamp_min(1e-30)/after.clamp_min(1e-30))
    detail=dict(waveform_MSE=float((out-x).square().mean()),max_relative_power_error=float(((out.square().mean((1,2),keepdim=True)-before).abs()/before.clamp_min(1e-30)).max()),max_abs_delay=float(delays.abs().max()))
    return out,detail
