"""Packet-local observations without an explicit frequency-offset scalar.

Alignment is phase-only; its coefficients never enter identity/condition inputs.
Whole-packet normalization means residual targets are not strict blind spots.
"""
import torch


def iq(x):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point():
        raise ValueError('Expected floating IQ[B,2,256]')
    return torch.complex(x[:,0],x[:,1])


def correlation_magnitudes(x,lags=(1,2,4,8,20)):
    z=iq(x);values=[]
    for lag in lags:
        a,b=z[:,lag:],z[:,:-lag]
        norm=(a.abs().square().sum(1)*b.abs().square().sum(1)).clamp_min(1e-20).sqrt()
        values.append(((a*b.conj()).sum(1).abs()/norm).clamp(0,1))
    return torch.stack(values,1)


def aligned_cycles(x):
    cycles=iq(x)[:,80:160].reshape(-1,4,20)
    cross=(cycles*cycles[:,:1].conj()).sum(-1)
    magnitude=cross.abs()
    unit=torch.where(magnitude>1e-10,cross.conj()/magnitude.clamp_min(1e-10),torch.ones_like(cross))
    # Unit modulus only: relative cycle amplitudes are preserved.
    return cycles*unit[:,:,None]


def repetition_quality(x):
    cycles=aligned_cycles(x);mean=cycles.mean(1)
    power=cycles.abs().square().mean((1,2)).clamp_min(1e-10)
    residual=(cycles-mean[:,None]).abs().square().mean((1,2))/power
    a,b=cycles[:,1:],cycles[:,:-1]
    den=(a.abs().square().sum(-1)*b.abs().square().sum(-1)).clamp_min(1e-20).sqrt()
    coherence=((a*b.conj()).sum(-1).abs()/den).mean(1).clamp(0,1)
    fraction=(mean.abs().square().mean(1)/power).clamp(0,1)
    return torch.stack((coherence,residual.clamp_min(0),fraction),1)


def signal_state(x):
    """40D compatible shape, new semantics; old fitted heads/caches forbidden."""
    z=iq(x);power=z.abs().square()
    psd=torch.fft.fft(z).abs().square().reshape(len(x),32,8).mean(2)
    psd=torch.log1p(psd/psd.mean(1,keepdim=True).clamp_min(1e-10))
    normalized=power/power.mean(1,keepdim=True).clamp_min(1e-10)
    moments=torch.stack((normalized.square().mean(1),normalized.pow(3).mean(1),normalized.max(1).values),1)
    return torch.cat((psd,repetition_quality(x),8*torch.tanh(moments/8),correlation_magnitudes(x,(1,2))),1)


def nonlinear_memory_features(x):
    """17D relative observations; not calibrated PA or hardware parameters."""
    z=iq(x);rms=z.abs().square().mean(1,keepdim=True).clamp_min(1e-12).sqrt();u=z/rms
    a=u.abs()
    envelope=torch.stack((a.mean(1),a.std(1,unbiased=False),a.pow(3).mean(1),a.pow(4).mean(1),a.max(1).values,(a[:,1:]-a[:,:-1]).abs().mean(1)),1)
    product=u[:,2:]*u[:,:-2]*u[:,1:-1].conj().square()
    magnitude=product.abs();unit=product/magnitude.clamp_min(1e-6)
    # Low-amplitude locations contribute little; no mean phase slope is used.
    weight=magnitude/(1+magnitude);den=weight.sum(1).clamp_min(1e-6)
    curvature=torch.stack(((weight*unit.real).sum(1)/den,(weight*unit.imag).sum(1)/den,weight.mean(1)),1)
    return torch.cat((torch.tanh(envelope/4),correlation_magnitudes(x),repetition_quality(x),curvature),1)


def known_linear(x,parameters,scale=.015):
    """Known two side taps around a fixed unit center, circular digital action.

    Four real targets are the actual applied side-tap components / scale.
    Circular boundary artifacts require a separate interior-only diagnostic.
    """
    z=iq(x)
    if parameters.shape!=(len(x),4) or not 0<scale<=.03:
        raise ValueError('Unregistered small linear action')
    if not torch.isfinite(parameters).all() or bool((parameters.abs()>1).any()):
        raise ValueError('Normalized actual tap components must be bounded')
    p=parameters.to(x)*scale
    left=torch.complex(p[:,0],p[:,1]);right=torch.complex(p[:,2],p[:,3])
    result=z+left[:,None]*z.roll(1,1)+right[:,None]*z.roll(-1,1)
    return torch.stack((result.real,result.imag),1)
