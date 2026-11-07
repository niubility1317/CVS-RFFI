"""Exact real-action Frobenius error in a compressed reflected-FIR basis."""
import torch


def aligned_factors(estimated,true):
    """Put both operators in a common unit-norm FIR gauge before mixing factors."""
    def normalize(parts):
        h,main,image=parts
        gain=h.norm(dim=1,keepdim=True)
        if (gain<=0).any() or not torch.isfinite(gain).all():
            raise ValueError('Nonzero finite FIR required for gauge alignment')
        return h/gain,main*gain,image*gain

    estimate=normalize(estimated)
    h,main,image=normalize(true)
    overlap=(estimate[0].conj()*h).sum(1,keepdim=True)
    magnitude=overlap.abs()
    # At zero overlap the angle is undetermined; use the identity convention.
    angle=torch.where(magnitude>0,overlap.conj()/magnitude.clamp_min(1e-300),torch.ones_like(overlap))
    return estimate,(h*angle,main*angle.conj(),image*angle),magnitude[:,0]


def relative_action_mse(kernel,main,image,target_kernel,target_main,target_image):
    """Match dense_action error/norm without constructing a [B,2N,2N] matrix.

    For causal reflect, input column |i-a| has at most two tap contributions.
    Sum each such pair before taking its squared magnitude, retaining one anchor.
    The widely linear real matrix norm is 2*(complex_main_norm+image_norm).
    Arithmetic matches the FP64 source loss; callers enforce finite training loss.
    """
    values=(kernel,main,image,target_kernel,target_main,target_image)
    if (kernel.ndim!=2 or main.ndim!=2 or kernel.shape!=target_kernel.shape
            or any(x.shape!=main.shape for x in (image,target_main,target_image))
            or len(kernel)!=len(main) or not 1<=kernel.shape[1]<=main.shape[1]
            or not all(x.is_complex() and x.device==kernel.device for x in values)):
        raise ValueError('Matched complex FIR/main/image factors on one device required')
    kernel,main,image,target_kernel,target_main,target_image=[x.to(torch.complex128) for x in values]
    n=main.shape[1];k=kernel.shape[1]
    rows=torch.arange(n,device=kernel.device)[:,None]
    taps=torch.arange(k,device=kernel.device)[None]
    partner=2*rows-taps
    paired=(partner>=0)&(partner<k)
    add=paired&(partner>taps)
    anchor=(~paired)|(partner>=taps)
    gather=partner.clamp(0,k-1)[None].expand(len(kernel),-1,-1)
    def grouped(h):
        expanded=h[:,None].expand(-1,n,-1)
        merged=expanded+torch.where(add[None],expanded.gather(2,gather),0)
        return torch.where(anchor[None],merged,0)
    predicted=grouped(kernel);target=grouped(target_kernel)
    difference_main=main[:,:,None]*predicted-target_main[:,:,None]*target
    difference_image=image[:,:,None]*predicted.conj()-target_image[:,:,None]*target.conj()
    square=lambda z:z.real.square()+z.imag.square()
    numerator=2*(square(difference_main)+square(difference_image)).sum((1,2))
    denominator=2*((square(target_main)+square(target_image))*square(target).sum(2)).sum(1)
    return numerator/denominator.clamp_min(1e-8)
