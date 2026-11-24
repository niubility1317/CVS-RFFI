"""Estimate local delay-coordinate signal support across the received LoRa chirps."""

import torch


EMBEDDING = 24
LOCAL_WIDTH = 128
LOCAL_HOP = 64


def subspace_filter(observed,noise_rms,width,hop,*,exclude_self):
    """Build a Wiener-like filter in the local delayed sample coordinates."""
    whitened = observed / noise_rms.clamp_min(.01)
    embeddings = whitened.unfold(-1,width,hop).unfold(-1,EMBEDDING,1)
    covariance = embeddings.mH @ embeddings / embeddings.shape[-2]
    pooled = covariance.sum(1,keepdim=True)
    if exclude_self:
        pooled = (pooled-covariance)/(observed.shape[1]-1)
    else:
        pooled = pooled/observed.shape[1]
    values, vectors = torch.linalg.eigh(pooled)
    gain = (1-values.clamp_min(1e-8).reciprocal()).clamp_min(0.)
    return (vectors*gain.unsqueeze(-2)) @ vectors.mH


def diagonal_average(trajectory,width):
    n,d = trajectory.shape[-2:]
    row=torch.arange(n,device=trajectory.device)
    delay=torch.arange(d,device=trajectory.device)
    place=(row[:,None]+delay[None,:]).flatten()
    target=trajectory.new_zeros((*trajectory.shape[:-2],width))
    target.index_add_(-1,place,trajectory.flatten(-2))
    count=torch.bincount(place,minlength=width).to(trajectory.real.dtype)
    return target/count


def apply_subspace(value,operator,width,hop):
    n=value.shape[-1]
    patch=value.unfold(-1,width,hop)
    vectors=patch.unfold(-1,EMBEDDING,1)
    local=diagonal_average(vectors @ operator,width)
    t=torch.arange(width,device=value.device,dtype=value.real.dtype)
    taper=.25+.75*(torch.pi*(t+.5)/width).sin().square()
    starts=torch.arange(patch.shape[-2],device=value.device)*hop
    positions=(starts[:,None]+torch.arange(width,device=value.device)[None,:]).flatten()
    output=value.new_zeros(value.shape)
    output.index_add_(-1,positions,(local*taper).flatten(-2))
    weight=taper.new_zeros(n)
    weight.index_add_(-1,positions,taper.repeat(len(starts)))
    return output/weight.clamp_min(1e-8)


def match_dose(reference,edited):
    """Cap the protected edit and match an unprojected control to its energy."""
    a=torch.linalg.vector_norm(reference,dim=-1,keepdim=True)
    b=torch.linalg.vector_norm(edited,dim=-1,keepdim=True)
    protected=edited*(a/b.clamp_min(1e-10)).clamp_max(1.)
    control=reference*(b/a.clamp_min(1e-10)).clamp_max(1.)
    return protected,control
