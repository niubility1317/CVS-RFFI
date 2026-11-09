"""Source same-record ranking; it is not transmitter-identity supervision."""
import math
import torch
from experiments.phase1_cumulant_context.model import cumulant_features

TEMPERATURE=.5
RANK_WEIGHT=.002
NORMALIZATION_IDS=512


def power_features(iq):
    if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('IQ windows required')
    z=torch.complex(iq[:,0],iq[:,1])
    spectrum=torch.fft.fftshift(torch.fft.fft(z,dim=1).abs().square(),dim=1)
    bands=spectrum.reshape(len(iq),32,8).mean(2)
    return torch.log((bands/bands.mean(1,keepdim=True).clamp_min(1e-12)).clamp_min(1e-8))


def raw_features(iq):
    return power_features(iq),cumulant_features(iq)


def normalize_features(pair,stats):
    if len(pair)!=2 or len(stats)!=2:raise ValueError('Two source feature blocks required')
    normalized=[]
    for x,(mean,scale) in zip(pair,stats):
        if x.ndim!=2 or mean.shape!=x.shape[1:] or scale.shape!=mean.shape:
            raise ValueError('Matching source normalization required')
        normalized.append((x-mean)/scale/math.sqrt(x.shape[1]))
    return torch.cat(normalized,1)


def rank_loss(query,reference,record_ids):
    """Match source physical record; duplicate views of the record are positives."""
    if query.ndim!=2 or query.shape!=reference.shape or len(query)<2 or record_ids.shape!=(len(query),):
        raise ValueError('Matching features and source record IDs required')
    scores=-(query[:,None,:]-reference[None,:,:]).square().sum(-1)/TEMPERATURE
    positive=record_ids[:,None]==record_ids[None,:]
    return (torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),1)).mean()
