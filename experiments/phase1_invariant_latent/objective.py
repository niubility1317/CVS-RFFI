"""Unlabeled paired-view latent consistency with variance and covariance terms."""
import torch
from torch.nn import functional as F

def latent_loss(latent,groups):
    if latent.ndim!=2 or latent.shape[1]!=64 or groups.shape!=(len(latent),) or len(latent)<6 or len(latent)%3:
        raise ValueError('Three paired source views and corresponding receiver groups required')
    size=len(latent)//3;g=groups.reshape(3,size)
    if not torch.equal(g[0],g[1]) or not torch.equal(g[0],g[2]):
        raise ValueError('Paired source views must preserve receiver ordering')
    if not torch.isfinite(latent).all():raise FloatingPointError('Finite source latent required')
    z=latent.reshape(3,size,64)
    inv=((z[0]-z[1]).square().mean()+(z[0]-z[2]).square().mean()+(z[1]-z[2]).square().mean())/3
    variance=F.relu(1-(latent.var(dim=0,unbiased=False)+1e-4).sqrt()).mean()
    centered=latent-latent.mean(0,keepdim=True)
    cov=centered.T@centered/(len(latent)-1)
    covariance=(cov.square().sum()-cov.diagonal().square().sum())/latent.shape[1]
    return 0.1*inv+0.01*variance+0.001*covariance,dict(invariance=inv,variance=variance,covariance=covariance)
