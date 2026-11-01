"""Noise-adaptive blind measurement proximal update."""
import torch

def noise_adaptive_prox(A,x,prior,noise_variance,prior_sigma,multiplier):
    if A.ndim!=3 or A.shape[1]!=A.shape[2] or x.shape!=prior.shape or x.ndim!=2 or A.shape[:2]!=x.shape or noise_variance.shape!=(len(x),) or prior_sigma.shape!=(len(x),) or multiplier<=0:raise ValueError('Matched square system and positive calibration required')
    if (noise_variance<=0).any() or (prior_sigma<=0).any():raise ValueError('Positive noise and prior uncertainty required')
    at=A.transpose(1,2);weight=multiplier*noise_variance/(.5*prior_sigma.square()).clamp_min(1e-12);eye=torch.eye(A.shape[-1],device=A.device,dtype=A.dtype);system=at@A+weight[:,None,None]*eye;rhs=at@x.unsqueeze(-1)+weight[:,None,None]*prior.unsqueeze(-1)
    return torch.cholesky_solve(rhs,torch.linalg.cholesky(system)).squeeze(-1)

def reconstruct_noise_adaptive(net,A,received,parent,noise_variance,multiplier):
    current=parent
    for frac in (.20,.10,.05):
        rms=current.square().sum(1).mean(1).sqrt().clamp_min(1e-8);sigma=frac*rms;prior=net(current,sigma);current=noise_adaptive_prox(A,received.reshape(len(received),-1),prior.reshape(len(received),-1),noise_variance,sigma,multiplier).reshape_as(received)
    return current
