"""Observation-space phase marginalization using a shared Gaussian factor."""
import math

import torch


def fixed_grid(device):
    phase = torch.arange(-128,128,dtype=torch.float64,device=device) * (math.pi/128)
    drift = torch.arange(-4,5,dtype=torch.float64,device=device) * (math.pi/16)
    log_prior = -.5*((phase[None,:]/(math.pi/12)).square()+(drift[:,None]/(math.pi/12)).square())
    return phase,drift,log_prior


def calibrated_action(action, gamma):
    """Left-multiply by a real block representation of a complex timewise gain."""
    i,q = action.chunk(2,dim=1)
    re,im = gamma.real[:,:,None],gamma.imag[:,:,None]
    return torch.cat((re*i-im*q,im*i+re*q),dim=1)


def ramp_posteriors(action, observed, mean, covariance, variance, phase, drift, log_prior):
    """Return scalar-control, phase-only and phase+drift posterior means.

    The phase rotation acts on the output of the fixed per-record action.
    Scalar noise is invariant, so all hypotheses have the same determinant.
    The whitened score is quadratic in cos(phase) and sin(phase) for each drift.
    """
    d = observed.shape[1]
    time = torch.linspace(-.5,.5,d//2,device=observed.device,dtype=observed.dtype)
    ramp = drift[:,None]*time
    sine,cosine = ramp.sin().T,ramp.cos().T
    i,q = observed.chunk(2,dim=1)
    # P is the observation with the inverse drift. J P is its quarter-turn.
    ip = i[:,:,None]*cosine + q[:,:,None]*sine
    qp = q[:,:,None]*cosine - i[:,:,None]*sine
    p = torch.cat((ip,qp),dim=1)
    jp = torch.cat((-qp,ip),dim=1)
    ac = action @ covariance
    normal = ac @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    factor = torch.linalg.cholesky(normal)
    rhs = torch.cat((p,jp,(action@mean)[:,:,None]),dim=2)
    p,jp,anchor = torch.linalg.solve_triangular(factor,rhs,upper=False).split(
        [len(drift),len(drift),1],dim=2)
    c,s = phase.cos(),phase.sin()
    # Remove the shared anchor norm from every log likelihood.
    loss = (p.square().sum(1)[:,:,None]*c.square() + jp.square().sum(1)[:,:,None]*s.square()
            -2*(p*jp).sum(1)[:,:,None]*c*s -2*(p*anchor).sum(1)[:,:,None]*c
            +2*(jp*anchor).sum(1)[:,:,None]*s)
    scores = -.5*loss + log_prior
    center = len(drift)//2
    w_phase = scores[:,center].softmax(dim=1)
    w_joint = scores.flatten(1).softmax(dim=1).reshape_as(scores)
    phase_c,phase_s = w_phase@c,w_phase@s
    joint_c,joint_s = w_joint@c,w_joint@s
    control = p[:,:,center]
    phase_only = control*phase_c[:,None] - jp[:,:,center]*phase_s[:,None]
    joint = (p*joint_c[:,None] - jp*joint_s[:,None]).sum(dim=2)
    innovation = torch.stack((control,phase_only,joint),dim=2)-anchor
    solved = torch.linalg.solve_triangular(factor.transpose(-1,-2),innovation,upper=True)
    output = mean + (ac.transpose(-1,-2) @ solved).transpose(1,2)
    phase_gain = torch.complex(phase_c,phase_s)[:,None].expand(-1,len(time))
    ramp_gain = torch.complex(joint_c,joint_s) @ torch.exp(1j*ramp)
    return output,torch.stack((phase_gain,ramp_gain),dim=1)
