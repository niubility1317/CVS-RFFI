"""A parameter-free phase coordinate change around the existing IQ prior."""
import torch
from torch import nn


def rotate_iq(value,unit):
    return torch.stack((value[:,0]*unit[:,0]-value[:,1]*unit[:,1],
                        value[:,1]*unit[:,0]+value[:,0]*unit[:,1]),dim=1)


def rotate_inputs(value,unit):
    pairs=value[:,:8].reshape(len(value),4,2,256)
    i=pairs[:,:,0]*unit[:,None,0]-pairs[:,:,1]*unit[:,None,1]
    q=pairs[:,:,1]*unit[:,None,0]+pairs[:,:,0]*unit[:,None,1]
    return torch.cat((torch.stack((i,q),dim=2).flatten(1,2),value[:,8:]),dim=1)


def coordinate(value):
    """Use the strongest coarse-base point; fall back only for a zero base.

    The selection is piecewise differentiable, as max pooling is. No sample is
    amplitude-normalized, permuted or averaged, and the scalar logvar is intact.
    """
    if value.ndim!=3 or value.shape[1:]!=(9,256):
        raise ValueError('Four IQ pairs and one log-variance channel required')
    coarse=value[:,2:4]
    index=coarse.square().sum(1).argmax(1,keepdim=True)
    anchor=coarse.gather(2,index[:,None,:].expand(-1,2,-1))
    pairs=value[:,:8].reshape(len(value),4,2,256)
    fallback_index=pairs.square().sum(2).flatten(1).argmax(1,keepdim=True)
    fallback=pairs.permute(0,2,1,3).flatten(2).gather(2,fallback_index[:,None,:].expand(-1,2,-1))
    anchor=torch.where(anchor.square().sum(1,keepdim=True)>1e-12,anchor,fallback)
    power=anchor.square().sum(1,keepdim=True);valid=power>1e-12
    unit=anchor/power.clamp_min(1e-12).sqrt()
    identity=torch.cat((torch.ones_like(power),torch.zeros_like(power)),dim=1)
    unit=torch.where(valid,unit,identity)
    inverse=torch.cat((unit[:,:1],-unit[:,1:]),dim=1)
    return rotate_inputs(value,inverse),unit,valid


class PhasePrior(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core

    def forward(self,value):
        canonical,unit,valid=coordinate(value)
        return rotate_iq(self.core(canonical),unit)*valid.to(value.dtype)


def wrap_priors(model):
    # Reuse the same objects so retained long-convolution aliases survive.
    model.prior=nn.ModuleList(PhasePrior(core) for core in model.prior)
    model.initialization_report.update(prior_phase_coordinate='strongest_coarse_base_point_with_zero_base_fallback',
        phase_prior_extra_parameters=0,phase_prior_core_calls=2,phase_prior_training='scratch_no_inherited_weights',
        phase_prior_zero_IQ_output=True,phase_prior_whole_G_equivariance_claim=False)
    return model
