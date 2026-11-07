"""Learn observation-dependent prior variance before a Gaussian IQ posterior."""

import torch
from torch import nn
from torch.nn import functional as F

from .covariance_model import scratch_covariance
from .packet_inverse_filter import TemporalBlock
from .projected_prior_detail import fixed_posterior_frame


def conditioned_posterior(action,observed,mean,covariance,variance,added_variance):
    adapted = covariance + torch.diag_embed(added_variance)
    restored,_,_,residual = fixed_posterior_frame(action,observed,mean,adapted,variance)
    return restored,residual


class PriorVarianceEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('position',torch.linspace(-1,1,256)[None,None])
        self.stem = nn.Conv1d(10,64,5,padding=2)
        self.context = nn.Sequential(*(TemporalBlock(64,dilation) for dilation in (1,2,4,8,16,32)))
        self.output = nn.Conv1d(64,2,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,received,mean,diagonal,innovation,variance):
        position = self.position.expand(len(received),-1,-1)
        noise = variance.log()[:,None,None].expand(-1,1,256)
        features = torch.cat((received,mean,diagonal.log(),innovation,noise,position),dim=1)
        raw = self.output(self.context(self.stem(features)))
        return .01*diagonal*F.softplus(raw)


class ConditionedCovariance(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.physical = scratch_covariance(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail = PriorVarianceEstimator()

    def _readout(self,received,prior):
        aux = self.physical.estimate(received)
        action = aux['action_matrix'].detach()
        variance = aux['variance'].detach()
        mean,covariance = ((self.physical.prior_mean,self.physical.prior_covariance) if prior is None else prior)
        mean,covariance = mean.detach(),covariance.detach()
        batch_mean = mean.expand(len(received),-1) if mean.ndim==1 else mean
        diagonal = covariance.diagonal(dim1=-2,dim2=-1)
        batch_diagonal = diagonal.expand(len(received),-1) if diagonal.ndim==1 else diagonal
        innovation = received.flatten(1).double()-(action@batch_mean.double()[:,:,None]).squeeze(-1)
        added = self.detail(received,batch_mean.float().reshape_as(received),
            batch_diagonal.float().reshape_as(received),innovation.float().reshape_as(received),variance)
        restored,residual = conditioned_posterior(action,received.flatten(1).double(),mean.double(),
            covariance.double(),variance.double(),added.flatten(1).double())
        restored = restored.float().reshape_as(received)
        return restored,dict(aux,base_iq=restored,relative_solve_residual=residual,added_variance=added,
            added_variance_relative_to_prior=(added.flatten(1)/batch_diagonal).mean(1))

    def forward(self,received,*,return_aux=False):
        restored,aux = self._readout(received,None)
        return (restored,aux) if return_aux else restored

    def source_readout(self,received,*,prior=None):
        return self._readout(received,prior)


def scratch_conditioned_covariance(seed):
    model = ConditionedCovariance(seed)
    model.initialization_report = dict(mode='scratch_observation_conditioned_prior_variance',
        model_seed=seed,variance_seed=seed+1,ancestor_checkpoints=[],training_stages=1,
        conditional_covariance='source covariance plus nonnegative per-coordinate diagonal',
        initial_added_variance_fraction=.01*float(F.softplus(torch.tensor(0.))))
    return model
