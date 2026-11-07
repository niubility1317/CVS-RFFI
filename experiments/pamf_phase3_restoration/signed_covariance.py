"""Observation-dependent, positive-definite signed source-covariance adaptation."""
import torch
from torch import nn
from torch.nn import functional as F

from .covariance_model import CovarianceMAP
from .packet_inverse_filter import TemporalBlock
from .projected_prior_detail import fixed_posterior_frame

RANK = 64
CHANGE_LIMIT = .9


def signed_coefficients(raw):
    # Both signs are reachable from an exact identity initialization. Softsign
    # has a polynomial rather than exponential tail derivative.
    return CHANGE_LIMIT * F.softsign(raw)


def source_modes(covariance, rank=RANK):
    """Factor the fixed source covariance's leading modes, without fitting IQ."""
    c = covariance.detach().float().double()
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not 1 <= rank <= len(c) or not torch.isfinite(c).all():
        raise ValueError('Finite square source covariance and valid fixed rank required')
    factor = torch.linalg.cholesky(c)
    values, vectors = torch.linalg.eigh(c)
    modes = (vectors[:,-rank:] * values[-rank:].sqrt()[None]).float()
    # Check the actual stored FP32 factor in the FP64 source-covariance frame.
    # C + B diag(d) B^T stays SPD for |d|<.9 when ||L^-1 B||^2<1/.9.
    whitened = torch.linalg.solve_triangular(factor,modes.double(),upper=False)
    if torch.linalg.eigvalsh(whitened.T @ whitened).max() > 1.01:
        raise ValueError('Rounded source mode factor does not preserve the SPD margin')
    return modes


def adapt_covariance(covariance, modes, coefficients):
    return covariance + (modes * coefficients.unsqueeze(-2)) @ modes.transpose(-1,-2)


class SpectralPhysical(CovarianceMAP):
    def __init__(self):
        super().__init__()
        self.register_buffer('prior_modes',torch.zeros(512,RANK))

    def bind_prior(self,mean,covariance):
        super().bind_prior(mean,covariance)
        self.prior_modes.copy_(source_modes(self.prior_covariance))


class SignedModeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('position',torch.linspace(-1,1,256)[None,None])
        self.stem = nn.Conv1d(10,64,5,padding=2)
        self.context = nn.Sequential(*(TemporalBlock(64,dilation) for dilation in (1,2,4,8,16,32)))
        self.output = nn.Linear(64,RANK)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,received,mean,diagonal,innovation,variance):
        noise = variance.log()[:,None,None].expand(-1,1,256)
        features = torch.cat((received,mean,diagonal.log(),innovation,noise,
                              self.position.expand(len(received),-1,-1)),dim=1)
        return self.output(self.context(self.stem(features)).mean(-1))


class SignedCovariance(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical = SpectralPhysical()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail = SignedModeEstimator()

    def _readout(self,received,prior):
        aux = self.physical.estimate(received)
        action,variance = aux['action_matrix'].detach(),aux['variance'].detach()
        mean,covariance,modes = ((self.physical.prior_mean,self.physical.prior_covariance,self.physical.prior_modes)
                                 if prior is None else prior)
        mean,covariance,modes = mean.detach(),covariance.detach(),modes.detach()
        batch_mean = mean.expand(len(received),-1)
        diagonal = covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
        innovation = received.flatten(1).double() - (action @ batch_mean.double()[:,:,None]).squeeze(-1)
        raw = self.detail(received,batch_mean.float().reshape_as(received),diagonal.float().reshape_as(received),
                          innovation.float().reshape_as(received),variance)
        coefficients = signed_coefficients(raw)
        adapted = adapt_covariance(covariance.double(),modes.double(),coefficients.double())
        restored,_,_,residual = fixed_posterior_frame(action,received.flatten(1).double(),mean.double(),adapted,variance.double())
        restored = restored.float().reshape_as(received)
        return restored,dict(aux,base_iq=restored,relative_solve_residual=residual,
                             mode_coefficients=coefficients,mode_raw=raw)

    def forward(self,received,*,return_aux=False):
        restored,aux = self._readout(received,None)
        return (restored,aux) if return_aux else restored

    def source_readout(self,received,*,prior=None):
        return self._readout(received,prior)


def scratch_signed_covariance(seed):
    model = SignedCovariance(seed)
    model.initialization_report = dict(mode='scratch_signed_low_rank_source_covariance',model_seed=seed,
        confidence_seed=seed+1,ancestor_checkpoints=[],training_stages=1,source_rank=RANK,
        initial_signed_coefficients=0.,signed_fraction_range=[-CHANGE_LIMIT,CHANGE_LIMIT])
    return model
