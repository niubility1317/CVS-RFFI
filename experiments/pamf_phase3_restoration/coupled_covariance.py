"""Positive-definite coupled source-mode covariance with an identity start."""
import math

import torch
from torch import nn

from .projected_prior_detail import fixed_posterior_frame
from .signed_covariance import RANK,SignedModeEstimator,SpectralPhysical

MODE_MARGIN = 1e-5


def positive_diagonal(raw):
    """sqrt(1+r^2)+r, evaluated stably in its polynomial negative tail."""
    root = torch.hypot(raw,torch.ones_like(raw))
    reciprocal = (root+(-raw).clamp_min(0)).reciprocal()
    return torch.where(raw>=0,root+raw,reciprocal)


def positive_diagonal_slope(raw):
    root = torch.hypot(raw,torch.ones_like(raw))
    negative = (root*(root+(-raw).clamp_min(0))).reciprocal()
    return torch.where(raw>=0,1+raw/root,negative)


def mode_factor(raw,rank=RANK):
    if raw.ndim != 2 or raw.shape[1] != rank*(rank+1)//2:
        raise ValueError('One lower-triangular source-mode factor per record required')
    rows,columns = torch.tril_indices(rank,rank,device=raw.device)
    matrix = torch.zeros(len(raw),rank,rank,device=raw.device,dtype=raw.dtype)
    matrix[:,rows,columns] = raw
    diagonal = matrix.diagonal(dim1=-2,dim2=-1)
    return torch.tril(matrix,-1)/math.sqrt(rank) + torch.diag_embed(positive_diagonal(diagonal))


def normalize_modes(covariance,modes):
    """Give the actual stored FP32 factor a strict source-whitened margin."""
    c = covariance.detach().float().double()
    factor = torch.linalg.cholesky(c)
    whitened = torch.linalg.solve_triangular(factor,modes.detach().double(),upper=False)
    upper = torch.linalg.eigvalsh(whitened.T@whitened).max()
    if not torch.isfinite(upper) or upper<=0:
        raise ValueError('Nonzero finite source mode factor required')
    scaled = (modes.detach().double()*torch.sqrt((1-MODE_MARGIN)/upper)).float()
    actual = torch.linalg.solve_triangular(factor,scaled.double(),upper=False)
    if torch.linalg.eigvalsh(actual.T@actual).max() > 1-1e-6:
        raise ValueError('Stored source modes do not preserve the strict SPD margin')
    return scaled


def coupled_covariance(covariance,modes,factor):
    # Form the Gram update in FP64, including when the neural factor is FP32.
    k = factor.to(covariance)
    identity = torch.eye(k.shape[-1],device=k.device,dtype=k.dtype)
    change = k@k.transpose(-1,-2)-identity
    return covariance + (modes@change)@modes.transpose(-1,-2),change


class CoupledPhysical(SpectralPhysical):
    def bind_prior(self,mean,covariance):
        super().bind_prior(mean,covariance)
        self.prior_modes.copy_(normalize_modes(self.prior_covariance,self.prior_modes))


class CoupledModeEstimator(SignedModeEstimator):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(64,RANK*(RANK+1)//2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class CoupledCovariance(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical = CoupledPhysical()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail = CoupledModeEstimator()

    def _readout(self,received,prior):
        aux = self.physical.estimate(received)
        action,variance = aux['action_matrix'].detach(),aux['variance'].detach()
        mean,covariance,modes = ((self.physical.prior_mean,self.physical.prior_covariance,self.physical.prior_modes)
                                 if prior is None else prior)
        mean,covariance,modes = mean.detach(),covariance.detach(),modes.detach()
        batch_mean = mean.expand(len(received),-1)
        diagonal = covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
        innovation = received.flatten(1).double()-(action@batch_mean.double()[:,:,None]).squeeze(-1)
        raw = self.detail(received,batch_mean.float().reshape_as(received),diagonal.float().reshape_as(received),
                          innovation.float().reshape_as(received),variance)
        factor = mode_factor(raw)
        adapted,change = coupled_covariance(covariance.double(),modes.double(),factor)
        restored,_,_,residual = fixed_posterior_frame(action,received.flatten(1).double(),mean.double(),adapted,variance.double())
        restored = restored.float().reshape_as(received)
        return restored,dict(aux,base_iq=restored,relative_solve_residual=residual,
                             mode_factor=factor,mode_change=change,mode_raw=raw)

    def forward(self,received,*,return_aux=False):
        restored,aux = self._readout(received,None)
        return (restored,aux) if return_aux else restored

    def source_readout(self,received,*,prior=None):
        return self._readout(received,prior)


def scratch_coupled_covariance(seed):
    model = CoupledCovariance(seed)
    model.initialization_report = dict(mode='scratch_coupled_source_covariance_factor',model_seed=seed,
        confidence_seed=seed+1,ancestor_checkpoints=[],training_stages=1,source_rank=RANK,
        initial_factor='identity',initial_covariance_change=0.,mode_margin=MODE_MARGIN,
        positive_factor_diagonal='polynomial_tail_sqrt(1+r^2)+r')
    return model
