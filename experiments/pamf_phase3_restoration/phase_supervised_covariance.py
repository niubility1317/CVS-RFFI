"""Scratch physical recovery with an explicit coherent-reference phase branch."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .covariance_inverse import dense_action
from .covariance_model import CovarianceMAP, source_action_loss


def reference_basis(reference, taps):
    rows = torch.arange(len(reference), device=reference.device)[:,None]
    shifts = torch.arange(taps, device=reference.device)[None,:]
    return reference[(rows-shifts).abs()]


def unit_direction(value, floor):
    active = torch.where(value.abs()>floor,value,torch.ones_like(value))
    return active/active.abs()


def coherent_direction(reference, response):
    return unit_direction((reference.conj()*response).sum(1),
                          1e-6*reference.norm()*response.norm(dim=1))


def canonical_factors(kernel, main, image, reference, basis):
    direction = coherent_direction(reference,kernel@basis.T)[:,None]
    return kernel*direction.conj(),main*direction,image*direction.conj()


def circular_supervision(angle, main_shape, canonical_main):
    target = unit_direction((main_shape.detach().conj()*canonical_main).sum(1),1e-8)
    predicted = torch.polar(torch.ones_like(angle),angle)
    difference = predicted*target.conj()
    return (1-difference.real).clamp_min(0), difference.angle().abs()*(180/math.pi)


class PacketShapeHead(nn.Module):
    """Separate positive main amplitude and a zero-intercept phase trajectory."""
    def __init__(self):
        super().__init__()
        self.gains = nn.Linear(128,3)
        self.slopes = nn.Linear(128,2)
        self.local_phase = nn.Conv1d(128,2,3,padding=1)
        self.register_buffer('phase_time',torch.linspace(-.5,.5,256)[None,None])
        for layer in (self.gains,self.slopes,self.local_phase):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self,time_tokens,pooled):
        gain = self.gains(pooled)
        main = F.softplus(gain[:,0]+math.log(math.expm1(1.)))
        image = torch.complex(gain[:,1],gain[:,2])
        local = self.local_phase(time_tokens.transpose(1,2))
        local = F.interpolate(local,size=self.phase_time.shape[-1],mode='linear',align_corners=False)
        local = local-local.mean(dim=-1,keepdim=True)
        linear = (local*self.phase_time).sum(dim=-1,keepdim=True)/self.phase_time.square().sum()
        phase = local+(self.slopes(pooled)[:,:,None]-linear)*self.phase_time
        rotation = torch.polar(torch.ones_like(phase),phase)
        return main[:,None]*rotation[:,0],image[:,None]*rotation[:,1]


class MainPhaseHead(nn.Module):
    """Learn the weights on coherent evidence in the observed packet."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(8,32,7,stride=2,padding=3),nn.GELU(),
            nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU())
        self.angle = nn.Linear(65,1)
        nn.init.zeros_(self.angle.weight)
        nn.init.zeros_(self.angle.bias)

    def forward(self,received,main_anchor,image_anchor,variance):
        observed = torch.complex(*received.unbind(1))
        energy = observed.abs().square().mean(1).clamp_min(1e-6)
        scale = energy.sqrt()[:,None]
        residual = (observed-image_anchor)/scale
        main = main_anchor/scale
        image = image_anchor/scale
        coherence = residual*main.conj()
        features = torch.stack((residual.real,residual.imag,main.real,main.imag,
            image.real,image.imag,coherence.real,coherence.imag),dim=1)
        encoded = self.encoder(features).mean(2)
        noise = (variance/energy).log().clamp(-16,4)
        return self.angle(torch.cat((encoded,noise[:,None]),dim=1)).squeeze(1)


class PhaseSupervisedCovarianceMAP(CovarianceMAP):
    def __init__(self):
        super().__init__()
        self.coefficient_head = PacketShapeHead()
        self.phase_head = MainPhaseHead()

    def estimate_modulation(self,time_tokens,pooled):
        return self.coefficient_head(time_tokens,pooled)

    def estimate(self,received,*,dense=True):
        aux = super().estimate(received,dense=False)
        kernel,main,image = (aux[name] for name in ('kernel','main','image'))
        reference = torch.complex(*self.prior_mean.reshape(2,256).unbind(0))
        basis = reference_basis(reference,kernel.shape[1])
        response = kernel@basis.T
        direction = coherent_direction(reference,response).conj()[:,None]
        kernel,response = kernel*direction,response*direction
        angle = self.phase_head(received,main*response,image*response.conj(),aux['variance'].detach())
        rotated = main*torch.polar(torch.ones_like(angle),angle)[:,None]
        if dense:
            action = dense_action(kernel.to(torch.complex128),rotated.to(torch.complex128),image.to(torch.complex128))
            return dict(action_matrix=action,variance=aux['variance'])
        return dict(kernel=kernel,main=rotated,image=image,variance=aux['variance'],main_shape=main,
            main_angle=angle,reference=reference,reference_basis=basis)


def phase_supervised_action_loss(aux,kernel,coefficients,variance):
    loss,terms = source_action_loss(aux,kernel,coefficients,variance)
    with torch.no_grad():
        h = torch.complex(*kernel.unbind(1))
        main = torch.complex(coefficients[:,0],coefficients[:,1])
        image = torch.complex(coefficients[:,2],coefficients[:,3])
        h,main,image = canonical_factors(h,main,image,aux['reference'],aux['reference_basis'])
        norm = h.norm(dim=1,keepdim=True)
        h,main,image = h/norm,main*norm,image*norm
    circular,degrees = circular_supervision(aux['main_angle'],aux['main_shape'],main)
    modulation = ((aux['main']-main).abs().square()+(aux['image']-image).abs().square()).sum(1)
    modulation = modulation/(main.abs().square()+image.abs().square()).sum(1).clamp_min(1e-8)
    fir = (aux['kernel']-h).abs().square().sum(1)
    loss = loss+.05*modulation+.02*fir+.1*circular
    return loss,dict(terms,modulation_relative_mse=modulation,fir_shape_squared_error=fir,
                     main_phase_circular_loss=circular,main_phase_absolute_degrees=degrees)


def scratch_phase_supervised_covariance(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PhaseSupervisedCovarianceMAP()
    model.initialization_report = dict(mode='scratch_single_stage_coherent_main_phase',model_seed=seed,
        ancestor_checkpoints=[],training_stages=1,prior_is_fitted_source_IQ_statistics=True)
    return model
