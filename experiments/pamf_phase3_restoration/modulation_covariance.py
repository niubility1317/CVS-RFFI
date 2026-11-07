"""Receiver-aware main/image modulation with factor-supervised Gaussian restoration."""
import torch
from torch import nn
from torch.nn import functional as F

from .covariance_model import CovarianceMAP, source_action_loss
from .factorized_action_loss import aligned_factors


class ReceiverModulationHead(nn.Module):
    """Predict a complex gain, a phase slope and a local phase residual per branch."""
    def __init__(self):
        super().__init__()
        self.gains = nn.Linear(128,4)
        self.slopes = nn.Linear(128,2)
        self.local_phase = nn.Conv1d(128,2,3,padding=1)
        self.register_buffer('phase_time',torch.linspace(-.5,.5,256)[None,None])
        for layer in (self.gains,self.slopes,self.local_phase):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self,time_tokens,pooled):
        gain = self.gains(pooled)
        main = 1+torch.complex(gain[:,0],gain[:,1])
        image = torch.complex(gain[:,2],gain[:,3])
        local = self.local_phase(time_tokens.transpose(1,2))
        local = F.interpolate(local,size=self.phase_time.shape[-1],mode='linear',align_corners=False)
        local = local-local.mean(dim=-1,keepdim=True)
        linear = (local*self.phase_time).sum(dim=-1,keepdim=True)/self.phase_time.square().sum()
        phase = local+(self.slopes(pooled)[:,:,None]-linear)*self.phase_time
        rotation = torch.polar(torch.ones_like(phase),phase)
        return main[:,None]*rotation[:,0],image[:,None]*rotation[:,1]


class ModulationCovarianceMAP(CovarianceMAP):
    def __init__(self):
        super().__init__()
        self.coefficient_head = ReceiverModulationHead()

    def estimate_modulation(self,time_tokens,pooled):
        return self.coefficient_head(time_tokens,pooled)


def modulation_supervision(aux,target_kernel,target_main,target_image):
    estimate = tuple(aux[name].to(torch.complex128) for name in ('kernel','main','image'))
    kernel,main,image = estimate
    with torch.no_grad():
        _,target,_ = aligned_factors(tuple(value.detach() for value in estimate),
                                    (target_kernel,target_main,target_image))
        gain = kernel.norm(dim=1,keepdim=True)
    true_h,true_main,true_image = target
    modulation = ((main*gain-true_main).abs().square()+(image*gain-true_image).abs().square()).sum(1)
    energy = (true_main.abs().square()+true_image.abs().square()).sum(1).clamp_min(1e-8)
    fir = (kernel/kernel.norm(dim=1,keepdim=True)-true_h).abs().square().sum(1)
    return (modulation/energy).float(),fir.float()


def modulation_action_loss(aux,kernel,coefficients,variance):
    loss,terms = source_action_loss(aux,kernel,coefficients,variance)
    h = torch.complex(*kernel.double().unbind(1))
    coeff = coefficients.double()
    main,image = torch.complex(coeff[:,0],coeff[:,1]),torch.complex(coeff[:,2],coeff[:,3])
    modulation,fir = modulation_supervision(aux,h,main,image)
    return loss+.05*modulation+.02*fir,dict(terms,modulation_relative_mse=modulation,fir_shape_squared_error=fir)


def scratch_modulation_covariance(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ModulationCovarianceMAP()
    model.initialization_report = dict(mode='scratch_single_stage_receiver_modulation',model_seed=seed,
        ancestor_checkpoints=[],training_stages=1,prior_is_fitted_source_IQ_statistics=True)
    return model
