"""Original joint-prior deployment; equivalent low-rank source learning path."""
import torch
from .joint_prior import JointPrior,bounded_prior_mean
from .coupled_covariance import mode_factor
from .low_rank_posterior import conditional_prior_posterior

EPSILON=.01
RESPONSE_WEIGHT=.01
DIRECTIONS=('conjugate','amplitude_cubic','phase_slope','local_difference')


def selected_perturbation(clean,source_indices,epoch):
    if clean.ndim!=3 or clean.shape[1:]!=(2,256) or source_indices.shape!=(len(clean),) or source_indices.dtype!=torch.long or not 0<=epoch<=30:
        raise ValueError('Original source IQ, explicit source indices and fixed epoch required')
    z=torch.complex(clean[:,0],clean[:,1]);power=z.abs().square();meanpower=power.mean(1,keepdim=True)
    if not torch.isfinite(clean).all() or (meanpower<=0).any():raise ValueError('Finite positive-energy source IQ required')
    t=torch.linspace(-1,1,256,device=clean.device,dtype=clean.dtype)[None]
    candidates=torch.stack((z.conj(),z*(power/meanpower-1),1j*z*t,torch.roll(z,1,-1)-z),1)
    choice=(source_indices+epoch)%4
    raw=candidates[torch.arange(len(clean),device=clean.device),choice]
    norm=raw.abs().square().mean(1,keepdim=True).sqrt()
    if not torch.isfinite(norm).all() or (norm<1e-10).any():raise ValueError('Selected source direction must be nonzero')
    delta=EPSILON*meanpower.sqrt()*raw/norm
    return torch.stack((delta.real,delta.imag),1),choice


def response_loss(base,changed,delta):
    if base.shape!=changed.shape or base.shape!=delta.shape or base.ndim!=3 or base.shape[1:]!=(2,256):
        raise ValueError('Matched IQ response pairs required')
    energy=delta.square().sum((1,2))
    if (energy<=0).any() or not torch.isfinite(energy).all():raise ValueError('Positive finite source perturbation energy required')
    return (changed-base-delta).square().sum((1,2))/energy


class ResponseJointPrior(JointPrior):
    def source_readout(self,received,*,prior=None):
        aux=self.physical.estimate(received)
        action,variance=aux['action_matrix'].detach(),aux['variance'].detach()
        mean,covariance,modes=((self.physical.prior_mean,self.physical.prior_covariance,self.physical.prior_modes) if prior is None else prior)
        mean,covariance,modes=mean.detach(),covariance.detach(),modes.detach()
        batch_mean=mean.expand(len(received),-1)
        diagonal=covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
        innovation=received.flatten(1).double()-(action@batch_mean.double()[:,:,None]).squeeze(-1)
        raw,mean_raw=self.detail(received,batch_mean.float().reshape_as(received),diagonal.float().reshape_as(received),innovation.float().reshape_as(received),variance)
        adapted_mean,delta,standardized=bounded_prior_mean(batch_mean.float(),diagonal.float(),mean_raw)
        factor=mode_factor(raw);k=factor.double()
        change=k@k.transpose(-1,-2)-torch.eye(k.shape[-1],device=k.device,dtype=k.dtype)
        restored,residual=conditional_prior_posterior(action,received.flatten(1).double(),adapted_mean.double(),covariance,modes,factor,variance)
        restored=restored.float().reshape_as(received)
        return restored,dict(aux,base_iq=restored,relative_solve_residual=residual,mode_factor=factor,mode_change=change,
            mode_raw=raw,mean_update=delta,mean_standardized_update=standardized)


def scratch_response_joint_prior(seed):
    model=ResponseJointPrior(seed)
    model.initialization_report=dict(mode='scratch_response_joint_prior',model_seed=seed,confidence_seed=seed+1,
        ancestor_checkpoints=[],training_stages=1,initial_factor='identity',initial_mean_change=0.,
        training_solver='equivalent_low_rank_FP64',deployment_solver='original_dense_FP64',new_learned_parameters=0)
    return model
