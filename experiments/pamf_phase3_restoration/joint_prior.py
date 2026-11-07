"""A bounded conditional mean and coupled covariance in one Gaussian posterior."""
import torch
from torch import nn
from torch.nn import functional as F

from .coupled_covariance import CoupledCovariance,CoupledModeEstimator,CoupledPhysical,coupled_covariance,mode_factor
from .projected_prior_detail import fixed_posterior_frame


def bounded_prior_mean(base_mean,diagonal,raw):
    """At most one prior standard deviation per real IQ coordinate."""
    standardized=F.softsign(raw)
    delta=diagonal.sqrt().reshape_as(raw)*standardized
    return base_mean+delta.flatten(1),delta,standardized


class JointPriorEstimator(CoupledModeEstimator):
    def __init__(self):
        super().__init__()
        self.mean_output=nn.Conv1d(64,2,1)
        nn.init.zeros_(self.mean_output.weight)
        nn.init.zeros_(self.mean_output.bias)

    def forward(self,received,mean,diagonal,innovation,variance):
        noise=variance.log()[:,None,None].expand(-1,1,256)
        features=torch.cat((received,mean,diagonal.log(),innovation,noise,
                            self.position.expand(len(received),-1,-1)),dim=1)
        hidden=self.context(self.stem(features))
        return self.output(hidden.mean(-1)),self.mean_output(hidden)


class JointPrior(CoupledCovariance):
    def __init__(self,seed):
        nn.Module.__init__(self)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.physical=CoupledPhysical()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail=JointPriorEstimator()

    def _readout(self,received,prior):
        aux=self.physical.estimate(received)
        action,variance=aux['action_matrix'].detach(),aux['variance'].detach()
        mean,covariance,modes=((self.physical.prior_mean,self.physical.prior_covariance,self.physical.prior_modes)
                              if prior is None else prior)
        mean,covariance,modes=mean.detach(),covariance.detach(),modes.detach()
        batch_mean=mean.expand(len(received),-1)
        diagonal=covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
        innovation=received.flatten(1).double()-(action@batch_mean.double()[:,:,None]).squeeze(-1)
        raw,mean_raw=self.detail(received,batch_mean.float().reshape_as(received),diagonal.float().reshape_as(received),
                                 innovation.float().reshape_as(received),variance)
        adapted_mean,delta,standardized=bounded_prior_mean(batch_mean.float(),diagonal.float(),mean_raw)
        factor=mode_factor(raw)
        adapted_covariance,change=coupled_covariance(covariance.double(),modes.double(),factor)
        restored,_,_,residual=fixed_posterior_frame(action,received.flatten(1).double(),adapted_mean.double(),
                                                   adapted_covariance,variance.double())
        restored=restored.float().reshape_as(received)
        return restored,dict(aux,base_iq=restored,relative_solve_residual=residual,
            mode_factor=factor,mode_change=change,mode_raw=raw,mean_update=delta,mean_standardized_update=standardized)


def scratch_joint_prior(seed):
    model=JointPrior(seed)
    model.initialization_report=dict(mode='scratch_joint_mean_and_coupled_covariance',model_seed=seed,
        confidence_seed=seed+1,ancestor_checkpoints=[],training_stages=1,initial_factor='identity',
        initial_mean_change=0.,conditional_mean_coordinate_bound='one original prior standard deviation via softsign',
        direct_post_posterior_decoder=False)
    return model
