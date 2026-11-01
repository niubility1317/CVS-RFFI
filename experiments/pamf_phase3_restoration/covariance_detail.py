"""Single-stage physical estimator and separately guided nonlinear IQ detail path."""
import torch
from torch import nn
from .covariance_model import scratch_covariance
from .covariance_inverse import gaussian_posterior
from .multiscale import MultiscaleAttentionRestorer


class CovarianceDetail(nn.Module):
    def __init__(self,seed):
        super().__init__()
        self.physical=scratch_covariance(seed)
        # Keep the physical branch's actual scratch initialization identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.detail=MultiscaleAttentionRestorer(context_channels=3)

    def forward(self,received,*,return_aux=False):
        aux=self.physical.estimate(received)
        # Physical parameters receive only simulator supervision. IQ/feature
        # objectives cannot bend A/noise through either the posterior or context.
        with torch.no_grad():
            base,residual=gaussian_posterior(aux['action_matrix'].detach(),received.flatten(1).double(),
                self.physical.prior_mean.double(),self.physical.prior_covariance.double(),aux['variance'].detach().double())
            base=base.float().reshape(-1,2,256)
            context=torch.cat((received.detach(),aux['variance'].detach().log()[:,None,None].expand(-1,1,256)),1)
        restored=self.detail(base,context)
        return (restored,dict(**aux,base_iq=base,relative_solve_residual=residual)) if return_aux else restored


def detail_objective(reader,restored,clean,target_features,scale,paired_objective):
    if target_features.shape!=(len(restored),160) or scale.shape!=(160,):raise ValueError('Matched original source fused160 targets/scale required')
    paired,metrics=paired_objective(restored,clean)
    features=reader.features(restored);target=target_features.detach()
    cosine=1-torch.nn.functional.cosine_similarity(features,target,dim=1,eps=1e-8)
    standardized=((features-target).square()/scale.detach()).mean(1)
    return paired+.01*(cosine+standardized),dict(metrics,feature_error=cosine,feature_standardized_error=standardized)


def clip_branch_gradients(model):
    # A shared norm would let detail gradients rescale the physical update.
    return {name:torch.nn.utils.clip_grad_norm_(getattr(model,name).parameters(),1.,error_if_nonfinite=True)
            for name in ('physical','detail')}


def scratch_covariance_detail(seed):
    model=CovarianceDetail(seed)
    model.initialization_report=dict(mode='scratch_single_stage_two_separate_gradient_branches',model_seed=seed,
        detail_seed=seed+1,ancestor_checkpoints=[],training_stages=1,physical_estimation_TX_labels=False,
        continuous_F0_guidance_detail_only=True,source_prior_statistics_required=True)
    return model
