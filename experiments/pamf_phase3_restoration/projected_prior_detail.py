"""Learn a conditional IQ prior mean and pass its update through the posterior."""

import torch

from .covariance_detail import CovarianceDetail, detail_objective


def fixed_posterior_frame(action, observed, mean, covariance, variance):
    ac = action @ covariance
    normal = ac @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    factor = torch.linalg.cholesky(normal)
    innovation = (observed-action@mean)[:,:,None]
    solved = torch.cholesky_solve(innovation,factor)
    error = (normal@solved-innovation).squeeze(2).norm(dim=1)/innovation.squeeze(2).norm(dim=1).clamp_min(1e-30)
    base = mean+(ac.transpose(-1,-2)@solved).squeeze(2)
    if not torch.isfinite(base).all() or not torch.isfinite(error).all():
        raise ValueError('Nonfinite conditioned prior frame')
    return base,ac,factor,error


def prior_update(action, ac, factor, delta):
    """Apply I-KA to learned mean updates without forming a dense projector."""
    passed = torch.cholesky_solve(action@delta,factor)
    return delta-ac.transpose(-1,-2)@passed


class ProjectedPriorDetail(CovarianceDetail):
    def _readout(self,received,source_clean):
        aux = self.physical.estimate(received)
        with torch.no_grad():
            action = aux['action_matrix'].detach()
            variance = aux['variance'].detach()
            mean = self.physical.prior_mean.double()
            base,ac,factor,residual = fixed_posterior_frame(action,received.flatten(1).double(),
                mean,self.physical.prior_covariance.double(),variance.double())
            base_iq = base.float().reshape_as(received)
            context = torch.cat((received.detach(),variance.log()[:,None,None].expand(-1,1,256)),dim=1)
        proposal = (self.detail(base_iq,context)-base_iq).flatten(1).double()
        if source_clean is None:
            updates = proposal[:,:,None]
        else:
            clean = source_clean.detach().flatten(1).double()
            updates = torch.stack((proposal,clean-mean),dim=2)
        projected = prior_update(action,ac,factor,updates)
        correction = projected[:,:,0]
        restored = (base+correction).float().reshape_as(received)
        debug = dict(aux,base_iq=base_iq,relative_solve_residual=residual)
        if source_clean is not None:
            shrinkage = projected[:,:,1].detach()
            debug.update(source_noiseless_corrected_iq=(clean-shrinkage+correction).float().reshape_as(received),
                source_shrinkage=shrinkage.float().reshape_as(received),
                projected_prior_update=correction.float().reshape_as(received))
        return restored,debug

    def forward(self,received,*,return_aux=False):
        restored,aux = self._readout(received,None)
        return (restored,aux) if return_aux else restored

    def source_readout(self,received,clean):
        return self._readout(received,clean)


def shrinkage_training_readout(model,reader,received,clean,features,scale,paired_objective):
    restored,aux = model.source_readout(received,clean)
    loss,values = detail_objective(reader,aux['source_noiseless_corrected_iq'],clean,features,scale,paired_objective)
    energy = clean.square().mean((1,2)).clamp_min(1e-4)
    metrics = {f'noiseless_{key}':value for key,value in values.items()}
    metrics.update(source_shrinkage_target_nmse=aux['source_shrinkage'].square().mean((1,2))/energy,
        projected_prior_update_nmse=aux['projected_prior_update'].square().mean((1,2))/energy,
        deployed_iq_nmse=(restored-clean).square().mean((1,2))/energy)
    return restored,aux,loss,metrics


def scratch_projected_prior_detail(seed):
    model = ProjectedPriorDetail(seed)
    model.initialization_report = dict(mode='scratch_conditional_prior_shrinkage_correction',model_seed=seed,
        detail_seed=seed+1,ancestor_checkpoints=[],training_stages=1,physical_estimation_TX_labels=False,
        continuous_F0_guidance_detail_only=True,source_prior_statistics_required=True)
    return model
