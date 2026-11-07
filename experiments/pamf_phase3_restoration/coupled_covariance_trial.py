"""Scratch single30 coupled source covariance and complete frozen Main testing."""
from functools import partial

import torch

from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import signed_covariance_parallel_trial as control
from .coupled_covariance import RANK,normalize_modes,positive_diagonal_slope,scratch_coupled_covariance
from .covariance_detail import detail_objective
from .covariance_inverse import gaussian_posterior
from .mismatched_prior_training import choose_prior,load_training_priors
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-CoupledCovariance-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.coupled_covariance_trial'
ARM = 'TF_CoupledCovariance'


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed coupled covariance candidate is allowed')
    cfg = control.configuration(control.ARM)
    cfg.pop('source_prior_mode_change_range',None)
    cfg.update(run_id=RUN,arm=ARM,
        architecture='Original scratch physical estimator + six64-channel temporal blocks +2080 lower-triangular coefficients + one FP64 posterior.',
        conditional_covariance='C + B(KK^T-I)B^T. Stored source B strictly contracts in the original covariance Cholesky frame.',
        learned_detail='64x64 lower-triangular conditional factor. Positive polynomial-tail diagonals, off-diagonal raw/sqrt64, identity zero-head start.',
        initial_covariance_addition='Exactly zero: K=I. Mean and full covariance stay original at scratch initialization.',
        source_prior_mode_fit='Existing top64 source eigenmodes, FP32-rounded then normalized to whitened norm squared about1-1e-5; actual stored bound<=1-1e-6.',
        source_prior_mode_factor='Fixed global and five other-RX factors; no new IQ/statistic fit, no same-RX covariance teacher.',
        source_prior_rank=RANK,mode_factor_coefficients=2080,source_mode_margin=1e-5,
        gradient_routing='Original source Action/noise updates physical only. Actual noisy IQ/STFT/original frozen F0 continuous features update factor CNN only.',
        training_path='All63000 records every epoch; single30 with15global/15other-RX prior exposure, differentiable coupled FP64 posterior.',
        source_diagnostic='20261107-analysis-Restore-source-covariance-capacity-s2026110707-r01/63afe5cb59da641d5bee751a182facaa9aad498d',
        scientific_change='Composite coupled factor head, wider positive variance range and polynomial diagonal. Source geometry motivates it; not an isolated off-diagonal ablation.',
        matched_control=control.RUN+'/'+control.ARM)
    return cfg


def factory(arm):
    return scratch_coupled_covariance(configuration(arm)['model_seed'])


def coupled_readout(model,reader,received,clean,features,scale,objective,*,source_indices,epoch,priors,assignment):
    prior = choose_prior(priors,assignment,source_indices,epoch)
    restored,aux = model.source_readout(received,prior=prior)
    loss,metrics = detail_objective(reader,restored,clean,features,scale,objective)
    factor,change = aux['mode_factor'],aux['mode_change']
    raw = aux['mode_raw']
    rows,columns = torch.tril_indices(RANK,RANK,device=raw.device)
    raw_diagonal = raw[:,rows==columns]
    off_diagonal = change-torch.diag_embed(change.diagonal(dim1=-2,dim2=-1))
    metrics.update(coupled_change_Frobenius_RMS=(change.square().sum((1,2))/RANK).sqrt(),
        coupled_off_diagonal_RMS=(off_diagonal.square().sum((1,2))/RANK).sqrt(),
        factor_diagonal_mean=factor.diagonal(dim1=-2,dim2=-1).mean(1),
        factor_diagonal_min=factor.diagonal(dim1=-2,dim2=-1).min(1).values,
        factor_diagonal_max=factor.diagonal(dim1=-2,dim2=-1).max(1).values,
        factor_diagonal_map_slope=positive_diagonal_slope(raw_diagonal).mean(1),
        receiver_mismatch_prior_fraction=torch.full_like(loss,float(prior is not None)))
    return restored,aux,loss,metrics


def initial_smoke(model,received,restored):
    with torch.no_grad():
        _,aux = model(received,return_aux=True)
        if aux['mode_change'].abs().max()!=0:
            raise ValueError('Coupled factor must start with exactly zero covariance change')
        original,_ = gaussian_posterior(aux['action_matrix'],received.flatten(1).double(),
            model.physical.prior_mean.double(),model.physical.prior_covariance.double(),aux['variance'].double())
        torch.testing.assert_close(restored,original.float().reshape_as(received),rtol=2e-5,atol=2e-6)
    return dict(coupled_covariance_initial_identity=True,source_mode_rank=RANK,
                factor_diagonal_zero_point_slope=1.,stored_mode_whitened_margin_verified=True)


def train(args,out):
    pair,assignment = load_training_priors(args.root,'cpu')
    modes = torch.stack([normalize_modes(covariance,source_modes(covariance)) for covariance in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair,modes))
    readout = partial(coupled_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout,smoke_check=initial_smoke)


def install():
    trial.RUN,trial.MODULE,trial.ARMS = RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train = configuration,factory,train
    trial.gpu_choice = partial(recovery_gpu_choice,allowed_gpus=(1,),maximum_processes=3)


if __name__=='__main__':
    install()
    trial.main()
