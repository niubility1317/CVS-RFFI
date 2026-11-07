"""Scratch single30 signed covariance adaptation after the active GPU6 run."""
from functools import partial
import time

import torch

from . import conditioned_covariance_shared_trial as predecessor
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .covariance_detail import detail_objective
from .covariance_inverse import gaussian_posterior
from .mismatched_prior_training import choose_prior,load_training_priors
from .signed_covariance import RANK,source_modes,scratch_signed_covariance
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.signed_covariance_trial'
ARM = 'TF_SignedCovariance'
original_controller = trial.controller


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed signed covariance candidate is allowed')
    cfg = predecessor.configuration(predecessor.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        architecture='Original scratch physical estimator + six full-rate64-channel blocks + global64-mode head + one FP64 posterior.',
        conditional_covariance='C + B diag(.9softsign(raw64)) B^T; B is fixed source top64 eigenvectors times sqrt(eigenvalues).',
        learned_detail='Signed per-packet coefficients on64 source covariance modes, pooled64-channel temporal context; no IQ decoder.',
        learned_IQ_decoder=False,detail_initial_zero=True,detail_initial_extra_variance_fraction=0.,
        initial_covariance_addition='Exactly zero; deployed prior is exactly original C at scratch initialization.',
        source_prior_rank=RANK,source_prior_mode_change_range=[-.9,.9],
        source_prior_mode_fit='Algebraic eigendecomposition of existing source-only global/other-RX covariance; no new IQ/statistic fit.',
        gradient_routing='Source simulator Action/noise updates physical only; actual noisy IQ/STFT/frozen continuous F0 updates signed-mode CNN only.',
        training_path='Single30, every63000record per epoch. Odd global/even other-RX prior; signed covariance and full FP64 posterior differentiated.',
        scientific_change='Composite signed low-rank covariance/global-head/softsign parameterization; not an isolated activation ablation. Source E02 diagnostic motivates change, no Main score used.',
        source_diagnostic='20261107-diagnostic-Restore-conditional-variance-E02-sourceLU-s2026092701-r01/055f41652902c577bc848a4a8364f4aa7a487b6c',
        matched_control=predecessor.RUN+'/'+predecessor.ARM)
    return cfg


def factory(arm):
    return scratch_signed_covariance(configuration(arm)['model_seed'])


def signed_readout(model,reader,received,clean,features,scale,objective,*,source_indices,epoch,priors,assignment):
    prior = choose_prior(priors,assignment,source_indices,epoch)
    restored,aux = model.source_readout(received,prior=prior)
    loss,metrics = detail_objective(reader,restored,clean,features,scale,objective)
    raw,coefficients = aux['mode_raw'],aux['mode_coefficients']
    metrics.update(signed_mode_absolute_change=coefficients.abs().mean(1),
                   signed_mode_mean_change=coefficients.mean(1),
                   signed_mode_negative_fraction=(coefficients<0).float().mean(1),
                   signed_map_slope=(.9/(1+raw.abs()).square()).mean(1),
                   receiver_mismatch_prior_fraction=torch.full_like(loss,float(prior is not None)))
    return restored,aux,loss,metrics


def initial_smoke(model,received,restored):
    with torch.no_grad():
        _,aux = model(received,return_aux=True)
        if aux['mode_coefficients'].abs().max() != 0:
            raise ValueError('Signed covariance must start with an exact zero update')
        original,_ = gaussian_posterior(aux['action_matrix'],received.flatten(1).double(),
            model.physical.prior_mean.double(),model.physical.prior_covariance.double(),aux['variance'].double())
        torch.testing.assert_close(restored,original.float().reshape_as(received),rtol=2e-5,atol=2e-6)
    return dict(signed_covariance_initial_identity=True,signed_zero_point_slope=.9,source_mode_rank=RANK)


def train(args,out,*,config_for=configuration):
    pair,assignment = load_training_priors(args.root,'cpu')
    modes = torch.stack([source_modes(covariance) for covariance in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair,modes))
    readout = partial(signed_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=config_for,training_readout=readout,smoke_check=initial_smoke)


def previous_gpu_finished(root):
    folder = root/'runs'/predecessor.RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{predecessor.ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code'] != 0 or stage == 'cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_CONDITIONED_COVARIANCE_LAST_GPU_STAGE',flush=True)
    while not previous_gpu_finished(args.root):
        time.sleep(5)
    original_controller(args)


def install():
    trial.RUN,trial.MODULE,trial.ARMS = RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller = configuration,factory,train,controller
    trial.gpu_choice = partial(recovery_gpu_choice,allowed_gpus=(6,),maximum_processes=5)


if __name__ == '__main__':
    install()
    trial.main()
