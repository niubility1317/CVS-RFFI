"""Scratch single30 joint mean/covariance, then the original frozen Main matrix."""
from functools import partial
import time

import torch

from . import coupled_covariance_trial as control
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .coupled_covariance import normalize_modes
from .joint_prior import scratch_joint_prior
from .mismatched_prior_training import load_training_priors
from .signed_covariance import source_modes
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN='20261107-phase1-Restore-JointPrior-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.joint_prior_trial'
ARM='TF_JointPrior'
original_controller=trial.controller


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed joint prior candidate is allowed')
    cfg=control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        architecture='Original physical estimator + shared six64-channel prior CNN +2080 factor coefficients +2-channel mean head + one FP64 posterior.',
        learned_prior_mean=True,learned_IQ_decoder=True,decoder_location='Bounded conditional prior mean, before posterior; no post-posterior decoder.',
        conditional_mean='mu_y=mu_source+sqrt(diag(C_source))*softsign(mean_head(y,source prior,innovation,estimated noise)).',
        conditional_mean_bound='At most one original source-prior standard deviation per real IQ coordinate; fixed before Main, no V/target fitting.',
        initial_mean_addition='Exactly zero; all existing coupled-control parameters initialize identically.',
        learned_detail='Shared full-rate context predicts bounded conditional mean and the original coupled positive-definite64 factor.',
        gradient_routing='Original source Action/noise updates physical only. Actual noisy IQ/STFT/frozen F0 continuous features update both prior heads and their shared context.',
        scientific_change='Adds a zero-initialized130-parameter conditional prior mean head to the fixed coupled control, with unchanged losses, priors, orders, budget and inference solver.',
        matched_control=control.RUN+'/'+control.ARM,
        source_mean_evidence='20261107-diagnostic-Restore-prior-factor-sourceV-s2026092701-r01; mean and covariance mismatch both affect fixed source predictions.',
        source_learning_evidence=control.RUN+'/source_family_progress_E20.md; no Main outcomes used to choose this addition.')
    return cfg


def factory(arm):
    return scratch_joint_prior(configuration(arm)['model_seed'])


def joint_readout(*args,**kwargs):
    restored,aux,loss,metrics=control.coupled_readout(*args,**kwargs)
    metrics.update(prior_mean_update_RMS=aux['mean_update'].square().mean((1,2)).sqrt(),
        prior_mean_standardized_update_RMS=aux['mean_standardized_update'].square().mean((1,2)).sqrt())
    return restored,aux,loss,metrics


def initial_smoke(model,received,restored):
    result=control.initial_smoke(model,received,restored)
    with torch.no_grad():
        _,aux=model(received,return_aux=True)
        if aux['mean_update'].count_nonzero():raise ValueError('Joint prior must start with exactly zero mean change')
    return dict(result,conditional_mean_initial_zero=True,conditional_mean_standard_deviation_bound=1.)


def mean_smoke_gradient_check(model):
    gradient=model.detail.mean_output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:
        raise ValueError('Actual source objective must reach the conditional mean head')
    return dict(actual_mean_feature_gradient=True,mean_head_source_gradient_absolute_sum=float(gradient.abs().sum()))


def train(args,out):
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(covariance,source_modes(covariance)) for covariance in pair[1]])
    priors=tuple(value.to('cuda:0') for value in (*pair,modes))
    readout=partial(joint_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout,
                   smoke_check=initial_smoke,smoke_gradient_check=mean_smoke_gradient_check)


def previous_gpu_finished(root):
    folder=root/'runs'/control.RUN
    for stage in ('train','cross_rx','cross_both'):
        path=folder/f'terminal-{control.ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code']!=0 or stage=='cross_both'):return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):print('WAITING_FOR_COUPLED_LAST_GPU_STAGE_NORMAL_GPU1',flush=True)
    while not previous_gpu_finished(args.root):time.sleep(5)
    original_controller(args)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller=configuration,factory,train,controller
    trial.gpu_choice=partial(recovery_gpu_choice,allowed_gpus=(1,),maximum_processes=3)


if __name__=='__main__':
    install()
    trial.main()
