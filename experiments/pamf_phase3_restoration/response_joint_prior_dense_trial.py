"""Scratch response-preserving candidate with the original dense posterior."""
from functools import partial

import torch

from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import response_joint_prior_trial as original
from .coupled_covariance import normalize_modes
from .mismatched_prior_training import load_training_priors
from .response_joint_prior import ResponseJointPrior
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-ResponseDense-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.response_joint_prior_dense_trial'
ARM='TF_ResponseDense'


class DenseResponseJointPrior(ResponseJointPrior):
    def source_readout(self,received,*,prior=None):
        return self._readout(received,prior)


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed dense response candidate is allowed')
    config=original.configuration(original.ARM)
    config.update(run_id=RUN,arm=ARM,
        training_solver='Original dense FP64 joint-prior posterior; physical/source moments fixed.',
        scientific_change='Original source-response loss with original dense posterior for both training and deployment; scratch single30.',
        source_solver_evidence='20261108-diagnostic-Restore-ResponseE10Dense-source-s2026092701-r01: same-source-microbatch dense and low-rank residuals; no target input.',
        prior_failed_run=original.RUN,prior_failed_run_loaded=False)
    return config


def factory(arm):
    return DenseResponseJointPrior(configuration(arm)['model_seed'])


def train(args,out):
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(c,source_modes(c)) for c in pair[1]])
    priors=tuple(v.to('cuda:0') for v in (*pair,modes))
    readout=partial(original.response_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout,
        smoke_check=original.control.initial_smoke,smoke_gradient_check=original.smoke_gradient_check,
        training_trace=True,training_posterior_multiplier=2)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train=configuration,factory,train
    trial.gpu_choice=partial(recovery_gpu_choice,allowed_gpus=(6,),maximum_processes=5)


if __name__=='__main__':install();trial.main()
