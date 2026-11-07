"""The fixed covariance candidate under the approved GPU6 sharing budget."""

from functools import partial

from . import conditioned_covariance_trial as method
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .mismatched_prior_training import load_training_priors
from .variance_trial import recovery_gpu_choice


RUN = '20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02'
MODULE = 'experiments.pamf_phase3_restoration.conditioned_covariance_shared_trial'
ARM = method.ARM


def configuration(arm):
    cfg = method.configuration(arm)
    cfg['run_id'] = RUN
    return cfg


def factory(arm):
    return method.factory(arm)


def train(args,out):
    priors,assignment = load_training_priors(args.root,'cuda:0')
    readout = partial(method.noisy_readout,priors=priors,assignment=assignment)
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout)


def install():
    trial.RUN = RUN
    trial.MODULE = MODULE
    trial.ARMS = (ARM,)
    trial.configuration = configuration
    trial.factory = factory
    trial.train = train
    trial.gpu_choice = partial(recovery_gpu_choice,allowed_gpus=(6,),maximum_processes=5)


if __name__=='__main__':
    install()
    trial.main()
