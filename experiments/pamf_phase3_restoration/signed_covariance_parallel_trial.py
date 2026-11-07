"""Unchanged signed covariance on GPU4's normal slots or later GPU6 succession."""
from functools import partial

from . import signed_covariance_trial as method
from . import multiview_trial as trial
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r02'
MODULE = 'experiments.pamf_phase3_restoration.signed_covariance_parallel_trial'
ARM = method.ARM


def configuration(arm):
    cfg = method.configuration(arm)
    cfg['run_id'] = RUN
    return cfg


def train(args,out):
    method.train(args,out,config_for=configuration)


def resource_choice(root,runner=None):
    choices = recovery_gpu_choice(runner,allowed_gpus=(4,),maximum_processes=3)
    if choices:
        return choices
    if method.previous_gpu_finished(root):
        return recovery_gpu_choice(runner,allowed_gpus=(6,),maximum_processes=5)
    return []


def controller(args):
    trial.gpu_choice = partial(resource_choice,args.root)
    print('SIGNED_GPU4_CAP3_OR_PREDECESSOR_GPU6_CAP5',flush=True)
    method.original_controller(args)


def install():
    trial.RUN,trial.MODULE,trial.ARMS = RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train,trial.controller = configuration,method.factory,train,controller


if __name__ == '__main__':
    install()
    trial.main()
