"""Single-stage projected-prior training and automatic full Main evaluation."""

import time

from . import covariance_detail_trial as parent
from . import multiview_trial as trial
from .phase_supervised_covariance_trial import ARM as PREVIOUS_ARM, RUN as PREVIOUS_RUN
from .projected_prior_detail import scratch_projected_prior_detail, shrinkage_training_readout
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.projected_prior_detail_trial'
ARM = 'TF_ProjectedPriorDetail'
original_controller = trial.controller


def configuration(arm):
    if arm!=ARM:
        raise ValueError('Only the fixed projected prior detail candidate is allowed')
    c = parent.configuration(parent.ARM)
    c.update(run_id=RUN,arm=arm,
        architecture='Scratch TF128x6 physical estimator + conditional MSAT prior mean update, posterior-projected in I-KA.',
        objective='Physical .1 Action relative MSE + .001 log noise; prior detail paired IQ/STFT and .01 F0 continuous '
                  'feature guidance on the corrected source noiseless inverse.',
        training_path='Every source record. Estimate physical Action/noise, detach, one FP64 factor, original posterior and '
                      'project learned mean plus clean-mean together. One joint AdamW update per logical batch.',
        learned_detail='Conditional IQ prior mean MSAT64-128-192-256, context received+lognoise, zero-init. Apply I-KA before output.',
        source_actual_IQ_F0_guidance=False,source_F0_guidance_readout='Source clean minus exact predicted-operator shrinkage plus projected learned update.',
        source_F0_knowledge='Original frozen source fused160 only, clean63000 target cache and corrected source noiseless IQ. Default head only smoke and Main.',
        scientific_change='Counterfactual prior shrinkage target and analytic posterior projection compared with the fixed additive CovarianceDetail.',
        matched_control=parent.RUN+'/'+parent.ARM)
    return c


def factory(arm):
    configuration(arm)
    return scratch_projected_prior_detail(2026092701)


def train(args,out):
    parent.train(args,out,build=factory,config_for=configuration,training_readout=shrinkage_training_readout)


def previous_gpu_finished(root):
    folder = root/'runs'/PREVIOUS_RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{PREVIOUS_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code']!=0 or stage=='cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_PHASE_SUPERVISED_LAST_GPU_STAGE',flush=True)
    while not previous_gpu_finished(args.root):
        time.sleep(5)
    original_controller(args)


def install():
    trial.RUN = RUN
    trial.MODULE = MODULE
    trial.ARMS = (ARM,)
    trial.configuration = configuration
    trial.factory = factory
    trial.train = train
    trial.controller = controller
    trial.gpu_choice = recovery_gpu_choice


if __name__=='__main__':
    install()
    trial.main()
