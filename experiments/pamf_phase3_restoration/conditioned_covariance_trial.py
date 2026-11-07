"""Single30 noisy-IQ training of an observation-conditioned prior covariance."""

from functools import partial
import time

from . import covariance_detail_trial as training
from . import mismatched_prior_trial as control
from . import multiview_trial as trial
from .conditioned_covariance import scratch_conditioned_covariance
from .covariance_detail import detail_objective
from .mismatched_prior_training import choose_prior, load_training_priors
from .source_pair import read
from .variance_trial import recovery_gpu_choice


RUN = '20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.conditioned_covariance_trial'
ARM = 'TF_ConditionedCovariance'
original_controller = trial.controller


def configuration(arm):
    if arm!=ARM:
        raise ValueError('Only the observation-conditioned covariance candidate is allowed')
    cfg = control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        architecture='Scratch original physical estimator plus six-layer full-rate prior variance network and one Gaussian posterior.',
        objective='Physical .1Action relative MSE+.001noise log MSE; confidence learns actual noisy restored IQNMSE+.05STFT+.01continuousF0.',
        training_path='Same single30 global/other-RX prior alternation. Confidence gradient passes through adapted FP64 Cholesky posterior; physical A/noise detached.',
        learned_detail='Per-time I/Q positive variance addition, received/mean/logdiag/innovation/lognoise/position,64 channels and dilations1..32.',
        learned_IQ_decoder=False,detail_initial_zero=False,detail_initial_extra_variance_fraction=.0069314718,
        gradient_routing='Physical gets original simulator loss. Conditional covariance gets actual noisy IQ/STFT/frozen F0 loss.',
        source_actual_IQ_F0_guidance=True,source_F0_guidance_readout='Actual restored IQ with realized source noise present.',
        conditional_covariance='C_source + diag(.01*diag(C_source)*softplus(network_output)).',
        learned_prior_mean=False,initial_covariance_addition='Zero output logits give 0.0069314718 times current prior diagonal.',
        scientific_change='Observation-conditioned variance network, actual noisy objective and source prior mismatch exposure; composite candidate.',
        matched_control=control.RUN+'/'+control.ARM)
    return cfg


def factory(arm):
    return scratch_conditioned_covariance(configuration(arm)['model_seed'])


def noisy_readout(model,reader,received,clean,features,scale,paired_objective,*,
                  source_indices,epoch,priors,assignment):
    prior = choose_prior(priors,assignment,source_indices,epoch)
    restored,aux = model.source_readout(received,prior=prior)
    loss,metrics = detail_objective(reader,restored,clean,features,scale,paired_objective)
    metrics['added_prior_variance'] = aux['added_variance'].mean((1,2))
    metrics['added_variance_relative_to_prior'] = aux['added_variance_relative_to_prior']
    return restored,aux,loss,metrics


def train(args,out):
    priors,assignment = load_training_priors(args.root,'cuda:0')
    readout = partial(noisy_readout,priors=priors,assignment=assignment)
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout)


def previous_gpu_finished(root):
    folder = root/'runs'/control.RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{control.ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code']!=0 or stage=='cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_MISMATCHED_PRIOR_LAST_GPU_STAGE',flush=True)
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
