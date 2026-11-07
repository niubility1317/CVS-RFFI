"""Single30 conditional-prior training with source-only receiver mismatch."""

from functools import partial
import time

from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import projected_prior_detail_trial as control
from .mismatched_prior_training import STATISTICS_CODE, STATISTICS_RUN, load_training_priors, mismatch_readout
from .packet_inverse_filter_trial import ARM as PREVIOUS_ARM, RUN as PREVIOUS_RUN
from .source_pair import read
from .variance_trial import recovery_gpu_choice


RUN = '20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.mismatched_prior_trial'
ARM = 'TF_MismatchedPrior'
original_controller = trial.controller


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed source mismatch candidate is allowed')
    cfg = control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        objective='Same projected-prior source noiseless IQ/STFT/continuous frozen F0 loss and separated physical loss.',
        training_path='Odd epochs use original full-source prior; even epochs use a source-only matched-count prior '
                      'excluding the current physical source RX. One factor and same source exposure per record.',
        training_prior_statistics=STATISTICS_RUN,training_prior_statistics_code=STATISTICS_CODE,
        training_prior_other_RX_samples=12600,training_prior_mismatch_epochs=list(range(2,31,2)),
        source_receiver_metadata='Training prior selection only. Reader and G deployment accept received IQ only.',
        scientific_change='Source prior mismatch exposure, fixed same scratch architecture, objective, optimizer, order and final30.',
        matched_control=control.RUN+'/'+control.ARM)
    return cfg


def factory(arm):
    configuration(arm)
    return control.factory(control.ARM)


def train(args,out):
    priors,assignment = load_training_priors(args.root,'cuda:0')
    readout = partial(mismatch_readout,priors=priors,assignment=assignment)
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout)


def previous_gpu_finished(root):
    folder = root/'runs'/PREVIOUS_RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{PREVIOUS_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code'] != 0 or stage == 'cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_PACKET_INVERSE_FILTER_LAST_GPU_STAGE',flush=True)
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
