"""Single-stage packet inverse filter and automatic full Main evaluation."""

import time

from . import multiview_trial as trial
from .packet_inverse_filter import scratch_packet_filter
from .projected_prior_detail_trial import ARM as PREVIOUS_ARM, RUN as PREVIOUS_RUN
from .source_pair import read
from .variance_trial import recovery_gpu_choice


RUN = '20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.packet_inverse_filter_trial'
ARM = 'TF_PacketInverseFilter'
TEMPLATE = trial.configuration('MSAT_M20')
original_controller = trial.controller
original_train = trial.train


def configuration(arm):
    if arm != ARM:
        raise ValueError('Expected the registered full-packet inverse filter')
    return dict(TEMPLATE, run_id=RUN, arm=ARM,
        architecture='Scratch 490866-parameter packet-conditional time-varying inverse filter',
        filter='Position-wise complex main and conjugate taps (17 each), plus IQ innovation. '
               'Front, center and tail have independent heads, blended over 16 samples.',
        reference='Received only, adjacent complex correlation across four 20-point blocks at [80:160].',
        matched_training_recipe='Original MSAT_M20 paired IQ, STFT, frozen F0 source fused160 cosine, same gradient cap.',
        training_uses_frozen_F0=True, identity_initialized=True)


def factory(arm):
    return scratch_packet_filter(configuration(arm)['model_seed'])


def smoke_weights(model):
    return tuple(head[-1].weight for head in model.heads)


def train(args, out):
    original_train(args, out, smoke_weights=smoke_weights)


def previous_gpu_finished(root):
    folder = root/'runs'/PREVIOUS_RUN
    for stage in ('train', 'cross_rx', 'cross_both'):
        path = folder/f'terminal-{PREVIOUS_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code'] != 0 or stage == 'cross_both'):
            return True
    queue_end = root/'logs'/PREVIOUS_RUN/'queue_terminal.json'
    return queue_end.is_file() and read(queue_end).get('exit_code') not in (None, 0)


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_PROJECTED_PRIOR_LAST_GPU_STAGE', flush=True)
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


if __name__ == '__main__':
    install()
    trial.main()
