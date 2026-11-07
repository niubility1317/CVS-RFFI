"""Continue the interrupted reliability run from its fixed E07 boundary."""
from functools import partial
from pathlib import Path

import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import response_reliability_trial as original
from .coupled_covariance import normalize_modes
from .mismatched_prior_training import load_training_priors
from .signed_covariance import source_modes
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-ResponseReliability-E30-s2026092701-r02'
MODULE = 'experiments.pamf_phase3_restoration.response_reliability_resume_trial'
ARM = original.ARM
PARENT_COMMIT = '936a15eebe15eb039c9aff635941f35389ea5d16'
PARENT_EPOCH = 7


def configuration(arm):
    cfg = original.configuration(arm)
    cfg.update(run_id=RUN, technical_recovery_parent=original.RUN,
               technical_recovery_commit=PARENT_COMMIT, technical_recovery_epoch=PARENT_EPOCH,
               resource_policy='GPU0/1; at most one existing compute PID at admission; monitor total cap3.',
               scientific_change='Original reliability objective and source schedule, technical E07 continuation only.')
    return cfg


def factory(arm):
    configuration(arm)
    return original.factory(arm)


def parent_training(root):
    parent = root/'runs'/original.RUN
    stop = read(parent/'resource_stop.json')
    if (stop['status'], stop['run_id'], stop['last_complete_epoch']) != (
            'STOPPED_OWN_PROCESSES_RESOURCE_CAP', original.RUN, PARENT_EPOCH):
        raise ValueError('Original fixed technical resource stop required')
    if any(parent.glob('submitted-'+ARM+'-cross*.json')):
        raise ValueError('Parent must not have started target predictions')
    for key in ('own_controller_pid', 'own_child_pid'):
        path = Path('/proc')/str(stop[key])/'cmdline'
        if path.exists() and original.MODULE in path.read_bytes().replace(b'\0', b' ').decode(errors='replace'):
            raise ValueError('Parent owner still live; do not duplicate')
    folder = parent/'rows'/ARM/'training'
    smoke = read(folder/'smoke.json')
    if smoke['status'] != 'PASS' or any(smoke[k] for k in ('TX_labels_read', 'V_reads', 'OE_reads', 'target_inputs')):
        raise ValueError('Original source-only smoke required')
    if not (folder/'recovery'/f'E{PARENT_EPOCH:02d}.pt').is_file() or (folder/'complete.json').exists():
        raise ValueError('Original incomplete E07 technical checkpoint required')
    return folder


def train(args, out):
    parent = parent_training(args.root)
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(v.to('cuda:0') for v in (*pair, modes))
    readout = partial(original.response_readout, priors=priors, assignment=assignment.to('cuda:0'))
    recovery = dict(run=original.RUN, training=parent, epoch=PARENT_EPOCH, commit=PARENT_COMMIT,
                    config=original.configuration(ARM))
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_check=original.original.control.initial_smoke,
                   smoke_gradient_check=original.original.smoke_gradient_check,
                   training_trace=True, training_posterior_multiplier=2, technical_resume=recovery)


def admitted_gpus():
    # Disjoint from the existing bounded-projection queue, retaining one spare slot.
    return [item for item in recovery_gpu_choice(allowed_gpus=(0, 1), maximum_processes=2)
            if item['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller


if __name__ == '__main__':
    install()
    trial.main()
