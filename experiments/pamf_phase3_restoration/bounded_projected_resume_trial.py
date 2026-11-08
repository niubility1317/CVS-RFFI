"""Continue the resource-interrupted bounded projected run at its fixed E26 boundary."""
from pathlib import Path

from . import bounded_projected_resource_trial as original
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r03'
MODULE = 'experiments.pamf_phase3_restoration.bounded_projected_resume_trial'
ARM = original.ARM
PARENT_COMMIT = 'f4e0527b6053584c59a70e16bd938ac7a95499e4'
PARENT_EPOCH = 26
PARENT_RUN = original.RUN
PARENT_CONFIG = original.configuration(ARM)


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the original arm is allowed')
    cfg = dict(PARENT_CONFIG)
    cfg.update(run_id=RUN, technical_recovery_parent=PARENT_RUN,
               technical_recovery_commit=PARENT_COMMIT, technical_recovery_epoch=PARENT_EPOCH,
               resource_policy='GPU2/4; at most one existing compute PID at admission; monitor total cap3.',
               scientific_change='Original bounded projected objective and source schedule, technical E26 continuation only.')
    return cfg


def factory(arm):
    configuration(arm)
    return original.factory(arm)


def parent_training(root):
    parent = root/'runs'/PARENT_RUN
    stop = read(parent/'resource_stop.json')
    terminal = read(parent/('terminal-'+ARM+'-train.json'))
    if (stop['stage'], stop['reason'], stop['cap'], stop['only_own_child_signaled'],
        stop['other_owner_processes_signaled'], terminal['pid'], terminal['exit_code']) != (
            'train', 'TOTAL_COMPUTE_PID_CAP_EXCEEDED', 3, True, False, stop['pid'], -15):
        raise ValueError('Original resource stop required')
    if any(parent.glob('submitted-'+ARM+'-cross*.json')):
        raise ValueError('Parent must not have started target predictions')
    for pid in (stop['pid'], read(parent/'launch.json')['pid']):
        path = Path('/proc')/str(pid)/'cmdline'
        if path.exists() and original.MODULE in path.read_bytes().replace(b'\0', b' ').decode(errors='replace'):
            raise ValueError('Parent owner still live; do not duplicate')
    folder = parent/'rows'/ARM/'training'
    smoke = read(folder/'smoke.json')
    if smoke['status'] != 'PASS' or any(smoke[k] for k in ('TX_labels_read', 'V_reads', 'OE_reads', 'target_inputs')):
        raise ValueError('Original source-only smoke required')
    if not (folder/'recovery'/f'E{PARENT_EPOCH:02d}.pt').is_file() or (folder/'complete.json').exists():
        raise ValueError('Original incomplete E26 technical checkpoint required')
    return folder


def train(args, out):
    parent = parent_training(args.root)
    recovery = dict(run=PARENT_RUN, training=parent, epoch=PARENT_EPOCH, commit=PARENT_COMMIT,
                    config=PARENT_CONFIG)
    training.train(args, out, build=factory, config_for=configuration,
                   training_readout=original.method.training_readout, technical_resume=recovery)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(2, 4), maximum_processes=2)
            if item['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    original.RUN, original.ARM, original.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = original.controller


if __name__ == '__main__':
    install()
    trial.main()
