"""Single-stage source training and automatic Main evaluation for explicit main phase."""

import time

from . import covariance_trial as control
from . import multiview_trial as trial
from .modulation_covariance_trial import ARM as PREVIOUS_ARM, RUN as PREVIOUS_RUN
from .phase_supervised_covariance import phase_supervised_action_loss, scratch_phase_supervised_covariance
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-PhaseSupervisedCovariance-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.phase_supervised_covariance_trial'
ARM = 'TF_PhaseSupervisedCovariance'
original_controller = trial.controller


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed coherent phase candidate is allowed')
    cfg = control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=arm,
        architecture='TF128x6, coherent-reference canonical FIR64, separate main amplitude/trajectory and image, '
                     'coherent I/Q convolutional main angle head, original FP64 Gaussian IQ posterior',
        objective='.1 exact Action relative MSE + .05 canonical modulation relative MSE + .02 canonical FIR shape '
                  '+ .1 circular main phase + .001 log noise MSE',
        operator_loss='Exact Frobenius error in compressed reflected-FIR basis. No dense Action in training updates.',
        source_action_supervision='Original R07 simulated factors. Canonicalize both FIRs against original L_s/U_s IQ mean. '
                                  'Target global phase from canonical true main projected onto detached predicted trajectory.',
        phase_observation='Current IQ, coherent main and image anchors, complex local matched products, relative estimated noise.',
        training_path='All source records. No F0 or posterior in updates. Exact Final30 only.',
        matched_control=PREVIOUS_RUN+'/'+PREVIOUS_ARM,final_deployment_phase_grid=False)
    return cfg


def factory(arm):
    configuration(arm)
    return scratch_phase_supervised_covariance(2026092701)


def smoke_weights(model):
    head = model.coefficient_head
    return (model.kernel_head.weight,head.gains.weight,head.slopes.weight,head.local_phase.weight,
            model.phase_head.angle.weight,model.noise_head.weight)


def train(args,out):
    control.train(args,out,build=factory,configure=configuration,objective=phase_supervised_action_loss,
                  dense_supervision=False,smoke_weights=smoke_weights)


def previous_gpu_finished(root):
    folder = root/'runs'/PREVIOUS_RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{PREVIOUS_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code'] != 0 or stage == 'cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_MODULATION_LAST_GPU_STAGE',flush=True)
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
