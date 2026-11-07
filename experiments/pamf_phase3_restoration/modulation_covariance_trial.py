"""Scratch source training with physical receiver-modulation factor supervision."""
import time

from . import covariance_trial as control
from . import multiview_trial as trial
from .modulation_covariance import modulation_action_loss, scratch_modulation_covariance
from .phase_only_eval import ARM as PREVIOUS_ARM, RUN as PREVIOUS_RUN
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.modulation_covariance_trial'
ARM = 'TF_ModulationCovariance'
original_controller = trial.controller


def configuration(arm):
    if arm!=ARM:
        raise ValueError('Only fixed receiver-modulation candidate allowed')
    cfg = control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=arm,
        architecture='TF128x6 + FIR64 and noise heads + global complex gain and per-branch phase slope plus local residual; original 512-real Gaussian posterior',
        objective='.1 exact full Action relative MSE + .05 gauge-aligned modulation relative MSE + .02 normalized FIR shape squared error + .001 noise log MSE',
        operator_loss='Exact Frobenius error in compressed reflected-FIR basis. No dense Action allocation in source updates.',
        source_action_supervision='Original R07 simulator traces only. Gauge-match true FIR/main/image to detached current estimated FIR.',
        training_path='All source records, factorized Action and direct modulation/FIR/noise losses, no posterior or F0 in updates.',
        matched_control=control.RUN+'/'+control.ARM,final_deployment_phase_grid=False)
    return cfg


def factory(arm):
    configuration(arm)
    return scratch_modulation_covariance(2026092701)


def smoke_weights(model):
    head = model.coefficient_head
    return (model.kernel_head.weight,head.gains.weight,head.slopes.weight,head.local_phase.weight,model.noise_head.weight)


def train(args,out):
    control.train(args,out,build=factory,configure=configuration,objective=modulation_action_loss,
                  dense_supervision=False,smoke_weights=smoke_weights)


def previous_gpu_finished(root):
    folder = root/'runs'/PREVIOUS_RUN
    for stage in ('cross_rx','cross_both'):
        path = folder/f'terminal-{PREVIOUS_ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code']!=0 or stage=='cross_both'):
            return True
    return False


def controller(args):
    if not previous_gpu_finished(args.root):
        print('WAITING_FOR_PHASE_ONLY_LAST_GPU_STAGE',flush=True)
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
