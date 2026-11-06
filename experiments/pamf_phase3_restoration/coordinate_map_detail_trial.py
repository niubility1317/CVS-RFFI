"""Scratch C4 recovery with original-source continuous F0 feature guidance."""
import torch

from . import covariance_detail_trial as detail
from . import covariance_trial as control
from . import multiview_trial as trial
from .coordinate_map_detail import CoordinateMAPDetail, coordinate_prior, rotate_iq
from .variance_trial import recovery_gpu_choice

RUN = '20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.coordinate_map_detail_trial'
ARM = 'TF_CoordinateMAPDetail'


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only fixed C4 operator and IQ detail candidate allowed')
    cfg = detail.configuration(detail.ARM)
    cfg.update(run_id=RUN, arm=arm,
        architecture='C4 group averaged TF128x6 physical operator + one FP64 Gaussian posterior + C4 shared MSAT detail',
        training_path='Four shared physical passes, one detached posterior, four shared detail passes, one F0 feature loss per original-source record',
        source_prior='C4 Gaussian moment match of original source full Gaussian: zero mean and averaged rotated second moment',
        source_prior_storage='FP64 source moments transformed before fixed FP32 storage; posterior arithmetic FP64',
        posterior='C4 conjugated action and invariant scalar noise averaged before one FP64 Cholesky solve',
        source_prior_buffers='physical.prior_mean and physical.prior_covariance fixed, C4-invariant',
        training_posterior_rows=1_890_000, training_physical_branch_rows=7_560_000, training_detail_branch_rows=7_560_000,
        deployment_inputs='currentreceivedIQonly, C4 source prior buffers and learned shared weights',
        scientific_change='Joint C4 physical and detail architecture plus moment-matched source prior; no G initialization checkpoint',
        matched_control=detail.RUN+'/'+detail.ARM)
    return cfg


def factory(arm):
    configuration(arm)
    return CoordinateMAPDetail(2026092701)


def bind_prior(root, physical):
    mean, covariance = control.load_prior(root)
    mean_tensor = torch.frombuffer(memoryview(mean), dtype=torch.float64)
    covariance_tensor = torch.frombuffer(memoryview(covariance), dtype=torch.float64).reshape(512, 512)
    physical.bind_prior(*coordinate_prior(mean_tensor, covariance_tensor))
    return dict(source_run=control.PRIOR_RUN, source_prior_code=control.PRIOR_COMMIT,
                source_fit_packets=63000, transform='C4_Gaussian_moment_match_FP64_before_FP32_storage')


def check_coordinate_response(model, received, original):
    errors = []
    for quarter in range(1, 4):
        rotated = model(rotate_iq(received, quarter))
        expected = rotate_iq(original, quarter)
        torch.testing.assert_close(rotated, expected, rtol=1e-5, atol=1e-6)
        errors.append(float((rotated - expected).abs().max()))
    return dict(C4_coordinate_check_max_abs_error=max(errors), extra_coordinate_G_rows=6)


def train(args, out):
    detail.train(args, out, build=factory, config_for=configuration, bind_prior=bind_prior,
                 smoke_check=check_coordinate_response)


def install():
    trial.RUN = RUN
    trial.MODULE = MODULE
    trial.ARMS = (ARM,)
    trial.configuration = configuration
    trial.factory = factory
    trial.train = train
    trial.gpu_choice = recovery_gpu_choice


if __name__ == '__main__':
    install()
    trial.main()
