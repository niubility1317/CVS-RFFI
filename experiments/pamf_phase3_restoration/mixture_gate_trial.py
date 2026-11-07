"""Single-stage learned waveform-mixture fusion, frozen Main evaluation."""
import numpy as np
import torch
from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import covariance_trial as covariance
from . import multiview_trial as trial
from .mixture_gate_model import scratch_mixture_gate
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-MixtureFusion-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.mixture_gate_trial'
ARM = 'TF_MixtureFusion'
PRIOR_RUN = '20261107-diagnostic-Restore-waveform-mixture-sourceV-s2026092701-r01'
PRIOR_COMMIT = 'f4dab0140b598bbc3cd212ab4259138e468e1a20'


def validate_mixture_metadata(result, complete):
    required = dict(status='SOURCE_WAVEFORM_MIXTURE_COMPLETE', code_commit=PRIOR_COMMIT,
                    TX_labels_read=0, OE_reads=0, novel_reads=0, target_inputs=0, optimizer_updates=0)
    if any(result.get(k) != v for k, v in required.items()):
        raise ValueError('Completed source-only mixture producer required')
    if complete.get('status') != 'ARTIFACTS_COMPLETE':
        raise ValueError('Completed source mixture artifact required')
    source = result['source_prior']
    if (source.get('source_Ls'), source.get('source_Us')) != (6300, 56700):
        raise ValueError('Only original L_s/U_s prior fit allowed')


def prior_tensor(value):
    array = np.array(value, dtype=np.float64, order='C', copy=True)
    return torch.frombuffer(memoryview(array), dtype=torch.float64).reshape(array.shape)

def bind_prior(root, physical):
    identity = covariance.bind_prior(root, physical)
    folder = root/'runs'/PRIOR_RUN
    result, complete = read(folder/'results.json'), read(folder/'complete.json')
    validate_mixture_metadata(result, complete)
    with np.load(folder/'waveform_prior.npz', allow_pickle=False) as f:
        if set(f.files) != {'within', 'offsets', 'weights'}:
            raise ValueError('Original source mixture statistics required')
        within, offsets, weights = (prior_tensor(f[name]) for name in ('within', 'offsets', 'weights'))
    physical.bind_mixture(within, offsets, weights)
    return dict(**identity, mixture_source_run=PRIOR_RUN, mixture_source_code=PRIOR_COMMIT,
                mixture_fit_roles={'L_s':6300,'U_s':56700}, mixture_source_TX_labels_read=0,
                mixture_modes=4, mixture_V_used_for_fit=False, mixture_target_used_for_fit=False)


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed source mixture fusion candidate is allowed')
    cfg = training.configuration(training.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        architecture=('Scratch time-frequency physical estimator; exact four-source-mixture posterior with one '
                      'FP64 factorization; 128-wide dilated learned mixture gate; independent multiscale '
                      '64/128/192/256 attention IQ refiner conditioned on all four hypotheses.'),
        objective='physical:.1 source-Action relative MSE+.001 noise-log MSE; IQ:NMSE+.05STFT+.01 frozen F0(cos+source-var MSE)',
        training_path='Full source exposure; physical supervised by source simulator; detached mixture components, '
                      'one observation-only gate and one multiscale IQ refinement jointly guided by source clean IQ and continuous F0.',
        scientific_change='Scratch single-stage learned continuous four-hypothesis fusion plus unrestricted multiscale IQ residual; '
                          'initial output exactly equals the source-fitted unadjusted mixture posterior.',
        source_mixture_run=PRIOR_RUN, source_mixture_code=PRIOR_COMMIT, source_mixture_fit_roles={'L_s':6300,'U_s':56700},
        source_mixture_modes=4, mixture_fit_TX_labels=False, mixture_fit_V_OE_target=False,
        deployment_inputs='current received IQ and saved source statistics only; no F0, labels, true channel/noise or target fitting',
        detail_initial_zero=True, training_F0_forward_rows=1890000, training_posterior_calls=59070,
        physical_detail_gradient_isolation=True, source_actual_IQ_F0_guidance=True,
        prior_buffers='source mean/covariance plus source mixture within/offsets/weights as FP32 buffers',
        resource_policy='GPU3; at most one existing compute PID before admission, monitor total cap3.')
    return cfg


def factory(arm):
    configuration(arm)
    return scratch_mixture_gate(2026092701)


def gradient_smoke(model):
    gradient = model.detail.refiner.output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise ValueError('Source IQ/F0 loss must reach the zero-initialized multiscale output')
    return dict(actual_refiner_gradient=True, refiner_output_gradient_absolute_sum=float(gradient.abs().sum()))


def train(args, out):
    training.train(args, out, build=factory, config_for=configuration,
                   bind_prior=bind_prior, smoke_gradient_check=gradient_smoke,
                   additional_prior_keys=('mixture_within', 'mixture_offsets', 'mixture_weights'))


def admitted_gpus():
    return [g for g in recovery_gpu_choice(allowed_gpus=(3,), maximum_processes=2)
            if g['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller


if __name__ == '__main__':
    install()
    trial.main()
