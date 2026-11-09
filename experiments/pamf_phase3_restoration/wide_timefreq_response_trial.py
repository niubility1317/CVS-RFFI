"""Scratch30 response-guided physical posterior with a time-frequency detail path."""
from functools import partial
from pathlib import Path
import sys

import torch
from torch import nn

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import feature_response_trial as response
from . import multiview_trial as trial
from . import response_joint_prior_trial as original
from .coupled_covariance import normalize_modes
from .mismatched_prior_training import load_training_priors
from .fixed_prediction_score_recovery import original_scorer
from .multiscale import MultiscaleAttentionRestorer
from .response_joint_prior_dense_trial import DenseResponseJointPrior
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice

RUN = '20261109-phase1-Restore-WideTimeFreqResponse-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.wide_timefreq_response_trial'
ARM = 'TF_WideTimeFreqResponse'


class SpectralBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, value):
        return value + self.path(value)


class SpectralContext(nn.Module):
    """Map complex local spectra of the received and posterior IQ back to IQ."""
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(64))
        self.net = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1),
                                 *[SpectralBlock(64) for _ in range(6)],
                                 nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 2, 1))

    def forward(self, received, posterior):
        spectra = []
        for value in (received, posterior):
            z = torch.complex(value[:, 0], value[:, 1])
            s = torch.stft(z, n_fft=64, hop_length=16, win_length=64,
                           window=self.window, center=True, normalized=True,
                           onesided=False, return_complex=True)
            spectra.extend((s.real, s.imag))
        value = self.net(torch.stack(spectra, dim=1))
        transformed = torch.complex(value[:, 0], value[:, 1])
        context = torch.istft(transformed, n_fft=64, hop_length=16, win_length=64,
                              window=self.window, center=True, normalized=True,
                              onesided=False, length=256, return_complex=True)
        return torch.stack((context.real, context.imag), dim=1)


class WideTimeFreqResponse(DenseResponseJointPrior):
    def __init__(self, seed):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 2)
            self.detail.spectral_context = SpectralContext()
            self.detail.refinement = MultiscaleAttentionRestorer(context_channels=5)
        self.initialization_report = dict(
            mode='scratch_single_stage_wide_time_frequency_response', model_seed=seed,
            detail_seed=seed+2, ancestor_checkpoints=[], training_stages=1,
            spectral_context='complex 64-point local spectra, hop16, six 64-channel residual blocks',
            detail='multiscale 64/128/192/256 plus four attention blocks; zero-output start',
            physical_gradient_isolated=True, F0_continuous_feature_guidance=True)

    def _readout(self, received, prior):
        posterior, aux = super()._readout(received, prior)
        spectral = self.detail.spectral_context(received, posterior)
        noise = aux['variance'].detach().log()[:, None, None].expand(-1, 1, 256)
        context = torch.cat((received, noise, spectral), dim=1)
        restored = self.detail.refinement(posterior, context)
        return restored, aux


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the preregistered wide time-frequency candidate is allowed')
    cfg = response.configuration(response.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        architecture=('Scratch physical estimator and adaptive posterior; complex local STFT context '
                      'and independent multiscale attention residual IQ refiner.'),
        scientific_change=('Adds time-frequency conditional post-posterior detail to source feature-response '
                           'training. The refiner starts as identity; no ancestor checkpoint.'),
        gradient_routing=('Source physical action/noise losses update physical estimator only; '
                          'IQ, STFT and continuous F0 response losses update prior and refiner.'),
        deployment_inputs='Received IQ only, fixed source prior, learned physical and detail paths; no F0 or true channel.',
        training_stages=1, epochs=30, updates=7410, checkpoint_selection='final30_only',
        resource_policy='GPU0-6 excluding7; admit at most 1 existing compute PID and >=12 GiB free; monitor total cap3.',
        matched_control=response.RUN+'/'+response.ARM,
        default_DURA_changed=False)
    return cfg


def factory(arm):
    return WideTimeFreqResponse(configuration(arm)['model_seed'])


def smoke_gradient_check(model):
    result = original.smoke_gradient_check(model)
    grad = model.detail.refinement.output.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
        raise ValueError('Actual source objective must reach the new detail output')
    return dict(result, actual_refinement_source_gradient=True,
                refinement_output_gradient_absolute_sum=float(grad.abs().sum()))


def train(args, out):
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair, modes))
    readout = partial(response.response_readout, priors=priors, assignment=assignment.to('cuda:0'))
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_check=original.control.initial_smoke, smoke_gradient_check=smoke_gradient_check,
                   training_trace=True, training_posterior_multiplier=2, training_F0_multiplier=4,
                   source_perturbed_F0_multiplier=1)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(0,1,2,3,4,5,6), maximum_processes=2)
            if item['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller
    if len(sys.argv) > 1 and sys.argv[1] == 'score':
        if '--root' not in sys.argv:
            raise ValueError('Root required for the original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root') + 1]))


if __name__ == '__main__':
    install()
    trial.main()
