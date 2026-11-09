"""Scratch30 matched wide restorer with source-only feature-tail protection."""
from functools import partial
from pathlib import Path
import math
import sys

import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import feature_response_trial as response
from . import multiview_trial as trial
from . import response_joint_prior_trial as original
from . import wide_timefreq_response_trial as wide
from .coupled_covariance import normalize_modes
from .fixed_prediction_score_recovery import original_scorer
from .mismatched_prior_training import load_training_priors
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice

RUN = '20261109-phase1-Restore-WideFeatureTail-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.wide_tail_feature_trial'
ARM = 'TF_WideFeatureTail'
TAIL_FRACTION = 0.25
TAIL_WEIGHT = 0.005


def tail_term(error):
    """Per-record terms whose mean is weight times the worst-quartile mean."""
    if error.ndim != 1 or not len(error) or not torch.isfinite(error).all() or (error < 0).any():
        raise ValueError('Nonempty finite nonnegative source feature errors required')
    count = math.ceil(TAIL_FRACTION * len(error))
    indices = torch.topk(error.detach(), count, sorted=False).indices
    mask = torch.zeros_like(error)
    mask.scatter_(0, indices, 1.)
    return TAIL_WEIGHT * error * mask * (len(error) / count), mask


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed source feature-tail arm is allowed')
    cfg = wide.configuration(wide.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        objective=(cfg['objective'] + '; add .005 times the within-microbatch top-quartile '
                   'source-variance-standardized F0(clean/restored) feature error.'),
        scientific_change=('Matched wide time-frequency architecture, source schedule and feature response '
                           'training, with additional within-microbatch source feature-tail loss.'),
        source_feature_tail_fraction=TAIL_FRACTION, source_feature_tail_weight=TAIL_WEIGHT,
        source_feature_tail_selection=('Top ceil(0.25 * actual microbatch size) standardized source feature '
                                       'errors; applied separately in each microbatch, including the final partial batch.'),
        resource_policy='GPU1-6 excluding0/7; at most 1 existing compute PID before admission, >=12 GiB free; total cap3.',
        matched_control=wide.RUN+'/'+wide.ARM,
        source_tail_evidence='20261108-diagnostic-Restore-FeatureViewDispersion-sourceLU-s2026092701-r01: source feature error concentration.')
    return cfg


def factory(arm):
    return wide.WideTimeFreqResponse(configuration(arm)['model_seed'])


def tail_readout(*args, **kwargs):
    restored, aux, loss, metrics = response.response_readout(*args, **kwargs)
    extra, selected = tail_term(metrics['feature_standardized_error'])
    metrics.update(source_feature_tail_weighted_loss=extra, source_feature_tail_selected=selected)
    return restored, aux, loss + extra, metrics


def train(args, out):
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair, modes))
    readout = partial(tail_readout, priors=priors, assignment=assignment.to('cuda:0'))
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_check=original.control.initial_smoke, smoke_gradient_check=wide.smoke_gradient_check,
                   training_trace=True, training_posterior_multiplier=2, training_F0_multiplier=4,
                   source_perturbed_F0_multiplier=1)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(1,2,3,4,5,6), maximum_processes=2)
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
