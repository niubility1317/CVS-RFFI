"""Scratch30 wide restorer with paired source-channel feature consistency."""
from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import feature_response_trial as response
from . import multiview_trial as trial
from . import response_joint_prior_trial as original
from . import wide_timefreq_response_trial as wide
from .cache import tensor
from .coupled_covariance import normalize_modes
from .fixed_prediction_score_recovery import original_scorer
from .mismatched_prior_training import choose_prior, load_training_priors
from .signed_covariance import source_modes
from .source_inverse import R07
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261109-phase1-Restore-WideViewConsistency-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.wide_multiview_consistency_trial'
ARM = 'TF_WideViewConsistency'
CONSISTENCY_WEIGHT = .01


def second_draw_number(epoch):
    if not 0 <= epoch <= 30:
        raise ValueError('Fixed single30 epoch required')
    actual_epoch = max(epoch, 1)
    first = (actual_epoch - 1) % 20 + 1
    return first % 20 + 1


class SecondSourceDraw:
    """One extra original R07 received view, indexed by the same source record."""
    def __init__(self, root, device):
        self.root, self.device, self.number, self.received = root, device, None, None

    def __call__(self, epoch, indices):
        number = second_draw_number(epoch)
        if number != self.number:
            folder = self.root/'runs'/R07/'cache'/'epochs'/f'{number:02d}'
            done = read(folder/'complete.json')
            if (done['count'], done['TX_labels_read'], done['target_reads']) != (63000, 0, 0):
                raise ValueError('Completed original source draw required')
            data = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
            try:
                if data.shape != (63000, 2, 256) or data.dtype != np.float32:
                    raise ValueError('Original matched source received IQ required')
                values = tensor(data, self.device)
            finally:
                data._mmap.close()
            if not torch.isfinite(values).all():
                raise ValueError('Finite paired source view required')
            self.received, self.number = values, number
        return self.received[indices]


def consistency(first, second, scale):
    if (first.ndim != 2 or first.shape[1] != 160 or second.shape != first.shape or scale.shape != (160,)
            or not torch.isfinite(first).all() or not torch.isfinite(second).all()
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('Matched finite continuous features and positive source scale required')
    return ((first - second).square() / scale.detach()).mean(1)


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed paired-view consistency arm is allowed')
    cfg = wide.configuration(wide.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        objective=(cfg['objective'] + '; add .01 standardized F0 feature distance between the same '
                   'source IQ restored from two original source channel draws.'),
        scientific_change=('Matched wide time-frequency architecture and source feature-response training; '
                           'add paired distinct source-channel feature consistency.'),
        second_source_draw=('For first draw d use 1+(d modulo20); original R07 source index, source clean '
                            'and same selected source prior; no TX labels or target.'),
        paired_feature_consistency_weight=CONSISTENCY_WEIGHT,
        training_posterior_calls_per_record=3, training_F0_forward_rows=6*1890000,
        source_other_received_rows=1890000, source_perturbed_F0_forward_rows=1890000,
        resource_policy='GPU2-6 excluding0/1/7; at most 1 existing compute PID before admission, >=12 GiB free; total cap3.',
        matched_control=wide.RUN+'/'+wide.ARM,
        source_view_evidence='20261108-diagnostic-Restore-FeatureViewDispersion-sourceLU-s2026092701-r01: source three-view feature variation.')
    return cfg


def factory(arm):
    return wide.WideTimeFreqResponse(configuration(arm)['model_seed'])


def paired_readout(model, reader, received, clean, features, scale, objective, *, source_indices,
                   epoch, priors, assignment, source_trace, second_source):
    restored, aux, loss, metrics = response.response_readout(
        model, reader, received, clean, features, scale, objective, source_indices=source_indices,
        epoch=epoch, priors=priors, assignment=assignment, source_trace=source_trace)
    other_received = second_source(epoch, source_indices)
    prior = choose_prior(priors, assignment, source_indices, epoch)
    other, other_aux = model.source_readout(other_received, prior=prior)
    first_feature, other_feature = reader.features(restored), reader.features(other)
    variation = consistency(first_feature, other_feature, scale)
    added = CONSISTENCY_WEIGHT * variation
    if epoch == 0:
        grad, = torch.autograd.grad(added.mean(), (model.detail.refinement.output.weight,), retain_graph=True)
        if not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
            raise ValueError('Paired source consistency must reach the new detail output')
        model.paired_view_smoke = dict(actual_source_view_consistency_gradient=True,
                                       source_view_gradient_absolute_sum=float(grad.abs().sum()))
    metrics.update(source_view_feature_distance=variation,
                   source_view_feature_weighted_loss=added)
    aux['relative_solve_residual'] = torch.maximum(aux['relative_solve_residual'],
                                                   other_aux['relative_solve_residual'])
    return restored, aux, loss + added, metrics


def smoke_gradient_check(model):
    return dict(wide.smoke_gradient_check(model), **model.paired_view_smoke)


def train(args, out):
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair, modes))
    source = SecondSourceDraw(args.root, 'cuda:0')
    readout = partial(paired_readout, priors=priors, assignment=assignment.to('cuda:0'), second_source=source)
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_check=original.control.initial_smoke, smoke_gradient_check=smoke_gradient_check,
                   training_trace=True, training_posterior_multiplier=3, training_F0_multiplier=6,
                   source_perturbed_F0_multiplier=1)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(2,3,4,5,6), maximum_processes=2)
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
