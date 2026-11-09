"""Scratch30 wide restoration with capped source F0 middle-feature guidance."""
from functools import partial
from pathlib import Path
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
from .source_multilevel_gradient_probe import normalized_map_error
from .variance_trial import recovery_gpu_choice

RUN = '20261109-phase1-Restore-WideMiddleFeature-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.wide_middle_feature_trial'
ARM = 'TF_WideMiddleFeature'
MIDDLE_WEIGHT = .01
MIDDLE_IQ_GRADIENT_CAP = .5
MIDDLE_PATH = 'core.id_backbone.t2.pool'


def middle_features(reader, value):
    seen = []
    layer = reader.get_submodule(MIDDLE_PATH)
    handle = layer.register_forward_hook(lambda _module, _inputs, output: seen.append(output))
    try:
        final = reader.features(value)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise ValueError('Actual F0 middle layer must execute exactly once')
    middle = seen[0]
    if (not torch.is_tensor(middle) or middle.ndim < 2 or middle.shape[0] != len(value)
            or not torch.isfinite(middle).all() or not torch.isfinite(final).all()):
        raise ValueError('Finite actual F0 middle and final features required')
    return final, middle


class MiddleTapReader:
    """Capture source F0 middle tensors during the existing feature calls."""
    def __init__(self, reader):
        self.reader = reader
        self.calls = []

    def features(self, value):
        final, middle = middle_features(self.reader, value)
        self.calls.append((value, middle))
        return final


def capped_middle_term(restored, clean, middle, target):
    if restored.shape != clean.shape or restored.shape[1:] != (2,256):
        raise ValueError('Exact source IQ pairs required')
    raw = MIDDLE_WEIGHT * normalized_map_error(middle, target)
    gradient = torch.autograd.grad(raw.sum(), restored, retain_graph=True)[0].detach()
    if not torch.isfinite(raw).all() or not torch.isfinite(gradient).all():
        raise ValueError('Finite source middle-feature loss and input gradient required')
    norm = gradient.flatten(1).norm(dim=1)
    energy = clean.detach().square().mean((1,2)).clamp_min(1e-4)
    iq_norm = 2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*energy)
    cap_weight = (MIDDLE_IQ_GRADIENT_CAP*iq_norm/norm.clamp_min(1e-12)).clamp(max=1).detach()
    term = cap_weight*raw
    return term, dict(source_middle_raw_loss=raw, source_middle_weighted_loss=term,
        source_middle_cap_weight=cap_weight, source_middle_cap_fraction=(cap_weight<1).to(raw.dtype),
        source_middle_input_gradient_norm=norm, source_middle_iq_input_gradient_norm=iq_norm)


def middle_readout(model, reader, received, clean, features, scale, objective, *,
                   source_indices, epoch, priors, assignment, source_trace):
    tap = MiddleTapReader(reader)
    restored, aux, loss, metrics = response.response_readout(
        model, tap, received, clean, features, scale, objective,
        source_indices=source_indices, epoch=epoch, priors=priors,
        assignment=assignment, source_trace=source_trace)
    candidates = [middle for value,middle in tap.calls if value is restored]
    if len(candidates) != 2:
        raise ValueError('Expected two existing F0 forwards on the same restored source IQ')
    with torch.no_grad():
        _, target = middle_features(reader, clean)
    term, extra = capped_middle_term(restored, clean, candidates[0], target.detach())
    if epoch == 0:
        parameter = model.detail.refinement.output.weight
        gradient = torch.autograd.grad(term.mean(), parameter, retain_graph=True)[0]
        if not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
            raise ValueError('Middle-feature source objective must reach refinement output')
        model.middle_smoke_report = dict(actual_source_middle_gradient=True,
            middle_refinement_gradient_absolute_sum=float(gradient.detach().abs().sum()))
    metrics.update(extra)
    return restored, aux, loss + term, metrics


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only fixed wide middle-feature variant allowed')
    cfg = wide.configuration(wide.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        scientific_change=('Matched scratch wide time-frequency architecture and source response losses; '
            'add per-record capped middle F0 feature preservation on paired source clean/restored IQ.'),
        middle_feature_path=MIDDLE_PATH, middle_feature_weight=MIDDLE_WEIGHT,
        middle_feature_normalization='Per-record clean middle-map mean square, floor 1e-8.',
        middle_input_gradient_cap='Detached min(1,0.5*IQ_NMSE_input_gradient_norm/middle_input_gradient_norm).',
        middle_reference='Original frozen F0 on the same source clean IQ; no logits, labels, V, OE or target.',
        middle_source_evidence='20261109-diagnostic-Restore-MultilevelFGradient-sourceLU-s2026092701-r01',
        training_F0_forward_rows=5*1890000,
        matched_control=wide.RUN+'/'+wide.ARM,
        resource_policy='GPU4 or6; admit at most one existing compute PID and >=12 GiB free; monitor total cap3.',
        training_stages=1, epochs=30, updates=7410, checkpoint_selection='final30_only',
        default_DURA_changed=False)
    return cfg


def factory(arm):
    configuration(arm)
    return wide.WideTimeFreqResponse(2026092701)


def smoke_gradient_check(model):
    result = wide.smoke_gradient_check(model)
    report = getattr(model, 'middle_smoke_report', None)
    if not report or not report.get('actual_source_middle_gradient'):
        raise ValueError('Actual source middle-gradient smoke required')
    result.update(report)
    return result


def train(args, out):
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair, modes))
    readout = partial(middle_readout, priors=priors, assignment=assignment.to('cuda:0'))
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
        smoke_check=original.control.initial_smoke, smoke_gradient_check=smoke_gradient_check,
        training_trace=True, training_posterior_multiplier=2, training_F0_multiplier=5,
        source_perturbed_F0_multiplier=1)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(4,6), maximum_processes=2)
            if item['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller
    if len(sys.argv)>1 and sys.argv[1]=='score':
        if '--root' not in sys.argv:
            raise ValueError('Root required for the original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root')+1]))


if __name__ == '__main__':
    install()
    trial.main()
