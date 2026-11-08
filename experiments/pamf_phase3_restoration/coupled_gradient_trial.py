"""Single-stage F-guided IQ restoration with bounded physical gradient coupling."""
import sys
from pathlib import Path

import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .covariance_detail import CovarianceDetail, detail_objective
from .covariance_inverse import gaussian_posterior
from .fixed_prediction_score_recovery import original_scorer
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-GradientCoupled-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.coupled_gradient_trial'
ARM = 'TF_GradientCoupled'
COUPLING = .05


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


class GradientCoupledRestorer(CovarianceDetail):
    def forward(self, received, *, return_aux=False):
        aux = self.physical.estimate(received)
        # Only IQ/feature gradients are scaled. Physical supervision consumes
        # the original aux and therefore retains its original gradient.
        action = ScaleGradient.apply(aux['action_matrix'], COUPLING)
        variance = ScaleGradient.apply(aux['variance'], COUPLING)
        base, residual = gaussian_posterior(action, received.flatten(1).double(),
            self.physical.prior_mean.double(), self.physical.prior_covariance.double(), variance.double())
        base = base.float().reshape_as(received)
        context = torch.cat((received, variance.log()[:, None, None].expand(-1, 1, 256)), 1)
        restored = self.detail(base, context)
        return (restored, dict(**aux, base_iq=base, relative_solve_residual=residual)) if return_aux else restored


def configuration(arm):
    if arm != ARM: raise ValueError('Only fixed coupled-gradient candidate allowed')
    cfg = training.configuration(training.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        architecture='CovarianceDetail single30 with differentiable physical posterior and shared detail path',
        training_path='Scratch single30 source IQ/action/noise and frozen F0 continuous features, one optimizer',
        gradient_routing='Physical supervised gradient unchanged; IQ and feature gradient to physical estimator scaled 0.05; detail full gradient',
        physical_gradient_coupling=COUPLING,
        coupling_source='sourceLU fixed 28 record gradient diagnostic; exploratory scale 0.05, preregistered before target',
        scientific_change='Adds physical-estimator gradient coupling vs source-only matched CovarianceDetail; no G weight inheritance',
        matched_control=training.RUN+'/'+training.ARM,
        resource_policy='GPU2/3/4 with at most one existing process; monitor total cap3; >=12 GiB free')
    return cfg


def factory(arm):
    configuration(arm)
    model = GradientCoupledRestorer(2026092701)
    model.initialization_report = dict(mode='scratch_single_stage_scaled_physical_gradient',
        model_seed=2026092701, detail_seed=2026092702, ancestor_checkpoints=[],
        training_stages=1, physical_gradient_coupling=COUPLING)
    return model


def readout(model, reader, received, clean, target, scale, objective, *, source_indices, epoch):
    restored, aux = model(received, return_aux=True)
    loss, metrics = detail_objective(reader, restored, clean, target, scale, objective)
    if epoch == 0:
        features = reader.features(restored)
        feature_only = .01 * ((1-torch.nn.functional.cosine_similarity(features, target, dim=1, eps=1e-8)) +
                               ((features-target).square()/scale).mean(1))
        params = tuple(p for head in (model.physical.kernel_head, model.physical.coefficient_head,
                                      model.physical.noise_head) for p in head.parameters())
        gradients = torch.autograd.grad(feature_only.mean(), params, retain_graph=True, allow_unused=True)
        present = [g for g in gradients if g is not None]
        if not present or any(not torch.isfinite(g).all() for g in present) or not any(g.abs().sum() > 0 for g in present):
            raise ValueError('Frozen source F0 continuous feature loss must reach physical heads')
        model.feature_coupling_smoke = dict(actual_feature_gradient_to_physical=True,
                                           physical_gradient_coupling=COUPLING)
    return restored, aux, loss, metrics


def smoke_gradient(model):
    return model.feature_coupling_smoke


def train(args, out):
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_gradient_check=smoke_gradient, physical_detail_coupled=True)


def admitted_gpus():
    return [g for g in recovery_gpu_choice(allowed_gpus=(2, 3, 4), maximum_processes=2)
            if g['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller
    # Use the exact original F0 scorer release for score mode; target truth is
    # opened only inside the established scorer after full prediction preflight.
    if len(sys.argv) > 1 and sys.argv[1] == 'score':
        if '--root' not in sys.argv: raise ValueError('Root required for original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root')+1]))


if __name__ == '__main__':
    install()
    trial.main()
