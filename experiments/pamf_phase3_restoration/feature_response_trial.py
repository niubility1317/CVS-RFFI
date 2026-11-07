"""Single-stage source feature-response guidance and frozen Main testing."""
from functools import partial

import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import response_joint_prior_dense_trial as dense
from . import response_joint_prior_trial as original
from .coupled_covariance import normalize_modes
from .feature_response_loss import feature_response_loss
from .mismatched_prior_training import choose_prior, load_training_priors
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice


RUN = '20261108-phase1-Restore-FeatureResponse-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.feature_response_trial'
ARM = 'TF_FeatureResponse'
FEATURE_RESPONSE_WEIGHT = .01


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed source feature-response candidate is allowed')
    cfg = dense.configuration(dense.ARM)
    cfg.update(
        run_id=RUN, arm=ARM,
        objective=('Original dense source action/noise, IQ/STFT, continuous F0 and IQ response objectives; '
                   'add .01 source continuous F0 response error.'),
        scientific_change=('Original dense response candidate from scratch with a single added '
                           'continuous F0 directional-response term; no extra deployment calls.'),
        feature_response_weight=FEATURE_RESPONSE_WEIGHT,
        feature_response_normalization=('Source F0 variance per dimension; reference response energy plus '
                                        '1e-4 times source feature energy, minimum 1e-12.'),
        feature_response_reference='F0(clean+delta)-F0(clean), no classifier output or TX label.',
        source_feature_response_evidence=('20261107-diagnostic-Restore-CovarianceResponse-sourceLU-'
                                          's2026092701-r01: fixed synthetic source-only responses.'),
        matched_control=dense.RUN+'/'+dense.ARM,
        training_F0_forward_rows=4*1890000,
        source_perturbed_F0_forward_rows=1890000,
        resource_policy='GPU5; at most one existing compute PID before admission, monitor total cap3.',
    )
    return cfg


def factory(arm):
    configuration(arm)
    return dense.factory(dense.ARM)


def response_readout(model, reader, received, clean, features, scale, objective, *,
                     source_indices, epoch, priors, assignment, source_trace):
    if source_trace is None or not torch.equal(source_trace['received'], received):
        raise ValueError('Exact current source simulator trace required')
    restored, aux, loss, metrics = original.control.joint_readout(
        model, reader, received, clean, features, scale, objective,
        source_indices=source_indices, epoch=epoch, priors=priors, assignment=assignment)
    delta, choice = original.selected_perturbation(clean, source_indices, epoch)
    kernel, main, image, _ = original.known_parameters(source_trace)
    action = original.dense_action(kernel, main, image)
    changed = original.inject(received, delta, action)
    prior = choose_prior(priors, assignment, source_indices, epoch)
    altered, altered_aux = model.source_readout(changed, prior=prior)
    iq_response = original.response_loss(restored, altered, delta)
    with torch.no_grad():
        source_changed_features = reader.features(clean + delta)
    restored_features = reader.features(restored)
    altered_features = reader.features(altered)
    f_response = feature_response_loss(restored_features, altered_features,
                                       features, source_changed_features, scale)
    if not torch.isfinite(iq_response).all() or not torch.isfinite(f_response).all():
        raise ValueError('Finite source IQ and feature response objectives required')
    feature_term = FEATURE_RESPONSE_WEIGHT * f_response
    if epoch == 0:
        with torch.no_grad():
            deployed, _ = model._readout(received, prior)
            torch.testing.assert_close(restored, deployed, rtol=2e-5, atol=2e-6)
        parameters = (model.detail.output.weight, model.detail.mean_output.weight)
        gradients = torch.autograd.grad(feature_term.mean(), parameters, retain_graph=True)
        if any(not torch.isfinite(g).all() or not g.abs().sum() > 0 for g in gradients):
            raise ValueError('Feature response objective must reach both prior heads')
        model.response_smoke_report = dict(
            actual_source_feature_response_gradients=True,
            actual_training_deployment_posterior_agreement=True,
            response_covariance_gradient_absolute_sum=float(gradients[0].abs().sum()),
            response_mean_gradient_absolute_sum=float(gradients[1].abs().sum()))
    metrics.update(source_response_relative_error=iq_response,
                   source_response_weighted_loss=original.RESPONSE_WEIGHT*iq_response,
                   source_feature_response_relative_error=f_response,
                   source_feature_response_weighted_loss=feature_term,
                   source_response_relative_RMS=(delta.square().mean((1, 2)) /
                                                 clean.square().mean((1, 2))).sqrt())
    for i, name in enumerate(original.DIRECTIONS):
        metrics['source_response_fraction_'+name] = (choice == i).to(loss.dtype)
    aux['relative_solve_residual'] = torch.maximum(aux['relative_solve_residual'],
                                                   altered_aux['relative_solve_residual'])
    return restored, aux, loss + original.RESPONSE_WEIGHT*iq_response + feature_term, metrics


def train(args, out):
    pair, assignment = load_training_priors(args.root, 'cpu')
    modes = torch.stack([normalize_modes(c, source_modes(c)) for c in pair[1]])
    priors = tuple(value.to('cuda:0') for value in (*pair, modes))
    readout = partial(response_readout, priors=priors, assignment=assignment.to('cuda:0'))
    training.train(args, out, build=factory, config_for=configuration, training_readout=readout,
                   smoke_check=original.control.initial_smoke,
                   smoke_gradient_check=original.smoke_gradient_check, training_trace=True,
                   training_posterior_multiplier=2, training_F0_multiplier=4,
                   source_perturbed_F0_multiplier=1)


def admitted_gpus():
    return [item for item in recovery_gpu_choice(allowed_gpus=(5,), maximum_processes=2)
            if item['free_memory_MB'] >= 12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller


if __name__ == '__main__':
    install()
    trial.main()
