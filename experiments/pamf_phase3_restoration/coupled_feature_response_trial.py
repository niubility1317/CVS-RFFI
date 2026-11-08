"""Scratch30 coupled-physics restorer with source IQ and feature response preservation."""
import sys
from pathlib import Path
import torch

from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import coupled_gradient_trial as coupled
from . import multiview_trial as trial
from .covariance_inverse import dense_action
from .feature_response_loss import feature_response_loss
from .fixed_prediction_score_recovery import original_scorer
from .response_joint_prior import selected_perturbation, response_loss
from .source_fingerprint_response import inject
from .structured_noise_source import known_parameters
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-CoupledFeatureResponse-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.coupled_feature_response_trial'
ARM = 'TF_CoupledFeatureResponse'
IQ_RESPONSE_WEIGHT = .01
FEATURE_RESPONSE_WEIGHT = .01


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only the fixed coupled feature-response candidate allowed')
    cfg = coupled.configuration(coupled.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        architecture='Scratch GradientCoupled covariance posterior plus multiscale detail; single-view inference',
        objective=('GradientCoupled action/noise and paired IQ/STFT/continuous F0 loss plus ''.01 IQ response and .01 continuous F0 response losses on one original source perturbation.'),
        scientific_change=('From scratch combine source directional IQ/F0 response preservation ''with scaled physical gradient coupling; no parent G checkpoint.'),
        matched_control=coupled.RUN+'/'+coupled.ARM,
        evidence='Fixed matched source-feature response experiment improved Main low urban by 1.683929 pp relative to its dense response control, but was below DURA; transfer to this architecture is unproven.',
        source_response_perturbation='1% source clean RMS; one of four directions selected by (source index+epoch) modulo 4.',
        source_response_observation='Same source noise realization, received+recorded source simulator action(delta); simulator action used only in source training.',
        feature_response_reference='F0(clean+delta)-F0(clean), continuous features only, no TX label or classification logits.',
        source_response_weights={'IQ':IQ_RESPONSE_WEIGHT,'F0':FEATURE_RESPONSE_WEIGHT},
        training_path='Scratch single30: one actual and one source-perturbed posterior, F0 features of both outputs and perturbed clean; one joint optimizer.',
        gradient_routing='Physical supervised gradient unchanged before clipping; IQ/feature and response gradient to physical estimator scaled 0.05; detail full gradient.',
        training_F0_forward_rows=3*1890000, source_perturbed_F0_forward_rows=1890000,
        training_posterior_calls=2*59070, training_posterior_rows=2*1890000,
        F0_forward_rows_per_source_record=3, source_perturbed_F0_forward_rows_per_source_record=1,
        posterior_rows_per_source_record=2, physical_gradient_coupling=coupled.COUPLING,
        resource_policy='GPU2/3/4 at most one existing process before admission, monitor total cap3, >=12 GiB free.')
    return cfg


def factory(arm):
    configuration(arm)
    model = coupled.GradientCoupledRestorer(2026092701)
    model.initialization_report = dict(mode='scratch_single_stage_coupled_feature_response',
        model_seed=2026092701, detail_seed=2026092702, ancestor_checkpoints=[],
        training_stages=1, physical_gradient_coupling=coupled.COUPLING)
    return model


def response_readout(model, reader, received, clean, target, scale, objective, *,
                     source_indices, epoch, source_trace):
    if source_trace is None or not torch.equal(source_trace['received'], received):
        raise ValueError('Exact matched source simulator trace required')
    restored, aux = model(received, return_aux=True)
    paired, metrics = objective(restored, clean)
    base_features = reader.features(restored)
    target = target.detach()
    cosine = 1-torch.nn.functional.cosine_similarity(base_features,target,dim=1,eps=1e-8)
    standardized = ((base_features-target).square()/scale.detach()).mean(1)
    base_loss = paired+.01*(cosine+standardized)
    delta, choice = selected_perturbation(clean, source_indices, epoch)
    kernel, main, image, _ = known_parameters(source_trace)
    changed = inject(received, delta, dense_action(kernel, main, image))
    altered, altered_aux = model(changed, return_aux=True)
    altered_features = reader.features(altered)
    with torch.no_grad():
        source_changed_features = reader.features(clean+delta)
    iq = response_loss(restored, altered, delta)
    feature = feature_response_loss(base_features, altered_features, target, source_changed_features, scale)
    if not torch.isfinite(iq).all() or not torch.isfinite(feature).all():
        raise ValueError('Finite IQ and feature response losses required')
    if epoch == 0:
        parameters = (model.physical.coefficient_head.weight, model.detail.output.weight)
        gradients = torch.autograd.grad(feature.mean(), parameters, retain_graph=True)
        if any(not torch.isfinite(g).all() or not g.abs().sum()>0 for g in gradients):
            raise ValueError('Actual source F0 response must reach physical and detail branches')
        model.response_smoke = dict(actual_feature_response_gradients_to_physical_and_detail=True,
            physical_gradient_coupling=coupled.COUPLING,
            response_physical_gradient_abs_sum=float(gradients[0].abs().sum()),
            response_detail_gradient_abs_sum=float(gradients[1].abs().sum()))
    metrics.update(feature_error=cosine, feature_standardized_error=standardized,
        source_response_relative_error=iq, source_feature_response_relative_error=feature,
        source_response_weighted_loss=IQ_RESPONSE_WEIGHT*iq,
        source_feature_response_weighted_loss=FEATURE_RESPONSE_WEIGHT*feature)
    for i, name in enumerate(('conjugate','amplitude_cubic','phase_slope','local_difference')):
        metrics['source_response_fraction_'+name]=(choice==i).to(base_loss.dtype)
    aux['relative_solve_residual']=torch.maximum(aux['relative_solve_residual'],altered_aux['relative_solve_residual'])
    return restored, aux, base_loss+IQ_RESPONSE_WEIGHT*iq+FEATURE_RESPONSE_WEIGHT*feature, metrics


def smoke_gradient(model):
    return model.response_smoke


def train(args, out):
    training.train(args, out, build=factory, config_for=configuration,
        training_readout=response_readout, smoke_gradient_check=smoke_gradient,
        training_trace=True, training_posterior_multiplier=2, training_F0_multiplier=3,
        source_perturbed_F0_multiplier=1, physical_detail_coupled=True)


def admitted_gpus():
    return [g for g in recovery_gpu_choice(allowed_gpus=(2,3,4),maximum_processes=2)
            if g['free_memory_MB']>=12288]


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = resource.controller
    if len(sys.argv)>1 and sys.argv[1]=='score':
        if '--root' not in sys.argv:
            raise ValueError('Root required for original scorer')
        original_scorer(Path(sys.argv[sys.argv.index('--root')+1]))


if __name__=='__main__':
    install()
    trial.main()
