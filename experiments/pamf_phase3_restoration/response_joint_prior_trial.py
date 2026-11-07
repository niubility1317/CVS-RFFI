"""Single30 source response supervision with unchanged dense joint-prior deployment."""
from functools import partial
import torch
from . import joint_prior_trial as control
from . import covariance_detail_trial as training
from . import multiview_trial as trial
from .coupled_covariance import normalize_modes
from .covariance_inverse import dense_action
from .mismatched_prior_training import choose_prior,load_training_priors
from .response_joint_prior import DIRECTIONS,EPSILON,RESPONSE_WEIGHT,response_loss,scratch_response_joint_prior,selected_perturbation
from .signed_covariance import source_modes
from .source_fingerprint_response import inject
from .structured_noise_source import known_parameters
from .variance_trial import recovery_gpu_choice

RUN='20261107-phase1-Restore-ResponseJointPrior-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.response_joint_prior_trial'
ARM='TF_ResponseJointPrior'


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed source-response joint prior is allowed')
    cfg=control.configuration(control.ARM)
    cfg.update(run_id=RUN,arm=ARM,
        objective='Original source Action/noise and actual noisy IQ/STFT/.01continuousF0 plus .01 source normalized squared IQ response error.',
        source_response_epsilon=EPSILON,source_response_weight=RESPONSE_WEIGHT,source_response_directions=list(DIRECTIONS),
        source_response_assignment='(original source row index + epoch) modulo4; all63000 each epoch, exact quarter exposure.',
        source_response_observation='received + recorded simulator Action(delta); same realized source noise, no second RMS or true identity.',
        source_response_gradient='Prior mean/covariance/context only, same physical gradient isolation as control.',
        training_posterior_calls_per_record=2,training_solver='Equivalent FP64 low-rank conditional-prior posterior; physical/source moments fixed.',
        deployment_solver='Original dense FP64 joint-prior posterior; no perturbation or extra deployment call.',
        scientific_change='Adds fixed source response supervision. Source solver algebra is equivalent within measured tolerances, not bitwise-identical training.',
        source_response_evidence='20261107-diagnostic-Restore-CovarianceResponse-sourceLU-s2026092701-r01;224 legal L/U rows,84 fixed response cells.',
        source_solver_evidence='20261107-diagnostic-Restore-CoupledLowRank-sourceLU-s2026092701-r03;all14 actual source numerical/gradient cells and42 paired timings.',
        matched_control=control.RUN+'/'+control.ARM)
    return cfg


def factory(arm):return scratch_response_joint_prior(configuration(arm)['model_seed'])


def response_readout(model,reader,received,clean,features,scale,objective,*,source_indices,epoch,priors,assignment,source_trace):
    if source_trace is None or not torch.equal(source_trace['received'],received):raise ValueError('Exact current source trace must accompany response training')
    restored,aux,loss,metrics=control.joint_readout(model,reader,received,clean,features,scale,objective,
        source_indices=source_indices,epoch=epoch,priors=priors,assignment=assignment)
    delta,choice=selected_perturbation(clean,source_indices,epoch)
    h,main,image,_=known_parameters(source_trace);action=dense_action(h,main,image)
    changed=inject(received,delta,action)
    prior=choose_prior(priors,assignment,source_indices,epoch)
    altered,altered_aux=model.source_readout(changed,prior=prior)
    response=response_loss(restored,altered,delta)
    if not torch.isfinite(response).all():raise ValueError('Finite source response objective required')
    if epoch==0:
        with torch.no_grad():
            dense,_=model._readout(received,prior)
            torch.testing.assert_close(restored,dense,rtol=2e-5,atol=2e-6)
        parameters=(model.detail.output.weight,model.detail.mean_output.weight)
        gradients=torch.autograd.grad(response.mean(),parameters,retain_graph=True)
        if any(not torch.isfinite(g).all() or not g.abs().sum()>0 for g in gradients):raise ValueError('Response objective must reach both prior heads')
        model.response_smoke_report=dict(actual_source_response_gradients=True,actual_training_deployment_posterior_agreement=True,
            response_covariance_gradient_absolute_sum=float(gradients[0].abs().sum()),response_mean_gradient_absolute_sum=float(gradients[1].abs().sum()))
    metrics.update(source_response_relative_error=response,source_response_weighted_loss=RESPONSE_WEIGHT*response,
        source_response_relative_RMS=(delta.square().mean((1,2))/clean.square().mean((1,2))).sqrt())
    for i,name in enumerate(DIRECTIONS):metrics['source_response_fraction_'+name]=(choice==i).to(loss.dtype)
    aux['relative_solve_residual']=torch.maximum(aux['relative_solve_residual'],altered_aux['relative_solve_residual'])
    return restored,aux,loss+RESPONSE_WEIGHT*response,metrics


def smoke_gradient_check(model):
    return dict(control.mean_smoke_gradient_check(model),**model.response_smoke_report)


def train(args,out):
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(c,source_modes(c)) for c in pair[1]])
    priors=tuple(v.to('cuda:0') for v in (*pair,modes))
    readout=partial(response_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout,
        smoke_check=control.initial_smoke,smoke_gradient_check=smoke_gradient_check,training_trace=True,training_posterior_multiplier=2)


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train=configuration,factory,train
    trial.gpu_choice=partial(recovery_gpu_choice,allowed_gpus=(6,),maximum_processes=5)


if __name__=='__main__':install();trial.main()
