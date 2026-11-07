"""Source reliability-weighted response constraint with the original dense posterior."""
from functools import partial

import torch

from . import covariance_detail_trial as training
from . import multiview_trial as trial
from . import response_joint_prior_dense_trial as dense
from . import response_joint_prior_trial as original
from .coupled_covariance import normalize_modes
from .mismatched_prior_training import choose_prior,load_training_priors
from .signed_covariance import source_modes
from .variance_trial import recovery_gpu_choice

RUN='20261108-phase1-Restore-ResponseReliability-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.response_reliability_trial'
ARM='TF_ResponseReliability'
KAPPA_REFERENCE=9.


def response_weight(delta,action,variance):
    if delta.ndim!=3 or delta.shape[1:]!=(2,256) or action.shape!=(len(delta),512,512) or variance.shape!=(len(delta),):
        raise ValueError('Matched source perturbation, action and noise required')
    with torch.no_grad():
        if not all(torch.isfinite(x).all() for x in (delta,action,variance)) or (variance<=0).any():
            raise ValueError('Finite source action and positive noise proxy required')
        observed=(action.double()@delta.flatten(1).double().unsqueeze(-1)).squeeze(-1)
        kappa=observed.square().sum(1)/variance.double()
        weight=kappa/(kappa+KAPPA_REFERENCE)
        if not torch.isfinite(weight).all():raise ValueError('Finite source response weight required')
    return weight,kappa


def configuration(arm):
    if arm!=ARM:raise ValueError('Only the fixed response reliability candidate is allowed')
    cfg=dense.configuration(dense.ARM)
    cfg.update(run_id=RUN,arm=ARM,matched_control=dense.RUN+'/'+dense.ARM,
        objective='Original source Action/noise and actual noisy IQ/STFT/.01continuousF0 plus .01 source reliability-weighted normalized squared IQ response error.',
        source_response_weighting='kappa/(kappa+9), kappa=||A_true delta||^2/recorded scalar source noise variance; source training only, no normalization.',
        source_response_reliability_reference=KAPPA_REFERENCE,
        scientific_change='Response weight changes continuously with the source recorded action/noise proxy, reducing both low-reliability contribution and total effective response strength.',
        source_reliability_evidence='20261108-diagnostic-Restore-ResponseObservability-sourceLU-s2026092701-r01; 224 source L/U records, no target or TX.',
        source_action_deployment_input=False,source_noise_deployment_input=False)
    return cfg


def factory(arm):
    configuration(arm)
    return dense.factory(dense.ARM)


def response_readout(model,reader,received,clean,features,scale,objective,*,source_indices,epoch,priors,assignment,source_trace):
    if source_trace is None or not torch.equal(source_trace['received'],received):
        raise ValueError('Exact current source trace must accompany response training')
    restored,aux,loss,metrics=original.control.joint_readout(model,reader,received,clean,features,scale,objective,
        source_indices=source_indices,epoch=epoch,priors=priors,assignment=assignment)
    delta,choice=original.selected_perturbation(clean,source_indices,epoch)
    h,main,image,variance=original.known_parameters(source_trace);action=original.dense_action(h,main,image)
    weight,kappa=response_weight(delta,action,variance)
    changed=original.inject(received,delta,action)
    prior=choose_prior(priors,assignment,source_indices,epoch)
    altered,altered_aux=model.source_readout(changed,prior=prior)
    response=original.response_loss(restored,altered,delta)
    if not torch.isfinite(response).all():raise ValueError('Finite source response objective required')
    weighted=original.RESPONSE_WEIGHT*weight.to(response.dtype)*response
    if epoch==0:
        with torch.no_grad():
            deployed,_=model._readout(received,prior)
            torch.testing.assert_close(restored,deployed,rtol=2e-5,atol=2e-6)
        parameters=(model.detail.output.weight,model.detail.mean_output.weight)
        gradients=torch.autograd.grad(weighted.mean(),parameters,retain_graph=True)
        if any(not torch.isfinite(g).all() or not g.abs().sum()>0 for g in gradients):
            raise ValueError('Weighted source response must reach both prior heads')
        model.response_smoke_report=dict(actual_source_response_gradients=True,actual_training_deployment_posterior_agreement=True,
            response_covariance_gradient_absolute_sum=float(gradients[0].abs().sum()),response_mean_gradient_absolute_sum=float(gradients[1].abs().sum()))
    metrics.update(source_response_relative_error=response,source_response_weighted_loss=weighted,
        source_response_reliability=weight.to(response.dtype),source_response_kappa=kappa.to(response.dtype),
        source_response_relative_RMS=(delta.square().mean((1,2))/clean.square().mean((1,2))).sqrt())
    for i,name in enumerate(original.DIRECTIONS):metrics['source_response_fraction_'+name]=(choice==i).to(loss.dtype)
    aux['relative_solve_residual']=torch.maximum(aux['relative_solve_residual'],altered_aux['relative_solve_residual'])
    return restored,aux,loss+weighted,metrics


def train(args,out):
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(c,source_modes(c)) for c in pair[1]])
    priors=tuple(v.to('cuda:0') for v in (*pair,modes))
    readout=partial(response_readout,priors=priors,assignment=assignment.to('cuda:0'))
    training.train(args,out,build=factory,config_for=configuration,training_readout=readout,
        smoke_check=original.control.initial_smoke,smoke_gradient_check=original.smoke_gradient_check,
        training_trace=True,training_posterior_multiplier=2)


def gpu_choice():
    return [item for item in recovery_gpu_choice(allowed_gpus=(0,2),maximum_processes=3)
            if item['free_memory_MB']>=12288]


def install():
    trial.RUN,trial.MODULE,trial.ARMS=RUN,MODULE,(ARM,)
    trial.configuration,trial.factory,trial.train=configuration,factory,train
    trial.gpu_choice=gpu_choice


if __name__=='__main__':install();trial.main()
