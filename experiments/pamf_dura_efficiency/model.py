import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_prior_refine.model import spectral_error, stage_details
from experiments.pamf_restoration.models import complex_iq


def scratch_model(seed):
    """Build only the graph: no checkpoint, ancestor state or IQ is loaded."""
    from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=StageDistilled()
    model.initialization_report=dict(mode='random_single_stage',model_seed=seed,
        checkpoint_sources=[],ancestors=[],training_updates=0,IQ_examples=0,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        initialization='Random graph parameters plus existing deterministic architecture defaults; no inherited learned state')
    return model.train().requires_grad_(True)


def supervised_action(kernel,coefficients):
    return Action(complex_iq(kernel),torch.complex(coefficients[:,0],coefficients[:,1]),
                  torch.complex(coefficients[:,2],coefficients[:,3]))


def direct_objective(model,received,clean,actual_action,actual_variance,probes,*,return_restored=False):
    """All source targets enter one joint update; no learned teacher is needed.

    Source simulator action/variance supervision is explicitly additional to IQ
    pairing. The deployment forward still consumes only received IQ.
    """
    context=model.estimate_context(received)
    # Avoid estimating the context twice while retaining the existing stages.
    base,action,native=context
    from experiments.pamf_action_restore.models import SCALE
    from experiments.pamf_restoration.models import pair
    from experiments.pamf_uncertainty_restore.models import weighted_map
    observed,current=complex_iq(received),base
    variance=native[:,None].expand(-1,256)
    logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
    priors=[]
    for core in model.prior:
        innovation=pair(action.adjoint(observed-action(complex_iq(current))))
        innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
        prior=base+SCALE*core(torch.cat((received,base,current,innovation,logvar),1))
        current=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=complex_iq(current)))
        priors.append(prior)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(current-clean.detach()).square().mean((1,2))/energy
    prior_error=torch.stack([(p-clean.detach()).square().mean((1,2))/energy for p in priors]).mean(0)
    action_errors=[]
    for probe in probes:
        with torch.no_grad():desired=actual_action(probe)
        difference=action(probe)-desired
        action_errors.append(difference.abs().square().mean(1)/desired.abs().square().mean(1).clamp_min(1e-4))
    action_error=torch.stack(action_errors).mean(0)
    noise_error=(native.log()-actual_variance.detach().log()).square()
    spectral=spectral_error(current,clean)
    loss=paired+.1*prior_error+.05*spectral+.05*action_error+.001*noise_error
    terms=dict(iq_nmse=paired,prior_nmse=prior_error,spectral_error=spectral,
               action_probe_nmse=action_error,log_variance_mse=noise_error)
    return (loss,terms,current) if return_restored else (loss,terms)
