import numpy as np
import torch
from . import design as d
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_rf_response.models import directions,oracle_response
from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_restoration.models import pair,complex_iq
from experiments.pamf_iqonly_restore.training import iq_loss


class ResponseRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:raise ValueError('Registered response-prior candidate required')
        super().__init__('unroll_unshared',frozen)
        self.candidate=candidate


@torch.no_grad()
def training_delta(clean,indices,epoch):
    if not 1<=epoch<=20 or len(indices)!=len(clean):raise ValueError('Original E20 source visit required')
    codes=np.stack([np.random.default_rng(np.random.SeedSequence([d.RESPONSE_SEED,epoch,int(i)]))
                    .integers([0,0],[len(d.DIRECTIONS),2]) for i in indices])
    delta=torch.zeros_like(complex_iq(clean));valid=torch.zeros(len(clean),dtype=torch.bool,device=clean.device)
    for amplitude_index,amplitude in enumerate((.01,.02)):
        # The validation panel retains the original R31 random-direction stream.
        # Use a disjoint seed namespace for L/U training random controls.
        values=directions(clean,[int(i)+(1<<32) for i in indices],amplitude)
        for direction_index,name in enumerate(d.DIRECTIONS):
            selected=torch.tensor(((codes[:,0]==direction_index)&(codes[:,1]==amplitude_index)).tolist(),device=clean.device)
            value,ok=values[name];delta=torch.where(selected[:,None],value,delta);valid|=selected&ok
    return delta,valid,codes


@torch.no_grad()
def teacher_response(action,delta,variance,valid):
    change=action(delta);target,residual=oracle_response(action,change,variance)
    energy=real_dot(delta,delta).clamp_min(1e-20)
    oracle_error=real_dot(target-delta,target-delta)/energy
    fisher=real_dot(change,change)/variance
    eligible=valid&(fisher>=9)&(oracle_error<=.1)&(residual<=.001)
    if not torch.isfinite(target).all() or not torch.isfinite(fisher).all():raise FloatingPointError('Invalid source response teacher')
    return change,target,eligible,dict(oracle_nmse=oracle_error,fisher_energy=fisher,CG_residual=residual)


def paired_objective(output,plus,minus,clean,delta):
    losses=[iq_loss(output,clean)[0],iq_loss(plus,clean+pair(delta))[0],iq_loss(minus,clean-pair(delta))[0]]
    return torch.stack(losses).mean(0)


def response_objective(plus,minus,target,delta,eligible):
    if target.requires_grad or delta.requires_grad or eligible.requires_grad:
        raise ValueError('Source-only fixed response targets must be detached')
    response=complex_iq((plus-minus)/2)
    denominator=real_dot(delta,delta).clamp_min(1e-20)
    error=real_dot(response-target,response-target)/denominator
    if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite response loss')
    return torch.where(eligible,error,torch.zeros_like(error)),error
