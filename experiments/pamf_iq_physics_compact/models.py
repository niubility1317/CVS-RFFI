import math
import torch
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,compact_from_teacher,convolution_linear_macs,objective as iq_objective
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_restoration.models import complex_iq
from . import design as d


def make_probes(count,device,generator):
    return [torch.complex(*torch.randn((2,count,256),device=device,generator=generator))/math.sqrt(2)
            for _ in range(d.PROBE_COUNT)]


def action_from_arrays(kernel,coefficients):
    if kernel.ndim!=3 or kernel.shape[1:]!=(2,64) or coefficients.shape!=(len(kernel),4,256):
        raise ValueError('Original simulated FIR64/main/image target required')
    if not torch.isfinite(kernel).all() or not torch.isfinite(coefficients).all():
        raise ValueError('Finite original source simulation target required')
    return Action(complex_iq(kernel.detach()),torch.complex(coefficients[:,0],coefficients[:,1]).detach(),
                  torch.complex(coefficients[:,2],coefficients[:,3]).detach())


def objective(outputs,teacher,clean,epoch,context,target,target_variance,probes):
    base,estimated,native=context
    count=len(clean)
    if (base.shape!=clean.shape or native.shape!=(count,) or target_variance.shape!=(count,) or
            not torch.isfinite(native).all() or not torch.isfinite(target_variance).all() or
            (native<=0).any() or (target_variance<=0).any() or len(probes)!=d.PROBE_COUNT or
            any(p.shape!=(count,256) or not p.is_complex() or not torch.isfinite(p).all() for p in probes)):
        raise ValueError('Matched positive native variances and two source-only complex probes required')
    loss,terms=iq_objective(outputs,teacher,clean,epoch)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    coarse=(base-clean.detach()).square().mean((1,2))/energy
    actions=[]
    for probe in probes:
        with torch.no_grad():truth=target(probe.detach())
        actions.append((estimated(probe.detach())-truth).abs().square().mean(-1)/truth.abs().square().mean(-1).clamp_min(1e-4))
    physical=dict(coarse_clean_nmse=coarse,action_probe_nmse=torch.stack(actions).mean(0),
                  noise_log_variance_mse=(native.log()-target_variance.detach().log()).square())
    return loss+sum(d.WEIGHTS[key]*value for key,value in physical.items()),dict(terms,**physical)
