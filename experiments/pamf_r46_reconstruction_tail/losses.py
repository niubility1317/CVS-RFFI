"""Current-batch, label-free reconstruction weighting, with all examples retained."""
import math
import time
import torch
from . import design as d
from experiments.pamf_r46_multipoint_fast.models import objective as original_objective


def reconstruction_risk(values,arm):
    if arm not in d.ARMS:raise ValueError('Fixed source-only tail fraction required')
    if (values.ndim!=1 or not len(values) or not values.is_floating_point()
            or not torch.isfinite(values).all() or bool((values<0).any())):
        raise ValueError('Finite nonnegative per-example source reconstruction errors required')
    started=time.perf_counter()
    n=len(values);k=max(1,math.ceil(d.FRACTION[arm]*n))
    indices=torch.argsort(values.detach(),descending=True,stable=True)[:k]
    selected=torch.zeros(n,dtype=torch.bool,device=values.device);selected[indices]=True
    weights=(1-d.MIX)+d.MIX*(n/k)*selected.to(values.dtype)
    return weights*values,dict(weights=weights,selected=selected,selected_count=k,batch=n,fraction=k/n,host_enqueue_seconds=time.perf_counter()-started)


def objective(student,teacher,clean,epoch,arm):
    _,terms=original_objective(student,teacher,clean,epoch)
    reconstruction=terms['iq_nmse']+.05*terms['spectral_error']
    risk,focus=reconstruction_risk(reconstruction,arm)
    loss=risk+d.teacher_weight(epoch)*terms['teacher_stage_nmse']
    tail=reconstruction*(focus['batch']/focus['selected_count'])*focus['selected'].to(reconstruction.dtype)
    return loss,dict(**terms,reconstruction_mean=reconstruction,reconstruction_tail=tail,reconstruction_risk=risk),focus
