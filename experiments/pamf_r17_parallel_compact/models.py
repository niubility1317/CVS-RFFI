import copy
import math
import torch
from torch import nn

from . import design as d
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_r17_compact.models import FactorConv
from experiments.pamf_r17_fidelity.models import FidelityRestorer,distillation_loss,stages


def _replace(core,name,value):
    pieces=name.split('.');parent=core
    for part in pieces[:-1]:parent=getattr(parent,part)
    setattr(parent,pieces[-1],value)


def share_factor_long(priors,fraction,*,initialize):
    """Compose mean sharing and balanced SVD, retaining every step's bias/state.

    Initialization is CPU-only so the matrix factorization does not allocate a
    CUDA solver workspace. No IQ, classifier, or target data are consumed.
    """
    if len(priors)!=3 or not math.isfinite(fraction) or fraction not in (.375,.25,.125):
        raise ValueError('Three priors and fixed registered rank required')
    maps=[dict(core.named_modules()) for core in priors]
    if any(set(values)!=set(maps[0]) for values in maps[1:]):raise ValueError('Matching prior structures required')
    reports=[]
    for name,module in maps[0].items():
        if not isinstance(module,nn.Conv1d) or module.kernel_size[0]<=1:continue
        members=[values[name] for values in maps]
        if any(not isinstance(v,nn.Conv1d) or v.weight.shape!=module.weight.shape or v.groups!=1 or v.padding_mode!='zeros'
               or v.stride!=module.stride or v.padding!=module.padding or v.dilation!=module.dilation
               or (v.bias is None)!=(module.bias is None) for v in members):
            raise ValueError('Matching ordinary long convolutions required')
        if any(v.weight.device.type!='cpu' for v in members):raise ValueError('Construct shared factors on CPU before device transfer')
        mean=torch.stack([v.weight.detach() for v in members]).mean(0)
        rank=max(1,int(min(module.out_channels,module.in_channels*module.kernel_size[0])*fraction))
        cost=rank*(module.in_channels*module.kernel_size[0]+module.out_channels)
        record=dict(name=name,rank=rank,shared_full_kernel_parameters=module.weight.numel(),shared_factor_parameters=min(cost,module.weight.numel()))
        if cost>=module.weight.numel():
            shared=nn.Parameter(mean)
            for member in members:member.weight=shared
            record.update(factored=False,relative_mean_kernel_error=0.)
        else:
            prototype=copy.deepcopy(module)
            with torch.no_grad():prototype.weight.copy_(mean)
            master=FactorConv.from_conv(prototype,fraction,initialize=initialize)
            for index,(core,member) in enumerate(zip(priors,members)):
                factor=master if index==0 else copy.deepcopy(master)
                factor.left.weight=master.left.weight;factor.right.weight=master.right.weight
                if member.bias is not None:
                    with torch.no_grad():factor.right.bias.copy_(member.bias)
                _replace(core,name,factor)
            if initialize:
                with torch.no_grad():
                    approximation=master.right.weight[:,:,0]@master.left.weight.flatten(1)
                    error=float((approximation-mean.flatten(1)).norm()/mean.norm().clamp_min(1e-12))
            else:error=None
            record.update(factored=True,relative_mean_kernel_error=error)
        reports.append(record)
    return reports


def share_pointwise(priors):
    """Share original 1x1 weights only; retain step-specific bias and norm."""
    if len(priors)!=3:raise ValueError('Three original priors required')
    maps=[dict(core.named_modules()) for core in priors];reports=[]
    for name,module in maps[0].items():
        if not isinstance(module,nn.Conv1d) or module.kernel_size!=(1,):continue
        members=[mapping[name] for mapping in maps]
        if any(not isinstance(value,nn.Conv1d) or value.weight.shape!=module.weight.shape or value.groups!=1 or value.weight.device.type!='cpu' for value in members):
            raise ValueError('Matching original CPU pointwise convolutions required')
        shared=nn.Parameter(torch.stack([value.weight.detach() for value in members]).mean(0))
        for value in members:value.weight=shared
        reports.append(dict(name=name,shared_parameters=shared.numel()))
    return reports


class SharedFactorRestorer(FidelityRestorer):
    def __init__(self,arm,frozen=None,priors=None):
        nn.Module.__init__(self)
        if arm not in d.ARMS:raise ValueError('Registered shared factor arm required')
        self.arm=arm;self.frozen=(ActionRestorer(ComplexTFUNet()) if frozen is None else frozen).eval().requires_grad_(False)
        if priors is None:
            priors=[Core(incoming=9) for _ in range(3)]
            if d.POINTWISE[arm]:share_pointwise(priors)
            share_factor_long(priors,d.FRACTIONS[arm],initialize=False)
        self.prior=nn.ModuleList(priors)
        self.initialization_report=None


def compact_from_teacher(teacher,arm):
    if arm not in d.ARMS or getattr(teacher,'candidate',None)!='unroll_unshared' or len(teacher.prior)!=3:
        raise ValueError('Original R17 unshared teacher and fixed compact arm required')
    priors=[copy.deepcopy(core).cpu() for core in teacher.prior]
    pointwise=share_pointwise(priors) if d.POINTWISE[arm] else []
    reports=share_factor_long(priors,d.FRACTIONS[arm],initialize=True)
    for core in priors:core.requires_grad_(True)
    model=SharedFactorRestorer(arm,copy.deepcopy(teacher.frozen).cpu(),priors)
    model.initialization_report=dict(mode='data_free_shared_SVD',fraction=d.FRACTIONS[arm],layers=reports,
        pointwise_weights_shared=d.POINTWISE[arm],pointwise_layers=pointwise,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0)
    return model
