import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_r17_compact.models import factor_core,stages


def share_long_weights(priors):
    maps=[dict(core.named_modules()) for core in priors]
    for name,module in maps[0].items():
        if isinstance(module,nn.Conv1d) and module.kernel_size[0]>1:
            shared=nn.Parameter(torch.stack([values[name].weight.detach() for values in maps]).mean(0))
            for values in maps:values[name].weight=shared
    return priors


class FidelityRestorer(nn.Module):
    def __init__(self,arm,frozen,priors=None):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered pure R17 fidelity arm required')
        self.arm=arm;self.frozen=frozen.eval().requires_grad_(False)
        if priors is None:
            priors=[Core(incoming=9) for _ in range(3)]
            if arm=='step_norm_fidelity':share_long_weights(priors)
            else:priors=[factor_core(core,initialize=False) for core in priors]
        self.prior=nn.ModuleList(priors)

    def train(self,mode=True):
        super().train(mode);self.frozen.eval();return self

    def stage_outputs(self,received,context=None):
        if context is None:
            with torch.no_grad():context=self.frozen.estimate(received)
        return stages(self.prior,received,context)

    def training_outputs(self,received):
        values,variance,action,base=self.stage_outputs(received)
        return values[-1],variance,action,base

    def forward(self,received):return self.training_outputs(received)[0]


def compact_from_teacher(teacher,arm):
    if teacher.candidate!='unroll_unshared' or len(teacher.prior)!=3 or arm not in d.ARMS:
        raise ValueError('Only original R17 unshared and registered pure fidelity arm required')
    priors=[copy.deepcopy(core) for core in teacher.prior]
    if arm=='step_norm_fidelity':share_long_weights(priors)
    else:priors=[factor_core(core,initialize=True) for core in priors]
    for core in priors:core.requires_grad_(True)
    return FidelityRestorer(arm,copy.deepcopy(teacher.frozen),priors)


def distillation_loss(student,teacher,clean):
    if len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):
        raise ValueError('Three matched original-R17/student stages and source energy reference required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    distance=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    # This paired-clean diagnostic is never included in the optimized objective.
    paired=(student[-1]-clean).square().mean((1,2))/energy
    return d.DISTILL_WEIGHT*distance,dict(iq_nmse=paired,teacher_stage_nmse=distance)
