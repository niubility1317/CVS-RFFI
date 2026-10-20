import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_r17_compact.models import FactorConv,stages
from experiments.pamf_r17_fidelity.models import distillation_loss
from tools.r17_activation_factorization import calibrated_factor,factor_rank


def factor_priors(priors,fraction,moments=None):
    reports=[]
    for stage,core in enumerate(priors):
        for name,conv in list(core.named_modules()):
            if not isinstance(conv,nn.Conv1d) or conv.kernel_size[0]<=1:continue
            try:factor_rank(conv,fraction)
            except ValueError:continue
            if moments is None:factor=FactorConv.from_conv(conv,fraction,initialize=False)
            else:
                if (stage,name) not in moments:raise ValueError('All eligible original-teacher layer moments required')
                factor,report=calibrated_factor(conv,moments[stage,name],fraction,d.RIDGE_FRACTION)
                reports.append(dict(stage=stage,name=name,**report))
            prefix,_,last=name.rpartition('.')
            setattr(core.get_submodule(prefix) if prefix else core,last,factor)
    return priors,reports


class CalibratedRestorer(nn.Module):
    def __init__(self,arm,frozen,priors=None):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered activation-calibrated R17 arm required')
        self.arm=arm;self.frozen=frozen.eval().requires_grad_(False);self.calibration_report=None
        if priors is None:priors=factor_priors([Core(incoming=9) for _ in range(3)],d.FRACTIONS[arm])[0]
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


def compact_from_teacher(teacher,arm,moments):
    if teacher.candidate!='unroll_unshared' or len(teacher.prior)!=3 or arm not in d.ARMS:
        raise ValueError('Only original R17 unshared teacher and registered calibrated arm required')
    priors,reports=factor_priors([copy.deepcopy(core) for core in teacher.prior],d.FRACTIONS[arm],moments)
    for core in priors:core.requires_grad_(True)
    return CalibratedRestorer(arm,copy.deepcopy(teacher.frozen),priors),reports
