import copy
import math
import torch
from torch import nn

from . import design as d
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


class FactorConv(nn.Module):
    @classmethod
    def from_conv(cls, conv, fraction=.5, *, initialize=True):
        if (not isinstance(conv, nn.Conv1d) or conv.groups != 1 or conv.padding_mode != 'zeros'
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError('Ordinary zero-padded Conv1d and positive rank fraction<=1 required')
        rank = max(1, int(min(conv.out_channels, conv.in_channels*conv.kernel_size[0])*fraction))
        value = cls()
        value.left = nn.Conv1d(conv.in_channels, rank, conv.kernel_size, stride=conv.stride,
                              padding=conv.padding, dilation=conv.dilation, bias=False).to(conv.weight)
        value.right = nn.Conv1d(rank, conv.out_channels, 1, bias=conv.bias is not None).to(conv.weight)
        if initialize:
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(conv.weight.flatten(1), full_matrices=False)
                root = s[:rank].sqrt()
                value.left.weight.copy_((root[:, None]*vh[:rank]).reshape_as(value.left.weight))
                value.right.weight.copy_((u[:, :rank]*root[None])[:, :, None])
                if conv.bias is not None:
                    value.right.bias.copy_(conv.bias)
        return value

    def forward(self, value):
        return self.right(self.left(value))


def factor_core(module, *, initialize):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d) and child.kernel_size[0] > 1:
            rank = max(1, min(child.out_channels, child.in_channels*child.kernel_size[0])//2)
            if rank*(child.in_channels*child.kernel_size[0]+child.out_channels) < child.weight.numel():
                setattr(module, name, FactorConv.from_conv(child, initialize=initialize))
        else:
            factor_core(child, initialize=initialize)
    return module


def stages(priors, received, context):
    if received.ndim != 3 or received.shape[1:] != (2, 256) or len(priors) not in (1, 3):
        raise ValueError('Current IQ2x256 and one or three priors required')
    base, action, native = context
    variance = native[:, None].expand(-1, 256)
    observed, current, outputs = complex_iq(received), base, []
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
    for index in range(3):
        innovation = pair(action.adjoint(observed-action(complex_iq(current))))
        innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
        inputs = torch.cat((received, base, current, innovation, logvar), 1)
        prior = base+SCALE*priors[0 if len(priors) == 1 else index](inputs)
        current = pair(weighted_map(action, observed, complex_iq(prior), variance,
                                    steps=2, initial=complex_iq(current)))
        outputs.append(current)
    return outputs, variance, action, base


class CompactRestorer(nn.Module):
    def __init__(self, arm, frozen, priors=None):
        super().__init__()
        if arm not in d.ARMS:
            raise ValueError('Registered R17 compression arm required')
        self.arm = arm
        self.frozen = frozen.eval().requires_grad_(False)
        if priors is None:
            priors = [Core(incoming=9)] if arm == 'shared_mean' else [factor_core(Core(incoming=9), initialize=False) for _ in range(3)]
        self.prior = nn.ModuleList(priors)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def stage_outputs(self, received, context=None):
        if context is None:
            with torch.no_grad():
                context = self.frozen.estimate(received)
        return stages(self.prior, received, context)

    def training_outputs(self, received):
        values, variance, action, base = self.stage_outputs(received)
        return values[-1], variance, action, base

    def forward(self, received):
        return self.training_outputs(received)[0]


def compact_from_teacher(teacher, arm):
    if teacher.candidate != 'unroll_unshared' or len(teacher.prior) != 3 or arm not in d.ARMS:
        raise ValueError('Original R17 unshared teacher and registered arm required')
    if arm == 'shared_mean':
        core = copy.deepcopy(teacher.prior[0])
        core.load_state_dict({key:torch.stack([v.state_dict()[key] for v in teacher.prior]).mean(0)
                              for key in core.state_dict()}, strict=True)
        priors = [core]
    else:
        priors = [factor_core(copy.deepcopy(core), initialize=True) for core in teacher.prior]
    # Teacher parameters are frozen; copied student priors must be trainable.
    for core in priors:
        core.requires_grad_(True)
    return CompactRestorer(arm, copy.deepcopy(teacher.frozen), priors)


def distillation_loss(student, teacher, clean):
    if len(student) != 3 or len(teacher) != 3 or any(v.shape != clean.shape for v in (*student, *teacher)):
        raise ValueError('Three matched student/teacher stages and paired source clean required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-4)
    paired = (student[-1]-clean).square().mean((1, 2))/energy
    distance = torch.stack([(a-b.detach()).square().mean((1, 2))/energy for a, b in zip(student, teacher)]).mean(0)
    return paired+d.DISTILL_WEIGHT*distance, dict(iq_nmse=paired, teacher_stage_nmse=distance)
