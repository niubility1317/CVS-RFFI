"""Source-activation factor initialization for the original shared-prior R46.

No files, labels, classifier, optimizer or experiment launcher are accessed here.
Callers must supply permitted source-training IQ and the verified frozen parent.
"""
import copy
import torch
from torch import nn
from experiments.pamf_r17_compact.models import FactorConv
from tools.r17_activation_factorization import InputSecondMoment, calibrated_factor, factor_rank


def roots(model):
    if getattr(model, 'arm', None) != 'prune_refine' or len(model.prior) != 3 or sum(p.numel() for p in model.parameters()) != 350521:
        raise ValueError('Original 350521-parameter R46 parent required')
    return dict(coarse_spectral=[model.frozen.coarse.spectral], coarse_temporal=[model.frozen.coarse.temporal],
                action=[model.frozen.estimator.core], prior=list(model.prior))


def eligible_groups(model, fraction):
    groups = {}
    for group, cores in roots(model).items():
        mappings = [dict(core.named_modules()) for core in cores]
        for name, conv in mappings[0].items():
            if not isinstance(conv, nn.Conv1d) or conv.kernel_size[0] <= 1:
                continue
            try:
                factor_rank(conv, fraction)
            except ValueError:
                continue
            members = [mapping[name] for mapping in mappings]
            if any(member.weight.shape != conv.weight.shape for member in members):
                raise ValueError('Matched kernel dimensions required')
            if group == 'prior' and any(member.weight is not conv.weight for member in members):
                raise ValueError('Original shared prior kernels required; do not average distinct teachers')
            groups[group+'::'+name] = dict(group=group, name=name, members=members)
    if not groups:
        raise ValueError('At least one saving kernel factorization required')
    return groups


class SharedInputMoments:
    """Pool actual teacher input moments over the three uses of shared kernels."""
    def __init__(self, teacher, minimum_fraction=.5, positions=8):
        if any(p.device.type != 'cpu' or p.requires_grad for p in teacher.parameters()) or teacher.training:
            raise ValueError('Frozen CPU parent required for source calibration')
        self.teacher = teacher
        self.groups = eligible_groups(teacher, minimum_fraction)
        self.moments = {key:InputSecondMoment(value['members'][0], positions) for key,value in self.groups.items()}
        self.positions = positions
        self.handles = []
        self.examples = 0

    def __enter__(self):
        if self.handles:
            raise ValueError('Calibration hooks already active')
        for key, value in self.groups.items():
            def observe(module, inputs, moment=self.moments[key]):
                moment.update(inputs[0].detach())
            for member in value['members']:
                self.handles.append(member.register_forward_pre_hook(observe))
        return self

    @torch.no_grad()
    def observe(self, values):
        if not self.handles or values.device.type != 'cpu' or values.dtype != torch.float32 or values.ndim != 3 or values.shape[1:] != (2,256):
            raise ValueError('Active hooks and native CPU source IQ[B,2,256] required')
        if not torch.isfinite(values).all():
            raise ValueError('Finite source IQ required')
        restored = self.teacher(values)
        if restored.shape != values.shape or not torch.isfinite(restored).all():
            raise ValueError('Finite complete teacher output required')
        self.examples += len(values)
        return restored

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def finish(self):
        if self.handles or not self.examples:
            raise ValueError('Close hooks after actual source observations before extracting moments')
        moments = {key:moment.value() for key,moment in self.moments.items()}
        report = dict(teacher_examples=self.examples, positions_per_call=self.positions,
            groups={key:dict(input_dimension=value.shape[0], patch_count=self.moments[key].count,
                shared_stage_uses=len(self.groups[key]['members'])) for key,value in moments.items()},
            optimizer_updates=0, TX_labels_read=0, F_reads=0, V_reads=0, target_reads=0)
        return moments, report


@torch.no_grad()
def calibrated_copy(parent, moments, fraction, ridge_fraction=1e-3):
    groups = eligible_groups(parent, fraction)
    if not set(groups) <= set(moments):
        raise ValueError('Moments for every factored original-parent kernel required')
    model = copy.deepcopy(parent)
    destinations = roots(model)
    reports = {}
    for key, value in groups.items():
        first, report = calibrated_factor(value['members'][0], moments[key], fraction, ridge_fraction)
        for index, (core, original) in enumerate(zip(destinations[value['group']], value['members'])):
            factor = first if index == 0 else FactorConv.from_conv(original, fraction, initialize=False)
            if index:
                factor.left.weight = first.left.weight
                factor.right.weight = first.right.weight
                if original.bias is not None:
                    factor.right.bias.copy_(original.bias)
            prefix, _, name = value['name'].rpartition('.')
            setattr(core.get_submodule(prefix) if prefix else core, name, factor)
        reports[key] = dict(report, shared_stage_uses=len(value['members']))
    model.requires_grad_(True)
    model.initialization_report = None
    return model, reports
