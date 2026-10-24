"""Disposable parameter-only first steps through the actual physical AdamW adapter."""
import math
import torch
from torch import nn

from experiments.cric_dura_comparison.f_optimizer import PREFIX, build, clip_physical


class CoordinateSinc(nn.Module):
    """Metadata and parameter container only; never used for a forward pass."""


def _parameter_shadow(model, named):
    observed = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) != len(observed) or any(n != a or p is not q for (n, p), (a, q) in zip(named, observed)):
        raise ValueError('Exact ordered original trainable parameters required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for _, p in named):
        raise ValueError('This disposable diagnostic is CPU FP32 only')
    original_sinc = model.get_submodule(PREFIX[:-1])
    if (type(original_sinc).__name__ != 'CoordinateSinc' or original_sinc.mode != 'CENTER_25K'
            or original_sinc.sample_rate != 25000000. or original_sinc.scale_hz != 25000.):
        raise ValueError('The actual centered Sinc unit contract is required')
    shadow = nn.Module()
    for name, parameter in named:
        node = shadow
        parts = name.split('.')
        for i, part in enumerate(parts[:-1]):
            if part not in node._modules:
                path = '.'.join(parts[:i + 1])
                node.add_module(part, CoordinateSinc() if path == PREFIX[:-1] else nn.Module())
            node = node._modules[part]
        node.register_parameter(parts[-1], nn.Parameter(parameter.detach().clone()))
    sinc = shadow.get_submodule(PREFIX[:-1])
    sinc.sample_rate = original_sinc.sample_rate
    sinc.scale_hz = original_sinc.scale_hz
    sinc.mode = original_sinc.mode
    for name in ('low_origin_hz', 'band_origin_hz'):
        sinc.register_buffer(name, getattr(original_sinc, name).detach().clone())
    return shadow


def shadow_step(model, named, gradients, *, base_lr, weight_decay):
    """Return theta_before-theta_after; original model and its gradients stay untouched."""
    named = list(named)
    if (not math.isfinite(base_lr) or base_lr <= 0 or not math.isfinite(weight_decay)
            or weight_decay < 0 or len(gradients) != len(named)):
        raise ValueError('One finite positive-LR native first-step scope required')
    shadow = _parameter_shadow(model, named)
    lookup = dict(shadow.named_parameters())
    parameters = [lookup[name] for name, _ in named]
    for (_, source), parameter, gradient in zip(named, parameters, gradients):
        if gradient is not None:
            if (gradient.device != source.device or gradient.dtype != source.dtype
                    or gradient.shape != source.shape or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Finite matching original CPU FP32 gradients required')
            parameter.grad = gradient.detach().clone()
    optimizer = build(shadow, parameters, 'CENTER_UNITS', lr=float(base_lr), weight_decay=float(weight_decay))
    if optimizer.state:
        raise ValueError('Only the fresh-moment first step is represented')
    before = [p.detach().clone() for p in parameters]
    norm = clip_physical(optimizer, parameters, 5.)
    if not bool(torch.isfinite(norm)):
        raise ValueError('Nonfinite physical gradient norm')
    optimizer.step()
    if optimizer._sinc_unit_steps != 1 or optimizer._sinc_physical_clip_calls != 1:
        raise ValueError('Exactly one actual physical clip and native AdamW step required')
    descent = tuple(a - p.detach() for a, p in zip(before, parameters))
    if any(not bool(torch.isfinite(d).all()) for d in descent):
        raise ValueError('Nonfinite disposable optimizer displacement')
    metadata = dict(optimizer='AdamW_with_actual_CENTER_UNITS_adapter', optimizer_steps=1,
                    physical_clip_calls=1, fresh_moments=True, max_grad_norm=5.,
                    physical_grad_norm_before_clip=float(norm), base_lr=float(base_lr),
                    base_weight_decay=float(weight_decay), source_parameter_or_grad_writes=0,
                    shadow_parameter_gradient_tensors=sum(g is not None for g in gradients),
                    parameter_groups=[dict(lr=float(g['lr']), eps=float(g['eps']),
                                           weight_decay=float(g['weight_decay']), betas=list(g['betas']),
                                           parameter_tensors=len(g['params'])) for g in optimizer.param_groups],
                    source_model_forward_calls=0, shadow_model_forward_calls=0,
                    interpretation='Actual CPU FP32 first-step displacement on independent parameter containers;fresh moments only;all shadows discarded.')
    return descent, metadata
