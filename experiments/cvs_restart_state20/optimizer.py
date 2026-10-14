"""Three hypothetical AdamW stage states on one unchanged parameter traversal.

This module consumes an already validated optimizer payload, never a checkpoint
path. It does not step, reset weights, change requires_grad/.grad, or create a
classifier. The caller must establish payload provenance and that the supplied
parameter sequence is the original saved optimizer traversal; unnamed state_dict
IDs alone cannot prove model/parameter identity.
"""
import copy
import math
from collections.abc import Mapping

import torch
from torch import nn

MODES = ('END_SAVED', 'RESTART_KEEP', 'RESTART_RESET')
RESTART_LR = 2e-4
_GROUP_KEYS = {'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad',
               'maximize', 'foreach', 'capturable', 'differentiable', 'fused'}


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not value.is_floating_point() or value.requires_grad:
            raise ValueError(name + ' must be a detached real scalar')
        value = float(value.detach())
    elif type(value) not in (int, float):
        raise ValueError(name + ' must be a real scalar')
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _finite_extra(value, name):
    """Preserve scheduler/custom metadata, rejecting its nonfinite numbers."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad or (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(name + ' contains attached/nonfinite metadata')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(name + ' contains nonfinite metadata')
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_extra(item, name + '.' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_extra(item, name + '[' + str(index) + ']')


def _validate(parameters, saved_state):
    if not isinstance(parameters, (list, tuple)) or not parameters:
        raise ValueError('Original nonempty ordered list/tuple of nn.Parameter is required')
    if any(not isinstance(p, nn.Parameter) or not p.is_floating_point()
           or p.layout != torch.strided or not bool(torch.isfinite(p.detach()).all())
           for p in parameters):
        raise ValueError('Parameters must be finite real floating dense nn.Parameters')
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Original optimizer traversal must not repeat parameter objects')
    if not isinstance(saved_state, Mapping) or set(saved_state) != {'state', 'param_groups'}:
        raise ValueError('Complete standard AdamW state_dict is required')
    groups = saved_state['param_groups']
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError('Only the original single AdamW parameter group is supported')
    group = groups[0]
    if (not _GROUP_KEYS.issubset(group) or not isinstance(group['params'], list)
            or any(type(index) is not int for index in group['params'])
            or group['params'] != list(range(len(parameters)))):
        raise ValueError('Saved IDs must be the complete exact ordered traversal 0..N-1')
    for name in ('lr', 'eps', 'weight_decay'):
        if _scalar(group[name], name) < 0:
            raise ValueError(name + ' must be nonnegative')
    if not isinstance(group['betas'], (tuple, list)) or len(group['betas']) != 2:
        raise ValueError('Saved AdamW betas must have length two')
    if any(not 0 <= _scalar(beta, 'beta') < 1 for beta in group['betas']):
        raise ValueError('Saved AdamW betas must be in [0,1)')
    for name in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
        if type(group[name]) is not bool:
            raise ValueError('Saved ' + name + ' must be bool')
    for name in ('foreach', 'fused'):
        if group[name] is not None and type(group[name]) is not bool:
            raise ValueError('Saved ' + name + ' must be bool or None')
    if group['foreach'] and group['fused']:
        raise ValueError('AdamW foreach and fused cannot both be True')
    if 'decoupled_weight_decay' in group and group['decoupled_weight_decay'] is not True:
        raise ValueError('AdamW requires decoupled weight decay')
    if group['differentiable']:
        raise ValueError('Differentiable AdamW is outside this temporary-state API')
    if group['capturable'] and any(p.device.type == 'cpu' for p in parameters):
        raise ValueError('CPU-capturable AdamW cannot perform the requested temporary step')
    if 'param_names' in group and (not isinstance(group['param_names'], list)
            or len(group['param_names']) != len(parameters)
            or any(not isinstance(name, str) for name in group['param_names'])
            or len(set(group['param_names'])) != len(parameters)):
        raise ValueError('Optional saved parameter names must form a unique complete traversal')
    for name, value in group.items():
        if name != 'params':
            _finite_extra(value, 'group.' + name)
    states = saved_state['state']
    if not isinstance(states, Mapping) or any(type(index) is not int or index not in group['params'] for index in states):
        raise ValueError('Saved state IDs must belong to the exact parameter group')
    state_keys = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if group['amsgrad'] else set())
    for index, state in states.items():
        if not isinstance(state, Mapping) or set(state) != state_keys:
            raise ValueError('Present AdamW moments must be complete and standard')
        parameter = parameters[index]
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or not step.is_floating_point()
                or step.requires_grad or not bool(torch.isfinite(step)) or float(step) < 0
                or float(step) != int(step)):
            raise ValueError('Saved step must be a detached finite nonnegative integral scalar tensor')
        if group['capturable'] or group['fused']:
            if step.device != parameter.device or step.dtype != torch.float32:
                raise ValueError('Saved capturable/fused step must match native device/float32')
        elif step.device.type != 'cpu':
            raise ValueError('Saved ordinary AdamW step must use its native CPU scalar')
        for name in state_keys - {'step'}:
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or value.device != parameter.device
                    or value.layout != torch.strided or value.requires_grad
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Saved ' + name + ' shape/dtype/device/value differs from parameter')
            if name != 'exp_avg' and bool((value < 0).any()):
                raise ValueError('Saved second moments must be nonnegative')
    return group


def make_optimizer(parameters, saved_state, mode, restart_lr=RESTART_LR):
    """Return one standard AdamW bound to the supplied *same* parameter objects.

    END_SAVED keeps all saved moments and the saved end LR. RESTART_KEEP keeps
    all saved moments but uses the fixed new-stage LR 2e-4. RESTART_RESET uses
    that same LR and clears only optimizer state. Every other saved group option
    and metadata entry is copied without alteration. Absent unused state remains
    absent. This API does not enforce an experiment-specific parameter count.

    Moments must already have the supplied parameter shape, dtype and device;
    no implicit device/dtype migration or recovery of malformed state is allowed.
    No step is taken here. A caller stepping this optimizer would update its live
    parameters, so hypothetical evaluation must use the existing simulate helper
    or separate temporary parameter copies.
    """
    if mode not in MODES:
        raise ValueError('Unknown stage-state mode')
    if _scalar(restart_lr, 'restart_lr') != RESTART_LR:
        raise ValueError('The fixed restart LR is exactly 2e-4')
    _validate(parameters, saved_state)
    copied = copy.deepcopy(saved_state)
    if mode != 'END_SAVED':
        copied['param_groups'][0]['lr'] = RESTART_LR
    constructor_group = copy.deepcopy(copied['param_groups'][0])
    constructor_group['params'] = list(parameters)
    try:
        optimizer = torch.optim.AdamW([constructor_group])
        optimizer.load_state_dict(copied)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError('Saved AdamW options/state are unsupported by the installed native optimizer') from error
    if mode == 'RESTART_RESET':
        optimizer.state.clear()
    return optimizer
