"""Restore parent AdamW moments while retaining the fresh segment's factory.

The checkpoint has positional optimizer IDs, not parameter names.  The full
restore therefore checks the current named layout against the parent model and
the already verified common CENTER_UNITS factory; it does not invent names for
historical optimizer entries.
"""
from collections.abc import Mapping
from copy import deepcopy
import json
import math

import torch


_OPTIONS = {
    'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize',
    'capturable', 'differentiable', 'fused', 'initial_lr',
}
_REQUIRED = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}
_MOMENTS = ('exp_avg', 'exp_avg_sq')


def _number(value, name, *, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and
            (value <= minimum if strict else value < minimum)):
        raise ValueError('Invalid ' + name)
    return value


def _options(group):
    if not isinstance(group, Mapping) or 'params' not in group:
        raise ValueError('AdamW parameter group required')
    keys = set(group) - {'params'}
    if not _REQUIRED <= keys or not keys <= _OPTIONS:
        raise ValueError('Unexpected or missing AdamW group options')
    _number(group['lr'], 'lr', minimum=0)
    _number(group['eps'], 'eps', minimum=0, strict=True)
    _number(group['weight_decay'], 'weight_decay', minimum=0)
    if 'initial_lr' in group:
        _number(group['initial_lr'], 'initial_lr', minimum=0)
    betas = group['betas']
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError('Two AdamW betas required')
    for value in betas:
        if not 0 <= _number(value, 'beta') < 1:
            raise ValueError('Invalid AdamW beta')
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
        if group.get(key, False) is not False:
            raise ValueError('Unsupported AdamW option: ' + key)
    for key in ('foreach', 'fused'):
        if group.get(key) is not None and not isinstance(group[key], bool):
            raise ValueError('Invalid AdamW option: ' + key)


def _step(value, expected_steps):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError('AdamW step must be a scalar integer value')
        value = value.detach().item()
    value = _number(value, 'AdamW step', minimum=0, strict=True)
    if not value.is_integer() or (expected_steps is not None and value > expected_steps):
        raise ValueError('AdamW step outside parent update budget')
    return int(value)


def _state(value, expected_steps, parameter=None):
    if not isinstance(value, Mapping) or set(value) != {'step', *_MOMENTS}:
        raise ValueError('Only native AdamW step and two moments are allowed')
    step = _step(value['step'], expected_steps)
    first = value['exp_avg']
    for key in _MOMENTS:
        moment = value[key]
        if not isinstance(moment, torch.Tensor) or moment.layout != torch.strided or not moment.is_floating_point() or moment.requires_grad:
            raise ValueError('Detached dense floating AdamW moments required')
        if not torch.isfinite(moment).all().item():
            raise ValueError('Nonfinite AdamW moment: ' + key)
        if moment.shape != first.shape or moment.dtype != first.dtype:
            raise ValueError('AdamW moment shapes/dtypes disagree')
        if parameter is not None and (moment.shape != parameter.shape or moment.dtype != parameter.dtype):
            raise ValueError('Parent moment does not match current parameter shape/dtype')
    if (value['exp_avg_sq'] < 0).any().item():
        raise ValueError('Negative AdamW second moment')
    return step


def _summary(saved, expected_steps, *, allow_empty=False):
    if expected_steps is not None and (isinstance(expected_steps, bool) or not isinstance(expected_steps, int) or expected_steps <= 0):
        raise ValueError('Positive parent update budget required')
    if not isinstance(saved, Mapping) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('Native optimizer state_dict required')
    groups, states = saved['param_groups'], saved['state']
    if not isinstance(groups, list) or not groups or not isinstance(states, Mapping) or (not states and not allow_empty):
        raise ValueError('Nonempty parent AdamW groups and state required')
    ids = []
    for group in groups:
        _options(group)
        params = group['params']
        if not isinstance(params, list) or not params or any(type(p) is not int for p in params):
            raise ValueError('Nonempty positional AdamW parameter IDs required')
        ids.extend(params)
    if ids != list(range(len(ids))):
        raise ValueError('Parent parameter ID grouping/order differs from fresh native factory')
    if any(type(key) is not int or key not in ids for key in states):
        raise ValueError('External AdamW state entry')
    steps = {}
    norms = {key: 0.0 for key in _MOMENTS}
    counts = {key: 0 for key in _MOMENTS}
    by_id = {}
    for pid, value in states.items():
        step = _state(value, expected_steps)
        by_id[pid] = step
        steps[str(step)] = steps.get(str(step), 0) + 1
        for key in _MOMENTS:
            counts[key] += value[key].numel()
            norms[key] += value[key].detach().double().square().sum().item()
    result = dict(
        state_entries=len(states), parameter_count=len(ids),
        state_missing_parameters=len(ids) - len(states),
        state_step_min=min(by_id.values(), default=None), state_step_max=max(by_id.values(), default=None),
        state_step_histogram=dict(sorted(steps.items(), key=lambda pair: int(pair[0]))),
        exp_avg_numel=counts['exp_avg'], exp_avg_sq_numel=counts['exp_avg_sq'],
        exp_avg_l2=math.sqrt(norms['exp_avg']), exp_avg_sq_l2=math.sqrt(norms['exp_avg_sq']),
        groups=[dict(index=i, parameters=len(group['params']),
                     state_entries=sum(pid in states for pid in group['params']),
                     lr=float(group['lr']),
                     state_step_min=min((by_id[pid] for pid in group['params'] if pid in by_id), default=None),
                     state_step_max=max((by_id[pid] for pid in group['params'] if pid in by_id), default=None))
                for i, group in enumerate(groups)],
    )
    json.dumps(result, allow_nan=False)
    return result


def validate_saved(saved, expected_steps=4440):
    """Basic checkpoint-only validation; no model or data access."""
    return _summary(saved, expected_steps)


def state_summary(opt):
    """JSON-safe moment/step statistics, including the fresh RESET state."""
    return _summary(opt.state_dict(), None, allow_empty=True)


def _named_layout(opt, named_parameters, parent_model):
    if (named_parameters is None) != (parent_model is None):
        raise ValueError('Both current parameter names and parent model layout are required')
    policy = getattr(opt, '_sinc_unit_policy', None)
    if policy is not None and named_parameters is None:
        raise ValueError('CENTER_UNITS restore requires named parent/current layout')
    if named_parameters is None:
        return False, False
    items = list(named_parameters)
    if not isinstance(parent_model, Mapping) or len({name for name, _ in items}) != len(items):
        raise ValueError('Unique current named parameters and parent model required')
    items = [(name, p) for name, p in items if p.requires_grad]
    named = dict(items)
    flat = [p for group in opt.param_groups for p in group['params']]
    if {id(p) for p in flat} != {id(p) for _, p in items} or len(flat) != len(items):
        raise ValueError('Fresh optimizer must cover all jointly trainable named parameters')
    if [name for name in parent_model if name in named] != list(named):
        raise ValueError('Parent/current named parameter order differs')
    for name, p in items:
        old = parent_model.get(name)
        if not isinstance(old, torch.Tensor) or old.shape != p.shape or old.dtype != p.dtype:
            raise ValueError('Parent/current named parameter layout differs: ' + name)
    if policy is None:
        expected = [p for _, p in items]
        if [id(p) for p in flat] != [id(p) for p in expected]:
            raise ValueError('Fresh parameter order differs from current named layout')
    else:
        if policy.get('mode') != 'CENTER_UNITS' or len(opt.param_groups) != 2:
            raise ValueError('Fresh CENTER_UNITS two-group factory required')
        names, sinc = tuple(policy['names']), list(policy['sinc'])
        if len(names) != 2 or len(sinc) != 2 or any(name not in named or named[name] is not p for name, p in zip(names, sinc)):
            raise ValueError('Fresh Sinc names/order differ')
        sinc_ids = {id(p) for p in sinc}
        expected = [[p for _, p in items if id(p) not in sinc_ids], sinc]
        if any([id(p) for p in group['params']] != [id(p) for p in wanted]
               for group, wanted in zip(opt.param_groups, expected)):
            raise ValueError('Fresh body/head and Sinc group order differs')
    return True, True


def restore_optimizer(opt, saved, expected_steps=4440, *, named_parameters=None, parent_model=None):
    """Keep parent's step/m/v and every fresh group option/current pointer.

    No parent Sinc hook, clipping counter, optimizer default, or scheduler state
    is installed.  The caller still owns the new segment's cosine schedule.
    """
    result = validate_saved(saved, expected_steps)
    if opt.state:
        raise ValueError('Restore must run before the first fresh optimizer update')
    fresh = opt.state_dict()
    if len(fresh['param_groups']) != len(saved['param_groups']):
        raise ValueError('Parent/fresh optimizer group counts differ')
    parameters = []
    fresh_options = []
    for current, packed, old in zip(opt.param_groups, fresh['param_groups'], saved['param_groups']):
        _options(current)
        if packed['params'] != old['params'] or len(current['params']) != len(old['params']):
            raise ValueError('Parent/fresh optimizer group sizes or order differ')
        now = {k: v for k, v in current.items() if k != 'params'}
        previous = {k: v for k, v in old.items() if k != 'params'}
        if set(now) != set(previous) or any(now[k] != previous[k] for k in now if k != 'lr'):
            raise ValueError('Parent/fresh native AdamW options differ beyond scheduled lr')
        fresh_options.append(deepcopy(now))
        parameters.append(list(current['params']))
    flat = [p for group in parameters for p in group]
    if len({id(p) for p in flat}) != len(flat):
        raise ValueError('Duplicate fresh optimizer parameter')
    for pid, value in saved['state'].items():
        _state(value, expected_steps, flat[pid])
    named_checked, parent_checked = _named_layout(opt, named_parameters, parent_model)
    custom = {k: v for k, v in vars(opt).items() if k.startswith('_sinc_')}
    hooks = {k: tuple(v.items()) for k, v in vars(opt).items()
             if k in ('_optimizer_step_pre_hooks', '_optimizer_step_post_hooks')}
    if custom and (custom.get('_sinc_unit_steps') != 0 or custom.get('_sinc_physical_clip_calls') != 0
                   or custom.get('_sinc_unit_lr_records') != [] or custom.get('_sinc_pre_step') != {}):
        raise ValueError('Fresh segment Sinc counters must start at zero')
    defaults = deepcopy(opt.defaults)
    # AdamW may retain CPU step tensors, and same-device .to() may alias moments.
    # The parent payload remains the immutable baseline for segment telemetry.
    independent = dict(param_groups=saved['param_groups'], state={
        pid: {key: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
              for key, value in state.items()}
        for pid, state in saved['state'].items()
    })
    try:
        opt.load_state_dict(independent)
    finally:
        for group, options, params in zip(opt.param_groups, fresh_options, parameters):
            group.clear()
            group.update(options)
            group['params'] = params
    if opt.defaults != defaults or any(getattr(opt, k) is not v for k, v in custom.items()) or any(
            tuple(getattr(opt, k).items()) != v for k, v in hooks.items()):
        raise ValueError('Fresh defaults, Sinc hooks or custom attributes changed')
    if len(opt.state) != len(saved['state']):
        raise ValueError('Loaded state count differs')
    for pid, value in saved['state'].items():
        loaded = opt.state[flat[pid]]
        if set(loaded) != set(value) or _step(loaded['step'], expected_steps) != _step(value['step'], expected_steps):
            raise ValueError('Loaded per-parameter AdamW step differs')
        if any(not torch.equal(loaded[key].detach().cpu(), value[key].detach().cpu()) for key in _MOMENTS):
            raise ValueError('Loaded AdamW moments differ from parent')
        for key, original in value.items():
            current = loaded[key]
            if (isinstance(original, torch.Tensor) and isinstance(current, torch.Tensor)
                    and original.device == current.device and original.numel()
                    and original.untyped_storage().data_ptr() == current.untyped_storage().data_ptr()):
                raise ValueError('Loaded AdamW state aliases parent tensor: ' + key)
    if any({k: v for k, v in group.items() if k != 'params'} != options
           or [id(p) for p in group['params']] != [id(p) for p in params]
           for group, options, params in zip(opt.param_groups, fresh_options, parameters)):
        raise ValueError('Fresh group hyperparameters/current pointers not retained')
    result.update(policy='KEEP_PARENT_ADAMW_MOMENTS_FRESH_GROUPS',
                  all_state_tensors_equal=True, per_parameter_steps_equal=True,
                  fresh_hyperparameters_equal=True, current_parameter_pointers_retained=True,
                  custom_attributes_retained=True, named_layout_checked=named_checked,
                  parent_named_layout_checked=parent_checked, parent_state_independent=True)
    for group, options, current, old in zip(result['groups'], fresh_options, opt.param_groups, saved['param_groups']):
        group.update(fresh_lr=float(options['lr']), parent_lr=float(old['lr']), restored_lr=float(current['lr']))
        group['lr'] = float(current['lr'])
    result['segment_counters'] = {k: (len(v) if isinstance(v, (list, dict)) else v)
                                  for k, v in custom.items() if k in (
                                      '_sinc_unit_steps', '_sinc_physical_clip_calls',
                                      '_sinc_unit_lr_records', '_sinc_pre_step')}
    json.dumps(result, allow_nan=False)
    return result
