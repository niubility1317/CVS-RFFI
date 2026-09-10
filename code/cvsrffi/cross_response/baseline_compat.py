"""Isolated July CORE90 losses; no process-global training mutation.

``legacy_losses.py`` is the byte-exact archived loss file. Later API arguments
are accepted only for disabled later mechanisms and historical radius semantics.
Callers bind ``training_bindings()`` locally for every configured row, including
U0. Ordinary SSDG runs continue using their normal loss implementations.
"""
from __future__ import annotations

import inspect
import math
from contextlib import nullcontext
from functools import wraps
from typing import Any

import torch

from . import legacy_losses as legacy


def _full_precision(function):
    """Preserve archived geometry while excluding AMP rounding at acos bounds.

    float() inputs alone are insufficient: an ambient autocast re-casts cosine
    matrix products to fp16 and rounds a clamped 0.9999 to 1 before acos. Keep
    the archived formulas byte-exact, executing their loss algebra in fp32.
    """
    @wraps(function)
    def invoke(*args, **kwargs):
        devices = []
        def promote(value):
            if torch.is_tensor(value):
                devices.append(value.device.type)
                return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
            if isinstance(value, dict):
                return {key:promote(item) for key,item in value.items()}
            if isinstance(value, tuple):
                return tuple(promote(item) for item in value)
            if isinstance(value, list):
                return [promote(item) for item in value]
            return value
        positional, named = promote(args), promote(kwargs)
        with torch.autocast(device_type=devices[0], enabled=False) if devices else nullcontext():
            return function(*positional, **named)
    return invoke

_PROXY_ZERO = {
    'bridge_accept_weight', 'shell_outward_accept_weight', 'low_density_accept_weight',
    'energy_margin_quantile_weight', 'radius_budget_weight', 'radius_inter_ratio_weight',
}
_PROXY_INERT = {
    'component_radius_quantile', 'accept_softplus_temperature', 'bridge_accept_target',
    'shell_outward_accept_target', 'tail_accept_target', 'overflow_accept_target',
    'energy_margin_q', 'energy_margin_target', 'radius_budget_rad', 'radius_max_budget_rad',
    'radius_inter_ratio_target', 'density_temperature_rad',
}
_SOURCE_ZERO = {
    'local_component_compact_weight', 'local_component_invariant_weight',
    'local_component_inter_weight', 'local_component_overlap_weight',
    'local_component_accept_weight', 'local_component_density_weight', 'leave_domain_target_weight',
}
_SOURCE_INERT = {
    'core_quantile', 'local_component_inter_margin_rad', 'local_component_center_target_rad',
    'local_component_overlap_margin_rad', 'local_component_min_samples',
    'local_component_radius_floor_rad', 'local_component_density_beta',
    'local_component_density_cap', 'local_component_term_cap',
    'leave_domain_target_rad', 'structural_cvar_alpha',
}


def _historical_kwargs(function, kwargs, *, zero, inert, mode_key):
    kwargs = dict(kwargs)
    mode = kwargs.pop(mode_key, 'three_sigma')
    if mode not in {'three_sigma', 'legacy_three_sigma'}:
        raise ValueError(f'historical CORE90 requires {mode_key}=three_sigma')
    for key in zero:
        if float(kwargs.pop(key, 0.0)) != 0.0:
            raise ValueError(f'later mechanism {key} is incompatible with historical CORE90')
    for key in inert:
        if key in kwargs and not math.isfinite(float(kwargs.pop(key))):
            raise ValueError(f'{key} must be finite even when inactive')
    unknown = set(kwargs) - set(inspect.signature(function).parameters)
    if unknown:
        raise TypeError(f'unsupported historical CORE90 arguments: {sorted(unknown)}')
    return kwargs


@_full_precision
def proxy_unknown_energy_loss(*args, **kwargs):
    return legacy.proxy_unknown_energy_loss(*args, **_historical_kwargs(
        legacy.proxy_unknown_energy_loss, kwargs, zero=_PROXY_ZERO, inert=_PROXY_INERT,
        mode_key='component_radius_mode'))


@_full_precision
def source_episode_three_sigma_loss(*args, **kwargs):
    return legacy.source_episode_three_sigma_loss(*args, **_historical_kwargs(
        legacy.source_episode_three_sigma_loss, kwargs, zero=_SOURCE_ZERO, inert=_SOURCE_INERT,
        mode_key='radius_mode'))


class PrototypeMemoryBank(legacy.PrototypeMemoryBank):
    """Keep archived gradient semantics; add only explicit state serialization."""
    _STATE_KEYS = ('class_proto', 'domain_proto', 'class_count', 'domain_count')

    loss = _full_precision(legacy.PrototypeMemoryBank.loss)
    update = _full_precision(legacy.PrototypeMemoryBank.update)

    def state_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key).detach().clone() if torch.is_tensor(getattr(self, key))
                else None for key in self._STATE_KEYS}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != set(self._STATE_KEYS):
            raise ValueError('incomplete historical prototype state')
        for key in self._STATE_KEYS:
            value = state[key]
            if value is not None and not torch.is_tensor(value):
                raise TypeError('prototype state entries must be tensors or None')
            setattr(self, key, value.detach().clone() if torch.is_tensor(value) else None)


def training_bindings() -> dict[str, Any]:
    names = ('compute_core_losses', 'fishr_logit_gradient_variance_loss',
             'make_soft_unknown_mixup', 'open_world_feature_space_loss',
             'one_way_kl_from_teacher', 'sanitize_loss', 'soft_unknown_mixup_loss',
             'zid_compactness_loss')
    result = {name: _full_precision(getattr(legacy, name)) for name in names}
    result.update(PrototypeMemoryBank=PrototypeMemoryBank,
                  proxy_unknown_energy_loss=proxy_unknown_energy_loss,
                  source_episode_three_sigma_loss=source_episode_three_sigma_loss)
    return result
