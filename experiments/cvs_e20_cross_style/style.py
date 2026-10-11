"""Training-only L MixStyle on native identity maps, with ordinary dropout.

The caller supplies only legal L_s labels and real physical IDs. Repeated batch
occurrences are allowed, but occurrence indices never replace physical IDs.
No auxiliary heads, model copy, input augmentation, or U labels are involved.
"""
from contextlib import contextmanager
import math
import numbers
import weakref

import torch
from torch import nn

from experiments.cvs_e20_budget_fishr.multilevel import weak_mixstyle_u


LEVELS = ('time_down', 't2.pool')
SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
_ACTIVE = weakref.WeakSet()


def _metadata(labels, receivers, days, ids):
    tensors = []
    for name, value, allowed in (('labels', labels, range(6)),
                                 ('receivers', receivers, SOURCE_RX),
                                 ('days', days, SOURCE_DAYS)):
        if (not isinstance(value, torch.Tensor) or value.ndim != 1 or not len(value)
                or value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError(f'L style {name} must be a nonempty integer vector')
        v = value.detach().cpu().to(torch.int64)
        if not set(v.tolist()).issubset(set(allowed)):
            raise ValueError(f'L style {name} contains unregistered source metadata')
        tensors.append(v)
    y, rx, day = tensors
    if any(len(v) != len(y) for v in tensors) or not isinstance(ids, (list, tuple)) or len(ids) != len(y):
        raise ValueError('L style metadata lengths differ')
    if any(isinstance(pid, bool) or not isinstance(pid, (str, numbers.Integral))
           or (isinstance(pid, str) and not pid.strip())
           or (isinstance(pid, numbers.Integral) and pid < 0) for pid in ids):
        raise ValueError('L style requires nonempty string or nonnegative integer physical IDs')
    pid = [p if isinstance(p, str) else int(p) for p in ids]
    seen = {}
    for p, tx, r, d in zip(pid, y.tolist(), rx.tolist(), day.tolist()):
        meta = (tx, r, d)
        if p in seen and seen[p] != meta:
            raise ValueError('One L physical ID has inconsistent TX/RX/day metadata')
        seen[p] = meta
    return y, rx, day, pid


def _paths(model):
    base = getattr(model, 'base', model)
    if not isinstance(base, nn.Module):
        raise ValueError('Missing native identity module')
    modules = dict(base.named_modules())
    candidates = [name for name in modules if 'id_backbone' in name.split('.')
                  and name.endswith('.time_down')]
    if len(candidates) != 1:
        raise ValueError('Exactly one native identity time_down hook path is required')
    prefix = candidates[0].rsplit('.', 1)[0]
    paths = {level: prefix+'.'+level for level in LEVELS}
    if any(path not in modules for path in paths.values()):
        raise ValueError('Missing native shallow/middle identity hook path')
    return modules, paths


@contextmanager
def labeled_style(model, y, rx, days, ids, step, config, ramp):
    """Yield mutable JSON-friendly details around exactly one styled L forward.

    ``config`` is the multilevel config plus ``model_seed``. ``enabled`` is the
    Fishr-head switch and is deliberately ignored. ``mixstyle_l`` and selected
    ``style_levels`` govern this context. The caller retains ordinary clean CE
    and applies its separate ``style_ce_weight * ramp`` to the returned logits.
    No module training flag is altered. CPU and model-device CUDA RNG states
    are restored after the complete extra forward, including on exceptions.
    """
    labels, receivers, day, physical_ids = _metadata(y, rx, days, ids)
    if config.get('source_role', 'L_s') != 'L_s':
        raise ValueError('L main identity style cannot consume another source role')
    selected = tuple(config.get('style_levels', ()))
    if selected != tuple(level for level in LEVELS if level in selected):
        raise ValueError('Style levels must be unique registered layers in native order')
    if isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0:
        raise ValueError('Style step must be a nonnegative integer')
    seed = config.get('model_seed')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**63:
        raise ValueError('L style requires the explicit model_seed')
    strength = float(config.get('mixstyle_strength', 1.))
    ramp = float(ramp)
    if not math.isfinite(strength+ramp) or not 0 <= strength <= 1 or not 0 <= ramp <= 1:
        raise ValueError('Invalid strength/ramp for L style')
    if config.get('mixstyle_p', .5) != .5 or config.get('mixstyle_alpha', .1) != .1:
        raise ValueError('L style requires registered p=.5 and alpha=.1')
    eta = strength*ramp
    active = bool(config.get('mixstyle_l', False) and selected and eta > 0 and model.training)
    detail = dict(role='L_main_identity', source_role='L_s', enabled=active,
                  style_input_valid=True, target_access=False, U_access=False, U_truth=False,
                  labels='legal L truth', batch_occurrences=len(labels),
                  distinct_physical_ids=len(set(physical_ids)), duplicate_occurrences=len(labels)-len(set(physical_ids)),
                  selected_levels=list(selected), actual_hook_order=[], levels={}, changed_packets=0,
                  changed_packets_semantics='sum of changed batch occurrences over selected layers',
                  p=.5, alpha=.1, strength=strength, ramp=ramp, eta=eta,
                  normal_training_dropout=True, training_flags_modified=False,
                  extra_forward_CPU_CUDA_RNG_restored=True,
                  auxiliary_heads_required=False, styled_forwards=0,
                  donor_rule='same TX/day, different source RX and real physical PID')
    if model in _ACTIVE:
        raise RuntimeError('Nested L style contexts are prohibited')
    if not active:
        detail['inactive_reason'] = ('eval_mode' if not model.training else 'no_selected_style' if not selected
                                     else 'mixstyle_l_disabled' if not config.get('mixstyle_l', False) else 'zero_eta')
    modules, paths = _paths(model) if active else ({}, {})
    devices = sorted({v.device.index for v in list(model.parameters())+list(model.buffers())
                      if v.device.type == 'cuda'})
    handles = []
    _ACTIVE.add(model)
    try:
        with torch.random.fork_rng(devices=devices):
            for level in selected if active else ():
                def hook(module, inputs, output, level=level):
                    if level in detail['levels']:
                        raise RuntimeError('Each L style context permits exactly one native forward')
                    if (not isinstance(output, torch.Tensor) or output.ndim != 3
                            or output.is_complex() or not output.is_floating_point()
                            or len(output) != len(labels)):
                        raise ValueError('Style hook must receive real [L batch,C,T] maps')
                    layer_seed = (int(seed)+int(step)*701+LEVELS.index(level)*1709) % 2**63
                    mixed, metrics = weak_mixstyle_u(output, labels, receivers, day, physical_ids,
                                                    layer_seed, p=.5, alpha=.1, eta=eta)
                    for i, j in enumerate(metrics['donors']):
                        if i != j and (labels[i] != labels[j] or receivers[i] == receivers[j]
                                or day[i] != day[j] or physical_ids[i] == physical_ids[j]):
                            raise ValueError('Illegal L style donor from native MixStyle')
                    metrics.update(path=paths[level], shape=list(output.shape), seed=layer_seed,
                                   eta=eta, role='L_main_identity', donor_rules_verified=True,
                                   input_requires_grad=output.requires_grad, output_requires_grad=mixed.requires_grad)
                    detail['levels'][level] = metrics
                    detail['actual_hook_order'].append(level)
                    detail['changed_packets'] += int(metrics['changed_packets'])
                    return mixed
                handles.append(modules[paths[level]].register_forward_hook(hook))
            yield detail
            if active:
                if tuple(detail['actual_hook_order']) != selected:
                    raise RuntimeError('Styled identity forward bypassed or reordered selected native layers')
                detail['styled_forwards'] = 1
    finally:
        for handle in handles:
            handle.remove()
        _ACTIVE.discard(model)
