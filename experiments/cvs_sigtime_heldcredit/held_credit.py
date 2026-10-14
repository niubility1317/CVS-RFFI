"""Fresh-cache default-head credit on disjoint, previously trained source L."""
from contextlib import contextmanager
import torch
from experiments.cvs_crossrx_credit50.credit import assess as same_state_assess

CACHE_ATTRIBUTES = ('_filter_cache_key', '_filter_cache', '_coordinate_cache_key',
                    '_norm_weight_cache_key', '_norm_weight_cache')


@contextmanager
def fresh_parameter_caches(model):
    """Functional states may share _version despite having different values.

    Clear only the existing parameter-derived eval caches before each model
    call, then restore their original objects. No parameters/buffers are fit.
    """
    saved = [(m, k, getattr(m, k)) for m in model.modules()
             for k in CACHE_ATTRIBUTES if hasattr(m, k)]
    for module, key, _ in saved:
        setattr(module, key, None)
    calls = [0]
    def before_call(module, args):
        calls[0] += 1
        for owner, key, _ in saved:
            setattr(owner, key, None)
    hook = model.register_forward_pre_hook(before_call)
    try:
        yield calls
    finally:
        hook.remove()
        for module, key, value in saved:
            setattr(module, key, value)


def physical_ids(values, count):
    if len(values) != count:
        raise ValueError('One complete physical ID per packet is required')
    result = []
    for value in values:
        if (not isinstance(value, (list, tuple)) or len(value) != 5
                or any(type(v) is not int or v < 0 for v in value)):
            raise ValueError('Physical ID must be five nonnegative native integers')
        result.append(tuple(value))
    if len(set(result)) != count or len({v[-1] for v in result}) != count:
        raise ValueError('Duplicate source physical packet/base_index')
    return result


def assess(model, optimizer, base_grads, auxiliary_grads, x, y, rx, anchor_rx,
           *, default_weight_name, evaluation_ids, current_ids, mode,
           role='L_s', tolerance=1e-6):
    evaluated = physical_ids(evaluation_ids, len(x))
    current = physical_ids(current_ids, len(current_ids))
    if not current:
        raise ValueError('Current gradient cohort IDs are required')
    if mode == 'HELD_L':
        if {v[-1] for v in current} & {v[-1] for v in evaluated}:
            raise ValueError('Held-L evaluation contains a current gradient packet')
    elif mode == 'SAME_L':
        if evaluated != current:
            raise ValueError('Same-L comparator must use the exact current cohort')
    else:
        raise ValueError('Only the preregistered SAME_L/HELD_L comparison')
    with fresh_parameter_caches(model) as calls:
        result = same_state_assess(model, optimizer, base_grads, auxiliary_grads,
                                  x, y, rx, anchor_rx,
                                  default_weight_name=default_weight_name,
                                  role=role, tolerance=tolerance)
    if calls[0] != 2:
        raise RuntimeError('Each AdamW functional state must get one fresh-cache forward')
    result['summary'].update(evaluation_mode=mode,
        evaluation_packets=len(evaluated), current_gradient_packets=len(current),
        excluded_current_packets=mode == 'HELD_L',
        previously_trained_source_L=True, never_trained_holdout=False,
        fresh_parameter_cache_forwards=calls[0], private_caches_restored=True)
    return result
