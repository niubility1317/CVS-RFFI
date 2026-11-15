"""Project selected U loss components before applying the original joint guard."""
import torch
from experiments.pamf_ssl_research.u_gradprotect_objective import (
    _ExplicitGradient, parameter_group, projected_gradients,
)


def protected_components(labeled, unlabeled, named, parts, *, max_u_ratio):
    params = tuple(p for _, p in named)
    gl = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
    gu = torch.autograd.grad(unlabeled, params, retain_graph=True, allow_unused=True)
    raw = tuple(None if g is None else g.detach() for g in gu)
    modified = [None if g is None else g.detach().clone() for g in gu]
    component_stats = {}
    for part, loss in parts.items():
        if not loss.requires_grad:
            raise ValueError('Registered component must retain its gradient graph')
        gc = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        for group in ('t2', 'cls_head', 'other'):
            indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group]
            ll = sum((gl[i].detach().double().square().sum() for i in indices if gl[i] is not None),
                     torch.zeros((), device=params[0].device, dtype=torch.float64))
            dot = sum(((gl[i].detach().double() * gc[i].detach().double()).sum()
                       for i in indices if gl[i] is not None and gc[i] is not None),
                      torch.zeros_like(ll))
            if not bool(torch.isfinite(ll) & torch.isfinite(dot)):
                raise ValueError('Nonfinite component gradient')
            conflict = bool(dot < 0 and ll > 1e-12)
            coefficient = dot.clamp_max(0) / ll if bool(ll > 1e-12) else torch.zeros_like(ll)
            for i in indices:
                if gl[i] is not None and conflict:
                    correction = -coefficient.to(gl[i].dtype) * gl[i].detach()
                    modified[i] = correction if modified[i] is None else modified[i] + correction
            component_stats[f'{part}_{group}_conflict'] = int(conflict)
            component_stats[f'{part}_{group}_dot_before'] = float(dot)
            component_stats[f'{part}_{group}_correction_norm'] = float((-coefficient * ll.sqrt()))
    gradients, details = projected_gradients(named, gl, modified, max_u_ratio=max_u_ratio)
    # Separate the raw total from the modified input to the original guard.
    _, raw_details = projected_gradients(named, gl, raw, max_u_ratio=max_u_ratio)
    for group in details:
        for key in ('U_norm', 'dot_before', 'conflict'):
            details[group]['raw_'+key] = raw_details[group][key]
    value = _ExplicitGradient.apply((labeled + unlabeled).detach(), gradients, *params)
    return value, details, gradients, component_stats
