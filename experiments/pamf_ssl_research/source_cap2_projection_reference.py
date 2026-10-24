"""Isolated native cap2 arithmetic;never replace the frozen cap1 dependencies."""
import math
import torch
from .u_gradprotect_objective import parameter_group


@torch.no_grad()
def projected_gradients(named, labeled, unlabeled, *, max_u_ratio=1.):
    if not isinstance(max_u_ratio,(int,float)) or not math.isfinite(max_u_ratio) or max_u_ratio<=0:
        raise ValueError('Finite positive U/L norm ratio required')
    if not len(named) == len(labeled) == len(unlabeled):
        raise ValueError('Gradient/parameter lengths differ')
    combined = [None] * len(named)
    details = {}
    for group in ('t2', 'cls_head', 'other'):
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group
                   and (labeled[i] is not None or unlabeled[i] is not None)]
        if not indices:
            details[group] = dict(L_norm=0.,U_norm=0.,applied_U_norm=0.,dot_before=0.,dot_after=0.,
                                  conflict=False,capped=False,tiny_L=True)
            continue
        gl = [torch.zeros_like(named[i][1]) if labeled[i] is None else labeled[i].detach() for i in indices]
        gu = [torch.zeros_like(named[i][1]) if unlabeled[i] is None else unlabeled[i].detach() for i in indices]
        if any(not bool(torch.isfinite(g).all()) for g in gl + gu):
            raise ValueError('Source gradients must be finite')
        # Float64 reductions avoid cancellation in the group projection coefficient.
        ll = sum(g.double().square().sum() for g in gl)
        uu = sum(g.double().square().sum() for g in gu)
        dot = sum((g.double() * h.double()).sum() for g, h in zip(gl, gu))
        tiny = bool(ll <= 1e-12)
        conflict = bool(dot < 0.)
        if tiny:
            projected = [torch.zeros_like(g) for g in gu]
        else:
            coefficient = dot.clamp_max(0.) / ll
            projected = [h - coefficient.to(g.dtype) * g for g, h in zip(gl, gu)]
        norm2 = sum(g.double().square().sum() for g in projected)
        scale = torch.ones_like(ll) if bool(norm2 == 0.) else (max_u_ratio*(ll / norm2).sqrt()).clamp_max(1.)
        applied = [g * scale.to(g.dtype) for g in projected]
        for i, g, h in zip(indices, gl, applied):
            combined[i] = g + h
        details[group] = dict(L_norm=float(ll.sqrt()),U_norm=float(uu.sqrt()),
            applied_U_norm=float(norm2.sqrt() * scale),dot_before=float(dot),
            dot_after=float(sum((g.double()*h.double()).sum() for g,h in zip(gl,applied))),
            conflict=conflict,capped=bool(scale < 1.),tiny_L=tiny)
    return tuple(combined), details
