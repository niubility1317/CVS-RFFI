"""Joint-gradient decomposition and groupwise reference dominance."""
import torch
from experiments.pamf_ssl_research.u_gradprotect_objective import _ExplicitGradient, parameter_group


def reference_loss(labeled, unlabeled, reference, named, *, cap):
    if cap is not None and cap != .5:
        raise ValueError('Only the fixed 0.5 cap or raw control is registered')
    params = tuple(p for _, p in named)
    gl = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
    gu = torch.autograd.grad(unlabeled, params, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(reference, params, retain_graph=True, allow_unused=True)
    combined = [None] * len(params); details = {}; stats = {}
    for group in ('t2', 'cls_head', 'other'):
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group]
        zero = torch.zeros((), device=params[0].device, dtype=torch.float64)
        l = [torch.zeros_like(params[i]) if gl[i] is None else gl[i].detach() for i in indices]
        u = [torch.zeros_like(params[i]) if gu[i] is None else gu[i].detach() for i in indices]
        ref = [torch.zeros_like(params[i]) if gr[i] is None else gr[i].detach() for i in indices]
        if any(not bool(torch.isfinite(x).all()) for x in l + u + ref):
            raise ValueError('All source gradients must be finite')
        sq = lambda v: sum((x.double().square().sum() for x in v), zero)
        dot = lambda a,b: sum(((x.double()*y.double()).sum() for x,y in zip(a,b)), zero)
        ll, uu, rr = sq(l), sq(u), sq(ref)
        lu = dot(l,u); total = [a+b for a,b in zip(l,u)]
        remainder = [a-r for a,r in zip(total,ref)]
        before = dot(ref,remainder); rest_norm = sq(remainder).sqrt()
        tiny = bool(rr <= 1e-12); conflict = bool(not tiny and before < 0)
        capped = False
        if cap is not None and not tiny:
            coefficient = before.clamp_max(0) / rr
            remainder = [a-coefficient.to(a.dtype)*r for a,r in zip(remainder,ref)]
            projected_norm = sq(remainder).sqrt()
            scale = (cap*rr.sqrt()/projected_norm).clamp_max(1.) if projected_norm > 0 else torch.ones_like(rr)
            capped = bool(scale < 1.)
            remainder = [a*scale.to(a.dtype) for a in remainder]
            total = [a+r for a,r in zip(remainder,ref)]
        for i, value in zip(indices,total):
            if gl[i] is not None or gu[i] is not None or gr[i] is not None:
                combined[i] = value
        effective_u = [a-b for a,b in zip(total,l)]
        details[group] = dict(L_norm=float(ll.sqrt()), U_norm=float(uu.sqrt()),
            applied_U_norm=float(sq(effective_u).sqrt()), dot_before=float(lu),
            dot_after=float(dot(l,effective_u)), conflict=bool(lu < 0), capped=capped, tiny_L=bool(ll <= 1e-12))
        stats.update({f'{group}_reference_norm':float(rr.sqrt()),
            f'{group}_raw_remainder_norm':float(rest_norm), f'{group}_applied_remainder_norm':float(sq(remainder).sqrt()),
            f'{group}_reference_remainder_dot_before':float(before),
            f'{group}_reference_remainder_dot_after':float(dot(ref,remainder)),
            f'{group}_zero_reference':int(tiny), f'{group}_conflict':int(conflict), f'{group}_capped':int(capped)})
    value = _ExplicitGradient.apply((labeled+unlabeled).detach(), tuple(combined), *params)
    return value, details, tuple(combined), stats
