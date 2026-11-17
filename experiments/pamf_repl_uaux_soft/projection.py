"""Project only U auxiliary gradients; preserve the L and U CE summands."""
import torch
from experiments.pamf_ssl_research.u_gradprotect_objective import _ExplicitGradient, parameter_group


def auxiliary_loss(labeled, unlabeled, reference, other, named, *, alpha, beta):
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError('Registered projection strengths must be between zero and one')
    params = tuple(p for _, p in named)
    gl, gu, gr, gh = [torch.autograd.grad(x, params, retain_graph=True, allow_unused=True)
                       for x in (labeled, unlabeled, reference, other)]
    combined = [None] * len(params); details = {}; stats = {}
    for group in ('t2', 'cls_head', 'other'):
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group]
        zero = torch.zeros((), device=params[0].device, dtype=torch.float64)
        unpack = lambda values: [torch.zeros_like(params[i]) if values[i] is None else values[i].detach() for i in indices]
        l, u, r, h = map(unpack, (gl, gu, gr, gh))
        if any(not bool(torch.isfinite(x).all()) for x in l+u+r+h):
            raise ValueError('Finite gradients required')
        dot = lambda a,b: sum(((x.double()*y.double()).sum() for x,y in zip(a,b)), zero)
        sq = lambda a: dot(a,a)
        ll, uu, rr = sq(l), sq(u), sq(r)
        rh = dot(r,h); tiny = bool(rr <= 1e-12)
        coef = alpha*rh.clamp_max(0)/rr if not tiny else zero
        hp = [x-coef.to(x.dtype)*y for x,y in zip(h,r)]
        aux = [x+y for x,y in zip(r,hp)]
        la = dot(l,aux)
        lc = beta*la.clamp_max(0)/ll if ll > 1e-12 else zero
        corrected = [a-lc.to(a.dtype)*b for a,b in zip(aux,l)]
        correction = [a-b-c for a,b,c in zip(corrected,r,h)]
        total = [a+b+c for a,b,c in zip(l,u,correction)]
        effective_u = [a-b for a,b in zip(total,l)]
        for i, value in zip(indices,total):
            if any(g[i] is not None for g in (gl,gu,gr,gh)):
                combined[i] = value
        details[group] = dict(L_norm=float(ll.sqrt()), U_norm=float(uu.sqrt()),
            applied_U_norm=float(sq(effective_u).sqrt()), dot_before=float(dot(l,u)),
            dot_after=float(dot(l,effective_u)), conflict=bool(dot(l,u)<0), capped=False, tiny_L=bool(ll<=1e-12))
        stats.update({f'{group}_reference_norm':float(rr.sqrt()), f'{group}_raw_remainder_norm':float(sq(h).sqrt()),
            f'{group}_applied_remainder_norm':float(sq([a-b for a,b in zip(corrected,r)]).sqrt()),
            f'{group}_reference_remainder_dot_before':float(rh),
            f'{group}_reference_remainder_dot_after':float(dot(r,[a-b for a,b in zip(corrected,r)])),
            f'{group}_zero_reference':int(tiny), f'{group}_conflict':int(not tiny and rh<0),
            f'{group}_capped':0, f'{group}_label_aux_dot_before':float(la),
            f'{group}_label_aux_dot_after':float(dot(l,corrected)),
            f'{group}_correction_norm':float(sq(correction).sqrt())})
    value = _ExplicitGradient.apply((labeled+unlabeled).detach(), tuple(combined), *params)
    return value, details, tuple(combined), stats
