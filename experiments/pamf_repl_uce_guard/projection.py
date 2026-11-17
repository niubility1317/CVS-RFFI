"""Apply a separately controlled base-U-CE/L soft projection per parameter group."""
import torch
from experiments.pamf_repl_uaux_soft.projection import auxiliary_loss
from experiments.pamf_ssl_research.u_gradprotect_objective import _ExplicitGradient, parameter_group

def guarded_loss(labeled, unlabeled, reference, other, named, *, alpha, beta, gamma):
    if not 0 <= gamma <= 1:
        raise ValueError('CE protection strength must be between zero and one')
    value, details, combined, stats = auxiliary_loss(labeled, unlabeled, reference, other, named,
                                                     alpha=alpha, beta=beta)
    params = tuple(p for _, p in named)
    gl = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(unlabeled-reference-other, params, retain_graph=True, allow_unused=True)
    corrected = list(combined)
    for group in ('t2', 'cls_head', 'other'):
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group]
        z = torch.zeros((), device=params[0].device, dtype=torch.float64)
        unpack = lambda x: [torch.zeros_like(params[i]) if x[i] is None else x[i].detach() for i in indices]
        l, ce, total = map(unpack, (gl, gc, combined))
        if any(not bool(torch.isfinite(x).all()) for x in l+ce+total):
            raise ValueError('Finite CE and labeled gradients required')
        dot = lambda a,b: sum(((x.double()*y.double()).sum() for x,y in zip(a,b)), z)
        ll = dot(l,l); before = dot(l,ce)
        coef = gamma*before.clamp_max(0)/ll if ll > 1e-12 else z
        changes = [-coef.to(x.dtype)*x for x in l]
        after = [x+y for x,y in zip(ce,changes)]
        total_after = [x+y for x,y in zip(total,changes)]
        effective_u = [x-y for x,y in zip(total_after,l)]
        for i, change, final in zip(indices,changes,total_after):
            if combined[i] is not None or gl[i] is not None or gc[i] is not None:
                corrected[i] = final
        details[group]['applied_U_norm'] = float(dot(effective_u,effective_u).sqrt())
        details[group]['dot_after'] = float(dot(l,effective_u))
        stats.update({f'{group}_base_ce_norm':float(dot(ce,ce).sqrt()),
            f'{group}_base_ce_l_dot_before':float(before),
            f'{group}_base_ce_l_dot_after':float(dot(l,after)),
            f'{group}_base_ce_correction_norm':float(dot(changes,changes).sqrt())})
    value = _ExplicitGradient.apply(value.detach(), tuple(corrected), *params)
    return value, details, tuple(corrected), stats
