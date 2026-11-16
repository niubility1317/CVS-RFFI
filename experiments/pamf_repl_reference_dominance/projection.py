"""Limit the real extra step while preserving the existing anchor constraints."""
import math
import torch
from experiments.pamf_ssl_research.branch_gradient_projection import GROUPS, ANCHORS
from experiments.pamf_ssl_research.u_gradprotect_objective import parameter_group


@torch.no_grad()
def cap_extra_to_reference(named, before, baseline, proposed, full, ce, physical, details, cap, *, inward_factors=(1.,)):
    vectors = (before, baseline, proposed, full, ce, physical)
    if (not named or not 0 < cap < 1 or any(len(v) != len(named) for v in vectors)
            or not inward_factors or inward_factors[0]!=1.
            or any(not 0 < x <= 1 for x in inward_factors)
            or any(a<=b for a,b in zip(inward_factors,inward_factors[1:]))):
        raise ValueError('Aligned actual parameters and a cap strictly between zero and one required')
    result = [v.detach().clone() for v in baseline]
    for group in GROUPS:
        indices = [i for i,(name,_) in enumerate(named) if parameter_group(name) == group]
        if not indices:
            continue
        def flatten(values):
            return torch.cat([(torch.zeros_like(named[i][1], dtype=torch.float64) if values[i] is None
                               else values[i].detach().double()).reshape(-1) for i in indices])
        pre, base, candidate = flatten(before), flatten(baseline), flatten(proposed)
        reference, extra = pre-base, base-candidate
        reference_norm, initial_norm = float(reference.norm()), float(extra.norm())
        zero_reference = reference_norm <= 1e-12
        allowed = cap*reference_norm
        scale = 0. if zero_reference else min(1., allowed/initial_norm) if initial_norm else 1.
        anchors = [flatten(v) for v in (full,ce,physical)]
        norms = [float(v.norm()) for v in anchors]
        valid = [i for i,n in enumerate(norms) if n*n > 1e-12]
        rounded=base.float();actual=torch.zeros_like(extra);actual_norm=0.;dots=[0.]*len(anchors)
        okay=False;attempts=0;used_factor=None
        for factor in inward_factors:
            attempts+=1
            candidate=(base-scale*factor*extra).float()
            displacement=base-candidate.double();norm=float(displacement.norm())
            candidate_dots=[float(v @ displacement) for v in anchors]
            if (norm <= allowed*(1+1e-6) and norm <= initial_norm*(1+1e-6)
                    and all(candidate_dots[i] >= -1e-6*max(norms[i]*norm,1e-12) for i in valid)):
                rounded,actual,actual_norm,dots=candidate,displacement,norm,candidate_dots
                okay=True;used_factor=factor;break
        rounding_fallback=not okay
        if rounding_fallback or zero_reference:
            rounded = base.float(); actual = torch.zeros_like(extra); actual_norm = 0.
            dots = [0.]*len(anchors)
        total = pre-rounded.double()
        total_norm = float(total.norm())
        cosine = float(reference @ total)/(reference_norm*total_norm) if not zero_reference and total_norm else None
        if cosine is not None and cosine < math.sqrt(1-cap*cap)-1e-6:
            raise ValueError('Actual total update does not remain near the labeled reference')
        offset = 0
        for i in indices:
            p = named[i][1]
            result[i] = rounded[offset:offset+p.numel()].view_as(p).clone()
            offset += p.numel()
        detail = details[group]
        raw_norm = detail['raw_increment_norm']
        if rounding_fallback and not detail['quantization_fallback']:
            detail['fallback_reason'] = 'reference_cap_quantization_constraint'
        detail.update(applied_increment_norm=actual_norm,
            retention=actual_norm/raw_norm if raw_norm else 1.,
            anchor_cosines_after={name: dot/(norm*actual_norm) if norm and actual_norm else None
                                  for name,dot,norm in zip(ANCHORS,dots,norms)},
            constraint_dots_after=dict(zip(ANCHORS,dots)),
            quantization_fallback=bool(detail['quantization_fallback'] or rounding_fallback),
            reference_zero_fallback=zero_reference,
            reference_rounding_fallback=rounding_fallback,
            reference_cap_repair_attempts=attempts,
            reference_cap_repair_used=bool(used_factor is not None and used_factor<1.),
            reference_cap_repair_nonzero=bool(used_factor is not None and used_factor<1. and actual_norm>0.),
            reference_cap_repair_factor=used_factor,
            reference_step_norm=reference_norm,
            reference_extra_cap=cap,
            reference_extra_ratio=actual_norm/reference_norm if reference_norm else None,
            reference_dominance_cosine=cosine,
            reference_dominance_checked=True)
    return tuple(result), details
