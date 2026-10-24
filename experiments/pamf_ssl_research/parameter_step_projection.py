"""Protect the actual quantized parameter increment relative to an L-only step."""
from itertools import combinations
import math
import torch
from .branch_gradient_projection import GROUPS, ANCHORS
from .u_gradprotect_objective import parameter_group


def affine_candidates(basis, coordinates, margins):
    """Closest small-basis point satisfying up to three affine half-spaces."""
    basis = torch.as_tensor(basis, dtype=torch.float64, device='cpu')
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device='cpu')
    margins = torch.as_tensor(margins, dtype=torch.float64, device='cpu')
    count, width = basis.shape
    if (not 1 <= count <= 3 or not 1 <= width <= 3 or coordinates.shape != (width,)
            or margins.shape != (count,) or not bool(torch.isfinite(basis).all())
            or not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(margins).all())
            or bool((margins < 0).any())):
        raise ValueError('Finite at-most-three affine constraints with nonnegative margins required')
    dots = basis @ coordinates
    tolerance = 1e-8 * max(float(dots.abs().max()), float(margins.abs().max()), 1e-12)
    result = []
    for size in range(count + 1):
        for selected in combinations(range(count), size):
            projected = coordinates.clone()
            if size:
                indices = torch.tensor(selected, dtype=torch.long)
                active = basis[indices]
                correction = torch.linalg.pinv(active, rtol=1e-10) @ (margins[indices] - dots[indices])
                multipliers = torch.linalg.pinv(active.T, rtol=1e-10) @ correction
                projected += correction
                if bool((multipliers < -tolerance).any()):
                    continue
                if bool(((active @ projected - margins[indices]).abs() > tolerance).any()):
                    continue
                if not torch.allclose(active.T @ multipliers, correction, rtol=1e-8, atol=tolerance):
                    continue
            if bool((basis @ projected - margins < -tolerance).any()):
                continue
            result.append(dict(coordinates=projected.tolist(), active_set=list(selected),
                               correction_squared_norm=float((projected - coordinates).square().sum())))
    return sorted(result, key=lambda item: item['correction_squared_norm'])


@torch.no_grad()
def project_parameters(named, baseline, joint, full, clean, physical):
    """Return new parameter values; never write the supplied model or tensors."""
    vectors = (baseline, joint, full, clean, physical)
    if not named or any(len(values) != len(named) for values in vectors):
        raise ValueError('Complete aligned source parameters and three anchor vectors required')
    for values in vectors:
        for (_, parameter), value in zip(named, values):
            if parameter.dtype != torch.float32:
                raise ValueError('Actual FP32 parameter values are required')
            if value is not None and (value.shape != parameter.shape or value.dtype != parameter.dtype
                                      or value.device != parameter.device or not bool(torch.isfinite(value).all())):
                raise ValueError('Finite aligned FP32 parameter/gradient tensors required')
    if any(value is None for values in (baseline, joint) for value in values):
        raise ValueError('Actual dense baseline and joint parameter values required')
    result = [value.detach().clone() for value in baseline]
    details = {}
    for group in GROUPS:
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group]
        if not indices:
            details[group] = dict(raw_increment_norm=0., applied_increment_norm=0., retention=1.,
                                  quantization_fallback=False, tiny_L=True, passthrough=True,
                                  actual_parameter_check=True, norm_nonexpansion=True,
                                  anchor_cosines_before={a: None for a in ANCHORS},
                                  anchor_cosines_after={a: None for a in ANCHORS},
                                  constraint_dots_after={a: 0. for a in ANCHORS}, quantization_attempts_used=0)
            continue
        def flatten(values):
            return torch.cat([(torch.zeros_like(named[i][1], dtype=torch.float64) if values[i] is None
                               else values[i].detach().double()).reshape(-1) for i in indices])
        base, target = flatten(baseline), flatten(joint)
        raw = base - target
        anchors = [flatten(values) for values in (full, clean, physical)]
        norms = [float(a.norm()) for a in anchors]
        raw_norm = float(raw.norm())
        valid = [i for i, norm in enumerate(norms) if norm * norm > 1e-12]
        before = [float(a @ raw) for a in anchors]
        tiny = norms[0] * norms[0] <= 1e-12
        def acceptable(delta):
            norm = float(delta.norm())
            dots = [float(a @ delta) for a in anchors]
            return (all(dots[i] >= -1e-6 * max(norms[i] * norm, 1e-12) for i in valid)
                    and norm <= raw_norm * (1 + 1e-6) + 1e-12), norm, dots
        feasible, _, _ = acceptable(raw)
        passthrough = feasible and not tiny
        fallback = False
        attempts = 0
        active = []
        if tiny:
            parameters, delta = base.float(), torch.zeros_like(raw)
        elif passthrough:
            parameters, delta = target.float(), raw
        else:
            units = [anchors[i] / norms[i] for i in valid]
            matrix = torch.stack(units, dim=1)
            q, r = torch.linalg.qr(matrix, mode='reduced')
            coordinates = q.T @ raw
            perpendicular = raw - q @ coordinates
            margins = torch.zeros(len(valid), dtype=torch.float64)
            chosen = None
            for attempt in range(4):
                attempts = attempt + 1
                candidates = affine_candidates(r.T.cpu(), coordinates.cpu(), margins)
                if not candidates:
                    break
                repair = None
                for candidate in candidates:
                    continuous = perpendicular + q @ torch.tensor(candidate['coordinates'], dtype=q.dtype, device=q.device)
                    rounded = (base - continuous).float()
                    actual = base - rounded.double()
                    okay, _, _ = acceptable(actual)
                    if okay:
                        chosen = (rounded, actual, candidate)
                        break
                    if repair is None:
                        rounding = actual - continuous
                        repair = torch.tensor([2 * float((unit * rounding).abs().sum())
                                               + 1e-12 * max(raw_norm, 1e-12) for unit in units], dtype=torch.float64)
                if chosen is not None:
                    break
                margins = repair
            if chosen is None:
                parameters, delta = base.float(), torch.zeros_like(raw)
                fallback = True
            else:
                parameters, delta, solution = chosen
                active = [ANCHORS[valid[i]] for i in solution['active_set']]
        okay, norm, after = acceptable(delta)
        if not okay:
            raise ValueError('The actual quantized source parameter constraints failed')
        offset = 0
        for index in indices:
            parameter = named[index][1]
            result[index] = parameters[offset:offset + parameter.numel()].view_as(parameter).clone()
            offset += parameter.numel()
        details[group] = dict(raw_increment_norm=raw_norm, applied_increment_norm=norm,
                              retention=norm / raw_norm if raw_norm else 1.,
                              quantization_fallback=fallback, fallback_reason='fixed_quantization_attempts_exhausted' if fallback else None,
                              tiny_L=tiny, passthrough=passthrough, quantization_attempts_used=attempts,
                              actual_parameter_check=True, norm_nonexpansion=True, active_set=active,
                              solver='thin_QR_active_set_SVD_affine_quantization_repair',
                              anchor_cosines_before={name: dot / (a_norm * raw_norm) if a_norm and raw_norm else None
                                                     for name, dot, a_norm in zip(ANCHORS, before, norms)},
                              anchor_cosines_after={name: dot / (a_norm * norm) if a_norm and norm else None
                                                    for name, dot, a_norm in zip(ANCHORS, after, norms)},
                              constraint_dots_after=dict(zip(ANCHORS, after)),
                              near_zero_component_anchors_omitted=[ANCHORS[i] for i in range(3) if i not in valid])
    return tuple(result), details
