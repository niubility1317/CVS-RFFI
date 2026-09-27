"""Local JVP/VJP diagnostics of an explicitly signed, ordinary autograd field.

Never pass the gradient of a GRL scalar surrogate as ``field``. Its custom
backward would reapply reversal during higher derivatives. The field must
include every loss claimed by the audit and use create_graph=True internally.
"""
from dataclasses import dataclass

import torch


@dataclass
class JacobianAudit:
    rotation_ratio: float
    jvp_norm: float
    vjp_norm: float
    parameter_norm: float
    direction_seed: int
    valid: bool
    jvp: torch.Tensor
    vjp: torch.Tensor
    finite_difference_relative_error: float | None = None


def local_jacobian_audit(field, point, *, direction=None, seed=0, eps=1e-12,
                         finite_difference_step=None, field_kind="explicit_signed"):
    """Audit a selected flat parameter block without constructing its matrix.

    ``field(point)`` returns a same-shaped vector with ordinary differentiable
    operations. Unsupported higher-order operators raise; no term is discarded.
    Parameter scaling is caller-defined and is reported through parameter_norm.
    """
    if field_kind != "explicit_signed":
        raise ValueError("Jacobian requires an explicit signed field, never a GRL Hessian")
    if point.ndim != 1 or not point.is_floating_point():
        raise ValueError("point must be a floating flat tensor")
    point = point.detach().requires_grad_(True)
    if direction is None:
        generator = torch.Generator(device=point.device).manual_seed(seed)
        direction = torch.randn(point.shape, dtype=point.dtype, device=point.device, generator=generator)
    if direction.shape != point.shape or not torch.isfinite(direction).all():
        raise ValueError("direction must be finite and match point")
    direction = direction.to(point)
    if direction.norm() <= eps:
        raise ValueError("direction cannot be zero")
    direction = direction / direction.norm()
    value, jvp = torch.autograd.functional.jvp(field, point, direction, strict=False)
    if value.shape != point.shape:
        raise ValueError("field output must match the selected parameter vector")
    _, vjp = torch.autograd.functional.vjp(field, point, direction, strict=False)
    jn, vn = float(jvp.norm()), float(vjp.norm())
    valid = bool(torch.isfinite(value).all() and torch.isfinite(jvp).all() and torch.isfinite(vjp).all() and jn + vn > eps)
    ratio = float((jvp - vjp).norm() / (jvp.norm() + vjp.norm() + eps)) if valid else float("nan")
    error = None
    if finite_difference_step is not None:
        if finite_difference_step <= 0:
            raise ValueError("finite_difference_step must be positive")
        h = finite_difference_step
        difference = (field(point + h * direction) - field(point - h * direction)) / (2 * h)
        error = float(((difference - jvp).norm() / (jvp.norm() + eps)).detach())
    return JacobianAudit(ratio, jn, vn, float(point.detach().norm()), seed, valid,
                         jvp.detach(), vjp.detach(), error)


jacobian_audit = local_jacobian_audit
