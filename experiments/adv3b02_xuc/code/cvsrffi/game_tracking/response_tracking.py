"""Bounded implicit head response with explicit curvature and fallback status."""
from dataclasses import dataclass

import torch


@dataclass
class ResponseResult:
    delta: torch.Tensor | None
    accepted: bool
    reason: str
    iterations: int
    residual_norm: float
    damping: float
    negative_curvature: bool = False
    fallback: str | None = None


def damped_cg(operator, rhs, *, damping, max_iterations=50, tolerance=1e-6,
              positive_definite=False):
    """Solve (H+damping I)x=rhs, conditional on a justified SPD contract.

    ``positive_definite=True`` must follow a convex regularized head objective
    or an independently justified positive-definite local operator; CG itself
    does not prove global definiteness. A failed solve returns no compensation.
    The caller may retry a preregistered larger damping or ordinary head steps.
    """
    if damping < 0 or max_iterations < 1 or tolerance <= 0 or rhs.ndim != 1:
        raise ValueError("Invalid damping, iteration budget, tolerance or rhs")
    fallback = "ordinary_finite_head_update"
    if not positive_definite:
        return ResponseResult(None, False, "positive_definiteness_unverified", 0, float("nan"), damping, fallback=fallback)
    if not torch.isfinite(rhs).all():
        return ResponseResult(None, False, "nonfinite_rhs", 0, float("nan"), damping, fallback=fallback)
    solution = torch.zeros_like(rhs)
    residual, direction = rhs.detach().clone(), rhs.detach().clone()
    squared = residual.dot(residual)
    threshold = tolerance * max(1., float(rhs.norm()))
    if float(residual.norm()) <= threshold:
        return ResponseResult(solution, True, "converged", 0, float(residual.norm()), damping)
    for iteration in range(1, max_iterations + 1):
        product = operator(direction).detach() + damping * direction
        curvature = direction.dot(product)
        if not torch.isfinite(product).all() or not torch.isfinite(curvature):
            return ResponseResult(None, False, "nonfinite_operator", iteration, float(residual.norm()), damping, fallback=fallback)
        if curvature <= 0:
            return ResponseResult(None, False, "nonpositive_curvature", iteration, float(residual.norm()), damping, True, fallback)
        alpha = squared / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * product
        new_squared = residual.dot(residual)
        if float(residual.norm()) <= threshold:
            # Verify the true residual, not just the recursively updated one.
            actual = rhs - operator(solution).detach() - damping * solution
            rn = float(actual.norm())
            if torch.isfinite(actual).all() and rn <= threshold:
                return ResponseResult(solution.detach(), True, "converged", iteration, rn, damping)
            return ResponseResult(None, False, "residual_verification_failed", iteration, rn, damping, fallback=fallback)
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
    return ResponseResult(None, False, "iteration_budget_exhausted", max_iterations,
                          float(residual.norm()), damping, fallback=fallback)


def implicit_head_response(loss, head_parameters, encoder_parameters, delta_theta, *,
                           damping, positive_definite=False, max_iterations=50, tolerance=1e-6):
    """Compute -H_phi_theta delta_theta using a plain, non-GRL head CE graph.

    ``delta_theta`` is the actual encoder/shared-parameter displacement. The
    returned compensation is a candidate only; the caller must validate it on
    legal monitor data before committing. No parameters are changed here.
    """
    heads, encoders, displacements = tuple(head_parameters), tuple(encoder_parameters), tuple(delta_theta)
    if not heads or len(encoders) != len(displacements):
        raise ValueError("Head block must be nonempty and encoder displacements must match")
    if len({id(p) for p in heads + encoders}) != len(heads + encoders):
        raise ValueError("Head and encoder/shared blocks must have unique disjoint ownership")
    if any(p.shape != d.shape for p, d in zip(encoders, displacements)):
        raise ValueError("Encoder displacement shapes do not match")
    if not loss.requires_grad or loss.numel() != 1:
        raise ValueError("Head loss must be a differentiable scalar without GRL")

    def derivatives(value, params, *, create_graph):
        if not value.requires_grad:
            return tuple(torch.zeros_like(p) for p in params)
        gradients = torch.autograd.grad(value, params, retain_graph=True, create_graph=create_graph, allow_unused=True)
        return tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients))

    def flatten(values):
        return torch.cat([value.reshape(-1) for value in values])

    encoder_gradient = derivatives(loss, encoders, create_graph=True)
    cross_scalar = sum((g * delta.detach()).sum() for g, delta in zip(encoder_gradient, displacements))
    if not isinstance(cross_scalar, torch.Tensor):
        cross_scalar = loss * 0
    rhs = -flatten(derivatives(cross_scalar, heads, create_graph=False)).detach()
    head_gradient = derivatives(loss, heads, create_graph=True)

    def operator(vector):
        return flatten(derivatives((flatten(head_gradient) * vector).sum(), heads, create_graph=False))

    return damped_cg(operator, rhs, damping=damping, max_iterations=max_iterations,
                     tolerance=tolerance, positive_definite=positive_definite)


solve_head_response = implicit_head_response
