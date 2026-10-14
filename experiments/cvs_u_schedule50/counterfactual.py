"""Observe two AdamW updates without modifying the training optimizer.

Gradients and returned tensors follow optimizer.param_groups order.  ``names``
are caller-defined group labels (duplicates are allowed), rather than a model
architecture convention.  Only ``summary`` is intended for JSON reports.
"""

from __future__ import annotations

import copy
import math
import random
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Sequence

import torch


# A caller that patches AdamW.step should pass its own saved original step.
_ADAMW_STEP = torch.optim.AdamW.step


def _parameters(optimizer: torch.optim.AdamW) -> list[torch.Tensor]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def _finite(value: Any, context: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{context} contains nonfinite values")
    elif isinstance(value, (float, int)) and not math.isfinite(value):
        raise ValueError(f"{context} is nonfinite")
    elif isinstance(value, dict):
        for key, member in value.items():
            _finite(member, f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, member in enumerate(value):
            _finite(member, f"{context}[{index}]")


def _gradients(
    values: Iterable[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
    branch: str,
) -> tuple[torch.Tensor | None, ...]:
    gradients = tuple(values)
    if len(gradients) != len(parameters):
        raise ValueError(
            f"{branch} gradient count {len(gradients)} != parameter count {len(parameters)}"
        )
    for index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
        if gradient is None:
            continue
        if not isinstance(gradient, torch.Tensor):
            raise TypeError(f"{branch} gradient {index} must be a Tensor or None")
        if gradient.layout != torch.strided:
            raise ValueError(f"{branch} gradient {index} must be dense and strided")
        if gradient.shape != parameter.shape:
            raise ValueError(f"{branch} gradient {index} shape differs from its parameter")
        if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            raise ValueError(f"{branch} gradient {index} device/dtype differs from its parameter")
        _finite(gradient, f"{branch} gradient {index}")
    return gradients


@contextmanager
def _preserve_rng():
    state = _rng_snapshot()
    try:
        yield state
    finally:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if state["numpy"] is not None:
            sys.modules["numpy"].random.set_state(state["numpy"])


def _rng_snapshot() -> dict[str, Any]:
    numpy = sys.modules.get("numpy")
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "numpy": numpy.random.get_state() if numpy is not None else None,
    }


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(left, numpy.ndarray):
        return isinstance(right, numpy.ndarray) and bool(numpy.array_equal(left, right))
    return bool(left == right)


def _live_snapshot(optimizer: torch.optim.AdamW) -> dict[str, Any]:
    parameters = _parameters(optimizer)
    return {
        "parameters": tuple(parameter.detach().clone() for parameter in parameters),
        "parameter_ids": tuple(id(parameter) for parameter in parameters),
        "gradients": tuple(None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters),
        "gradient_ids": tuple(id(parameter.grad) for parameter in parameters),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
    }


def _live_readback(optimizer: torch.optim.AdamW, before: dict[str, Any]) -> dict[str, bool]:
    parameters = _parameters(optimizer)
    return {
        "live_parameters_unchanged": (
            before["parameter_ids"] == tuple(id(parameter) for parameter in parameters)
            and _equal(before["parameters"], tuple(parameter.detach() for parameter in parameters))
        ),
        "live_gradients_unchanged": (
            before["gradient_ids"] == tuple(id(parameter.grad) for parameter in parameters)
            and _equal(before["gradients"], tuple(parameter.grad for parameter in parameters))
        ),
        "live_optimizer_state_unchanged": _equal(before["optimizer_state"], optimizer.state_dict()),
    }


def _norm(values: Iterable[torch.Tensor | None]) -> float:
    # Aggregate on the host so groups may contain different devices/dtypes.
    norms = [
        float(torch.linalg.vector_norm(value.detach().abs().to(torch.float64)).item())
        for value in values if value is not None
    ]
    result = math.hypot(*norms)
    if not math.isfinite(result):
        raise ValueError("a counterfactual summary norm is nonfinite")
    return result


def _step_summary(
    base_steps: Sequence[torch.Tensor],
    total_steps: Sequence[torch.Tensor],
    increments: Sequence[torch.Tensor],
) -> dict[str, float | None]:
    base_norm = _norm(base_steps)
    total_norm = _norm(total_steps)
    increment_norm = _norm(increments)
    return {
        "base_step_norm": base_norm,
        "total_step_norm": total_norm,
        "U_increment_norm": increment_norm,
        # A zero denominator is explicitly undefined, rather than NaN/Infinity.
        "U_to_total_step_ratio": increment_norm / total_norm if total_norm else None,
    }


def group_summary(
    base_steps: Sequence[torch.Tensor],
    total_steps: Sequence[torch.Tensor],
    increments: Sequence[torch.Tensor],
    labels: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate update norms for arbitrary caller-supplied parameter labels."""
    count = len(base_steps)
    if any(len(values) != count for values in (total_steps, increments, labels)):
        raise ValueError("group_summary inputs must have equal lengths")
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if not isinstance(label, str) or not label:
            raise ValueError("parameter group labels must be nonempty strings")
        groups.setdefault(label, []).append(index)
    return {
        label: {
            "parameter_count": len(indices),
            **_step_summary(
                [base_steps[index] for index in indices],
                [total_steps[index] for index in indices],
                [increments[index] for index in indices],
            ),
        }
        for label, indices in groups.items()
    }


def _branch(
    optimizer: torch.optim.AdamW,
    gradients: Sequence[torch.Tensor | None],
    max_grad_norm: float,
    step_fn: Callable[[torch.optim.AdamW], Any],
) -> tuple[tuple[torch.Tensor, ...], dict[str, float | int], tuple[torch.Tensor | None, ...]]:
    parameters = _parameters(optimizer)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_grad_norm, norm_type=2.0, error_if_nonfinite=True
    )
    factor = float((max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0).item())
    clipped_norm = _norm(parameter.grad for parameter in parameters)
    clipped_gradients = tuple(None if parameter.grad is None else parameter.grad.detach() for parameter in parameters)
    step_fn(optimizer)
    after = tuple(parameter.detach().clone() for parameter in parameters)
    for index, parameter in enumerate(after):
        _finite(parameter, f"updated parameter {index}")
    return after, {
        "grad_norm": float(total_norm.item()),
        "clipped_grad_norm": clipped_norm,
        "clip_factor": factor,
        "active_gradient_count": sum(gradient is not None for gradient in gradients),
    }, clipped_gradients


@torch.no_grad()
def simulate(
    optimizer: torch.optim.AdamW,
    base_grads: Iterable[torch.Tensor | None],
    total_grads: Iterable[torch.Tensor | None],
    *,
    names: Sequence[str] | None = None,
    max_grad_norm: float = 5.0,
    step_fn: Callable[[torch.optim.AdamW], Any] | None = None,
) -> dict[str, Any]:
    """Simulate independently clipped base/total steps from the same AdamW state.

    ``None`` skips a parameter exactly as AdamW does; an explicit zero gradient
    still advances its moments and applies weight decay.  The live parameters,
    optimizer state, gradients, and Python/NumPy/Torch RNG state are untouched.
    ``step_fn`` accepts a temporary optimizer and should be the saved original
    unbound AdamW step when the training code intercepts that method.

    Only standard dense, nondifferentiable AdamW is supported.  Tensor learning
    rates, per-group options, moments, AMSGrad, foreach/fused, and AMP optimizer
    attributes are copied by deepcopy; unsupported options fail explicitly in
    the actual AdamW call rather than falling back to a different algorithm.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("counterfactual simulate supports standard torch.optim.AdamW only")
    if not isinstance(max_grad_norm, (int, float)) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if step_fn is not None and not callable(step_fn):
        raise TypeError("step_fn must be callable")
    parameters = _parameters(optimizer)
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("duplicate optimizer parameters are unsupported")
    for group in optimizer.param_groups:
        if group.get("differentiable", False):
            raise ValueError("differentiable AdamW is unsupported for detached observation")
        _finite({key: value for key, value in group.items() if key != "params"}, "optimizer group")
    for index, parameter in enumerate(parameters):
        if parameter.layout != torch.strided:
            raise ValueError(f"parameter {index} must be dense and strided")
        _finite(parameter, f"parameter {index}")
        _finite(optimizer.state.get(parameter, {}), f"optimizer state {index}")
    base = _gradients(base_grads, parameters, "base")
    total = _gradients(total_grads, parameters, "total")
    labels = tuple(names) if names is not None else tuple(f"parameter_{i}" for i in range(len(parameters)))
    if len(labels) != len(parameters):
        raise ValueError("names count differs from optimizer parameter count")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("names must contain nonempty strings")
    original_step = step_fn if step_fn is not None else _ADAMW_STEP
    with _preserve_rng() as saved_rng:
        live_before = _live_snapshot(optimizer)
        # Clone both branches before either step.  Deepcopy preserves all group
        # settings, initialized moments, scalar step state and AMP attributes.
        base_optimizer = copy.deepcopy(optimizer)
        total_optimizer = copy.deepcopy(optimizer)
        base_after, base_clip, base_clipped = _branch(base_optimizer, base, max_grad_norm, original_step)
        total_after, total_clip, total_clipped = _branch(total_optimizer, total, max_grad_norm, original_step)
        base_steps = tuple(after - before.detach() for after, before in zip(base_after, parameters))
        total_steps = tuple(after - before.detach() for after, before in zip(total_after, parameters))
        increments = tuple(total_value - base_value for base_value, total_value in zip(base_after, total_after))
        groups = group_summary(base_steps, total_steps, increments, labels)
        for label, group in groups.items():
            indices = [index for index, candidate in enumerate(labels) if candidate == label]
            group.update({
                "base_grad_norm": _norm(base[index] for index in indices),
                "total_grad_norm": _norm(total[index] for index in indices),
                "base_clipped_grad_norm": _norm(base_clipped[index] for index in indices),
                "total_clipped_grad_norm": _norm(total_clipped[index] for index in indices),
                "base_global_clip_factor": base_clip["clip_factor"],
                "total_global_clip_factor": total_clip["clip_factor"],
            })
        summary: dict[str, Any] = {
            **_step_summary(base_steps, total_steps, increments),
            "parameter_count": len(parameters),
            "max_grad_norm": float(max_grad_norm),
            "base_grad_norm": base_clip["grad_norm"],
            "total_grad_norm": total_clip["grad_norm"],
            "base_clipped_grad_norm": base_clip["clipped_grad_norm"],
            "total_clipped_grad_norm": total_clip["clipped_grad_norm"],
            "base_clip_factor": base_clip["clip_factor"],
            "total_clip_factor": total_clip["clip_factor"],
            "base_active_gradient_count": base_clip["active_gradient_count"],
            "total_active_gradient_count": total_clip["active_gradient_count"],
            "groups": groups,
        }
        preservation = _live_readback(optimizer, live_before)
        if not all(preservation.values()):
            raise RuntimeError(f"counterfactual step modified live training state: {preservation}")
        summary.update(preservation)
    summary["rng_unchanged"] = _equal(saved_rng, _rng_snapshot())
    if not summary["rng_unchanged"]:
        raise RuntimeError("counterfactual step could not preserve all initialized RNG states")
    return {
        "base_after": base_after,
        "total_after": total_after,
        "increments": increments,
        "summary": summary,
    }
