"""First-order, adapter-only inner loop for the V1 meta-adapter route.

The inner loop is deliberately functional: it replaces only the Task3 adapter
allowlist, never creates an optimizer, and keeps the original module state
available for the outer query loss.  A detached support gradient gives the
first-order approximation while the updated value still depends on the
initial adapter tensor and its module-level Meta-SGD step size.
"""

from __future__ import annotations

import inspect
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .meta_adapter import adapter_step_size_by_parameter, iter_inner_adapter_parameters

try:
    from torch.func import functional_call as _functional_call
except ImportError:  # pragma: no cover - retained for older supported torch builds.
    from torch.nn.utils.stateless import functional_call as _stateless_functional_call

    def _functional_call(module, parameter_and_buffer_values, *, args, kwargs, strict=False):
        del strict
        return _stateless_functional_call(
            module,
            parameter_and_buffer_values,
            args=args,
            kwargs=kwargs,
        )


class MetaInnerLoopError(RuntimeError):
    """Raised when a functional adapter update cannot be proven valid."""


class _ReadOnlyOrderedDict(OrderedDict):
    """Ordered mapping exposed by ``FastAdapterState`` without mutators."""

    _ERROR = "FastAdapterState.parameters is read-only"

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self)
        source = OrderedDict(*args, **kwargs)
        for key, value in source.items():
            OrderedDict.__setitem__(self, key, value)

    def __setitem__(self, key, value):
        raise TypeError(self._ERROR)

    def __delitem__(self, key):
        raise TypeError(self._ERROR)

    def clear(self):
        raise TypeError(self._ERROR)

    def pop(self, key, default=None):
        raise TypeError(self._ERROR)

    def popitem(self, last=True):
        raise TypeError(self._ERROR)

    def setdefault(self, key, default=None):
        raise TypeError(self._ERROR)

    def update(self, *args, **kwargs):
        raise TypeError(self._ERROR)

    def move_to_end(self, key, last=True):
        raise TypeError(self._ERROR)

    def __ior__(self, other):
        raise TypeError(self._ERROR)


@dataclass(frozen=True)
class FastAdapterState:
    """Ordered fast adapter tensors and detached support-loss history."""

    parameters: Mapping[str, Tensor]
    steps: int
    support_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError("FastAdapterState.steps must be an integer")
        if self.steps < 0:
            raise ValueError("FastAdapterState.steps must be non-negative")
        if not isinstance(self.parameters, OrderedDict):
            raise TypeError("FastAdapterState.parameters must be an OrderedDict")
        if not self.parameters:
            raise ValueError("FastAdapterState.parameters must be non-empty")
        if not isinstance(self.support_losses, tuple):
            raise TypeError("FastAdapterState.support_losses must be a tuple")
        if len(self.support_losses) != self.steps:
            raise ValueError("FastAdapterState.support_losses length must equal steps")

        normalized = OrderedDict()
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("FastAdapterState parameter names must be non-empty strings")
            if not torch.is_tensor(value):
                raise TypeError(f"FastAdapterState.parameters[{name!r}] must be a tensor")
            if not value.is_floating_point():
                raise ValueError(f"FastAdapterState.parameters[{name!r}] must be floating-point")
            normalized[name] = value

        losses = []
        for loss in self.support_losses:
            if isinstance(loss, bool) or not isinstance(loss, (float, int)):
                raise TypeError("FastAdapterState.support_losses must contain Python numbers")
            loss = float(loss)
            if not math.isfinite(loss):
                raise ValueError("FastAdapterState.support_losses must be finite")
            losses.append(loss)

        object.__setattr__(self, "parameters", _ReadOnlyOrderedDict(normalized))
        object.__setattr__(self, "support_losses", tuple(losses))


def _is_finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _forward_kwargs(model: nn.Module, y: Tensor | None) -> dict[str, object]:
    """Choose the real label keyword without retrying arbitrary TypeErrors."""

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as exc:
        raise MetaInnerLoopError("cannot inspect model.forward signature") from exc

    if "return_aux" not in parameters:
        raise MetaInnerLoopError(
            "model.forward must expose return_aux=True for functional meta adaptation"
        )

    label_names = [name for name in ("y", "y_tx") if name in parameters]
    if len(label_names) > 1:
        raise MetaInnerLoopError(
            "model.forward exposes ambiguous label arguments; expected y or y_tx"
        )
    if y is not None and not label_names:
        raise MetaInnerLoopError("model.forward exposes neither y nor y_tx for support labels")

    kwargs: dict[str, object] = {"return_aux": True}
    if label_names:
        kwargs[label_names[0]] = y
    return kwargs


def _validate_fast_parameters(
    model: nn.Module,
    values: Mapping[str, Tensor],
) -> OrderedDict[str, Tensor]:
    if not isinstance(values, Mapping):
        raise MetaInnerLoopError("fast adapter parameters must be a mapping")
    if not values:
        raise MetaInnerLoopError("fast adapter parameters must be non-empty")

    expected = OrderedDict(iter_inner_adapter_parameters(model))
    if not expected:
        raise MetaInnerLoopError("model has no Task3 inner adapter parameters")

    actual_keys = tuple(values.keys())
    if any(not isinstance(key, str) for key in actual_keys):
        raise MetaInnerLoopError("fast adapter parameter keys must be strings")
    expected_keys = tuple(expected.keys())
    if set(actual_keys) != set(expected_keys):
        missing = tuple(key for key in expected_keys if key not in values)
        extra = tuple(key for key in actual_keys if key not in expected)
        raise MetaInnerLoopError(
            f"fast adapter parameter key set mismatch: missing={missing}, extra={extra}"
        )

    normalized = OrderedDict()
    for name, reference in expected.items():
        value = values[name]
        if not torch.is_tensor(value):
            raise MetaInnerLoopError(f"fast parameter {name!r} must be a tensor")
        if value.shape != reference.shape:
            raise MetaInnerLoopError(
                f"fast parameter {name!r} shape must match {tuple(reference.shape)}"
            )
        if value.dtype != reference.dtype or value.device != reference.device:
            raise MetaInnerLoopError(
                f"fast parameter {name!r} dtype/device must match the model parameter"
            )
        if not value.is_floating_point():
            raise MetaInnerLoopError(f"fast parameter {name!r} must be floating-point")
        if not _is_finite(value):
            raise MetaInnerLoopError(f"fast parameter {name!r} must be finite")
        normalized[name] = value
    return normalized


def _clone_inner_parameters(model: nn.Module) -> OrderedDict[str, Tensor]:
    """Clone adapter leaves while retaining the autograd path to initialization."""

    fast = OrderedDict()
    for name, parameter in iter_inner_adapter_parameters(model):
        # ``clone`` is intentionally not detached: outer query gradients must
        # flow through the clone back to the model's initialization parameter,
        # while the clone's storage remains independent from that parameter.
        fast[name] = parameter.clone()
    return fast


def _snapshot_module_state(model: nn.Module):
    modules = tuple(model.modules())
    parameters = tuple(
        (name, parameter, parameter.detach().clone())
        for name, parameter in model.named_parameters()
    )
    buffers = tuple(
        (name, buffer, buffer.detach().clone())
        for name, buffer in model.named_buffers()
        if buffer is not None
    )
    return modules, parameters, buffers, tuple(module.training for module in modules)


def _restore_module_state(snapshot) -> None:
    modules, parameters, buffers, training = snapshot
    with torch.no_grad():
        for _, parameter, before in parameters:
            if not torch.equal(parameter.detach(), before):
                parameter.copy_(before)
        for _, buffer, before in buffers:
            if not torch.equal(buffer.detach(), before):
                buffer.copy_(before)
    for module, was_training in zip(modules, training):
        module.training = was_training


def _functional_forward_from_mapping(
    model: nn.Module,
    fast_values: Mapping[str, Tensor],
    x: Tensor,
    y: Tensor | None = None,
) -> Mapping[str, object]:
    """Run the actual model with a validated adapter-only fast state."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not torch.is_tensor(x):
        raise TypeError("x must be a tensor")
    if y is not None and not torch.is_tensor(y):
        raise TypeError("y must be a tensor or None")

    fast = _validate_fast_parameters(model, fast_values)
    kwargs = _forward_kwargs(model, y)
    snapshot = _snapshot_module_state(model)

    # Supplying cloned buffers keeps running statistics and other mutable
    # module state outside the functional call, while the strict allowlist
    # above still limits replacement parameters to Task3 adapter keys.
    parameter_and_buffer_values: OrderedDict[str, Tensor] = OrderedDict(fast)
    for name, buffer in model.named_buffers():
        if buffer is not None:
            parameter_and_buffer_values[name] = buffer.detach().clone()

    try:
        output = _functional_call(
            model,
            parameter_and_buffer_values,
            args=(x,),
            kwargs=kwargs,
            strict=False,
        )
        if not isinstance(output, Mapping):
            raise MetaInnerLoopError(
                "model.forward(return_aux=True) must return a mapping for meta objectives"
            )
        return output
    finally:
        _restore_module_state(snapshot)


def functional_forward(
    model: nn.Module,
    fast_state: FastAdapterState,
    x: Tensor,
    y: Tensor | None = None,
) -> Mapping[str, object]:
    """Run the actual model with a validated, immutable fast-state carrier."""

    if not isinstance(fast_state, FastAdapterState):
        raise TypeError("functional_forward fast_state must be a FastAdapterState")
    return _functional_forward_from_mapping(model, fast_state.parameters, x, y)


def _validate_support_loss(loss: object) -> Tensor:
    if not torch.is_tensor(loss):
        raise MetaInnerLoopError("support loss must be a scalar floating tensor")
    if loss.ndim != 0:
        raise MetaInnerLoopError("support loss must be a scalar floating tensor")
    if not loss.is_floating_point():
        raise MetaInnerLoopError("support loss must be a scalar floating tensor")
    if not loss.requires_grad:
        raise MetaInnerLoopError("support loss must require gradients")
    if not _is_finite(loss):
        raise MetaInnerLoopError("support loss must be finite")
    return loss


def first_order_adapt(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    support_loss_fn: Callable[[Mapping[str, object], Tensor, Mapping[str, Tensor]], Tensor],
    steps: int,
) -> FastAdapterState:
    """Perform fixed-step FOMAML updates over only Task3 adapter parameters."""

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("V1 source meta inner steps must be in [0, 10]")
    if steps < 0 or steps > 10:
        raise ValueError("V1 source meta inner steps must be in [0, 10]")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError("x and y must be tensors")
    if not callable(support_loss_fn):
        raise TypeError("support_loss_fn must be callable")

    fast = _clone_inner_parameters(model)
    if not fast:
        raise MetaInnerLoopError("model has no Task3 inner adapter parameters")
    step_sizes = dict(adapter_step_size_by_parameter(model))
    if tuple(step_sizes) != tuple(fast):
        raise MetaInnerLoopError("Task3 adapter step-size mapping does not match inner keys")

    history: list[float] = []
    for step_index in range(steps):
        outputs = _functional_forward_from_mapping(model, fast, x, y)
        try:
            loss = support_loss_fn(outputs, y, _ReadOnlyOrderedDict(fast))
        except MetaInnerLoopError:
            raise
        loss = _validate_support_loss(loss)
        history.append(float(loss.detach().cpu().item()))

        try:
            grads = torch.autograd.grad(
                loss,
                tuple(fast.values()),
                create_graph=False,
                allow_unused=False,
            )
        except (RuntimeError, ValueError) as exc:
            raise MetaInnerLoopError(
                f"missing or unused adapter gradient at inner step {step_index + 1}"
            ) from exc

        updated = OrderedDict()
        for (name, value), grad in zip(fast.items(), grads):
            if grad is None:
                raise MetaInnerLoopError(
                    f"missing adapter gradient for {name!r} at inner step {step_index + 1}"
                )
            if grad.shape != value.shape or grad.dtype != value.dtype or grad.device != value.device:
                raise MetaInnerLoopError(f"gradient contract mismatch for adapter {name!r}")
            if not _is_finite(grad):
                raise MetaInnerLoopError(
                    f"adapter gradient for {name!r} is non-finite at inner step {step_index + 1}"
                )
            step_size = step_sizes[name]
            if not torch.is_tensor(step_size) or step_size.ndim != 0:
                raise MetaInnerLoopError(f"step size for adapter {name!r} must be scalar")
            if step_size.device != value.device or step_size.dtype != value.dtype:
                raise MetaInnerLoopError(f"step size contract mismatch for adapter {name!r}")
            if not _is_finite(step_size):
                raise MetaInnerLoopError(f"step size for adapter {name!r} is non-finite")
            next_value = value - step_size * grad.detach()
            if not _is_finite(next_value):
                raise MetaInnerLoopError(
                    f"updated adapter {name!r} is non-finite at inner step {step_index + 1}"
                )
            updated[name] = next_value
        fast = updated

    return FastAdapterState(fast, steps, tuple(history))


__all__ = [
    "FastAdapterState",
    "MetaInnerLoopError",
    "first_order_adapt",
    "functional_forward",
]
