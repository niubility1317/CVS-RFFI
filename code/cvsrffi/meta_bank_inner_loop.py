"""First-order support-only adaptation from an explicit MARC-OT bank state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .meta_weight_bank import parameter_block_key


class MetaBankInnerLoopError(RuntimeError):
    """Raised when a bank fast-state update cannot be proven valid."""


@dataclass(frozen=True)
class BankFastState:
    """Ordered fast parameters plus the learned per-block step sizes used."""

    parameters: OrderedDict[str, Tensor]
    block_lrs: dict[str, Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, OrderedDict) or not self.parameters:
            raise ValueError("BankFastState.parameters must be a non-empty OrderedDict")
        if not isinstance(self.block_lrs, dict) or not self.block_lrs:
            raise ValueError("BankFastState.block_lrs must be a non-empty dict")


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _validate_inputs(
    initial_state: Mapping[str, Tensor],
    block_lrs: Mapping[str, Tensor],
    steps: int,
) -> tuple[OrderedDict[str, Tensor], dict[str, Tensor]]:
    if not isinstance(initial_state, Mapping) or not initial_state:
        raise ValueError("initial bank fast state must be a non-empty mapping")
    if not isinstance(block_lrs, Mapping) or not block_lrs:
        raise ValueError("block_lrs must be a non-empty mapping")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 10:
        raise ValueError("bank inner steps must be an integer in [0, 10]")

    fast: OrderedDict[str, Tensor] = OrderedDict()
    used_blocks: set[str] = set()
    for name, value in initial_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("bank fast parameter names must be non-empty strings")
        block_name = parameter_block_key(name)
        if block_name is None:
            raise ValueError(f"bank fast parameter is outside canonical blocks: {name!r}")
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"bank fast parameter {name!r} must be floating point")
        if not _finite(value):
            raise ValueError(f"bank fast parameter {name!r} must be finite")
        fast[name] = value
        used_blocks.add(block_name)

    lrs: dict[str, Tensor] = {}
    if set(block_lrs) != used_blocks:
        raise ValueError("block_lrs keys must exactly match the bank fast-state blocks")
    for block_name in sorted(used_blocks):
        value = block_lrs[block_name]
        if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 0:
            raise ValueError(f"learning rate for block {block_name!r} must be a floating scalar")
        if not _finite(value) or bool(value <= 0):
            raise ValueError(f"learning rate for block {block_name!r} must be finite and positive")
        lrs[block_name] = value
    return fast, lrs


def first_order_bank_adapt(
    functional_forward: Callable[[Mapping[str, Tensor], Any], Tensor],
    initial_state: Mapping[str, Tensor],
    block_lrs: Mapping[str, Tensor],
    support: Any,
    steps: int,
) -> BankFastState:
    """Run FOMAML updates using only the explicitly supplied support carrier.

    Support gradients are computed with ``create_graph=False`` and detached
    before the update.  The subtraction itself intentionally retains paths to
    the supplied initial state and learned block learning rates.
    """

    if not callable(functional_forward):
        raise TypeError("functional_forward must be callable")
    fast, lrs = _validate_inputs(initial_state, block_lrs, steps)

    for step_index in range(steps):
        loss = functional_forward(fast, support)
        if (
            not isinstance(loss, Tensor)
            or loss.ndim != 0
            or not loss.is_floating_point()
            or not loss.requires_grad
        ):
            raise MetaBankInnerLoopError("support loss must be a scalar floating tensor with gradients")
        if not _finite(loss):
            raise MetaBankInnerLoopError("support loss is non-finite")
        names = tuple(fast)
        try:
            grads = torch.autograd.grad(
                loss,
                tuple(fast.values()),
                create_graph=False,
                allow_unused=False,
            )
        except (RuntimeError, ValueError) as error:
            raise MetaBankInnerLoopError(
                f"missing or unused bank gradient at inner step {step_index + 1}"
            ) from error

        updated: OrderedDict[str, Tensor] = OrderedDict()
        for name, grad in zip(names, grads, strict=True):
            value = fast[name]
            if grad is None or grad.shape != value.shape:
                raise MetaBankInnerLoopError(f"gradient contract mismatch for {name!r}")
            if not _finite(grad):
                raise MetaBankInnerLoopError(f"gradient for {name!r} is non-finite")
            block_name = parameter_block_key(name)
            assert block_name is not None
            step_size = lrs[block_name].to(device=value.device, dtype=value.dtype)
            next_value = value - step_size * grad.detach()
            if not _finite(next_value):
                raise MetaBankInnerLoopError(f"updated bank parameter {name!r} is non-finite")
            updated[name] = next_value
        fast = updated

    return BankFastState(parameters=fast, block_lrs=lrs)


__all__ = ["BankFastState", "MetaBankInnerLoopError", "first_order_bank_adapt"]
