"""A1 execution helpers preserving per-parameter gradient arithmetic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class GradientSnapshot:
    """One gradient observation; recapture after clipping or any gradient mutation.

    Detached gradients are retained only to inspect an exceptional parameter.
    Call ``first_nonfinite`` before mutating gradients, and never reuse a snapshot
    across optimizer steps. Normal finite checks and norms share one CPU readback.
    """

    names: tuple[str, ...]
    gradients: tuple[torch.Tensor, ...]
    values: tuple[tuple[float, bool], ...]

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "GradientSnapshot":
        entries = [(name, param.grad.detach()) for name, param in model.named_parameters()
                   if param.grad is not None]
        if not entries:
            return cls((), (), ())
        devices = {gradient.device for _, gradient in entries}
        if len(devices) != 1:
            raise ValueError("GradientSnapshot requires one device per model")
        # Keep the original float32 norm and original-dtype finite operations.
        # Do not replace them with a fused/reordered global norm reduction.
        scalars = [torch.stack((gradient.float().norm(2),
                                torch.isfinite(gradient).all().float()))
                   for _, gradient in entries]
        values = torch.stack(scalars).cpu().tolist()
        return cls(tuple(name for name, _ in entries),
                   tuple(gradient for _, gradient in entries),
                   tuple((float(norm), bool(finite)) for norm, finite in values))

    def norm(self, name_filter: Callable[[str], bool] | None = None) -> float:
        total = 0.0
        seen = 0
        for name, (value, _) in zip(self.names, self.values):
            if name_filter is not None and not name_filter(name):
                continue
            total += value * value
            seen += 1
        return total ** 0.5 if seen else float("nan")

    def first_nonfinite(self) -> dict | None:
        for name, gradient, (_, finite) in zip(self.names, self.gradients, self.values):
            if finite:
                continue
            counts = torch.stack((
                (~torch.isfinite(gradient)).sum(), torch.isnan(gradient).sum(),
                torch.isposinf(gradient).sum(), torch.isneginf(gradient).sum(),
            )).cpu().tolist()
            return dict(zip(("parameter_name", "nonfinite_elements", "nan_elements",
                             "posinf_elements", "neginf_elements"), [name, *map(int, counts)]))
        return None
