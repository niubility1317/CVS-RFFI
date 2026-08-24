"""Small residual adapters and parameter allowlists for CVS_META_ADAPTER_TRI_R4_V1."""

from __future__ import annotations

import math
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


META_ADAPTER_SITES = ("time", "freq", "fusion")
_INNER_PARAMETER_SUFFIXES = (
    "down.weight",
    "down.bias",
    "up.weight",
    "up.bias",
    "gate",
)


class ResidualMetaAdapter(nn.Module):
    """Rank-constrained, near-identity residual adapter."""

    def __init__(self, dim: int, rank: int = 4, init_step_size: float = 1e-3):
        super().__init__()
        dim = int(dim)
        rank = int(rank)
        init_step_size = float(init_step_size)
        if dim <= 0:
            raise ValueError("dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not math.isfinite(init_step_size) or init_step_size <= 0.0:
            raise ValueError("init_step_size must be a finite positive number")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=True)
        self.up = nn.Linear(rank, dim, bias=True)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.log_step_size = nn.Parameter(torch.log(torch.tensor(init_step_size)))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: Tensor) -> Tensor:
        delta = self.up(F.silu(self.down(self.norm(z))))
        return z + torch.tanh(self.gate) * delta

    def step_size(self) -> Tensor:
        return F.softplus(self.log_step_size).clamp(1e-6, 5e-2)


def parse_meta_adapter_sites(sites: str, rank: int) -> tuple[str, ...]:
    """Validate and canonicalize the comma-separated site allowlist."""

    rank = int(rank)
    if rank < 0:
        raise ValueError("meta_adapter_rank must be non-negative")
    raw = str(sites or "").strip()
    if not raw:
        if rank > 0:
            raise ValueError("meta_adapter_sites is required when meta_adapter_rank > 0")
        return ()
    requested = tuple(item.strip().lower() for item in raw.split(","))
    if any(item not in META_ADAPTER_SITES for item in requested):
        raise ValueError(
            "meta_adapter_sites must contain only time,freq,fusion"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("meta_adapter_sites must not contain duplicate sites")
    return tuple(site for site in META_ADAPTER_SITES if site in requested)


def _adapter_modules(model: nn.Module) -> list[tuple[str, ResidualMetaAdapter]]:
    return sorted(
        (
            name,
            module,
        )
        for name, module in model.named_modules()
        if isinstance(module, ResidualMetaAdapter)
    )


def iter_inner_adapter_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only adapter inner-loop parameters in stable full-name order."""

    for module_name, module in _adapter_modules(model):
        for suffix in _INNER_PARAMETER_SUFFIXES:
            parameter = module
            parts = suffix.split(".")
            for part in parts[:-1]:
                parameter = getattr(parameter, part)
            parameter = getattr(parameter, parts[-1])
            full_name = f"{module_name}.{suffix}" if module_name else suffix
            yield full_name, parameter


def adapter_step_size_by_parameter(model: nn.Module) -> Mapping[str, Tensor]:
    """Map each inner parameter to its owning module's differentiable step size."""

    result: dict[str, Tensor] = {}
    for module_name, module in _adapter_modules(model):
        step_size = module.step_size()
        prefix = f"{module_name}." if module_name else ""
        for suffix in _INNER_PARAMETER_SUFFIXES:
            result[f"{prefix}{suffix}"] = step_size
    return result


def adapter_parameter_budget(model: nn.Module) -> dict[str, int | float]:
    """Return auditable total, adapter and inner-loop parameter counts."""

    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    adapter_parameters = int(
        sum(parameter.numel() for _, module in _adapter_modules(model) for parameter in module.parameters())
    )
    inner_parameters = int(
        sum(parameter.numel() for _, parameter in iter_inner_adapter_parameters(model))
    )
    inner_ratio = float(inner_parameters / total_parameters) if total_parameters else 0.0
    return {
        "total_parameters": total_parameters,
        "adapter_parameters": adapter_parameters,
        "inner_parameters": inner_parameters,
        "inner_ratio": inner_ratio,
        "adapter_ratio": float(adapter_parameters / total_parameters) if total_parameters else 0.0,
    }


__all__ = [
    "META_ADAPTER_SITES",
    "ResidualMetaAdapter",
    "adapter_parameter_budget",
    "adapter_step_size_by_parameter",
    "iter_inner_adapter_parameters",
    "parse_meta_adapter_sites",
]
