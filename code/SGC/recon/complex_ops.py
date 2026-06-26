from __future__ import annotations

import torch


def require_iq(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    if x.ndim != 3 or x.size(1) != 2:
        raise ValueError(f"{name} must be shaped [B, 2, T].")
    return x


def iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    x = require_iq(x)
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def complex_to_iq(z: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    out = torch.stack([z.real, z.imag], dim=1)
    return out.to(dtype=dtype) if dtype is not None else out


def rms_iq(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = require_iq(x)
    return torch.sqrt(x.float().square().mean(dim=(1, 2)) + float(eps))


def rms_normalize_iq(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    rms = rms_iq(x, eps=eps).view(-1, 1, 1)
    return x / rms.to(dtype=x.dtype), rms


def residual_ratio(x_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_hat = require_iq(x_hat, "x_hat")
    y = require_iq(y, "y")
    num = (x_hat.float() - y.float()).flatten(1).norm(p=2, dim=1)
    den = y.float().flatten(1).norm(p=2, dim=1).clamp_min(float(eps))
    return num / den


def apply_bounded_residual(
    y: torch.Tensor,
    delta: torch.Tensor,
    gate: torch.Tensor,
    *,
    rho: float = 0.15,
) -> torch.Tensor:
    y = require_iq(y, "y")
    delta = require_iq(delta, "delta")
    if gate.ndim == 2:
        gate = gate.view(gate.size(0), 1, 1)
    if gate.ndim != 3 or gate.size(0) != y.size(0):
        raise ValueError("gate must be shaped [B, 1, 1] or [B, 1].")
    return y + float(rho) * gate.to(dtype=y.dtype).clamp(0.0, 1.0) * torch.tanh(delta)


def residual_safety_loss(x_hat: torch.Tensor, y: torch.Tensor, r_max: float = 0.15) -> torch.Tensor:
    ratio = residual_ratio(x_hat, y)
    return torch.relu(ratio - float(r_max)).square().mean()
