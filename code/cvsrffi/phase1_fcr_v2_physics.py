from __future__ import annotations

import math

import torch
from torch import nn

from .phase1_fcr_types import FCRConfig, FCRDecodeOutput


def _require_complex_sequence(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_complex(value) or value.ndim != 2:
        raise ValueError(f"{name} must be complex [B,input_len]")
    return value.to(torch.complex64)


def _expand_real(value: torch.Tensor | float, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"expected {batch_size} values, got {tensor.numel()}")
    return tensor


def _expand_complex(
    value: torch.Tensor | complex | float,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if not torch.is_complex(tensor):
        tensor = torch.complex(tensor.float(), torch.zeros_like(tensor.float()))
    tensor = tensor.to(torch.complex64).reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"expected {batch_size} values, got {tensor.numel()}")
    return tensor


def complex_gram(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        raise ValueError("x must be complex")
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError("x must be [B,T,C] or [B,T]")
    scale = float(max(1, x.size(1)))
    return torch.einsum("btc,btd->bcd", x.conj(), x) / scale


def apply_iq_imbalance(
    signal: torch.Tensor,
    alpha: torch.Tensor | complex | float,
    beta: torch.Tensor | complex | float,
) -> torch.Tensor:
    signal_c = _require_complex_sequence(signal, "signal")
    batch_size = signal_c.size(0)
    alpha_c = _expand_complex(alpha, batch_size=batch_size, device=signal_c.device)
    beta_c = _expand_complex(beta, batch_size=batch_size, device=signal_c.device)
    return alpha_c[:, None] * signal_c + beta_c[:, None] * signal_c.conj()


def apply_sto(signal: torch.Tensor, sto: torch.Tensor | float) -> torch.Tensor:
    signal_c = _require_complex_sequence(signal, "signal")
    batch_size, length = signal_c.shape
    shift = _expand_real(
        sto,
        batch_size=batch_size,
        device=signal_c.device,
        dtype=signal_c.real.dtype,
    )
    index = torch.arange(length, device=signal_c.device, dtype=signal_c.real.dtype).unsqueeze(0)
    source = index - shift[:, None]
    left = torch.floor(source)
    right = left + 1.0
    right_weight = source - left
    left_weight = 1.0 - right_weight
    left_index = left.clamp(0, length - 1).long()
    right_index = right.clamp(0, length - 1).long()
    left_value = torch.gather(signal_c, 1, left_index)
    right_value = torch.gather(signal_c, 1, right_index)
    left_value = left_value * ((left >= 0.0) & (left < length))
    right_value = right_value * ((right >= 0.0) & (right < length))
    return left_value * left_weight + right_value * right_weight


def apply_sfo(signal: torch.Tensor, sfo: torch.Tensor | float) -> torch.Tensor:
    signal_c = _require_complex_sequence(signal, "signal")
    batch_size, length = signal_c.shape
    slope = _expand_real(
        sfo,
        batch_size=batch_size,
        device=signal_c.device,
        dtype=signal_c.real.dtype,
    )
    index = torch.arange(length, device=signal_c.device, dtype=signal_c.real.dtype)
    phase = 2.0 * math.pi * slope[:, None] * index[None, :]
    return signal_c * torch.exp(1j * phase)


def normalize_taps(taps: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(taps):
        raise ValueError("taps must be complex")
    if taps.ndim == 1:
        taps = taps.unsqueeze(0)
    if taps.ndim != 2:
        raise ValueError("taps must be [B,K] or [K]")
    energy = taps.abs().square().sum(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    return taps / energy


def apply_multipath(signal: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    signal_c = _require_complex_sequence(signal, "signal")
    taps_c = normalize_taps(taps.to(signal_c.device, dtype=torch.complex64))
    if taps_c.size(0) == 1 and signal_c.size(0) != 1:
        taps_c = taps_c.expand(signal_c.size(0), -1)
    if taps_c.size(0) != signal_c.size(0):
        raise ValueError("taps batch dimension must match signal")
    padded = torch.nn.functional.pad(signal_c, (taps_c.size(1) - 1, 0))
    windows = padded.unfold(-1, taps_c.size(1), 1)
    return (windows * taps_c.flip(-1)[:, None, :]).sum(dim=-1)


class IdentityInitializedPhysicsDecoder(nn.Module):
    def __init__(self, config: FCRConfig, *, multipath_taps: int = 3) -> None:
        super().__init__()
        self.config = config
        self.multipath_taps = int(max(1, multipath_taps))
        self.log_variance_bias = nn.Parameter(
            torch.tensor(math.log(max(config.variance_floor, 1e-8)), dtype=torch.float32)
        )

    def _identity_nuisance(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        alpha = torch.ones(batch_size, dtype=torch.complex64, device=device)
        beta = torch.zeros(batch_size, dtype=torch.complex64, device=device)
        taps = torch.zeros(batch_size, self.multipath_taps, dtype=torch.complex64, device=device)
        taps[:, 0] = 1.0 + 0.0j
        zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
        return {
            "alpha": alpha,
            "beta": beta,
            "sto": zeros,
            "sfo": zeros,
            "phase": zeros,
            "taps": taps,
        }

    def identity_forward(self, iq: torch.Tensor) -> FCRDecodeOutput:
        signal = _require_complex_sequence(iq, "iq")
        delta = torch.zeros_like(signal)
        return self.forward(signal, delta, self._identity_nuisance(signal.size(0), signal.device))

    def forward(
        self,
        s_hat: torch.Tensor,
        delta_f: torch.Tensor,
        z_n: dict[str, torch.Tensor],
    ) -> FCRDecodeOutput:
        s_hat_c = _require_complex_sequence(s_hat, "s_hat")
        delta_f_c = _require_complex_sequence(delta_f, "delta_f")
        if s_hat_c.shape != delta_f_c.shape:
            raise ValueError("s_hat and delta_f must share shape")
        batch_size, length = s_hat_c.shape
        combined = s_hat_c + delta_f_c
        default = self._identity_nuisance(batch_size, combined.device)
        alpha = z_n.get("alpha", default["alpha"])
        beta = z_n.get("beta", default["beta"])
        sto = z_n.get("sto", default["sto"])
        sfo = z_n.get("sfo", default["sfo"])
        phase = z_n.get("phase", default["phase"])
        taps = z_n.get("taps", default["taps"])

        distorted = apply_iq_imbalance(combined, alpha, beta)
        phase_offset = _expand_real(
            phase,
            batch_size=batch_size,
            device=combined.device,
            dtype=combined.real.dtype,
        )
        distorted = distorted * torch.exp(1j * phase_offset[:, None])
        distorted = apply_sto(distorted, sto)
        distorted = apply_sfo(distorted, sfo)
        distorted = apply_multipath(distorted, taps)

        log_variance = self.log_variance_bias.to(distorted.real.dtype).expand(batch_size, length)
        return FCRDecodeOutput(
            mu_iq=torch.stack((distorted.real, distorted.imag), dim=1),
            log_variance=log_variance,
            delta_f=delta_f_c,
            decoder_mode="identity_initialized",
        )
