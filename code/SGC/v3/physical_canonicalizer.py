from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class PhysicalSafeCanonicalizerConfig:
    eps: float = 1e-8
    cfo_betas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5)
    shifts: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    envelope_gammas: tuple[float, ...] = (0.0, 0.15, 0.3)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object]) -> "PhysicalSafeCanonicalizerConfig":
        data = dict(cfg)
        for key in ("cfo_betas", "shifts", "envelope_gammas"):
            if key in data:
                data[key] = tuple(data[key])  # type: ignore[arg-type]
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError("IQ input must be shaped [B, 2, T].")
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def _from_complex(s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([s.real, s.imag], dim=1).to(dtype=dtype)


def _rms_normalize(s: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    rms = torch.sqrt((s.real.square() + s.imag.square()).mean(dim=-1, keepdim=True) + float(eps))
    return s / rms, rms


class PhysicalSafeCanonicalizer(nn.Module):
    """Deterministic low-degree physical IQ candidate generator.

    This module intentionally has no trainable waveform residual branch.
    """

    def __init__(
        self,
        cfg: PhysicalSafeCanonicalizerConfig | Mapping[str, object] | None = None,
        *,
        eps: float | None = None,
        cfo_betas: Sequence[float] | None = None,
        shifts: Sequence[int] | None = None,
        envelope_gammas: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = PhysicalSafeCanonicalizerConfig()
        elif isinstance(cfg, Mapping):
            cfg = PhysicalSafeCanonicalizerConfig.from_mapping(cfg)
        if eps is not None:
            cfg.eps = float(eps)
        if cfo_betas is not None:
            cfg.cfo_betas = tuple(float(v) for v in cfo_betas)
        if shifts is not None:
            cfg.shifts = tuple(int(v) for v in shifts)
        if envelope_gammas is not None:
            cfg.envelope_gammas = tuple(float(v) for v in envelope_gammas)
        self.cfg = cfg

    @property
    def num_candidate_views(self) -> int:
        cfo_count = sum(1 for beta in self.cfg.cfo_betas if abs(float(beta)) > 1e-12)
        env_count = sum(1 for gamma in self.cfg.envelope_gammas if abs(float(gamma)) > 1e-12)
        return 2 + cfo_count + len(self.cfg.shifts) + env_count

    def _cfo_hat(self, s: torch.Tensor) -> torch.Tensor:
        prod = s[:, 1:] * torch.conj(s[:, :-1])
        return torch.angle(prod.sum(dim=-1, keepdim=True))

    def _cfo_correct(self, s: torch.Tensor, beta: float, cfo_hat: torch.Tensor) -> torch.Tensor:
        n = torch.arange(s.size(-1), device=s.device, dtype=s.real.dtype).view(1, -1)
        phase = -float(beta) * cfo_hat.to(dtype=s.real.dtype) * n
        return s * torch.exp(torch.complex(torch.zeros_like(phase), phase))

    def _spectral_envelope(self, s: torch.Tensor, gamma: float) -> torch.Tensor:
        spectrum = torch.fft.fft(s, dim=-1)
        amp = spectrum.abs()
        kernel_size = min(9, max(3, (s.size(-1) // 16) * 2 + 1))
        pad = kernel_size // 2
        smooth = F.avg_pool1d(amp.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)
        corrected = spectrum / (smooth.clamp_min(self.cfg.eps).pow(float(gamma)))
        return torch.fft.ifft(corrected, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        dtype = x.dtype
        s_raw = _to_complex(x)
        s, rms = _rms_normalize(s_raw, self.cfg.eps)
        cfo_hat = self._cfo_hat(s)
        views = [s]
        names = ["identity"]

        s_dc, _ = _rms_normalize(s_raw - s_raw.mean(dim=-1, keepdim=True), self.cfg.eps)
        views.append(s_dc)
        names.append("dc_removed")

        for beta in self.cfg.cfo_betas:
            beta_f = float(beta)
            if abs(beta_f) <= 1e-12:
                continue
            view, _ = _rms_normalize(self._cfo_correct(s, beta_f, cfo_hat), self.cfg.eps)
            views.append(view)
            names.append(f"cfo_beta_{beta_f:g}")

        for shift in self.cfg.shifts:
            view = torch.roll(s, shifts=int(shift), dims=-1)
            views.append(view)
            names.append(f"shift_{int(shift)}")

        for gamma in self.cfg.envelope_gammas:
            gamma_f = float(gamma)
            if abs(gamma_f) <= 1e-12:
                continue
            view, _ = _rms_normalize(self._spectral_envelope(s, gamma_f), self.cfg.eps)
            views.append(view)
            names.append(f"envelope_gamma_{gamma_f:g}")

        stacked = torch.stack([_from_complex(v, dtype) for v in views], dim=1)
        power = s.abs().square()
        papr = power.max(dim=-1, keepdim=True).values / power.mean(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        return {
            "views": stacked,
            "view_names": names,
            "stats": {
                "rms": rms.detach(),
                "papr": papr.detach(),
                "cfo_hat": cfo_hat.detach(),
            },
        }
