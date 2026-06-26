from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from .complex_ops import complex_to_iq, iq_to_complex


def _meta_tensor(phi: Mapping[str, object], key: str, batch: int, device, default: float) -> torch.Tensor:
    value = phi.get(key, default)
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.float32).view(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.float32).view(-1)
    return out.expand(batch) if out.numel() == 1 and batch > 1 else out


class DifferentiableSatChannel(nn.Module):
    def __init__(self, *, fs_hz: float = 25e6, fc_hz: float = 2.462e9, eps: float = 1e-8) -> None:
        super().__init__()
        self.fs_hz = float(fs_hz)
        self.fc_hz = float(fc_hz)
        self.eps = float(eps)

    def _apply_multipath(self, z: torch.Tensor, phi: Mapping[str, object]) -> torch.Tensor:
        taps = phi.get("taps")
        delays = phi.get("delays")
        if taps is None or delays is None:
            return z
        B, T = z.shape
        taps_t = torch.as_tensor(taps, device=z.device)
        if not torch.is_complex(taps_t):
            taps_t = torch.complex(taps_t.float(), torch.zeros_like(taps_t.float()))
        taps_t = taps_t.to(device=z.device, dtype=z.dtype)
        if taps_t.ndim == 1:
            taps_t = taps_t.view(1, -1).expand(B, -1)
        delays_t = torch.as_tensor(delays, device=z.device).long().view(-1)
        out = torch.zeros_like(z)
        for idx in range(taps_t.size(1)):
            delay = int(delays_t[min(idx, delays_t.numel() - 1)].item())
            shifted = torch.roll(z, shifts=delay, dims=-1)
            if delay > 0:
                shifted[:, :delay] = 0
            out = out + taps_t[:, idx].view(B, 1) * shifted
        return out

    def forward(self, x_hat: torch.Tensor, phi: Mapping[str, object]) -> torch.Tensor:
        z = iq_to_complex(x_hat)
        B, T = z.shape
        device = x_hat.device
        dtype = x_hat.dtype

        pl_db = _meta_tensor(phi, "pl_db", B, device, 180.0)
        gain = torch.pow(10.0, -(pl_db - pl_db.detach().mean()) / 20.0).to(z.dtype)
        K_db = _meta_tensor(phi, "K_db", B, device, 9.0)
        K = torch.pow(10.0, K_db / 10.0)
        fading_amp = torch.sqrt((K + 0.25) / (K + 1.0)).clamp(0.1, 2.0).to(z.dtype)
        z = z * gain.view(B, 1) * fading_amp.view(B, 1)
        z = self._apply_multipath(z, phi)

        fD = _meta_tensor(phi, "fD_hz", B, device, 0.0)
        cfo = _meta_tensor(phi, "cfo_hz", B, device, 0.0)
        n = torch.arange(T, device=device, dtype=torch.float32).view(1, -1)
        phase = 2.0 * math.pi * (fD + cfo).view(B, 1) * n / self.fs_hz
        z = z * torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(z.dtype)

        pn_std = _meta_tensor(phi, "phase_noise_inc_std", B, device, 0.0)
        if bool((pn_std.abs() > 0).any()):
            inc = pn_std.view(B, 1) * torch.sin(2.0 * math.pi * n / max(T, 1))
            pn = torch.cumsum(inc, dim=-1)
            z = z * torch.exp(torch.complex(torch.zeros_like(pn), pn)).to(z.dtype)

        rms = torch.sqrt((z.real.square() + z.imag.square()).mean(dim=-1, keepdim=True) + self.eps)
        z = z / rms
        resid_db = _meta_tensor(phi, "agc_resid_db", B, device, 0.0)
        z = z * torch.pow(10.0, resid_db.view(B, 1) / 20.0).to(z.dtype)

        amp_db = _meta_tensor(phi, "iq_amp_db", B, device, 0.0)
        phase_deg = _meta_tensor(phi, "iq_phase_deg", B, device, 0.0)
        eps_amp = torch.pow(10.0, amp_db / 20.0)
        phi_rad = torch.deg2rad(phase_deg)
        alpha = 0.5 * (1.0 + eps_amp) * torch.exp(torch.complex(torch.zeros_like(phi_rad), -phi_rad / 2.0))
        beta = 0.5 * (1.0 - eps_amp) * torch.exp(torch.complex(torch.zeros_like(phi_rad), phi_rad / 2.0))
        z = alpha.to(z.dtype).view(B, 1) * z + beta.to(z.dtype).view(B, 1) * torch.conj(z)

        return complex_to_iq(z, dtype=dtype)
