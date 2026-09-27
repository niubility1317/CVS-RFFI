"""Channel--Receiver Robust Adapter (CRRA) primitives for ADVB02.

The module is deliberately independent from the ADVB02 model so that the
identity path can be tested without constructing the complete dual backbone.
It operates on paired Sinc/IQ feature maps and keeps the intervention bounded
and close to the identity at the beginning of training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def crra_gate_scale(
    epoch: int,
    *,
    start_epoch: int = 17,
    ramp_epochs: int = 30,
) -> float:
    """Return the deterministic E1--16/E17--46/E47+ intervention schedule."""

    epoch = int(epoch)
    start_epoch = max(1, int(start_epoch))
    ramp_epochs = max(1, int(ramp_epochs))
    if epoch < start_epoch:
        return 0.0
    return min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))


def _finite_iq(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)


def compute_crra_rcn_stats(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the fixed 18-dimensional RCN nuisance statistic vector.

    The statistics are detached by construction. This is intentional: the
    first CRRA condition path must not let the condition estimator pull the
    identity representation toward nuisance metadata.
    """

    if x.dim() != 3 or int(x.size(1)) != 2:
        raise ValueError("CRRA RCN statistics expect IQ tensors shaped [B, 2, T]")
    x = _finite_iq(x)
    i = x[:, 0, :]
    q = x[:, 1, :]
    amp = torch.sqrt(i * i + q * q + float(eps))
    power = torch.log1p(amp * amp)

    def moments(v: torch.Tensor):
        return v.mean(dim=1), v.std(dim=1, unbiased=False).clamp_min(float(eps)), v.abs().mean(dim=1)

    i_mean, i_std, i_abs = moments(i)
    q_mean, q_std, q_abs = moments(q)
    a_mean, a_std, a_abs = moments(amp)
    p_mean, p_std, _ = moments(power)
    corr = ((i - i_mean[:, None]) * (q - q_mean[:, None])).mean(dim=1)
    corr = corr / (i_std * q_std).clamp_min(float(eps))
    imbalance = torch.log((i_std + float(eps)) / (q_std + float(eps)))

    if int(x.size(-1)) > 1:
        di = i[:, 1:] - i[:, :-1]
        dq = q[:, 1:] - q[:, :-1]
        da = amp[:, 1:] - amp[:, :-1]
        cross = q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]
        dot = i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]
        dphi = torch.atan2(cross, dot + float(eps))
        di_abs = di.abs().mean(dim=1)
        dq_abs = dq.abs().mean(dim=1)
        da_abs = da.abs().mean(dim=1)
        dphi_mean, dphi_std, dphi_abs = moments(dphi)
    else:
        zero = i.new_zeros(i.size(0))
        di_abs = dq_abs = da_abs = dphi_mean = dphi_std = dphi_abs = zero

    dphi_summary = dphi_std + 0.1 * dphi_abs
    return torch.stack(
        [
            i_mean,
            i_std,
            i_abs,
            q_mean,
            q_std,
            q_abs,
            a_mean,
            a_std,
            a_abs,
            p_mean,
            p_std,
            corr.clamp(-5.0, 5.0),
            imbalance.clamp(-5.0, 5.0),
            di_abs,
            dq_abs,
            da_abs,
            dphi_mean.clamp(-torch.pi, torch.pi),
            dphi_summary,
        ],
        dim=1,
    ).detach()


class ComplexIQShrinkageWhitening(nn.Module):
    """Per-sample 2x2 I/Q covariance shrinkage whitening.

    The input layout is the existing CV-SincNet layout: the first
    ``iq_channels`` channels are I filter responses and the remaining
    ``iq_channels`` channels are Q filter responses.
    """

    def __init__(self, iq_channels: int, shrinkage: float = 0.10, eps: float = 1e-5):
        super().__init__()
        self.iq_channels = int(iq_channels)
        if self.iq_channels <= 0:
            raise ValueError("iq_channels must be positive")
        self.shrinkage = float(max(0.0, min(1.0, shrinkage)))
        self.eps = float(eps)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() != 3:
            raise ValueError("CRRA feature must be shaped [B, C, T]")
        expected = 2 * self.iq_channels
        if int(feature.size(1)) != expected:
            raise ValueError(f"CRRA paired feature expects {expected} channels, got {feature.size(1)}")
        source_dtype = feature.dtype
        y = torch.nan_to_num(feature.float(), nan=0.0, posinf=0.0, neginf=0.0)
        bsz, _, steps = y.shape
        pair = y.reshape(bsz, 2, self.iq_channels, steps).permute(0, 2, 1, 3)
        mean = pair.mean(dim=-1, keepdim=True)
        centered = pair - mean
        c00 = (centered[:, :, 0, :] ** 2).mean(dim=-1)
        c11 = (centered[:, :, 1, :] ** 2).mean(dim=-1)
        c01 = (centered[:, :, 0, :] * centered[:, :, 1, :]).mean(dim=-1)
        trace_half = 0.5 * (c00 + c11)
        shrink = self.shrinkage
        a = (1.0 - shrink) * c00 + shrink * trace_half + self.eps
        d = (1.0 - shrink) * c11 + shrink * trace_half + self.eps
        off = (1.0 - shrink) * c01
        cov = torch.stack(
            [torch.stack([a, off], dim=-1), torch.stack([off, d], dim=-1)],
            dim=-2,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        inv_sqrt = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(self.eps).rsqrt()) @ eigenvectors.transpose(-1, -2)
        whitened = torch.einsum("bsij,bsjt->bsit", inv_sqrt, centered) + mean
        return whitened.permute(0, 2, 1, 3).reshape(bsz, expected, steps).to(dtype=source_dtype)


class LowRankDepthwiseResidual(nn.Module):
    """Low-rank depthwise temporal residual with zero-init up projection."""

    def __init__(self, feature_channels: int, rank: int = 8, kernel_size: int = 5):
        super().__init__()
        feature_channels = int(feature_channels)
        rank = int(rank)
        if feature_channels <= 0 or rank <= 0:
            raise ValueError("feature_channels and rank must be positive")
        if int(kernel_size) % 2 == 0:
            raise ValueError("CRRA depthwise kernel_size must be odd")
        self.depthwise = nn.Conv1d(
            feature_channels,
            feature_channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=feature_channels,
            bias=False,
        )
        self.down = nn.Conv1d(feature_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, feature_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.depthwise(x))))


class SourceSupportGate(nn.Module):
    """Source-only running support statistics for the condition vector."""

    def __init__(self, q_dim: int, momentum: float = 0.95, eps: float = 1e-5):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("center", torch.zeros(int(q_dim)))
        self.register_buffer("scale", torch.ones(int(q_dim)))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, q: torch.Tensor) -> None:
        if q.numel() == 0:
            return
        q = q.detach().float()
        mean = q.mean(dim=0)
        std = q.std(dim=0, unbiased=False).clamp_min(self.eps)
        if int(self.count.item()) == 0:
            self.center.copy_(mean)
            self.scale.copy_(std)
        else:
            m = self.momentum
            self.center.mul_(m).add_(mean, alpha=1.0 - m)
            self.scale.mul_(m).add_(std, alpha=1.0 - m)
        self.count.add_(int(q.size(0)))

    def forward(
        self,
        q: torch.Tensor,
        *,
        update_source: bool = False,
        update_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and update_source:
            q_update = q
            if update_mask is not None:
                mask = update_mask.to(device=q.device).view(-1).bool()
                if mask.numel() != q.size(0):
                    raise ValueError("CRRA source support mask must align with the condition batch")
                q_update = q[mask]
            if q_update.numel() > 0:
                self.update(q_update)
        normalized = (q - self.center.to(device=q.device, dtype=q.dtype)) / self.scale.to(device=q.device, dtype=q.dtype).clamp_min(self.eps)
        distance = normalized.pow(2).mean(dim=1).sqrt()
        support = torch.exp(-0.5 * distance).clamp(0.0, 1.0)
        return support, distance


@dataclass
class CRRAOutput:
    feature: torch.Tensor
    alpha: torch.Tensor
    gate: torch.Tensor
    correction_energy: torch.Tensor
    support_distance: torch.Tensor
    q: torch.Tensor
    q_raw: Optional[torch.Tensor] = None
    nuisance_pred: Optional[torch.Tensor] = None


class CRRAAdapter(nn.Module):
    """Bounded channel/receiver robust adapter used on the identity path."""

    def __init__(
        self,
        *,
        iq_channels: Optional[int],
        feature_channels: int,
        rank: int = 8,
        alpha_max: float = 0.25,
        condition_dim: int = 32,
        nuisance_dim: int = 0,
        use_whitening: bool = True,
        shrinkage: float = 0.10,
        kernel_size: int = 5,
        start_epoch: int = 17,
        ramp_epochs: int = 30,
    ):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.rank = int(rank)
        self.alpha_max = float(max(0.0, alpha_max))
        self.condition_dim = int(condition_dim)
        self.use_whitening = bool(use_whitening)
        self.start_epoch = max(1, int(start_epoch))
        self.ramp_epochs = max(1, int(ramp_epochs))
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if self.use_whitening:
            if iq_channels is None or int(iq_channels) <= 0:
                raise ValueError("iq_channels must be positive when whitening is enabled")
            if self.feature_channels != 2 * int(iq_channels):
                raise ValueError("paired CRRA feature channels must be even")
            self.iq_channels = int(iq_channels)
            self.whitening = ComplexIQShrinkageWhitening(self.iq_channels, shrinkage=shrinkage)
        else:
            self.iq_channels = None
            self.whitening = nn.Identity()

        condition_input = 18 + self.feature_channels
        hidden = max(64, 2 * self.condition_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.condition_dim),
        )
        self.alpha_head = nn.Linear(self.condition_dim, 1)
        self.gate_head = nn.Linear(self.condition_dim, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.residual = LowRankDepthwiseResidual(self.feature_channels, rank=self.rank, kernel_size=kernel_size)
        self.support = SourceSupportGate(self.condition_dim)
        self.nuisance_head = nn.Linear(self.condition_dim, int(nuisance_dim)) if int(nuisance_dim) > 0 else None

    def _condition(
        self,
        feature: torch.Tensor,
        raw_iq: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_iq is None:
            stats = feature.new_zeros((feature.size(0), 18))
        else:
            stats = compute_crra_rcn_stats(raw_iq).to(device=feature.device, dtype=feature.dtype)
        gap = torch.nan_to_num(feature.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
        condition_input = torch.cat([stats.detach().float(), gap.detach().float()], dim=1)
        q_raw = self.condition_encoder(condition_input).tanh()
        return q_raw, q_raw.detach()

    def forward(
        self,
        feature: torch.Tensor,
        *,
        raw_iq: Optional[torch.Tensor] = None,
        epoch: int = 1,
        update_source_support: bool = False,
        source_support_mask: Optional[torch.Tensor] = None,
    ) -> CRRAOutput:
        if feature.dim() != 3 or int(feature.size(1)) != self.feature_channels:
            raise ValueError(f"CRRA feature expects [B, {self.feature_channels}, T]")
        q_raw, q = self._condition(feature, raw_iq)
        support, support_distance = self.support(
            q,
            update_source=bool(update_source_support),
            update_mask=source_support_mask,
        )
        scale = feature.new_tensor(
            crra_gate_scale(
                int(epoch),
                start_epoch=self.start_epoch,
                ramp_epochs=self.ramp_epochs,
            )
        )
        alpha = self.alpha_max * torch.sigmoid(self.alpha_head(q)).view(-1)
        gate = scale * support * torch.sigmoid(self.gate_head(q)).view(-1)
        whitened = self.whitening(feature) if self.use_whitening else feature
        intervention = whitened - feature
        low_rank = self.residual(feature)
        correction = alpha[:, None, None] * intervention + low_rank
        robust = feature + gate[:, None, None] * correction
        correction_energy = (robust - feature).pow(2).mean(dim=(1, 2)).sqrt()
        nuisance_pred = self.nuisance_head(q_raw) if self.nuisance_head is not None else None
        return CRRAOutput(
            feature=robust,
            alpha=alpha,
            gate=gate,
            correction_energy=correction_energy,
            support_distance=support_distance,
            q=q,
            q_raw=q_raw,
            nuisance_pred=nuisance_pred,
        )


__all__ = [
    "CRRAAdapter",
    "CRRAOutput",
    "ComplexIQShrinkageWhitening",
    "LowRankDepthwiseResidual",
    "compute_crra_rcn_stats",
    "crra_gate_scale",
]
