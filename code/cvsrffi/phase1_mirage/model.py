"""Lightweight, role-blind MIRAGE IQ feature encoder.

This module exposes only the frozen Phase1 feature boundary.  Open-world
classification, unknown scoring, and training losses intentionally belong to
later modules so the same encoder can serve training and deployment paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class MIRAGEConfig:
    """Frozen architecture settings for the lightweight MIRAGE encoder."""

    patch_kernel: int = 32
    patch_stride: int = 16
    token_dim: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    z_id_dim: int = 160
    z_dom_dim: int = 32
    formal_mode: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "patch_kernel",
            "patch_stride",
            "token_dim",
            "transformer_layers",
            "transformer_heads",
            "z_id_dim",
            "z_dom_dim",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.token_dim % self.transformer_heads != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if not isinstance(self.formal_mode, bool):
            raise ValueError("formal_mode must be a bool")


@dataclass
class MIRAGEFeatures:
    """Role-blind features consumed by later heads, losses, and deployment code."""

    z_id: Tensor
    z_dom: Tensor
    quality: Tensor
    tokens: Tensor


def _validate_iq_tensor(x: Tensor, *, patch_kernel: int | None = None) -> None:
    """Reject malformed inputs before numeric preprocessing or learned compute."""

    if not isinstance(x, Tensor):
        raise TypeError("IQ input must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("IQ input must use a floating dtype")
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("IQ input must have shape [B, 2, T]")
    if x.shape[0] < 1:
        raise ValueError("IQ batch must be non-empty")
    if x.shape[-1] < 1:
        raise ValueError("IQ temporal length must be positive")
    if patch_kernel is not None and x.shape[-1] < patch_kernel:
        raise ValueError("IQ temporal length must be at least patch_kernel")


def preprocess_iq(x: Tensor) -> tuple[Tensor, Tensor]:
    """Center and RMS-normalize IQ while retaining two interpretable quality cues.

    ``torch.nan_to_num`` is intentionally used only here, at the public IQ
    boundary.  The rest of the encoder fails explicitly in formal mode instead
    of silently repairing an intermediate numerical error.
    """

    _validate_iq_tensor(x)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centered = x - x.mean(dim=-1, keepdim=True)
    rms = centered.square().sum(dim=1).mean(dim=-1).clamp_min(1e-8).sqrt()
    normalized = centered / rms[:, None, None]
    peak = centered.square().sum(dim=1).sqrt().amax(dim=-1)
    quality_aux = torch.stack([rms.log(), peak / rms.clamp_min(1e-8)], dim=1)
    return normalized, quality_aux


class _MultiScaleDepthwise(nn.Module):
    """Parallel depthwise local filters for transient, modulation, and envelope cues."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            (
                nn.Conv1d(token_dim, token_dim, kernel_size=3, padding=1, groups=token_dim),
                nn.Conv1d(token_dim, token_dim, kernel_size=7, padding=3, groups=token_dim),
                nn.Conv1d(token_dim, token_dim, kernel_size=15, padding=7, groups=token_dim),
            )
        )
        self.mix = nn.Conv1d(token_dim * 3, token_dim, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, tokens: Tensor) -> Tensor:
        """Return local tokens with the input's ``[B,N,D]`` layout."""

        channels_first = tokens.transpose(1, 2)
        multi_scale = torch.cat(tuple(branch(channels_first) for branch in self.branches), dim=1)
        return self.activation(self.mix(multi_scale)).transpose(1, 2)


class _PreNormTransformerLayer(nn.Module):
    """A compact pre-norm Transformer layer without duplicated encoder modules."""

    def __init__(self, token_dim: int, transformer_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            token_dim,
            transformer_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Aggregate global relationships while preserving a residual token path."""

        attention_input = self.attention_norm(tokens)
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        tokens = tokens + attention_output
        return tokens + self.feedforward(self.feedforward_norm(tokens))


def _sinusoidal_positions(*, length: int, token_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build non-learned positional features for any valid patch count."""

    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequency_indices = torch.arange(0, token_dim, 2, device=device, dtype=dtype)
    frequencies = torch.exp(-math.log(10_000.0) * frequency_indices / token_dim)
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros((length, token_dim), device=device, dtype=dtype)
    encoding[:, 0::2] = angles.sin()
    if token_dim > 1:
        encoding[:, 1::2] = angles[:, : token_dim // 2].cos()
    return encoding.unsqueeze(0)


class MIRAGEEncoder(nn.Module):
    """Encode role-blind IQ into identity, domain, quality, and token features."""

    def __init__(self, config: MIRAGEConfig | None = None) -> None:
        super().__init__()
        self.config = MIRAGEConfig() if config is None else config
        if not isinstance(self.config, MIRAGEConfig):
            raise TypeError("config must be a MIRAGEConfig")

        token_dim = self.config.token_dim
        self.patch_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=token_dim,
                kernel_size=self.config.patch_kernel,
                stride=self.config.patch_stride,
            ),
            nn.GELU(),
        )
        self.local_encoder = _MultiScaleDepthwise(token_dim)
        self.local_norm = nn.LayerNorm(token_dim)
        self.transformer_layers = nn.ModuleList(
            _PreNormTransformerLayer(token_dim, self.config.transformer_heads)
            for _ in range(self.config.transformer_layers)
        )
        self.fusion_norm = nn.LayerNorm(token_dim)
        self.quality_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.identity_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.config.z_id_dim),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(token_dim + 2),
            nn.Linear(token_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, self.config.z_dom_dim),
        )

    def _require_finite(self, value: Tensor, stage: str) -> Tensor:
        """Raise instead of silently correcting a non-finite internal result."""

        if self.config.formal_mode and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite internal tensor at {stage}")
        return value

    def forward(self, x: Tensor) -> MIRAGEFeatures:
        """Return frozen, role-blind MIRAGE features for one IQ batch."""

        _validate_iq_tensor(x, patch_kernel=self.config.patch_kernel)
        parameter = self.patch_stem[0].weight
        if x.device != parameter.device:
            raise ValueError("IQ input device must match encoder device")
        normalized, quality_aux = preprocess_iq(x.to(dtype=parameter.dtype))
        normalized = self._require_finite(normalized, "preprocess.normalized")
        quality_aux = self._require_finite(quality_aux, "preprocess.quality_aux")

        patch_tokens = self.patch_stem(normalized).transpose(1, 2)
        patch_tokens = self._require_finite(patch_tokens, "patch_stem")
        local_tokens = self.local_norm(self.local_encoder(patch_tokens))
        local_tokens = self._require_finite(local_tokens, "local_encoder")

        tokens = local_tokens + _sinusoidal_positions(
            length=local_tokens.shape[1],
            token_dim=self.config.token_dim,
            device=local_tokens.device,
            dtype=local_tokens.dtype,
        )
        tokens = self._require_finite(tokens, "positional_encoding")
        for layer_index, transformer_layer in enumerate(self.transformer_layers):
            tokens = transformer_layer(tokens)
            tokens = self._require_finite(tokens, f"transformer_layer_{layer_index}")

        quality = torch.sigmoid(self.quality_head(quality_aux)).squeeze(-1)
        quality = self._require_finite(quality, "quality_head")
        local_summary = local_tokens.mean(dim=1)
        global_summary = tokens.mean(dim=1)
        fused = quality[:, None] * local_summary + (1.0 - quality[:, None]) * global_summary
        fused = self.fusion_norm(fused)
        fused = self._require_finite(fused, "quality_gated_fusion")

        z_id_raw = self.identity_head(fused)
        z_id_raw = self._require_finite(z_id_raw, "identity_head")
        z_id_norm = torch.linalg.vector_norm(z_id_raw, dim=1)
        if bool((z_id_norm <= 1e-12).any()):
            raise FloatingPointError("identity_head produced a zero-norm identity representation")
        z_id = functional.normalize(z_id_raw, p=2.0, dim=1, eps=1e-12)
        z_id = self._require_finite(z_id, "z_id")

        z_dom = self.domain_head(torch.cat((fused, quality_aux), dim=1))
        z_dom = self._require_finite(z_dom, "domain_head")
        tokens = self._require_finite(tokens, "tokens")
        return MIRAGEFeatures(z_id=z_id, z_dom=z_dom, quality=quality, tokens=tokens)


__all__ = ["MIRAGEConfig", "MIRAGEEncoder", "MIRAGEFeatures", "preprocess_iq"]
