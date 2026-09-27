from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from .phase1_fcr_types import FCRConfig


def excitation_features(s: torch.Tensor) -> torch.Tensor:
    """Return fixed amplitude and slew features for a complex excitation."""

    if not torch.is_complex(s) or s.ndim != 2:
        raise ValueError("s must be complex [B,input_len]")
    amp = s.abs()
    slew = torch.diff(amp, dim=-1, prepend=amp[..., :1])
    return torch.stack((amp, amp.square(), amp.pow(3), slew), dim=-1)


@dataclass
class ContentOutput:
    z_s: torch.Tensor
    s_hat: torch.Tensor
    content_confidence: torch.Tensor


class ContentSequenceEncoder(nn.Module):
    """Produce local stride-four content tokens without early global pooling."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        self.local_encoder = nn.Sequential(
            nn.Conv1d(2, config.content_dim, kernel_size=7, stride=config.content_stride, padding=3),
            nn.GELU(),
            nn.Conv1d(config.content_dim, config.content_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.size(1) != 2 or iq.size(2) != self.config.input_len:
            raise ValueError(
                "iq must have shape [B,2,{}]".format(self.config.input_len)
            )
        tokens = self.local_encoder(iq.float()).transpose(1, 2)
        expected_tokens = self.config.input_len // self.config.content_stride
        if tokens.size(1) != expected_tokens:
            raise RuntimeError("content encoder produced an unexpected token count")
        return tokens


class ContentGenerator(nn.Module):
    """Use short local filters around bounded interpolation to reconstruct content."""

    def __init__(self, config: FCRConfig) -> None:
        super().__init__()
        self.config = config
        self.local_decoder = nn.Sequential(
            nn.Conv1d(config.content_dim, config.content_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.content_dim, 2, kernel_size=5, padding=2),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.config.input_len // self.config.content_stride
        if z_s.ndim != 3 or z_s.size(1) != expected_tokens or z_s.size(2) != self.config.content_dim:
            raise ValueError(
                "z_s must have shape [B,{},{}]".format(expected_tokens, self.config.content_dim)
            )
        with torch.autocast(device_type=z_s.device.type, enabled=False):
            upsampled = functional.interpolate(
                z_s.float().transpose(1, 2),
                size=self.config.input_len,
                mode="linear",
                align_corners=False,
            )
            reconstructed = torch.tanh(self.local_decoder(upsampled))
            return torch.complex(reconstructed[:, 0], reconstructed[:, 1])


class ContentFactorEncoder(nn.Module):
    """Content tokens, local reconstruction and a detached default identity view."""

    def __init__(self, config: FCRConfig, *, detach_identity_input: bool = True) -> None:
        super().__init__()
        self.config = config
        self.detach_identity_input = detach_identity_input
        self.sequence_encoder = ContentSequenceEncoder(config)
        self.generator = ContentGenerator(config)
        self.confidence_head = nn.Linear(config.content_dim, 1)

    def forward(self, canonical_iq: torch.Tensor, *, mask: torch.Tensor | None = None) -> ContentOutput:
        if mask is not None:
            if mask.shape != (canonical_iq.size(0), canonical_iq.size(-1)) or mask.dtype != torch.bool:
                raise ValueError("mask must be bool [B,input_len]")
            canonical_iq = canonical_iq.masked_fill(mask[:, None, :], 0.0)
        z_s = self.sequence_encoder(canonical_iq)
        s_hat = self.generator(z_s)
        content_confidence = torch.sigmoid(self.confidence_head(z_s.mean(dim=1)).squeeze(-1))
        return ContentOutput(
            z_s=z_s,
            s_hat=s_hat,
            content_confidence=content_confidence,
        )

    def identity_input(
        self, z_s: torch.Tensor, *, detach_identity_input: bool | None = None
    ) -> torch.Tensor:
        """Return the content summary for identity CE, detached by default."""

        if z_s.ndim != 3 or z_s.size(-1) != self.config.content_dim:
            raise ValueError("z_s must be [B,T,content_dim]")
        summary = z_s.mean(dim=1)
        should_detach = self.detach_identity_input if detach_identity_input is None else detach_identity_input
        return summary.detach() if should_detach else summary
