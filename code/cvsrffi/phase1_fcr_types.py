from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch


if TYPE_CHECKING:
    from .phase1_fcr_canonicalizer import CanonicalOutput
    from .phase1_fcr_factors import ContentOutput
    from .phase1_fcr_fingerprint import (
        FingerprintFactorOutput,
        FingerprintResponseOutput,
    )
    from .phase1_fcr_nuisance import NuisanceOutput


@dataclass(frozen=True)
class FCRConfig:
    input_len: int = 256
    content_stride: int = 4
    content_dim: int = 32
    tx_state_dim: int = 16
    channel_dim: int = 16
    receiver_dim: int = 8
    sync_dim: int = 6
    gain_dim: int = 3
    variance_floor: float = 1e-4
    variance_ceiling: float = 1.0
    decoder_mode: str = "full_physics"


@dataclass
class FCRPairBatch:
    clean_iq: torch.Tensor
    leo_iq: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    receiver_id: torch.Tensor
    day_id: torch.Tensor
    nuisance: torch.Tensor
    nuisance_valid: torch.Tensor
    physical_sample_id: tuple[str, ...]
    pair_id: tuple[str, ...]
    clean_crop_offset: torch.Tensor
    leo_crop_offset: torch.Tensor
    nuisance_pair_index: torch.Tensor
    content_pair_index: torch.Tensor
    fingerprint_pair_index: torch.Tensor
    pair_valid_mask: dict[str, torch.Tensor]


@dataclass
class FCRFactorOutput:
    z_s: torch.Tensor
    z_f_id: torch.Tensor
    z_tx_state: torch.Tensor
    z_n_parts: dict[str, torch.Tensor]
    s_hat: torch.Tensor
    content_confidence: torch.Tensor
    response_coef: Optional[torch.Tensor] = None
    response_quality: Optional[dict[str, torch.Tensor]] = None


@dataclass
class FCRDecodeOutput:
    mu_iq: torch.Tensor
    log_variance: torch.Tensor
    delta_f: torch.Tensor
    decoder_mode: str = "full_physics"


@dataclass
class FCRAggregateOutput:
    canonical: "CanonicalOutput"
    content: "ContentOutput"
    fingerprint: "FingerprintFactorOutput"
    response: "FingerprintResponseOutput"
    nuisance: "NuisanceOutput"
    factors: FCRFactorOutput
    decode: FCRDecodeOutput
    quality: dict[str, torch.Tensor]


@dataclass
class FCRLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float]
