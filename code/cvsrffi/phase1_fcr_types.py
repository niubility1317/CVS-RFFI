from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch


if TYPE_CHECKING:
    from .phase1_fcr_canonicalizer import CanonicalOutput
    from .phase1_fcr_factors import ContentOutput
    from .phase1_fcr_fingerprint import (
        FingerprintFactorOutput,
        FingerprintResponseOutput,
    )
    from .phase1_fcr_nuisance import NuisanceOutput


FCR_V2_ETA_SCHEMA_VERSION = "fcr-v2/eta-v1"
FCR_V2_ETA_FIELDS = (
    "snr_db",
    "cfo_hz",
    "residual_cfo_hz",
    "fD_hz",
    "pl_db",
    "K_db",
    "theta_deg",
    "h_km",
    "state",
)
FCR_V2_ETA_UNITS = (
    "dB",
    "Hz",
    "Hz",
    "Hz",
    "dB",
    "dB",
    "degree",
    "km",
    "category_index",
)
FCR_V2_ETA_SCALES = (
    20.0,
    100_000.0,
    100_000.0,
    100_000.0,
    200.0,
    20.0,
    90.0,
    2_000.0,
    2.0,
)
_FCR_V2_REQUIRED_METADATA_FIELDS = (
    "physical_sample_id",
    "content_record_id",
    "crop_offset",
    "common_preamble_id",
    "tx_id",
    "rx_i",
    "day_i",
    "view_type",
    "link_condition",
    "excitation_bin",
    "eta_schema_version",
    "eta_fields",
    "eta_units",
    "eta_scales",
    "eta",
    "eta_valid_mask",
)


def _require_tensor(name: str, value: Any, *, batch_size: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have a leading batch dimension")
    if int(tensor.shape[0]) != int(batch_size):
        raise ValueError(f"{name} must have leading batch size {batch_size}, got {tuple(tensor.shape)}")
    return tensor


def _require_string_tuple(name: str, value: Any, *, batch_size: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of strings")
    result = tuple(str(item) for item in value)
    if len(result) != int(batch_size):
        raise ValueError(f"{name} must contain {batch_size} entries, got {len(result)}")
    return result


def _reverse_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(reversed(values))


def _reverse_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    index = torch.arange(tensor.shape[0] - 1, -1, -1, device=tensor.device)
    return tensor.index_select(0, index)


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


@dataclass(frozen=True)
class FCRV2Metadata:
    physical_sample_id: tuple[str, ...]
    content_record_id: tuple[str, ...]
    crop_offset: torch.Tensor
    common_preamble_id: tuple[str, ...]
    tx_id: torch.Tensor
    rx_i: torch.Tensor
    day_i: torch.Tensor
    view_type: tuple[str, ...]
    link_condition: tuple[str, ...]
    excitation_bin: torch.Tensor
    eta_schema_version: str
    eta_fields: tuple[str, ...]
    eta_units: tuple[str, ...]
    eta_scales: tuple[float, ...]
    eta: torch.Tensor
    eta_valid_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.eta.shape[0])

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, batch_size: int) -> "FCRV2Metadata":
        missing = [name for name in _FCR_V2_REQUIRED_METADATA_FIELDS if name not in mapping]
        if missing:
            raise ValueError(f"missing FCR-V2 metadata fields: {', '.join(missing)}")
        eta_schema_version = str(mapping["eta_schema_version"])
        if eta_schema_version != FCR_V2_ETA_SCHEMA_VERSION:
            raise ValueError(f"eta_schema_version must be {FCR_V2_ETA_SCHEMA_VERSION}, got {eta_schema_version}")
        eta_fields = tuple(str(value) for value in mapping["eta_fields"])
        eta_units = tuple(str(value) for value in mapping["eta_units"])
        eta_scales = tuple(float(value) for value in mapping["eta_scales"])
        if eta_fields != FCR_V2_ETA_FIELDS:
            raise ValueError(f"eta_fields must be {FCR_V2_ETA_FIELDS}, got {eta_fields}")
        if eta_units != FCR_V2_ETA_UNITS:
            raise ValueError(f"eta_units must be {FCR_V2_ETA_UNITS}, got {eta_units}")
        if eta_scales != FCR_V2_ETA_SCALES:
            raise ValueError(f"eta_scales must be {FCR_V2_ETA_SCALES}, got {eta_scales}")
        eta = _require_tensor("eta", mapping["eta"], batch_size=batch_size)
        eta_valid_mask = _require_tensor("eta_valid_mask", mapping["eta_valid_mask"], batch_size=batch_size).to(dtype=torch.bool)
        if eta.ndim != 2:
            raise ValueError(f"eta must have shape (batch, eta_dim), got {tuple(eta.shape)}")
        if eta_valid_mask.shape != eta.shape:
            raise ValueError(
                f"eta_valid_mask must match eta shape {tuple(eta.shape)}, got {tuple(eta_valid_mask.shape)}"
            )
        if int(eta.shape[1]) != len(FCR_V2_ETA_FIELDS):
            raise ValueError(
                f"eta width must match named schema ({len(FCR_V2_ETA_FIELDS)}), got {int(eta.shape[1])}"
            )
        valid_eta = eta[eta_valid_mask]
        if valid_eta.numel() > 0 and not torch.isfinite(valid_eta).all():
            raise ValueError("eta contains non-finite values under eta_valid_mask")
        return cls(
            physical_sample_id=_require_string_tuple("physical_sample_id", mapping["physical_sample_id"], batch_size=batch_size),
            content_record_id=_require_string_tuple("content_record_id", mapping["content_record_id"], batch_size=batch_size),
            crop_offset=_require_tensor("crop_offset", mapping["crop_offset"], batch_size=batch_size),
            common_preamble_id=_require_string_tuple("common_preamble_id", mapping["common_preamble_id"], batch_size=batch_size),
            tx_id=_require_tensor("tx_id", mapping["tx_id"], batch_size=batch_size),
            rx_i=_require_tensor("rx_i", mapping["rx_i"], batch_size=batch_size),
            day_i=_require_tensor("day_i", mapping["day_i"], batch_size=batch_size),
            view_type=_require_string_tuple("view_type", mapping["view_type"], batch_size=batch_size),
            link_condition=_require_string_tuple("link_condition", mapping["link_condition"], batch_size=batch_size),
            excitation_bin=_require_tensor("excitation_bin", mapping["excitation_bin"], batch_size=batch_size),
            eta_schema_version=eta_schema_version,
            eta_fields=eta_fields,
            eta_units=eta_units,
            eta_scales=eta_scales,
            eta=eta,
            eta_valid_mask=eta_valid_mask,
        )

    def flip_batch(self) -> "FCRV2Metadata":
        return FCRV2Metadata(
            physical_sample_id=_reverse_tuple(self.physical_sample_id),
            content_record_id=_reverse_tuple(self.content_record_id),
            crop_offset=_reverse_tensor(self.crop_offset),
            common_preamble_id=_reverse_tuple(self.common_preamble_id),
            tx_id=_reverse_tensor(self.tx_id),
            rx_i=_reverse_tensor(self.rx_i),
            day_i=_reverse_tensor(self.day_i),
            view_type=_reverse_tuple(self.view_type),
            link_condition=_reverse_tuple(self.link_condition),
            excitation_bin=_reverse_tensor(self.excitation_bin),
            eta_schema_version=self.eta_schema_version,
            eta_fields=self.eta_fields,
            eta_units=self.eta_units,
            eta_scales=self.eta_scales,
            eta=_reverse_tensor(self.eta),
            eta_valid_mask=_reverse_tensor(self.eta_valid_mask),
        )


@dataclass(frozen=True)
class FCRV2FactorOutput:
    z_s: torch.Tensor
    z_f_id: torch.Tensor
    z_f_dev: torch.Tensor
    z_n: dict[str, torch.Tensor]
    s_hat: torch.Tensor
    delta_f: torch.Tensor
    canonical_residual: Optional[torch.Tensor] = None
    response_quality: Optional[dict[str, torch.Tensor]] = None
    eta_pred: Optional[torch.Tensor] = None

    @property
    def z_n_parts(self) -> dict[str, torch.Tensor]:
        return self.z_n

    def decoder_inputs(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.s_hat, self.delta_f, self.z_n


@dataclass(frozen=True)
class FCRV2CapabilityState:
    eta_ready: bool
    decoder_ready: bool
    swap_ready: bool
    fingerprint_ready: bool
    reasons: dict[str, str] = field(default_factory=dict)

    def reason_for(self, name: str) -> str | None:
        return self.reasons.get(name)


@dataclass(frozen=True)
class FCRV2LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float | str]
    active_losses: frozenset[str] = field(default_factory=frozenset)
    weights: dict[str, float] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
