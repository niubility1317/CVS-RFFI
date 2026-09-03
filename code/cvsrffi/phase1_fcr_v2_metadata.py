from __future__ import annotations

from typing import Any, Mapping

import torch

from cvsrffi.phase1_fcr_types import (
    FCRV2Metadata,
    FCR_V2_ETA_SCHEMA_VERSION,
)


_REQUIRED_BATCH_FIELDS = (
    "physical_sample_id",
    "content_record_id",
    "crop_offset",
    "common_preamble_id",
    "tx_id",
    "rx_i",
    "day_i",
    "link_condition",
    "excitation_bin",
)


def _require_mapping(batch: Any) -> Mapping[str, Any]:
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a metadata mapping")
    return batch


def _batch_size(batch: Mapping[str, Any]) -> int:
    for key in _REQUIRED_BATCH_FIELDS:
        if key not in batch:
            raise ValueError(f"missing FCR-V2 batch metadata field: {key}")
    physical_sample_id = batch["physical_sample_id"]
    if not isinstance(physical_sample_id, (list, tuple)):
        raise ValueError("physical_sample_id must be a list or tuple of strings")
    return int(len(physical_sample_id))


def _string_tuple(values: Any, *, name: str, batch_size: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    result = tuple(str(value) for value in values)
    if len(result) != int(batch_size):
        raise ValueError(f"{name} must contain {batch_size} entries, got {len(result)}")
    return result


def _long_tensor(values: Any, *, name: str, batch_size: int) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    tensor = tensor.reshape(-1)
    if int(tensor.numel()) != int(batch_size):
        raise ValueError(f"{name} must have leading batch size {batch_size}, got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.long)


def _require_augmented_rows(augmentation: Any, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(augmentation, "eta"):
        raise ValueError("augmentation must expose eta")
    if not hasattr(augmentation, "eta_valid_mask"):
        raise ValueError("augmentation must expose eta_valid_mask")
    eta = torch.as_tensor(getattr(augmentation, "eta"))
    eta_valid_mask = torch.as_tensor(getattr(augmentation, "eta_valid_mask"), dtype=torch.bool)
    if eta.ndim != 2 or int(eta.shape[0]) != int(batch_size):
        raise ValueError(f"augmentation eta must have shape ({batch_size}, eta_dim), got {tuple(eta.shape)}")
    if tuple(eta_valid_mask.shape) != tuple(eta.shape):
        raise ValueError(
            f"augmentation eta_valid_mask must match eta shape {tuple(eta.shape)}, got {tuple(eta_valid_mask.shape)}"
        )
    schema = str(getattr(augmentation, "eta_schema_version", FCR_V2_ETA_SCHEMA_VERSION))
    if schema != FCR_V2_ETA_SCHEMA_VERSION:
        raise ValueError(f"eta_schema_version must be {FCR_V2_ETA_SCHEMA_VERSION}, got {schema}")
    valid_eta = eta[eta_valid_mask]
    if valid_eta.numel() > 0 and not torch.isfinite(valid_eta).all():
        raise ValueError("eta contains non-finite values under eta_valid_mask")
    return eta.to(dtype=torch.float32), eta_valid_mask


def _require_view_alignment(
    batch: Mapping[str, Any],
    augmentation: Any,
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], torch.Tensor]:
    expected_ids = _string_tuple(batch["physical_sample_id"], name="physical_sample_id", batch_size=batch_size)
    actual_ids = _string_tuple(
        getattr(augmentation, "physical_sample_id", None),
        name="augmentation.physical_sample_id",
        batch_size=batch_size,
    )
    if actual_ids != expected_ids:
        raise ValueError("augmentation physical_sample_id must match batch physical_sample_id")
    expected_crop = _long_tensor(batch["crop_offset"], name="crop_offset", batch_size=batch_size)
    actual_crop = _long_tensor(
        getattr(augmentation, "crop_offset", None),
        name="augmentation.crop_offset",
        batch_size=batch_size,
    )
    if not torch.equal(actual_crop, expected_crop):
        raise ValueError("augmentation crop_offset must match batch crop_offset")
    return actual_ids, actual_crop


def build_fcr_v2_metadata(batch: Mapping[str, Any], augmentation: Any) -> FCRV2Metadata:
    batch = _require_mapping(batch)
    batch_size = _batch_size(batch)
    eta, eta_valid_mask = _require_augmented_rows(augmentation, batch_size=batch_size)
    physical_sample_id, crop_offset = _require_view_alignment(batch, augmentation, batch_size=batch_size)
    content_record_id = _string_tuple(batch["content_record_id"], name="content_record_id", batch_size=batch_size)
    common_preamble_id = _string_tuple(
        batch["common_preamble_id"],
        name="common_preamble_id",
        batch_size=batch_size,
    )
    link_condition = _string_tuple(batch["link_condition"], name="link_condition", batch_size=batch_size)
    tx_id = _long_tensor(batch["tx_id"], name="tx_id", batch_size=batch_size)
    rx_i = _long_tensor(batch["rx_i"], name="rx_i", batch_size=batch_size)
    day_i = _long_tensor(batch["day_i"], name="day_i", batch_size=batch_size)
    excitation_bin = _long_tensor(batch["excitation_bin"], name="excitation_bin", batch_size=batch_size)
    scenario = str(getattr(augmentation, "scenario", "") or "satellite")
    clean_eta = torch.zeros_like(eta)
    clean_valid = torch.ones_like(eta_valid_mask, dtype=torch.bool)
    return FCRV2Metadata.from_mapping(
        {
            "physical_sample_id": physical_sample_id + physical_sample_id,
            "content_record_id": content_record_id + content_record_id,
            "crop_offset": torch.cat([crop_offset, crop_offset], dim=0),
            "common_preamble_id": common_preamble_id + common_preamble_id,
            "tx_id": torch.cat([tx_id, tx_id], dim=0),
            "rx_i": torch.cat([rx_i, rx_i], dim=0),
            "day_i": torch.cat([day_i, day_i], dim=0),
            "view_type": ("clean",) * batch_size + (scenario,) * batch_size,
            "link_condition": link_condition + link_condition,
            "excitation_bin": torch.cat([excitation_bin, excitation_bin], dim=0),
            "eta_schema_version": FCR_V2_ETA_SCHEMA_VERSION,
            "eta": torch.cat([clean_eta, eta], dim=0),
            "eta_valid_mask": torch.cat([clean_valid, eta_valid_mask], dim=0),
        },
        batch_size=batch_size * 2,
    )
