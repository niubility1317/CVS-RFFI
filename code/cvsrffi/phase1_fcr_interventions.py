from __future__ import annotations

from dataclasses import dataclass
import hmac
import os
from typing import Any, Mapping, Sequence

import torch

from cvsrffi.phase1_fcr_types import FCRPairBatch


_TX_IDENTITY_KEYS = ("true_tx_i", "tx_i", "tx", "transmitter_id", "tx_id")
_OPAQUE_TOKEN_KEY = os.urandom(32)


def build_physical_sample_id(meta: Mapping[str, Any]) -> str:
    """Build the stable physical identity used only where TX labels are visible."""

    required = ("tx_i", "rx_i", "day_i", "eq_i", "sig_i")
    missing = [key for key in required if key not in meta]
    if missing:
        raise KeyError(f"physical sample metadata missing {missing}")
    return "tx{tx_i}:rx{rx_i}:day{day_i}:eq{eq_i}:sig{sig_i}".format(**meta)


def _opaque_sample_id(full_id: str) -> str:
    return "sample:" + hmac.new(_OPAQUE_TOKEN_KEY, full_id.encode("utf-8"), "sha256").hexdigest()


def sanitize_fcr_meta(meta: Mapping[str, Any], label_visible: bool = False) -> dict[str, Any]:
    """Return FCR metadata without a reversible hidden TX identity.

    Visible roles retain the physical record identity needed for strict
    fingerprint intervention.  Hidden roles retain only an opaque physical
    token for clean/LEO synchronization; no TX identity field is exposed.
    """

    clean = dict(meta)
    visible = bool(label_visible)
    full_id = str(clean.get("physical_sample_id") or build_physical_sample_id(clean))
    opaque_id = clean.pop("_fcr_opaque_physical_id", None)
    clean["label_visible"] = visible
    if visible:
        clean["physical_sample_id"] = full_id
        return clean
    for key in _TX_IDENTITY_KEYS:
        clean.pop(key, None)
    clean.pop("base_index", None)
    clean["physical_sample_id"] = str(opaque_id or _opaque_sample_id(full_id))
    if "content_record_id" in clean:
        # Content-record identity is needed for same-record synchronization but
        # must not retain the reversible TX-bearing ManySig identifier in U_s.
        clean["content_record_id"] = clean["physical_sample_id"]
    return clean


@dataclass(frozen=True)
class InterventionCapability:
    nuisance_pair: bool
    content_pair: bool
    fingerprint_pair: bool
    reason: dict[str, str]


def invalid_indices(batch_size: int, device) -> torch.Tensor:
    return torch.full((int(batch_size),), -1, dtype=torch.long, device=device)


def _column(meta: Any, key: str, batch_size: int, default: Any = None) -> list[Any]:
    if isinstance(meta, Mapping):
        value = meta.get(key, default)
    elif isinstance(meta, Sequence) and all(isinstance(item, Mapping) for item in meta):
        return [item.get(key, default) for item in meta]
    elif hasattr(meta, key):
        value = getattr(meta, key)
    else:
        value = default
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (tuple, list)):
        if len(value) == int(batch_size):
            return list(value)
        if len(value) == 1:
            return list(value) * int(batch_size)
        return [default] * int(batch_size)
    return [value] * int(batch_size)


def _long_column(meta: Any, key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    values = _column(meta, key, batch_size, -1)
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            parsed.append(-1)
    return torch.tensor(parsed, dtype=torch.long, device=device)


def _first_strict_match(keys: list[tuple[Any, ...]], disallow_same_label: torch.Tensor | None = None) -> torch.Tensor:
    indices = invalid_indices(len(keys), disallow_same_label.device if disallow_same_label is not None else "cpu")
    for i, key in enumerate(keys):
        if any(value is None or value == "" for value in key):
            continue
        for j, candidate in enumerate(keys):
            if i == j or candidate != key:
                continue
            if disallow_same_label is not None and int(disallow_same_label[i]) == int(disallow_same_label[j]):
                continue
            indices[i] = int(j)
            break
    return indices


class InterventionCubeBatchBuilder:
    """Build strict FCR intervention indices without label-derived fallback."""

    def __init__(self) -> None:
        self.capability = InterventionCapability(False, False, False, {})

    def build(
        self,
        clean_iq: torch.Tensor,
        leo_view: Any,
        labels: torch.Tensor,
        domains: torch.Tensor | None,
        batch_meta: Any,
    ) -> FCRPairBatch:
        if not torch.is_tensor(clean_iq) or clean_iq.ndim < 2:
            raise ValueError("clean_iq must be a batched IQ tensor")
        leo_iq = getattr(leo_view, "x", leo_view)
        if not torch.is_tensor(leo_iq) or tuple(leo_iq.shape) != tuple(clean_iq.shape):
            raise ValueError("leo_iq must match clean_iq shape exactly")
        batch_size = int(clean_iq.size(0))
        device = clean_iq.device
        labels = labels.to(device=device, dtype=torch.long).reshape(-1)
        if int(labels.numel()) != batch_size:
            raise ValueError("labels must have one entry per IQ sample")
        receiver_id = _long_column(batch_meta, "rx_i", batch_size, device)
        day_id = _long_column(batch_meta, "day_i", batch_size, device)
        clean_ids = tuple(str(value or "") for value in _column(batch_meta, "physical_sample_id", batch_size, ""))
        clean_crop = _long_column(batch_meta, "crop_offset", batch_size, device)
        leo_ids = tuple(str(value or "") for value in _column(leo_view, "physical_sample_id", batch_size, ""))
        leo_crop = getattr(leo_view, "crop_offset", None)
        if leo_crop is None:
            leo_crop_t = invalid_indices(batch_size, device)
        elif torch.is_tensor(leo_crop):
            leo_crop_t = leo_crop.to(device=device, dtype=torch.long).reshape(-1)
        else:
            leo_crop_t = torch.tensor(list(leo_crop), dtype=torch.long, device=device).reshape(-1)
        if int(leo_crop_t.numel()) != batch_size:
            leo_crop_t = invalid_indices(batch_size, device)
        label_visible = torch.tensor(
            [bool(value) for value in _column(batch_meta, "label_visible", batch_size, False)],
            dtype=torch.bool,
            device=device,
        )
        label_mask = (labels >= 0) & label_visible

        applied = bool(getattr(leo_view, "applied", False))
        raw_nuisance_valid = getattr(leo_view, "nuisance_valid", None)
        if raw_nuisance_valid is None:
            nuisance_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            nuisance_valid = torch.as_tensor(
                raw_nuisance_valid,
                device=device,
                dtype=torch.bool,
            ).reshape(-1)
        if int(nuisance_valid.numel()) != batch_size:
            nuisance_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        nuisance = getattr(leo_view, "nuisance", None)
        if nuisance is None:
            nuisance = clean_iq.new_zeros((batch_size, 0))
        else:
            nuisance = torch.as_tensor(nuisance, device=device, dtype=clean_iq.dtype)
        nuisance_index = invalid_indices(batch_size, device)
        synchronized = torch.tensor(
            [bool(applied and clean_ids[i] and clean_ids[i] == leo_ids[i]) for i in range(batch_size)],
            dtype=torch.bool,
            device=device,
        ) & (clean_crop == leo_crop_t)
        nuisance_index[synchronized] = torch.arange(batch_size, device=device)[synchronized]

        content_records = _column(batch_meta, "content_record_id", batch_size, None)
        content_keys: list[tuple[Any, ...]] = []
        for index, record in enumerate(content_records):
            if not clean_ids[index] or record in (None, "") or int(clean_crop[index]) < 0:
                content_keys.append((None,))
            else:
                content_keys.append((str(record),))
        content_index = invalid_indices(batch_size, device)
        for i, key in enumerate(content_keys):
            if key == (None,):
                continue
            for j, candidate in enumerate(content_keys):
                if i != j and candidate == key and int(clean_crop[i]) != int(clean_crop[j]):
                    content_index[i] = int(j)
                    break

        preambles = _column(batch_meta, "common_preamble_id", batch_size, None)
        view_types = _column(batch_meta, "view_type", batch_size, None)
        links = _column(batch_meta, "link_condition", batch_size, None)
        excitation = _column(batch_meta, "excitation_bin", batch_size, None)
        fingerprint_keys = [
            (preambles[i], int(receiver_id[i]), int(day_id[i]), view_types[i], links[i], excitation[i])
            if bool(label_mask[i]) and int(receiver_id[i]) >= 0 and int(day_id[i]) >= 0
            else (None,)
            for i in range(batch_size)
        ]
        fingerprint_index = _first_strict_match(fingerprint_keys, labels)
        fingerprint_index = fingerprint_index.to(device=device)

        reasons = {
            "nuisance": "available" if bool(synchronized.any()) else "missing_synchronized_generated_leo_view",
            "content": (
                "available"
                if bool((content_index >= 0).any())
                else "missing_content_window_metadata"
                if all(key == (None,) for key in content_keys)
                else "no_distinct_window_for_same_physical_record"
            ),
            "fingerprint": (
                "available"
                if bool((fingerprint_index >= 0).any())
                else "missing_common_preamble_metadata"
                if any(value in (None, "") for value in preambles)
                else "no_matched_different_tx_common_preamble_pair"
            ),
        }
        self.capability = InterventionCapability(
            nuisance_pair=bool((nuisance_index >= 0).any()),
            content_pair=bool((content_index >= 0).any()),
            fingerprint_pair=bool((fingerprint_index >= 0).any()),
            reason=reasons,
        )
        return FCRPairBatch(
            clean_iq=clean_iq,
            leo_iq=leo_iq.to(device=device, dtype=clean_iq.dtype),
            labels=labels,
            label_mask=label_mask,
            receiver_id=receiver_id,
            day_id=day_id,
            nuisance=nuisance,
            nuisance_valid=nuisance_valid,
            physical_sample_id=clean_ids,
            pair_id=clean_ids,
            clean_crop_offset=clean_crop,
            leo_crop_offset=leo_crop_t,
            nuisance_pair_index=nuisance_index,
            content_pair_index=content_index,
            fingerprint_pair_index=fingerprint_index,
            pair_valid_mask={
                "nuisance": nuisance_index >= 0,
                "content": content_index >= 0,
                "fingerprint": fingerprint_index >= 0,
            },
        )
