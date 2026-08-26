"""Strict int8 aggregate deployment bundle for CVS-FSFA-V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .factored_slow_fast import FactoredSlowFastState


FACTORED_BUNDLE_SCHEMA = "cvs.factored_slow_fast.bundle.int8.v1"
_FIELDS = frozenset(
    {
        "schema",
        "candidate",
        "base_checkpoint_id",
        "class_ids",
        "ridge_receiver",
        "ridge_leo",
        "receiver_basis_q",
        "receiver_basis_scale",
        "leo_basis_q",
        "leo_basis_scale",
        "geometric_centers_q",
        "geometric_centers_scale",
    }
)


def _quantize(value: Tensor) -> tuple[Tensor, float]:
    maximum = float(value.detach().abs().max())
    scale = max(maximum / 127.0, 1.0e-8)
    quantized = torch.round(value.detach().cpu().float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _dequantize(value: object, scale: object, *, name: str) -> Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.int8 or value.ndim != 2:
        raise ValueError(f"{name} aggregate must be an int8 matrix")
    resolved_scale = float(scale)
    if not torch.isfinite(torch.tensor(resolved_scale)) or resolved_scale <= 0.0:
        raise ValueError(f"{name} scale must be finite and positive")
    return value.float() * resolved_scale


def save_factored_bundle(
    path: str | Path,
    state: FactoredSlowFastState,
    *,
    candidate: str,
    base_checkpoint_id: str,
) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"factored bundle already exists: {output}")
    if candidate not in {"B3", "B5"}:
        raise ValueError("factored deployment candidate must be B3 or B5")
    if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id.strip():
        raise ValueError("base_checkpoint_id must be nonempty")
    receiver_q, receiver_scale = _quantize(state.receiver_basis)
    leo_q, leo_scale = _quantize(state.leo_basis)
    centers_q, centers_scale = _quantize(state.geometric_centers)
    payload: dict[str, Any] = {
        "schema": FACTORED_BUNDLE_SCHEMA,
        "candidate": candidate,
        "base_checkpoint_id": base_checkpoint_id,
        "class_ids": state.class_ids.detach().cpu().long(),
        "ridge_receiver": state.ridge_receiver,
        "ridge_leo": state.ridge_leo,
        "receiver_basis_q": receiver_q,
        "receiver_basis_scale": receiver_scale,
        "leo_basis_q": leo_q,
        "leo_basis_scale": leo_scale,
        "geometric_centers_q": centers_q,
        "geometric_centers_scale": centers_scale,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_factored_bundle_strict(
    path: str | Path,
    *,
    decision_prototypes: Tensor,
) -> tuple[FactoredSlowFastState, dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"factored bundle is not a regular file: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(_FIELDS):
        raise ValueError("factored bundle field allowlist mismatch")
    if payload["schema"] != FACTORED_BUNDLE_SCHEMA or payload["candidate"] not in {"B3", "B5"}:
        raise ValueError("factored bundle schema/candidate mismatch")
    class_ids = payload["class_ids"]
    if not torch.is_tensor(class_ids) or class_ids.ndim != 1:
        raise ValueError("factored bundle class IDs are invalid")
    prototypes = decision_prototypes.detach().cpu().float()
    if prototypes.ndim != 2 or prototypes.shape[0] != class_ids.numel():
        raise ValueError("external decision prototypes do not align with factored bundle")
    state = FactoredSlowFastState(
        receiver_basis=_dequantize(payload["receiver_basis_q"], payload["receiver_basis_scale"], name="receiver_basis"),
        leo_basis=_dequantize(payload["leo_basis_q"], payload["leo_basis_scale"], name="leo_basis"),
        geometric_centers=_dequantize(payload["geometric_centers_q"], payload["geometric_centers_scale"], name="geometric_centers"),
        decision_prototypes=prototypes,
        class_ids=class_ids,
        ridge_receiver=float(payload["ridge_receiver"]),
        ridge_leo=float(payload["ridge_leo"]),
    )
    return state, {
        "schema": FACTORED_BUNDLE_SCHEMA,
        "candidate": payload["candidate"],
        "base_checkpoint_id": payload["base_checkpoint_id"],
        "aggregate_storage_dtype": "int8",
        "source_samples_representable": False,
        "fast_parameter_count": state.fast_parameter_count,
    }


__all__ = ["FACTORED_BUNDLE_SCHEMA", "load_factored_bundle_strict", "save_factored_bundle"]
