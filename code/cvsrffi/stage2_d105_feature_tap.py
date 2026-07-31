"""Exact one-pass D105 feature tap for the frozen ADV3B02 checkpoint.

The public entry point accepts only a reconstructed eval-mode model and fixed
received IQ. It executes the identity and domain backbones once each, captures
the input/output of ``joint_proj.0``, and returns the four tensors required by
D105: ``z_id``, ``z_dom``, ``hidden`` and ``pre_relu``. No labels, roles,
query truth, class counts, or mutable state are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn

from cvsrffi.dual_feature_forward import dual_feature_forward


Z_DIM = 160
HIDDEN_DIM = 320
SCHEMA = "cvs.phase2.d105.feature_tap.v1"


class D105FeatureTapError(ValueError):
    """Raised when the D105 one-pass feature contract drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _to_numpy(value: torch.Tensor, *, width: int, name: str) -> np.ndarray:
    tensor = value.detach().cpu().contiguous()
    if (
        tensor.dtype != torch.float32
        or tensor.ndim != 2
        or int(tensor.shape[1]) != int(width)
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise D105FeatureTapError(f"{name} must be finite float32 [N,{width}]")
    # N607's Torch 2.1 / NumPy 2.x C-API bridge can reject Tensor.numpy() even
    # for a valid finite float32 tensor.  Python values preserve every finite
    # float32 value exactly through the float64 intermediate and avoid that
    # cached ndarray-type identity entirely.
    array = np.asarray(tensor.tolist(), dtype=np.float32)
    if array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
        raise D105FeatureTapError(f"{name} tensor-to-array bridge output drift")
    return np.ascontiguousarray(array, dtype=np.float32)


def _tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    return bool(
        torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


@dataclass(frozen=True, slots=True)
class D105FeatureTapBatch:
    z_id: np.ndarray
    z_dom: np.ndarray
    hidden: np.ndarray
    pre_relu: np.ndarray
    receipt_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        arrays = {
            "z_id": (self.z_id, Z_DIM),
            "z_dom": (self.z_dom, Z_DIM),
            "hidden": (self.hidden, HIDDEN_DIM),
            "pre_relu": (self.pre_relu, Z_DIM),
        }
        row_counts: set[int] = set()
        for name, (value, width) in arrays.items():
            array = np.asarray(value)
            if (
                array.dtype != np.float32
                or array.ndim != 2
                or array.shape[1] != width
                or len(array) < 1
                or not np.isfinite(array).all()
            ):
                raise D105FeatureTapError(f"{name} array contract drift")
            row_counts.add(len(array))
            copied = np.ascontiguousarray(array, dtype=np.float32).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if self.schema != SCHEMA or len(row_counts) != 1:
            raise D105FeatureTapError("D105 feature tap schema/alignment drift")
        expected = _feature_receipt(self)
        if self.receipt_sha256 != expected:
            raise D105FeatureTapError("D105 feature tap receipt drift")


def _feature_receipt(batch: D105FeatureTapBatch) -> str:
    payload = {
        "schema": SCHEMA,
        "rows": len(batch.z_id),
        "arrays": {
            name: _array_receipt(np.asarray(getattr(batch, name)))
            for name in ("z_id", "z_dom", "hidden", "pre_relu")
        },
        "query_rows_used_for_fit": 0,
        "state_updates_from_query": 0,
        "execution_counts": {
            "id_backbone": 1,
            "dom_backbone": 1,
            "dom_enhancer": 1,
        },
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def extract_d105_feature_tap(
    model: nn.Module, received_iq: torch.Tensor
) -> D105FeatureTapBatch:
    """Return exact D105 taps from one same-IQ dual-backbone invocation."""

    if (
        not isinstance(model, nn.Module)
        or model.training
        or not torch.is_tensor(received_iq)
        or received_iq.dtype != torch.float32
        or received_iq.ndim != 3
        or int(received_iq.shape[0]) < 1
        or int(received_iq.shape[1]) != 2
        or not bool(torch.isfinite(received_iq).all().item())
    ):
        raise D105FeatureTapError(
            "D105 feature tap requires eval model and finite float32 [N,2,T] IQ"
        )
    try:
        joint = model.id_backbone.cls_head.joint_proj
        linear = joint[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise D105FeatureTapError("ADV3B02 joint_proj path is absent") from exc
    if (
        not isinstance(linear, nn.Linear)
        or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.weight.dtype != torch.float32
    ):
        raise D105FeatureTapError("ADV3B02 joint_proj.0 contract drift")

    captured: dict[str, torch.Tensor] = {}

    def capture_linear(
        _module: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if len(args) != 1 or not torch.is_tensor(args[0]):
            raise D105FeatureTapError("joint_proj.0 hidden input drift")
        captured["hidden"] = args[0].detach().clone()
        captured["pre_relu"] = output.detach().clone()

    def capture_joint(
        _module: nn.Module,
        _args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["joint"] = output.detach().clone()

    linear_hook = linear.register_forward_hook(capture_linear)
    joint_hook = joint.register_forward_hook(capture_joint)
    try:
        z_id_tensor, z_dom_tensor, _tx_logits = dual_feature_forward(
            model, received_iq
        )
    finally:
        linear_hook.remove()
        joint_hook.remove()

    if set(captured) != {"hidden", "pre_relu", "joint"}:
        raise D105FeatureTapError("D105 joint projection tap is incomplete")
    with torch.no_grad():
        recomputed_pre = linear(captured["hidden"])
        recomputed_zid = torch.relu(recomputed_pre)
    if (
        not _tensor_bytes_equal(captured["pre_relu"], recomputed_pre)
        or not _tensor_bytes_equal(captured["joint"], recomputed_zid)
        or not _tensor_bytes_equal(z_id_tensor, captured["joint"])
    ):
        raise D105FeatureTapError(
            "D105 tap is not byte-bound to joint_proj.0 and ReLU"
        )

    values = {
        "z_id": _to_numpy(z_id_tensor, width=Z_DIM, name="z_id"),
        "z_dom": _to_numpy(z_dom_tensor, width=Z_DIM, name="z_dom"),
        "hidden": _to_numpy(captured["hidden"], width=HIDDEN_DIM, name="hidden"),
        "pre_relu": _to_numpy(
            captured["pre_relu"], width=Z_DIM, name="pre_relu"
        ),
    }
    provisional = object.__new__(D105FeatureTapBatch)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "schema", SCHEMA)
    receipt = _feature_receipt(provisional)
    return D105FeatureTapBatch(**values, receipt_sha256=receipt)


__all__ = [
    "D105FeatureTapBatch",
    "D105FeatureTapError",
    "HIDDEN_DIM",
    "SCHEMA",
    "Z_DIM",
    "extract_d105_feature_tap",
]
