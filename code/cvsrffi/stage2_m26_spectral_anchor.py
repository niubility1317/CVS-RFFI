"""Checkpoint-bound source-only spectral anchors for ERBT-IDR M2.6.

The persisted component contains exactly six class-aggregated identity160 and
FFT96 centres.  It exposes no sample, member, raw-IQ, query, or update API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.erbt_idr.m26.phase1_spectral_anchor.v1"
FEATURE_SCHEMA = "ADV3B02:z_id160+fft_logmag96:source_only_class_aggregate:v1"
IDENTITY_DIM = 160
FFT_DIM = 96
ENVELOPE_DIM = 32
RIPPLE_DIM = 64
GEOMETRY_DIM = 96
GEOMETRY_BLOCK_DIM = 32
TREND_DCT_DIM = 8
CLASS_COUNT = 6
CHECKPOINT_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EPS = 1.0e-12
_MEMBERS = {
    "schema",
    "feature_schema",
    "checkpoint_sha256",
    "class_registry",
    "identity_q",
    "identity_scale",
    "fft_q",
    "fft_scale",
}


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or not np.isfinite(rows).all():
        raise ValueError("feature rows must be finite and nonempty")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise ValueError("feature rows must be nondegenerate")
    return rows / norm


def _cauchy_center(value: Any) -> np.ndarray:
    rows = _unit_rows(value)
    centre = np.mean(rows, axis=0)
    centre /= max(float(np.linalg.norm(centre)), _EPS)
    for _ in range(12):
        residual = np.linalg.norm(rows - centre[None, :], axis=1)
        positive = residual[residual > _EPS]
        scale = float(np.median(positive)) if len(positive) else 1.0
        weight = 1.0 / (1.0 + np.square(residual / max(2.3849 * scale, _EPS)))
        updated = np.sum(weight[:, None] * rows, axis=0) / max(float(np.sum(weight)), _EPS)
        updated /= max(float(np.linalg.norm(updated)), _EPS)
        if float(np.linalg.norm(updated - centre)) <= 1.0e-8:
            centre = updated
            break
        centre = updated
    return centre.astype(np.float32)


def _dct_matrix() -> np.ndarray:
    sample = np.arange(FFT_DIM, dtype=np.float64)[None, :]
    order = np.arange(FFT_DIM, dtype=np.float64)[:, None]
    matrix = np.sqrt(2.0 / FFT_DIM) * np.cos(
        np.pi * (sample + 0.5) * order / FFT_DIM
    )
    matrix[0] /= np.sqrt(2.0)
    return matrix


_DCT96 = _dct_matrix()


def fft_envelope_ripple(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split normalized FFT96 into orthogonal low/high-order blocks."""

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FFT_DIM or not np.isfinite(rows).all():
        raise ValueError("FFT descriptor must be finite N x 96")
    coefficient = _unit_rows(rows) @ _DCT96.T
    envelope = _unit_rows(coefficient[:, :ENVELOPE_DIM]).astype(np.float32)
    ripple = _unit_rows(coefficient[:, ENVELOPE_DIM:]).astype(np.float32)
    return envelope, ripple


def _safe_unit_rows(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or not np.isfinite(rows).all():
        raise ValueError("spectral geometry rows must be finite and nonempty")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, np.maximum(norm, _EPS), out=np.zeros_like(rows), where=norm > _EPS)


def _pool_fft96_to_32(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FFT_DIM:
        raise ValueError("FFT96 pooling requires N x 96 rows")
    return rows.reshape(rows.shape[0], GEOMETRY_BLOCK_DIM, FFT_DIM // GEOMETRY_BLOCK_DIM).mean(axis=2)


def fft_magnitude_geometry(value: Any) -> np.ndarray:
    """Build a 96-D magnitude-geometry view from the frozen FFT96 row.

    The three equal blocks retain high-order trend residual, local slope, and
    exact fftshift mirror asymmetry.  The transform consumes only the same
    deterministic FFT96 observation and therefore does not create another
    physical support view or increase K.
    """

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FFT_DIM or not np.isfinite(rows).all():
        raise ValueError("FFT descriptor must be finite N x 96")
    normalized = _unit_rows(rows)
    coefficient = normalized @ _DCT96.T
    smooth = coefficient[:, :TREND_DCT_DIM] @ _DCT96[:TREND_DCT_DIM]
    residual = normalized - smooth
    slope = np.diff(residual, axis=1, prepend=residual[:, :1])
    mirror_indices = np.remainder(-np.arange(FFT_DIM), FFT_DIM)
    mirror_asymmetry = residual - residual[:, mirror_indices]
    joined = np.concatenate(
        [
            _safe_unit_rows(_pool_fft96_to_32(residual)),
            _safe_unit_rows(_pool_fft96_to_32(slope)),
            _safe_unit_rows(_pool_fft96_to_32(mirror_asymmetry)),
        ],
        axis=1,
    )
    if np.any(np.linalg.norm(joined, axis=1) <= _EPS):
        raise ValueError("FFT magnitude geometry is degenerate")
    return _unit_rows(joined).astype(np.float32)


def _quantize(value: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(value, dtype=np.float32)
    maximum = np.max(np.abs(rows), axis=1)
    scale32 = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or np.any(scale16 <= 0.0):
        raise ValueError("anchor quantization scale is invalid")
    quantized = np.clip(np.rint(rows / scale32[:, None]), -127, 127).astype(np.int8)
    if np.any(quantized == -128):
        raise ValueError("anchor quantization emitted forbidden -128")
    return quantized, scale16


@dataclass(frozen=True)
class Phase1SpectralAnchor:
    class_registry: tuple[str, ...]
    checkpoint_sha256: str
    identity_q: np.ndarray
    identity_scale: np.ndarray
    fft_q: np.ndarray
    fft_scale: np.ndarray

    def __post_init__(self) -> None:
        classes = tuple(str(item) for item in self.class_registry)
        checkpoint = str(self.checkpoint_sha256).lower()
        identity_q = np.asarray(self.identity_q)
        identity_scale = np.asarray(self.identity_scale)
        fft_q = np.asarray(self.fft_q)
        fft_scale = np.asarray(self.fft_scale)
        if (
            len(classes) != CLASS_COUNT
            or len(set(classes)) != CLASS_COUNT
            or any(not item for item in classes)
            or CHECKPOINT_SHA256_PATTERN.fullmatch(checkpoint) is None
            or identity_q.dtype != np.int8
            or identity_q.shape != (CLASS_COUNT, IDENTITY_DIM)
            or identity_scale.dtype != np.float16
            or identity_scale.shape != (CLASS_COUNT,)
            or fft_q.dtype != np.int8
            or fft_q.shape != (CLASS_COUNT, FFT_DIM)
            or fft_scale.dtype != np.float16
            or fft_scale.shape != (CLASS_COUNT,)
            or np.any(identity_q == -128)
            or np.any(fft_q == -128)
            or not np.isfinite(identity_scale).all()
            or not np.isfinite(fft_scale).all()
            or np.any(identity_scale <= 0.0)
            or np.any(fft_scale <= 0.0)
        ):
            raise ValueError("M2.6 Phase1 spectral anchor state drift")
        object.__setattr__(self, "class_registry", classes)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(self, "identity_q", _readonly(identity_q, np.int8))
        object.__setattr__(self, "identity_scale", _readonly(identity_scale, np.float16))
        object.__setattr__(self, "fft_q", _readonly(fft_q, np.int8))
        object.__setattr__(self, "fft_scale", _readonly(fft_scale, np.float16))

    @property
    def state_bytes(self) -> int:
        return int(
            self.identity_q.nbytes
            + self.identity_scale.nbytes
            + self.fft_q.nbytes
            + self.fft_scale.nbytes
        )

    @property
    def component_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "feature_schema": FEATURE_SCHEMA,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "class_registry": list(self.class_registry),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name in ("identity_q", "identity_scale", "fft_q", "fft_scale"):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(
                json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
            )
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def centres(self) -> np.ndarray:
        identity = self.identity_q.astype(np.float32) * self.identity_scale.astype(np.float32)[:, None]
        fft = self.fft_q.astype(np.float32) * self.fft_scale.astype(np.float32)[:, None]
        return np.concatenate([_unit_rows(identity), _unit_rows(fft)], axis=1).astype(np.float32)


def build_phase1_spectral_anchor(
    source_features: Any,
    source_labels: Any,
    *,
    class_registry: Sequence[str],
    checkpoint_sha256: str,
    dataset_roles: Any | None = None,
) -> tuple[Phase1SpectralAnchor, Mapping[str, Any]]:
    rows = np.asarray(source_features, dtype=np.float64)
    labels = np.asarray(source_labels).astype(str)
    classes = tuple(str(item) for item in class_registry)
    checkpoint = str(checkpoint_sha256).lower()
    if (
        rows.ndim != 2
        or rows.shape[1] != IDENTITY_DIM + FFT_DIM
        or len(rows) != len(labels)
        or not np.isfinite(rows).all()
        or len(classes) != CLASS_COUNT
        or len(set(classes)) != CLASS_COUNT
        or CHECKPOINT_SHA256_PATTERN.fullmatch(checkpoint) is None
        or set(labels.tolist()) != set(classes)
    ):
        raise ValueError("source anchor feature/registry/checkpoint drift")
    if dataset_roles is not None:
        roles = np.asarray(dataset_roles).astype(str)
        if roles.shape != (len(rows),) or set(roles.tolist()) != {"source"}:
            raise ValueError("M2.6 Phase1 anchor accepts source-only rows")
    counts = np.asarray([np.sum(labels == name) for name in classes], dtype=np.int64)
    if np.any(counts <= 0) or len(set(counts.tolist())) != 1:
        raise ValueError("source anchor rows must be class-symmetric")
    identity = np.stack([_cauchy_center(rows[labels == name, :IDENTITY_DIM]) for name in classes])
    fft = np.stack([_cauchy_center(rows[labels == name, IDENTITY_DIM:]) for name in classes])
    identity_q, identity_scale = _quantize(identity)
    fft_q, fft_scale = _quantize(fft)
    component = Phase1SpectralAnchor(
        class_registry=classes,
        checkpoint_sha256=checkpoint,
        identity_q=identity_q,
        identity_scale=identity_scale,
        fft_q=fft_q,
        fft_scale=fft_scale,
    )
    audit = {
        "schema": "cvs.erbt_idr.m26.phase1_spectral_anchor_build_audit.v1",
        "source_row_count": int(len(rows)),
        "class_count": CLASS_COUNT,
        "rows_per_class": int(counts[0]),
        "feature_dim": IDENTITY_DIM + FFT_DIM,
        "persisted_member_or_sample_count": 0,
        "raw_iq_persisted": False,
        "query_rows_used": 0,
        "source_only": True,
        "state_bytes": component.state_bytes,
        "component_id": component.component_id,
    }
    return component, audit


def publish_m26_spectral_anchor(path: str | Path, component: Phase1SpectralAnchor) -> None:
    target = Path(path).absolute()
    if not target.parent.is_dir():
        raise FileNotFoundError("spectral anchor destination parent is missing")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez(
                stream,
                schema=np.asarray(SCHEMA),
                feature_schema=np.asarray(FEATURE_SCHEMA),
                checkpoint_sha256=np.asarray(component.checkpoint_sha256),
                class_registry=np.asarray(component.class_registry),
                identity_q=component.identity_q,
                identity_scale=component.identity_scale,
                fft_q=component.fft_q,
                fft_scale=component.fft_scale,
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def load_m26_spectral_anchor(
    path: str | Path, *, expected_checkpoint_sha256: str | None = None
) -> Phase1SpectralAnchor:
    source = Path(path).absolute()
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != _MEMBERS:
            raise ValueError("M2.6 spectral anchor member allowlist drift")
        if (
            _scalar_string(arrays["schema"], "schema") != SCHEMA
            or _scalar_string(arrays["feature_schema"], "feature_schema") != FEATURE_SCHEMA
        ):
            raise ValueError("M2.6 spectral anchor schema drift")
        component = Phase1SpectralAnchor(
            class_registry=tuple(np.asarray(arrays["class_registry"]).astype(str).tolist()),
            checkpoint_sha256=_scalar_string(arrays["checkpoint_sha256"], "checkpoint_sha256"),
            identity_q=np.array(arrays["identity_q"], copy=True),
            identity_scale=np.array(arrays["identity_scale"], copy=True),
            fft_q=np.array(arrays["fft_q"], copy=True),
            fft_scale=np.array(arrays["fft_scale"], copy=True),
        )
    if expected_checkpoint_sha256 is not None and component.checkpoint_sha256 != str(expected_checkpoint_sha256).lower():
        raise ValueError("M2.6 spectral anchor checkpoint binding drift")
    return component


__all__ = [
    "CHECKPOINT_SHA256_PATTERN",
    "ENVELOPE_DIM",
    "FEATURE_SCHEMA",
    "FFT_DIM",
    "GEOMETRY_BLOCK_DIM",
    "GEOMETRY_DIM",
    "IDENTITY_DIM",
    "Phase1SpectralAnchor",
    "RIPPLE_DIM",
    "SCHEMA",
    "build_phase1_spectral_anchor",
    "fft_envelope_ripple",
    "fft_magnitude_geometry",
    "load_m26_spectral_anchor",
    "publish_m26_spectral_anchor",
]
