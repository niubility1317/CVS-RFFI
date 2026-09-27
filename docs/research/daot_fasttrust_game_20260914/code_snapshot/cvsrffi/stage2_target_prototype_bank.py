"""Compact append-only target-support prototype banks for Phase2.

The bank is built only from registered support features and opaque class
handles.  It has no query-side API.  FP32 and FP16 banks store prototype
vectors directly; INT8 banks use one symmetric scale per vector.  Radius and
support count use the same representation for every storage format so that
storage and accuracy comparisons remain paired.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
from typing import Any, Sequence

import numpy as np


SCHEMA = "cvs.phase2.target_prototype_bank.v1"
RADIUS_DEFINITION = "unit_l2_rms_chord_k1_locked_r0"
STORAGE_FORMATS = ("fp32", "fp16", "int8")
OPAQUE_CLASS_RE = re.compile(r"cls_[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
EPS = 1.0e-8
MAX_UINT16 = int(np.iinfo(np.uint16).max)


class TargetPrototypeBankError(ValueError):
    """Raised when support input or an encoded bank fails closed."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
        array.shape
    )
    immutable.setflags(write=False)
    return immutable


def _opaque_classes(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if (
        not classes
        or len(set(classes)) != len(classes)
        or any(OPAQUE_CLASS_RE.fullmatch(value) is None for value in classes)
    ):
        raise TargetPrototypeBankError("opaque class registry drift")
    return classes


def _normalized_rows(
    value: np.ndarray,
    *,
    require_unit: bool,
    expected_rows: int | None = None,
) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] < 1
        or (expected_rows is not None and len(rows) != int(expected_rows))
        or not np.isfinite(rows).all()
    ):
        raise TargetPrototypeBankError("prototype feature matrix drift")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norms <= EPS)):
        raise TargetPrototypeBankError("prototype features must be non-zero")
    if require_unit and not np.allclose(norms[:, 0], 1.0, atol=1.0e-5):
        raise TargetPrototypeBankError("vectors must already be unit normalized")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _validated_radius(value: Sequence[float], class_count: int) -> np.ndarray:
    radius32 = np.asarray(value, dtype=np.float32)
    if (
        radius32.shape != (class_count,)
        or not np.isfinite(radius32).all()
        or bool(np.any(radius32 < 0.0))
        or bool(np.any(radius32 > 2.0))
    ):
        raise TargetPrototypeBankError("prototype radius drift")
    radius16 = radius32.astype(np.float16)
    if not np.isfinite(radius16).all():
        raise TargetPrototypeBankError("prototype radius is not FP16 representable")
    return np.ascontiguousarray(radius16)


def _validated_count(value: Sequence[int], class_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != (class_count,)
        or raw.dtype.kind not in "iu"
        or bool(np.any(raw < 1))
        or bool(np.any(raw > MAX_UINT16))
        or len(set(int(item) for item in raw.tolist())) != 1
    ):
        raise TargetPrototypeBankError(
            "support count must be one class-symmetric physical K-shot"
        )
    return np.ascontiguousarray(raw, dtype=np.uint16)


def _validate_r0(value: float) -> float:
    r0 = float(value)
    if not math.isfinite(r0) or not 0.0 <= r0 <= 2.0:
        raise TargetPrototypeBankError("locked K1 radius r0 is out of range")
    if not math.isfinite(float(np.float16(r0))):
        raise TargetPrototypeBankError("locked K1 radius r0 is not FP16 representable")
    return r0


def _encode_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(vectors), axis=1)
    scale = (max_abs / np.float32(127.0)).astype(np.float16)
    if bool(np.any(scale <= 0.0)) or not np.isfinite(scale).all():
        raise TargetPrototypeBankError("INT8 per-vector scale drift")
    persisted_scale = scale.astype(np.float32)
    quantized = np.rint(vectors / persisted_scale[:, None])
    quantized = np.clip(quantized, -127.0, 127.0).astype(np.int8)
    if bool(np.any(quantized == -128)) or bool(
        np.any(np.all(quantized == 0, axis=1))
    ):
        raise TargetPrototypeBankError("INT8 prototype payload drift")
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scale)


def _prefix_sha256(
    *,
    storage_format: str,
    vectors: np.ndarray | None,
    q: np.ndarray | None,
    scale: np.ndarray | None,
    radius: np.ndarray,
    count: np.ndarray,
    classes: tuple[str, ...],
    old_class_count: int,
) -> str:
    old_count = int(old_class_count)
    digest = hashlib.sha256()
    digest.update(b"cvs.phase2.target_prototype_bank.old_prefix.v1\0")
    format_bytes = storage_format.encode("ascii")
    digest.update(struct.pack("<B", len(format_bytes)))
    digest.update(format_bytes)
    digest.update(struct.pack("<IH", old_count, int(radius.ndim)))
    for class_handle in classes[:old_count]:
        encoded = class_handle.encode("ascii")
        digest.update(struct.pack("<H", len(encoded)))
        digest.update(encoded)
    payloads: tuple[tuple[bytes, np.ndarray], ...]
    if storage_format == "int8":
        assert q is not None and scale is not None
        payloads = ((b"q", q[:old_count]), (b"scale", scale[:old_count]))
    else:
        assert vectors is not None
        payloads = ((b"vectors", vectors[:old_count]),)
    payloads += ((b"radius", radius[:old_count]), (b"count", count[:old_count]))
    for name, array in payloads:
        contiguous = np.ascontiguousarray(array)
        digest.update(struct.pack("<B", len(name)))
        digest.update(name)
        digest.update(struct.pack("<B", contiguous.ndim))
        for dimension in contiguous.shape:
            digest.update(struct.pack("<I", int(dimension)))
        dtype = contiguous.dtype.str.encode("ascii")
        digest.update(struct.pack("<B", len(dtype)))
        digest.update(dtype)
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetPrototypeBank:
    """Immutable target-prototype state with a byte-locked old prefix."""

    schema: str
    storage_format: str
    classes: tuple[str, ...]
    old_class_count: int
    radius: np.ndarray
    count: np.ndarray
    old_prefix_sha256: str
    vectors: np.ndarray | None = None
    q: np.ndarray | None = None
    scale: np.ndarray | None = None

    def __post_init__(self) -> None:
        storage_format = str(self.storage_format)
        classes = _opaque_classes(self.classes)
        old_count = int(self.old_class_count)
        radius = np.asarray(self.radius)
        count = np.asarray(self.count)
        vectors = None if self.vectors is None else np.asarray(self.vectors)
        q = None if self.q is None else np.asarray(self.q)
        scale = None if self.scale is None else np.asarray(self.scale)
        if (
            self.schema != SCHEMA
            or storage_format not in STORAGE_FORMATS
            or isinstance(self.old_class_count, bool)
            or not 1 <= old_count <= len(classes)
            or radius.dtype != np.float16
            or radius.shape != (len(classes),)
            or count.dtype != np.uint16
            or count.shape != (len(classes),)
            or not np.isfinite(radius).all()
            or bool(np.any(radius < 0.0))
            or bool(np.any(radius > 2.0))
            or bool(np.any(count < 1))
            or len(set(int(item) for item in count.tolist())) != 1
        ):
            raise TargetPrototypeBankError("target prototype bank state drift")
        if storage_format == "int8":
            if (
                vectors is not None
                or q is None
                or q.dtype != np.int8
                or q.ndim != 2
                or q.shape[0] != len(classes)
                or q.shape[1] < 1
                or bool(np.any(q == -128))
                or bool(np.any(np.all(q == 0, axis=1)))
                or scale is None
                or scale.dtype != np.float16
                or scale.shape != (len(classes),)
                or not np.isfinite(scale).all()
                or bool(np.any(scale <= 0.0))
            ):
                raise TargetPrototypeBankError("INT8 target prototype bank drift")
        else:
            expected_dtype = np.float32 if storage_format == "fp32" else np.float16
            if (
                vectors is None
                or vectors.dtype != expected_dtype
                or vectors.ndim != 2
                or vectors.shape[0] != len(classes)
                or vectors.shape[1] < 1
                or not np.isfinite(vectors).all()
                or bool(np.any(np.linalg.norm(vectors.astype(np.float32), axis=1) <= EPS))
                or q is not None
                or scale is not None
            ):
                raise TargetPrototypeBankError(
                    f"{storage_format.upper()} target prototype bank drift"
                )
        expected_prefix = _prefix_sha256(
            storage_format=storage_format,
            vectors=vectors,
            q=q,
            scale=scale,
            radius=radius,
            count=count,
            classes=classes,
            old_class_count=old_count,
        )
        if (
            SHA256_RE.fullmatch(str(self.old_prefix_sha256)) is None
            or str(self.old_prefix_sha256) != expected_prefix
        ):
            raise TargetPrototypeBankError("old prototype prefix SHA256 drift")
        object.__setattr__(self, "storage_format", storage_format)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "radius", _readonly(radius, np.float16))
        object.__setattr__(self, "count", _readonly(count, np.uint16))
        object.__setattr__(self, "old_prefix_sha256", expected_prefix)
        if storage_format == "int8":
            assert q is not None and scale is not None
            object.__setattr__(self, "q", _readonly(q, np.int8))
            object.__setattr__(self, "scale", _readonly(scale, np.float16))
            object.__setattr__(self, "vectors", None)
        else:
            assert vectors is not None
            dtype = np.float32 if storage_format == "fp32" else np.float16
            object.__setattr__(self, "vectors", _readonly(vectors, dtype))
            object.__setattr__(self, "q", None)
            object.__setattr__(self, "scale", None)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def feature_dim(self) -> int:
        value = self.q if self.storage_format == "int8" else self.vectors
        assert value is not None
        return int(value.shape[1])

    def decoded_vectors(self) -> np.ndarray:
        """Return a transient FP32 reconstruction of every class vector."""

        if self.storage_format == "int8":
            assert self.q is not None and self.scale is not None
            value = self.q.astype(np.float32) * self.scale.astype(np.float32)[:, None]
        else:
            assert self.vectors is not None
            value = self.vectors.astype(np.float32)
        return _readonly(value, np.float32)

    @property
    def array_state_bytes(self) -> int:
        payload = (
            (self.q, self.scale)
            if self.storage_format == "int8"
            else (self.vectors,)
        )
        return int(
            sum(value.nbytes for value in payload if value is not None)
            + self.radius.nbytes
            + self.count.nbytes
        )

    @property
    def metadata_state_bytes(self) -> int:
        # One format tag, uint32 old count, raw SHA256, and uint16-prefixed
        # opaque handles.  Python/container overhead is deliberately excluded.
        return int(
            1
            + np.dtype(np.uint32).itemsize
            + hashlib.sha256().digest_size
            + sum(2 + len(value.encode("ascii")) for value in self.classes)
        )

    @property
    def logical_state_bytes(self) -> int:
        return self.array_state_bytes + self.metadata_state_bytes

    def storage_audit(
        self, reference_vectors: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Compare actual state with paired FP32, FP16, and INT8 formats."""

        common = int(
            self.radius.nbytes + self.count.nbytes + self.metadata_state_bytes
        )
        vector_elements = self.class_count * self.feature_dim
        by_format = {
            "fp32": common + vector_elements * np.dtype(np.float32).itemsize,
            "fp16": common + vector_elements * np.dtype(np.float16).itemsize,
            "int8": common
            + vector_elements * np.dtype(np.int8).itemsize
            + self.class_count * np.dtype(np.float16).itemsize,
        }
        audit: dict[str, Any] = {
            "schema": "cvs.phase2.target_prototype_bank.resource.v1",
            "storage_format": self.storage_format,
            "class_count": self.class_count,
            "old_class_count": self.old_class_count,
            "feature_dim": self.feature_dim,
            "radius_definition": RADIUS_DEFINITION,
            "array_state_bytes": self.array_state_bytes,
            "metadata_state_bytes": self.metadata_state_bytes,
            "logical_state_bytes": self.logical_state_bytes,
            "state_bytes_by_format": by_format,
            "actual_matches_format_bytes": (
                self.logical_state_bytes == by_format[self.storage_format]
            ),
            "compression_ratio_vs_fp32": float(
                by_format["fp32"] / self.logical_state_bytes
            ),
            "compression_ratio_vs_fp16": float(
                by_format["fp16"] / self.logical_state_bytes
            ),
            "mixed_fp32_prototype_macs_per_scored_row": int(vector_elements),
            "prototype_rescale_multiplies_per_scored_row": (
                self.class_count if self.storage_format == "int8" else 0
            ),
            "persistent_fp32_prototype_vectors": (
                self.class_count if self.storage_format == "fp32" else 0
            ),
            "int8_negative_128_present": bool(
                self.q is not None and np.any(self.q == -128)
            ),
            "old_prefix_sha256": self.old_prefix_sha256,
            "quantization_error_available": reference_vectors is not None,
        }
        if reference_vectors is not None:
            reference = _normalized_rows(
                reference_vectors,
                require_unit=True,
                expected_rows=self.class_count,
            )
            candidates = {
                "fp32": reference,
                "fp16": reference.astype(np.float16).astype(np.float32),
            }
            q, scale = _encode_int8(reference)
            candidates["int8"] = q.astype(np.float32) * scale.astype(np.float32)[:, None]
            audit["error_by_format"] = {
                name: _reconstruction_error(reference, value)
                for name, value in candidates.items()
            }
            audit["actual_error"] = _reconstruction_error(
                reference, self.decoded_vectors()
            )
        return audit


def _reconstruction_error(
    reference: np.ndarray, reconstructed: np.ndarray
) -> dict[str, float]:
    value = np.asarray(reconstructed, dtype=np.float32)
    delta = value - reference
    norms = np.linalg.norm(value, axis=1)
    cosine = np.sum(reference * value, axis=1) / np.maximum(norms, EPS)
    return {
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_l2_error": float(np.mean(np.linalg.norm(delta, axis=1))),
        "max_cosine_distance": float(np.max(1.0 - cosine)),
    }


def _build_bank(
    vectors: np.ndarray,
    classes: Sequence[str],
    *,
    radius: Sequence[float],
    count: Sequence[int],
    storage_format: str,
    old_class_count: int,
    old_prefix_sha256: str | None = None,
) -> TargetPrototypeBank:
    class_tuple = _opaque_classes(classes)
    normalized = _normalized_rows(
        vectors, require_unit=True, expected_rows=len(class_tuple)
    )
    radius16 = _validated_radius(radius, len(class_tuple))
    count16 = _validated_count(count, len(class_tuple))
    if int(count16[0]) == 1 and not np.all(radius16 == radius16[0]):
        raise TargetPrototypeBankError("K1 radius must use one locked r0")
    storage = str(storage_format)
    if storage not in STORAGE_FORMATS:
        raise TargetPrototypeBankError("unsupported prototype storage format")
    payload: dict[str, np.ndarray | None]
    if storage == "int8":
        q, scale = _encode_int8(normalized)
        payload = {"vectors": None, "q": q, "scale": scale}
    else:
        dtype = np.float32 if storage == "fp32" else np.float16
        payload = {
            "vectors": np.ascontiguousarray(normalized, dtype=dtype),
            "q": None,
            "scale": None,
        }
    prefix = _prefix_sha256(
        storage_format=storage,
        vectors=payload["vectors"],
        q=payload["q"],
        scale=payload["scale"],
        radius=radius16,
        count=count16,
        classes=class_tuple,
        old_class_count=int(old_class_count),
    )
    if old_prefix_sha256 is not None and prefix != str(old_prefix_sha256):
        raise TargetPrototypeBankError("append changed the locked old prefix")
    return TargetPrototypeBank(
        schema=SCHEMA,
        storage_format=storage,
        classes=class_tuple,
        old_class_count=int(old_class_count),
        radius=radius16,
        count=count16,
        old_prefix_sha256=prefix,
        vectors=payload["vectors"],
        q=payload["q"],
        scale=payload["scale"],
    )


def encode_normalized_vectors(
    vectors: np.ndarray,
    classes: Sequence[str],
    *,
    radius: Sequence[float],
    count: Sequence[int],
    storage_format: str,
    old_class_count: int | None = None,
) -> TargetPrototypeBank:
    """Encode already unit-normalized class vectors with supplied metadata."""

    class_tuple = _opaque_classes(classes)
    old_count = len(class_tuple) if old_class_count is None else int(old_class_count)
    return _build_bank(
        vectors,
        class_tuple,
        radius=radius,
        count=count,
        storage_format=storage_format,
        old_class_count=old_count,
    )


def _support_prototypes_and_radius(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    *,
    r0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_tuple = _opaque_classes(classes)
    features = _normalized_rows(support_features, require_unit=False)
    labels = np.asarray(tuple(str(value) for value in support_labels))
    if labels.ndim != 1 or len(labels) != len(features):
        raise TargetPrototypeBankError("support features and labels must align")
    if set(labels.tolist()) != set(class_tuple):
        raise TargetPrototypeBankError("support class registry drift")
    counts = np.asarray(
        [int(np.sum(labels == label)) for label in class_tuple], dtype=np.int64
    )
    count16 = _validated_count(counts, len(class_tuple))
    locked_r0 = _validate_r0(r0)
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    for class_handle, count in zip(class_tuple, counts.tolist()):
        rows = features[labels == class_handle]
        prototype = np.mean(rows, axis=0, dtype=np.float64).astype(np.float32)
        prototype = _normalized_rows(prototype[None, :], require_unit=False)[0]
        prototypes.append(prototype)
        if int(count) == 1:
            radii.append(locked_r0)
        else:
            squared = np.sum((rows - prototype[None, :]) ** 2, axis=1)
            radii.append(float(np.sqrt(np.mean(squared, dtype=np.float64))))
    return (
        np.ascontiguousarray(np.stack(prototypes), dtype=np.float32),
        np.ascontiguousarray(radii, dtype=np.float32),
        count16,
    )


def encode_support_prototypes(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    *,
    storage_format: str,
    r0: float,
) -> TargetPrototypeBank:
    """Build an old-class bank from registered physical support only."""

    class_tuple = _opaque_classes(classes)
    vectors, radius, count = _support_prototypes_and_radius(
        support_features, support_labels, class_tuple, r0=r0
    )
    return _build_bank(
        vectors,
        class_tuple,
        radius=radius,
        count=count,
        storage_format=storage_format,
        old_class_count=len(class_tuple),
    )


def _append_bank(
    bank: TargetPrototypeBank,
    new_vectors: np.ndarray,
    new_classes: Sequence[str],
    *,
    new_radius: Sequence[float],
    new_count: Sequence[int],
) -> TargetPrototypeBank:
    if not isinstance(bank, TargetPrototypeBank):
        raise TargetPrototypeBankError("valid target prototype bank required")
    class_tuple = _opaque_classes(new_classes)
    if set(class_tuple) & set(bank.classes):
        raise TargetPrototypeBankError("new classes overlap registered classes")
    normalized = _normalized_rows(
        new_vectors, require_unit=True, expected_rows=len(class_tuple)
    )
    radius16 = _validated_radius(new_radius, len(class_tuple))
    count16 = _validated_count(new_count, len(class_tuple))
    if (
        len(set(int(item) for item in bank.count.tolist())) != 1
        or int(count16[0]) != int(bank.count[0])
    ):
        raise TargetPrototypeBankError("append K-shot differs from locked bank")
    if int(bank.count[0]) == 1 and not np.all(radius16 == bank.radius[0]):
        raise TargetPrototypeBankError("append changed the locked K1 radius r0")
    all_classes = bank.classes + class_tuple
    all_radius = np.concatenate([bank.radius, radius16])
    all_count = np.concatenate([bank.count, count16])
    if bank.storage_format == "int8":
        assert bank.q is not None and bank.scale is not None
        suffix_q, suffix_scale = _encode_int8(normalized)
        q = np.concatenate([bank.q, suffix_q], axis=0)
        scale = np.concatenate([bank.scale, suffix_scale])
        prefix = _prefix_sha256(
            storage_format=bank.storage_format,
            vectors=None,
            q=q,
            scale=scale,
            radius=all_radius,
            count=all_count,
            classes=all_classes,
            old_class_count=bank.old_class_count,
        )
        return TargetPrototypeBank(
            schema=SCHEMA,
            storage_format=bank.storage_format,
            classes=all_classes,
            old_class_count=bank.old_class_count,
            radius=all_radius,
            count=all_count,
            old_prefix_sha256=prefix,
            q=q,
            scale=scale,
        )
    assert bank.vectors is not None
    dtype = np.float32 if bank.storage_format == "fp32" else np.float16
    vectors = np.concatenate(
        [bank.vectors, normalized.astype(dtype)], axis=0
    ).astype(dtype, copy=False)
    prefix = _prefix_sha256(
        storage_format=bank.storage_format,
        vectors=vectors,
        q=None,
        scale=None,
        radius=all_radius,
        count=all_count,
        classes=all_classes,
        old_class_count=bank.old_class_count,
    )
    return TargetPrototypeBank(
        schema=SCHEMA,
        storage_format=bank.storage_format,
        classes=all_classes,
        old_class_count=bank.old_class_count,
        radius=all_radius,
        count=all_count,
        old_prefix_sha256=prefix,
        vectors=vectors,
    )


def append_normalized_vectors(
    bank: TargetPrototypeBank,
    new_vectors: np.ndarray,
    new_classes: Sequence[str],
    *,
    radius: Sequence[float],
    count: Sequence[int],
) -> TargetPrototypeBank:
    """Append new normalized vectors without rewriting the old prefix."""

    result = _append_bank(
        bank,
        new_vectors,
        new_classes,
        new_radius=radius,
        new_count=count,
    )
    if result.old_prefix_sha256 != bank.old_prefix_sha256:
        raise TargetPrototypeBankError("old prefix changed during append")
    return result


def append_support_prototypes(
    bank: TargetPrototypeBank,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    r0: float,
) -> TargetPrototypeBank:
    """Append new-class support prototypes while freezing all old bytes."""

    class_tuple = _opaque_classes(new_classes)
    vectors, radius, count = _support_prototypes_and_radius(
        support_features, support_labels, class_tuple, r0=r0
    )
    result = _append_bank(
        bank,
        vectors,
        class_tuple,
        new_radius=radius,
        new_count=count,
    )
    if result.old_prefix_sha256 != bank.old_prefix_sha256:
        raise TargetPrototypeBankError("old prefix changed during registration")
    return result


def score_mixed_fp32(
    bank: TargetPrototypeBank, normalized_features: np.ndarray
) -> np.ndarray:
    """Score FP32 rows against every stored class with one shared API."""

    if not isinstance(bank, TargetPrototypeBank):
        raise TargetPrototypeBankError("valid target prototype bank required")
    raw = np.asarray(normalized_features, dtype=np.float32)
    one_row = raw.ndim == 1
    if one_row:
        raw = raw[None, :]
    rows = _normalized_rows(raw, require_unit=True)
    if rows.shape[1] != bank.feature_dim:
        raise TargetPrototypeBankError("score feature dimension drift")
    if bank.storage_format == "int8":
        assert bank.q is not None and bank.scale is not None
        scores = (rows @ bank.q.astype(np.float32).T) * bank.scale.astype(
            np.float32
        )[None, :]
    else:
        assert bank.vectors is not None
        scores = rows @ bank.vectors.astype(np.float32).T
    result = np.ascontiguousarray(scores, dtype=np.float32)
    return _readonly(result[0] if one_row else result, np.float32)


__all__ = [
    "RADIUS_DEFINITION",
    "SCHEMA",
    "STORAGE_FORMATS",
    "TargetPrototypeBank",
    "TargetPrototypeBankError",
    "append_normalized_vectors",
    "append_support_prototypes",
    "encode_normalized_vectors",
    "encode_support_prototypes",
    "score_mixed_fp32",
]
