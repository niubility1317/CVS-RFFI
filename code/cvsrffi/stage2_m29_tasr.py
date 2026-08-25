"""Target-aware spectral residual representation for ERBT-IDR M2.9.

Phase1 builds one checkpoint-bound aggregate from source-only FFT96 rows.
Phase2 estimates one frozen target spectral shift from labelled support and
applies it to support/query rows without exposing any query update surface.
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


SCHEMA = "cvs.erbt_idr.m29.phase1_tasr_bundle.v2"
FEATURE_SCHEMA = "ADV3B02:fft_logmag96:source_receiver_class_int8_aggregate:tasr48:v2"
FFT_DIM = 96
TASR_DIM = 48
DEFAULT_RANK = 8
CHECKPOINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_EPS = 1.0e-12
_MEMBERS = {
    "schema",
    "feature_schema",
    "checkpoint_sha256",
    "class_registry",
    "receiver_registry",
    "rank",
    "global_mean_q",
    "global_mean_scale",
    "basis_q",
    "basis_scale",
    "eigenvalues_q",
    "eigenvalues_scale",
    "tasr_location_q",
    "tasr_location_scale",
    "tasr_scale_q",
    "tasr_scale_scale",
}


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _rows(value: Any, *, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 0
        or rows.shape[1] != int(dimension)
        or not np.isfinite(rows).all()
    ):
        raise ValueError(f"{name} must be finite N x {dimension}")
    return rows


def _quantize_vector(value: Any) -> tuple[np.ndarray, np.float16]:
    vector = np.asarray(value, dtype=np.float32)
    maximum = float(np.max(np.abs(vector)))
    scale32 = np.float32(max(maximum / 127.0, np.finfo(np.float16).tiny))
    scale16 = np.float16(scale32)
    quantized = np.clip(np.rint(vector / np.float32(scale16)), -127, 127).astype(np.int8)
    if not np.isfinite(scale16) or scale16 <= 0 or np.any(quantized == -128):
        raise ValueError("TASR vector quantization drift")
    return quantized, scale16


def _quantize_columns(value: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float32)
    columns: list[np.ndarray] = []
    scales: list[np.float16] = []
    for index in range(matrix.shape[1]):
        code, scale = _quantize_vector(matrix[:, index])
        columns.append(code)
        scales.append(scale)
    return np.stack(columns, axis=1), np.asarray(scales, dtype=np.float16)


def _dequantize_vector(code: np.ndarray, scale: np.float16) -> np.ndarray:
    return code.astype(np.float32) * np.float32(scale)


def _dequantize_columns(code: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return code.astype(np.float32) * scale.astype(np.float32)[None, :]


def _pool_mean(rows: np.ndarray, count: int) -> np.ndarray:
    return np.stack(
        [np.stack([part.mean() for part in np.array_split(row, count)]) for row in rows]
    )


def _pool_rms(rows: np.ndarray, count: int) -> np.ndarray:
    return np.stack(
        [
            np.stack([np.sqrt(np.mean(np.square(part))) for part in np.array_split(row, count)])
            for row in rows
        ]
    )


def tasr48_raw(value: Any) -> np.ndarray:
    """Return the fixed 16+16+8+8 TASR descriptor before frozen scaling."""

    rows = _rows(value, dimension=FFT_DIM, name="FFT96")
    padded = np.pad(rows, ((0, 0), (4, 4)), mode="reflect")
    smooth = np.stack(
        [np.convolve(row, np.ones(9, dtype=np.float64) / 9.0, mode="valid") for row in padded]
    )
    residual = rows - smooth
    first = np.diff(rows, axis=1)
    second = np.diff(first, axis=1)
    result = np.concatenate(
        [
            _pool_mean(residual, 16),
            _pool_rms(residual, 16),
            _pool_rms(first, 8),
            _pool_rms(second, 8),
        ],
        axis=1,
    )
    if result.shape != (len(rows), TASR_DIM) or not np.isfinite(result).all():
        raise ValueError("TASR48 fixed pooling drift")
    return result.astype(np.float32)


@dataclass(frozen=True)
class Phase1TASRBundle:
    class_registry: tuple[str, ...]
    receiver_registry: tuple[str, ...]
    checkpoint_sha256: str
    rank: int
    global_mean_q: np.ndarray
    global_mean_scale: np.float16
    basis_q: np.ndarray
    basis_scale: np.ndarray
    eigenvalues_q: np.ndarray
    eigenvalues_scale: np.float16
    tasr_location_q: np.ndarray
    tasr_location_scale: np.float16
    tasr_scale_q: np.ndarray
    tasr_scale_scale: np.float16

    def __post_init__(self) -> None:
        classes = tuple(str(item) for item in self.class_registry)
        receivers = tuple(str(item) for item in self.receiver_registry)
        checkpoint = str(self.checkpoint_sha256).lower()
        rank = int(self.rank)
        arrays_ok = (
            np.asarray(self.global_mean_q).dtype == np.int8
            and np.asarray(self.global_mean_q).shape == (FFT_DIM,)
            and np.asarray(self.basis_q).dtype == np.int8
            and np.asarray(self.basis_q).shape == (FFT_DIM, rank)
            and np.asarray(self.basis_scale).dtype == np.float16
            and np.asarray(self.basis_scale).shape == (rank,)
            and np.asarray(self.eigenvalues_q).dtype == np.int8
            and np.asarray(self.eigenvalues_q).shape == (rank,)
            and np.asarray(self.eigenvalues_scale).dtype == np.float16
            and np.asarray(self.eigenvalues_scale).shape == ()
            and np.asarray(self.tasr_location_q).dtype == np.int8
            and np.asarray(self.tasr_location_q).shape == (TASR_DIM,)
            and np.asarray(self.tasr_scale_q).dtype == np.int8
            and np.asarray(self.tasr_scale_q).shape == (TASR_DIM,)
        )
        positive = np.concatenate(
            [
                np.asarray(self.basis_scale, dtype=np.float32),
                np.asarray([self.global_mean_scale, self.eigenvalues_scale, self.tasr_location_scale, self.tasr_scale_scale], dtype=np.float32),
            ]
        )
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or len(receivers) < 2
            or len(set(receivers)) != len(receivers)
            or any(not item for item in classes + receivers)
            or CHECKPOINT_PATTERN.fullmatch(checkpoint) is None
            or rank <= 0
            or rank > FFT_DIM
            or not arrays_ok
            or not np.isfinite(positive).all()
            or np.any(positive <= 0.0)
            or np.any(np.asarray(self.global_mean_q) == -128)
            or np.any(np.asarray(self.basis_q) == -128)
            or np.any(np.asarray(self.eigenvalues_q) <= 0)
            or np.any(np.asarray(self.tasr_location_q) == -128)
            or np.any(np.asarray(self.tasr_scale_q) <= 0)
        ):
            raise ValueError("Phase1 TASR bundle state drift")
        object.__setattr__(self, "class_registry", classes)
        object.__setattr__(self, "receiver_registry", receivers)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(self, "rank", rank)
        for name, dtype in (
            ("global_mean_q", np.int8),
            ("basis_q", np.int8),
            ("basis_scale", np.float16),
            ("eigenvalues_q", np.int8),
            ("tasr_location_q", np.int8),
            ("tasr_scale_q", np.int8),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        object.__setattr__(self, "eigenvalues_scale", np.float16(self.eigenvalues_scale))

    def global_mean(self) -> np.ndarray:
        return _dequantize_vector(self.global_mean_q, self.global_mean_scale)

    def basis(self) -> np.ndarray:
        value = _dequantize_columns(self.basis_q, self.basis_scale)
        # Restore the orthonormal basis after storage quantization.
        q, _ = np.linalg.qr(value.astype(np.float64))
        return q[:, : self.rank].astype(np.float32)

    def eigenvalues(self) -> np.ndarray:
        return (
            self.eigenvalues_q.astype(np.float32)
            * np.float32(self.eigenvalues_scale)
        )

    def tasr_location(self) -> np.ndarray:
        return _dequantize_vector(self.tasr_location_q, self.tasr_location_scale)

    def tasr_scale(self) -> np.ndarray:
        value = _dequantize_vector(self.tasr_scale_q, self.tasr_scale_scale)
        return np.maximum(value, np.float32(1.0e-6))

    @property
    def tau(self) -> float:
        return float(np.median(self.eigenvalues()))

    @property
    def state_bytes(self) -> int:
        return int(
            self.global_mean_q.nbytes
            + np.asarray(self.global_mean_scale).nbytes
            + self.basis_q.nbytes
            + self.basis_scale.nbytes
            + self.eigenvalues_q.nbytes
            + np.asarray(self.eigenvalues_scale).nbytes
            + self.tasr_location_q.nbytes
            + np.asarray(self.tasr_location_scale).nbytes
            + self.tasr_scale_q.nbytes
            + np.asarray(self.tasr_scale_scale).nbytes
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
                    "receiver_registry": list(self.receiver_registry),
                    "rank": self.rank,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name in (
            "global_mean_q",
            "global_mean_scale",
            "basis_q",
            "basis_scale",
            "eigenvalues_q",
            "eigenvalues_scale",
            "tasr_location_q",
            "tasr_location_scale",
            "tasr_scale_q",
            "tasr_scale_scale",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()


def build_phase1_tasr_bundle(
    fft96: Any,
    labels: Any,
    receivers: Any,
    *,
    class_registry: Sequence[str],
    checkpoint_sha256: str,
    dataset_roles: Any | None = None,
    rank: int = DEFAULT_RANK,
) -> tuple[Phase1TASRBundle, Mapping[str, Any]]:
    rows = _rows(fft96, dimension=FFT_DIM, name="source FFT96")
    y = np.asarray(labels).astype(str)
    d = np.asarray(receivers).astype(str)
    classes = tuple(str(item) for item in class_registry)
    domains = tuple(sorted(set(d.tolist())))
    checkpoint = str(checkpoint_sha256).lower()
    requested_rank = int(rank)
    if (
        y.shape != (len(rows),)
        or d.shape != (len(rows),)
        or set(y.tolist()) != set(classes)
        or len(classes) < 2
        or len(set(classes)) != len(classes)
        or len(domains) < 2
        or CHECKPOINT_PATTERN.fullmatch(checkpoint) is None
        or requested_rank <= 0
        or requested_rank > FFT_DIM
    ):
        raise ValueError("source receiver/class/checkpoint geometry drift")
    if dataset_roles is not None:
        roles = np.asarray(dataset_roles).astype(str)
        if roles.shape != (len(rows),) or set(roles.tolist()) != {"source"}:
            raise ValueError("Phase1 TASR bundle accepts source-only rows")
    cells = np.empty((len(classes), len(domains), FFT_DIM), dtype=np.float64)
    for class_index, name in enumerate(classes):
        for domain_index, domain in enumerate(domains):
            mask = (y == name) & (d == domain)
            if not np.any(mask):
                raise ValueError("source receiver-class cell is missing")
            cells[class_index, domain_index] = rows[mask].mean(axis=0)
    class_centres = cells.mean(axis=1)
    residual = cells - class_centres[:, None, :]
    covariance = np.einsum("cdr,cde->re", residual, residual) / float(
        len(classes) * len(domains)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(eigenvalues)[::-1]
    retained_values = eigenvalues[order[:requested_rank]]
    basis = eigenvectors[:, order[:requested_rank]]
    if np.any(retained_values <= _EPS):
        raise ValueError("source receiver-class covariance has insufficient rank")
    # Fix eigenvector sign so row ordering/platform details do not change identity.
    for index in range(requested_rank):
        pivot = int(np.argmax(np.abs(basis[:, index])))
        if basis[pivot, index] < 0.0:
            basis[:, index] *= -1.0
    global_mean = class_centres.mean(axis=0)
    receiver_means = cells.mean(axis=0)
    receiver_delta = (receiver_means - global_mean[None, :]) @ basis @ basis.T
    corrected = rows - np.stack(
        [receiver_delta[domains.index(item)] for item in d.tolist()]
    )
    raw_tasr = tasr48_raw(corrected)
    location = np.median(raw_tasr, axis=0)
    scale = 1.4826 * np.median(np.abs(raw_tasr - location[None, :]), axis=0)
    positive = scale[scale > 1.0e-8]
    floor = max(float(np.median(positive)) * 1.0e-3 if len(positive) else 0.0, 1.0e-6)
    scale = np.maximum(scale, floor)
    global_q, global_scale = _quantize_vector(global_mean)
    basis_q, basis_scale = _quantize_columns(basis)
    eigenvalues_scale = np.float16(float(np.max(retained_values)) / 127.0)
    if not np.isfinite(eigenvalues_scale) or eigenvalues_scale <= 0.0:
        raise ValueError("source receiver-class eigenvalue quantization underflow")
    eigenvalues_q = np.clip(
        np.rint(retained_values / float(eigenvalues_scale)), 1, 127
    ).astype(np.int8)
    location_q, location_scale = _quantize_vector(location)
    scale_q, scale_scale = _quantize_vector(scale)
    bundle = Phase1TASRBundle(
        class_registry=classes,
        receiver_registry=domains,
        checkpoint_sha256=checkpoint,
        rank=requested_rank,
        global_mean_q=global_q,
        global_mean_scale=global_scale,
        basis_q=basis_q,
        basis_scale=basis_scale,
        eigenvalues_q=eigenvalues_q,
        eigenvalues_scale=eigenvalues_scale,
        tasr_location_q=location_q,
        tasr_location_scale=location_scale,
        tasr_scale_q=scale_q,
        tasr_scale_scale=scale_scale,
    )
    audit = {
        "schema": "cvs.erbt_idr.m29.phase1_tasr_build_audit.v1",
        "source_row_count": int(len(rows)),
        "class_count": len(classes),
        "receiver_count": len(domains),
        "receiver_class_cell_count": int(len(classes) * len(domains)),
        "rank": requested_rank,
        "tau": bundle.tau,
        "source_only": True,
        "target_rows_used": 0,
        "query_rows_used": 0,
        "persisted_member_or_sample_count": 0,
        "raw_iq_persisted": False,
        "state_bytes": bundle.state_bytes,
        "component_id": bundle.component_id,
    }
    return bundle, audit


@dataclass(frozen=True)
class TargetSpectralCalibration:
    delta: np.ndarray
    support_class_count: int
    support_k_shot: int
    shrinkage: np.ndarray
    query_rows_used: int = 0
    frozen: bool = True

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta, dtype=np.float32)
        shrinkage = np.asarray(self.shrinkage, dtype=np.float32)
        if (
            delta.shape != (FFT_DIM,)
            or shrinkage.ndim != 1
            or not np.isfinite(delta).all()
            or not np.isfinite(shrinkage).all()
            or np.any(shrinkage < 0.0)
            or np.any(shrinkage > 1.0)
            or int(self.support_class_count) < 2
            or int(self.support_k_shot) < 1
            or int(self.query_rows_used) != 0
            or self.frozen is not True
        ):
            raise ValueError("target spectral calibration state drift")
        object.__setattr__(self, "delta", _readonly(delta, np.float32))
        object.__setattr__(self, "shrinkage", _readonly(shrinkage, np.float32))


def estimate_target_spectral_calibration(
    support_fft96: Any,
    support_labels: Any,
    bundle: Phase1TASRBundle,
) -> TargetSpectralCalibration:
    rows = _rows(support_fft96, dimension=FFT_DIM, name="target support FFT96")
    labels = np.asarray(support_labels).astype(str)
    classes = tuple(sorted(set(labels.tolist())))
    if labels.shape != (len(rows),) or len(classes) < 2:
        raise ValueError("target support labels are incomplete")
    counts = np.asarray([np.sum(labels == name) for name in classes], dtype=np.int64)
    if np.any(counts <= 0) or len(set(counts.tolist())) != 1:
        raise ValueError("target support must be balanced K-shot")
    target_mean = np.stack([rows[labels == name].mean(axis=0) for name in classes]).mean(axis=0)
    basis = bundle.basis().astype(np.float64)
    eigenvalues = bundle.eigenvalues().astype(np.float64)
    shrinkage = eigenvalues / (eigenvalues + float(bundle.tau))
    difference = target_mean - bundle.global_mean().astype(np.float64)
    delta = basis @ (shrinkage * (basis.T @ difference))
    return TargetSpectralCalibration(
        delta=delta.astype(np.float32),
        support_class_count=len(classes),
        support_k_shot=int(counts[0]),
        shrinkage=shrinkage.astype(np.float32),
    )


def transform_tasr48(
    fft96: Any,
    calibration: TargetSpectralCalibration,
    bundle: Phase1TASRBundle,
) -> np.ndarray:
    rows = _rows(fft96, dimension=FFT_DIM, name="received FFT96")
    corrected = rows - calibration.delta.astype(np.float64)[None, :]
    raw = tasr48_raw(corrected).astype(np.float64)
    scaled = (raw - bundle.tasr_location().astype(np.float64)[None, :]) / bundle.tasr_scale().astype(np.float64)[None, :]
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    if np.any(norms <= _EPS) or not np.isfinite(norms).all():
        raise ValueError("TASR48 frozen scaling is degenerate")
    return (scaled / norms).astype(np.float32)


def publish_phase1_tasr_bundle(path: str | Path, bundle: Phase1TASRBundle) -> None:
    target = Path(path).absolute()
    if not target.parent.is_dir():
        raise FileNotFoundError("TASR bundle destination parent is missing")
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
                checkpoint_sha256=np.asarray(bundle.checkpoint_sha256),
                class_registry=np.asarray(bundle.class_registry),
                receiver_registry=np.asarray(bundle.receiver_registry),
                rank=np.asarray(bundle.rank, dtype=np.int64),
                global_mean_q=bundle.global_mean_q,
                global_mean_scale=np.asarray(bundle.global_mean_scale, dtype=np.float16),
                basis_q=bundle.basis_q,
                basis_scale=bundle.basis_scale,
                eigenvalues_q=bundle.eigenvalues_q,
                eigenvalues_scale=bundle.eigenvalues_scale,
                tasr_location_q=bundle.tasr_location_q,
                tasr_location_scale=np.asarray(bundle.tasr_location_scale, dtype=np.float16),
                tasr_scale_q=bundle.tasr_scale_q,
                tasr_scale_scale=np.asarray(bundle.tasr_scale_scale, dtype=np.float16),
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


def load_phase1_tasr_bundle(
    path: str | Path, *, expected_checkpoint_sha256: str | None = None
) -> Phase1TASRBundle:
    with np.load(Path(path).absolute(), allow_pickle=False) as arrays:
        if set(arrays.files) != _MEMBERS:
            raise ValueError("TASR bundle member allowlist drift")
        if (
            _scalar_string(arrays["schema"], "schema") != SCHEMA
            or _scalar_string(arrays["feature_schema"], "feature_schema") != FEATURE_SCHEMA
        ):
            raise ValueError("TASR bundle schema drift")
        bundle = Phase1TASRBundle(
            class_registry=tuple(np.asarray(arrays["class_registry"]).astype(str).tolist()),
            receiver_registry=tuple(np.asarray(arrays["receiver_registry"]).astype(str).tolist()),
            checkpoint_sha256=_scalar_string(arrays["checkpoint_sha256"], "checkpoint_sha256"),
            rank=int(np.asarray(arrays["rank"]).item()),
            global_mean_q=np.array(arrays["global_mean_q"], copy=True),
            global_mean_scale=np.float16(np.asarray(arrays["global_mean_scale"]).item()),
            basis_q=np.array(arrays["basis_q"], copy=True),
            basis_scale=np.array(arrays["basis_scale"], copy=True),
            eigenvalues_q=np.array(arrays["eigenvalues_q"], copy=True),
            eigenvalues_scale=np.float16(np.asarray(arrays["eigenvalues_scale"]).item()),
            tasr_location_q=np.array(arrays["tasr_location_q"], copy=True),
            tasr_location_scale=np.float16(np.asarray(arrays["tasr_location_scale"]).item()),
            tasr_scale_q=np.array(arrays["tasr_scale_q"], copy=True),
            tasr_scale_scale=np.float16(np.asarray(arrays["tasr_scale_scale"]).item()),
        )
    if (
        expected_checkpoint_sha256 is not None
        and bundle.checkpoint_sha256 != str(expected_checkpoint_sha256).lower()
    ):
        raise ValueError("TASR bundle checkpoint binding drift")
    return bundle


__all__ = [
    "CHECKPOINT_PATTERN",
    "DEFAULT_RANK",
    "FEATURE_SCHEMA",
    "FFT_DIM",
    "Phase1TASRBundle",
    "SCHEMA",
    "TASR_DIM",
    "TargetSpectralCalibration",
    "build_phase1_tasr_bundle",
    "estimate_target_spectral_calibration",
    "load_phase1_tasr_bundle",
    "publish_phase1_tasr_bundle",
    "tasr48_raw",
    "transform_tasr48",
]
