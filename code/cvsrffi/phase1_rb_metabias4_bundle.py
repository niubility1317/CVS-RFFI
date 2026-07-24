"""D102 Phase1 class-free RB-MetaBias4 aggregate bundle.

This module deliberately accepts only a Phase1 tap archive.  It emits a
checkpoint-bound INT8 aggregate component without class handles, receiver/day
names, physical IDs, or sample-level features.  The component is not formally
Phase2-authorized until an outer deployment bundle jointly seals it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA = "cvs.phase1.rb_metabias4.bundle.v1"
WIRE_PROFILE = "rb_metabias4_u32_b4_classfree_int8_v1"
NPZ_NAME = "phase1_rb_metabias4_bundle.npz"
MANIFEST_NAME = "phase1_rb_metabias4_bundle.manifest.json"
SEAL_NAME = "phase1_rb_metabias4_bundle.seal.sha256"
Z_DIM = 160
DOMAIN_DIM = 32
RANK = 4
MIN_PHYSICAL_PER_CLASS_CELL = 2
MIN_CLASSES_PER_CELL = 2
EPS = 1.0e-12

PAYLOAD_MEMBERS = (
    "basis_codes_qint8",
    "basis_scales_fp16",
    "domain_encoder_codes_qint8",
    "domain_encoder_scales_fp16",
    "bank_g_codes_qint8",
    "bank_g_scales_fp16",
    "bank_t_codes_qint8",
    "bank_t_scales_fp16",
    "bank_precision_diag_fp16",
    "bank_sigma_fp16",
    "lambda0_diag_fp16",
    "amax_fp16",
)


class RBMetaBias4BundleError(ValueError):
    """Raised when the Phase1 aggregate or sealed component drifts."""


def _canon(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise RBMetaBias4BundleError(f"{name} must be lowercase SHA256")
    return result


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return _sha_bytes(
        _canon({"dtype": array.dtype.str, "shape": list(array.shape)})
        + b"\0"
        + array.tobytes()
    )


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise RBMetaBias4BundleError(f"{name} must be finite rows")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise RBMetaBias4BundleError(f"{name} contains zero rows")
    return np.asarray(rows / norms, dtype=np.float32)


def _quantize_rows(value: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise RBMetaBias4BundleError(f"{name} must be finite rows")
    maximum = np.max(np.abs(rows), axis=1)
    scale32 = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if np.any(scale16 <= 0.0) or not np.isfinite(scale16).all():
        raise RBMetaBias4BundleError(f"{name} FP16 scale closure failed")
    codes = np.clip(np.rint(rows / scale32[:, None]), -127, 127).astype(np.int8)
    decoded = codes.astype(np.float32) * scale16.astype(np.float32)[:, None]
    return codes, scale16, float(np.max(np.abs(decoded - rows)))


def _decode_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.asarray(codes, dtype=np.float32)
        * np.asarray(scales, dtype=np.float32)[:, None],
        dtype=np.float32,
    )


def _canonical_basis(rows: np.ndarray, width: int, rank: int, name: str) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all():
        raise RBMetaBias4BundleError(f"{name} shape drift")
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    basis: list[np.ndarray] = []
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * (
        float(singular[0]) if len(singular) else 1.0
    )
    for index, value in enumerate(singular):
        if value <= tolerance or len(basis) == rank:
            break
        row = np.array(vt[index], copy=True)
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
        basis.append(row)
    # A deterministic coordinate completion keeps the fixed tensor contract
    # when a held fold has fewer than rank independent source-domain shifts.
    for coordinate in range(width):
        if len(basis) == rank:
            break
        row = np.zeros(width, dtype=np.float64)
        row[coordinate] = 1.0
        for prior in basis:
            row -= np.dot(row, prior) * prior
        norm = float(np.linalg.norm(row))
        if norm > 1.0e-10:
            basis.append(row / norm)
    if len(basis) != rank:
        raise RBMetaBias4BundleError(f"{name} cannot provide rank {rank}")
    return np.asarray(basis, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class RBMetaBias4Config:
    temperature: float = 0.25
    lambda0_diag: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    amax: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    trust_radius: float = 0.35
    precision_floor: float = 0.05
    precision_ceiling: float = 20.0
    sigma_floor: float = 0.05
    deterministic_seed: int = 102

    def __post_init__(self) -> None:
        values = (
            self.temperature,
            self.trust_radius,
            self.precision_floor,
            self.precision_ceiling,
            self.sigma_floor,
            *self.lambda0_diag,
            *self.amax,
        )
        if (
            type(self.deterministic_seed) is not int
            or self.deterministic_seed < 0
            or any(not np.isfinite(value) or value <= 0.0 for value in values)
            or self.precision_floor > self.precision_ceiling
        ):
            raise RBMetaBias4BundleError("invalid frozen MetaBias4 config")

    @property
    def digest(self) -> str:
        return _sha_bytes(_canon({"schema": SCHEMA + ".config", **asdict(self)}))


@dataclass(frozen=True, slots=True)
class Phase1RBMetaBias4Bundle:
    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    domain_encoder_codes_qint8: np.ndarray
    domain_encoder_scales_fp16: np.ndarray
    bank_g_codes_qint8: np.ndarray
    bank_g_scales_fp16: np.ndarray
    bank_t_codes_qint8: np.ndarray
    bank_t_scales_fp16: np.ndarray
    bank_precision_diag_fp16: np.ndarray
    bank_sigma_fp16: np.ndarray
    lambda0_diag_fp16: np.ndarray
    amax_fp16: np.ndarray
    temperature: float
    trust_radius: float
    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    source_aggregate_digest_sha256: str
    config_digest_sha256: str
    aggregation_receipt: Mapping[str, Any]
    quantization_receipt: Mapping[str, Any]
    content_root_sha256: str = ""
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        arrays = {name: np.asarray(getattr(self, name)) for name in PAYLOAD_MEMBERS}
        m = int(arrays["bank_g_codes_qint8"].shape[0])
        expected = {
            "basis_codes_qint8": (np.int8, (RANK, Z_DIM)),
            "basis_scales_fp16": (np.float16, (RANK,)),
            "domain_encoder_codes_qint8": (np.int8, (DOMAIN_DIM, Z_DIM)),
            "domain_encoder_scales_fp16": (np.float16, (DOMAIN_DIM,)),
            "bank_g_codes_qint8": (np.int8, (m, DOMAIN_DIM)),
            "bank_g_scales_fp16": (np.float16, (m,)),
            "bank_t_codes_qint8": (np.int8, (m, RANK)),
            "bank_t_scales_fp16": (np.float16, (m,)),
            "bank_precision_diag_fp16": (np.float16, (m, RANK)),
            "bank_sigma_fp16": (np.float16, (m,)),
            "lambda0_diag_fp16": (np.float16, (RANK,)),
            "amax_fp16": (np.float16, (RANK,)),
        }
        if self.schema != SCHEMA or m < 2:
            raise RBMetaBias4BundleError("bundle schema/bank count drift")
        for name, (dtype, shape) in expected.items():
            array = arrays[name]
            if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
                raise RBMetaBias4BundleError(f"{name} dtype/shape/finite drift")
            if dtype == np.int8 and np.any(array == -128):
                raise RBMetaBias4BundleError(f"{name} contains forbidden -128")
        for name in (
            "basis_scales_fp16",
            "domain_encoder_scales_fp16",
            "bank_g_scales_fp16",
            "bank_t_scales_fp16",
            "bank_precision_diag_fp16",
            "bank_sigma_fp16",
            "lambda0_diag_fp16",
            "amax_fp16",
        ):
            if np.any(arrays[name] <= 0.0):
                raise RBMetaBias4BundleError(f"{name} must be positive")
        if (
            not np.isfinite(self.temperature)
            or self.temperature <= 0.0
            or not np.isfinite(self.trust_radius)
            or self.trust_radius <= 0.0
        ):
            raise RBMetaBias4BundleError("bundle scalar drift")
        for value, name in (
            (self.checkpoint_sha256, "checkpoint"),
            (self.runtime_sha256, "runtime"),
            (self.method_lock_sha256, "method lock"),
            (self.source_aggregate_digest_sha256, "source aggregate"),
            (self.config_digest_sha256, "config"),
        ):
            _sha(value, name)
        aggregation = dict(self.aggregation_receipt)
        quantization = dict(self.quantization_receipt)
        if (
            aggregation.get("all_observed_class_cells_ge2") is not True
            or int(aggregation.get("minimum_observed_class_cell_physical_count", 0)) < 2
            or aggregation.get("class_free_payload") is not True
            or quantization.get("persistent_fp32_sidecar") is not False
        ):
            raise RBMetaBias4BundleError("bundle receipt drift")
        for name, array in arrays.items():
            object.__setattr__(self, name, _readonly(array, expected[name][0]))
        object.__setattr__(self, "aggregation_receipt", MappingProxyType(aggregation))
        object.__setattr__(self, "quantization_receipt", MappingProxyType(quantization))
        expected_root = self._content_root()
        if self.content_root_sha256 and self.content_root_sha256 != expected_root:
            raise RBMetaBias4BundleError("bundle content root drift")
        object.__setattr__(self, "content_root_sha256", expected_root)

    def _content_root(self) -> str:
        return _sha_bytes(
            _canon(
                {
                    "schema": self.schema,
                    "profile": WIRE_PROFILE,
                    "arrays": {
                        name: _array_sha(np.asarray(getattr(self, name)))
                        for name in PAYLOAD_MEMBERS
                    },
                    "temperature": float(self.temperature),
                    "trust_radius": float(self.trust_radius),
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "runtime_sha256": self.runtime_sha256,
                    "method_lock_sha256": self.method_lock_sha256,
                    "source_aggregate_digest_sha256": self.source_aggregate_digest_sha256,
                    "config_digest_sha256": self.config_digest_sha256,
                    "aggregation_receipt": dict(self.aggregation_receipt),
                    "quantization_receipt": dict(self.quantization_receipt),
                }
            )
        )

    @property
    def bank_count(self) -> int:
        return int(self.bank_g_codes_qint8.shape[0])

    def basis(self) -> np.ndarray:
        result = _decode_rows(self.basis_codes_qint8, self.basis_scales_fp16).T
        result.setflags(write=False)
        return result

    def domain_encoder(self) -> np.ndarray:
        result = _decode_rows(
            self.domain_encoder_codes_qint8, self.domain_encoder_scales_fp16
        )
        result.setflags(write=False)
        return result

    def bank_g(self) -> np.ndarray:
        result = _unit_rows(
            _decode_rows(self.bank_g_codes_qint8, self.bank_g_scales_fp16), "bank_g"
        )
        result.setflags(write=False)
        return result

    def bank_t(self) -> np.ndarray:
        result = _decode_rows(self.bank_t_codes_qint8, self.bank_t_scales_fp16)
        result.setflags(write=False)
        return result

    @property
    def numeric_state_bytes(self) -> int:
        return int(sum(np.asarray(getattr(self, name)).nbytes for name in PAYLOAD_MEMBERS))


def _tap_arrays(tap_archive: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "pre_relu",
        "z_dom",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "class_ids",
    }
    if not isinstance(tap_archive, Mapping) or not required.issubset(tap_archive):
        raise RBMetaBias4BundleError("tap archive required member drift")
    result = {name: np.asarray(tap_archive[name]) for name in required}
    n = len(result["pre_relu"])
    if (
        result["pre_relu"].shape != (n, Z_DIM)
        or result["z_dom"].shape != (n, Z_DIM)
        or not np.isfinite(result["pre_relu"]).all()
        or not np.isfinite(result["z_dom"]).all()
        or any(result[name].shape != (n,) for name in ("labels", "receiver_ids", "day_ids", "physical_ids"))
        or result["class_ids"].ndim != 1
    ):
        raise RBMetaBias4BundleError("tap archive shape/finite drift")
    for name in ("labels", "receiver_ids", "day_ids", "physical_ids", "class_ids"):
        if result[name].dtype.kind not in {"U", "S"}:
            raise RBMetaBias4BundleError(f"{name} must be a non-object string array")
        result[name] = result[name].astype(str)
    if (
        n < 1
        or len(set(result["physical_ids"].tolist())) != n
        or set(result["labels"].tolist()) != set(result["class_ids"].tolist())
    ):
        raise RBMetaBias4BundleError("tap physical/class closure drift")
    return result


def merge_verified_phase1_tap_and_dual_archives(
    tap_archive: Mapping[str, np.ndarray],
    dual_archive: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Join byte-bound pre-ReLU taps with the same-row dual-feature archive."""

    tap_required = {
        "z_id",
        "pre_relu",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "class_ids",
    }
    dual_required = tap_required - {"pre_relu"} | {"z_dom"}
    if (
        not isinstance(tap_archive, Mapping)
        or not tap_required.issubset(tap_archive)
        or not isinstance(dual_archive, Mapping)
        or not dual_required.issubset(dual_archive)
    ):
        raise RBMetaBias4BundleError("tap/dual archive member closure drift")
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "class_ids",
    ):
        if not np.array_equal(np.asarray(tap_archive[name]), np.asarray(dual_archive[name])):
            raise RBMetaBias4BundleError(f"tap/dual row binding drift: {name}")
    tap_zid = np.asarray(tap_archive["z_id"])
    dual_zid = np.asarray(dual_archive["z_id"])
    if (
        tap_zid.dtype != np.float32
        or dual_zid.dtype != np.float32
        or tap_zid.shape != dual_zid.shape
        or tap_zid.ndim != 2
        or tap_zid.shape[1] != Z_DIM
        or not np.isfinite(tap_zid).all()
        or not np.isfinite(dual_zid).all()
    ):
        raise RBMetaBias4BundleError("tap/dual z_id contract drift")
    maximum = float(
        np.max(np.abs(tap_zid.astype(np.float64) - dual_zid.astype(np.float64)))
    )
    if not np.isfinite(maximum) or maximum > 1.0e-5:
        raise RBMetaBias4BundleError(
            f"tap/dual z_id parity failed: max_abs={maximum}"
        )
    merged = {
        "pre_relu": np.array(tap_archive["pre_relu"], copy=True),
        "z_dom": np.array(dual_archive["z_dom"], copy=True),
        "labels": np.array(dual_archive["labels"], copy=True),
        "receiver_ids": np.array(dual_archive["receiver_ids"], copy=True),
        "day_ids": np.array(dual_archive["day_ids"], copy=True),
        "physical_ids": np.array(dual_archive["physical_ids"], copy=True),
        "class_ids": np.array(dual_archive["class_ids"], copy=True),
    }
    _tap_arrays(merged)
    return merged


def _training_views(
    arrays: Mapping[str, np.ndarray],
    excluded_receivers: Sequence[str],
    excluded_classes: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[str, ...]]:
    excluded_rx = {str(value) for value in excluded_receivers}
    excluded_cls = {str(value) for value in excluded_classes}
    mask = np.asarray(
        [
            receiver not in excluded_rx and label not in excluded_cls
            for receiver, label in zip(arrays["receiver_ids"], arrays["labels"])
        ],
        dtype=bool,
    )
    selected = {name: np.asarray(value[mask]) for name, value in arrays.items() if name != "class_ids"}
    classes = tuple(sorted(set(selected["labels"].tolist())))
    cells = tuple(
        sorted(
            set(
                f"{receiver}\0{day}"
                for receiver, day in zip(selected["receiver_ids"], selected["day_ids"])
            )
        )
    )
    if len(classes) < 2 or len(cells) < 2:
        raise RBMetaBias4BundleError("held exclusions leave insufficient classes/domains")
    return selected, classes, cells


def _class_means(rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> dict[str, np.ndarray]:
    return {
        class_id: np.asarray(rows[labels == class_id].mean(axis=0), dtype=np.float64)
        for class_id in classes
    }


def build_phase1_rb_metabias4_bundle(
    tap_archive: Mapping[str, np.ndarray],
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    config: RBMetaBias4Config = RBMetaBias4Config(),
    excluded_receivers: Sequence[str] = (),
    excluded_classes: Sequence[str] = (),
) -> Phase1RBMetaBias4Bundle:
    """Build the deterministic source-only aggregate component."""

    config.__post_init__()
    arrays = _tap_arrays(tap_archive)
    train, classes, cells = _training_views(arrays, excluded_receivers, excluded_classes)
    labels = train["labels"]
    row_cells = np.asarray(
        [f"{receiver}\0{day}" for receiver, day in zip(train["receiver_ids"], train["day_ids"])],
        dtype=str,
    )
    class_pre = _class_means(train["pre_relu"], labels, classes)
    class_dom = _class_means(train["z_dom"], labels, classes)

    cell_pre: list[np.ndarray] = []
    cell_dom: list[np.ndarray] = []
    retained_cells: list[str] = []
    counts_by_cell: list[list[int]] = []
    for cell in cells:
        local_pre: list[np.ndarray] = []
        local_dom: list[np.ndarray] = []
        counts: list[int] = []
        for class_id in classes:
            indices = np.flatnonzero((row_cells == cell) & (labels == class_id))
            if len(indices) == 1:
                raise RBMetaBias4BundleError("observed class-cell must aggregate at least two physical samples")
            if len(indices) >= MIN_PHYSICAL_PER_CLASS_CELL:
                local_pre.append(train["pre_relu"][indices].mean(axis=0) - class_pre[class_id])
                local_dom.append(train["z_dom"][indices].mean(axis=0) - class_dom[class_id])
                counts.append(int(len(indices)))
        if len(local_pre) >= MIN_CLASSES_PER_CELL:
            retained_cells.append(cell)
            cell_pre.append(np.mean(local_pre, axis=0))
            cell_dom.append(np.mean(local_dom, axis=0))
            counts_by_cell.append(counts)
    if len(retained_cells) < 2:
        raise RBMetaBias4BundleError("class-free bank requires at least two valid domain cells")

    pre_shift = np.asarray(cell_pre, dtype=np.float32)
    dom_shift = np.asarray(cell_dom, dtype=np.float32)
    b_teacher = _canonical_basis(pre_shift, Z_DIM, RANK, "MetaBias B")
    u_teacher = _canonical_basis(dom_shift, Z_DIM, DOMAIN_DIM, "domain encoder U")
    b_q, b_s, b_err = _quantize_rows(b_teacher, "MetaBias B")
    u_q, u_s, u_err = _quantize_rows(u_teacher, "domain encoder U")
    b_decoded = _decode_rows(b_q, b_s)
    u_decoded = _decode_rows(u_q, u_s)

    g_rows: list[np.ndarray] = []
    t_rows: list[np.ndarray] = []
    precision_rows: list[np.ndarray] = []
    sigma_rows: list[float] = []
    for cell, pre_center, counts in zip(retained_cells, pre_shift, counts_by_cell):
        projected: list[np.ndarray] = []
        code_by_class: list[np.ndarray] = []
        for class_id in classes:
            indices = np.flatnonzero((row_cells == cell) & (labels == class_id))
            if len(indices) < MIN_PHYSICAL_PER_CLASS_CELL:
                continue
            dom_mean = train["z_dom"][indices].mean(axis=0) - class_dom[class_id]
            encoded = _unit_rows((u_decoded @ dom_mean)[None, :], "class-cell domain")[0]
            projected.append(encoded)
            pre_mean = train["pre_relu"][indices].mean(axis=0) - class_pre[class_id]
            code_by_class.append(-(b_decoded @ pre_mean))
        g = _unit_rows(np.mean(projected, axis=0, keepdims=True), "bank g")[0]
        class_codes = np.asarray(code_by_class, dtype=np.float64)
        t = np.mean(class_codes, axis=0)
        # The reciprocal ceiling is a fixed Phase1 variance floor.  Adding the
        # reciprocal *minimum* would collapse every bank item to the weakest
        # allowed precision and erase the intended continuous coverage signal.
        variance = np.var(class_codes, axis=0) + config.precision_ceiling ** -1
        precision = np.clip(
            1.0 / variance, config.precision_floor, config.precision_ceiling
        )
        angular = [max(0.0, 1.0 - float(np.dot(item, g))) for item in projected]
        sigma = max(config.sigma_floor, float(np.sqrt(np.mean(angular) + EPS)))
        g_rows.append(g)
        t_rows.append(np.asarray(t, dtype=np.float32))
        precision_rows.append(np.asarray(precision, dtype=np.float32))
        sigma_rows.append(sigma)

    g_teacher = np.asarray(g_rows, dtype=np.float32)
    t_teacher = np.asarray(t_rows, dtype=np.float32)
    g_q, g_s, g_err = _quantize_rows(g_teacher, "domain bank g")
    t_q, t_s, t_err = _quantize_rows(t_teacher, "domain bank t")
    precision16 = np.asarray(precision_rows, dtype=np.float16)
    sigma16 = np.asarray(sigma_rows, dtype=np.float16)
    lambda016 = np.asarray(config.lambda0_diag, dtype=np.float16)
    amax16 = np.asarray(config.amax, dtype=np.float16)
    if (
        np.any(precision16 <= 0.0)
        or np.any(sigma16 <= 0.0)
        or np.any(lambda016 <= 0.0)
        or np.any(amax16 <= 0.0)
    ):
        raise RBMetaBias4BundleError("FP16 positive state closure failed")

    source_digest = _sha_bytes(
        _canon(
            {
                "schema": SCHEMA + ".source_aggregate",
                "pre_relu_sha256": _array_sha(train["pre_relu"]),
                "z_dom_sha256": _array_sha(train["z_dom"]),
                "role_partition_sha256": _sha_bytes(
                    _canon(
                        {
                            "labels": train["labels"].tolist(),
                            "receivers": train["receiver_ids"].tolist(),
                            "days": train["day_ids"].tolist(),
                        }
                    )
                ),
                "physical_set_sha256": _sha_bytes(
                    _canon(sorted(train["physical_ids"].tolist()))
                ),
                "excluded_receiver_count": len(tuple(excluded_receivers)),
                "excluded_class_count": len(tuple(excluded_classes)),
            }
        )
    )
    flat_counts = [count for row in counts_by_cell for count in row]
    aggregation_receipt = {
        "schema": SCHEMA + ".aggregation_receipt",
        "bank_cell_count": len(retained_cells),
        "minimum_classes_per_bank_cell": min(len(row) for row in counts_by_cell),
        "minimum_observed_class_cell_physical_count": min(flat_counts),
        "maximum_observed_class_cell_physical_count": max(flat_counts),
        "all_observed_class_cells_ge2": min(flat_counts) >= 2,
        "class_balanced_cell_aggregation": True,
        "class_free_payload": True,
        "payload_contains_class_handles": False,
        "payload_contains_receiver_or_day_names": False,
        "payload_contains_member_or_physical_ids": False,
        "excluded_receiver_count": len(tuple(excluded_receivers)),
        "excluded_class_count": len(tuple(excluded_classes)),
    }
    quantization_receipt = {
        "schema": SCHEMA + ".quantization_receipt",
        "mode": "symmetric_per_row_int8_fp16_scale",
        "basis_max_abs_error": b_err,
        "domain_encoder_max_abs_error": u_err,
        "bank_g_max_abs_error": g_err,
        "bank_t_max_abs_error": t_err,
        "persistent_fp32_sidecar": False,
        "teacher_arrays_persisted": False,
    }
    return Phase1RBMetaBias4Bundle(
        b_q,
        b_s,
        u_q,
        u_s,
        g_q,
        g_s,
        t_q,
        t_s,
        precision16,
        sigma16,
        lambda016,
        amax16,
        float(np.float16(config.temperature)),
        float(np.float16(config.trust_radius)),
        _sha(checkpoint_sha256, "checkpoint"),
        _sha(runtime_sha256, "runtime"),
        _sha(method_lock_sha256, "method lock"),
        source_digest,
        config.digest,
        aggregation_receipt,
        quantization_receipt,
    )


def infer_metabias4_coefficient(
    bundle: Phase1RBMetaBias4Bundle,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen class-equal four-dimensional normal equation."""

    bundle.__post_init__()
    zdom = np.asarray(support_zdom, dtype=np.float64)
    labels = tuple(str(value) for value in support_labels)
    if zdom.ndim != 2 or zdom.shape[1] != Z_DIM or len(labels) != len(zdom) or not np.isfinite(zdom).all():
        raise RBMetaBias4BundleError("support z_dom/label alignment drift")
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2 or any(not value for value in classes):
        raise RBMetaBias4BundleError("support requires at least two opaque classes")
    u = bundle.domain_encoder().astype(np.float64)
    r = _unit_rows(zdom @ u.T, "encoded support domain").astype(np.float64)
    g = bundle.bank_g().astype(np.float64)
    t = bundle.bank_t().astype(np.float64)
    precision = bundle.bank_precision_diag_fp16.astype(np.float64)
    sigma = bundle.bank_sigma_fp16.astype(np.float64)
    similarity = np.clip(r @ g.T, -1.0, 1.0)
    logits = similarity / float(bundle.temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    pi = np.exp(logits)
    pi /= np.sum(pi, axis=1, keepdims=True)
    coverage = np.sum(
        pi * np.exp(-(1.0 - similarity) / np.square(sigma[None, :])), axis=1
    )
    p_diag = coverage[:, None] * (pi @ precision)
    meta = pi @ t
    a_data = np.zeros(RANK, dtype=np.float64)
    b_data = np.zeros(RANK, dtype=np.float64)
    class_counts: list[int] = []
    for class_id in classes:
        indices = np.asarray([value == class_id for value in labels], dtype=bool)
        class_counts.append(int(indices.sum()))
        a_data += np.mean(p_diag[indices], axis=0) / len(classes)
        b_data += np.mean(p_diag[indices] * meta[indices], axis=0) / len(classes)
    lambda0 = bundle.lambda0_diag_fp16.astype(np.float64)
    system = lambda0 + a_data
    unconstrained = b_data / system
    amax = bundle.amax_fp16.astype(np.float64)
    boxed = np.clip(unconstrained, -amax, amax)
    quadratic = float(np.sum(lambda0 * np.square(boxed)))
    ellipsoid_active = quadratic > float(bundle.trust_radius) ** 2
    coefficient = (
        boxed
        if not ellipsoid_active
        else boxed * float(bundle.trust_radius) / np.sqrt(quadratic)
    )
    audit = {
        "schema": SCHEMA + ".coefficient_receipt",
        "class_count": len(classes),
        "class_support_counts": class_counts,
        "all_classes_equal_outer_weight": True,
        "old_new_role_access": False,
        "query_rows_used_for_fit": 0,
        "data_information_rank": int(np.count_nonzero(a_data > 1.0e-10)),
        "system_eigmin": float(np.min(system)),
        "system_eigmax": float(np.max(system)),
        "system_condition": float(np.max(system) / np.min(system)),
        "prior_fraction": float(np.sum(lambda0) / np.sum(system)),
        "coverage_min": float(np.min(coverage)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": float(np.max(coverage)),
        "coverage_hard_gate": False,
        "unconstrained_norm": float(np.linalg.norm(unconstrained)),
        "coefficient_norm": float(np.linalg.norm(coefficient)),
        "box_active": bool(np.any(unconstrained != boxed)),
        "ellipsoid_active": bool(ellipsoid_active),
        "deterministic_projection_order": "box_then_lambda0_ellipsoid_radial",
    }
    return np.asarray(coefficient, dtype=np.float32), audit


def apply_metabias4(
    bundle: Phase1RBMetaBias4Bundle, pre_relu: np.ndarray, coefficient: np.ndarray
) -> np.ndarray:
    rows = np.asarray(pre_relu, dtype=np.float64)
    a = np.asarray(coefficient, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != Z_DIM or a.shape != (RANK,) or not np.isfinite(rows).all() or not np.isfinite(a).all():
        raise RBMetaBias4BundleError("MetaBias application input drift")
    shifted = rows + (bundle.basis().astype(np.float64) @ a)[None, :]
    return _unit_rows(np.maximum(shifted, 0.0), "MetaBias z_id")


def _payload(bundle: Phase1RBMetaBias4Bundle) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(bundle, name)) for name in PAYLOAD_MEMBERS}


def _manifest(bundle: Phase1RBMetaBias4Bundle, npz_sha256: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "profile": WIRE_PROFILE,
        "status": "PHASE1_HELD_ASSET_PENDING_OUTER_JOINT_SEAL",
        "formal_phase2_eligible": False,
        "outer_joint_seal_required": True,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "runtime_sha256": bundle.runtime_sha256,
        "method_lock_sha256": bundle.method_lock_sha256,
        "source_aggregate_digest_sha256": bundle.source_aggregate_digest_sha256,
        "config_digest_sha256": bundle.config_digest_sha256,
        "temperature": bundle.temperature,
        "trust_radius": bundle.trust_radius,
        "content_root_sha256": bundle.content_root_sha256,
        "payload_path": NPZ_NAME,
        "payload_sha256": npz_sha256,
        "payload_member_allowlist": list(PAYLOAD_MEMBERS),
        "numeric_state_bytes": bundle.numeric_state_bytes,
        "aggregation_receipt": dict(bundle.aggregation_receipt),
        "quantization_receipt": dict(bundle.quantization_receipt),
        "phase2_raw_zdom_bank_matching": False,
        "phase2_class_or_receiver_name_access": False,
        "phase2_member_or_physical_id_access": False,
        "phase2_source_replay_access": False,
    }


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in PAYLOAD_MEMBERS:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(payload[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def save_phase1_rb_metabias4_bundle(
    output_dir: str | Path, bundle: Phase1RBMetaBias4Bundle
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        raise RBMetaBias4BundleError(f"output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    npz_path = root / NPZ_NAME
    _write_npz(npz_path, _payload(bundle))
    manifest = _manifest(bundle, sha256_file(npz_path))
    manifest_bytes = _canon(manifest) + b"\n"
    (root / MANIFEST_NAME).write_bytes(manifest_bytes)
    seal = _sha_bytes(manifest_bytes)
    (root / SEAL_NAME).write_text(f"{seal}  {MANIFEST_NAME}\n", encoding="ascii")
    validate_phase1_rb_metabias4_bundle(root)
    return {
        "bundle_root": str(root),
        "content_root_sha256": bundle.content_root_sha256,
        "manifest_sha256": seal,
        "payload_sha256": manifest["payload_sha256"],
        "numeric_state_bytes": bundle.numeric_state_bytes,
        "formal_phase2_eligible": False,
    }


def load_phase1_rb_metabias4_bundle(
    bundle_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
    expected_method_lock_sha256: str | None = None,
) -> Phase1RBMetaBias4Bundle:
    root = Path(bundle_dir)
    manifest_path, npz_path, seal_path = (
        root / MANIFEST_NAME,
        root / NPZ_NAME,
        root / SEAL_NAME,
    )
    if not all(path.is_file() and not path.is_symlink() for path in (manifest_path, npz_path, seal_path)):
        raise RBMetaBias4BundleError("bundle member missing/symlink")
    manifest_bytes = manifest_path.read_bytes()
    seal_tokens = seal_path.read_text(encoding="ascii").strip().split()
    if seal_tokens != [_sha_bytes(manifest_bytes), MANIFEST_NAME]:
        raise RBMetaBias4BundleError("manifest seal drift")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("profile") != WIRE_PROFILE
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("outer_joint_seal_required") is not True
        or manifest.get("payload_member_allowlist") != list(PAYLOAD_MEMBERS)
        or manifest.get("payload_sha256") != sha256_file(npz_path)
    ):
        raise RBMetaBias4BundleError("bundle manifest drift")
    with np.load(npz_path, allow_pickle=False) as archive:
        if tuple(archive.files) != PAYLOAD_MEMBERS:
            raise RBMetaBias4BundleError("bundle payload allowlist drift")
        payload = {name: np.array(archive[name], copy=True) for name in PAYLOAD_MEMBERS}
    bundle = Phase1RBMetaBias4Bundle(
        **payload,
        temperature=float(manifest["temperature"]),
        trust_radius=float(manifest["trust_radius"]),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        runtime_sha256=str(manifest["runtime_sha256"]),
        method_lock_sha256=str(manifest["method_lock_sha256"]),
        source_aggregate_digest_sha256=str(manifest["source_aggregate_digest_sha256"]),
        config_digest_sha256=str(manifest["config_digest_sha256"]),
        aggregation_receipt=manifest["aggregation_receipt"],
        quantization_receipt=manifest["quantization_receipt"],
        content_root_sha256=str(manifest["content_root_sha256"]),
    )
    for actual, expected, name in (
        (bundle.checkpoint_sha256, expected_checkpoint_sha256, "checkpoint"),
        (bundle.runtime_sha256, expected_runtime_sha256, "runtime"),
        (bundle.method_lock_sha256, expected_method_lock_sha256, "method lock"),
    ):
        if expected is not None and actual != _sha(expected, f"expected {name}"):
            raise RBMetaBias4BundleError(f"{name} binding drift")
    if int(manifest["numeric_state_bytes"]) != bundle.numeric_state_bytes:
        raise RBMetaBias4BundleError("numeric state receipt drift")
    return bundle


def validate_phase1_rb_metabias4_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = load_phase1_rb_metabias4_bundle(bundle_dir)
    return {
        "schema": SCHEMA,
        "content_root_sha256": bundle.content_root_sha256,
        "bank_count": bundle.bank_count,
        "numeric_state_bytes": bundle.numeric_state_bytes,
        "class_free_payload": True,
        "int8_joint_seal_verified": True,
        "formal_phase2_eligible": False,
    }


__all__ = [
    "DOMAIN_DIM",
    "MANIFEST_NAME",
    "NPZ_NAME",
    "PAYLOAD_MEMBERS",
    "Phase1RBMetaBias4Bundle",
    "RANK",
    "RBMetaBias4BundleError",
    "RBMetaBias4Config",
    "SCHEMA",
    "SEAL_NAME",
    "Z_DIM",
    "apply_metabias4",
    "build_phase1_rb_metabias4_bundle",
    "infer_metabias4_coefficient",
    "load_phase1_rb_metabias4_bundle",
    "merge_verified_phase1_tap_and_dual_archives",
    "save_phase1_rb_metabias4_bundle",
    "sha256_file",
    "validate_phase1_rb_metabias4_bundle",
]
