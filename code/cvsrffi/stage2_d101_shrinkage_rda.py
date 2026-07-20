"""Typed, analytic shrinkage-RDA alternative global head for D99.

D101 is a Phase1 nested-LODO diagnostic core.  Its sole support input is the
exact typed D99 INT8 bank.  Ground knowledge contributes only a shared
nuisance covariance; every class mean comes from decoded registered support.
Formal target execution remains unavailable until an external D101 LODO lock
and the complete combined resource authority exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d100_ra_cgspr_lgf as d100


FEATURE_DIM = 288
Z_DIM = 160
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
BLOCK_DIMS = (160, 96, 32)
ALLOWED_K = (1, 5, 10, 20)
EPSILON = 1e-12
STATE_LIMIT_BYTES = 256 * 1024
STATE_SCHEMA = "cvs.phase2.d101.shrinkage_rda_state.v1"
FIT_SCHEMA = "cvs.phase2.d101.shrinkage_rda_fit.v1"
QUANTIZATION_SCHEMA = "cvs.phase2.d101.int8_linear_head_audit.v1"
RESOURCE_SCHEMA = "cvs.phase2.d101.incremental_resource.v1"
CANONICAL_FUSION_SCHEMA = "cvs.phase2.d101.canonical_d81_d99_d101_fusion.v1"
KNOWN_PARTIAL_RESOURCE_SCHEMA = "cvs.phase2.d101.known_partial_combined_resource.v1"
WIRE_MAGIC = b"CVS_D101_SRDA_V1\0"


class D101ShrinkageRDAError(ValueError):
    """Raised when typed lifecycle, covariance, or receipt invariants drift."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D101ShrinkageRDAError(f"{name} must be lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        # Exact D100 TypedD81LogitBatch query-receipt representation.
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise D101ShrinkageRDAError(f"{name} must be finite and positive")
    return result


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise D101ShrinkageRDAError(f"{name} must be in [0,1]")
    return result


def _classes(value: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(item) for item in value)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise D101ShrinkageRDAError("D101 requires a unique external class registry")
    return classes


@dataclass(frozen=True, slots=True)
class Phase1D101Lock:
    """K-specific values that nested Phase1 LODO must freeze."""

    k_shot: int
    block_variance_prior: tuple[float, float, float]
    prior_degrees_of_freedom: float
    target_residual_rank: int
    lambda_relative: float
    temperature: float
    d99_temperature: float
    alpha: float
    d99_phase1_lock_digest: str
    ground_geometry_receipt_sha256: str
    phase1_lodo_receipt_sha256: str

    def __post_init__(self) -> None:
        variances = tuple(_positive(item, "block variance prior") for item in self.block_variance_prior)
        if (
            int(self.k_shot) not in ALLOWED_K
            or len(variances) != 3
            or int(self.target_residual_rank) not in (0, 1, 2)
            or (int(self.k_shot) == 1 and int(self.target_residual_rank) != 0)
        ):
            raise D101ShrinkageRDAError("D101 K/rank lock invariant drift")
        _positive(self.prior_degrees_of_freedom, "prior degrees of freedom")
        _positive(self.lambda_relative, "relative ridge")
        _positive(self.temperature, "RDA temperature")
        _positive(self.d99_temperature, "D99 temperature")
        _probability(self.alpha, "D101 alpha")
        _require_sha256(self.d99_phase1_lock_digest, "D99 lock digest")
        _require_sha256(self.ground_geometry_receipt_sha256, "ground geometry receipt")
        _require_sha256(self.phase1_lodo_receipt_sha256, "Phase1 LODO receipt")
        object.__setattr__(self, "block_variance_prior", variances)

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256({"schema": "cvs.phase1.d101_lock.v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class _SharedGroundCovarianceView:
    nuisance_basis_fp32: np.ndarray
    nuisance_spectrum_fp32: np.ndarray
    geometry_receipt_sha256: str

    def __post_init__(self) -> None:
        basis = np.asarray(self.nuisance_basis_fp32)
        spectrum = np.asarray(self.nuisance_spectrum_fp32)
        rank = basis.shape[1] if basis.ndim == 2 else -1
        if (
            basis.dtype != np.float32
            or basis.shape != (Z_DIM, rank)
            or spectrum.dtype != np.float32
            or spectrum.shape != (rank,)
            or rank > 4
            or not np.isfinite(basis).all()
            or not np.isfinite(spectrum).all()
            or np.any(spectrum <= 0.0)
            or (rank and not np.allclose(basis.T @ basis, np.eye(rank), atol=3e-5))
        ):
            raise D101ShrinkageRDAError("D101 shared ground covariance view drift")
        _require_sha256(self.geometry_receipt_sha256, "ground geometry receipt")
        object.__setattr__(self, "nuisance_basis_fp32", _readonly(basis, np.float32))
        object.__setattr__(self, "nuisance_spectrum_fp32", _readonly(spectrum, np.float32))


def _shared_ground_covariance_view(ground: d99.GroundGeometry) -> _SharedGroundCovarianceView:
    """Extract only class-agnostic ground state; class means are never accessed."""

    if type(ground) is not d99.GroundGeometry:
        raise D101ShrinkageRDAError("D101 requires exact D99 GroundGeometry")
    return _SharedGroundCovarianceView(
        nuisance_basis_fp32=ground.nuisance_basis_fp32,
        nuisance_spectrum_fp32=ground.nuisance_spectrum_fp32,
        geometry_receipt_sha256=ground.geometry_receipt_sha256,
    )


def _normalize_rows(value: np.ndarray, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != dimension or not np.isfinite(rows).all():
        raise D101ShrinkageRDAError(f"{name} must be finite [N,{dimension}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise D101ShrinkageRDAError(f"{name} contains a zero-norm row")
    return rows / norms


def _feature_map(bank: d99.TypedINT8MetricKernelBank, features: np.ndarray) -> np.ndarray:
    if type(bank) is not d99.TypedINT8MetricKernelBank:
        raise D101ShrinkageRDAError("D101 feature map requires exact typed D99 bank")
    rows = np.asarray(features)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or not np.isfinite(rows).all()
    ):
        raise D101ShrinkageRDAError("D101 features must be finite float32 [N,288]")
    weights = np.asarray(
        (bank.config.z_weight, bank.config.fft_weight, bank.config.rf_weight),
        dtype=np.float64,
    )
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise D101ShrinkageRDAError("D101 D99 block weight drift")
    z = d100.generalized_precision_sqrt_transform(
        rows[:, :Z_DIM],
        bank.metric.metric_basis_fp32,
        bank.metric.precision_attenuation_fp32,
    )
    fft = _normalize_rows(rows[:, 160:256], 96, "D101 FFT96 rows")
    rf = _normalize_rows(rows[:, 256:288], 32, "D101 RF32 rows")
    mapped = np.concatenate(
        (
            math.sqrt(weights[0]) * z,
            math.sqrt(weights[1]) * fft,
            math.sqrt(weights[2]) * rf,
        ),
        axis=1,
    )
    if not np.allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=2e-6):
        raise D101ShrinkageRDAError("D101 mapped feature norm drift")
    return mapped


def _linear_d99_sqrt_on_ground_basis(
    bank: d99.TypedINT8MetricKernelBank,
    view: _SharedGroundCovarianceView,
) -> np.ndarray:
    ground_basis = view.nuisance_basis_fp32.astype(np.float64)
    metric_basis = bank.metric.metric_basis_fp32.astype(np.float64)
    attenuation = bank.metric.precision_attenuation_fp32.astype(np.float64)
    if metric_basis.shape[1]:
        sqrt_attenuation = 1.0 - np.sqrt(1.0 - attenuation)
        ground_basis = ground_basis - metric_basis @ (
            sqrt_attenuation[:, None] * (metric_basis.T @ ground_basis)
        )
    embedded = np.zeros((FEATURE_DIM, ground_basis.shape[1]), dtype=np.float64)
    embedded[:Z_DIM] = math.sqrt(float(bank.config.z_weight)) * ground_basis
    return embedded


def _within_class_covariance(
    mapped: np.ndarray, indices: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.stack([np.mean(mapped[indices == index], axis=0) for index in range(class_count)])
    residual = mapped - means[indices]
    if k_shot == 1:
        covariance = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    else:
        covariance = residual.T @ residual / float(class_count * (k_shot - 1))
        covariance = 0.5 * (covariance + covariance.T)
    return means, residual, covariance


def _block_isotropic_variances(covariance: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.trace(covariance[block, block])) / dim for block, dim in zip(BLOCK_SLICES, BLOCK_DIMS)],
        dtype=np.float64,
    )


def _woodbury_precision_apply(rows: np.ndarray, diagonal: np.ndarray, factor: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    diag = np.asarray(diagonal, dtype=np.float64)
    low_rank = np.asarray(factor, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_DIM
        or diag.shape != (FEATURE_DIM,)
        or low_rank.ndim != 2
        or low_rank.shape[0] != FEATURE_DIM
        or low_rank.shape[1] > 6
        or not np.isfinite(values).all()
        or not np.isfinite(diag).all()
        or not np.isfinite(low_rank).all()
        or np.any(diag <= 0.0)
    ):
        raise D101ShrinkageRDAError("D101 Woodbury inputs drift")
    inverse_diagonal = 1.0 / diag
    base = values * inverse_diagonal[None, :]
    if low_rank.shape[1] == 0:
        return base
    system = np.eye(low_rank.shape[1]) + low_rank.T @ (
        inverse_diagonal[:, None] * low_rank
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (system + system.T))
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    if np.min(eigenvalues) <= 0.0 or not math.isfinite(condition):
        raise D101ShrinkageRDAError("D101 Woodbury system is not finite SPD")
    correction = np.linalg.solve(system, (base @ low_rank).T).T
    return base - (correction @ low_rank.T) * inverse_diagonal[None, :]


def _woodbury_precision_dense(diagonal: np.ndarray, factor: np.ndarray) -> np.ndarray:
    diag = np.asarray(diagonal, dtype=np.float64)
    low_rank = np.asarray(factor, dtype=np.float64)
    return np.linalg.inv(np.diag(diag) + low_rank @ low_rank.T)


def _quantize_weight_rows(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.isfinite(values).all():
        raise D101ShrinkageRDAError("D101 teacher weights must be finite [C,288]")
    codes = np.empty(values.shape, dtype=np.int8)
    scales = np.empty((len(values), 3), dtype=np.float16)
    decoded = np.empty(values.shape, dtype=np.float64)
    for row in range(len(values)):
        for block_index, block in enumerate(BLOCK_SLICES):
            scale = max(float(np.max(np.abs(values[row, block]))) / 127.0, np.finfo(np.float16).tiny)
            stored = np.float16(scale)
            scales[row, block_index] = stored
            code = np.clip(np.rint(values[row, block] / float(stored)), -127, 127).astype(np.int8)
            codes[row, block] = code
            decoded[row, block] = code.astype(np.float64) * float(stored)
    return codes, scales, decoded


def _decode_weight_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    code = np.asarray(codes)
    scale = np.asarray(scales)
    if code.dtype != np.int8 or code.ndim != 2 or code.shape[1] != FEATURE_DIM:
        raise D101ShrinkageRDAError("D101 INT8 weight code drift")
    if scale.dtype != np.float16 or scale.shape != (len(code), 3):
        raise D101ShrinkageRDAError("D101 FP16 weight scale drift")
    result = np.empty(code.shape, dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        result[:, block] = code[:, block].astype(np.float32) * scale[:, block_index, None].astype(np.float32)
    return result


def _serialize_wire(header: Mapping[str, Any], arrays: Sequence[tuple[str, np.ndarray]]) -> bytes:
    header_bytes = _canonical_bytes(header)
    parts = [WIRE_MAGIC, struct.pack("<Q", len(header_bytes)), header_bytes]
    for name, value in arrays:
        name_bytes = name.encode("ascii")
        payload = np.ascontiguousarray(value).tobytes(order="C")
        parts.extend((struct.pack("<H", len(name_bytes)), name_bytes, struct.pack("<Q", len(payload)), payload))
    return b"".join(parts)


def _resource_from_state(
    *, class_count: int, k_shot: int, metric_rank: int, alpha: float, numeric_bytes: int, wire_bytes: int, d99_query_macs: int
) -> dict[str, Any]:
    branch_macs = int(
        class_count * FEATURE_DIM
        + FEATURE_DIM * 5
        + 2 * Z_DIM * metric_rank
        + metric_rank
        + FEATURE_DIM
        + class_count * FEATURE_DIM
        + class_count
    )
    probability_ops = int(class_count * (8 + 3))
    branch_enabled = float(alpha) > 0.0
    known_query_macs = int(d99_query_macs + (branch_macs if branch_enabled else 0))
    return {
        "schema": RESOURCE_SCHEMA,
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "trainable_parameters": 0,
        "optimizer_steps": 0,
        "query_state_updates": 0,
        "persistent_query_batch_state_bytes": 0,
        "persistent_parameter_equivalent": int(class_count * (FEATURE_DIM + 1)),
        "numeric_logical_state_bytes": int(numeric_bytes),
        "actual_serialized_state_bytes": int(wire_bytes),
        "incremental_state_below_256kib": bool(wire_bytes <= STATE_LIMIT_BYTES),
        "d101_branch_query_mac_upper_bound": branch_macs,
        "d101_branch_probability_scalar_ops": probability_ops,
        "alpha_zero_skips_query_branch": not branch_enabled,
        "d99_canonical_query_mac_upper_bound": int(d99_query_macs),
        "known_d99_plus_d101_query_mac_upper_bound": known_query_macs,
        "typed_d81_head_query_mac_available": False,
        "complete_combined_query_mac_available": False,
        "complete_combined_resource_claim": False,
    }


def _state_core(
    *, classes: tuple[str, ...], k_shot: int, codes: np.ndarray, scales: np.ndarray, bias: np.ndarray,
    temperature: float, d99_temperature: float, alpha: float, d99_bank_receipt_sha256: str,
    d99_lock_digest: str, ground_geometry_receipt_sha256: str, config_lock_digest: str,
    phase1_lodo_receipt_sha256: str, fit_receipt_sha256: str, fit_audit: Mapping[str, Any],
    quantization_audit: Mapping[str, Any], formal_phase2_eligible: bool,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "classes": list(classes),
        "k_shot": int(k_shot),
        "weight_codes_qint8": _array_receipt(codes),
        "weight_scales_fp16": _array_receipt(scales),
        "bias_fp16": _array_receipt(bias),
        "temperature": float(temperature),
        "d99_temperature": float(d99_temperature),
        "alpha": float(alpha),
        "d99_bank_receipt_sha256": d99_bank_receipt_sha256,
        "d99_lock_digest": d99_lock_digest,
        "ground_geometry_receipt_sha256": ground_geometry_receipt_sha256,
        "config_lock_digest": config_lock_digest,
        "phase1_lodo_receipt_sha256": phase1_lodo_receipt_sha256,
        "fit_receipt_sha256": fit_receipt_sha256,
        "fit_audit": fit_audit,
        "quantization_audit": quantization_audit,
        "formal_phase2_eligible": bool(formal_phase2_eligible),
    }


def _closed_state_artifact(**kwargs: Any) -> tuple[dict[str, Any], str, bytes]:
    codes = np.asarray(kwargs["codes"])
    scales = np.asarray(kwargs["scales"])
    bias = np.asarray(kwargs["bias"])
    arrays = (("weight_codes_qint8", codes), ("weight_scales_fp16", scales), ("bias_fp16", bias))
    numeric_bytes = int(sum(value.nbytes for _, value in arrays))
    wire_size = 0
    for _ in range(16):
        resource = _resource_from_state(
            class_count=len(kwargs["classes"]),
            k_shot=kwargs["k_shot"],
            metric_rank=kwargs["metric_rank"],
            alpha=kwargs["alpha"],
            numeric_bytes=numeric_bytes,
            wire_bytes=wire_size,
            d99_query_macs=kwargs["d99_query_macs"],
        )
        core = _state_core(
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"metric_rank", "d99_query_macs"}
            }
        )
        receipt = _canonical_sha256({**core, "resource_audit": resource})
        header = {**core, "resource_audit": resource, "state_receipt_sha256": receipt}
        wire = _serialize_wire(header, arrays)
        if len(wire) == wire_size:
            return resource, receipt, wire
        wire_size = len(wire)
    raise D101ShrinkageRDAError("D101 wire-size fixed point did not converge")


@dataclass(frozen=True, slots=True)
class D101ShrinkageRDAState:
    classes: tuple[str, ...]
    k_shot: int
    weight_codes_qint8: np.ndarray
    weight_scales_fp16: np.ndarray
    bias_fp16: np.ndarray
    temperature: float
    d99_temperature: float
    alpha: float
    d99_bank_receipt_sha256: str
    d99_lock_digest: str
    ground_geometry_receipt_sha256: str
    config_lock_digest: str
    phase1_lodo_receipt_sha256: str
    fit_receipt_sha256: str
    fit_audit: Mapping[str, Any]
    quantization_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    state_receipt_sha256: str
    formal_phase2_eligible: bool = False
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        if (
            self.schema != STATE_SCHEMA
            or int(self.k_shot) not in ALLOWED_K
            or self.weight_codes_qint8.dtype != np.int8
            or self.weight_codes_qint8.shape != (len(classes), FEATURE_DIM)
            or self.weight_scales_fp16.dtype != np.float16
            or self.weight_scales_fp16.shape != (len(classes), 3)
            or self.bias_fp16.dtype != np.float16
            or self.bias_fp16.shape != (len(classes),)
            or np.any(self.weight_scales_fp16 <= 0.0)
            or not np.isfinite(self.weight_scales_fp16).all()
            or not np.isfinite(self.bias_fp16).all()
            or self.formal_phase2_eligible is not False
        ):
            raise D101ShrinkageRDAError("D101 state invariant drift")
        _positive(self.temperature, "RDA temperature")
        _positive(self.d99_temperature, "D99 temperature")
        _probability(self.alpha, "D101 alpha")
        for value, name in (
            (self.d99_bank_receipt_sha256, "D99 bank receipt"),
            (self.d99_lock_digest, "D99 lock digest"),
            (self.ground_geometry_receipt_sha256, "ground geometry receipt"),
            (self.config_lock_digest, "D101 lock digest"),
            (self.phase1_lodo_receipt_sha256, "Phase1 LODO receipt"),
            (self.fit_receipt_sha256, "fit receipt"),
            (self.state_receipt_sha256, "state receipt"),
        ):
            _require_sha256(value, name)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "weight_codes_qint8", _readonly(self.weight_codes_qint8, np.int8))
        object.__setattr__(self, "weight_scales_fp16", _readonly(self.weight_scales_fp16, np.float16))
        object.__setattr__(self, "bias_fp16", _readonly(self.bias_fp16, np.float16))
        object.__setattr__(self, "fit_audit", MappingProxyType(dict(self.fit_audit)))
        object.__setattr__(self, "quantization_audit", MappingProxyType(dict(self.quantization_audit)))
        object.__setattr__(self, "resource_audit", MappingProxyType(dict(self.resource_audit)))
        if not _verify_state(self):
            raise D101ShrinkageRDAError("D101 state resource/receipt closure drift")


def _state_artifact_kwargs(state: D101ShrinkageRDAState) -> dict[str, Any]:
    return {
        "classes": state.classes,
        "k_shot": state.k_shot,
        "codes": state.weight_codes_qint8,
        "scales": state.weight_scales_fp16,
        "bias": state.bias_fp16,
        "temperature": state.temperature,
        "d99_temperature": state.d99_temperature,
        "alpha": state.alpha,
        "d99_bank_receipt_sha256": state.d99_bank_receipt_sha256,
        "d99_lock_digest": state.d99_lock_digest,
        "ground_geometry_receipt_sha256": state.ground_geometry_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "phase1_lodo_receipt_sha256": state.phase1_lodo_receipt_sha256,
        "fit_receipt_sha256": state.fit_receipt_sha256,
        "fit_audit": state.fit_audit,
        "quantization_audit": state.quantization_audit,
        "formal_phase2_eligible": state.formal_phase2_eligible,
        "metric_rank": int(state.fit_audit["d99_metric_rank"]),
        "d99_query_macs": int(state.resource_audit["d99_canonical_query_mac_upper_bound"]),
    }


def _verify_state(state: D101ShrinkageRDAState) -> bool:
    try:
        resource, receipt, _wire = _closed_state_artifact(**_state_artifact_kwargs(state))
        return _jsonable(resource) == _jsonable(state.resource_audit) and receipt == state.state_receipt_sha256
    except (D101ShrinkageRDAError, KeyError, TypeError, ValueError):
        return False


def build_shrinkage_rda_state(
    bank: d99.TypedINT8MetricKernelBank,
    ground: d99.GroundGeometry,
    *,
    config: Phase1D101Lock,
) -> D101ShrinkageRDAState:
    """Build D101 from the exact typed D99 support lifecycle only."""

    if type(bank) is not d99.TypedINT8MetricKernelBank or type(config) is not Phase1D101Lock:
        raise D101ShrinkageRDAError("D101 requires exact typed D99 bank/config")
    if (
        bank.metric.k_shot != config.k_shot
        or bank.config.lock_digest != config.d99_phase1_lock_digest
        or bank.metric.ground_geometry_receipt_sha256 != config.ground_geometry_receipt_sha256
        or tuple(bank.support_counts) != tuple(config.k_shot for _ in bank.classes)
    ):
        raise D101ShrinkageRDAError("D101 typed bank/config binding drift")
    view = _shared_ground_covariance_view(ground)
    if view.geometry_receipt_sha256 != config.ground_geometry_receipt_sha256:
        raise D101ShrinkageRDAError("D101 ground receipt drift")
    classes = _classes(bank.classes)
    decoded_support = d99.decode_support_bank(bank)
    mapped = _feature_map(bank, decoded_support)
    indices = np.asarray(bank.class_indices_int16, dtype=np.int64)
    means, residual, support_covariance = _within_class_covariance(
        mapped, indices, len(classes), config.k_shot
    )
    empirical_block_variance = _block_isotropic_variances(support_covariance)
    residual_dof = len(classes) * (config.k_shot - 1)
    shrinkage = float(
        residual_dof / (config.prior_degrees_of_freedom + residual_dof)
    )
    if config.k_shot == 1 and (shrinkage != 0.0 or np.any(support_covariance != 0.0)):
        raise AssertionError("K1 must have exactly zero target covariance degrees of freedom")
    prior_variance = np.asarray(config.block_variance_prior, dtype=np.float64)
    base_variance = (1.0 - shrinkage) * prior_variance + shrinkage * empirical_block_variance

    target_rank = 0
    target_values = np.empty(0, dtype=np.float64)
    target_vectors = np.empty((FEATURE_DIM, 0), dtype=np.float64)
    if config.k_shot > 1 and config.target_residual_rank > 0:
        support_minus_block = support_covariance.copy()
        for block, variance in zip(BLOCK_SLICES, empirical_block_variance):
            support_minus_block[block, block] -= variance * np.eye(block.stop - block.start)
        values, vectors = np.linalg.eigh(0.5 * (support_minus_block + support_minus_block.T))
        positive = np.flatnonzero(values > EPSILON)
        target_rank = min(config.target_residual_rank, len(positive))
        if target_rank:
            order = positive[np.argsort(values[positive], kind="stable")[-target_rank:][::-1]]
            target_values = values[order]
            target_vectors = vectors[:, order]

    ground_weight = float(bank.metric.ground_weight)
    if not 0.0 <= ground_weight <= 1.0:
        raise D101ShrinkageRDAError("D101 must reuse finite D99 ground weight")
    transformed_ground = _linear_d99_sqrt_on_ground_basis(bank, view)
    if ground_weight == 0.0 or transformed_ground.shape[1] == 0:
        # Low coverage is a real rank-zero fallback, not a bank of zero
        # columns that would inflate Woodbury rank and retain ground coupling.
        ground_factor = np.zeros((FEATURE_DIM, 0), dtype=np.float64)
    else:
        ground_scale = np.sqrt(
            (1.0 - shrinkage)
            * ground_weight
            * view.nuisance_spectrum_fp32.astype(np.float64)
        )
        ground_factor = transformed_ground * ground_scale[None, :]
    target_factor = target_vectors * np.sqrt(shrinkage * target_values)[None, :]
    factor = np.concatenate((ground_factor, target_factor), axis=1)
    if factor.shape[1] > 6:
        raise D101ShrinkageRDAError("D101 total Woodbury rank exceeds six")
    diagonal_without_ridge = np.concatenate(
        [np.full(dim, value, dtype=np.float64) for dim, value in zip(BLOCK_DIMS, base_variance)]
    )
    sigma_bar_trace = float(np.sum(diagonal_without_ridge) + np.sum(np.square(factor)))
    relative_ridge = config.lambda_relative * sigma_bar_trace / FEATURE_DIM
    if not math.isfinite(relative_ridge) or relative_ridge <= 0.0:
        raise D101ShrinkageRDAError("D101 relative ridge is not finite positive")
    diagonal = diagonal_without_ridge + relative_ridge
    precision_means = _woodbury_precision_apply(means, diagonal, factor)
    teacher_weights = precision_means
    teacher_bias = -0.5 * np.sum(means * precision_means, axis=1)
    codes, scales, decoded_weights = _quantize_weight_rows(teacher_weights)
    deployed_bias = np.asarray(teacher_bias, dtype=np.float16)
    teacher_logits = mapped @ teacher_weights.T + teacher_bias[None, :]
    deployed_logits = mapped @ decoded_weights.T + deployed_bias.astype(np.float64)[None, :]
    order = np.argsort(teacher_logits, axis=1, kind="stable")
    rows = np.arange(len(mapped))
    winner = order[:, -1]
    runner_up = order[:, -2]
    teacher_margin = teacher_logits[rows, winner] - teacher_logits[rows, runner_up]
    deployed_margin = deployed_logits[rows, winner] - deployed_logits[rows, runner_up]
    quantization_audit = {
        "schema": QUANTIZATION_SCHEMA,
        "scope": "support_fit_diagnostic_not_held_lodo_margin_authority",
        "teacher_deployed_logit_max_abs_error": float(np.max(np.abs(teacher_logits - deployed_logits))),
        "support_top1_agreement": float(np.mean(np.argmax(teacher_logits, axis=1) == np.argmax(deployed_logits, axis=1))),
        "teacher_winner_margin_sign_flip_count": int(np.sum(deployed_margin <= 0.0)),
        "teacher_margin_min": float(np.min(teacher_margin)),
        "deployed_teacher_winner_margin_min": float(np.min(deployed_margin)),
        "same_quantization_formula_all_registered_classes": True,
        "held_lodo_margin_audit_present": False,
        "query_rows_used": 0,
    }
    fit_payload = {
        "schema": FIT_SCHEMA,
        "classes": list(classes),
        "k_shot": config.k_shot,
        "d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "d99_metric_receipt_sha256": bank.metric.metric_receipt_sha256,
        "ground_geometry_receipt_sha256": view.geometry_receipt_sha256,
        "config_lock_digest": config.lock_digest,
        "mapped_support": _array_receipt(mapped),
        "class_means": _array_receipt(means),
        "support_covariance": _array_receipt(support_covariance),
        "diagonal": _array_receipt(diagonal),
        "woodbury_factor": _array_receipt(factor),
        "weight_codes": _array_receipt(codes),
        "weight_scales": _array_receipt(scales),
        "bias": _array_receipt(deployed_bias),
    }
    fit_receipt = _canonical_sha256(fit_payload)
    dense_covariance = np.diag(diagonal) + factor @ factor.T
    dense_eigenvalues = np.linalg.eigvalsh(dense_covariance)
    condition = float(np.max(dense_eigenvalues) / np.min(dense_eigenvalues))
    fit_audit = {
        "schema": FIT_SCHEMA,
        "support_fit_receipt_sha256": fit_receipt,
        "support_source": "exact_typed_D99_INT8_bank_decode_only",
        "all_registered_class_means_from_support_only": True,
        "class_means_are_arithmetic_not_renormalized": True,
        "ground_class_means_accessed": False,
        "ground_class_logit_or_bias": False,
        "equal_class_prior": True,
        "external_registered_class_order_preserved": True,
        "support_row_count": int(len(mapped)),
        "residual_degrees_of_freedom": int(residual_dof),
        "support_shrinkage_a": shrinkage,
        "block_variance_prior": list(config.block_variance_prior),
        "empirical_block_variance": empirical_block_variance.tolist(),
        "posterior_block_variance_before_ridge": base_variance.tolist(),
        "relative_ridge": relative_ridge,
        "ground_coverage_rho_reused_from_d99": float(bank.metric.ground_coverage_rho),
        "ground_weight_reused_from_d99": ground_weight,
        "ground_factor_scaled_by_sqrt_one_minus_a": True,
        "ground_basis_push_forward_model": (
            "metric_sqrt_linear_map_before_per_sample_normalization;"
            "first_order_proxy_for_the_normalized_feature_map"
        ),
        "ground_basis_exact_normalized_push_forward_claimed": False,
        "ground_rank": int(ground_factor.shape[1]),
        "target_rank": int(target_factor.shape[1]),
        "woodbury_rank": int(factor.shape[1]),
        "d99_metric_rank": int(bank.metric.metric_basis_fp32.shape[1]),
        "k1_target_covariance_exact_zero": bool(config.k_shot == 1),
        "low_coverage_fallback": (
            "three_block_isotropic" if ground_weight == 0.0 and config.k_shot == 1 else "not_applicable"
        ),
        "covariance_min_eigenvalue": float(np.min(dense_eigenvalues)),
        "covariance_condition_number": condition,
        "common_transform_claimed_as_gain": False,
        "benefit_mechanism": "fixed_ground_prior_structured_shrinkage_rank2_truncation_and_quantization",
        "optimizer_steps": 0,
        "query_rows_used": 0,
        "formal_phase2_eligible": False,
    }
    d99_query_macs = int(bank.resource_audit["query_mac_upper_bound"])
    kwargs = {
        "classes": classes,
        "k_shot": config.k_shot,
        "codes": codes,
        "scales": scales,
        "bias": deployed_bias,
        "temperature": config.temperature,
        "d99_temperature": config.d99_temperature,
        "alpha": config.alpha,
        "d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "d99_lock_digest": bank.config.lock_digest,
        "ground_geometry_receipt_sha256": view.geometry_receipt_sha256,
        "config_lock_digest": config.lock_digest,
        "phase1_lodo_receipt_sha256": config.phase1_lodo_receipt_sha256,
        "fit_receipt_sha256": fit_receipt,
        "fit_audit": fit_audit,
        "quantization_audit": quantization_audit,
        "formal_phase2_eligible": False,
        "metric_rank": int(bank.metric.metric_basis_fp32.shape[1]),
        "d99_query_macs": d99_query_macs,
    }
    resource, receipt, _wire = _closed_state_artifact(**kwargs)
    return D101ShrinkageRDAState(
        classes=classes,
        k_shot=config.k_shot,
        weight_codes_qint8=codes,
        weight_scales_fp16=scales,
        bias_fp16=deployed_bias,
        temperature=config.temperature,
        d99_temperature=config.d99_temperature,
        alpha=config.alpha,
        d99_bank_receipt_sha256=bank.bank_receipt_sha256,
        d99_lock_digest=bank.config.lock_digest,
        ground_geometry_receipt_sha256=view.geometry_receipt_sha256,
        config_lock_digest=config.lock_digest,
        phase1_lodo_receipt_sha256=config.phase1_lodo_receipt_sha256,
        fit_receipt_sha256=fit_receipt,
        fit_audit=fit_audit,
        quantization_audit=quantization_audit,
        resource_audit=resource,
        state_receipt_sha256=receipt,
    )


def serialize_shrinkage_rda_state(state: D101ShrinkageRDAState) -> bytes:
    if type(state) is not D101ShrinkageRDAState or not _verify_state(state):
        raise D101ShrinkageRDAError("D101 serialization state verification failed")
    resource, receipt, wire = _closed_state_artifact(**_state_artifact_kwargs(state))
    if receipt != state.state_receipt_sha256 or _jsonable(resource) != _jsonable(state.resource_audit):
        raise D101ShrinkageRDAError("D101 serialization receipt drift")
    return wire


def _score_compiled_rda_logits(
    state: D101ShrinkageRDAState,
    bank: d99.TypedINT8MetricKernelBank,
    query_features: np.ndarray,
) -> np.ndarray:
    mapped = _feature_map(bank, query_features)
    weights = _decode_weight_rows(state.weight_codes_qint8, state.weight_scales_fp16).astype(np.float64)
    logits = mapped @ weights.T + state.bias_fp16.astype(np.float64)[None, :]
    if not np.isfinite(logits).all():
        raise D101ShrinkageRDAError("D101 RDA logits became non-finite")
    return _readonly(logits, np.float32)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / _positive(temperature, "temperature")
    values -= np.max(values, axis=1, keepdims=True)
    exponential = np.exp(values)
    return _readonly(exponential / np.sum(exponential, axis=1, keepdims=True), np.float32)


@dataclass(frozen=True, slots=True)
class D101CanonicalFusionResult:
    d81_probability_fp32: np.ndarray
    student_t_probability_fp32: np.ndarray
    d99_probability_fp32: np.ndarray
    rda_probability_fp32: np.ndarray | None
    fused_probability_fp32: np.ndarray
    prediction: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays = (self.d81_probability_fp32, self.student_t_probability_fp32, self.d99_probability_fp32, self.fused_probability_fp32)
        reference = np.asarray(arrays[0])
        if (
            any(np.asarray(item).dtype != np.float32 or np.asarray(item).shape != reference.shape for item in arrays)
            or any(not np.isfinite(np.asarray(item)).all() for item in arrays)
            or any(np.any(np.asarray(item) < 0.0) for item in arrays)
            or any(np.any(np.asarray(item) > 1.0 + 2e-6) for item in arrays)
            or any(not np.allclose(np.sum(np.asarray(item), axis=1), 1.0, atol=2e-6) for item in arrays)
            or self.prediction.shape != (len(reference),)
        ):
            raise D101ShrinkageRDAError("D101 canonical fusion result drift")
        if self.rda_probability_fp32 is not None:
            rda_probability = np.asarray(self.rda_probability_fp32)
            if (
                rda_probability.dtype != np.float32
                or rda_probability.shape != reference.shape
                or not np.isfinite(rda_probability).all()
                or np.any(rda_probability < 0.0)
                or np.any(rda_probability > 1.0 + 2e-6)
                or not np.allclose(np.sum(rda_probability, axis=1), 1.0, atol=2e-6)
            ):
                raise D101ShrinkageRDAError("D101 RDA branch probability drift")
            object.__setattr__(self, "rda_probability_fp32", _readonly(rda_probability, np.float32))
        for name in ("d81_probability_fp32", "student_t_probability_fp32", "d99_probability_fp32", "fused_probability_fp32"):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float32))
        prediction = np.asarray(self.prediction, dtype=str).copy()
        prediction.setflags(write=False)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def canonical_fuse_typed_d81_d99_d101(
    state: D101ShrinkageRDAState,
    bank: d99.TypedINT8MetricKernelBank,
    typed_d81: d100.TypedD81LogitBatch,
    query_features: np.ndarray,
    *,
    evaluate_complementarity_branch: bool = False,
) -> D101CanonicalFusionResult:
    """Sole public query path: D101 replaces, never stacks on, D100."""

    if type(state) is not D101ShrinkageRDAState or not _verify_state(state):
        raise D101ShrinkageRDAError("D101 canonical state verification failed")
    if type(bank) is not d99.TypedINT8MetricKernelBank:
        raise D101ShrinkageRDAError("D101 canonical path requires exact typed D99 bank")
    if type(typed_d81) is not d100.TypedD81LogitBatch:
        raise D101ShrinkageRDAError("D101 canonical path requires exact typed D81 batch")
    expected_d81_batch_receipt = d100._canonical_sha256(
        d100._typed_d81_batch_payload(
            classes=typed_d81.classes,
            k_shot=typed_d81.k_shot,
            logits=typed_d81.logits_fp32,
            query_feature_receipt=typed_d81.query_feature_receipt,
            source_schema=typed_d81.source_schema,
            source_receipt_sha256=typed_d81.source_receipt_sha256,
        )
    )
    if typed_d81.batch_receipt_sha256 != expected_d81_batch_receipt:
        raise D101ShrinkageRDAError("D101 typed D81 batch receipt drift")
    if (
        state.classes != bank.classes
        or state.classes != typed_d81.classes
        or state.k_shot != bank.metric.k_shot
        or state.k_shot != typed_d81.k_shot
        or state.d99_bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.d99_lock_digest != bank.config.lock_digest
    ):
        raise D101ShrinkageRDAError("D101 canonical typed lifecycle binding drift")
    query = np.asarray(query_features)
    if (
        query.dtype != np.float32
        or query.shape != (len(typed_d81.logits_fp32), FEATURE_DIM)
        or not np.isfinite(query).all()
        or _array_receipt(query) != dict(typed_d81.query_feature_receipt)
    ):
        raise D101ShrinkageRDAError("D101 canonical query receipt drift")
    d81_probability = _softmax(typed_d81.logits_fp32, 1.0)
    student_probability = _softmax(
        d99.score_metric_kernel_raw_logits(bank, query), state.d99_temperature
    )
    eta = float(bank.eta_phase1_locked)
    probability99 = _readonly(
        (1.0 - eta) * d81_probability.astype(np.float64)
        + eta * student_probability.astype(np.float64),
        np.float32,
    )
    evaluate_rda = bool(state.alpha > 0.0 or evaluate_complementarity_branch)
    rda_probability = (
        _softmax(_score_compiled_rda_logits(state, bank, query), state.temperature)
        if evaluate_rda
        else None
    )
    if state.alpha == 0.0:
        fused = _readonly(probability99, np.float32)
    elif state.alpha == 1.0:
        if rda_probability is None:
            raise AssertionError("alpha one requires D101 branch")
        fused = _readonly(rda_probability, np.float32)
    else:
        if rda_probability is None:
            raise AssertionError("nonzero alpha requires D101 branch")
        fused = _readonly(
            (1.0 - state.alpha) * probability99.astype(np.float64)
            + state.alpha * rda_probability.astype(np.float64),
            np.float32,
        )
    prediction = np.asarray(state.classes, dtype=str)[np.argmax(fused, axis=1)]
    audit = {
        "schema": CANONICAL_FUSION_SCHEMA,
        "typed_d81_batch_receipt_sha256": typed_d81.batch_receipt_sha256,
        "d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "d101_state_receipt_sha256": state.state_receipt_sha256,
        "formula": "p99=(1-eta)*p81+eta*pStudentT;p101=(1-alpha)*p99+alpha*pRDA",
        "d101_replaces_d100_not_third_alpha": True,
        "eta_phase1_locked": eta,
        "alpha_phase1_locked": float(state.alpha),
        "RDA_branch_evaluated": evaluate_rda,
        "query_batch_dependency": False,
        "query_state_updates": 0,
        "formal_phase2_eligible": False,
    }
    return D101CanonicalFusionResult(
        d81_probability_fp32=d81_probability,
        student_t_probability_fp32=student_probability,
        d99_probability_fp32=probability99,
        rda_probability_fp32=rda_probability,
        fused_probability_fp32=fused,
        prediction=prediction,
        audit=audit,
    )


def audit_known_partial_combined_resources(
    state: D101ShrinkageRDAState,
    bank: d99.TypedINT8MetricKernelBank,
    ground: d99.GroundGeometry,
) -> Mapping[str, Any]:
    """Recompute exact available bytes; never accept caller-supplied sizes."""

    if type(state) is not D101ShrinkageRDAState or type(bank) is not d99.TypedINT8MetricKernelBank or type(ground) is not d99.GroundGeometry:
        raise D101ShrinkageRDAError("D101 partial resource audit requires exact objects")
    state_wire = serialize_shrinkage_rda_state(state)
    bank_wire = d99._serialize_receipt_bearing_bank(bank)
    view = _shared_ground_covariance_view(ground)
    ground_numeric_bytes = int(view.nuisance_basis_fp32.nbytes + view.nuisance_spectrum_fp32.nbytes)
    known_wire_bytes = int(len(state_wire) + len(bank_wire))
    return MappingProxyType(
        {
            "schema": KNOWN_PARTIAL_RESOURCE_SCHEMA,
            "d101_state_wire_bytes": len(state_wire),
            "d101_state_sha256": _sha256_bytes(state_wire),
            "d99_bank_wire_bytes": len(bank_wire),
            "d99_bank_sha256": _sha256_bytes(bank_wire),
            "ground_shared_numeric_bytes_not_wire": ground_numeric_bytes,
            "known_partial_wire_bytes": known_wire_bytes,
            "known_partial_below_256kib": bool(known_wire_bytes <= STATE_LIMIT_BYTES),
            "typed_d81_persistent_head_wire_available": False,
            "ground_complete_wire_available": False,
            "typed_d81_logit_batch_not_counted_as_persistent_head": True,
            "all_reported_wire_sizes_recomputed_from_exact_serializers": True,
            "complete_combined_resource_claim": False,
            "formal_under_256kib_claim": False,
        }
    )


def predict_formal(*_args: Any, **_kwargs: Any) -> None:
    raise D101ShrinkageRDAError(
        "D101 formal prediction is blocked pending independent Phase1 LODO and complete resources"
    )


__all__ = [
    "D101CanonicalFusionResult",
    "D101ShrinkageRDAError",
    "D101ShrinkageRDAState",
    "Phase1D101Lock",
    "audit_known_partial_combined_resources",
    "build_shrinkage_rda_state",
    "canonical_fuse_typed_d81_d99_d101",
    "predict_formal",
    "serialize_shrinkage_rda_state",
]
