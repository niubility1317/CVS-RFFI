"""Pure z_id160 local-global head for the Stage2 B causal arm.

This module adds only a class-balanced shrinkage RDA head and a Phase1-locked
scalar fusion to the reviewed Patch-A Student-t qKNN baseline.  It never fits
an encoder or metric, never reads z_dom/FFT/RF, and never accepts query data in
state construction.
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

from cvsrffi import stage2_zid_student_t_qknn as zid


LOCK_SCHEMA = "cvs.phase2.zid_srda.phase1_lock.v1"
GROUND_PROVENANCE_SCHEMA = "cvs.phase2.zid_srda.ground_provenance.v1"
GROUND_SCHEMA = "cvs.phase2.zid_srda.shared_covariance_prior.v1"
STATE_SCHEMA = "cvs.phase2.zid_srda.state.v1"
FUSION_SCHEMA = "cvs.phase2.zid_srda.fusion.v1"
WIRE_SCHEMA = "cvs.phase2.zid_srda.wire.v1"
WIRE_MAGIC = b"CVSZSRDA1\0"
ZERO_SHA256 = "0" * 64
Z_DIM = zid.Z_DIM
MAX_GROUND_RANK = 4
MAX_TARGET_RANK = 2
MAX_TOTAL_RANK = MAX_GROUND_RANK + MAX_TARGET_RANK
EPSILON = 1e-10
DIAGNOSTIC_AUTHORITY_SCOPE = "builder_reported_non_authoritative_not_for_promotion"


class ZIDSRDAError(ValueError):
    """Raised when the pure-B typed lifecycle or numerical contract drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: Any, name: str, *, allow_zero: bool = False) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ZIDSRDAError(f"{name} must be lowercase SHA256")
    if not allow_zero and text == ZERO_SHA256:
        raise ZIDSRDAError(f"{name} must not use the zero sentinel")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    array.setflags(write=False)
    return array


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _positive(value: Any, name: str) -> float:
    if type(value) is not float:
        raise ZIDSRDAError(f"{name} must be an exact float")
    number = value
    if not math.isfinite(number) or number <= 0.0:
        raise ZIDSRDAError(f"{name} must be finite and positive")
    return number


def _probability(value: Any, name: str) -> float:
    if type(value) is not float:
        raise ZIDSRDAError(f"{name} must be an exact float")
    number = value
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ZIDSRDAError(f"{name} must be in [0,1]")
    return number


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if len(classes) < 2 or len(set(classes)) != len(classes) or any(not value for value in classes):
        raise ZIDSRDAError("registered classes must be unique non-empty strings")
    return classes


@dataclass(frozen=True, slots=True)
class Phase1ZIDSRDALock:
    """K-specific RDA and fusion parameters sealed by Phase1 LODO.

    ``a_component_sha256`` is an external Git/release-handoff authority.  This
    nonformal core binds it into receipts but cannot self-attest its own source
    blob; the sealed admission layer must verify that lineage before release.
    """

    active_k: int
    sigma0_sq: float
    nu0: float
    target_rank: int
    lambda_relative: float
    rda_temperature: float
    ground_weight: float
    ground_prior_rank: int
    alpha_phase1: float
    a_component_sha256: str
    a_config_lock_digest: str
    a_identity_metric_receipt_sha256: str
    ground_prior_receipt_sha256: str
    ground_source_receipt_sha256: str
    phase1_lodo_receipt_sha256: str
    quantization_margin_audit_sha256: str
    schema: str = LOCK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != LOCK_SCHEMA
            or type(self.active_k) is not int
            or self.active_k not in zid.ALLOWED_K
            or type(self.target_rank) is not int
            or not 0 <= self.target_rank <= MAX_TARGET_RANK
            or type(self.ground_prior_rank) is not int
            or not 0 <= self.ground_prior_rank <= MAX_GROUND_RANK
        ):
            raise ZIDSRDAError("SRDA lock active K/rank drift")
        if self.active_k == 1 and self.target_rank != 0:
            raise ZIDSRDAError("K1 SRDA lock must request target rank zero")
        for value, name in (
            (self.sigma0_sq, "sigma0_sq"),
            (self.nu0, "nu0"),
            (self.lambda_relative, "lambda_relative"),
            (self.rda_temperature, "rda_temperature"),
        ):
            _positive(value, name)
        _probability(self.ground_weight, "ground_weight")
        _probability(self.alpha_phase1, "alpha_phase1")
        for value, name in (
            (self.a_component_sha256, "a_component_sha256"),
            (self.a_config_lock_digest, "a_config_lock_digest"),
            (
                self.a_identity_metric_receipt_sha256,
                "a_identity_metric_receipt_sha256",
            ),
            (self.phase1_lodo_receipt_sha256, "phase1_lodo_receipt_sha256"),
            (
                self.quantization_margin_audit_sha256,
                "quantization_margin_audit_sha256",
            ),
        ):
            _require_sha256(value, name)
        for value, name in (
            (self.ground_prior_receipt_sha256, "ground_prior_receipt_sha256"),
            (self.ground_source_receipt_sha256, "ground_source_receipt_sha256"),
        ):
            _require_sha256(value, name, allow_zero=True)
        ground_is_off = float(self.ground_weight) == 0.0
        ground_fields_are_zero = (
            self.ground_prior_receipt_sha256 == ZERO_SHA256
            and self.ground_source_receipt_sha256 == ZERO_SHA256
            and self.ground_prior_rank == 0
        )
        ground_fields_are_live = (
            self.ground_prior_receipt_sha256 != ZERO_SHA256
            and self.ground_source_receipt_sha256 != ZERO_SHA256
            and 1 <= self.ground_prior_rank <= MAX_GROUND_RANK
        )
        if (ground_is_off and not ground_fields_are_zero) or (
            not ground_is_off and not ground_fields_are_live
        ):
            raise ZIDSRDAError(
                "Phase1 ground weight and expected prior/source receipts must close"
            )

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class TypedZIDGroundPriorProvenance:
    """Typed Phase1-only provenance for a class-agnostic covariance prior."""

    source_receipt_sha256: str
    query_rows_used_for_fit: int = 0
    fit_scope: str = "phase1_lodo"
    schema: str = GROUND_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != GROUND_PROVENANCE_SCHEMA
            or self.fit_scope != "phase1_lodo"
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
        ):
            raise ZIDSRDAError("ground prior provenance must be Phase1 LODO only")
        _require_sha256(self.source_receipt_sha256, "ground provenance source receipt")

    @property
    def provenance_receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _ground_payload(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    spectrum: np.ndarray,
    provenance: TypedZIDGroundPriorProvenance,
) -> dict[str, Any]:
    return {
        "schema": GROUND_SCHEMA,
        "basis_codes_qint8": _array_receipt(codes),
        "basis_scales_fp16": _array_receipt(scales),
        "spectrum_fp16": _array_receipt(spectrum),
        "rank": int(codes.shape[1]),
        "provenance": asdict(provenance),
        "provenance_receipt_sha256": provenance.provenance_receipt_sha256,
        "class_agnostic": True,
        "contains_class_mean_or_bias": False,
        "query_rows_used_for_fit": 0,
    }


@dataclass(frozen=True, slots=True)
class TypedZIDSharedCovariancePrior:
    """Class-agnostic quantized Phase1 nuisance covariance basis."""

    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    spectrum_fp16: np.ndarray
    provenance: TypedZIDGroundPriorProvenance
    prior_receipt_sha256: str
    schema: str = GROUND_SCHEMA

    def __post_init__(self) -> None:
        codes = np.asarray(self.basis_codes_qint8)
        scales = np.asarray(self.basis_scales_fp16)
        spectrum = np.asarray(self.spectrum_fp16)
        if (
            self.schema != GROUND_SCHEMA
            or codes.dtype != np.int8
            or codes.ndim != 2
            or codes.shape[0] != Z_DIM
            or not 1 <= codes.shape[1] <= MAX_GROUND_RANK
            or scales.dtype != np.float16
            or scales.shape != (codes.shape[1],)
            or spectrum.dtype != np.float16
            or spectrum.shape != (codes.shape[1],)
            or np.any(codes == np.int8(-128))
            or not np.isfinite(scales).all()
            or not np.isfinite(spectrum).all()
            or np.any(scales <= 0.0)
            or np.any(spectrum <= 0.0)
        ):
            raise ZIDSRDAError("shared covariance prior array/rank drift")
        if type(self.provenance) is not TypedZIDGroundPriorProvenance:
            raise ZIDSRDAError("ground prior requires exact typed provenance")
        _require_sha256(self.prior_receipt_sha256, "ground prior receipt")
        payload = _ground_payload(
            codes=codes,
            scales=scales,
            spectrum=spectrum,
            provenance=self.provenance,
        )
        if _canonical_sha256(payload) != self.prior_receipt_sha256:
            raise ZIDSRDAError("ground prior receipt verification failed")
        decoded = _decode_ground_basis_arrays(codes, scales)
        gram = decoded.T @ decoded
        if not np.allclose(gram, np.eye(codes.shape[1]), atol=2e-2, rtol=0.0):
            raise ZIDSRDAError("quantized ground prior basis lost orthogonality")
        object.__setattr__(self, "basis_codes_qint8", _readonly(codes, np.int8))
        object.__setattr__(self, "basis_scales_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "spectrum_fp16", _readonly(spectrum, np.float16))

    @property
    def rank(self) -> int:
        return int(self.basis_codes_qint8.shape[1])


def _decode_ground_basis_arrays(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    decoded = codes.astype(np.float64) * scales.astype(np.float64)[None, :]
    norms = np.linalg.norm(decoded, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= EPSILON):
        raise ZIDSRDAError("ground prior basis contains a degenerate vector")
    return decoded / norms[None, :]


def _verify_ground_prior(prior: TypedZIDSharedCovariancePrior) -> None:
    """Revalidate mutable-in-memory arrays and the sealed Phase1 provenance."""

    if type(prior) is not TypedZIDSharedCovariancePrior:
        raise ZIDSRDAError("ground prior verification requires exact typed state")
    codes = np.asarray(prior.basis_codes_qint8)
    scales = np.asarray(prior.basis_scales_fp16)
    spectrum = np.asarray(prior.spectrum_fp16)
    provenance = prior.provenance
    if (
        prior.schema != GROUND_SCHEMA
        or codes.dtype != np.int8
        or codes.ndim != 2
        or codes.shape[0] != Z_DIM
        or not 1 <= codes.shape[1] <= MAX_GROUND_RANK
        or scales.dtype != np.float16
        or scales.shape != (codes.shape[1],)
        or spectrum.dtype != np.float16
        or spectrum.shape != (codes.shape[1],)
        or np.any(codes == np.int8(-128))
        or not np.isfinite(scales).all()
        or not np.isfinite(spectrum).all()
        or np.any(scales <= 0.0)
        or np.any(spectrum <= 0.0)
        or type(provenance) is not TypedZIDGroundPriorProvenance
    ):
        raise ZIDSRDAError("shared covariance prior array/rank drift")
    if (
        provenance.schema != GROUND_PROVENANCE_SCHEMA
        or provenance.fit_scope != "phase1_lodo"
        or type(provenance.query_rows_used_for_fit) is not int
        or provenance.query_rows_used_for_fit != 0
    ):
        raise ZIDSRDAError("ground prior provenance drift")
    _require_sha256(provenance.source_receipt_sha256, "ground provenance source receipt")
    _require_sha256(prior.prior_receipt_sha256, "ground prior receipt")
    payload = _ground_payload(
        codes=codes, scales=scales, spectrum=spectrum, provenance=provenance
    )
    if _canonical_sha256(payload) != prior.prior_receipt_sha256:
        raise ZIDSRDAError("ground prior receipt verification failed")
    decoded = _decode_ground_basis_arrays(codes, scales)
    if not np.allclose(
        decoded.T @ decoded, np.eye(codes.shape[1]), atol=2e-2, rtol=0.0
    ):
        raise ZIDSRDAError("quantized ground prior basis lost orthogonality")


def decode_shared_covariance_prior(prior: TypedZIDSharedCovariancePrior) -> np.ndarray:
    if type(prior) is not TypedZIDSharedCovariancePrior:
        raise ZIDSRDAError("ground prior decode requires exact typed state")
    _verify_ground_prior(prior)
    return _readonly(
        _decode_ground_basis_arrays(prior.basis_codes_qint8, prior.basis_scales_fp16),
        np.float64,
    )


def build_typed_shared_covariance_prior(
    basis_fp32: np.ndarray,
    spectrum_fp32: np.ndarray,
    *,
    provenance: TypedZIDGroundPriorProvenance,
) -> TypedZIDSharedCovariancePrior:
    """Quantize a Phase1-provided orthonormal class-agnostic basis."""

    basis = np.asarray(basis_fp32)
    spectrum = np.asarray(spectrum_fp32)
    if type(provenance) is not TypedZIDGroundPriorProvenance:
        raise ZIDSRDAError("ground prior builder requires exact typed provenance")
    if (
        basis.dtype != np.float32
        or basis.ndim != 2
        or basis.shape[0] != Z_DIM
        or not 1 <= basis.shape[1] <= MAX_GROUND_RANK
        or spectrum.dtype != np.float32
        or spectrum.shape != (basis.shape[1],)
        or not np.isfinite(basis).all()
        or not np.isfinite(spectrum).all()
        or np.any(spectrum <= 0.0)
    ):
        raise ZIDSRDAError("ground prior source array/rank drift")
    gram = basis.astype(np.float64).T @ basis.astype(np.float64)
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-5, rtol=0.0):
        raise ZIDSRDAError("ground prior source basis must be orthonormal")
    scales = np.max(np.abs(basis), axis=0).astype(np.float64) / 127.0
    if np.any(scales <= 0.0):
        raise ZIDSRDAError("ground prior source contains a zero basis vector")
    scales16 = np.asarray(scales, dtype=np.float16)
    codes = np.clip(
        np.rint(basis.astype(np.float64) / scales16.astype(np.float64)[None, :]),
        -127,
        127,
    ).astype(np.int8)
    spectrum16 = np.asarray(spectrum, dtype=np.float16)
    payload = _ground_payload(
        codes=codes,
        scales=scales16,
        spectrum=spectrum16,
        provenance=provenance,
    )
    return TypedZIDSharedCovariancePrior(
        basis_codes_qint8=_readonly(codes, np.int8),
        basis_scales_fp16=_readonly(scales16, np.float16),
        spectrum_fp16=_readonly(spectrum16, np.float16),
        provenance=provenance,
        prior_receipt_sha256=_canonical_sha256(payload),
    )


def _class_balanced_scatter(
    support: np.ndarray,
    class_indices: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    means = np.stack(
        [np.mean(support[class_indices == index], axis=0) for index in range(class_count)]
    ).astype(np.float64)
    residual = support.astype(np.float64) - means[class_indices]
    if k_shot == 1:
        covariance = np.zeros((Z_DIM, Z_DIM), dtype=np.float64)
        nres = 0
    else:
        covariance = residual.T @ residual / float(class_count * (k_shot - 1))
        covariance = 0.5 * (covariance + covariance.T)
        nres = class_count * (k_shot - 1)
    return means, residual, covariance, nres


def woodbury_precision_apply(
    rows: np.ndarray, diagonal: float, factor: np.ndarray
) -> np.ndarray:
    """Apply (diagonal*I + factor*factor.T)^-1 to row vectors."""

    value = np.asarray(rows, dtype=np.float64)
    low_rank = np.asarray(factor, dtype=np.float64)
    scalar = float(diagonal)
    if (
        value.ndim != 2
        or value.shape[1] != Z_DIM
        or low_rank.ndim != 2
        or low_rank.shape[0] != Z_DIM
        or low_rank.shape[1] > MAX_TOTAL_RANK
        or not np.isfinite(value).all()
        or not np.isfinite(low_rank).all()
        or not math.isfinite(scalar)
        or scalar <= 0.0
    ):
        raise ZIDSRDAError("Woodbury input/diagonal/rank drift")
    base = value / scalar
    if low_rank.shape[1] == 0:
        return base
    scaled = low_rank / scalar
    middle = np.eye(low_rank.shape[1]) + low_rank.T @ scaled
    correction = (base @ low_rank) @ np.linalg.solve(middle, scaled.T)
    result = base - correction
    if not np.isfinite(result).all():
        raise ZIDSRDAError("Woodbury precision application became non-finite")
    return result


def _quantize_weight_rows(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64)
    maximum = np.max(np.abs(values), axis=1)
    if np.any(maximum <= EPSILON) or not np.isfinite(maximum).all():
        raise ZIDSRDAError("RDA weight row is degenerate")
    scales16 = np.asarray(maximum / 127.0, dtype=np.float16)
    if np.any(scales16 <= 0.0) or not np.isfinite(scales16).all():
        raise ZIDSRDAError("RDA weight scale cannot be represented in FP16")
    codes = np.clip(
        np.rint(values / scales16.astype(np.float64)[:, None]), -127, 127
    ).astype(np.int8)
    decoded = codes.astype(np.float64) * scales16.astype(np.float64)[:, None]
    return codes, scales16, decoded


def _state_payload(
    *,
    classes: tuple[str, ...],
    active_k: int,
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray,
    alpha: float,
    a_bank_receipt_sha256: str,
    a_config_lock_digest: str,
    a_metric_receipt_sha256: str,
    ground_prior_receipt_sha256: str,
    lock: Phase1ZIDSRDALock,
    fit_audit: Mapping[str, Any],
    quantization_audit: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "classes": list(classes),
        "active_k": int(active_k),
        "weight_codes_qint8": _array_receipt(codes),
        "weight_scales_fp16": _array_receipt(scales),
        "bias_fp16": _array_receipt(bias),
        "alpha_phase1": float(alpha),
        "rda_temperature": float(lock.rda_temperature),
        "a_bank_receipt_sha256": a_bank_receipt_sha256,
        "a_config_lock_digest": a_config_lock_digest,
        "a_metric_receipt_sha256": a_metric_receipt_sha256,
        "a_component_sha256": lock.a_component_sha256,
        "ground_prior_receipt_sha256": ground_prior_receipt_sha256,
        "phase1_lodo_receipt_sha256": lock.phase1_lodo_receipt_sha256,
        "lock": asdict(lock),
        "lock_digest": lock.lock_digest,
        "fit_audit": _json_value(fit_audit),
        "quantization_audit": _json_value(quantization_audit),
        "resource_audit": _json_value(resource_audit),
        "diagnostic_authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "feature_space": "z_id160_only",
        "metric_component": "Patch_A_identity_rank0",
        "alpha_source": "phase1_lodo_only",
        "target_support_crossfit": False,
        "same_formula_all_registered_classes": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "formal_phase2_eligible": False,
        "bundle_created": False,
    }


@dataclass(frozen=True, slots=True)
class TypedZIDSRDAState:
    classes: tuple[str, ...]
    active_k: int
    weight_codes_qint8: np.ndarray
    weight_scales_fp16: np.ndarray
    bias_fp16: np.ndarray
    alpha_phase1: float
    rda_temperature: float
    a_bank_receipt_sha256: str
    a_config_lock_digest: str
    a_metric_receipt_sha256: str
    ground_prior_receipt_sha256: str
    lock: Phase1ZIDSRDALock
    fit_audit: Mapping[str, Any]
    quantization_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    state_receipt_sha256: str
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        codes = np.asarray(self.weight_codes_qint8)
        scales = np.asarray(self.weight_scales_fp16)
        bias = np.asarray(self.bias_fp16)
        if type(self.lock) is not Phase1ZIDSRDALock:
            raise ZIDSRDAError("SRDA state requires exact Phase1 lock")
        if (
            self.schema != STATE_SCHEMA
            or type(self.active_k) is not int
            or self.active_k != self.lock.active_k
            or type(self.alpha_phase1) is not float
            or type(self.rda_temperature) is not float
            or codes.dtype != np.int8
            or codes.shape != (len(classes), Z_DIM)
            or scales.dtype != np.float16
            or scales.shape != (len(classes),)
            or bias.dtype != np.float16
            or bias.shape != (len(classes),)
            or np.any(codes == np.int8(-128))
            or not np.isfinite(scales).all()
            or not np.isfinite(bias).all()
            or np.any(scales <= 0.0)
        ):
            raise ZIDSRDAError("SRDA state array/class invariant drift")
        if float(self.alpha_phase1) != float(self.lock.alpha_phase1):
            raise ZIDSRDAError("SRDA state alpha drift")
        if float(self.rda_temperature) != float(self.lock.rda_temperature):
            raise ZIDSRDAError("SRDA state temperature drift")
        for value, name, allow_zero in (
            (self.a_bank_receipt_sha256, "A bank receipt", False),
            (self.a_config_lock_digest, "A config lock", False),
            (self.a_metric_receipt_sha256, "A metric receipt", False),
            (self.ground_prior_receipt_sha256, "ground prior receipt", True),
            (self.state_receipt_sha256, "SRDA state receipt", False),
        ):
            _require_sha256(value, name, allow_zero=allow_zero)
        fit = dict(self.fit_audit)
        quantization = dict(self.quantization_audit)
        resource = dict(self.resource_audit)
        payload = _state_payload(
            classes=classes,
            active_k=self.active_k,
            codes=codes,
            scales=scales,
            bias=bias,
            alpha=self.alpha_phase1,
            a_bank_receipt_sha256=self.a_bank_receipt_sha256,
            a_config_lock_digest=self.a_config_lock_digest,
            a_metric_receipt_sha256=self.a_metric_receipt_sha256,
            ground_prior_receipt_sha256=self.ground_prior_receipt_sha256,
            lock=self.lock,
            fit_audit=fit,
            quantization_audit=quantization,
            resource_audit=resource,
        )
        if _canonical_sha256(payload) != self.state_receipt_sha256:
            raise ZIDSRDAError("SRDA state receipt verification failed")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "weight_codes_qint8", _readonly(codes, np.int8))
        object.__setattr__(self, "weight_scales_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "bias_fp16", _readonly(bias, np.float16))
        object.__setattr__(self, "fit_audit", MappingProxyType(fit))
        object.__setattr__(self, "quantization_audit", MappingProxyType(quantization))
        object.__setattr__(self, "resource_audit", MappingProxyType(resource))
        _verify_state(self)


_FIT_AUDIT_FIELDS = {
    "schema",
    "authority_scope",
    "support_rows",
    "class_count",
    "active_k",
    "n_residual_degrees",
    "target_scatter_exact_zero",
    "target_rank_requested",
    "target_rank_actual",
    "ground_rank_actual",
    "woodbury_rank_total",
    "shrinkage",
    "sigma_support_sq",
    "sigma_base_sq",
    "ridge",
    "diagonal",
    "positive_target_eigenvalues",
    "class_balanced_scatter",
    "class_means_source",
    "class_means_renormalized",
    "equal_class_prior",
    "ground_class_mean_or_logit_access",
    "ground_prior_receipt_sha256",
    "ground_source_receipt_sha256",
    "alpha_source",
    "target_support_crossfit",
    "query_rows_used_for_fit",
}
_QUANTIZATION_AUDIT_FIELDS = {
    "schema",
    "authority_scope",
    "support_rows",
    "class_count",
    "support_fit_top1_agreement",
    "support_fit_top1_flip_count",
    "support_fit_max_logit_abs_error",
    "teacher_margin_min",
    "deployed_margin_min",
    "teacher_margin_sign_flip_count",
    "large_margin_flip_count",
    "support_fit_only_not_phase1_authority",
    "phase1_margin_authority_sha256",
    "same_formula_all_registered_classes",
    "query_rows_used",
}
_RESOURCE_AUDIT_FIELDS = {
    "schema",
    "authority_scope",
    "incremental_numeric_array_state_bytes",
    "incremental_rda_linear_matmul_mac_per_query",
    "incremental_query_matmul_scope",
    "alpha_zero_skips_rda_query_branch",
    "fit_transient_numpy_array_bytes_estimate",
    "persistent_decoded_cache_bytes",
    "trainable_parameters",
    "epochs",
    "optimizer_steps",
    "query_state_updates",
    "query_batch_dependency",
    "dense_query_graph",
    "query_dependent_batch_optimization",
    "complete_combined_wire_container_available",
    "latency_measurement_included",
    "not_end_to_end_mac_or_latency",
}


def _exact_audit(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ZIDSRDAError(f"{name} must be a mapping")
    result = dict(value)
    if set(result) != fields:
        raise ZIDSRDAError(f"{name} field set drift")
    return result


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise ZIDSRDAError(f"{name} must be an exact float")
    number = value
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ZIDSRDAError(f"{name} must be finite and within range")
    return number


def _verify_state(state: TypedZIDSRDAState) -> None:
    """Recompute state receipts and audit semantics at every public consumer."""

    if type(state) is not TypedZIDSRDAState or type(state.lock) is not Phase1ZIDSRDALock:
        raise ZIDSRDAError("SRDA verification requires exact typed state and lock")
    classes = _registry(state.classes)
    class_count = len(classes)
    codes = np.asarray(state.weight_codes_qint8)
    scales = np.asarray(state.weight_scales_fp16)
    bias = np.asarray(state.bias_fp16)
    if (
        state.schema != STATE_SCHEMA
        or type(state.active_k) is not int
        or state.active_k != state.lock.active_k
        or type(state.alpha_phase1) is not float
        or type(state.rda_temperature) is not float
        or codes.dtype != np.int8
        or codes.shape != (class_count, Z_DIM)
        or scales.dtype != np.float16
        or scales.shape != (class_count,)
        or bias.dtype != np.float16
        or bias.shape != (class_count,)
        or np.any(codes == np.int8(-128))
        or not np.isfinite(scales).all()
        or not np.isfinite(bias).all()
        or np.any(scales <= 0.0)
    ):
        raise ZIDSRDAError("SRDA state array/class invariant drift")
    if (
        float(state.alpha_phase1) != float(state.lock.alpha_phase1)
        or float(state.rda_temperature) != float(state.lock.rda_temperature)
        or state.a_config_lock_digest != state.lock.a_config_lock_digest
        or state.a_metric_receipt_sha256
        != state.lock.a_identity_metric_receipt_sha256
        or state.ground_prior_receipt_sha256
        != state.lock.ground_prior_receipt_sha256
    ):
        raise ZIDSRDAError("SRDA state/Phase1 lock binding drift")
    for value, name, allow_zero in (
        (state.a_bank_receipt_sha256, "A bank receipt", False),
        (state.a_config_lock_digest, "A config lock", False),
        (state.a_metric_receipt_sha256, "A metric receipt", False),
        (state.ground_prior_receipt_sha256, "ground prior receipt", True),
        (state.state_receipt_sha256, "SRDA state receipt", False),
    ):
        _require_sha256(value, name, allow_zero=allow_zero)

    fit = _exact_audit(state.fit_audit, _FIT_AUDIT_FIELDS, "fit audit")
    quant = _exact_audit(
        state.quantization_audit, _QUANTIZATION_AUDIT_FIELDS, "quantization audit"
    )
    resource = _exact_audit(
        state.resource_audit, _RESOURCE_AUDIT_FIELDS, "resource audit"
    )
    expected_nres = class_count * (state.active_k - 1)
    target_actual = fit["target_rank_actual"]
    ground_actual = fit["ground_rank_actual"]
    if (
        fit["schema"] != "cvs.phase2.zid_srda.fit_audit.v1"
        or fit["authority_scope"] != DIAGNOSTIC_AUTHORITY_SCOPE
        or type(fit["support_rows"]) is not int
        or fit["support_rows"] != class_count * state.active_k
        or type(fit["class_count"]) is not int
        or fit["class_count"] != class_count
        or type(fit["active_k"]) is not int
        or fit["active_k"] != state.active_k
        or type(fit["n_residual_degrees"]) is not int
        or fit["n_residual_degrees"] != expected_nres
        or type(fit["target_rank_requested"]) is not int
        or type(target_actual) is not int
        or not 0 <= target_actual <= state.lock.target_rank <= MAX_TARGET_RANK
        or type(ground_actual) is not int
        or ground_actual != state.lock.ground_prior_rank
        or type(fit["woodbury_rank_total"]) is not int
        or fit["woodbury_rank_total"] != target_actual + ground_actual
        or fit["woodbury_rank_total"] > MAX_TOTAL_RANK
        or fit["target_rank_requested"] != state.lock.target_rank
        or fit["target_scatter_exact_zero"] is not (state.active_k == 1)
        or (state.active_k == 1 and target_actual != 0)
        or fit["class_balanced_scatter"] is not True
        or fit["class_means_source"]
        != "decoded_current_row_target_support_only"
        or fit["class_means_renormalized"] is not False
        or fit["equal_class_prior"] is not True
        or fit["ground_class_mean_or_logit_access"] is not False
        or fit["ground_prior_receipt_sha256"]
        != state.lock.ground_prior_receipt_sha256
        or fit["ground_source_receipt_sha256"]
        != state.lock.ground_source_receipt_sha256
        or fit["alpha_source"] != "phase1_lodo_only"
        or fit["target_support_crossfit"] is not False
        or type(fit["query_rows_used_for_fit"]) is not int
        or fit["query_rows_used_for_fit"] != 0
    ):
        raise ZIDSRDAError("SRDA fit audit semantic drift")
    shrinkage_expected = expected_nres / (float(state.lock.nu0) + expected_nres) if expected_nres else 0.0
    if not math.isclose(
        _finite_number(fit["shrinkage"], "shrinkage", minimum=0.0),
        shrinkage_expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ZIDSRDAError("SRDA shrinkage audit drift")
    for key in ("sigma_support_sq", "sigma_base_sq", "ridge"):
        _finite_number(fit[key], key, minimum=0.0)
    _positive(fit["diagonal"], "fit diagonal")
    eigenvalues = fit["positive_target_eigenvalues"]
    if (
        type(eigenvalues) is not list
        or len(eigenvalues) != target_actual
        or any(_finite_number(value, "positive target eigenvalue") <= 0.0 for value in eigenvalues)
    ):
        raise ZIDSRDAError("positive target eigenvalue audit drift")

    support_rows = class_count * state.active_k
    flips = quant["support_fit_top1_flip_count"]
    if (
        quant["schema"] != "cvs.phase2.zid_srda.quantization_audit.v1"
        or quant["authority_scope"] != DIAGNOSTIC_AUTHORITY_SCOPE
        or type(quant["support_rows"]) is not int
        or quant["support_rows"] != support_rows
        or type(quant["class_count"]) is not int
        or quant["class_count"] != class_count
        or type(flips) is not int
        or not 0 <= flips <= support_rows
        or not math.isclose(
            _probability(quant["support_fit_top1_agreement"], "support agreement"),
            (support_rows - flips) / support_rows,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _finite_number(
            quant["support_fit_max_logit_abs_error"], "max logit error", minimum=0.0
        )
        < 0.0
        or type(quant["teacher_margin_sign_flip_count"]) is not int
        or not 0 <= quant["teacher_margin_sign_flip_count"] <= support_rows
        or type(quant["large_margin_flip_count"]) is not int
        or not 0 <= quant["large_margin_flip_count"] <= support_rows
        or quant["support_fit_only_not_phase1_authority"] is not True
        or quant["phase1_margin_authority_sha256"]
        != state.lock.quantization_margin_audit_sha256
        or quant["same_formula_all_registered_classes"] is not True
        or type(quant["query_rows_used"]) is not int
        or quant["query_rows_used"] != 0
    ):
        raise ZIDSRDAError("SRDA quantization audit semantic drift")
    _finite_number(quant["teacher_margin_min"], "teacher margin")
    _finite_number(quant["deployed_margin_min"], "deployed margin")

    numeric_bytes = int(codes.nbytes + scales.nbytes + bias.nbytes)
    expected_mac = 0 if state.alpha_phase1 == 0.0 else class_count * Z_DIM
    expected_transient = int(
        support_rows * Z_DIM * 8
        + class_count * Z_DIM * 8
        + support_rows * Z_DIM * 8
        + 2 * Z_DIM * Z_DIM * 8
        + Z_DIM * (target_actual + ground_actual) * 8
        + class_count * Z_DIM * 8
    )
    integer_resource_fields = (
        "incremental_numeric_array_state_bytes",
        "incremental_rda_linear_matmul_mac_per_query",
        "fit_transient_numpy_array_bytes_estimate",
        "persistent_decoded_cache_bytes",
        "trainable_parameters",
        "epochs",
        "optimizer_steps",
        "query_state_updates",
    )
    if (
        resource["schema"] != "cvs.phase2.zid_srda.resource_audit.v1"
        or resource["authority_scope"] != DIAGNOSTIC_AUTHORITY_SCOPE
        or any(type(resource[key]) is not int for key in integer_resource_fields)
        or resource["incremental_numeric_array_state_bytes"] != numeric_bytes
        or resource["incremental_rda_linear_matmul_mac_per_query"] != expected_mac
        or resource["incremental_query_matmul_scope"]
        != "compiled RDA linear head only; excludes bias, softmax and fusion"
        or resource["alpha_zero_skips_rda_query_branch"]
        is not (state.alpha_phase1 == 0.0)
        or resource["fit_transient_numpy_array_bytes_estimate"] != expected_transient
        or resource["persistent_decoded_cache_bytes"] != 0
        or resource["trainable_parameters"] != 0
        or resource["epochs"] != 0
        or resource["optimizer_steps"] != 0
        or resource["query_state_updates"] != 0
        or resource["query_batch_dependency"] is not False
        or resource["dense_query_graph"] is not False
        or resource["query_dependent_batch_optimization"] is not False
        or resource["complete_combined_wire_container_available"] is not False
        or resource["latency_measurement_included"] is not False
        or resource["not_end_to_end_mac_or_latency"] is not True
    ):
        raise ZIDSRDAError("SRDA resource audit semantic drift")

    payload = _state_payload(
        classes=classes,
        active_k=state.active_k,
        codes=codes,
        scales=scales,
        bias=bias,
        alpha=state.alpha_phase1,
        a_bank_receipt_sha256=state.a_bank_receipt_sha256,
        a_config_lock_digest=state.a_config_lock_digest,
        a_metric_receipt_sha256=state.a_metric_receipt_sha256,
        ground_prior_receipt_sha256=state.ground_prior_receipt_sha256,
        lock=state.lock,
        fit_audit=fit,
        quantization_audit=quant,
        resource_audit=resource,
    )
    if _canonical_sha256(payload) != state.state_receipt_sha256:
        raise ZIDSRDAError("SRDA state receipt verification failed")


def _verify_bank_state_binding(
    bank: zid.TypedINT8ZIDSupportBank, state: TypedZIDSRDAState
) -> tuple[zid.TypedSharedPSDMetric, bytes]:
    if type(bank) is not zid.TypedINT8ZIDSupportBank:
        raise ZIDSRDAError("A/B binding requires exact Patch-A bank")
    _verify_state(state)
    identity = zid.identity_shared_psd_metric(config=bank.config)
    if (
        bank.classes != state.classes
        or bank.active_k != state.active_k
        or bank.bank_receipt_sha256 != state.a_bank_receipt_sha256
        or bank.config_lock_digest != state.a_config_lock_digest
        or bank.config_lock_digest != state.lock.a_config_lock_digest
        or identity.metric_receipt_sha256 != state.a_metric_receipt_sha256
        or identity.metric_receipt_sha256
        != state.lock.a_identity_metric_receipt_sha256
    ):
        raise ZIDSRDAError("A bank/B state binding drift")
    a_wire = zid.serialize_typed_zid_runtime_state(bank, identity)
    return identity, a_wire


def build_zid_srda_state(
    bank: zid.TypedINT8ZIDSupportBank,
    ground_prior: TypedZIDSharedCovariancePrior | None,
    *,
    lock: Phase1ZIDSRDALock,
) -> TypedZIDSRDAState:
    """Build one support-only compiled RDA head; no query argument exists."""

    if type(bank) is not zid.TypedINT8ZIDSupportBank or type(lock) is not Phase1ZIDSRDALock:
        raise ZIDSRDAError("SRDA build requires exact Patch-A bank and Phase1 lock")
    if bank.active_k != lock.active_k:
        raise ZIDSRDAError("SRDA bank/lock active K drift")
    if bank.config_lock_digest != lock.a_config_lock_digest:
        raise ZIDSRDAError("SRDA bank does not match the Phase1-locked A config")
    if ground_prior is not None and type(ground_prior) is not TypedZIDSharedCovariancePrior:
        raise ZIDSRDAError("SRDA ground prior must be exact typed state or None")
    if (ground_prior is None) != (float(lock.ground_weight) == 0.0):
        raise ZIDSRDAError(
            "ground prior presence must exactly match a nonzero Phase1 ground weight"
        )
    identity_metric = zid.identity_shared_psd_metric(config=bank.config)
    if identity_metric.metric_receipt_sha256 != lock.a_identity_metric_receipt_sha256:
        raise ZIDSRDAError("SRDA identity metric does not match the Phase1 lock")
    if ground_prior is None:
        if (
            lock.ground_prior_receipt_sha256 != ZERO_SHA256
            or lock.ground_source_receipt_sha256 != ZERO_SHA256
            or lock.ground_prior_rank != 0
        ):
            raise ZIDSRDAError("ground-off lock closure drift")
    else:
        _verify_ground_prior(ground_prior)
        if (
            ground_prior.prior_receipt_sha256
            != lock.ground_prior_receipt_sha256
            or ground_prior.provenance.source_receipt_sha256
            != lock.ground_source_receipt_sha256
            or ground_prior.rank != lock.ground_prior_rank
        ):
            raise ZIDSRDAError("ground prior/source/rank does not match Phase1 lock")
    support = zid.decode_zid_support_bank(bank).astype(np.float64)
    indices = bank.class_indices_int16.astype(np.int64)
    class_count = len(bank.classes)
    means, residual, support_covariance, nres = _class_balanced_scatter(
        support, indices, class_count, bank.active_k
    )
    if bank.active_k == 1 and (
        nres != 0 or np.any(residual != 0.0) or np.any(support_covariance != 0.0)
    ):
        raise AssertionError("K1 target residual covariance must be exactly zero")
    shrinkage = float(nres / (float(lock.nu0) + nres)) if nres else 0.0
    sigma_sup_sq = float(np.trace(support_covariance) / Z_DIM) if nres else 0.0
    sigma_base_sq = (1.0 - shrinkage) * float(lock.sigma0_sq) + shrinkage * sigma_sup_sq
    centered = support_covariance - sigma_sup_sq * np.eye(Z_DIM)
    if bank.active_k == 1 or lock.target_rank == 0:
        target_factor = np.zeros((Z_DIM, 0), dtype=np.float64)
        positive_eigenvalues = np.empty(0, dtype=np.float64)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(centered)
        order = np.argsort(eigenvalues)[::-1]
        positive = [index for index in order if eigenvalues[index] > 1e-10]
        selected = positive[: lock.target_rank]
        positive_eigenvalues = np.asarray(
            [eigenvalues[index] for index in selected], dtype=np.float64
        )
        if selected:
            target_factor = eigenvectors[:, selected] * np.sqrt(
                shrinkage * positive_eigenvalues
            )[None, :]
        else:
            target_factor = np.zeros((Z_DIM, 0), dtype=np.float64)
    if ground_prior is None or lock.ground_weight == 0.0:
        ground_factor = np.zeros((Z_DIM, 0), dtype=np.float64)
        ground_receipt = ZERO_SHA256
    else:
        basis = decode_shared_covariance_prior(ground_prior)
        spectrum = ground_prior.spectrum_fp16.astype(np.float64)
        ground_factor = basis * np.sqrt(
            (1.0 - shrinkage) * float(lock.ground_weight) * spectrum
        )[None, :]
        ground_receipt = ground_prior.prior_receipt_sha256
    factor = np.concatenate((ground_factor, target_factor), axis=1)
    if factor.shape[1] > MAX_TOTAL_RANK:
        raise AssertionError("SRDA total Woodbury rank exceeded six")
    covariance_trace_without_ridge = Z_DIM * sigma_base_sq + float(np.sum(factor * factor))
    ridge = float(lock.lambda_relative) * covariance_trace_without_ridge / Z_DIM
    diagonal = sigma_base_sq + ridge
    _positive(diagonal, "SRDA isotropic diagonal")
    weights = woodbury_precision_apply(means, diagonal, factor)
    teacher_bias = -0.5 * np.sum(means * weights, axis=1)
    codes, scales, decoded_weights = _quantize_weight_rows(weights)
    bias16 = np.asarray(teacher_bias, dtype=np.float16)
    if not np.isfinite(bias16).all():
        raise ZIDSRDAError("RDA bias cannot be represented in FP16")
    teacher_logits = support @ weights.T + teacher_bias[None, :]
    deployed_logits = support @ decoded_weights.T + bias16.astype(np.float64)[None, :]
    teacher_winner = np.argmax(teacher_logits, axis=1)
    deployed_winner = np.argmax(deployed_logits, axis=1)
    row_indices = np.arange(len(support))
    teacher_other = teacher_logits.copy()
    teacher_other[row_indices, indices] = -np.inf
    deployed_other = deployed_logits.copy()
    deployed_other[row_indices, indices] = -np.inf
    teacher_margin = teacher_logits[row_indices, indices] - np.max(teacher_other, axis=1)
    deployed_margin = deployed_logits[row_indices, indices] - np.max(deployed_other, axis=1)
    top1_flip_count = int(np.count_nonzero(teacher_winner != deployed_winner))
    fit_audit = {
        "schema": "cvs.phase2.zid_srda.fit_audit.v1",
        "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "support_rows": int(len(support)),
        "class_count": class_count,
        "active_k": bank.active_k,
        "n_residual_degrees": int(nres),
        "target_scatter_exact_zero": bool(bank.active_k == 1 and np.all(support_covariance == 0.0)),
        "target_rank_requested": int(lock.target_rank),
        "target_rank_actual": int(target_factor.shape[1]),
        "ground_rank_actual": int(ground_factor.shape[1]),
        "woodbury_rank_total": int(factor.shape[1]),
        "shrinkage": shrinkage,
        "sigma_support_sq": sigma_sup_sq,
        "sigma_base_sq": sigma_base_sq,
        "ridge": ridge,
        "diagonal": diagonal,
        "positive_target_eigenvalues": positive_eigenvalues.tolist(),
        "class_balanced_scatter": True,
        "class_means_source": "decoded_current_row_target_support_only",
        "class_means_renormalized": False,
        "equal_class_prior": True,
        "ground_class_mean_or_logit_access": False,
        "ground_prior_receipt_sha256": ground_receipt,
        "ground_source_receipt_sha256": lock.ground_source_receipt_sha256,
        "alpha_source": "phase1_lodo_only",
        "target_support_crossfit": False,
        "query_rows_used_for_fit": 0,
    }
    quantization_audit = {
        "schema": "cvs.phase2.zid_srda.quantization_audit.v1",
        "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "support_rows": int(len(support)),
        "class_count": class_count,
        "support_fit_top1_agreement": float(np.mean(teacher_winner == deployed_winner)),
        "support_fit_top1_flip_count": top1_flip_count,
        "support_fit_max_logit_abs_error": float(np.max(np.abs(teacher_logits - deployed_logits))),
        "teacher_margin_min": float(np.min(teacher_margin)),
        "deployed_margin_min": float(np.min(deployed_margin)),
        "teacher_margin_sign_flip_count": int(
            np.count_nonzero(np.signbit(teacher_margin) != np.signbit(deployed_margin))
        ),
        "large_margin_flip_count": int(
            np.count_nonzero((teacher_margin >= 0.25) & (deployed_margin <= 0.0))
        ),
        "support_fit_only_not_phase1_authority": True,
        "phase1_margin_authority_sha256": lock.quantization_margin_audit_sha256,
        "same_formula_all_registered_classes": True,
        "query_rows_used": 0,
    }
    numeric_bytes = int(codes.nbytes + scales.nbytes + bias16.nbytes)
    fit_transient = int(
        support.nbytes
        + means.nbytes
        + residual.nbytes
        + support_covariance.nbytes
        + centered.nbytes
        + factor.nbytes
        + weights.nbytes
    )
    resource_audit = {
        "schema": "cvs.phase2.zid_srda.resource_audit.v1",
        "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "incremental_numeric_array_state_bytes": numeric_bytes,
        "incremental_rda_linear_matmul_mac_per_query": (
            0 if lock.alpha_phase1 == 0.0 else class_count * Z_DIM
        ),
        "incremental_query_matmul_scope": "compiled RDA linear head only; excludes bias, softmax and fusion",
        "alpha_zero_skips_rda_query_branch": bool(lock.alpha_phase1 == 0.0),
        "fit_transient_numpy_array_bytes_estimate": fit_transient,
        "persistent_decoded_cache_bytes": 0,
        "trainable_parameters": 0,
        "epochs": 0,
        "optimizer_steps": 0,
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "dense_query_graph": False,
        "query_dependent_batch_optimization": False,
        "complete_combined_wire_container_available": False,
        "latency_measurement_included": False,
        "not_end_to_end_mac_or_latency": True,
    }
    placeholder_payload = _state_payload(
        classes=bank.classes,
        active_k=bank.active_k,
        codes=codes,
        scales=scales,
        bias=bias16,
        alpha=lock.alpha_phase1,
        a_bank_receipt_sha256=bank.bank_receipt_sha256,
        a_config_lock_digest=bank.config_lock_digest,
        a_metric_receipt_sha256=identity_metric.metric_receipt_sha256,
        ground_prior_receipt_sha256=ground_receipt,
        lock=lock,
        fit_audit=fit_audit,
        quantization_audit=quantization_audit,
        resource_audit=resource_audit,
    )
    return TypedZIDSRDAState(
        classes=bank.classes,
        active_k=bank.active_k,
        weight_codes_qint8=_readonly(codes, np.int8),
        weight_scales_fp16=_readonly(scales, np.float16),
        bias_fp16=_readonly(bias16, np.float16),
        alpha_phase1=lock.alpha_phase1,
        rda_temperature=lock.rda_temperature,
        a_bank_receipt_sha256=bank.bank_receipt_sha256,
        a_config_lock_digest=bank.config_lock_digest,
        a_metric_receipt_sha256=identity_metric.metric_receipt_sha256,
        ground_prior_receipt_sha256=ground_receipt,
        lock=lock,
        fit_audit=fit_audit,
        quantization_audit=quantization_audit,
        resource_audit=resource_audit,
        state_receipt_sha256=_canonical_sha256(placeholder_payload),
    )


def _decode_rda_weights(state: TypedZIDSRDAState) -> np.ndarray:
    return state.weight_codes_qint8.astype(np.float64) * state.weight_scales_fp16.astype(
        np.float64
    )[:, None]


def _score_rda_logits(state: TypedZIDSRDAState, query_zid: np.ndarray) -> np.ndarray:
    query = zid.normalize_zid_rows(query_zid).astype(np.float64)
    logits = query @ _decode_rda_weights(state).T + state.bias_fp16.astype(np.float64)[None, :]
    if not np.isfinite(logits).all():
        raise ZIDSRDAError("RDA logits became non-finite")
    return _readonly(logits, np.float32)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(logits)
    temp = _positive(temperature, "RDA temperature")
    if scores.dtype != np.float32 or scores.ndim != 2 or not np.isfinite(scores).all():
        raise ZIDSRDAError("RDA softmax logits drift")
    scaled = scores.astype(np.float64) / temp
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return _readonly(exp / np.sum(exp, axis=1, keepdims=True), np.float32)


@dataclass(frozen=True, slots=True)
class TypedZIDSRDAFusionResult:
    classes: tuple[str, ...]
    a_probability_fp32: np.ndarray
    rda_probability_fp32: np.ndarray | None
    fused_probability_fp32: np.ndarray
    predicted_class_indices_int16: np.ndarray
    audit: Mapping[str, Any]
    schema: str = FUSION_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        a_probability = np.asarray(self.a_probability_fp32)
        fused = np.asarray(self.fused_probability_fp32)
        prediction = np.asarray(self.predicted_class_indices_int16)
        rda_probability = self.rda_probability_fp32
        if (
            self.schema != FUSION_SCHEMA
            or a_probability.dtype != np.float32
            or fused.dtype != np.float32
            or a_probability.ndim != 2
            or fused.shape != a_probability.shape
            or fused.shape[1] != len(classes)
            or prediction.dtype != np.int16
            or prediction.shape != (len(fused),)
            or not np.isfinite(a_probability).all()
            or not np.isfinite(fused).all()
            or not np.allclose(np.sum(a_probability, axis=1), 1.0, atol=2e-6)
            or not np.allclose(np.sum(fused, axis=1), 1.0, atol=2e-6)
        ):
            raise ZIDSRDAError("fusion result probability/prediction drift")
        if rda_probability is not None:
            rda = np.asarray(rda_probability)
            if (
                rda.dtype != np.float32
                or rda.shape != fused.shape
                or not np.isfinite(rda).all()
                or np.any(rda < 0.0)
                or np.any(rda > 1.0)
                or not np.allclose(np.sum(rda, axis=1), 1.0, atol=2e-6)
            ):
                raise ZIDSRDAError("RDA probability drift")
            object.__setattr__(self, "rda_probability_fp32", _readonly(rda, np.float32))
        if (
            np.any(a_probability < 0.0)
            or np.any(a_probability > 1.0)
            or np.any(fused < 0.0)
            or np.any(fused > 1.0)
            or np.any(prediction < 0)
            or np.any(prediction >= len(classes))
            or not np.array_equal(prediction, np.argmax(fused, axis=1).astype(np.int16))
        ):
            raise ZIDSRDAError("fusion probability range or argmax drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "a_probability_fp32", _readonly(a_probability, np.float32))
        object.__setattr__(self, "fused_probability_fp32", _readonly(fused, np.float32))
        object.__setattr__(
            self, "predicted_class_indices_int16", _readonly(prediction, np.int16)
        )
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def fuse_zid_qknn_srda(
    bank: zid.TypedINT8ZIDSupportBank,
    state: TypedZIDSRDAState,
    query_zid: np.ndarray,
) -> TypedZIDSRDAFusionResult:
    """Score independent queries; alpha is sealed in the Phase1 lock."""

    identity_metric, a_before = _verify_bank_state_binding(bank, state)
    before = serialize_zid_srda_state(state)
    a_logits = zid.score_zid_student_t_logits(bank, query_zid, metric=identity_metric)
    a_probability = zid.softmax_probabilities(a_logits, config=bank.config)
    alpha = float(state.alpha_phase1)
    if alpha == 0.0:
        fused = a_probability
        rda_probability = None
        rda_branch_evaluated = False
    else:
        rda_logits = _score_rda_logits(state, query_zid)
        rda_probability = _softmax(rda_logits, state.rda_temperature)
        rda_branch_evaluated = True
        if alpha == 1.0:
            fused = rda_probability
        else:
            fused = _readonly(
                (1.0 - alpha) * a_probability.astype(np.float64)
                + alpha * rda_probability.astype(np.float64),
                np.float32,
            )
    prediction = np.argmax(fused, axis=1).astype(np.int16)
    after = serialize_zid_srda_state(state)
    a_after = zid.serialize_typed_zid_runtime_state(bank, identity_metric)
    if before != after:
        raise AssertionError("query scoring mutated the SRDA state")
    if a_before != a_after:
        raise AssertionError("query scoring mutated the Patch-A runtime state")
    audit = {
        "schema": "cvs.phase2.zid_srda.fusion_audit.v1",
        "formula": "pB=(1-alpha_phase1)*pA+alpha_phase1*pRDA",
        "alpha_phase1": alpha,
        "alpha_source": "phase1_lodo_only",
        "target_support_crossfit": False,
        "rda_branch_evaluated": rda_branch_evaluated,
        "alpha_zero_bit_exact_a": bool(alpha != 0.0 or np.array_equal(fused, a_probability)),
        "a_probability_sha256": _sha256_bytes(a_probability.tobytes(order="C")),
        "fused_probability_sha256": _sha256_bytes(fused.tobytes(order="C")),
        "state_sha256_before": _sha256_bytes(before),
        "state_sha256_after": _sha256_bytes(after),
        "a_state_sha256_before": _sha256_bytes(a_before),
        "a_state_sha256_after": _sha256_bytes(a_after),
        "same_formula_all_registered_classes": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_batch_dependency": False,
    }
    return TypedZIDSRDAFusionResult(
        classes=bank.classes,
        a_probability_fp32=a_probability,
        rda_probability_fp32=rda_probability,
        fused_probability_fp32=fused,
        predicted_class_indices_int16=prediction,
        audit=audit,
    )


_WIRE_ARRAYS = (
    ("weight_codes_qint8", np.dtype(np.int8)),
    ("weight_scales_fp16", np.dtype(np.float16)),
    ("bias_fp16", np.dtype(np.float16)),
)


def _wire_array(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    name_bytes = name.encode("utf-8")
    dtype_bytes = array.dtype.str.encode("ascii")
    payload = array.tobytes(order="C")
    frame = bytearray()
    frame.extend(struct.pack("<H", len(name_bytes)))
    frame.extend(name_bytes)
    frame.extend(struct.pack("<H", len(dtype_bytes)))
    frame.extend(dtype_bytes)
    frame.extend(struct.pack("<B", array.ndim))
    for dimension in array.shape:
        frame.extend(struct.pack("<I", int(dimension)))
    frame.extend(struct.pack("<Q", len(payload)))
    frame.extend(payload)
    return bytes(frame)


def _wire_header(state: TypedZIDSRDAState) -> dict[str, Any]:
    return {
        "schema": WIRE_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "classes": list(state.classes),
        "active_k": state.active_k,
        "alpha_phase1": state.alpha_phase1,
        "rda_temperature": state.rda_temperature,
        "a_bank_receipt_sha256": state.a_bank_receipt_sha256,
        "a_config_lock_digest": state.a_config_lock_digest,
        "a_metric_receipt_sha256": state.a_metric_receipt_sha256,
        "ground_prior_receipt_sha256": state.ground_prior_receipt_sha256,
        "lock": asdict(state.lock),
        "fit_audit": _json_value(state.fit_audit),
        "quantization_audit": _json_value(state.quantization_audit),
        "resource_audit": _json_value(state.resource_audit),
        "diagnostic_authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "state_receipt_sha256": state.state_receipt_sha256,
        "array_order": [name for name, _dtype in _WIRE_ARRAYS],
    }


def serialize_zid_srda_state(state: TypedZIDSRDAState) -> bytes:
    if type(state) is not TypedZIDSRDAState:
        raise ZIDSRDAError("wire serialization requires exact SRDA state")
    _verify_state(state)
    header = _canonical_bytes(_wire_header(state))
    if len(header) > 4_000_000:
        raise ZIDSRDAError("SRDA wire header exceeds fixed bound")
    output = bytearray(WIRE_MAGIC)
    output.extend(struct.pack("<I", len(header)))
    output.extend(header)
    output.extend(struct.pack("<H", len(_WIRE_ARRAYS)))
    for name, _dtype in _WIRE_ARRAYS:
        output.extend(_wire_array(name, np.asarray(getattr(state, name))))
    wire = bytes(output)
    if len(wire) > 16_000_000:
        raise ZIDSRDAError("SRDA wire exceeds fixed total bound")
    return wire


def _take(data: bytes, position: int, size: int, context: str) -> tuple[bytes, int]:
    if size < 0 or position < 0 or position + size > len(data):
        raise ZIDSRDAError(f"truncated SRDA wire while reading {context}")
    return data[position : position + size], position + size


def _uint(data: bytes, position: int, fmt: str, context: str) -> tuple[int, int]:
    size = struct.calcsize(fmt)
    raw, position = _take(data, position, size, context)
    return int(struct.unpack(fmt, raw)[0]), position


def deserialize_zid_srda_state(wire: bytes) -> TypedZIDSRDAState:
    if type(wire) is not bytes or len(wire) > 16_000_000:
        raise ZIDSRDAError("SRDA wire must be bounded exact bytes")
    position = 0
    magic, position = _take(wire, position, len(WIRE_MAGIC), "magic")
    if magic != WIRE_MAGIC:
        raise ZIDSRDAError("SRDA wire magic drift")
    header_size, position = _uint(wire, position, "<I", "header size")
    if header_size > 4_000_000:
        raise ZIDSRDAError("SRDA wire header exceeds fixed bound")
    header_raw, position = _take(wire, position, header_size, "header")
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ZIDSRDAError("SRDA wire header contains a duplicate key")
            result[key] = value
        return result

    try:
        header = json.loads(
            header_raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ZIDSRDAError("SRDA wire header is not canonical JSON") from error
    try:
        canonical_header = _canonical_bytes(header)
    except (TypeError, ValueError) as error:
        raise ZIDSRDAError("SRDA wire header contains unsupported JSON values") from error
    if canonical_header != header_raw:
        raise ZIDSRDAError("SRDA wire header is not canonical")
    expected_header_fields = {
        "schema",
        "state_schema",
        "classes",
        "active_k",
        "alpha_phase1",
        "rda_temperature",
        "a_bank_receipt_sha256",
        "a_config_lock_digest",
        "a_metric_receipt_sha256",
        "ground_prior_receipt_sha256",
        "lock",
        "fit_audit",
        "quantization_audit",
        "resource_audit",
        "diagnostic_authority_scope",
        "state_receipt_sha256",
        "array_order",
    }
    if type(header) is not dict or set(header) != expected_header_fields:
        raise ZIDSRDAError("SRDA wire header field set drift")
    if (
        header["schema"] != WIRE_SCHEMA
        or header["state_schema"] != STATE_SCHEMA
        or header["diagnostic_authority_scope"] != DIAGNOSTIC_AUTHORITY_SCOPE
        or header["array_order"] != [name for name, _dtype in _WIRE_ARRAYS]
    ):
        raise ZIDSRDAError("SRDA wire schema/array order drift")
    array_count, position = _uint(wire, position, "<H", "array count")
    if array_count != len(_WIRE_ARRAYS):
        raise ZIDSRDAError("SRDA wire array count drift")
    arrays: dict[str, np.ndarray] = {}
    for expected_name, expected_dtype in _WIRE_ARRAYS:
        name_size, position = _uint(wire, position, "<H", "array name size")
        name_raw, position = _take(wire, position, name_size, "array name")
        dtype_size, position = _uint(wire, position, "<H", "dtype size")
        dtype_raw, position = _take(wire, position, dtype_size, "dtype")
        ndim, position = _uint(wire, position, "<B", "ndim")
        if ndim > 4:
            raise ZIDSRDAError("SRDA wire array ndim exceeds fixed bound")
        shape = []
        for _index in range(ndim):
            dimension, position = _uint(wire, position, "<I", "shape")
            shape.append(dimension)
        payload_size, position = _uint(wire, position, "<Q", "payload size")
        payload, position = _take(wire, position, payload_size, "array payload")
        try:
            name = name_raw.decode("utf-8")
            dtype = np.dtype(dtype_raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ZIDSRDAError("SRDA wire array name/dtype decode failed") from error
        if name != expected_name or dtype != expected_dtype:
            raise ZIDSRDAError("SRDA wire array name/dtype order drift")
        element_count = 1
        for dimension in shape:
            if dimension > len(wire) or element_count > len(wire) // max(dimension, 1):
                raise ZIDSRDAError("SRDA wire array shape exceeds payload bound")
            element_count *= dimension
        expected_size = element_count * dtype.itemsize
        if expected_size != payload_size:
            raise ZIDSRDAError("SRDA wire array payload length drift")
        try:
            arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(tuple(shape)).copy()
        except ValueError as error:
            raise ZIDSRDAError("SRDA wire array reshape failed") from error
    if position != len(wire):
        raise ZIDSRDAError("SRDA wire contains trailing bytes")
    try:
        lock = Phase1ZIDSRDALock(**header["lock"])
        state = TypedZIDSRDAState(
            classes=tuple(header["classes"]),
            active_k=header["active_k"],
            weight_codes_qint8=arrays["weight_codes_qint8"],
            weight_scales_fp16=arrays["weight_scales_fp16"],
            bias_fp16=arrays["bias_fp16"],
            alpha_phase1=header["alpha_phase1"],
            rda_temperature=header["rda_temperature"],
            a_bank_receipt_sha256=header["a_bank_receipt_sha256"],
            a_config_lock_digest=header["a_config_lock_digest"],
            a_metric_receipt_sha256=header["a_metric_receipt_sha256"],
            ground_prior_receipt_sha256=header["ground_prior_receipt_sha256"],
            lock=lock,
            fit_audit=header["fit_audit"],
            quantization_audit=header["quantization_audit"],
            resource_audit=header["resource_audit"],
            state_receipt_sha256=header["state_receipt_sha256"],
        )
    except (TypeError, KeyError, ZIDSRDAError) as error:
        if isinstance(error, ZIDSRDAError):
            raise
        raise ZIDSRDAError("SRDA wire typed reconstruction failed") from error
    if serialize_zid_srda_state(state) != wire:
        raise ZIDSRDAError("SRDA wire failed byte-exact roundtrip")
    return state


def audit_combined_resources(
    bank: zid.TypedINT8ZIDSupportBank,
    state: TypedZIDSRDAState,
) -> dict[str, Any]:
    identity, a_wire = _verify_bank_state_binding(bank, state)
    a_audit = zid.audit_runtime_state(bank, identity)
    b_wire = serialize_zid_srda_state(state)
    if int(a_audit["actual_serialized_state_bytes"]) != len(a_wire):
        raise ZIDSRDAError("Patch-A resource audit/wire drift")
    b_numeric = int(
        state.weight_codes_qint8.nbytes
        + state.weight_scales_fp16.nbytes
        + state.bias_fp16.nbytes
    )
    return {
        "schema": "cvs.phase2.zid_srda.combined_resource_audit.v1",
        "a_numeric_array_state_bytes": int(a_audit["numeric_array_state_bytes"]),
        "b_incremental_numeric_array_state_bytes": b_numeric,
        "component_numeric_sum_bytes": int(a_audit["numeric_array_state_bytes"] + b_numeric),
        "a_actual_wire_bytes": int(a_audit["actual_serialized_state_bytes"]),
        "b_actual_wire_bytes": len(b_wire),
        "component_wire_sum_bytes": int(a_audit["actual_serialized_state_bytes"] + len(b_wire)),
        "complete_combined_wire_container_available": False,
        "a_matmul_audit": {
            key: value for key, value in a_audit.items() if "matmul" in key
        },
        "b_incremental_rda_linear_matmul_mac_per_query": int(
            0 if state.alpha_phase1 == 0.0 else len(state.classes) * Z_DIM
        ),
        "not_end_to_end_mac_or_latency": True,
        "persistent_decoded_cache_bytes": 0,
        "query_state_updates": 0,
    }


__all__ = [
    "DIAGNOSTIC_AUTHORITY_SCOPE",
    "FUSION_SCHEMA",
    "GROUND_PROVENANCE_SCHEMA",
    "GROUND_SCHEMA",
    "LOCK_SCHEMA",
    "MAX_GROUND_RANK",
    "MAX_TARGET_RANK",
    "MAX_TOTAL_RANK",
    "Phase1ZIDSRDALock",
    "STATE_SCHEMA",
    "TypedZIDSRDAFusionResult",
    "TypedZIDGroundPriorProvenance",
    "TypedZIDSRDAState",
    "TypedZIDSharedCovariancePrior",
    "WIRE_MAGIC",
    "ZERO_SHA256",
    "ZIDSRDAError",
    "audit_combined_resources",
    "build_typed_shared_covariance_prior",
    "build_zid_srda_state",
    "decode_shared_covariance_prior",
    "deserialize_zid_srda_state",
    "fuse_zid_qknn_srda",
    "serialize_zid_srda_state",
    "woodbury_precision_apply",
]
