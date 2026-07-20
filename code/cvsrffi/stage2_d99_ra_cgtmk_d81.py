"""Protocol-minimal local core for D99 RA-CGTMK-D81.

The module implements a redundancy-aware ground geometry, a support-only
coverage-gated metric, and a typed INT8 Student-t metric-kernel bank.  Ground
knowledge can only change a shared z160 distance; it never contributes a class
logit or a target class mean.  Every registered class is represented by the
same target-support construction and the query path is stateless.

The current typed target-row D81 implementation is independently blocked: it
fits its 20-step metric on all registered classes instead of sealed Y_old only.
Therefore this module deliberately exposes no base-logit/probability fusion or
deploy prediction API.  The Phase1-locked eta values are sealed for a future
corrected typed integration, while the current status remains local-core-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.phase2.d99.ra_cgtmk_d81.local_core.v1"
LOCK_SCHEMA = "cvs.phase1.d99.ra_cgtmk_d81_lock.v1"
GROUND_SCHEMA = "cvs.phase1.d99.redundancy_aware_ground_geometry.v1"
METRIC_SCHEMA = "cvs.phase2.d99.support_only_metric.v1"
BANK_SCHEMA = "cvs.phase2.d99.typed_int8_metric_kernel_bank.v1"
GROUND_AGGREGATION_RECEIPT_SCHEMA = (
    "cvs.phase1.d99.external_ground_aggregation_receipt.v1"
)
GROUND_BUNDLE_SCHEMA = "cvs.phase1.d99.typed_int8_ground_aggregate_bundle.v1"
VALIDATION_ARTIFACT_SCHEMA = "cvs.phase1.d99.typed_validation_artifact.v1"
VALIDATION_RECEIPT_SCHEMA = "cvs.phase1.d99.external_validation_receipt.v1"
VALIDATION_METHOD_LOCK_SCHEMA = "cvs.phase1.d99.validation_method_lock.v1"
AUTHORITY_ENVELOPE_SCHEMA = "cvs.phase1.d99.external_authority_envelope.v1"
VALIDATION_ARCHIVE_SCHEMA = "cvs.phase1.d99.validation_feature_archive.v1"
VALIDATION_MANIFEST_SCHEMA = "cvs.phase1.d99.validation_manifest.v1"
VALIDATION_PRODUCER_ID = "d97_phase1_singleobs_lodo_validation_exporter"
VALIDATION_LIFECYCLE = "PHASE1_SOURCE_VALIDATION_ONLY"
DEPLOYMENT_STATUS = (
    "LOCAL_CORE_BLOCKED_EXTERNAL_PHASE1_AUTHORITY_AND_CORRECTED_TYPED_D81_P0"
)
REQUIRED_TYPED_D81_STATE_SCHEMA = (
    "cvs.phase2.d81.typed_target_state.corrected_old_only_metric.pending"
)

FEATURE_DIM = 288
Z_DIM = 160
FFT_DIM = 96
RF_DIM = 32
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
BLOCK_DIMS = (Z_DIM, FFT_DIM, RF_DIM)
ALLOWED_K = (1, 5, 10)
INT8_MAX = 127.0
EPSILON = 1.0e-12
VALIDATION_ARCHIVE_MAGIC = b"D99-PHASE1-VALIDATION\0"
_VALIDATION_LOADER_TOKEN = object()
# No independently published authority envelope exists in this repository.
# Provisioning must replace this with a separately reviewed immutable SHA; a
# caller-supplied expected SHA can verify bytes but can never grant authority.
TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256: str | None = None


class D99RACGTMKError(ValueError):
    """Raised when D99 protocol, receipt, registry, or numeric state drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value)
    if (
        normalized != normalized.lower()
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise D99RACGTMKError(f"{name} must be lowercase SHA256 hex")
    return normalized


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _registry(values: Sequence[str], name: str, minimum: int = 2) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        len(result) < minimum
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise D99RACGTMKError(
            f"{name} must contain at least {minimum} unique nonempty values"
        )
    return result


def _finite_features(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D99RACGTMKError(f"{name} must be finite float32 [N,{FEATURE_DIM}]")
    return np.ascontiguousarray(rows)


def _normalized_rows(value: np.ndarray, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1] != dimension
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D99RACGTMKError(f"{name} must be finite [N,{dimension}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise D99RACGTMKError(f"{name} contains a zero-norm row")
    return rows / norms


def normalize_feature_blocks(value: np.ndarray) -> np.ndarray:
    """Normalize z160, FFT96, and RF32 independently without creating views."""

    rows = _finite_features(value, "features").astype(np.float64)
    normalized = np.empty_like(rows)
    for block in BLOCK_SLICES:
        normalized[:, block] = _normalized_rows(
            rows[:, block], block.stop - block.start, "feature block"
        )
    return _readonly(normalized, np.float32)


@dataclass(frozen=True)
class Phase1D99Lock:
    """All tunable D99 values, selected before any target access."""

    density_tau: float
    max_ground_rank: int
    max_target_rank: int
    coverage_floor: float
    ground_energy_scale: float
    target_energy_scale: float
    shrinkage_prior_strength: float
    ground_weight_max: float
    target_weight_max: float
    student_nu: float
    kernel_effective_dim: int
    kernel_volume_gamma: float
    shared_h0: float
    scale_prior_strength: float
    scale_min_ratio: float
    scale_max_ratio: float
    z_weight: float
    fft_weight: float
    rf_weight: float
    eta_k1: float
    eta_k5: float
    eta_k10: float
    phase1_receipt_sha256: str
    ground_aggregation_receipt_sha256: str
    ground_bundle_receipt_sha256: str
    quantization_margin_audit_sha256: str
    validation_method_lock_sha256: str
    d81_phase1_lock_sha256: str
    ground_old_registry: tuple[str, ...]

    def __post_init__(self) -> None:
        positive = (
            self.density_tau,
            self.ground_energy_scale,
            self.target_energy_scale,
            self.shrinkage_prior_strength,
            self.student_nu,
            self.kernel_volume_gamma,
            self.shared_h0,
            self.scale_prior_strength,
            self.scale_min_ratio,
            self.scale_max_ratio,
        )
        finite = positive + (
            self.coverage_floor,
            self.ground_weight_max,
            self.target_weight_max,
            self.z_weight,
            self.fft_weight,
            self.rf_weight,
            self.eta_k1,
            self.eta_k5,
            self.eta_k10,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise D99RACGTMKError("D99 Phase1 lock values must be finite")
        if any(float(value) <= 0.0 for value in positive):
            raise D99RACGTMKError("D99 positive Phase1 lock values must be >0")
        if (
            isinstance(self.max_ground_rank, bool)
            or isinstance(self.max_target_rank, bool)
            or not isinstance(self.max_ground_rank, (int, np.integer))
            or not isinstance(self.max_target_rank, (int, np.integer))
            or not 1 <= int(self.max_ground_rank) <= 4
            or not 0 <= int(self.max_target_rank) <= 4
        ):
            raise D99RACGTMKError("D99 ground/target ranks must be integers <=4")
        if (
            isinstance(self.kernel_effective_dim, bool)
            or not isinstance(self.kernel_effective_dim, (int, np.integer))
            or not 1 <= int(self.kernel_effective_dim) <= FEATURE_DIM
        ):
            raise D99RACGTMKError("D99 kernel_effective_dim must be an integer in [1,288]")
        if not 0.0 <= float(self.coverage_floor) < 1.0:
            raise D99RACGTMKError("D99 coverage_floor must be in [0,1)")
        if not 0.0 <= float(self.ground_weight_max) <= 1.0 or not 0.0 <= float(
            self.target_weight_max
        ) <= 1.0:
            raise D99RACGTMKError("D99 metric weights must be in [0,1]")
        if self.scale_min_ratio > 1.0 or self.scale_max_ratio < 1.0:
            raise D99RACGTMKError("D99 scale ratios must bracket shared_h0")
        if not np.isclose(
            self.z_weight + self.fft_weight + self.rf_weight, 1.0, atol=1e-12
        ) or min(self.z_weight, self.fft_weight, self.rf_weight) <= 0.0:
            raise D99RACGTMKError("D99 three-block weights must be positive and sum to one")
        if any(
            not 0.0 <= float(value) <= 1.0
            for value in (self.eta_k1, self.eta_k5, self.eta_k10)
        ):
            raise D99RACGTMKError("D99 Phase1 eta values must be in [0,1]")
        for value, name in (
            (self.phase1_receipt_sha256, "phase1_receipt_sha256"),
            (
                self.ground_aggregation_receipt_sha256,
                "ground_aggregation_receipt_sha256",
            ),
            (self.ground_bundle_receipt_sha256, "ground_bundle_receipt_sha256"),
            (
                self.quantization_margin_audit_sha256,
                "quantization_margin_audit_sha256",
            ),
            (self.validation_method_lock_sha256, "validation_method_lock_sha256"),
            (self.d81_phase1_lock_sha256, "d81_phase1_lock_sha256"),
        ):
            _require_sha256(value, name)
        old_registry = _registry(self.ground_old_registry, "ground_old_registry")
        object.__setattr__(self, "ground_old_registry", old_registry)

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256({"schema": LOCK_SCHEMA, **asdict(self)})

    def eta_for_k(self, k_shot: int) -> float:
        if int(k_shot) == 1:
            return float(self.eta_k1)
        if int(k_shot) == 5:
            return float(self.eta_k5)
        if int(k_shot) == 10:
            return float(self.eta_k10)
        raise D99RACGTMKError(f"D99 supports only K in {ALLOWED_K}")


@dataclass(frozen=True)
class ExternalGroundAggregationReceipt:
    """Typed Phase1 aggregation receipt without an authority claim.

    This local type binds the producer/config/checkpoint identities and the
    aggregate-only policy.  It is not a cryptographic signer or certification
    boundary; formal authority must be supplied by a future external producer.
    """

    aggregation_manifest_sha256: str
    producer_code_sha256: str
    phase1_checkpoint_sha256: str
    receipt_sha256: str
    minimum_physical_sample_count: int = 2
    member_ids_present: bool = False
    target_rows_used: int = 0
    cryptographic_external_authority_claimed: bool = False
    schema: str = GROUND_AGGREGATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for value, name in (
            (self.aggregation_manifest_sha256, "aggregation_manifest_sha256"),
            (self.producer_code_sha256, "producer_code_sha256"),
            (self.phase1_checkpoint_sha256, "phase1_checkpoint_sha256"),
            (self.receipt_sha256, "receipt_sha256"),
        ):
            _require_sha256(value, name)
        if (
            self.schema != GROUND_AGGREGATION_RECEIPT_SCHEMA
            or self.minimum_physical_sample_count != 2
            or self.member_ids_present
            or self.target_rows_used != 0
            or self.cryptographic_external_authority_claimed
        ):
            raise D99RACGTMKError("D99 external aggregation receipt policy drift")
        payload = {
            "schema": self.schema,
            "aggregation_manifest_sha256": self.aggregation_manifest_sha256,
            "producer_code_sha256": self.producer_code_sha256,
            "phase1_checkpoint_sha256": self.phase1_checkpoint_sha256,
            "minimum_physical_sample_count": 2,
            "member_ids_present": False,
            "target_rows_used": 0,
            "cryptographic_external_authority_claimed": False,
        }
        if self.receipt_sha256 != _canonical_sha256(payload):
            raise D99RACGTMKError("D99 external aggregation receipt SHA drift")


@dataclass(frozen=True)
class Phase1GroundAggregateBundle:
    """Exact typed INT8 aggregate grid consumed by the public ground builder."""

    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    domain_class_mask: np.ndarray
    physical_sample_count_floor_uint16: np.ndarray
    domain_ids: tuple[str, ...]
    ground_old_registry: tuple[str, ...]
    aggregation_receipt: ExternalGroundAggregationReceipt
    bundle_sha256: str
    schema: str = GROUND_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        domains = _registry(self.domain_ids, "ground bundle domain_ids")
        classes = _registry(self.ground_old_registry, "ground bundle old registry")
        expected = (len(domains), len(classes), Z_DIM)
        if (
            self.schema != GROUND_BUNDLE_SCHEMA
            or type(self.aggregation_receipt) is not ExternalGroundAggregationReceipt
            or self.codes_qint8.dtype != np.int8
            or self.codes_qint8.shape != expected
            or self.scales_fp16.dtype != np.float16
            or self.scales_fp16.shape != expected[:2]
            or self.domain_class_mask.dtype != np.bool_
            or self.domain_class_mask.shape != expected[:2]
            or not np.all(self.domain_class_mask)
            or self.physical_sample_count_floor_uint16.dtype != np.uint16
            or self.physical_sample_count_floor_uint16.shape != expected[:2]
            or np.any(self.physical_sample_count_floor_uint16 < 2)
            or not np.isfinite(self.scales_fp16).all()
            or np.any(self.scales_fp16 <= 0.0)
        ):
            raise D99RACGTMKError("D99 typed ground aggregate bundle invariant drift")
        _require_sha256(self.bundle_sha256, "ground bundle SHA256")
        expected_sha = _canonical_sha256(
            _ground_bundle_payload(
                codes=self.codes_qint8,
                scales=self.scales_fp16,
                mask=self.domain_class_mask,
                count_floor=self.physical_sample_count_floor_uint16,
                domains=domains,
                classes=classes,
                aggregation_receipt=self.aggregation_receipt,
            )
        )
        if self.bundle_sha256 != expected_sha:
            raise D99RACGTMKError("D99 typed ground bundle SHA drift")
        for name, dtype in (
            ("codes_qint8", np.int8),
            ("scales_fp16", np.float16),
            ("domain_class_mask", np.bool_),
            ("physical_sample_count_floor_uint16", np.uint16),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))


def _ground_bundle_payload(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
    count_floor: np.ndarray,
    domains: tuple[str, ...],
    classes: tuple[str, ...],
    aggregation_receipt: ExternalGroundAggregationReceipt,
) -> dict[str, Any]:
    return {
        "schema": GROUND_BUNDLE_SCHEMA,
        "codes_qint8": _array_receipt(codes),
        "scales_fp16": _array_receipt(scales),
        "domain_class_mask": _array_receipt(mask),
        "physical_sample_count_floor_uint16": _array_receipt(count_floor),
        "domain_ids": list(domains),
        "ground_old_registry": list(classes),
        "external_aggregation_receipt_sha256": aggregation_receipt.receipt_sha256,
        "member_ids_present": False,
        "raw_or_clean_rows_present": False,
    }


def produce_typed_ground_aggregate_bundle(
    *,
    codes_qint8: np.ndarray,
    scales_fp16: np.ndarray,
    domain_class_mask: np.ndarray,
    physical_sample_count_floor_uint16: np.ndarray,
    domain_ids: Sequence[str],
    ground_old_registry: Sequence[str],
    aggregation_receipt: ExternalGroundAggregationReceipt,
) -> Phase1GroundAggregateBundle:
    """Bind exact INT8 aggregate arrays to a typed external receipt.

    No member-ID, exemplar, raw/clean row, or target input exists.  The
    resulting Python type is a local fail-closed interface, not certification.
    """

    if type(aggregation_receipt) is not ExternalGroundAggregationReceipt:
        raise D99RACGTMKError("D99 ground producer requires an exact typed receipt")
    codes = np.asarray(codes_qint8)
    scales = np.asarray(scales_fp16)
    mask = np.asarray(domain_class_mask)
    count_floor = np.asarray(physical_sample_count_floor_uint16)
    domains = _registry(domain_ids, "ground bundle domain_ids")
    classes = _registry(ground_old_registry, "ground bundle old registry")
    payload = _ground_bundle_payload(
        codes=codes,
        scales=scales,
        mask=mask,
        count_floor=count_floor,
        domains=domains,
        classes=classes,
        aggregation_receipt=aggregation_receipt,
    )
    return Phase1GroundAggregateBundle(
        codes_qint8=codes,
        scales_fp16=scales,
        domain_class_mask=mask,
        physical_sample_count_floor_uint16=count_floor,
        domain_ids=domains,
        ground_old_registry=classes,
        aggregation_receipt=aggregation_receipt,
        bundle_sha256=_canonical_sha256(payload),
    )


def _forbid_target_query_identifier(value: str, name: str) -> str:
    result = str(value)
    forbidden = ("target", "query", "phase2", "test-row", "held-query")
    if not result or any(token in result.casefold() for token in forbidden):
        raise D99RACGTMKError(f"D99 {name} is not Phase1-source-only")
    return result


@dataclass(frozen=True)
class Phase1ValidationMethodLock:
    """Externally preregistered source-validation authority allowlist."""

    expected_external_validation_receipt_sha256: str
    allowlisted_producer_code_sha256: str
    allowlisted_phase1_checkpoint_sha256: str
    allowlisted_feature_archive_sha256: str
    allowlisted_validation_manifest_sha256: str
    expected_phase1_episode_support_receipt_sha256: str
    expected_phase1_episode_id: str
    producer_id: str = VALIDATION_PRODUCER_ID
    lifecycle: str = VALIDATION_LIFECYCLE
    target_rows_used: int = 0
    query_rows_used: int = 0
    single_received_observation: bool = True
    schema: str = VALIDATION_METHOD_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != VALIDATION_METHOD_LOCK_SCHEMA
            or self.producer_id != VALIDATION_PRODUCER_ID
            or self.lifecycle != VALIDATION_LIFECYCLE
            or int(self.target_rows_used) != 0
            or int(self.query_rows_used) != 0
            or self.single_received_observation is not True
        ):
            raise D99RACGTMKError("D99 Phase1 validation method lock policy drift")
        _forbid_target_query_identifier(
            self.expected_phase1_episode_id, "validation episode ID"
        )
        for value, name in (
            (self.expected_external_validation_receipt_sha256, "expected validation receipt"),
            (self.allowlisted_producer_code_sha256, "allowlisted producer code"),
            (self.allowlisted_phase1_checkpoint_sha256, "allowlisted Phase1 checkpoint"),
            (self.allowlisted_feature_archive_sha256, "allowlisted feature archive"),
            (self.allowlisted_validation_manifest_sha256, "allowlisted validation manifest"),
            (self.expected_phase1_episode_support_receipt_sha256, "episode support authority"),
        ):
            _require_sha256(value, name)

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class Phase1AuthorityEnvelope:
    """Loaded authority evidence; currently always blocked without a trusted SHA."""

    envelope_sha256: str
    externally_expected_envelope_sha256: str
    validation_method_lock_sha256: str
    expected_external_validation_receipt_sha256: str
    authority_status: str
    formal_phase1_eligible: bool
    loader_token: object = field(repr=False, compare=False)
    schema: str = AUTHORITY_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        for value, name in (
            (self.envelope_sha256, "authority envelope"),
            (self.externally_expected_envelope_sha256, "externally expected envelope"),
            (self.validation_method_lock_sha256, "authority method lock"),
            (
                self.expected_external_validation_receipt_sha256,
                "authority validation receipt",
            ),
        ):
            _require_sha256(value, name)
        trusted = TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256
        expected_status = (
            "PROVISIONED"
            if trusted is not None
            and self.envelope_sha256 == trusted
            and self.externally_expected_envelope_sha256 == trusted
            else "BLOCKED"
        )
        if (
            self.schema != AUTHORITY_ENVELOPE_SCHEMA
            or self.loader_token is not _VALIDATION_LOADER_TOKEN
            or self.envelope_sha256 != self.externally_expected_envelope_sha256
            or self.authority_status != expected_status
            or self.formal_phase1_eligible is not (expected_status == "PROVISIONED")
        ):
            raise D99RACGTMKError("D99 external Phase1 authority envelope drift")


def load_phase1_authority_envelope(
    *,
    authority_envelope_bytes: bytes,
    externally_expected_envelope_sha256: str,
    method_lock: Phase1ValidationMethodLock,
) -> Phase1AuthorityEnvelope:
    """Load bytes against an external expected SHA without self-granting trust."""

    if type(method_lock) is not Phase1ValidationMethodLock:
        raise D99RACGTMKError("D99 authority loader requires exact method lock")
    raw = bytes(authority_envelope_bytes)
    expected_sha = _require_sha256(
        externally_expected_envelope_sha256, "externally expected authority envelope"
    )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise D99RACGTMKError("D99 authority envelope bytes/expected SHA mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99RACGTMKError("D99 authority envelope is not UTF-8 JSON") from exc
    expected_payload = {
        "schema": AUTHORITY_ENVELOPE_SCHEMA,
        "validation_method_lock_sha256": method_lock.lock_digest,
        "expected_external_validation_receipt_sha256": (
            method_lock.expected_external_validation_receipt_sha256
        ),
        "source_validation_lifecycle": VALIDATION_LIFECYCLE,
        "single_received_observation": True,
        "target_rows_used": 0,
        "query_rows_used": 0,
    }
    if payload != expected_payload or raw != _canonical_bytes(expected_payload):
        raise D99RACGTMKError("D99 authority envelope payload drift")
    trusted = TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256
    provisioned = trusted is not None and actual_sha == trusted
    return Phase1AuthorityEnvelope(
        envelope_sha256=actual_sha,
        externally_expected_envelope_sha256=expected_sha,
        validation_method_lock_sha256=method_lock.lock_digest,
        expected_external_validation_receipt_sha256=(
            method_lock.expected_external_validation_receipt_sha256
        ),
        authority_status="PROVISIONED" if provisioned else "BLOCKED",
        formal_phase1_eligible=bool(provisioned),
        loader_token=_VALIDATION_LOADER_TOKEN,
    )


@dataclass(frozen=True)
class ExternalPhase1ValidationReceipt:
    """Externally sealed Phase1 validation provenance, never self-issued here."""

    phase1_episode_id: str
    phase1_episode_support_receipt_sha256: str
    feature_archive_sha256: str
    validation_manifest_sha256: str
    producer_code_sha256: str
    phase1_checkpoint_sha256: str
    receipt_sha256: str
    producer_id: str = VALIDATION_PRODUCER_ID
    lifecycle: str = VALIDATION_LIFECYCLE
    target_rows_used: int = 0
    query_rows_used: int = 0
    single_received_observation: bool = True
    schema: str = VALIDATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        episode = _forbid_target_query_identifier(
            self.phase1_episode_id, "validation episode ID"
        )
        if (
            self.schema != VALIDATION_RECEIPT_SCHEMA
            or self.producer_id != VALIDATION_PRODUCER_ID
            or self.lifecycle != VALIDATION_LIFECYCLE
            or int(self.target_rows_used) != 0
            or int(self.query_rows_used) != 0
            or self.single_received_observation is not True
        ):
            raise D99RACGTMKError("D99 external Phase1 validation receipt policy drift")
        for value, name in (
            (self.phase1_episode_support_receipt_sha256, "Phase1 episode support receipt"),
            (self.feature_archive_sha256, "validation feature archive"),
            (self.validation_manifest_sha256, "validation manifest"),
            (self.producer_code_sha256, "validation producer code"),
            (self.phase1_checkpoint_sha256, "Phase1 checkpoint"),
            (self.receipt_sha256, "external validation receipt"),
        ):
            _require_sha256(value, name)
        if self.receipt_sha256 != _canonical_sha256(_validation_receipt_payload(self)):
            raise D99RACGTMKError("D99 external validation receipt SHA drift")


def _validation_receipt_payload(
    receipt: ExternalPhase1ValidationReceipt,
) -> dict[str, Any]:
    return {
        "schema": VALIDATION_RECEIPT_SCHEMA,
        "producer_id": VALIDATION_PRODUCER_ID,
        "lifecycle": VALIDATION_LIFECYCLE,
        "phase1_episode_id": str(receipt.phase1_episode_id),
        "phase1_episode_support_receipt_sha256": (
            receipt.phase1_episode_support_receipt_sha256
        ),
        "feature_archive_sha256": receipt.feature_archive_sha256,
        "validation_manifest_sha256": receipt.validation_manifest_sha256,
        "producer_code_sha256": receipt.producer_code_sha256,
        "phase1_checkpoint_sha256": receipt.phase1_checkpoint_sha256,
        "target_rows_used": 0,
        "query_rows_used": 0,
        "single_received_observation": True,
    }


def _validation_archive_metadata(
    *, features: np.ndarray, physical_ids: Sequence[str], episode_id: str
) -> dict[str, Any]:
    physical = tuple(
        _forbid_target_query_identifier(value, "validation physical ID")
        for value in physical_ids
    )
    if (
        len(physical) != len(features)
        or len(set(physical)) != len(physical)
    ):
        raise D99RACGTMKError("D99 validation physical ID closure drift")
    return {
        "schema": VALIDATION_ARCHIVE_SCHEMA,
        "lifecycle": VALIDATION_LIFECYCLE,
        "phase1_episode_id": str(episode_id),
        "validation_features": _array_receipt(features),
        "physical_ids": list(physical),
        "target_rows_used": 0,
        "query_rows_used": 0,
        "single_received_observation": True,
    }


def _serialize_phase1_validation_archive(
    validation_features: np.ndarray,
    physical_ids: Sequence[str],
    phase1_episode_id: str,
) -> bytes:
    """Private deterministic format helper used by the external producer/tests."""

    features = _finite_features(validation_features, "Phase1 validation features")
    physical = tuple(str(value) for value in physical_ids)
    if (
        len(physical) != len(features)
        or len(set(physical)) != len(physical)
        or any(not value for value in physical)
    ):
        raise D99RACGTMKError("D99 validation physical ID closure drift")
    metadata = _validation_archive_metadata(
        features=features,
        physical_ids=physical,
        episode_id=phase1_episode_id,
    )
    header = _canonical_bytes(metadata)
    return (
        VALIDATION_ARCHIVE_MAGIC
        + len(header).to_bytes(8, "little")
        + header
        + np.ascontiguousarray(features).tobytes(order="C")
    )


@dataclass(frozen=True)
class Phase1ValidationArtifact:
    """Loaded exact archive whose source lifecycle is externally receipt-bound."""

    validation_features_fp32: np.ndarray
    physical_ids_sha256: str
    external_receipt: ExternalPhase1ValidationReceipt
    authority_envelope: Phase1AuthorityEnvelope
    validation_method_lock_sha256: str
    archive_serialized_bytes: int
    artifact_sha256: str
    loader_token: object = field(repr=False, compare=False)
    lifecycle: str = VALIDATION_LIFECYCLE
    source_validation_only: bool = True
    target_or_query_rows_present: bool = False
    single_received_observation: bool = True
    schema: str = VALIDATION_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        features = _finite_features(self.validation_features_fp32, "Phase1 validation features")
        if (
            self.schema != VALIDATION_ARTIFACT_SCHEMA
            or type(self.external_receipt) is not ExternalPhase1ValidationReceipt
            or type(self.authority_envelope) is not Phase1AuthorityEnvelope
            or self.authority_envelope.loader_token is not _VALIDATION_LOADER_TOKEN
            or self.authority_envelope.validation_method_lock_sha256
            != self.validation_method_lock_sha256
            or self.authority_envelope.expected_external_validation_receipt_sha256
            != self.external_receipt.receipt_sha256
            or self.lifecycle != VALIDATION_LIFECYCLE
            or not self.source_validation_only
            or self.target_or_query_rows_present
            or not self.single_received_observation
            or int(self.archive_serialized_bytes) <= 0
            or self.loader_token is not _VALIDATION_LOADER_TOKEN
        ):
            raise D99RACGTMKError("D99 Phase1 validation lifecycle drift")
        _require_sha256(self.physical_ids_sha256, "validation physical_ids_sha256")
        _require_sha256(self.validation_method_lock_sha256, "validation method lock")
        _require_sha256(self.artifact_sha256, "validation artifact_sha256")
        payload = {
            "schema": self.schema,
            "validation_features": _array_receipt(features),
            "physical_ids_sha256": self.physical_ids_sha256,
            "external_validation_receipt_sha256": self.external_receipt.receipt_sha256,
            "authority_envelope_sha256": self.authority_envelope.envelope_sha256,
            "authority_status": self.authority_envelope.authority_status,
            "formal_phase1_eligible": self.authority_envelope.formal_phase1_eligible,
            "validation_method_lock_sha256": self.validation_method_lock_sha256,
            "archive_serialized_bytes": int(self.archive_serialized_bytes),
            "lifecycle": self.lifecycle,
            "source_validation_only": True,
            "target_or_query_rows_present": False,
            "single_received_observation": True,
        }
        if self.artifact_sha256 != _canonical_sha256(payload):
            raise D99RACGTMKError("D99 Phase1 validation artifact SHA drift")
        object.__setattr__(self, "validation_features_fp32", _readonly(features, np.float32))

    @property
    def external_validation_receipt_sha256(self) -> str:
        return self.external_receipt.receipt_sha256


def load_phase1_validation_artifact(
    *,
    feature_archive_bytes: bytes,
    validation_manifest_bytes: bytes,
    producer_code_bytes: bytes,
    phase1_checkpoint_bytes: bytes,
    external_receipt: ExternalPhase1ValidationReceipt,
    method_lock: Phase1ValidationMethodLock,
    authority_envelope: Phase1AuthorityEnvelope,
    config: Phase1D99Lock,
) -> Phase1ValidationArtifact:
    """Load an externally sealed Phase1 artifact; raw arrays are never accepted."""

    if (
        type(external_receipt) is not ExternalPhase1ValidationReceipt
        or type(method_lock) is not Phase1ValidationMethodLock
        or type(authority_envelope) is not Phase1AuthorityEnvelope
        or type(config) is not Phase1D99Lock
    ):
        raise D99RACGTMKError("D99 validation load requires exact external lock/receipt types")
    if (
        method_lock.lock_digest != config.validation_method_lock_sha256
        or authority_envelope.validation_method_lock_sha256 != method_lock.lock_digest
        or authority_envelope.expected_external_validation_receipt_sha256
        != external_receipt.receipt_sha256
        or external_receipt.receipt_sha256
        != method_lock.expected_external_validation_receipt_sha256
        or external_receipt.producer_code_sha256
        != method_lock.allowlisted_producer_code_sha256
        or external_receipt.phase1_checkpoint_sha256
        != method_lock.allowlisted_phase1_checkpoint_sha256
        or external_receipt.feature_archive_sha256
        != method_lock.allowlisted_feature_archive_sha256
        or external_receipt.validation_manifest_sha256
        != method_lock.allowlisted_validation_manifest_sha256
        or external_receipt.phase1_episode_support_receipt_sha256
        != method_lock.expected_phase1_episode_support_receipt_sha256
        or external_receipt.phase1_episode_id
        != method_lock.expected_phase1_episode_id
    ):
        raise D99RACGTMKError("D99 external validation authority lock mismatch")
    archive = bytes(feature_archive_bytes)
    manifest_raw = bytes(validation_manifest_bytes)
    producer_raw = bytes(producer_code_bytes)
    checkpoint_raw = bytes(phase1_checkpoint_bytes)
    if (
        hashlib.sha256(archive).hexdigest() != external_receipt.feature_archive_sha256
        or hashlib.sha256(manifest_raw).hexdigest()
        != external_receipt.validation_manifest_sha256
        or hashlib.sha256(producer_raw).hexdigest() != external_receipt.producer_code_sha256
        or hashlib.sha256(checkpoint_raw).hexdigest()
        != external_receipt.phase1_checkpoint_sha256
    ):
        raise D99RACGTMKError("D99 external validation source bytes do not match receipt")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99RACGTMKError("D99 validation manifest is not canonical UTF-8 JSON") from exc
    expected_manifest = {
        "schema": VALIDATION_MANIFEST_SCHEMA,
        "producer_id": VALIDATION_PRODUCER_ID,
        "lifecycle": VALIDATION_LIFECYCLE,
        "phase1_episode_id": external_receipt.phase1_episode_id,
        "phase1_episode_support_receipt_sha256": (
            external_receipt.phase1_episode_support_receipt_sha256
        ),
        "feature_archive_sha256": external_receipt.feature_archive_sha256,
        "producer_code_sha256": external_receipt.producer_code_sha256,
        "phase1_checkpoint_sha256": external_receipt.phase1_checkpoint_sha256,
        "target_rows_used": 0,
        "query_rows_used": 0,
        "single_received_observation": True,
    }
    if manifest != expected_manifest or manifest_raw != _canonical_bytes(expected_manifest):
        raise D99RACGTMKError("D99 validation manifest/source lifecycle is unverifiable")
    prefix = len(VALIDATION_ARCHIVE_MAGIC)
    if not archive.startswith(VALIDATION_ARCHIVE_MAGIC) or len(archive) < prefix + 8:
        raise D99RACGTMKError("D99 validation feature archive framing drift")
    header_size = int.from_bytes(archive[prefix : prefix + 8], "little")
    header_start = prefix + 8
    header_end = header_start + header_size
    try:
        header_raw = archive[header_start:header_end]
        metadata = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99RACGTMKError("D99 validation feature archive header drift") from exc
    if header_raw != _canonical_bytes(metadata):
        raise D99RACGTMKError("D99 validation feature archive is not canonical")
    feature_receipt = metadata.get("validation_features", {})
    shape = tuple(int(value) for value in feature_receipt.get("shape", ()))
    if (
        metadata.get("schema") != VALIDATION_ARCHIVE_SCHEMA
        or metadata.get("lifecycle") != VALIDATION_LIFECYCLE
        or metadata.get("phase1_episode_id") != external_receipt.phase1_episode_id
        or metadata.get("target_rows_used") != 0
        or metadata.get("query_rows_used") != 0
        or metadata.get("single_received_observation") is not True
        or feature_receipt.get("dtype") != np.dtype(np.float32).str
        or len(shape) != 2
        or shape[1:] != (FEATURE_DIM,)
        or shape[0] < 1
    ):
        raise D99RACGTMKError("D99 validation feature archive lifecycle/schema drift")
    body = archive[header_end:]
    expected_body_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if len(body) != expected_body_bytes:
        raise D99RACGTMKError("D99 validation feature archive payload length drift")
    features = np.frombuffer(body, dtype=np.float32).reshape(shape)
    if _array_receipt(features) != feature_receipt:
        raise D99RACGTMKError("D99 validation feature archive payload SHA drift")
    physical = tuple(str(value) for value in metadata.get("physical_ids", ()))
    expected_metadata = _validation_archive_metadata(
        features=features,
        physical_ids=physical,
        episode_id=external_receipt.phase1_episode_id,
    )
    if metadata != expected_metadata:
        raise D99RACGTMKError("D99 validation feature archive metadata drift")
    physical_digest = _canonical_sha256(sorted(physical))
    payload = {
        "schema": VALIDATION_ARTIFACT_SCHEMA,
        "validation_features": _array_receipt(features),
        "physical_ids_sha256": physical_digest,
        "external_validation_receipt_sha256": external_receipt.receipt_sha256,
        "authority_envelope_sha256": authority_envelope.envelope_sha256,
        "authority_status": authority_envelope.authority_status,
        "formal_phase1_eligible": authority_envelope.formal_phase1_eligible,
        "validation_method_lock_sha256": method_lock.lock_digest,
        "archive_serialized_bytes": len(archive),
        "lifecycle": VALIDATION_LIFECYCLE,
        "source_validation_only": True,
        "target_or_query_rows_present": False,
        "single_received_observation": True,
    }
    return Phase1ValidationArtifact(
        validation_features_fp32=features,
        physical_ids_sha256=physical_digest,
        external_receipt=external_receipt,
        authority_envelope=authority_envelope,
        validation_method_lock_sha256=method_lock.lock_digest,
        archive_serialized_bytes=len(archive),
        artifact_sha256=_canonical_sha256(payload),
        loader_token=_VALIDATION_LOADER_TOKEN,
    )


@dataclass(frozen=True)
class GroundGeometry:
    """Immutable aggregate-only ground geometry; no member IDs or class score."""

    ground_classes: tuple[str, ...]
    class_means_fp32: np.ndarray
    nuisance_basis_fp32: np.ndarray
    nuisance_spectrum_fp32: np.ndarray
    density_weights_fp32: np.ndarray
    effective_domain_count: float
    domain_registry_sha256: str
    class_registry_sha256: str
    config_lock_digest: str
    ground_bundle_receipt_sha256: str
    geometry_receipt_sha256: str
    coverage_certificate: Mapping[str, Any]

    def __post_init__(self) -> None:
        classes = _registry(self.ground_classes, "ground_classes")
        rank = int(np.asarray(self.nuisance_basis_fp32).shape[1])
        domains = int(np.asarray(self.density_weights_fp32).shape[0])
        if (
            self.class_means_fp32.dtype != np.float32
            or self.class_means_fp32.shape != (len(classes), Z_DIM)
            or self.nuisance_basis_fp32.dtype != np.float32
            or self.nuisance_basis_fp32.shape != (Z_DIM, rank)
            or self.nuisance_spectrum_fp32.dtype != np.float32
            or self.nuisance_spectrum_fp32.shape != (rank,)
            or self.density_weights_fp32.dtype != np.float32
            or self.density_weights_fp32.shape != (domains,)
            or domains < 2
            or rank > 4
            or not np.isfinite(self.class_means_fp32).all()
            or not np.isfinite(self.nuisance_basis_fp32).all()
            or not np.isfinite(self.nuisance_spectrum_fp32).all()
            or not np.isfinite(self.density_weights_fp32).all()
            or np.any(self.nuisance_spectrum_fp32 <= 0.0)
            or np.any(self.density_weights_fp32 <= 0.0)
            or not np.isclose(np.sum(self.density_weights_fp32), 1.0, atol=2e-6)
            or not 1.0 <= float(self.effective_domain_count) <= domains
        ):
            raise D99RACGTMKError("D99 ground geometry invariant drift")
        if rank and not np.allclose(
            self.nuisance_basis_fp32.T @ self.nuisance_basis_fp32,
            np.eye(rank),
            atol=3e-5,
        ):
            raise D99RACGTMKError("D99 ground nuisance basis is not orthonormal")
        if not np.allclose(
            np.linalg.norm(self.class_means_fp32, axis=1), 1.0, atol=3e-5
        ):
            raise D99RACGTMKError("D99 ground class means are not normalized")
        for value, name in (
            (self.domain_registry_sha256, "domain_registry_sha256"),
            (self.class_registry_sha256, "class_registry_sha256"),
            (self.config_lock_digest, "config_lock_digest"),
            (self.ground_bundle_receipt_sha256, "ground_bundle_receipt_sha256"),
            (self.geometry_receipt_sha256, "geometry_receipt_sha256"),
        ):
            _require_sha256(value, name)
        object.__setattr__(self, "class_means_fp32", _readonly(self.class_means_fp32, np.float32))
        object.__setattr__(self, "nuisance_basis_fp32", _readonly(self.nuisance_basis_fp32, np.float32))
        object.__setattr__(self, "nuisance_spectrum_fp32", _readonly(self.nuisance_spectrum_fp32, np.float32))
        object.__setattr__(self, "density_weights_fp32", _readonly(self.density_weights_fp32, np.float32))
        object.__setattr__(self, "coverage_certificate", MappingProxyType(dict(self.coverage_certificate)))


def _geometry_payload(
    *,
    ground_classes: tuple[str, ...],
    class_means: np.ndarray,
    basis: np.ndarray,
    spectrum: np.ndarray,
    weights: np.ndarray,
    effective_domains: float,
    domain_registry_sha256: str,
    class_registry_sha256: str,
    config: Phase1D99Lock,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": GROUND_SCHEMA,
        "ground_classes": list(ground_classes),
        "class_means": _array_receipt(class_means),
        "nuisance_basis": _array_receipt(basis),
        "nuisance_spectrum": _array_receipt(spectrum),
        "density_weights": _array_receipt(weights),
        "effective_domain_count": float(effective_domains),
        "domain_registry_sha256": domain_registry_sha256,
        "class_registry_sha256": class_registry_sha256,
        "config_lock_digest": config.lock_digest,
        "ground_bundle_receipt_sha256": config.ground_bundle_receipt_sha256,
        "coverage_certificate": certificate,
    }


def build_ground_geometry(
    bundle: Phase1GroundAggregateBundle,
    *,
    config: Phase1D99Lock,
) -> GroundGeometry:
    """Build density inverse weighting and a rank<=4 shared nuisance basis."""

    if type(config) is not Phase1D99Lock or type(bundle) is not Phase1GroundAggregateBundle:
        raise D99RACGTMKError(
            "D99 ground builder requires exact typed bundle and Phase1 lock"
        )
    if (
        bundle.bundle_sha256 != config.ground_bundle_receipt_sha256
        or bundle.aggregation_receipt.receipt_sha256
        != config.ground_aggregation_receipt_sha256
        or bundle.ground_old_registry != config.ground_old_registry
    ):
        raise D99RACGTMKError("D99 typed ground bundle/lock registry receipt drift")
    # Reconstruct aggregate centroids only from sealed INT8 payload and scales.
    centroids = (
        bundle.codes_qint8.astype(np.float32)
        * bundle.scales_fp16.astype(np.float32)[:, :, None]
    )
    mask = bundle.domain_class_mask
    domains = bundle.domain_ids
    classes = bundle.ground_old_registry
    if (
        centroids.shape != (len(domains), len(classes), Z_DIM)
        or mask.dtype != np.bool_
        or mask.shape != centroids.shape[:2]
        or not np.isfinite(centroids).all()
        or not np.all(mask)
    ):
        raise D99RACGTMKError(
            "D99 ground input must be finite float32 [D,C,160] with a complete bool mask"
        )
    active = _normalized_rows(
        centroids.reshape(-1, Z_DIM), Z_DIM, "aggregate ground centroids"
    ).reshape(len(domains), len(classes), Z_DIM)
    initial_mean = np.mean(active, axis=0)
    raw_residual = active - initial_mean[None, :, :]
    signatures = raw_residual.reshape(len(domains), -1)
    signature_norm = np.linalg.norm(signatures, axis=1, keepdims=True)
    if float(np.max(signature_norm)) <= EPSILON:
        normalized_signatures = np.zeros_like(signatures)
        weights = np.full(len(domains), 1.0 / len(domains), dtype=np.float64)
        effective_domains = 1.0
    else:
        normalized_signatures = signatures / np.maximum(signature_norm, EPSILON)
        cosine = np.clip(normalized_signatures @ normalized_signatures.T, -1.0, 1.0)
        density = np.sum(
            np.exp(-(1.0 - cosine) / float(config.density_tau)), axis=1
        )
        if not np.isfinite(density).all() or np.any(density <= EPSILON):
            raise D99RACGTMKError("D99 ground density is not positive and finite")
        weights = 1.0 / density
        weights /= np.sum(weights)
        centered = normalized_signatures - np.sum(
            weights[:, None] * normalized_signatures, axis=0
        )
        weighted = np.sqrt(weights)[:, None] * centered
        diversity_values = np.linalg.eigvalsh(weighted @ weighted.T)
        diversity_values = diversity_values[diversity_values > EPSILON]
        effective_domains = (
            1.0
            if len(diversity_values) == 0
            else float(
                np.square(np.sum(diversity_values))
                / np.sum(np.square(diversity_values))
            )
        )
    class_mean_raw = np.einsum("d,dcz->cz", weights, active)
    class_means = _normalized_rows(class_mean_raw, Z_DIM, "ground class means")
    residual = active - class_mean_raw[None, :, :]
    covariance = np.einsum("d,dcz,dcw->zw", weights, residual, residual)
    covariance /= float(len(classes))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    positive = np.flatnonzero(eigenvalues > EPSILON)
    adaptive = max(0, int(math.floor(effective_domains)) - 1)
    rank = min(
        int(config.max_ground_rank), adaptive, len(positive), len(domains) - 1
    )
    if rank:
        order = positive[np.argsort(eigenvalues[positive], kind="stable")[-rank:][::-1]]
        basis = eigenvectors[:, order]
        spectrum = eigenvalues[order]
    else:
        basis = np.empty((Z_DIM, 0), dtype=np.float64)
        spectrum = np.empty(0, dtype=np.float64)
    total_trace = float(np.sum(np.clip(eigenvalues, 0.0, None)))
    retained_trace = float(np.sum(spectrum))
    ground_resource = _ground_resource_from_dimensions(len(domains), len(classes))
    certificate = {
        "schema": "cvs.phase1.d99.coverage_certificate.v1",
        "aggregate_only": True,
        "member_or_sample_ids_stored": False,
        "ground_class_score_access": False,
        "domain_count": len(domains),
        "class_count": len(classes),
        "effective_domain_count": effective_domains,
        "adaptive_rank_policy": "min(max_rank,floor(D_eff)-1,positive_rank,D-1)",
        "retained_rank": rank,
        "retained_trace_fraction": (
            retained_trace / total_trace if total_trace > EPSILON else 0.0
        ),
        "density_weight_min": float(np.min(weights)),
        "density_weight_max": float(np.max(weights)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "query_or_target_rows_used": 0,
        "typed_int8_aggregate_bundle": True,
        "external_aggregation_receipt_sha256": (
            bundle.aggregation_receipt.receipt_sha256
        ),
        "ground_bundle_sha256": bundle.bundle_sha256,
        "physical_sample_count_floor_min": int(
            np.min(bundle.physical_sample_count_floor_uint16)
        ),
        **ground_resource,
        "combined_method_resource_status": "BLOCKED_CORRECTED_TYPED_D81_REVIEW_P0",
    }
    class_means32 = _readonly(class_means, np.float32)
    basis32 = _readonly(basis, np.float32)
    spectrum32 = _readonly(spectrum, np.float32)
    weights32 = _readonly(weights, np.float32)
    domain_digest = _canonical_sha256(list(domains))
    class_digest = _canonical_sha256(list(classes))
    payload = _geometry_payload(
        ground_classes=classes,
        class_means=class_means32,
        basis=basis32,
        spectrum=spectrum32,
        weights=weights32,
        effective_domains=effective_domains,
        domain_registry_sha256=domain_digest,
        class_registry_sha256=class_digest,
        config=config,
        certificate=certificate,
    )
    return GroundGeometry(
        ground_classes=classes,
        class_means_fp32=class_means32,
        nuisance_basis_fp32=basis32,
        nuisance_spectrum_fp32=spectrum32,
        density_weights_fp32=weights32,
        effective_domain_count=effective_domains,
        domain_registry_sha256=domain_digest,
        class_registry_sha256=class_digest,
        config_lock_digest=config.lock_digest,
        ground_bundle_receipt_sha256=config.ground_bundle_receipt_sha256,
        geometry_receipt_sha256=_canonical_sha256(payload),
        coverage_certificate=certificate,
    )


def _support_closure(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    registered_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int, str]:
    features = _finite_features(support_features, "support_features")
    classes = _registry(registered_classes, "registered_classes")
    labels = np.asarray(tuple(str(value) for value in support_labels), dtype=np.str_)
    physical = np.asarray(tuple(str(value) for value in physical_ids), dtype=np.str_)
    if labels.shape != (len(features),) or physical.shape != labels.shape:
        raise D99RACGTMKError("support labels/physical IDs must align with features")
    if any(not value for value in physical.tolist()) or len(set(physical.tolist())) != len(
        physical
    ):
        raise D99RACGTMKError("support physical IDs must be unique and nonempty")
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        positions = np.asarray([lookup[value] for value in labels], dtype=np.int16)
    except KeyError as exc:
        raise D99RACGTMKError("support label absent from registered_classes") from exc
    counts = np.bincount(positions.astype(np.int64), minlength=len(classes))
    if np.any(counts <= 0) or len(np.unique(counts)) != 1:
        raise D99RACGTMKError("D99 requires balanced K-shot support for all classes")
    k_shot = int(counts[0])
    if k_shot not in ALLOWED_K:
        raise D99RACGTMKError(f"D99 supports only K in {ALLOWED_K}")
    order = np.asarray(
        sorted(range(len(features)), key=lambda index: str(physical[index])),
        dtype=np.int64,
    )
    receipt = _canonical_sha256(
        {
            "schema": "cvs.phase2.d99.support_input.v1",
            "classes": list(classes),
            "labels": labels[order].tolist(),
            "physical_ids_sha256": _canonical_sha256(physical[order].tolist()),
            "features": _array_receipt(features[order]),
        }
    )
    # Canonicalize all downstream support-only reductions by physical identity.
    # The identifiers themselves are represented only through the receipt and
    # are never stored in the compiled metric or INT8 bank.
    return (
        np.ascontiguousarray(features[order]),
        labels[order],
        positions[order],
        classes,
        k_shot,
        receipt,
    )


def _ground_resource_from_dimensions(domain_count: int, class_count: int) -> dict[str, int]:
    d_count, c_count = int(domain_count), int(class_count)
    if d_count < 2 or c_count < 2:
        raise D99RACGTMKError("D99 ground resource dimensions drift")
    return {
        "ground_build_mac_upper_bound": int(
            d_count**2 * c_count * Z_DIM + d_count * c_count * Z_DIM**2
        ),
        "ground_build_dense_eigh_flop_upper_bound": int(9 * Z_DIM**3),
        "ground_build_peak_live_dcz_float64_array_count": 6,
        "ground_build_peak_transient_bytes_upper_bound": int(
            d_count * c_count * Z_DIM * np.dtype(np.float32).itemsize
            + 6 * d_count * c_count * Z_DIM * np.dtype(np.float64).itemsize
            + 5 * Z_DIM * Z_DIM * np.dtype(np.float64).itemsize
            + 2 * d_count**2 * np.dtype(np.float64).itemsize
            + 4 * c_count * Z_DIM * np.dtype(np.float64).itemsize
        ),
    }


def _metric_resource_from_dimensions(
    *,
    class_count: int,
    k_shot: int,
    ground_domain_count: int,
    ground_class_count: int,
    ground_rank: int,
    target_rank: int,
    combined_rank: int,
) -> dict[str, Any]:
    support_rows = int(class_count) * int(k_shot)
    normalization_macs = support_rows * FEATURE_DIM * 3
    coverage_macs = int(ground_class_count) * Z_DIM * max(1, int(ground_rank)) * 2
    residual_covariance_macs = (
        0 if int(k_shot) == 1 else support_rows * Z_DIM * Z_DIM
    )
    low_rank_macs = (
        Z_DIM * max(1, int(combined_rank)) ** 2
        + support_rows * Z_DIM * max(1, int(target_rank))
    )
    ground_resource = _ground_resource_from_dimensions(
        ground_domain_count, ground_class_count
    )
    return {
        "schema": "cvs.phase2.d99.metric_resource.v1",
        "support_fit_mac_upper_bound": int(
            normalization_macs + coverage_macs + residual_covariance_macs + low_rank_macs
        ),
        "support_fit_dense_linear_algebra_flop_upper_bound": int(
            (0 if int(k_shot) == 1 else 9 * Z_DIM**3)
            + 9 * Z_DIM * max(1, int(combined_rank)) ** 2
        ),
        "support_fit_peak_transient_bytes_upper_bound": int(
            support_rows * FEATURE_DIM * np.dtype(np.float64).itemsize
            + 4 * Z_DIM * Z_DIM * np.dtype(np.float64).itemsize
            + support_rows * Z_DIM * np.dtype(np.float64).itemsize
        ),
        "residual_covariance_mac_upper_bound": int(residual_covariance_macs),
        "ground_build_mac_upper_bound": ground_resource["ground_build_mac_upper_bound"],
        "ground_build_dense_eigh_flop_upper_bound": ground_resource[
            "ground_build_dense_eigh_flop_upper_bound"
        ],
        "ground_build_peak_transient_bytes_upper_bound": ground_resource[
            "ground_build_peak_transient_bytes_upper_bound"
        ],
        "optimizer_steps": 0,
        "optimizer_steps_scope": "D99_incremental_only",
        "d99_incremental_optimizer_steps": 0,
        "trainable_parameters": 0,
        "d81_base_fit_included": False,
        "d81_base_single_fit_resource_status": "BLOCKED_CORRECTED_TYPED_D81_REVIEW_P0",
        "total_combined_resource_status": "BLOCKED_NOT_CLAIMED",
        "complete_method_resource_claim": False,
        "scope": "D99_incremental_only",
    }


@dataclass(frozen=True)
class SupportMetricState:
    classes: tuple[str, ...]
    k_shot: int
    metric_basis_fp32: np.ndarray
    precision_attenuation_fp32: np.ndarray
    target_basis_fp32: np.ndarray
    ground_coverage_rho: float
    target_shift_energy: float
    ground_weight: float
    target_weight: float
    ground_domain_count: int
    ground_class_count: int
    ground_rank: int
    support_input_sha256: str
    ground_geometry_receipt_sha256: str
    config_lock_digest: str
    metric_receipt_sha256: str
    fit_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        classes = _registry(self.classes, "metric classes")
        rank = self.metric_basis_fp32.shape[1]
        target_rank = self.target_basis_fp32.shape[1]
        if (
            self.k_shot not in ALLOWED_K
            or self.metric_basis_fp32.dtype != np.float32
            or self.metric_basis_fp32.shape != (Z_DIM, rank)
            or self.precision_attenuation_fp32.dtype != np.float32
            or self.precision_attenuation_fp32.shape != (rank,)
            or self.target_basis_fp32.dtype != np.float32
            or self.target_basis_fp32.shape != (Z_DIM, target_rank)
            or rank > 8
            or target_rank > 4
            or (self.k_shot == 1 and target_rank != 0)
            or not np.isfinite(self.metric_basis_fp32).all()
            or not np.isfinite(self.precision_attenuation_fp32).all()
            or np.any(self.precision_attenuation_fp32 < 0.0)
            or np.any(self.precision_attenuation_fp32 >= 1.0)
            or not all(
                0.0 <= float(value) <= 1.0
                for value in (
                    self.ground_coverage_rho,
                    self.ground_weight,
                    self.target_weight,
                )
            )
            or float(self.target_shift_energy) < 0.0
            or int(self.ground_domain_count) < 2
            or int(self.ground_class_count) < 2
            or not 0 <= int(self.ground_rank) <= 4
        ):
            raise D99RACGTMKError("D99 support metric invariant drift")
        if rank and not np.allclose(
            self.metric_basis_fp32.T @ self.metric_basis_fp32,
            np.eye(rank),
            atol=3e-5,
        ):
            raise D99RACGTMKError("D99 combined metric basis is not orthonormal")
        for value, name in (
            (self.support_input_sha256, "support_input_sha256"),
            (self.ground_geometry_receipt_sha256, "ground_geometry_receipt_sha256"),
            (self.config_lock_digest, "config_lock_digest"),
            (self.metric_receipt_sha256, "metric_receipt_sha256"),
        ):
            _require_sha256(value, name)
        object.__setattr__(self, "metric_basis_fp32", _readonly(self.metric_basis_fp32, np.float32))
        object.__setattr__(self, "precision_attenuation_fp32", _readonly(self.precision_attenuation_fp32, np.float32))
        object.__setattr__(self, "target_basis_fp32", _readonly(self.target_basis_fp32, np.float32))
        object.__setattr__(self, "fit_audit", MappingProxyType(dict(self.fit_audit)))
        object.__setattr__(self, "resource_audit", MappingProxyType(dict(self.resource_audit)))
        if not _verify_metric_numeric_resource(self):
            raise D99RACGTMKError("D99 metric numeric resource/receipt drift")

    @property
    def is_identity(self) -> bool:
        return self.metric_basis_fp32.shape[1] == 0

    def apply_precision(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != Z_DIM or not np.isfinite(values).all():
            raise D99RACGTMKError("precision input must be finite [N,160]")
        basis = self.metric_basis_fp32.astype(np.float64)
        if basis.shape[1] == 0:
            return values.copy()
        attenuation = self.precision_attenuation_fp32.astype(np.float64)
        return values - ((values @ basis) * attenuation[None, :]) @ basis.T


def _metric_payload(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    basis: np.ndarray,
    attenuation: np.ndarray,
    target_basis: np.ndarray,
    rho: float,
    shift_energy: float,
    ground_weight: float,
    target_weight: float,
    support_receipt: str,
    ground: GroundGeometry,
    config: Phase1D99Lock,
    audit: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": METRIC_SCHEMA,
        "classes": list(classes),
        "k_shot": k_shot,
        "metric_basis": _array_receipt(basis),
        "precision_attenuation": _array_receipt(attenuation),
        "target_basis": _array_receipt(target_basis),
        "ground_coverage_rho": rho,
        "target_shift_energy": shift_energy,
        "ground_weight": ground_weight,
        "target_weight": target_weight,
        "ground_domain_count": int(len(ground.density_weights_fp32)),
        "ground_class_count": int(len(ground.ground_classes)),
        "ground_rank": int(ground.nuisance_basis_fp32.shape[1]),
        "support_input_sha256": support_receipt,
        "ground_geometry_receipt_sha256": ground.geometry_receipt_sha256,
        "config_lock_digest": config.lock_digest,
        "fit_audit": audit,
        "resource_audit": resource,
    }


def fit_support_metric(
    ground: GroundGeometry,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    ground_target_class_ids: Sequence[str],
    *,
    config: Phase1D99Lock,
) -> SupportMetricState:
    """Fit one class-agnostic PSD metric using target support only."""

    if type(ground) is not GroundGeometry or type(config) is not Phase1D99Lock:
        raise D99RACGTMKError("D99 fit requires exact ground/config types")
    if (
        ground.config_lock_digest != config.lock_digest
        or ground.ground_bundle_receipt_sha256 != config.ground_bundle_receipt_sha256
        or ground.ground_classes != config.ground_old_registry
        or ground.class_registry_sha256
        != _canonical_sha256(list(config.ground_old_registry))
    ):
        raise D99RACGTMKError("D99 ground/config receipt drift")
    expected_geometry_receipt = _canonical_sha256(
        _geometry_payload(
            ground_classes=ground.ground_classes,
            class_means=ground.class_means_fp32,
            basis=ground.nuisance_basis_fp32,
            spectrum=ground.nuisance_spectrum_fp32,
            weights=ground.density_weights_fp32,
            effective_domains=ground.effective_domain_count,
            domain_registry_sha256=ground.domain_registry_sha256,
            class_registry_sha256=ground.class_registry_sha256,
            config=config,
            certificate=ground.coverage_certificate,
        )
    )
    if expected_geometry_receipt != ground.geometry_receipt_sha256:
        raise D99RACGTMKError("D99 ground geometry receipt verification failed")
    features, labels, positions, classes, k_shot, support_receipt = _support_closure(
        support_features, support_labels, physical_ids, registered_classes
    )
    ground_targets = tuple(str(value) for value in ground_target_class_ids)
    if (
        ground_targets != config.ground_old_registry
        or ground_targets != ground.ground_classes
        or any(value not in classes for value in ground_targets)
    ):
        raise D99RACGTMKError(
            "ground-to-target mapping must equal the sealed Y_old registry"
        )
    normalized = normalize_feature_blocks(features).astype(np.float64)
    z = normalized[:, :Z_DIM]
    class_means = np.stack(
        [np.mean(z[positions == index], axis=0) for index in range(len(classes))]
    )
    class_means = _normalized_rows(class_means, Z_DIM, "target support class means")
    target_lookup = {label: index for index, label in enumerate(classes)}
    old_centers = np.stack([class_means[target_lookup[value]] for value in ground_targets])
    delta = old_centers - ground.class_means_fp32.astype(np.float64)
    delta -= np.mean(delta, axis=0, keepdims=True)
    shift_energy = float(np.mean(np.sum(np.square(delta), axis=1)))
    ground_basis = ground.nuisance_basis_fp32.astype(np.float64)
    if shift_energy <= EPSILON or ground_basis.shape[1] == 0:
        rho = 0.0
    else:
        projected = (delta @ ground_basis) @ ground_basis.T
        rho = float(
            np.clip(
                np.sum(np.square(projected)) / np.sum(np.square(delta)), 0.0, 1.0
            )
        )
    old_support_count = len(ground_targets) * k_shot
    diversity_factor = max(0.0, ground.effective_domain_count - 1.0) / max(
        1.0, ground.effective_domain_count
    )
    energy_factor = shift_energy / (shift_energy + config.ground_energy_scale)
    count_factor = old_support_count / (
        old_support_count + config.shrinkage_prior_strength
    )
    ground_weight = float(
        min(
            config.ground_weight_max,
            rho * diversity_factor * energy_factor * count_factor,
        )
    )
    low_coverage_identity = bool(
        rho <= config.coverage_floor
        or ground.effective_domain_count <= 1.0 + 1e-9
        or ground_basis.shape[1] == 0
    )
    if low_coverage_identity:
        ground_weight = 0.0

    if k_shot == 1:
        target_basis = np.empty((Z_DIM, 0), dtype=np.float64)
        target_values = np.empty(0, dtype=np.float64)
        residual_energy = 0.0
    else:
        residual = z - class_means[positions]
        covariance = residual.T @ residual / float(len(classes) * (k_shot - 1))
        covariance = 0.5 * (covariance + covariance.T)
        values, vectors = np.linalg.eigh(covariance)
        positive = np.flatnonzero(values > EPSILON)
        target_rank = min(
            int(config.max_target_rank), len(positive), len(classes) * (k_shot - 1)
        )
        if target_rank:
            order = positive[np.argsort(values[positive], kind="stable")[-target_rank:][::-1]]
            target_basis = vectors[:, order]
            target_values = values[order]
        else:
            target_basis = np.empty((Z_DIM, 0), dtype=np.float64)
            target_values = np.empty(0, dtype=np.float64)
        residual_energy = float(np.trace(covariance))
    residual_count = len(classes) * max(0, k_shot - 1)
    target_weight = float(
        min(
            config.target_weight_max,
            (residual_energy / (residual_energy + config.target_energy_scale))
            * (
                residual_count
                / (residual_count + config.shrinkage_prior_strength)
                if residual_count
                else 0.0
            ),
        )
    )
    factors: list[np.ndarray] = []
    if ground_weight > 0.0:
        spectrum = ground.nuisance_spectrum_fp32.astype(np.float64)
        normalized_spectrum = spectrum / max(float(np.sum(spectrum)), EPSILON)
        factors.append(
            ground_basis * np.sqrt(ground_weight * len(spectrum) * normalized_spectrum)[None, :]
        )
    if target_weight > 0.0 and target_basis.shape[1]:
        normalized_target = target_values / max(float(np.sum(target_values)), EPSILON)
        factors.append(
            target_basis
            * np.sqrt(target_weight * len(target_values) * normalized_target)[None, :]
        )
    if factors:
        factor = np.concatenate(factors, axis=1)
        combined_basis, singular, _ = np.linalg.svd(factor, full_matrices=False)
        positive = singular > EPSILON
        combined_basis = combined_basis[:, positive]
        penalties = np.square(singular[positive])
        attenuation = penalties / (1.0 + penalties)
    else:
        combined_basis = np.empty((Z_DIM, 0), dtype=np.float64)
        attenuation = np.empty(0, dtype=np.float64)
    min_precision_eigenvalue = (
        1.0 if len(attenuation) == 0 else float(1.0 - np.max(attenuation))
    )
    if min_precision_eigenvalue <= 0.0 or not np.isfinite(min_precision_eigenvalue):
        raise D99RACGTMKError("D99 analytic precision is not strictly PSD")
    basis32 = _readonly(combined_basis, np.float32)
    attenuation32 = _readonly(attenuation, np.float32)
    target_basis32 = _readonly(target_basis, np.float32)
    audit = {
        "schema": "cvs.phase2.d99.support_metric_fit_audit.v1",
        "support_only": True,
        "balanced_all_registered_classes": True,
        "old_ground_centers_use": "rho_e_delta_omega_g_only",
        "ground_class_logit_or_bonus": False,
        "target_basis_source": "all_registered_class_balanced_within_class_residual",
        "old_new_role_specific_scoring": False,
        "k_shot": k_shot,
        "ground_coverage_rho": rho,
        "target_shift_energy": shift_energy,
        "e_delta": shift_energy,
        "ground_weight": ground_weight,
        "target_weight": target_weight,
        "omega_g": ground_weight,
        "omega_t": target_weight,
        "ground_rank": int(ground_basis.shape[1]),
        "ground_domain_count": int(len(ground.density_weights_fp32)),
        "ground_class_count": int(len(ground.ground_classes)),
        "target_rank": int(target_basis.shape[1]),
        "combined_rank": int(combined_basis.shape[1]),
        "k1_target_rank_exact_zero": bool(k_shot == 1),
        "low_coverage_ground_identity_fallback": low_coverage_identity,
        "target_metric_survives_low_ground_coverage": bool(
            low_coverage_identity and target_weight > 0.0
        ),
        "precision_min_eigenvalue": min_precision_eigenvalue,
        "precision_formula": "I-B*diag(lambda/(1+lambda))*B.T",
        "full_coordinate_transport": False,
        "query_rows_used": 0,
    }
    resource = _metric_resource_from_dimensions(
        class_count=len(classes),
        k_shot=k_shot,
        ground_domain_count=len(ground.density_weights_fp32),
        ground_class_count=len(ground.ground_classes),
        ground_rank=ground_basis.shape[1],
        target_rank=target_basis.shape[1],
        combined_rank=combined_basis.shape[1],
    )
    payload = _metric_payload(
        classes=classes,
        k_shot=k_shot,
        basis=basis32,
        attenuation=attenuation32,
        target_basis=target_basis32,
        rho=rho,
        shift_energy=shift_energy,
        ground_weight=ground_weight,
        target_weight=target_weight,
        support_receipt=support_receipt,
        ground=ground,
        config=config,
        audit=audit,
        resource=resource,
    )
    return SupportMetricState(
        classes=classes,
        k_shot=k_shot,
        metric_basis_fp32=basis32,
        precision_attenuation_fp32=attenuation32,
        target_basis_fp32=target_basis32,
        ground_coverage_rho=rho,
        target_shift_energy=shift_energy,
        ground_weight=ground_weight,
        target_weight=target_weight,
        ground_domain_count=len(ground.density_weights_fp32),
        ground_class_count=len(ground.ground_classes),
        ground_rank=ground_basis.shape[1],
        support_input_sha256=support_receipt,
        ground_geometry_receipt_sha256=ground.geometry_receipt_sha256,
        config_lock_digest=config.lock_digest,
        metric_receipt_sha256=_canonical_sha256(payload),
        fit_audit=audit,
        resource_audit=resource,
    )


def _quantize_rows(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(normalized, dtype=np.float32)
    codes = np.zeros(rows.shape, dtype=np.int8)
    scales = np.zeros((len(rows), len(BLOCK_SLICES)), dtype=np.float16)
    decoded = np.zeros(rows.shape, dtype=np.float32)
    minimum_scale = float(np.finfo(np.float16).tiny)
    for row_index in range(len(rows)):
        for block_index, block in enumerate(BLOCK_SLICES):
            part = rows[row_index, block]
            scale = np.float16(
                max(float(np.max(np.abs(part))) / INT8_MAX, minimum_scale)
            )
            if not np.isfinite(scale) or scale <= 0:
                raise D99RACGTMKError("D99 INT8 quantization scale overflow")
            code = np.clip(np.rint(part / float(scale)), -127, 127).astype(np.int8)
            codes[row_index, block] = code
            scales[row_index, block_index] = scale
            decoded[row_index, block] = code.astype(np.float32) * np.float32(scale)
    return codes, scales, normalize_feature_blocks(decoded)


def _precision_cosine(
    left: np.ndarray, right: np.ndarray, metric: SupportMetricState
) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    p_left = metric.apply_precision(left64)
    p_right = metric.apply_precision(right64)
    numerator = left64 @ p_right.T
    left_norm = np.sqrt(np.maximum(np.sum(left64 * p_left, axis=1), EPSILON))
    right_norm = np.sqrt(np.maximum(np.sum(right64 * p_right, axis=1), EPSILON))
    return np.clip(numerator / (left_norm[:, None] * right_norm[None, :]), -1.0, 1.0)


def _pair_distance_squared(
    left: np.ndarray,
    right: np.ndarray,
    metric: SupportMetricState,
    config: Phase1D99Lock,
) -> np.ndarray:
    z_cosine = _precision_cosine(left[:, :Z_DIM], right[:, :Z_DIM], metric)
    fft_cosine = np.clip(left[:, 160:256] @ right[:, 160:256].T, -1.0, 1.0)
    rf_cosine = np.clip(left[:, 256:288] @ right[:, 256:288].T, -1.0, 1.0)
    distance = 2.0 * (
        config.z_weight * (1.0 - z_cosine)
        + config.fft_weight * (1.0 - fft_cosine)
        + config.rf_weight * (1.0 - rf_cosine)
    )
    return np.maximum(distance, 0.0)


@dataclass(frozen=True)
class TypedINT8MetricKernelBank:
    classes: tuple[str, ...]
    support_counts: tuple[int, ...]
    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    class_indices_int16: np.ndarray
    class_scales_fp16: np.ndarray
    metric: SupportMetricState
    config: Phase1D99Lock
    eta_phase1_locked: float
    bank_receipt_sha256: str
    quantization_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    deployment_status: str = DEPLOYMENT_STATUS

    def __post_init__(self) -> None:
        classes = _registry(self.classes, "bank classes")
        rows = sum(self.support_counts)
        if (
            self.metric.classes != classes
            or self.metric.config_lock_digest != self.config.lock_digest
            or self.metric.ground_class_count != len(self.config.ground_old_registry)
            or len(self.support_counts) != len(classes)
            or any(value != self.metric.k_shot for value in self.support_counts)
            or self.codes_qint8.dtype != np.int8
            or self.codes_qint8.shape != (rows, FEATURE_DIM)
            or self.scales_fp16.dtype != np.float16
            or self.scales_fp16.shape != (rows, 3)
            or self.class_indices_int16.dtype != np.int16
            or self.class_indices_int16.shape != (rows,)
            or self.class_scales_fp16.dtype != np.float16
            or self.class_scales_fp16.shape != (len(classes),)
            or not np.isfinite(self.scales_fp16).all()
            or not np.isfinite(self.class_scales_fp16).all()
            or np.any(self.scales_fp16 <= 0.0)
            or np.any(self.class_scales_fp16 <= 0.0)
            or not np.array_equal(
                np.bincount(self.class_indices_int16.astype(np.int64), minlength=len(classes)),
                np.asarray(self.support_counts),
            )
            or self.eta_phase1_locked != self.config.eta_for_k(self.metric.k_shot)
            or self.deployment_status != DEPLOYMENT_STATUS
        ):
            raise D99RACGTMKError("D99 typed support bank invariant drift")
        _require_sha256(self.bank_receipt_sha256, "bank_receipt_sha256")
        for name, dtype in (
            ("codes_qint8", np.int8),
            ("scales_fp16", np.float16),
            ("class_indices_int16", np.int16),
            ("class_scales_fp16", np.float16),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        object.__setattr__(self, "quantization_audit", MappingProxyType(dict(self.quantization_audit)))
        object.__setattr__(self, "resource_audit", MappingProxyType(dict(self.resource_audit)))
        if not _verify_bank_numeric_resource_and_receipt(self):
            raise D99RACGTMKError("D99 bank numeric resource/receipt closure drift")


def _bank_metadata(
    *,
    classes: tuple[str, ...],
    counts: tuple[int, ...],
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
    config: Phase1D99Lock,
    quantization: Mapping[str, Any],
    resource: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": BANK_SCHEMA,
        "classes": list(classes),
        "support_counts": list(counts),
        "codes": _array_receipt(codes),
        "scales": _array_receipt(scales),
        "class_indices": _array_receipt(indices),
        "class_scales": _array_receipt(class_scales),
        "metric_receipt_sha256": metric.metric_receipt_sha256,
        "config_lock_digest": config.lock_digest,
        "phase1_locked_config": asdict(config),
        "eta_phase1_locked": config.eta_for_k(metric.k_shot),
        "quantization_audit": quantization,
        "deployment_status": DEPLOYMENT_STATUS,
        "required_typed_d81_state_schema": REQUIRED_TYPED_D81_STATE_SCHEMA,
    }
    if resource is not None:
        payload["resource_audit"] = resource
    return payload


def _serialize_runtime_artifact(
    metadata: Mapping[str, Any], arrays: Sequence[tuple[str, np.ndarray]]
) -> bytes:
    """Deterministic runtime serialization used for actual byte accounting."""

    output = bytearray(b"D99RA-CGTMK-RUNTIME\0")
    metadata_bytes = _canonical_bytes(metadata)
    output.extend(len(metadata_bytes).to_bytes(8, "little"))
    output.extend(metadata_bytes)
    for name, value in arrays:
        name_bytes = name.encode("ascii")
        array = np.ascontiguousarray(value)
        header = _canonical_bytes(
            {"dtype": array.dtype.str, "shape": list(array.shape)}
        )
        body = array.tobytes(order="C")
        output.extend(len(name_bytes).to_bytes(4, "little"))
        output.extend(name_bytes)
        output.extend(len(header).to_bytes(8, "little"))
        output.extend(header)
        output.extend(len(body).to_bytes(8, "little"))
        output.extend(body)
    return bytes(output)


def _bank_runtime_arrays(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("codes_qint8", codes),
        ("scales_fp16", scales),
        ("class_indices_int16", indices),
        ("class_scales_fp16", class_scales),
        ("metric_basis_fp32", metric.metric_basis_fp32),
        ("precision_attenuation_fp32", metric.precision_attenuation_fp32),
    )


def _receipt_bearing_bank_artifact(
    *,
    classes: tuple[str, ...],
    counts: tuple[int, ...],
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
    config: Phase1D99Lock,
    quantization: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> tuple[str, bytes]:
    core = _bank_metadata(
        classes=classes,
        counts=counts,
        codes=codes,
        scales=scales,
        indices=indices,
        class_scales=class_scales,
        metric=metric,
        config=config,
        quantization=quantization,
        resource=resource,
    )
    receipt = _canonical_sha256(core)
    wire_metadata = {**core, "bank_receipt_sha256": receipt}
    return receipt, _serialize_runtime_artifact(
        wire_metadata,
        _bank_runtime_arrays(
            codes=codes,
            scales=scales,
            indices=indices,
            class_scales=class_scales,
            metric=metric,
        ),
    )


def _bank_resource_from_numeric_state(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
    actual_serialized_bytes: int,
) -> dict[str, Any]:
    rows = int(codes.shape[0])
    rank = int(metric.metric_basis_fp32.shape[1])
    expected_metric = _metric_resource_from_dimensions(
        class_count=len(metric.classes),
        k_shot=metric.k_shot,
        ground_domain_count=metric.ground_domain_count,
        ground_class_count=metric.ground_class_count,
        ground_rank=metric.ground_rank,
        target_rank=metric.target_basis_fp32.shape[1],
        combined_rank=rank,
    )
    numeric_bytes = int(
        codes.nbytes
        + scales.nbytes
        + indices.nbytes
        + class_scales.nbytes
        + metric.metric_basis_fp32.nbytes
        + metric.precision_attenuation_fp32.nbytes
    )
    support_decode_normalize_macs = rows * FEATURE_DIM * 5
    query_precision_norm_macs = Z_DIM
    query_kernel_pair_macs = rows * (
        Z_DIM * (2 * rank + 3) + FFT_DIM + RF_DIM
    )
    query_kernel_macs = query_kernel_pair_macs + query_precision_norm_macs
    peak_transient = int(
        rows * FEATURE_DIM * np.dtype(np.float32).itemsize
        + rows * FEATURE_DIM * np.dtype(np.float64).itemsize
        + rows * np.dtype(np.float64).itemsize * 4
        + expected_metric["support_fit_peak_transient_bytes_upper_bound"]
    )
    return {
        "schema": "cvs.phase2.d99.resource_closure.v1",
        "logical_runtime_numeric_state_bytes": numeric_bytes,
        "actual_serialized_runtime_artifact_bytes": int(actual_serialized_bytes),
        "support_fit_mac_upper_bound": expected_metric["support_fit_mac_upper_bound"],
        "support_fit_dense_linear_algebra_flop_upper_bound": expected_metric[
            "support_fit_dense_linear_algebra_flop_upper_bound"
        ],
        "ground_build_mac_upper_bound": expected_metric["ground_build_mac_upper_bound"],
        "ground_build_dense_eigh_flop_upper_bound": expected_metric[
            "ground_build_dense_eigh_flop_upper_bound"
        ],
        "support_decode_normalize_mac_per_prediction_call": support_decode_normalize_macs,
        "query_precision_norm_mac_upper_bound": query_precision_norm_macs,
        "query_kernel_pair_mac_upper_bound": query_kernel_pair_macs,
        "query_kernel_mac_upper_bound": query_kernel_macs,
        "query_mac_upper_bound": int(
            support_decode_normalize_macs + query_kernel_macs + FEATURE_DIM * 5
        ),
        "peak_transient_bytes_upper_bound": peak_transient,
        "optimizer_steps": 0,
        "optimizer_steps_scope": "D99_incremental_only",
        "trainable_parameters": 0,
        "query_state_updates": 0,
        "persistent_query_batch_state_bytes": 0,
        "d81_base_fit_included": False,
        "d81_base_single_fit_resource_status": "BLOCKED_CORRECTED_TYPED_D81_REVIEW_P0",
        "total_combined_resource_status": "BLOCKED_NOT_CLAIMED",
        "complete_method_resource_claim": False,
        "scope": "D99_incremental_only",
    }


def _closed_bank_resource_receipt_artifact(
    *,
    classes: tuple[str, ...],
    counts: tuple[int, ...],
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
    config: Phase1D99Lock,
    quantization: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bytes]:
    actual = 0
    for _ in range(8):
        resource = _bank_resource_from_numeric_state(
            codes=codes,
            scales=scales,
            indices=indices,
            class_scales=class_scales,
            metric=metric,
            actual_serialized_bytes=actual,
        )
        receipt, artifact = _receipt_bearing_bank_artifact(
            classes=classes,
            counts=counts,
            codes=codes,
            scales=scales,
            indices=indices,
            class_scales=class_scales,
            metric=metric,
            config=config,
            quantization=quantization,
            resource=resource,
        )
        observed = len(artifact)
        if observed == actual:
            return resource, receipt, artifact
        actual = observed
    raise D99RACGTMKError("D99 receipt-bearing serialized size did not converge")


def _verify_bank_numeric_resource_and_receipt(
    bank: TypedINT8MetricKernelBank,
) -> bool:
    try:
        expected_resource, expected_receipt, artifact = (
            _closed_bank_resource_receipt_artifact(
                classes=bank.classes,
                counts=bank.support_counts,
                codes=bank.codes_qint8,
                scales=bank.scales_fp16,
                indices=bank.class_indices_int16,
                class_scales=bank.class_scales_fp16,
                metric=bank.metric,
                config=bank.config,
                quantization=bank.quantization_audit,
            )
        )
        return (
            _json_value(bank.resource_audit) == expected_resource
            and _verify_metric_numeric_resource(bank.metric)
            and bank.bank_receipt_sha256 == expected_receipt
            and len(artifact)
            == expected_resource["actual_serialized_runtime_artifact_bytes"]
        )
    except (D99RACGTMKError, KeyError, TypeError, ValueError):
        return False


def _metric_state_receipt(state: SupportMetricState) -> str:
    payload = {
        "schema": METRIC_SCHEMA,
        "classes": list(state.classes),
        "k_shot": state.k_shot,
        "metric_basis": _array_receipt(state.metric_basis_fp32),
        "precision_attenuation": _array_receipt(
            state.precision_attenuation_fp32
        ),
        "target_basis": _array_receipt(state.target_basis_fp32),
        "ground_coverage_rho": state.ground_coverage_rho,
        "target_shift_energy": state.target_shift_energy,
        "ground_weight": state.ground_weight,
        "target_weight": state.target_weight,
        "ground_domain_count": state.ground_domain_count,
        "ground_class_count": state.ground_class_count,
        "ground_rank": state.ground_rank,
        "support_input_sha256": state.support_input_sha256,
        "ground_geometry_receipt_sha256": state.ground_geometry_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "fit_audit": state.fit_audit,
        "resource_audit": state.resource_audit,
    }
    return _canonical_sha256(payload)


def _verify_metric_numeric_resource(state: SupportMetricState) -> bool:
    """Recompute metric resources from immutable dimensions and numeric arrays."""

    if type(state) is not SupportMetricState:
        return False
    try:
        expected = _metric_resource_from_dimensions(
            class_count=len(state.classes),
            k_shot=state.k_shot,
            ground_domain_count=state.ground_domain_count,
            ground_class_count=state.ground_class_count,
            ground_rank=state.ground_rank,
            target_rank=state.target_basis_fp32.shape[1],
            combined_rank=state.metric_basis_fp32.shape[1],
        )
        return (
            _json_value(state.resource_audit) == expected
            and _metric_state_receipt(state) == state.metric_receipt_sha256
        )
    except (D99RACGTMKError, KeyError, TypeError, ValueError):
        return False


def build_typed_support_bank(
    metric: SupportMetricState,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: Phase1D99Lock,
) -> TypedINT8MetricKernelBank:
    """Quantize target support and compile the uniform metric-kernel head."""

    if type(metric) is not SupportMetricState or type(config) is not Phase1D99Lock:
        raise D99RACGTMKError("D99 bank requires exact metric/config types")
    features, labels, positions, classes, k_shot, support_receipt = _support_closure(
        support_features, support_labels, physical_ids, registered_classes
    )
    if (
        classes != metric.classes
        or k_shot != metric.k_shot
        or support_receipt != metric.support_input_sha256
        or config.lock_digest != metric.config_lock_digest
        or _metric_state_receipt(metric) != metric.metric_receipt_sha256
    ):
        raise D99RACGTMKError("D99 bank support/metric/config receipt drift")
    normalized = normalize_feature_blocks(features)
    codes, scales, decoded = _quantize_rows(normalized)
    order = np.asarray(
        sorted(
            range(len(features)),
            key=lambda index: (
                int(positions[index]),
                codes[index].tobytes(),
                scales[index].tobytes(),
            ),
        ),
        dtype=np.int64,
    )
    codes, scales, decoded, positions = (
        codes[order],
        scales[order],
        decoded[order],
        positions[order],
    )
    normalized_ordered = np.asarray(normalized, dtype=np.float32)[order]
    if k_shot == 1:
        class_scales = np.full(len(classes), config.shared_h0, dtype=np.float64)
        scale_source = "phase1_locked_shared_h0"
    else:
        values = []
        for class_index in range(len(classes)):
            local = normalized_ordered[positions == class_index].astype(np.float64)
            pair = _pair_distance_squared(local, local, metric, config)
            upper = pair[np.triu_indices(k_shot, 1)]
            empirical = float(np.mean(upper))
            shrunk = (
                empirical + config.scale_prior_strength * config.shared_h0**2
            ) / (1.0 + config.scale_prior_strength)
            values.append(
                np.clip(
                    math.sqrt(max(shrunk, EPSILON)),
                    config.shared_h0 * config.scale_min_ratio,
                    config.shared_h0 * config.scale_max_ratio,
                )
            )
        class_scales = np.asarray(values, dtype=np.float64)
        scale_source = "support_only_uniform_class_formula"
    class_scales16 = _readonly(class_scales, np.float16)
    reconstruction_error = np.abs(
        decoded.astype(np.float64) - normalized_ordered.astype(np.float64)
    )
    reconstruction_cosine = np.mean(
        [
            np.sum(
                decoded[:, block].astype(np.float64)
                * normalized_ordered[:, block].astype(np.float64),
                axis=1,
            )
            for block in BLOCK_SLICES
        ],
        axis=0,
    )
    quantization = {
        "schema": "cvs.phase2.d99.int8_quantization_audit.v1",
        "support_only": True,
        "single_received_observation": True,
        "block_dims": list(BLOCK_DIMS),
        "quantization_error_mean": float(np.mean(reconstruction_error)),
        "quantization_error_max": float(np.max(reconstruction_error)),
        "reconstruction_block_cosine_mean": float(np.mean(reconstruction_cosine)),
        "class_scale_source": scale_source,
        "class_count_normalization": "logsumexp_minus_log_Kc",
        "student_t_kernel_formula": (
            "-gamma_v*d_k*log(h)-(nu+d_k)/2*log1p(d2/(nu*h2))"
        ),
        "kernel_effective_dim": int(config.kernel_effective_dim),
        "kernel_volume_gamma": float(config.kernel_volume_gamma),
        "same_formula_all_registered_classes": True,
        "query_rows_used_for_fit": 0,
    }
    counts = tuple(k_shot for _ in classes)
    resource, receipt, serialized_runtime_artifact = (
        _closed_bank_resource_receipt_artifact(
        classes=classes,
        counts=counts,
        codes=codes,
        scales=scales,
        indices=positions,
        class_scales=class_scales16,
        metric=metric,
        config=config,
        quantization=quantization,
        )
    )
    return TypedINT8MetricKernelBank(
        classes=classes,
        support_counts=counts,
        codes_qint8=codes,
        scales_fp16=scales,
        class_indices_int16=positions,
        class_scales_fp16=class_scales16,
        metric=metric,
        config=config,
        eta_phase1_locked=config.eta_for_k(k_shot),
        bank_receipt_sha256=receipt,
        quantization_audit=quantization,
        resource_audit=resource,
    )


def decode_support_bank(bank: TypedINT8MetricKernelBank) -> np.ndarray:
    if type(bank) is not TypedINT8MetricKernelBank:
        raise D99RACGTMKError("D99 decode requires an exact typed bank")
    decoded = np.zeros((len(bank.codes_qint8), FEATURE_DIM), dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        decoded[:, block] = (
            bank.codes_qint8[:, block].astype(np.float32)
            * bank.scales_fp16[:, block_index].astype(np.float32)[:, None]
        )
    return normalize_feature_blocks(decoded)


def _student_t_logits(
    support: np.ndarray,
    class_indices: np.ndarray,
    class_scales: np.ndarray,
    metric: SupportMetricState,
    config: Phase1D99Lock,
    query_features: np.ndarray,
) -> np.ndarray:
    query = normalize_feature_blocks(query_features).astype(np.float64)
    support64 = np.asarray(support, dtype=np.float64)
    distance = _pair_distance_squared(query, support64, metric, config)
    columns = []
    for class_index in range(len(metric.classes)):
        local = distance[:, class_indices == class_index]
        expected = metric.k_shot
        if local.shape[1] != expected:
            raise D99RACGTMKError("D99 support count drift during scoring")
        h = float(class_scales[class_index])
        effective_dim = int(config.kernel_effective_dim)
        kernel = (
            -float(config.kernel_volume_gamma) * effective_dim * math.log(h)
            - 0.5
            * (config.student_nu + effective_dim)
            * np.log1p(local / (config.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        column = maximum[:, 0] + np.log(
            np.sum(np.exp(kernel - maximum), axis=1)
        ) - math.log(expected)
        columns.append(column)
    result = np.stack(columns, axis=1)
    if not np.isfinite(result).all():
        raise D99RACGTMKError("D99 metric-kernel raw logits became non-finite")
    return _readonly(result, np.float32)


def score_metric_kernel_raw_logits(
    bank: TypedINT8MetricKernelBank, query_features: np.ndarray
) -> np.ndarray:
    """Score independent queries against all registered classes with one formula."""

    if type(bank) is not TypedINT8MetricKernelBank:
        raise D99RACGTMKError("D99 scoring requires an exact typed bank")
    if not _verify_metric_numeric_resource(bank.metric):
        raise D99RACGTMKError("D99 support metric numeric resource verification failed")
    if not _verify_bank_numeric_resource_and_receipt(bank):
        raise D99RACGTMKError("D99 typed bank numeric resource/receipt verification failed")
    return _student_t_logits(
        decode_support_bank(bank),
        bank.class_indices_int16,
        bank.class_scales_fp16,
        bank.metric,
        bank.config,
        query_features,
    )


def _serialize_receipt_bearing_bank(bank: TypedINT8MetricKernelBank) -> bytes:
    """Return the exact complete wire artifact used by the byte receipt."""

    if type(bank) is not TypedINT8MetricKernelBank:
        raise D99RACGTMKError("D99 serialization requires an exact typed bank")
    if not _verify_bank_numeric_resource_and_receipt(bank):
        raise D99RACGTMKError("D99 serialization numeric resource/receipt drift")
    resource, receipt, artifact = _closed_bank_resource_receipt_artifact(
        classes=bank.classes,
        counts=bank.support_counts,
        codes=bank.codes_qint8,
        scales=bank.scales_fp16,
        indices=bank.class_indices_int16,
        class_scales=bank.class_scales_fp16,
        metric=bank.metric,
        config=bank.config,
        quantization=bank.quantization_audit,
    )
    if receipt != bank.bank_receipt_sha256 or resource != _json_value(bank.resource_audit):
        raise D99RACGTMKError("D99 bank receipt drift during serialization")
    return artifact


def _phase1_quantized_margin_payload(
    bank: TypedINT8MetricKernelBank,
    full_precision_support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    validation_artifact: Phase1ValidationArtifact,
) -> dict[str, Any]:
    if type(validation_artifact) is not Phase1ValidationArtifact:
        raise D99RACGTMKError(
            "D99 margin audit requires an exact typed Phase1 validation artifact"
        )

    if (
        validation_artifact.validation_method_lock_sha256
        != bank.config.validation_method_lock_sha256
        or validation_artifact.external_receipt.phase1_episode_support_receipt_sha256
        != bank.metric.support_input_sha256
    ):
        raise D99RACGTMKError(
            "D99 margin audit bank is outside the sealed Phase1 episode lifecycle"
        )

    features, labels, positions, classes, k_shot, support_receipt = _support_closure(
        full_precision_support_features,
        support_labels,
        physical_ids,
        bank.classes,
    )
    if (
        classes != bank.classes
        or k_shot != bank.metric.k_shot
        or support_receipt != bank.metric.support_input_sha256
    ):
        raise D99RACGTMKError("D99 margin audit support receipt drift")
    validation = validation_artifact.validation_features_fp32
    fp_support = normalize_feature_blocks(features)
    fp_logits = _student_t_logits(
        fp_support,
        positions,
        bank.class_scales_fp16,
        bank.metric,
        bank.config,
        validation,
    ).astype(np.float64)
    int8_logits = score_metric_kernel_raw_logits(bank, validation).astype(
        np.float64
    )
    order = np.argsort(fp_logits, axis=1, kind="stable")
    winner, runner_up = order[:, -1], order[:, -2]
    rows = np.arange(len(fp_logits))
    teacher_margin = fp_logits[rows, winner] - fp_logits[rows, runner_up]
    quantized_margin = int8_logits[rows, winner] - int8_logits[rows, runner_up]
    flips = quantized_margin <= 0.0
    return {
        "schema": "cvs.phase1.d99.quantized_margin_audit.v1",
        "validation_artifact_sha256": validation_artifact.artifact_sha256,
        "external_validation_receipt_sha256": (
            validation_artifact.external_validation_receipt_sha256
        ),
        "validation_method_lock_sha256": (
            validation_artifact.validation_method_lock_sha256
        ),
        "authority_envelope_sha256": (
            validation_artifact.authority_envelope.envelope_sha256
        ),
        "authority_status": validation_artifact.authority_envelope.authority_status,
        "formal_phase1_eligible": (
            validation_artifact.authority_envelope.formal_phase1_eligible
        ),
        "phase1_episode_id": validation_artifact.external_receipt.phase1_episode_id,
        "phase1_episode_support_receipt_sha256": (
            validation_artifact.external_receipt.phase1_episode_support_receipt_sha256
        ),
        "feature_archive_sha256": (
            validation_artifact.external_receipt.feature_archive_sha256
        ),
        "validation_manifest_sha256": (
            validation_artifact.external_receipt.validation_manifest_sha256
        ),
        "producer_code_sha256": (
            validation_artifact.external_receipt.producer_code_sha256
        ),
        "phase1_checkpoint_sha256": (
            validation_artifact.external_receipt.phase1_checkpoint_sha256
        ),
        "validation_lifecycle": validation_artifact.lifecycle,
        "source_validation_only": validation_artifact.source_validation_only,
        "target_or_query_rows_present": (
            validation_artifact.target_or_query_rows_present
        ),
        "support_codes_qint8": _array_receipt(bank.codes_qint8),
        "support_scales_fp16": _array_receipt(bank.scales_fp16),
        "class_scales_fp16": _array_receipt(bank.class_scales_fp16),
        "metric_basis_fp32": _array_receipt(bank.metric.metric_basis_fp32),
        "precision_attenuation_fp32": _array_receipt(
            bank.metric.precision_attenuation_fp32
        ),
        "kernel_numeric_lock": {
            "student_nu": float(bank.config.student_nu),
            "kernel_effective_dim": int(bank.config.kernel_effective_dim),
            "kernel_volume_gamma": float(bank.config.kernel_volume_gamma),
            "z_weight": float(bank.config.z_weight),
            "fft_weight": float(bank.config.fft_weight),
            "rf_weight": float(bank.config.rf_weight),
        },
        "validation_row_count": int(len(fp_logits)),
        "top1_agreement": float(
            np.mean(np.argmax(fp_logits, axis=1) == np.argmax(int8_logits, axis=1))
        ),
        "logit_abs_error_mean": float(np.mean(np.abs(fp_logits - int8_logits))),
        "logit_abs_error_max": float(np.max(np.abs(fp_logits - int8_logits))),
        "teacher_margin_mean": float(np.mean(teacher_margin)),
        "quantized_teacher_margin_mean": float(np.mean(quantized_margin)),
        "margin_sign_flip_count": int(np.sum(flips)),
        "margin_sign_flip_rate": float(np.mean(flips)),
    }


def _require_formal_phase1_authority(
    validation_artifact: Phase1ValidationArtifact,
) -> None:
    if type(validation_artifact) is not Phase1ValidationArtifact:
        raise D99RACGTMKError(
            "D99 formal margin audit requires an exact typed Phase1 artifact"
        )
    envelope = validation_artifact.authority_envelope
    trusted = TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256
    if (
        trusted is None
        or envelope.envelope_sha256 != trusted
        or envelope.authority_status != "PROVISIONED"
        or envelope.formal_phase1_eligible is not True
        or envelope.loader_token is not _VALIDATION_LOADER_TOKEN
    ):
        raise D99RACGTMKError(
            "D99 formal margin audit blocked: external Phase1 authority is not provisioned"
        )


def diagnose_quantized_margin_development(
    bank: TypedINT8MetricKernelBank,
    full_precision_support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    validation_artifact: Phase1ValidationArtifact,
) -> dict[str, Any]:
    """Non-formal numerical diagnostic that can never grant Phase1 eligibility."""

    payload = _phase1_quantized_margin_payload(
        bank,
        full_precision_support_features,
        support_labels,
        physical_ids,
        validation_artifact,
    )
    diagnostic_sha = _canonical_sha256(payload)
    return {
        **payload,
        "development_diagnostic_sha256": diagnostic_sha,
        "development_lock_digest_matches": (
            diagnostic_sha == bank.config.quantization_margin_audit_sha256
        ),
        "authority_status": validation_artifact.authority_envelope.authority_status,
        "formal_phase1_eligible": False,
        "matches_phase1_lock": False,
        "formal_result_claimed": False,
    }


def precompute_phase1_quantized_margin_audit_sha256(
    bank: TypedINT8MetricKernelBank,
    full_precision_support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    validation_artifact: Phase1ValidationArtifact,
) -> str:
    """Compute the Phase1-only numerical audit identity before sealing the lock."""

    _require_formal_phase1_authority(validation_artifact)
    return _canonical_sha256(
        _phase1_quantized_margin_payload(
            bank,
            full_precision_support_features,
            support_labels,
            physical_ids,
            validation_artifact,
        )
    )


def audit_quantized_margin(
    bank: TypedINT8MetricKernelBank,
    full_precision_support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    validation_artifact: Phase1ValidationArtifact,
) -> dict[str, Any]:
    """Verify the actual Phase1-only INT8 margin audit against the sealed lock."""

    _require_formal_phase1_authority(validation_artifact)
    payload = _phase1_quantized_margin_payload(
        bank,
        full_precision_support_features,
        support_labels,
        physical_ids,
        validation_artifact,
    )
    actual = _canonical_sha256(payload)
    if actual != bank.config.quantization_margin_audit_sha256:
        raise D99RACGTMKError(
            "D99 actual margin audit SHA does not match the Phase1 lock"
        )
    return {**payload, "audit_sha256": actual, "matches_phase1_lock": True}


__all__ = [
    "ALLOWED_K",
    "AUTHORITY_ENVELOPE_SCHEMA",
    "BANK_SCHEMA",
    "DEPLOYMENT_STATUS",
    "D99RACGTMKError",
    "ExternalGroundAggregationReceipt",
    "ExternalPhase1ValidationReceipt",
    "GROUND_AGGREGATION_RECEIPT_SCHEMA",
    "GROUND_BUNDLE_SCHEMA",
    "GROUND_SCHEMA",
    "GroundGeometry",
    "LOCK_SCHEMA",
    "METRIC_SCHEMA",
    "Phase1D99Lock",
    "Phase1AuthorityEnvelope",
    "Phase1GroundAggregateBundle",
    "Phase1ValidationMethodLock",
    "Phase1ValidationArtifact",
    "REQUIRED_TYPED_D81_STATE_SCHEMA",
    "SCHEMA",
    "SupportMetricState",
    "TypedINT8MetricKernelBank",
    "TRUSTED_EXTERNAL_AUTHORITY_ENVELOPE_SHA256",
    "audit_quantized_margin",
    "build_ground_geometry",
    "build_typed_support_bank",
    "decode_support_bank",
    "diagnose_quantized_margin_development",
    "fit_support_metric",
    "normalize_feature_blocks",
    "load_phase1_validation_artifact",
    "load_phase1_authority_envelope",
    "precompute_phase1_quantized_margin_audit_sha256",
    "produce_typed_ground_aggregate_bundle",
    "score_metric_kernel_raw_logits",
    "VALIDATION_ARTIFACT_SCHEMA",
    "VALIDATION_ARCHIVE_SCHEMA",
    "VALIDATION_LIFECYCLE",
    "VALIDATION_MANIFEST_SCHEMA",
    "VALIDATION_METHOD_LOCK_SCHEMA",
    "VALIDATION_PRODUCER_ID",
    "VALIDATION_RECEIPT_SCHEMA",
]
