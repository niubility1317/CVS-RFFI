"""Externally receipt-bound, typed D81 Stage2-B/C lifecycle.

This module deliberately does not produce Phase1 or Phase2 data authority.  It
only consumes canonical authority bytes whose SHA256 is supplied out of band.
The fit surface materializes the exact D81/D62/D42 lifecycle: the twenty-step
metric and the registration-before head read old support only, then the metric
is frozen and a second head is fitted from all registered support.  Both heads
are stored in the same immutable INT8 wire artifact.

Deployment remains pending until the project capsule builder emits the two
external authority artifacts used by the loaders below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d81_phase1_episode_scorer import (
    D81Phase1EpisodeScorer,
    FEATURE_DIM,
    _FIT_LOCK,
    _dependency_hashes as _d81_dependency_hashes,
    raw_concat_to_d81_registered_feature,
)


SCHEMA = "cvs.phase2.d81.typed_target_lifecycle.v2"
CONFIG_SCHEMA = "cvs.phase2.d81.typed_target_config.v2"
RESOURCE_SCHEMA = "cvs.phase2.d81.typed_target_resource.v2"
PHASE1_AUTHORITY_SCHEMA = "cvs.phase1.d81.external_authority.v1"
ROW_AUTHORITY_SCHEMA = "cvs.phase2.d81.external_row_authority.v1"
DEPLOYMENT_STATUS = "LOCAL_CORE_PENDING_EXTERNAL_CAPSULE_PRODUCER_AND_REVIEW"
PROTOCOL_SCHEMA = "p2_min_v1"
ALLOWED_K_SHOT = (1, 5, 10)
MAGIC = b"D81TYP2\x00"
ARRAY_ORDER = (
    "log_diag_fp32",
    "before_coef1_qint8",
    "before_coef2_qint8",
    "before_scale1_fp16",
    "before_scale2_fp16",
    "before_intercept_fp16",
    "final_coef1_qint8",
    "final_coef2_qint8",
    "final_scale1_fp16",
    "final_scale2_fp16",
    "final_intercept_fp16",
)
_AUTHORITY_TOKEN = object()
_STATE_TOKEN = object()


class D81TypedTargetStateError(ValueError):
    """Raised when authority, lifecycle, wire, or resource closure drifts."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise D81TypedTargetStateError(f"{name} must be lowercase SHA256")
    return normalized


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise D81TypedTargetStateError(f"noncanonical value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _decode_canonical_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D81TypedTargetStateError(f"{name} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise D81TypedTargetStateError(f"{name} is not canonical JSON")
    return value


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(source.tobytes(), dtype=source.dtype).reshape(source.shape)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, dict):
        return MappingProxyType({key: _freeze(item) for key, item in safe.items()})
    if isinstance(safe, list):
        return tuple(_freeze(item) for item in safe)
    return safe


def _current_dependency_hashes() -> tuple[tuple[str, str], ...]:
    values = dict(_d81_dependency_hashes())
    values["typed_target_state"] = _sha256_file(Path(__file__).resolve())
    return tuple(sorted((str(name), str(value)) for name, value in values.items()))


@dataclass(frozen=True, slots=True)
class D81Phase1Authority:
    artifact_sha256: str
    method_lock_sha256: str
    phase1_bundle_receipt_sha256: str
    d81_scorer_receipt_sha256: str
    phase1_checkpoint_sha256: str
    ground_manifest_sha256: str
    ground_component_npz_sha256: str
    dependency_closure_sha256: str
    metric_seed: int
    ground_component_npz_serialized_bytes: int
    ground_manifest_serialized_bytes: int
    ground_bundle_logical_state_bytes: int
    ground_retained_rank: int
    ground_covariance_statistics_mac_upper_bound: int
    ground_transient_dequantized_bytes: int
    schema: str = PHASE1_AUTHORITY_SCHEMA
    bundle_status: str = "IMMUTABLE_SEALED"
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _AUTHORITY_TOKEN:
            raise D81TypedTargetStateError("Phase1 authority must come from external loader")
        if self.schema != PHASE1_AUTHORITY_SCHEMA or self.bundle_status != "IMMUTABLE_SEALED":
            raise D81TypedTargetStateError("Phase1 authority schema/status drift")
        for item in fields(self):
            if item.name.endswith("sha256"):
                object.__setattr__(self, item.name, _require_sha256(getattr(self, item.name), item.name))
        if (
            int(self.metric_seed) < 0 or int(self.metric_seed) > 0x7FFFFFFF
            or int(self.ground_component_npz_serialized_bytes) <= 0
            or int(self.ground_manifest_serialized_bytes) <= 0
            or int(self.ground_bundle_logical_state_bytes) <= 0
            or int(self.ground_retained_rank) < 1
            or int(self.ground_covariance_statistics_mac_upper_bound) < 0
            or int(self.ground_transient_dequantized_bytes) < 0
        ):
            raise D81TypedTargetStateError("Phase1 authority resource/seed drift")


def load_d81_phase1_authority(
    artifact: bytes, *, expected_artifact_sha256: str
) -> D81Phase1Authority:
    """Load authority bytes against a separately supplied immutable identity."""

    raw = bytes(artifact)
    expected = _require_sha256(expected_artifact_sha256, "expected Phase1 authority artifact")
    if _sha256_bytes(raw) != expected:
        raise D81TypedTargetStateError("Phase1 authority artifact SHA mismatch")
    value = _decode_canonical_object(raw, "Phase1 authority")
    allowed = {item.name for item in fields(D81Phase1Authority) if not item.name.startswith("_")}
    if set(value) != allowed - {"artifact_sha256"}:
        raise D81TypedTargetStateError("Phase1 authority field closure drift")
    return D81Phase1Authority(artifact_sha256=expected, _token=_AUTHORITY_TOKEN, **value)


@dataclass(frozen=True, slots=True)
class D81TargetRowAuthority:
    artifact_sha256: str
    capsule_id: str
    split_id: str
    opaque_row_receipt_sha256: str
    method_lock_sha256: str
    phase1_bundle_receipt_sha256: str
    phase1_authority_artifact_sha256: str
    k_shot: int
    old_registry: tuple[str, ...]
    final_registry: tuple[str, ...]
    old_support: Mapping[str, Any]
    all_registered_support: Mapping[str, Any]
    schema: str = ROW_AUTHORITY_SCHEMA
    protocol_schema: str = PROTOCOL_SCHEMA
    phase2_data_status: str = "VALIDATED_ONCE"
    single_leo_observation: bool = True
    clean_source_runtime_access: bool = False
    query_fit_access: bool = False
    query_decision_policy: str = "per_sample_all_registered_classes"
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _AUTHORITY_TOKEN:
            raise D81TypedTargetStateError("row authority must come from external loader")
        if (
            self.schema != ROW_AUTHORITY_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or self.phase2_data_status != "VALIDATED_ONCE"
            or self.single_leo_observation is not True
            or self.clean_source_runtime_access is not False
            or self.query_fit_access is not False
            or self.query_decision_policy != "per_sample_all_registered_classes"
            or int(self.k_shot) not in ALLOWED_K_SHOT
        ):
            raise D81TypedTargetStateError("row authority protocol closure drift")
        for name in (
            "artifact_sha256", "capsule_id", "split_id", "opaque_row_receipt_sha256",
            "method_lock_sha256", "phase1_bundle_receipt_sha256",
            "phase1_authority_artifact_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        old = tuple(str(value) for value in self.old_registry)
        final = tuple(str(value) for value in self.final_registry)
        if (
            len(old) < 2 or len(final) <= len(old) or final[: len(old)] != old
            or len(set(final)) != len(final) or any(not value for value in final)
        ):
            raise D81TypedTargetStateError("old registry must be exact final-registry prefix")
        object.__setattr__(self, "old_registry", old)
        object.__setattr__(self, "final_registry", final)
        for name, support, registry in (
            ("old_support", self.old_support, old),
            ("all_registered_support", self.all_registered_support, final),
        ):
            if not isinstance(support, Mapping):
                raise D81TypedTargetStateError(f"{name} receipt missing")
            expected_fields = {
                "scope", "row_count", "k_shot", "ordered_registry",
                "ordered_physical_ids_root_sha256", "ordered_feature_root_sha256",
                "ordered_row_root_sha256", "support_receipt_sha256",
            }
            plain = _json_safe(support)
            if set(plain) != expected_fields or plain["scope"] != name:
                raise D81TypedTargetStateError(f"{name} receipt field closure drift")
            if (
                int(plain["row_count"]) != len(registry) * int(self.k_shot)
                or int(plain["k_shot"]) != int(self.k_shot)
                or tuple(plain["ordered_registry"]) != registry
            ):
                raise D81TypedTargetStateError(f"{name} registry/K closure drift")
            for key in (
                "ordered_physical_ids_root_sha256", "ordered_feature_root_sha256",
                "ordered_row_root_sha256", "support_receipt_sha256",
            ):
                plain[key] = _require_sha256(plain[key], f"{name}.{key}")
            expected_receipt = _canonical_sha256({key: plain[key] for key in sorted(expected_fields - {"support_receipt_sha256"})})
            if plain["support_receipt_sha256"] != expected_receipt:
                raise D81TypedTargetStateError(f"{name} self-consistency drift")
            object.__setattr__(self, name, _freeze(plain))


def load_d81_target_row_authority(
    artifact: bytes,
    *,
    expected_artifact_sha256: str,
    phase1_authority: D81Phase1Authority,
) -> D81TargetRowAuthority:
    if type(phase1_authority) is not D81Phase1Authority:
        raise D81TypedTargetStateError("exact externally loaded Phase1 authority required")
    raw = bytes(artifact)
    expected = _require_sha256(expected_artifact_sha256, "expected row authority artifact")
    if _sha256_bytes(raw) != expected:
        raise D81TypedTargetStateError("row authority artifact SHA mismatch")
    value = _decode_canonical_object(raw, "row authority")
    allowed = {item.name for item in fields(D81TargetRowAuthority) if not item.name.startswith("_")}
    if set(value) != allowed - {"artifact_sha256"}:
        raise D81TypedTargetStateError("row authority field closure drift")
    result = D81TargetRowAuthority(artifact_sha256=expected, _token=_AUTHORITY_TOKEN, **value)
    if (
        result.phase1_authority_artifact_sha256 != phase1_authority.artifact_sha256
        or result.phase1_bundle_receipt_sha256 != phase1_authority.phase1_bundle_receipt_sha256
        or result.method_lock_sha256 != phase1_authority.method_lock_sha256
    ):
        raise D81TypedTargetStateError("row/Phase1 external authority binding drift")
    return result


@dataclass(frozen=True, slots=True)
class D81TypedTargetConfig:
    phase1_authority_artifact_sha256: str
    d81_scorer_receipt_sha256: str
    phase1_checkpoint_sha256: str
    ground_manifest_sha256: str
    ground_component_npz_sha256: str
    dependency_code_sha256: tuple[tuple[str, str], ...]
    dependency_closure_sha256: str
    metric_seed: int
    ground_component_npz_serialized_bytes: int
    ground_manifest_serialized_bytes: int
    ground_bundle_logical_state_bytes: int
    ground_retained_rank: int
    ground_covariance_statistics_mac_upper_bound: int
    ground_transient_dequantized_bytes: int
    schema: str = CONFIG_SCHEMA
    protocol_schema: str = PROTOCOL_SCHEMA
    feature_geometry: str = "raw_concat288_to_D81_registered_feature"
    metric_optimizer_steps: int = 20

    def __post_init__(self) -> None:
        if (
            self.schema != CONFIG_SCHEMA or self.protocol_schema != PROTOCOL_SCHEMA
            or self.feature_geometry != "raw_concat288_to_D81_registered_feature"
            or int(self.metric_optimizer_steps) != 20
            or int(self.metric_seed) < 0 or int(self.metric_seed) > 0x7FFFFFFF
            or int(self.ground_component_npz_serialized_bytes) <= 0
            or int(self.ground_manifest_serialized_bytes) <= 0
            or int(self.ground_bundle_logical_state_bytes) <= 0
            or int(self.ground_retained_rank) < 1
            or int(self.ground_covariance_statistics_mac_upper_bound) < 0
            or int(self.ground_transient_dequantized_bytes) < 0
        ):
            raise D81TypedTargetStateError("typed config lock drift")
        for name in (
            "phase1_authority_artifact_sha256", "d81_scorer_receipt_sha256",
            "phase1_checkpoint_sha256", "ground_manifest_sha256",
            "ground_component_npz_sha256", "dependency_closure_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        dependencies = tuple((str(name), _require_sha256(sha, name)) for name, sha in self.dependency_code_sha256)
        if dependencies != tuple(sorted(dependencies)) or len({name for name, _ in dependencies}) != len(dependencies):
            raise D81TypedTargetStateError("dependency registry drift")
        if self.dependency_closure_sha256 != _canonical_sha256(dict(dependencies)):
            raise D81TypedTargetStateError("dependency closure drift")
        object.__setattr__(self, "dependency_code_sha256", dependencies)

    @classmethod
    def from_scorer(
        cls, scorer: D81Phase1EpisodeScorer, phase1_authority: D81Phase1Authority
    ) -> "D81TypedTargetConfig":
        if type(scorer) is not D81Phase1EpisodeScorer or type(phase1_authority) is not D81Phase1Authority:
            raise D81TypedTargetStateError("exact scorer and external Phase1 authority required")
        if (
            scorer.scorer_id != phase1_authority.d81_scorer_receipt_sha256
            or scorer.phase1_checkpoint_sha256 != phase1_authority.phase1_checkpoint_sha256
            or scorer.ground_manifest_sha256 != phase1_authority.ground_manifest_sha256
            or scorer.ground_component_npz_sha256 != phase1_authority.ground_component_npz_sha256
            or int(scorer.metric_seed) != int(phase1_authority.metric_seed)
        ):
            raise D81TypedTargetStateError("scorer/Phase1 authority binding drift")
        dependencies = _current_dependency_hashes()
        if _canonical_sha256(dict(dependencies)) != phase1_authority.dependency_closure_sha256:
            raise D81TypedTargetStateError("Phase1 authority dependency closure drift")
        component_path = Path(str(scorer.ground_audit.get("component_path", ""))).resolve()
        manifest_path = component_path.parent / "manifest.json"
        if not component_path.is_file() or not manifest_path.is_file():
            raise D81TypedTargetStateError("sealed ground bundle files missing")
        if _sha256_file(component_path) != scorer.ground_component_npz_sha256 or _sha256_file(manifest_path) != scorer.ground_manifest_sha256:
            raise D81TypedTargetStateError("sealed ground bundle file drift")
        if (
            int(component_path.stat().st_size) != phase1_authority.ground_component_npz_serialized_bytes
            or int(manifest_path.stat().st_size) != phase1_authority.ground_manifest_serialized_bytes
            or int(scorer.ground_audit.get("ground_int8_component_logical_state_bytes", -1))
            != phase1_authority.ground_bundle_logical_state_bytes
            or int(scorer.ground_audit.get("d81_retained_rank", -1))
            != phase1_authority.ground_retained_rank
            or int(scorer.ground_audit.get("ground_covariance_statistics_mac_upper_bound", -1))
            != phase1_authority.ground_covariance_statistics_mac_upper_bound
            or int(scorer.ground_audit.get("transient_dequantized_ground_bytes", -1))
            != phase1_authority.ground_transient_dequantized_bytes
        ):
            raise D81TypedTargetStateError("scorer/Phase1 resource authority drift")
        return cls(
            phase1_authority_artifact_sha256=phase1_authority.artifact_sha256,
            d81_scorer_receipt_sha256=scorer.scorer_id,
            phase1_checkpoint_sha256=scorer.phase1_checkpoint_sha256,
            ground_manifest_sha256=scorer.ground_manifest_sha256,
            ground_component_npz_sha256=scorer.ground_component_npz_sha256,
            dependency_code_sha256=dependencies,
            dependency_closure_sha256=_canonical_sha256(dict(dependencies)),
            metric_seed=int(scorer.metric_seed),
            ground_component_npz_serialized_bytes=int(component_path.stat().st_size),
            ground_manifest_serialized_bytes=int(manifest_path.stat().st_size),
            ground_bundle_logical_state_bytes=int(
                scorer.ground_audit["ground_int8_component_logical_state_bytes"]
            ),
            ground_retained_rank=int(scorer.ground_audit["d81_retained_rank"]),
            ground_covariance_statistics_mac_upper_bound=int(
                scorer.ground_audit["ground_covariance_statistics_mac_upper_bound"]
            ),
            ground_transient_dequantized_bytes=int(
                scorer.ground_audit["transient_dequantized_ground_bytes"]
            ),
        )

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256(self)

    def verify_current(self, phase1_authority: D81Phase1Authority) -> None:
        if type(phase1_authority) is not D81Phase1Authority:
            raise D81TypedTargetStateError("external Phase1 authority required")
        current = _current_dependency_hashes()
        if current != self.dependency_code_sha256 or _canonical_sha256(dict(current)) != self.dependency_closure_sha256:
            raise D81TypedTargetStateError("current dependency code drift")
        if (
            self.phase1_authority_artifact_sha256 != phase1_authority.artifact_sha256
            or self.d81_scorer_receipt_sha256 != phase1_authority.d81_scorer_receipt_sha256
            or self.phase1_checkpoint_sha256 != phase1_authority.phase1_checkpoint_sha256
            or self.ground_manifest_sha256 != phase1_authority.ground_manifest_sha256
            or self.ground_component_npz_sha256 != phase1_authority.ground_component_npz_sha256
            or int(self.metric_seed) != int(phase1_authority.metric_seed)
            or int(self.ground_component_npz_serialized_bytes)
            != int(phase1_authority.ground_component_npz_serialized_bytes)
            or int(self.ground_manifest_serialized_bytes)
            != int(phase1_authority.ground_manifest_serialized_bytes)
            or int(self.ground_bundle_logical_state_bytes)
            != int(phase1_authority.ground_bundle_logical_state_bytes)
            or int(self.ground_retained_rank) != int(phase1_authority.ground_retained_rank)
            or int(self.ground_covariance_statistics_mac_upper_bound)
            != int(phase1_authority.ground_covariance_statistics_mac_upper_bound)
            or int(self.ground_transient_dequantized_bytes)
            != int(phase1_authority.ground_transient_dequantized_bytes)
        ):
            raise D81TypedTargetStateError("config/Phase1 authority drift")


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype), "shape": list(array.shape),
        "sha256": _sha256_bytes(array.view(np.uint8).tobytes()), "nbytes": int(array.nbytes),
    }


def _support_closure(
    features: np.ndarray, labels: Sequence[str], physical_ids: Sequence[str], registry: Sequence[str], scope: str
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, int, dict[str, Any]]:
    rows = np.asarray(features)
    text_labels = tuple(str(value) for value in labels)
    ids = tuple(str(value) for value in physical_ids)
    classes = tuple(str(value) for value in registry)
    if (
        rows.dtype != np.float32 or rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or len(rows) == 0
        or not np.isfinite(rows).all() or len(text_labels) != len(rows) or len(ids) != len(rows)
        or len(classes) < 2 or len(set(classes)) != len(classes) or any(not value for value in classes)
        or len(set(ids)) != len(ids) or any(not value for value in ids) or set(text_labels) != set(classes)
    ):
        raise D81TypedTargetStateError(f"{scope} support closure drift")
    counts = tuple(sum(label == class_id for label in text_labels) for class_id in classes)
    if len(set(counts)) != 1 or counts[0] not in ALLOWED_K_SHOT:
        raise D81TypedTargetStateError(f"{scope} must be balanced K1/K5/K10")
    rank = {value: index for index, value in enumerate(classes)}
    # The external capsule order is authoritative.  No class, physical-ID, or
    # lexical canonicalization is allowed after the row receipt is sealed.
    payload_rows = np.ascontiguousarray(rows, dtype=np.float32)
    payload_labels = text_labels
    payload_ids = ids
    targets = np.asarray([rank[value] for value in payload_labels], dtype=np.int64)
    id_hashes = [_sha256_bytes(value.encode("utf-8")) for value in payload_ids]
    feature_hashes = [_sha256_bytes(payload_rows[index].view(np.uint8).tobytes()) for index in range(len(payload_rows))]
    row_hashes = [
        _canonical_sha256(
            {
                "row_index": index,
                "class": payload_labels[index],
                "physical_id_sha256": id_hashes[index],
                "feature_sha256": feature_hashes[index],
            }
        )
        for index in range(len(payload_rows))
    ]
    receipt = {
        "scope": scope, "row_count": len(payload_rows), "k_shot": counts[0],
        "ordered_registry": list(classes),
        "ordered_physical_ids_root_sha256": _canonical_sha256(id_hashes),
        "ordered_feature_root_sha256": _canonical_sha256(feature_hashes),
        "ordered_row_root_sha256": _canonical_sha256(row_hashes),
    }
    receipt["support_receipt_sha256"] = _canonical_sha256(receipt)
    return payload_rows, payload_labels, payload_ids, classes, targets, counts[0], receipt


def _validate_support_authority(receipt: Mapping[str, Any], expected: Mapping[str, Any], scope: str) -> None:
    if _json_safe(receipt) != _json_safe(expected):
        raise D81TypedTargetStateError(f"{scope} does not match external row authority")


def _sanitize_d81_audit(audit: Mapping[str, Any], class_count: int, k_shot: int) -> dict[str, Any]:
    transform = audit.get("d81_transform_audit")
    if (
        audit.get("d81_probe_arm") != "ground_nuisance_cauchy_center"
        or audit.get("d81_structure") != "d62_with_ground_spectrum_support_only_class_center_translation"
        or audit.get("d81_ground_int8_component_used") is not True
        or audit.get("d81_ground_component_update_access") is not False
        or audit.get("d81_old_new_role_specific_branch") is not False
        or audit.get("d81_class_id_specific_formula") is not False
        or audit.get("d81_uses_outer_held_or_query") is not False
        or int(audit.get("d81_query_rows_used", -1)) != 0
        or not isinstance(transform, Mapping)
        or int(transform.get("class_count", -1)) != class_count
        or int(transform.get("k_shot", -1)) != k_shot
        or transform.get("uses_outer_held_or_query") is not False
        or int(transform.get("query_rows_used", -1)) != 0
    ):
        raise D81TypedTargetStateError("exact D81 support-only head audit drift")
    return {
        "covariance_policy": str(audit.get("covariance_policy")),
        "d81_probe_arm": str(audit.get("d81_probe_arm")),
        "d81_structure": str(audit.get("d81_structure")),
        "class_count": int(class_count), "k_shot": int(k_shot),
        "retained_rank": int(transform.get("retained_rank", -1)),
        "translation_scope": str(transform.get("translation_scope")),
        "support_rows": int(transform.get("support_rows", -1)),
        "query_rows_used": 0,
        "center_shift_l2_max": float(transform.get("center_shift_l2_max", 0.0)),
        "within_class_residual_max_abs_error": float(transform.get("within_class_residual_max_abs_error", 0.0)),
        "fft96_rf32_max_abs_error": float(transform.get("fft96_rf32_max_abs_error", 0.0)),
    }


def _quantized_head(coefficient: np.ndarray, intercept: np.ndarray, prefix: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    code1, code2, scale1, scale2, decoded = d42._quantize_coefficients(coefficient)
    intercept16 = np.asarray(intercept, dtype=np.float16)
    if not np.isfinite(intercept16).all():
        raise D81TypedTargetStateError("INT8 head intercept FP16 overflow")
    arrays = {
        f"{prefix}_coef1_qint8": np.asarray(code1, dtype=np.int8),
        f"{prefix}_coef2_qint8": np.asarray(code2, dtype=np.int8),
        f"{prefix}_scale1_fp16": np.asarray(scale1, dtype=np.float16),
        f"{prefix}_scale2_fp16": np.asarray(scale2, dtype=np.float16),
        f"{prefix}_intercept_fp16": intercept16,
    }
    error = np.abs(decoded - np.asarray(coefficient, dtype=np.float32))
    return arrays, {"mean": float(np.mean(error)), "max": float(np.max(error))}


def _core_payload(
    *, old_classes: tuple[str, ...], classes: tuple[str, ...], k_shot: int,
    before_covariance_policy: str, final_covariance_policy: str,
    arrays: Mapping[str, np.ndarray], config: D81TypedTargetConfig,
    phase1_authority_sha256: str, row_authority_sha256: str,
    old_support_receipt_sha256: str, all_support_receipt_sha256: str,
    row_authority_binding: Mapping[str, Any],
    phase1_lock_binding: Mapping[str, Any],
    formal_query_authorized: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "deployment_status": DEPLOYMENT_STATUS,
        "old_classes": list(old_classes), "classes": list(classes), "k_shot": int(k_shot),
        "before_covariance_policy": before_covariance_policy,
        "final_covariance_policy": final_covariance_policy,
        "arrays": {name: _array_receipt(arrays[name]) for name in ARRAY_ORDER},
        "config": _json_safe(config),
        "phase1_authority_artifact_sha256": phase1_authority_sha256,
        "row_authority_artifact_sha256": row_authority_sha256,
        "old_support_receipt_sha256": old_support_receipt_sha256,
        "all_registered_support_receipt_sha256": all_support_receipt_sha256,
        "row_authority_binding": _json_safe(row_authority_binding),
        "phase1_lock_binding": _json_safe(phase1_lock_binding),
        "formal_query_authorized": bool(formal_query_authorized),
    }


def _row_authority_binding(authority: D81TargetRowAuthority) -> dict[str, Any]:
    return {
        "schema": authority.schema,
        "protocol_schema": authority.protocol_schema,
        "phase2_data_status": authority.phase2_data_status,
        "single_leo_observation": authority.single_leo_observation,
        "clean_source_runtime_access": authority.clean_source_runtime_access,
        "query_fit_access": authority.query_fit_access,
        "query_decision_policy": authority.query_decision_policy,
        "capsule_id": authority.capsule_id,
        "split_id": authority.split_id,
        "opaque_row_receipt_sha256": authority.opaque_row_receipt_sha256,
        "method_lock_sha256": authority.method_lock_sha256,
        "phase1_bundle_receipt_sha256": authority.phase1_bundle_receipt_sha256,
        "phase1_authority_artifact_sha256": authority.phase1_authority_artifact_sha256,
        "k_shot": int(authority.k_shot),
        "old_registry": list(authority.old_registry),
        "final_registry": list(authority.final_registry),
        "old_support": _json_safe(authority.old_support),
        "all_registered_support": _json_safe(authority.all_registered_support),
    }


def _phase1_lock_binding(authority: D81Phase1Authority) -> dict[str, Any]:
    return {
        item.name: _json_safe(getattr(authority, item.name))
        for item in fields(D81Phase1Authority)
        if item.name not in {"artifact_sha256", "_token"}
    }


def _wire_from_parts(core: Mapping[str, Any], fit_audit: Mapping[str, Any], resource: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> tuple[bytes, str]:
    audit = {"fit_audit": _json_safe(fit_audit), "resource_audit": _json_safe(resource)}
    receipt = _canonical_sha256({"core": core, "audit": audit})
    header = _canonical_bytes({**_json_safe(core), "state_receipt_sha256": receipt})
    audit_raw = _canonical_bytes(audit)
    wire = b"".join(
        [MAGIC, struct.pack("<Q", len(header)), header, struct.pack("<Q", len(audit_raw)), audit_raw]
        + [np.ascontiguousarray(arrays[name]).view(np.uint8).tobytes() for name in ARRAY_ORDER]
    )
    return wire, receipt


def _resource_with_exact_wire_sizes(
    base: Mapping[str, Any], core: Mapping[str, Any], fit_audit: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], str]:
    resource = dict(base)
    resource.update({"deploy_state_serialized_bytes": 0, "audit_serialized_bytes": 0, "total_wire_serialized_bytes": 0})
    for _ in range(12):
        wire, receipt = _wire_from_parts(core, fit_audit, resource, arrays)
        header_len = struct.unpack("<Q", wire[len(MAGIC): len(MAGIC) + 8])[0]
        audit_offset = len(MAGIC) + 8 + header_len
        audit_len = struct.unpack("<Q", wire[audit_offset: audit_offset + 8])[0]
        numeric_bytes = sum(int(arrays[name].nbytes) for name in ARRAY_ORDER)
        updated = {
            **resource,
            "deploy_state_serialized_bytes": int(len(MAGIC) + 8 + header_len + numeric_bytes),
            "audit_serialized_bytes": int(8 + audit_len),
            "total_wire_serialized_bytes": int(len(wire)),
        }
        if "total_deployment_serialized_bytes_including_ground" in resource:
            updated["total_deployment_serialized_bytes_including_ground"] = int(
                len(wire) + int(resource["ground_bundle_serialized_state_bytes"])
            )
        if updated == resource:
            return resource, receipt
        resource = updated
    raise D81TypedTargetStateError("wire-size fixed point did not converge")


def _head_component_inventory(
    scope: str, class_count: int, k_shot: int
) -> list[dict[str, Any]]:
    classes, shots = int(class_count), int(k_shot)
    rows = classes * shots
    values = [
        {
            "fit_group": f"{scope}_d46_main_full_block",
            "fit_count": 2,
            "row_count_per_fit": rows,
            "class_count": classes,
        }
    ]
    if shots > 1:
        values.append(
            {
                "fit_group": f"{scope}_d46_inner_loo_full_block",
                "fit_count": 2 * shots,
                "row_count_per_fit": classes * (shots - 1),
                "class_count": classes,
            }
        )
    if shots > 2:
        values.extend(
            [
                {
                    "fit_group": f"{scope}_d62_outer_fisher_full_block",
                    "fit_count": 2,
                    "row_count_per_fit": rows,
                    "class_count": classes,
                },
                {
                    "fit_group": f"{scope}_d62_inner_fisher_full_block",
                    "fit_count": 2 * shots,
                    "row_count_per_fit": classes * (shots - 1),
                    "class_count": classes,
                },
            ]
        )
    for item in values:
        item["macs_per_fit"] = int(
            d42._lda_fit_macs(item["row_count_per_fit"], item["class_count"])
        )
        item["group_macs"] = int(item["fit_count"] * item["macs_per_fit"])
    return values


def _exact_resource_inventory(
    *,
    old_class_count: int,
    final_class_count: int,
    k_shot: int,
    config: D81TypedTargetConfig,
    numeric_bytes: int,
) -> dict[str, Any]:
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    old_count, final_count, shots = (
        int(old_class_count), int(final_class_count), int(k_shot)
    )
    old_rows, final_rows = old_count * shots, final_count * shots
    inventory = _head_component_inventory("before", old_count, shots) + _head_component_inventory(
        "final", final_count, shots
    )
    component_fit_count = sum(int(item["fit_count"]) for item in inventory)
    lda_macs = sum(int(item["group_macs"]) for item in inventory)
    d62_fit_count = (
        0 if shots <= 2 else 2 * (shots + 1) * 2
    )
    fisher_macs = int(d62_fit_count * 8 * FEATURE_DIM**3)
    reliability_macs = int(
        2
        * shots
        * (1 if shots <= 1 else shots + 1)
        * FEATURE_DIM
        * (old_count**2 + final_count**2)
    )
    fusion_macs = int(2 * (FEATURE_DIM + 1) * (old_count + final_count))
    gate_scalar_macs = int(shots * (old_count**2 + final_count**2) * 8)
    metric_macs = int(3 * FEATURE_DIM * 20 * old_rows * old_count)
    raw_geometry_macs = int((old_rows + final_rows) * 1856)
    metric_transform_macs = int((old_rows + final_rows) * 4 * FEATURE_DIM)
    translation_macs = int(
        probe._d62_translation_chain_macs(
            old_count, shots, int(config.ground_retained_rank)
        )
        + probe._d62_translation_chain_macs(
            final_count, shots, int(config.ground_retained_rank)
        )
    )
    quantization_macs = int(10 * FEATURE_DIM * (old_count + final_count))
    fit_total = int(
        lda_macs
        + fisher_macs
        + reliability_macs
        + fusion_macs
        + gate_scalar_macs
        + metric_macs
        + raw_geometry_macs
        + metric_transform_macs
        + translation_macs
        + quantization_macs
        + int(config.ground_covariance_statistics_mac_upper_bound)
    )
    max_rows = max(old_rows, final_rows)
    max_classes = max(old_count, final_count)
    metric_parameter_count = FEATURE_DIM * (1 + old_count)
    metric_optimizer_peak = int(
        metric_parameter_count * 16
        + old_rows * FEATURE_DIM * 16
        + old_count * FEATURE_DIM * 8
    )
    # Sequential component fits can reuse buffers, but a conservative proof
    # includes dense LDA/Fisher workspaces, the largest row/class workspace, and
    # retained audit evidence for every actually executed component fit.
    dense_component_workspace = int(
        32 * FEATURE_DIM**2 * 8
        + max_rows * FEATURE_DIM * 8 * 12
        + max_classes * FEATURE_DIM * 8 * 12
    )
    retained_component_audit = int(
        component_fit_count
        * (FEATURE_DIM**2 * 8 + max_rows * FEATURE_DIM * 8)
    )
    caller_and_canonical_support = int(
        2 * (old_rows + final_rows) * FEATURE_DIM * 4
    )
    registered_and_transformed_support = int(
        2 * (old_rows + final_rows) * FEATURE_DIM * 4
    )
    two_fp32_heads = int((old_count + final_count) * (FEATURE_DIM + 1) * 4)
    ground_serialized = int(
        config.ground_component_npz_serialized_bytes
        + config.ground_manifest_serialized_bytes
    )
    peak = int(
        ground_serialized
        + int(config.ground_bundle_logical_state_bytes)
        + int(config.ground_transient_dequantized_bytes)
        + caller_and_canonical_support
        + registered_and_transformed_support
        + two_fp32_heads
        + int(numeric_bytes)
        + metric_optimizer_peak
        + dense_component_workspace
        + retained_component_audit
    )
    return {
        "component_fit_inventory": inventory,
        "component_fit_count": int(component_fit_count),
        "d46_base_component_fit_count": int(
            4 if shots <= 1 else 4 + 4 * shots
        ),
        "d62_additional_component_fit_count": int(d62_fit_count),
        "lda_component_fit_macs": int(lda_macs),
        "fisher_dense_algebra_macs": int(fisher_macs),
        "reliability_scoring_macs": int(reliability_macs),
        "classwise_fusion_macs": int(fusion_macs),
        "pareto_gate_scalar_macs": int(gate_scalar_macs),
        "metric_fit_macs": int(metric_macs),
        "raw288_geometry_fit_macs": int(raw_geometry_macs),
        "metric_transform_fit_macs": int(metric_transform_macs),
        "support_center_translation_macs": int(translation_macs),
        "coefficient_quantization_macs": int(quantization_macs),
        "ground_statistics_macs": int(
            config.ground_covariance_statistics_mac_upper_bound
        ),
        "fit_mac_upper_bound": int(fit_total),
        "metric_parameter_count": int(metric_parameter_count),
        "dense_component_workspace_bytes": int(dense_component_workspace),
        "retained_component_audit_bytes_upper_bound": int(retained_component_audit),
        "complete_fit_lifecycle_peak_bytes_upper_bound": int(peak),
    }


@dataclass(frozen=True, slots=True)
class D81TypedTargetState:
    old_classes: tuple[str, ...]
    classes: tuple[str, ...]
    k_shot: int
    log_diag_fp32: np.ndarray
    before_coef1_qint8: np.ndarray
    before_coef2_qint8: np.ndarray
    before_scale1_fp16: np.ndarray
    before_scale2_fp16: np.ndarray
    before_intercept_fp16: np.ndarray
    final_coef1_qint8: np.ndarray
    final_coef2_qint8: np.ndarray
    final_scale1_fp16: np.ndarray
    final_scale2_fp16: np.ndarray
    final_intercept_fp16: np.ndarray
    before_covariance_policy: str
    final_covariance_policy: str
    config: D81TypedTargetConfig
    phase1_authority_artifact_sha256: str
    row_authority_artifact_sha256: str
    old_support_receipt_sha256: str
    all_registered_support_receipt_sha256: str
    row_authority_binding: Mapping[str, Any]
    phase1_lock_binding: Mapping[str, Any]
    formal_query_authorized: bool
    fit_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    state_receipt_sha256: str
    schema: str = SCHEMA
    deployment_status: str = DEPLOYMENT_STATUS
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _STATE_TOKEN:
            raise D81TypedTargetStateError("typed state must come from fit or external wire loader")
        old, final = tuple(self.old_classes), tuple(self.classes)
        if (
            self.schema != SCHEMA or self.deployment_status != DEPLOYMENT_STATUS
            or final[: len(old)] != old or len(old) < 2 or len(final) <= len(old)
            or len(set(final)) != len(final) or int(self.k_shot) not in ALLOWED_K_SHOT
            or self.formal_query_authorized is not False
        ):
            raise D81TypedTargetStateError("typed state registry/lifecycle drift")
        expected = {
            "log_diag_fp32": (np.float32, (FEATURE_DIM,)),
            "before_coef1_qint8": (np.int8, (len(old), FEATURE_DIM)),
            "before_coef2_qint8": (np.int8, (len(old), FEATURE_DIM)),
            "before_scale1_fp16": (np.float16, (len(old), len(d42.BLOCK_SLICES))),
            "before_scale2_fp16": (np.float16, (len(old), len(d42.BLOCK_SLICES))),
            "before_intercept_fp16": (np.float16, (len(old),)),
            "final_coef1_qint8": (np.int8, (len(final), FEATURE_DIM)),
            "final_coef2_qint8": (np.int8, (len(final), FEATURE_DIM)),
            "final_scale1_fp16": (np.float16, (len(final), len(d42.BLOCK_SLICES))),
            "final_scale2_fp16": (np.float16, (len(final), len(d42.BLOCK_SLICES))),
            "final_intercept_fp16": (np.float16, (len(final),)),
        }
        for name, (dtype, shape) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                raise D81TypedTargetStateError(f"typed state array drift: {name}")
            if "scale" in name and np.any(value <= 0):
                raise D81TypedTargetStateError(f"typed state scale drift: {name}")
            object.__setattr__(self, name, _readonly(value, dtype))
        for name in (
            "phase1_authority_artifact_sha256", "row_authority_artifact_sha256",
            "old_support_receipt_sha256", "all_registered_support_receipt_sha256", "state_receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        object.__setattr__(self, "fit_audit", _freeze(self.fit_audit))
        object.__setattr__(self, "resource_audit", _freeze(self.resource_audit))
        binding = _json_safe(self.row_authority_binding)
        if (
            binding.get("protocol_schema") != PROTOCOL_SCHEMA
            or binding.get("phase2_data_status") != "VALIDATED_ONCE"
            or int(binding.get("k_shot", -1)) != int(self.k_shot)
            or tuple(binding.get("old_registry", ())) != old
            or tuple(binding.get("final_registry", ())) != final
            or binding.get("old_support", {}).get("support_receipt_sha256")
            != self.old_support_receipt_sha256
            or binding.get("all_registered_support", {}).get("support_receipt_sha256")
            != self.all_registered_support_receipt_sha256
        ):
            raise D81TypedTargetStateError("typed state row-authority binding drift")
        object.__setattr__(self, "row_authority_binding", _freeze(binding))
        phase1_binding = _json_safe(self.phase1_lock_binding)
        if (
            _canonical_sha256(phase1_binding) != self.phase1_authority_artifact_sha256
            or _canonical_sha256(binding) != self.row_authority_artifact_sha256
            or int(phase1_binding.get("metric_seed", -1)) != int(self.config.metric_seed)
            or phase1_binding.get("d81_scorer_receipt_sha256")
            != self.config.d81_scorer_receipt_sha256
            or int(phase1_binding.get("ground_retained_rank", -1))
            != int(self.config.ground_retained_rank)
        ):
            raise D81TypedTargetStateError("typed state external lock binding drift")
        object.__setattr__(self, "phase1_lock_binding", _freeze(phase1_binding))
        if not verify_d81_typed_target_state(self):
            raise D81TypedTargetStateError("typed state receipt/resource drift")

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ARRAY_ORDER}


def fit_d81_typed_target_state(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_physical_ids: Sequence[str],
    all_registered_support_features: np.ndarray,
    all_registered_support_labels: Sequence[str],
    all_registered_physical_ids: Sequence[str],
    *,
    d81_scorer: D81Phase1EpisodeScorer,
    config: D81TypedTargetConfig,
    phase1_authority: D81Phase1Authority,
    row_authority: D81TargetRowAuthority,
) -> D81TypedTargetState:
    """Fit a receipt-bound old-before/all-final D81 lifecycle exactly once."""

    if any(type(value) is not expected for value, expected in (
        (d81_scorer, D81Phase1EpisodeScorer), (config, D81TypedTargetConfig),
        (phase1_authority, D81Phase1Authority), (row_authority, D81TargetRowAuthority),
    )):
        raise D81TypedTargetStateError("fit requires exact typed scorer/config/authority objects")
    config.verify_current(phase1_authority)
    if D81TypedTargetConfig.from_scorer(d81_scorer, phase1_authority) != config:
        raise D81TypedTargetStateError("scorer/config external receipt drift")
    if (
        row_authority.phase1_authority_artifact_sha256 != phase1_authority.artifact_sha256
        or row_authority.method_lock_sha256 != phase1_authority.method_lock_sha256
        or row_authority.phase1_bundle_receipt_sha256 != phase1_authority.phase1_bundle_receipt_sha256
    ):
        raise D81TypedTargetStateError("row/Phase1 authority drift")
    old = _support_closure(old_support_features, old_support_labels, old_physical_ids, row_authority.old_registry, "old_support")
    all_support = _support_closure(
        all_registered_support_features, all_registered_support_labels,
        all_registered_physical_ids, row_authority.final_registry, "all_registered_support",
    )
    old_rows, old_labels, old_ids, old_classes, old_targets, old_k, old_receipt = old
    all_rows, all_labels, all_ids, classes, all_targets, all_k, all_receipt = all_support
    if old_k != all_k or old_k != row_authority.k_shot:
        raise D81TypedTargetStateError("old/all K-shot lifecycle drift")
    _validate_support_authority(old_receipt, row_authority.old_support, "old support")
    _validate_support_authority(all_receipt, row_authority.all_registered_support, "all registered support")
    old_count = len(old_rows)
    if (
        tuple(all_labels[:old_count]) != old_labels
        or tuple(all_ids[:old_count]) != old_ids
        or not np.array_equal(all_rows[:old_count], old_rows)
    ):
        raise D81TypedTargetStateError(
            "all registered support must preserve identical old rows in sealed payload order"
        )

    old_registered = raw_concat_to_d81_registered_feature(old_rows)
    all_registered = raw_concat_to_d81_registered_feature(all_rows)
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    with _FIT_LOCK:
        d81_fit, call_records, transform_records = probe.build_d81_fit(
            d42, d81_scorer.nuisance_basis_fp64, d81_scorer.spectral_weights_fp64,
            _json_safe(d81_scorer.ground_audit),
        )
        log_diag, trace, metric_resource = d42._fit_old_only_b3_metric(
            old_registered, old_targets, len(old_classes), seed=config.metric_seed,
            device=torch.device(d81_scorer.device),
        )
        if len(trace) != 20 or [int(row.get("optimizer_step", -1)) for row in trace] != list(range(1, 21)):
            raise D81TypedTargetStateError("old-only metric lifecycle drift")
        frozen_log_diag = np.array(log_diag, dtype=np.float32, copy=True)
        before_record_start = len(call_records)
        before_coef, before_intercept, before_raw_audit = d81_fit(
            d42._transform(old_registered, frozen_log_diag), old_targets, len(old_classes), old_k
        )
        before_arrays, before_quant = _quantized_head(before_coef, before_intercept, "before")
        before_d62_record_count = len(call_records) - before_record_start
        before_snapshot = tuple(value.tobytes() for value in before_arrays.values()) + (frozen_log_diag.tobytes(),)
        final_record_start = len(call_records)
        final_coef, final_intercept, final_raw_audit = d81_fit(
            d42._transform(all_registered, frozen_log_diag), all_targets, len(classes), all_k
        )
        final_arrays, final_quant = _quantized_head(final_coef, final_intercept, "final")
        final_d62_record_count = len(call_records) - final_record_start
        if before_snapshot != tuple(value.tobytes() for value in before_arrays.values()) + (frozen_log_diag.tobytes(),):
            raise D81TypedTargetStateError("registration-before state mutated during final fit")

    arrays = {"log_diag_fp32": frozen_log_diag, **before_arrays, **final_arrays}
    before_audit = _sanitize_d81_audit(before_raw_audit, len(old_classes), old_k)
    final_audit = _sanitize_d81_audit(final_raw_audit, len(classes), all_k)
    numeric_bytes = sum(int(value.nbytes) for value in arrays.values())
    inventory = _exact_resource_inventory(
        old_class_count=len(old_classes), final_class_count=len(classes),
        k_shot=old_k, config=config, numeric_bytes=numeric_bytes,
    )
    expected_d62_per_head = 0 if old_k <= 2 else 2 * (old_k + 1)
    if (
        before_d62_record_count != expected_d62_per_head
        or final_d62_record_count != expected_d62_per_head
        or len(call_records) != int(inventory["d62_additional_component_fit_count"])
        or int(metric_resource.get("estimated_adaptation_macs", -1))
        != int(inventory["metric_fit_macs"])
    ):
        raise D81TypedTargetStateError("D46/D62 exact component resource inventory drift")
    fit_audit = {
        "schema": "cvs.phase2.d81.typed_target_fit_lifecycle.v2",
        "metric_fit_scope": "old_support_only", "metric_fit_execution_count": 1,
        "metric_optimizer_steps": 20, "metric_frozen_before_all_registered_support_read": True,
        "before_head_scope": "old_support_only", "before_head_fit_count": 1,
        "final_head_scope": "all_registered_support", "final_head_fit_count": 1,
        "old_registry_is_final_registry_prefix": True,
        "old_support_receipt_sha256": old_receipt["support_receipt_sha256"],
        "all_registered_support_receipt_sha256": all_receipt["support_receipt_sha256"],
        "physical_ids_persisted": False, "raw_support_features_persisted": False,
        "query_rows_used": 0, "query_state_updates": 0,
        "old_new_role_specific_query_branch": False, "all_registered_classes_same_formula": True,
        "before_head_audit": before_audit, "final_head_audit": final_audit,
        "before_quantization_error": before_quant, "final_quantization_error": final_quant,
        "metric_trace_sha256": _canonical_sha256(trace),
        "d81_component_fit_call_record_count": len(call_records),
        "before_d62_component_fit_call_record_count": before_d62_record_count,
        "final_d62_component_fit_call_record_count": final_d62_record_count,
        "exact_d46_d62_resource_inventory": inventory,
        "d81_support_transform_record_count": len(transform_records),
    }

    core = _core_payload(
        old_classes=old_classes, classes=classes, k_shot=old_k,
        before_covariance_policy=before_audit["covariance_policy"],
        final_covariance_policy=final_audit["covariance_policy"], arrays=arrays,
        config=config, phase1_authority_sha256=phase1_authority.artifact_sha256,
        row_authority_sha256=row_authority.artifact_sha256,
        old_support_receipt_sha256=old_receipt["support_receipt_sha256"],
        all_support_receipt_sha256=all_receipt["support_receipt_sha256"],
        row_authority_binding=_row_authority_binding(row_authority),
        phase1_lock_binding=_phase1_lock_binding(phase1_authority),
        formal_query_authorized=False,
    )
    ground_logical = int(config.ground_bundle_logical_state_bytes)
    ground_serialized = int(config.ground_component_npz_serialized_bytes + config.ground_manifest_serialized_bytes)
    if ground_logical <= 0:
        raise D81TypedTargetStateError("ground logical state byte audit missing")
    query_geometry_macs = 1856
    query_metric_transform_macs = 4 * FEATURE_DIM
    query_decode_macs = 3 * FEATURE_DIM * len(classes)
    query_affine_macs = (2 * FEATURE_DIM + 1) * len(classes)
    base_resource = {
        "schema": RESOURCE_SCHEMA, "scope": "complete_old_before_all_final_D81_lifecycle",
        "support_only": True, "single_received_observation_per_physical_id": True,
        "optimizer_steps": 20, "metric_optimizer_steps": 20,
        "metric_fit_execution_count": 1, "closed_form_head_fit_count": 2,
        "trainable_parameters": int(inventory["metric_parameter_count"]),
        "head_numeric_logical_state_bytes": int(numeric_bytes),
        "ground_bundle_logical_state_bytes": ground_logical,
        "ground_bundle_serialized_state_bytes": ground_serialized,
        "fit_mac_upper_bound": int(inventory["fit_mac_upper_bound"]),
        "raw288_geometry_fit_macs_upper_bound": int(inventory["raw288_geometry_fit_macs"]),
        "metric_fit_macs": int(inventory["metric_fit_macs"]),
        "d46_d62_component_fit_count": int(inventory["component_fit_count"]),
        "d46_base_component_fit_count": int(inventory["d46_base_component_fit_count"]),
        "d62_additional_component_fit_count": int(inventory["d62_additional_component_fit_count"]),
        "exact_component_fit_inventory": inventory["component_fit_inventory"],
        "exact_lda_component_fit_macs": int(inventory["lda_component_fit_macs"]),
        "fisher_dense_algebra_macs": int(inventory["fisher_dense_algebra_macs"]),
        "reliability_scoring_macs": int(inventory["reliability_scoring_macs"]),
        "classwise_fusion_macs": int(inventory["classwise_fusion_macs"]),
        "pareto_gate_scalar_macs": int(inventory["pareto_gate_scalar_macs"]),
        "support_center_translation_macs_upper_bound": int(inventory["support_center_translation_macs"]),
        "coefficient_quantization_macs_upper_bound": int(inventory["coefficient_quantization_macs"]),
        "complete_fit_lifecycle_peak_bytes_upper_bound": int(inventory["complete_fit_lifecycle_peak_bytes_upper_bound"]),
        "dense_component_workspace_bytes_upper_bound": int(inventory["dense_component_workspace_bytes"]),
        "retained_component_audit_bytes_upper_bound": int(inventory["retained_component_audit_bytes_upper_bound"]),
        "peak_includes_input_and_canonical_support": True,
        "peak_includes_registered_and_transformed_support": True,
        "peak_includes_optimizer_gradient_and_activation_bound": True,
        "peak_includes_ground_decode_and_two_fp32_heads": True,
        "query_raw288_geometry_macs_upper_bound": query_geometry_macs,
        "query_metric_transform_macs_upper_bound": query_metric_transform_macs,
        "query_int8_decode_macs_upper_bound": query_decode_macs,
        "query_affine_macs_upper_bound": query_affine_macs,
        "query_mac_upper_bound_per_sample": int(query_geometry_macs + query_metric_transform_macs + query_decode_macs + query_affine_macs),
        "query_decoded_coefficient_cache_bytes": 0,
        "query_state_updates": 0, "query_rows_used_for_fit": 0,
        "query_file_io": False, "query_external_hash_reads": False,
        "query_full_state_hash_reads": False,
        "runtime_device": str(d81_scorer.device),
    }
    resource, receipt = _resource_with_exact_wire_sizes(base_resource, core, fit_audit, arrays)
    resource = {
        **resource,
        "total_deployment_serialized_bytes_including_ground": int(
            resource["total_wire_serialized_bytes"] + ground_serialized
        ),
        "total_deployment_logical_bytes_including_ground": int(numeric_bytes + ground_logical),
    }
    # The additional total fields change the canonical audit length; close the
    # actual wire sizes once more with the final resource schema.
    resource, receipt = _resource_with_exact_wire_sizes(resource, core, fit_audit, arrays)
    return D81TypedTargetState(
        old_classes=old_classes, classes=classes, k_shot=old_k,
        log_diag_fp32=arrays["log_diag_fp32"],
        before_coef1_qint8=arrays["before_coef1_qint8"], before_coef2_qint8=arrays["before_coef2_qint8"],
        before_scale1_fp16=arrays["before_scale1_fp16"], before_scale2_fp16=arrays["before_scale2_fp16"],
        before_intercept_fp16=arrays["before_intercept_fp16"],
        final_coef1_qint8=arrays["final_coef1_qint8"], final_coef2_qint8=arrays["final_coef2_qint8"],
        final_scale1_fp16=arrays["final_scale1_fp16"], final_scale2_fp16=arrays["final_scale2_fp16"],
        final_intercept_fp16=arrays["final_intercept_fp16"],
        before_covariance_policy=before_audit["covariance_policy"], final_covariance_policy=final_audit["covariance_policy"],
        config=config, phase1_authority_artifact_sha256=phase1_authority.artifact_sha256,
        row_authority_artifact_sha256=row_authority.artifact_sha256,
        old_support_receipt_sha256=old_receipt["support_receipt_sha256"],
        all_registered_support_receipt_sha256=all_receipt["support_receipt_sha256"],
        row_authority_binding=_row_authority_binding(row_authority),
        phase1_lock_binding=_phase1_lock_binding(phase1_authority),
        formal_query_authorized=False,
        fit_audit=fit_audit, resource_audit=resource, state_receipt_sha256=receipt,
        _token=_STATE_TOKEN,
    )


def _state_core(state: D81TypedTargetState) -> dict[str, Any]:
    return _core_payload(
        old_classes=state.old_classes, classes=state.classes, k_shot=state.k_shot,
        before_covariance_policy=state.before_covariance_policy,
        final_covariance_policy=state.final_covariance_policy, arrays=state.arrays,
        config=state.config, phase1_authority_sha256=state.phase1_authority_artifact_sha256,
        row_authority_sha256=state.row_authority_artifact_sha256,
        old_support_receipt_sha256=state.old_support_receipt_sha256,
        all_support_receipt_sha256=state.all_registered_support_receipt_sha256,
        row_authority_binding=state.row_authority_binding,
        phase1_lock_binding=state.phase1_lock_binding,
        formal_query_authorized=state.formal_query_authorized,
    )


def _verify_independent_resource(state: D81TypedTargetState) -> bool:
    numeric = sum(int(value.nbytes) for value in state.arrays.values())
    expected = _exact_resource_inventory(
        old_class_count=len(state.old_classes), final_class_count=len(state.classes),
        k_shot=state.k_shot, config=state.config, numeric_bytes=numeric,
    )
    resource = state.resource_audit
    audit_inventory = state.fit_audit.get("exact_d46_d62_resource_inventory")
    exact_fields = {
        "fit_mac_upper_bound": "fit_mac_upper_bound",
        "metric_fit_macs": "metric_fit_macs",
        "d46_d62_component_fit_count": "component_fit_count",
        "d46_base_component_fit_count": "d46_base_component_fit_count",
        "d62_additional_component_fit_count": "d62_additional_component_fit_count",
        "exact_lda_component_fit_macs": "lda_component_fit_macs",
        "fisher_dense_algebra_macs": "fisher_dense_algebra_macs",
        "reliability_scoring_macs": "reliability_scoring_macs",
        "classwise_fusion_macs": "classwise_fusion_macs",
        "pareto_gate_scalar_macs": "pareto_gate_scalar_macs",
        "support_center_translation_macs_upper_bound": "support_center_translation_macs",
        "coefficient_quantization_macs_upper_bound": "coefficient_quantization_macs",
        "complete_fit_lifecycle_peak_bytes_upper_bound": "complete_fit_lifecycle_peak_bytes_upper_bound",
        "dense_component_workspace_bytes_upper_bound": "dense_component_workspace_bytes",
        "retained_component_audit_bytes_upper_bound": "retained_component_audit_bytes_upper_bound",
    }
    if _json_safe(audit_inventory) != _json_safe(expected):
        return False
    if _json_safe(resource.get("exact_component_fit_inventory")) != _json_safe(
        expected["component_fit_inventory"]
    ):
        return False
    if any(int(resource.get(name, -1)) != int(expected[key]) for name, key in exact_fields.items()):
        return False
    query_expected = int(
        1856
        + 4 * FEATURE_DIM
        + 3 * FEATURE_DIM * len(state.classes)
        + (2 * FEATURE_DIM + 1) * len(state.classes)
    )
    return bool(
        int(resource.get("head_numeric_logical_state_bytes", -1)) == numeric
        and int(resource.get("ground_bundle_logical_state_bytes", -1))
        == int(state.config.ground_bundle_logical_state_bytes)
        and int(resource.get("ground_bundle_serialized_state_bytes", -1))
        == int(
            state.config.ground_component_npz_serialized_bytes
            + state.config.ground_manifest_serialized_bytes
        )
        and int(resource.get("trainable_parameters", -1))
        == int(expected["metric_parameter_count"])
        and int(resource.get("query_mac_upper_bound_per_sample", -1)) == query_expected
        and int(state.fit_audit.get("d81_component_fit_call_record_count", -1))
        == int(expected["d62_additional_component_fit_count"])
    )


def verify_d81_typed_target_state(state: D81TypedTargetState) -> bool:
    """Pure in-memory verification; performs no file I/O or dependency hashing."""

    if type(state) is not D81TypedTargetState or state._token is not _STATE_TOKEN:
        return False
    try:
        wire, receipt = _wire_from_parts(_state_core(state), state.fit_audit, state.resource_audit, state.arrays)
        resource = state.resource_audit
        numeric = sum(int(value.nbytes) for value in state.arrays.values())
        return bool(
            receipt == state.state_receipt_sha256
            and _verify_independent_resource(state)
            and int(resource["head_numeric_logical_state_bytes"]) == numeric
            and int(resource["total_wire_serialized_bytes"]) == len(wire)
            and int(resource["deploy_state_serialized_bytes"]) + int(resource["audit_serialized_bytes"]) == len(wire)
            and int(resource["query_state_updates"]) == 0
            and resource["query_file_io"] is False
            and resource["query_external_hash_reads"] is False
            and resource["query_full_state_hash_reads"] is False
        )
    except (KeyError, TypeError, ValueError, D81TypedTargetStateError):
        return False


def serialize_d81_typed_target_state(state: D81TypedTargetState) -> bytes:
    if not verify_d81_typed_target_state(state):
        raise D81TypedTargetStateError("cannot serialize invalid typed state")
    wire, receipt = _wire_from_parts(_state_core(state), state.fit_audit, state.resource_audit, state.arrays)
    if receipt != state.state_receipt_sha256 or len(wire) != int(state.resource_audit["total_wire_serialized_bytes"]):
        raise D81TypedTargetStateError("serialized wire closure drift")
    return wire


def save_d81_typed_target_state(state: D81TypedTargetState, path: str | Path) -> str:
    raw = serialize_d81_typed_target_state(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
    except FileExistsError as exc:
        raise D81TypedTargetStateError("typed state artifact already exists") from exc
    return _sha256_bytes(raw)


def _config_from_json(value: Mapping[str, Any]) -> D81TypedTargetConfig:
    allowed = {item.name for item in fields(D81TypedTargetConfig)}
    if set(value) != allowed:
        raise D81TypedTargetStateError("wire config field closure drift")
    return D81TypedTargetConfig(**value)


def load_d81_typed_target_state(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
    phase1_authority: D81Phase1Authority,
    row_authority: D81TargetRowAuthority,
) -> D81TypedTargetState:
    """Load and externally authorize the wire state before query deployment."""

    if type(phase1_authority) is not D81Phase1Authority or type(row_authority) is not D81TargetRowAuthority:
        raise D81TypedTargetStateError("external authorities required at state load")
    expected = _require_sha256(expected_artifact_sha256, "expected state artifact")
    raw = Path(path).read_bytes()
    if _sha256_bytes(raw) != expected:
        raise D81TypedTargetStateError("state artifact SHA mismatch")
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 16:
        raise D81TypedTargetStateError("state wire magic/truncation drift")
    cursor = len(MAGIC)
    header_len = struct.unpack("<Q", raw[cursor: cursor + 8])[0]
    cursor += 8
    header = _decode_canonical_object(raw[cursor: cursor + header_len], "state header")
    cursor += header_len
    audit_len = struct.unpack("<Q", raw[cursor: cursor + 8])[0]
    cursor += 8
    audit = _decode_canonical_object(raw[cursor: cursor + audit_len], "state audit")
    cursor += audit_len
    if set(audit) != {"fit_audit", "resource_audit"}:
        raise D81TypedTargetStateError("state audit field closure drift")
    if (
        header.get("schema") != SCHEMA or header.get("deployment_status") != DEPLOYMENT_STATUS
        or header.get("phase1_authority_artifact_sha256") != phase1_authority.artifact_sha256
        or header.get("row_authority_artifact_sha256") != row_authority.artifact_sha256
        or tuple(header.get("old_classes", ())) != row_authority.old_registry
        or tuple(header.get("classes", ())) != row_authority.final_registry
        or int(header.get("k_shot", -1)) != row_authority.k_shot
        or header.get("old_support_receipt_sha256") != row_authority.old_support["support_receipt_sha256"]
        or header.get("all_registered_support_receipt_sha256") != row_authority.all_registered_support["support_receipt_sha256"]
        or header.get("row_authority_binding") != _row_authority_binding(row_authority)
        or header.get("phase1_lock_binding") != _phase1_lock_binding(phase1_authority)
        or header.get("formal_query_authorized") is not False
    ):
        raise D81TypedTargetStateError("wire/external authority lifecycle drift")
    config = _config_from_json(header.get("config", {}))
    config.verify_current(phase1_authority)
    array_receipts = header.get("arrays")
    if not isinstance(array_receipts, dict) or set(array_receipts) != set(ARRAY_ORDER):
        raise D81TypedTargetStateError("wire array order drift")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_ORDER:
        receipt = array_receipts[name]
        try:
            dtype = np.dtype(receipt["dtype"])
            shape = tuple(int(value) for value in receipt["shape"])
            nbytes = int(receipt["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise D81TypedTargetStateError("wire array receipt drift") from exc
        if nbytes != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize or cursor + nbytes > len(raw):
            raise D81TypedTargetStateError("wire array size/truncation drift")
        block = raw[cursor: cursor + nbytes]
        cursor += nbytes
        if _sha256_bytes(block) != receipt.get("sha256"):
            raise D81TypedTargetStateError("wire array SHA drift")
        arrays[name] = np.frombuffer(block, dtype=dtype).reshape(shape)
    if cursor != len(raw):
        raise D81TypedTargetStateError("wire trailing bytes drift")
    core = {key: value for key, value in header.items() if key != "state_receipt_sha256"}
    expected_receipt = _canonical_sha256({"core": core, "audit": audit})
    if header.get("state_receipt_sha256") != expected_receipt:
        raise D81TypedTargetStateError("wire state receipt drift")
    state = D81TypedTargetState(
        old_classes=tuple(header["old_classes"]), classes=tuple(header["classes"]), k_shot=int(header["k_shot"]),
        before_covariance_policy=str(header["before_covariance_policy"]), final_covariance_policy=str(header["final_covariance_policy"]),
        config=config, phase1_authority_artifact_sha256=header["phase1_authority_artifact_sha256"],
        row_authority_artifact_sha256=header["row_authority_artifact_sha256"],
        old_support_receipt_sha256=header["old_support_receipt_sha256"],
        all_registered_support_receipt_sha256=header["all_registered_support_receipt_sha256"],
        row_authority_binding=header["row_authority_binding"],
        phase1_lock_binding=header["phase1_lock_binding"],
        formal_query_authorized=header["formal_query_authorized"],
        fit_audit=audit["fit_audit"], resource_audit=audit["resource_audit"],
        state_receipt_sha256=header["state_receipt_sha256"], _token=_STATE_TOKEN,
        **arrays,
    )
    if serialize_d81_typed_target_state(state) != raw:
        raise D81TypedTargetStateError("wire canonical round-trip drift")
    return state


def _require_query_ready(state: D81TypedTargetState) -> None:
    """O(1) deployment guard; full hashing is completed at fit/load time."""

    if type(state) is not D81TypedTargetState or state._token is not _STATE_TOKEN:
        raise D81TypedTargetStateError("query requires a fit/loaded typed state")
    if (
        state.deployment_status != DEPLOYMENT_STATUS
        or state.classes[: len(state.old_classes)] != state.old_classes
        or int(state.resource_audit.get("query_state_updates", -1)) != 0
        or state.resource_audit.get("query_file_io") is not False
        or state.resource_audit.get("query_external_hash_reads") is not False
        or state.resource_audit.get("query_full_state_hash_reads") is not False
    ):
        raise D81TypedTargetStateError("typed query lifecycle guard failed")
    # Shapes and dtypes were fully checked during construction.  Read-only flags
    # make later caller mutation impossible without constructing a new state,
    # which itself re-enters the full receipt verifier.
    if any(np.asarray(getattr(state, name)).flags.writeable for name in ARRAY_ORDER):
        raise D81TypedTargetStateError("typed query state became writable")


def _score_head(
    state: D81TypedTargetState,
    query_features: np.ndarray,
    prefix: str,
) -> np.ndarray:
    _require_query_ready(state)
    query = np.asarray(query_features)
    if query.dtype != np.float32 or query.ndim != 2 or query.shape[1] != FEATURE_DIM or len(query) < 1 or not np.isfinite(query).all():
        raise D81TypedTargetStateError("query must be finite float32 [N,288]")
    registered = raw_concat_to_d81_registered_feature(query)
    transformed = d42._transform(registered, state.log_diag_fp32)
    coefficient = d42._decode_coefficients(
        getattr(state, f"{prefix}_coef1_qint8"), getattr(state, f"{prefix}_coef2_qint8"),
        getattr(state, f"{prefix}_scale1_fp16"), getattr(state, f"{prefix}_scale2_fp16"),
    )
    intercept = getattr(state, f"{prefix}_intercept_fp16").astype(np.float32)
    # Deliberately score one query at a time.  This makes the numerical path
    # independent of query batching and prevents BLAS batch shape from changing
    # the last bits of an otherwise per-sample decision.
    result = np.stack(
        [
            np.asarray(row @ coefficient.T + intercept, dtype=np.float32)
            for row in transformed
        ],
        axis=0,
    )
    result = _readonly(result, np.float32)
    expected_classes = len(state.old_classes) if prefix == "before" else len(state.classes)
    if result.shape != (len(query), expected_classes) or not np.isfinite(result).all():
        raise D81TypedTargetStateError("typed query score drift")
    return result


def score_d81_typed_old_before_raw_logits(state: D81TypedTargetState, query_features: np.ndarray) -> np.ndarray:
    # Do not inspect ``state`` while the external formal producer is absent.
    # Future enablement must first introduce a slots-only formal-state type and
    # call the pure in-memory receipt/config/registry/shape/resource verifier.
    raise D81TypedTargetStateError(
        "formal query unavailable: external formal-state producer is absent"
    )


def score_d81_typed_target_raw_logits(state: D81TypedTargetState, query_features: np.ndarray) -> np.ndarray:
    # Unconditional fail-closed: caller-controlled flags or copied objects can
    # never turn a LOCAL artifact into a formal deployment capability.
    raise D81TypedTargetStateError(
        "formal query unavailable: external formal-state producer is absent"
    )


def score_d81_typed_local_diagnostic_old_before_logits(
    state: D81TypedTargetState, query_features: np.ndarray
) -> np.ndarray:
    """Local exactness-only scorer; never authorizes deployment claims."""

    return _score_head(state, query_features, "before")


def score_d81_typed_local_diagnostic_target_logits(
    state: D81TypedTargetState, query_features: np.ndarray
) -> np.ndarray:
    """Local exactness-only scorer; never authorizes deployment claims."""

    return _score_head(state, query_features, "final")


__all__ = [
    "ALLOWED_K_SHOT", "CONFIG_SCHEMA", "DEPLOYMENT_STATUS",
    "D81Phase1Authority", "D81TargetRowAuthority", "D81TypedTargetConfig",
    "D81TypedTargetState", "D81TypedTargetStateError", "PHASE1_AUTHORITY_SCHEMA",
    "RESOURCE_SCHEMA", "ROW_AUTHORITY_SCHEMA", "SCHEMA",
    "fit_d81_typed_target_state", "load_d81_phase1_authority",
    "load_d81_target_row_authority", "load_d81_typed_target_state",
    "save_d81_typed_target_state", "score_d81_typed_old_before_raw_logits",
    "score_d81_typed_target_raw_logits",
    "score_d81_typed_local_diagnostic_old_before_logits",
    "score_d81_typed_local_diagnostic_target_logits",
    "serialize_d81_typed_target_state",
    "verify_d81_typed_target_state",
]
