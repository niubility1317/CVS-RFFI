"""Receipt-bound target-row D81 state with no caller-supplied logit surface.

The only public producer consumes balanced target support and the exact sealed
``D81Phase1EpisodeScorer`` capability.  It executes the historical D81-before
chain once, persists the compiled INT8 affine head, and returns an immutable
typed state.  The public scorer accepts only that state and raw concat288 query
features.  It cannot accept logits, probabilities, query labels, receiver or
old/new role information.

This is a local implementation core pending independent review.  It does not
itself authorize deployment or a performance claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
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


SCHEMA = "cvs.phase2.d81.typed_target_state.v1"
CONFIG_SCHEMA = "cvs.phase2.d81.typed_target_config.v1"
RESOURCE_SCHEMA = "cvs.phase2.d81.typed_target_resource.v1"
DEPLOYMENT_STATUS = "LOCAL_CORE_PENDING_INDEPENDENT_REVIEW"
PROTOCOL_SCHEMA = "p2_min_v1"
ALLOWED_K_SHOT = (1, 5, 10)
ARRAY_ORDER = (
    "log_diag_fp32",
    "coef1_qint8",
    "coef2_qint8",
    "scale1_fp16",
    "scale2_fp16",
    "intercept_fp16",
)


class D81TypedTargetStateError(ValueError):
    """Raised when typed D81 inputs, provenance, state, or resources drift."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.view(np.uint8).tobytes()),
        "nbytes": int(array.nbytes),
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
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
    raise D81TypedTargetStateError(
        f"D81 typed state contains noncanonical {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise D81TypedTargetStateError(f"{name} must be lowercase SHA256")
    return normalized


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
    dependencies = dict(_d81_dependency_hashes())
    dependencies["typed_target_state"] = _sha256_file(Path(__file__).resolve())
    return tuple(sorted((str(key), str(value)) for key, value in dependencies.items()))


def _ground_serialized_sizes(scorer: D81Phase1EpisodeScorer) -> tuple[int, int]:
    audit = scorer.ground_audit
    component_path_value = audit.get("component_path")
    if not isinstance(component_path_value, str) or not component_path_value:
        raise D81TypedTargetStateError(
            "D81 ground audit must seal the component_path for serialized-byte closure"
        )
    component_path = Path(component_path_value).resolve()
    manifest_path = component_path.parent / "manifest.json"
    if not component_path.is_file() or not manifest_path.is_file():
        raise D81TypedTargetStateError("D81 sealed ground component files are missing")
    if _sha256_file(component_path) != scorer.ground_component_npz_sha256:
        raise D81TypedTargetStateError("D81 sealed ground component SHA drift")
    if _sha256_file(manifest_path) != scorer.ground_manifest_sha256:
        raise D81TypedTargetStateError("D81 sealed ground manifest SHA drift")
    return int(component_path.stat().st_size), int(manifest_path.stat().st_size)


@dataclass(frozen=True)
class D81TypedTargetConfig:
    d81_scorer_receipt_sha256: str
    phase1_checkpoint_sha256: str
    ground_manifest_sha256: str
    ground_component_npz_sha256: str
    dependency_code_sha256: tuple[tuple[str, str], ...]
    dependency_closure_sha256: str
    metric_seed: int
    ground_component_npz_serialized_bytes: int
    ground_manifest_serialized_bytes: int
    schema: str = CONFIG_SCHEMA
    protocol_schema: str = PROTOCOL_SCHEMA
    feature_geometry: str = "raw_concat288_to_D81_registered_feature"
    metric_optimizer_steps: int = d42.METRIC_EPOCHS
    affine_precision: str = "three_block_two_level_residual_int8_fp16_scale_intercept"

    def __post_init__(self) -> None:
        if (
            self.schema != CONFIG_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or self.feature_geometry != "raw_concat288_to_D81_registered_feature"
            or int(self.metric_optimizer_steps) != 20
            or int(self.metric_optimizer_steps) != d42.METRIC_EPOCHS
            or self.affine_precision
            != "three_block_two_level_residual_int8_fp16_scale_intercept"
            or not 0 <= int(self.metric_seed) <= 0x7FFFFFFF
            or int(self.ground_component_npz_serialized_bytes) <= 0
            or int(self.ground_manifest_serialized_bytes) <= 0
        ):
            raise D81TypedTargetStateError("D81 typed target config lock drift")
        for value, name in (
            (self.d81_scorer_receipt_sha256, "D81 scorer receipt"),
            (self.phase1_checkpoint_sha256, "Phase1 checkpoint"),
            (self.ground_manifest_sha256, "ground manifest"),
            (self.ground_component_npz_sha256, "ground component"),
            (self.dependency_closure_sha256, "dependency closure"),
        ):
            _require_sha256(value, name)
        dependencies = tuple((str(name), _require_sha256(sha, name)) for name, sha in self.dependency_code_sha256)
        if (
            dependencies != tuple(sorted(dependencies))
            or len({name for name, _sha in dependencies}) != len(dependencies)
            or self.dependency_closure_sha256
            != _canonical_sha256(dict(dependencies))
        ):
            raise D81TypedTargetStateError("D81 dependency receipt drift")
        object.__setattr__(self, "dependency_code_sha256", dependencies)

    @classmethod
    def from_scorer(cls, scorer: D81Phase1EpisodeScorer) -> "D81TypedTargetConfig":
        if type(scorer) is not D81Phase1EpisodeScorer:
            raise D81TypedTargetStateError(
                "D81 target config requires exact D81Phase1EpisodeScorer"
            )
        component_bytes, manifest_bytes = _ground_serialized_sizes(scorer)
        dependencies = _current_dependency_hashes()
        return cls(
            d81_scorer_receipt_sha256=scorer.scorer_id,
            phase1_checkpoint_sha256=scorer.phase1_checkpoint_sha256,
            ground_manifest_sha256=scorer.ground_manifest_sha256,
            ground_component_npz_sha256=scorer.ground_component_npz_sha256,
            dependency_code_sha256=dependencies,
            dependency_closure_sha256=_canonical_sha256(dict(dependencies)),
            metric_seed=int(scorer.metric_seed),
            ground_component_npz_serialized_bytes=component_bytes,
            ground_manifest_serialized_bytes=manifest_bytes,
        )

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256(self)

    def verify_current(self) -> None:
        current = _current_dependency_hashes()
        if current != self.dependency_code_sha256:
            raise D81TypedTargetStateError("D81 dependency code drift")
        if _canonical_sha256(dict(current)) != self.dependency_closure_sha256:
            raise D81TypedTargetStateError("D81 dependency closure drift")


def _support_closure(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    registered_classes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], int, str]:
    features = np.asarray(support_features)
    labels = tuple(str(value) for value in support_labels)
    ids = tuple(str(value) for value in physical_ids)
    classes = tuple(str(value) for value in registered_classes)
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != FEATURE_DIM
        or len(features) < 2
        or not np.isfinite(features).all()
        or len(labels) != len(features)
        or len(ids) != len(features)
        or len(classes) < 2
        or len(set(classes)) != len(classes)
        or any(not value for value in classes)
        or any(not value for value in ids)
        or len(set(ids)) != len(ids)
        or set(labels) != set(classes)
    ):
        raise D81TypedTargetStateError("D81 target support/registry closure drift")
    counts = tuple(sum(label == class_id for label in labels) for class_id in classes)
    if min(counts) < 1 or len(set(counts)) != 1 or counts[0] not in ALLOWED_K_SHOT:
        raise D81TypedTargetStateError("D81 target support must be balanced K1/K5/K10")
    order = tuple(sorted(range(len(features)), key=lambda index: (labels[index], ids[index])))
    canonical_features = np.ascontiguousarray(features[np.asarray(order)], dtype=np.float32)
    canonical_labels = tuple(labels[index] for index in order)
    canonical_ids = tuple(ids[index] for index in order)
    members = [
        {
            "class": label,
            "physical_id_sha256": _sha256_bytes(physical_id.encode("utf-8")),
            "feature_sha256": _sha256_bytes(canonical_features[index].view(np.uint8).tobytes()),
        }
        for index, (label, physical_id) in enumerate(zip(canonical_labels, canonical_ids, strict=True))
    ]
    support_receipt = _canonical_sha256(
        {
            "schema": "cvs.phase2.d81.typed_support_receipt.v1",
            "k_shot": counts[0],
            "class_set": sorted(classes),
            "members": members,
            "single_received_observation_per_physical_id": True,
        }
    )
    return canonical_features, canonical_labels, canonical_ids, classes, counts[0], support_receipt


def _state_core_payload(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    covariance_policy: str,
    arrays: Mapping[str, np.ndarray],
    config: D81TypedTargetConfig,
    support_receipt_sha256: str,
    fit_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "deployment_status": DEPLOYMENT_STATUS,
        "classes": list(classes),
        "k_shot": int(k_shot),
        "feature_geometry": config.feature_geometry,
        "covariance_policy": str(covariance_policy),
        "arrays": {name: _array_receipt(arrays[name]) for name in ARRAY_ORDER},
        "config": _json_safe(config),
        "config_lock_digest": config.lock_digest,
        "support_receipt_sha256": support_receipt_sha256,
        "fit_audit": _json_safe(fit_audit),
    }


def _state_receipt_payload(
    core: Mapping[str, Any], resource_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {**_json_safe(core), "resource_audit": _json_safe(resource_audit)}


def _serialized_head_bytes(core: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> int:
    # Exact size of the locked deterministic wire format: an 8-byte little-endian
    # header length, canonical UTF-8 header, then contiguous arrays in ARRAY_ORDER.
    return int(8 + len(_canonical_bytes(core)) + sum(arrays[name].nbytes for name in ARRAY_ORDER))


@dataclass(frozen=True)
class D81TypedTargetState:
    classes: tuple[str, ...]
    k_shot: int
    log_diag_fp32: np.ndarray
    coef1_qint8: np.ndarray
    coef2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    intercept_fp16: np.ndarray
    covariance_policy: str
    config: D81TypedTargetConfig
    support_receipt_sha256: str
    fit_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    state_receipt_sha256: str
    schema: str = SCHEMA
    deployment_status: str = DEPLOYMENT_STATUS

    def __post_init__(self) -> None:
        count = len(self.classes)
        if (
            self.schema != SCHEMA
            or self.deployment_status != DEPLOYMENT_STATUS
            or type(self.config) is not D81TypedTargetConfig
            or type(self.classes) is not tuple
            or count < 2
            or len(set(self.classes)) != count
            or any(not value for value in self.classes)
            or int(self.k_shot) not in ALLOWED_K_SHOT
            or self.covariance_policy
            not in {
                "sklearn_lsqr_auto_shrinkage_equal_prior",
                "unit_covariance_equal_prior_nearest_centroid",
            }
        ):
            raise D81TypedTargetStateError("D81 typed state metadata drift")
        expected = {
            "log_diag_fp32": (np.float32, (FEATURE_DIM,)),
            "coef1_qint8": (np.int8, (count, FEATURE_DIM)),
            "coef2_qint8": (np.int8, (count, FEATURE_DIM)),
            "scale1_fp16": (np.float16, (count, len(d42.BLOCK_SLICES))),
            "scale2_fp16": (np.float16, (count, len(d42.BLOCK_SLICES))),
            "intercept_fp16": (np.float16, (count,)),
        }
        for name, (dtype, shape) in expected.items():
            source = np.asarray(getattr(self, name))
            if source.dtype != dtype or source.shape != shape or not np.isfinite(source).all():
                raise D81TypedTargetStateError(f"D81 typed state {name} drift")
            if "scale" in name and bool(np.any(source <= 0)):
                raise D81TypedTargetStateError("D81 typed state scale drift")
            object.__setattr__(self, name, _readonly(source, dtype))
        object.__setattr__(self, "fit_audit", _freeze(self.fit_audit))
        object.__setattr__(self, "resource_audit", _freeze(self.resource_audit))
        _require_sha256(self.support_receipt_sha256, "support receipt")
        _require_sha256(self.state_receipt_sha256, "state receipt")
        if not verify_d81_typed_target_state(self, verify_dependencies=False):
            raise D81TypedTargetStateError("D81 typed state receipt drift")

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ARRAY_ORDER}


def _sanitize_fit_audit(
    audit: Mapping[str, Any], *, class_count: int, k_shot: int, trace: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    required = {
        "covariance_policy": audit.get("covariance_policy"),
        "d81_probe_arm": audit.get("d81_probe_arm"),
        "d81_structure": audit.get("d81_structure"),
        "d81_ground_int8_component_used": audit.get("d81_ground_int8_component_used"),
        "d81_ground_component_update_access": audit.get("d81_ground_component_update_access"),
        "d81_old_new_role_specific_branch": audit.get("d81_old_new_role_specific_branch"),
        "d81_class_id_specific_formula": audit.get("d81_class_id_specific_formula"),
        "d81_uses_outer_held_or_query": audit.get("d81_uses_outer_held_or_query"),
        "d81_query_rows_used": audit.get("d81_query_rows_used"),
        "d81_single_affine_state_only": audit.get("d81_single_affine_state_only"),
        "d81_transform_audit": audit.get("d81_transform_audit"),
    }
    transform = required["d81_transform_audit"]
    if (
        required["covariance_policy"]
        not in {
            "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_equal_prior_nearest_centroid",
        }
        or required["d81_ground_int8_component_used"] is not True
        or required["d81_ground_component_update_access"] is not False
        or required["d81_old_new_role_specific_branch"] is not False
        or required["d81_class_id_specific_formula"] is not False
        or required["d81_uses_outer_held_or_query"] is not False
        or int(required["d81_query_rows_used"] if required["d81_query_rows_used"] is not None else -1) != 0
        or required["d81_single_affine_state_only"] is not True
        or required["d81_probe_arm"] != "ground_nuisance_cauchy_center"
        or required["d81_structure"]
        != "d62_with_ground_spectrum_support_only_class_center_translation"
        or not isinstance(transform, Mapping)
        or transform.get("schema") != "cvs.phase2.d81.support_center_translation.v1"
        or int(transform.get("support_rows", -1)) != int(class_count) * int(k_shot)
        or int(transform.get("class_count", -1)) != int(class_count)
        or int(transform.get("k_shot", -1)) != int(k_shot)
        or int(transform.get("retained_rank", -1)) < 1
        or transform.get("translation_scope") != "z160_class_common_only"
        or transform.get("old_new_role_specific_branch") is not False
        or transform.get("class_id_specific_formula") is not False
        or transform.get("uses_outer_held_or_query") is not False
        or int(transform.get("query_rows_used", -1)) != 0
        or float(transform.get("within_class_residual_max_abs_error", 1.0)) > 2.0e-12
        or float(transform.get("fft96_rf32_max_abs_error", 1.0)) != 0.0
        or len(trace) != 20
        or [int(row.get("optimizer_step", -1)) for row in trace] != list(range(1, 21))
    ):
        raise D81TypedTargetStateError("D81 exact support-only fit audit drift")
    effective_samples = tuple(
        float(value) for value in transform.get("effective_sample_size_by_class", ())
    )
    if len(effective_samples) != int(class_count) or any(
        not np.isfinite(value) or value <= 0.0 for value in effective_samples
    ):
        raise D81TypedTargetStateError("D81 transform effective-sample audit drift")
    # Persist aggregate invariants only.  Per-support energy/weight vectors and
    # physical identifiers are deliberately absent from the compiled state.
    required["d81_transform_audit"] = {
        "schema": transform.get("schema"),
        "support_rows": int(transform.get("support_rows", -1)),
        "class_count": int(transform.get("class_count", -1)),
        "k_shot": int(transform.get("k_shot", -1)),
        "retained_rank": int(transform.get("retained_rank", -1)),
        "translation_scope": transform.get("translation_scope"),
        "k1_k2_exact_identity": transform.get("k1_k2_exact_identity"),
        "uses_outer_held_or_query": transform.get("uses_outer_held_or_query"),
        "query_rows_used": int(transform.get("query_rows_used", -1)),
        "within_class_residual_max_abs_error": float(
            transform.get("within_class_residual_max_abs_error", 1.0)
        ),
        "fft96_rf32_max_abs_error": float(
            transform.get("fft96_rf32_max_abs_error", 1.0)
        ),
        "center_shift_l2_max": float(transform.get("center_shift_l2_max", -1.0)),
        "normalized_weight_min": float(transform.get("normalized_weight_min", -1.0)),
        "normalized_weight_max": float(transform.get("normalized_weight_max", -1.0)),
        "effective_sample_size_min": float(min(effective_samples)),
        "effective_sample_size_max": float(max(effective_samples)),
        "per_support_energy_or_weight_vectors_persisted": False,
    }
    return {
        "schema": "cvs.phase2.d81.typed_target_fit_audit.v1",
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "metric_fit_execution_count": 1,
        "metric_optimizer_steps": 20,
        "d81_head_fit_execution_count": 1,
        "all_registered_classes_same_formula": True,
        "old_new_role_input": False,
        "query_rows_used": 0,
        "query_state_updates": 0,
        **required,
    }


def _translation_macs_upper_bound(
    class_count: int, k_shot: int, retained_rank: int
) -> int:
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    if hasattr(probe, "_d62_translation_chain_macs"):
        return int(probe._d62_translation_chain_macs(class_count, k_shot, retained_rank))
    return int(16 * class_count * k_shot * 160 * max(1, retained_rank))


def fit_d81_typed_target_state(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    *,
    d81_scorer: D81Phase1EpisodeScorer,
    config: D81TypedTargetConfig,
) -> D81TypedTargetState:
    """Fit one immutable D81 target-row state from all registered support."""

    if type(d81_scorer) is not D81Phase1EpisodeScorer or type(config) is not D81TypedTargetConfig:
        raise D81TypedTargetStateError("D81 typed fit requires exact scorer/config types")
    expected_config = D81TypedTargetConfig.from_scorer(d81_scorer)
    if config != expected_config or config.lock_digest != expected_config.lock_digest:
        raise D81TypedTargetStateError("D81 typed fit scorer/config receipt drift")
    config.verify_current()
    features, labels, _ids, requested_classes, k_shot, support_receipt = _support_closure(
        support_features, support_labels, physical_ids, registered_classes
    )
    canonical_classes = tuple(sorted(requested_classes))
    lookup = {value: index for index, value in enumerate(canonical_classes)}
    targets = np.asarray([lookup[value] for value in labels], dtype=np.int64)
    registered = raw_concat_to_d81_registered_feature(features)
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    with _FIT_LOCK:
        d81_fit, call_records, transform_records = probe.build_d81_fit(
            d42,
            d81_scorer.nuisance_basis_fp64,
            d81_scorer.spectral_weights_fp64,
            _json_safe(d81_scorer.ground_audit),
        )
        log_diag, trace, metric_resource = d42._fit_old_only_b3_metric(
            registered,
            targets,
            len(canonical_classes),
            seed=config.metric_seed,
            device=torch.device(d81_scorer.device),
        )
        transformed = d42._transform(registered, log_diag)
        coefficient, intercept, raw_audit = d81_fit(
            transformed, targets, len(canonical_classes), k_shot
        )
        code1, code2, scale1, scale2, decoded = d42._quantize_coefficients(coefficient)
    intercept16 = np.asarray(intercept, dtype=np.float16)
    if not np.isfinite(intercept16).all():
        raise D81TypedTargetStateError("D81 typed intercept FP16 overflow")
    order = np.asarray([canonical_classes.index(value) for value in requested_classes], dtype=np.int64)
    arrays = {
        "log_diag_fp32": np.asarray(log_diag, dtype=np.float32),
        "coef1_qint8": code1[order],
        "coef2_qint8": code2[order],
        "scale1_fp16": scale1[order],
        "scale2_fp16": scale2[order],
        "intercept_fp16": intercept16[order],
    }
    fit_audit = _sanitize_fit_audit(
        raw_audit, class_count=len(requested_classes), k_shot=k_shot, trace=trace
    )
    coefficient_error = np.abs(decoded - np.asarray(coefficient, dtype=np.float32))
    fit_audit.update(
        {
            "closed_form_component_fit_count": int(len(call_records)),
            "support_center_transform_execution_count": int(len(transform_records)),
            "coefficient_quantization_error_mean": float(np.mean(coefficient_error)),
            "coefficient_quantization_error_max": float(np.max(coefficient_error)),
            "fp32_coefficient_persisted": False,
            "physical_ids_persisted": False,
        }
    )
    core = _state_core_payload(
        classes=requested_classes,
        k_shot=k_shot,
        covariance_policy=str(raw_audit["covariance_policy"]),
        arrays=arrays,
        config=config,
        support_receipt_sha256=support_receipt,
        fit_audit=fit_audit,
    )
    numeric_bytes = int(sum(value.nbytes for value in arrays.values()))
    serialized_head_bytes = _serialized_head_bytes(core, arrays)
    ground_logical_bytes = int(
        d81_scorer.ground_audit.get("ground_int8_component_logical_state_bytes", -1)
    )
    if ground_logical_bytes <= 0:
        raise D81TypedTargetStateError("D81 ground logical state byte audit missing")
    ground_serialized_bytes = int(
        config.ground_component_npz_serialized_bytes
        + config.ground_manifest_serialized_bytes
    )
    class_count = len(requested_classes)
    support_rows = len(features)
    metric_macs = int(
        metric_resource.get(
            "estimated_adaptation_macs",
            3 * FEATURE_DIM * 20 * support_rows * class_count,
        )
    )
    closed_form_macs = int(
        max(1, len(call_records)) * d42._lda_fit_macs(support_rows, class_count)
    )
    retained_rank = int(d81_scorer.ground_audit.get("d81_retained_rank", 1))
    ground_statistics_macs = int(
        d81_scorer.ground_audit.get("ground_covariance_statistics_mac_upper_bound", 0)
    )
    translation_macs = _translation_macs_upper_bound(class_count, k_shot, retained_rank)
    transient_bytes = int(
        d81_scorer.nuisance_basis_fp64.nbytes
        + d81_scorer.spectral_weights_fp64.nbytes
        + d81_scorer.ground_audit.get("transient_dequantized_ground_bytes", 0)
        + transformed.nbytes
        + np.asarray(coefficient, dtype=np.float32).nbytes
        + np.asarray(intercept, dtype=np.float32).nbytes
    )
    resource = {
        "schema": RESOURCE_SCHEMA,
        "scope": "complete_D81_target_row_fit_and_query_head",
        "support_only": True,
        "single_received_observation_per_physical_id": True,
        "optimizer_steps": 20,
        "metric_optimizer_steps": 20,
        "d81_closed_form_optimizer_steps": 0,
        "metric_fit_execution_count": 1,
        "d81_head_fit_execution_count": 1,
        "trainable_parameters": int(FEATURE_DIM * (1 + class_count)),
        "head_numeric_logical_state_bytes": numeric_bytes,
        "head_serialized_state_bytes": serialized_head_bytes,
        "ground_bundle_logical_state_bytes": ground_logical_bytes,
        "ground_bundle_serialized_state_bytes": ground_serialized_bytes,
        "logical_persistent_state_bytes_including_ground": int(numeric_bytes + len(_canonical_bytes(core)) + ground_logical_bytes),
        "serialized_persistent_state_bytes_including_ground": int(serialized_head_bytes + ground_serialized_bytes),
        "peak_state_bytes_upper_bound_including_ground": int(serialized_head_bytes + ground_logical_bytes + transient_bytes),
        "fit_mac_upper_bound": int(metric_macs + closed_form_macs + ground_statistics_macs + translation_macs),
        "metric_fit_macs": metric_macs,
        "closed_form_fit_macs_upper_bound": closed_form_macs,
        "ground_statistics_macs_upper_bound": ground_statistics_macs,
        "support_center_translation_macs_upper_bound": translation_macs,
        "query_mac_upper_bound_per_sample": int(FEATURE_DIM + 2 * FEATURE_DIM * class_count),
        "query_state_updates": 0,
        "query_batch_graph_bytes": 0,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_role_or_receiver_or_scenario_input": False,
        "query_latency_measurement_protocol": "external_wall_clock_cold_and_warm_per_sample_no_batch_amortization",
        "query_latency_value_in_deterministic_receipt": None,
        "runtime_device": str(d81_scorer.device),
    }
    # Wall-clock timing would destroy support-order receipt invariance.  Bind
    # only a deterministic measurement protocol; runners record measured values.
    resource["fit_latency_measurement_protocol"] = (
        "external_wall_clock_one_complete_row_fit_no_foldwise_metric_refit"
    )
    core = _state_core_payload(
        classes=requested_classes,
        k_shot=k_shot,
        covariance_policy=str(raw_audit["covariance_policy"]),
        arrays=arrays,
        config=config,
        support_receipt_sha256=support_receipt,
        fit_audit=fit_audit,
    )
    receipt = _canonical_sha256(_state_receipt_payload(core, resource))
    return D81TypedTargetState(
        classes=requested_classes,
        k_shot=k_shot,
        log_diag_fp32=arrays["log_diag_fp32"],
        coef1_qint8=arrays["coef1_qint8"],
        coef2_qint8=arrays["coef2_qint8"],
        scale1_fp16=arrays["scale1_fp16"],
        scale2_fp16=arrays["scale2_fp16"],
        intercept_fp16=arrays["intercept_fp16"],
        covariance_policy=str(raw_audit["covariance_policy"]),
        config=config,
        support_receipt_sha256=support_receipt,
        fit_audit=fit_audit,
        resource_audit=resource,
        state_receipt_sha256=receipt,
    )


def verify_d81_typed_target_state(
    state: D81TypedTargetState, *, verify_dependencies: bool = True
) -> bool:
    if type(state) is not D81TypedTargetState or type(state.config) is not D81TypedTargetConfig:
        return False
    try:
        if verify_dependencies:
            state.config.verify_current()
        core = _state_core_payload(
            classes=state.classes,
            k_shot=state.k_shot,
            covariance_policy=state.covariance_policy,
            arrays=state.arrays,
            config=state.config,
            support_receipt_sha256=state.support_receipt_sha256,
            fit_audit=state.fit_audit,
        )
        resource = _json_safe(state.resource_audit)
        numeric = int(sum(value.nbytes for value in state.arrays.values()))
        if (
            resource.get("schema") != RESOURCE_SCHEMA
            or int(resource.get("optimizer_steps", -1)) != 20
            or int(resource.get("metric_fit_execution_count", -1)) != 1
            or int(resource.get("query_state_updates", -1)) != 0
            or int(resource.get("head_numeric_logical_state_bytes", -1)) != numeric
            or int(resource.get("head_serialized_state_bytes", -1))
            != _serialized_head_bytes(core, state.arrays)
            or int(resource.get("ground_bundle_serialized_state_bytes", -1))
            != state.config.ground_component_npz_serialized_bytes
            + state.config.ground_manifest_serialized_bytes
        ):
            return False
        expected = _canonical_sha256(_state_receipt_payload(core, resource))
        return expected == state.state_receipt_sha256
    except (D81TypedTargetStateError, KeyError, TypeError, ValueError):
        return False


def score_d81_typed_target_raw_logits(
    state: D81TypedTargetState, query_features: np.ndarray
) -> np.ndarray:
    """Score independent raw concat288 queries with an exact typed D81 state."""

    if type(state) is not D81TypedTargetState or not verify_d81_typed_target_state(state):
        raise D81TypedTargetStateError("D81 typed state receipt/dependency verification failed")
    query = np.asarray(query_features)
    if (
        query.dtype != np.float32
        or query.ndim != 2
        or query.shape[1] != FEATURE_DIM
        or len(query) < 1
        or not np.isfinite(query).all()
    ):
        raise D81TypedTargetStateError("D81 query must be finite float32 [N,288]")
    registered = raw_concat_to_d81_registered_feature(query)
    transformed = d42._transform(registered, state.log_diag_fp32)
    coefficient = d42._decode_coefficients(
        state.coef1_qint8,
        state.coef2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
    )
    rows = [
        np.asarray(row @ coefficient.T + state.intercept_fp16.astype(np.float32), dtype=np.float32)
        for row in transformed
    ]
    result = _readonly(np.stack(rows), np.float32)
    if result.shape != (len(query), len(state.classes)) or not np.isfinite(result).all():
        raise D81TypedTargetStateError("D81 typed raw logit output drift")
    if not verify_d81_typed_target_state(state):
        raise D81TypedTargetStateError("D81 typed state mutated during query scoring")
    return result


__all__ = [
    "ALLOWED_K_SHOT",
    "CONFIG_SCHEMA",
    "DEPLOYMENT_STATUS",
    "D81TypedTargetConfig",
    "D81TypedTargetState",
    "D81TypedTargetStateError",
    "RESOURCE_SCHEMA",
    "SCHEMA",
    "fit_d81_typed_target_state",
    "score_d81_typed_target_raw_logits",
    "verify_d81_typed_target_state",
]
