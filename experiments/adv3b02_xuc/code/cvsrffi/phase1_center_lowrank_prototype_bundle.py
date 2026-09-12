"""Phase1 center-domain plus low-rank int8 prototype deployment component.

This module is the offline v1-dense-to-v2 compressor and the strict Phase2
reader.  It accepts only many-to-one aggregate v1 centroids and an optional
already-aggregated P90 radius matrix.  It has no source dataset/sample path,
query, role, quota, member-count, or sample-level feature interface.

The Phase2 object can dequantize the center ``[C,P]`` or reconstruct exactly
one requested domain ``[C,P]``.  It intentionally exposes no dense-bank
dequantization/export API and never caches a reconstructed float bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "int8_domain_class_center_lowrank_residual_radius_v2"
NPZ_NAME = "int8_domain_class_center_lowrank_residual_radius_v2.npz"
REPRESENTATION = "domain_class_center_lowrank_residual_radius_v2"
MANIFEST_NAME = "manifest.json"
MANIFEST_SHA_NAME = "manifest.sha256"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
FEATURE_DIM = 160
RESIDUAL_RANK = 3
ZERO_VECTOR_SCALE = np.float16(1.0)
SVD_SIGN_SCHEMA = "largest_abs_basis_entry_positive_lowest_index_tie_v1"
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_v1"
RADIUS_DEFINITION = "p90_cosine_distance_to_phase1_domain_class_centroid"
RADIUS_GENERATION_PROOF_SCHEMA = "phase1_aggregate_radius_generation_proof_v1"
PENDING_OUTER_JOINT_SEAL = "PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL"

V1_ALLOWED_MEMBERS = {
    "domain_class_q",
    "domain_class_scale",
    "domain_class_mask",
    "domain_registry",
    "class_registry",
    "feature_schema",
}

ALLOWED_NPZ_MEMBERS = {
    "schema",
    "feature_schema",
    "residual_rank",
    "center_domain_handle",
    "domain_registry",
    "residual_domain_registry",
    "class_registry",
    "core_q",
    "core_scale",
    "residual_basis_q",
    "residual_basis_scale",
    "residual_coeff_q",
    "residual_coeff_scale",
    "radius_q",
    "radius_scale",
}

BASE_MANIFEST_FIELDS = {
    "schema",
    "feature_schema",
    "feature_dim",
    "residual_rank",
    "center_domain_handle",
    "domain_count",
    "class_count",
    "checkpoint_sha256",
    "class_handle_binding_sha256",
    "v1_component_sha256",
    "phase1_stream_sha256",
    "radius_generation_proof_sha256",
    "radius_generation_proof_schema",
    "generation_code_sha256",
    "generation_config_sha256",
    "registry_sha256",
    "provenance_status",
    "component_state",
    "outer_bundle_signature_required",
    "formal_phase2_eligible",
    "radius_provenance",
    "radius_definition",
    "svd_sign_canonicalization",
    "rounding_rule",
    "member_allowlist",
    "npz_member_allowlist",
    "quantization",
    "resource_audit",
    "phase2_authorized_phase1_model_knowledge_policy",
    "phase2_phase1_prototype_generation_stage",
    "phase2_phase1_prototype_payload",
    "phase2_phase1_prototype_representation",
    "phase2_phase1_prototype_center_domain_policy",
    "phase2_phase1_prototype_center_domain_target_conditioned",
    "phase2_phase1_prototype_center_domain_query_conditioned",
    "phase2_phase1_prototype_residual_rank",
    "phase2_phase1_prototype_radius_definition",
    "phase2_phase1_prototype_radius_member_values_exposed",
    "phase2_phase1_prototype_dense_bank_persistent",
    "phase2_phase1_prototype_dequantized_persistence",
    "phase2_phase1_prototype_dequantized_export",
    "phase2_phase1_prototype_component_immutable",
    "phase2_phase1_prototype_update_access",
    "phase2_phase1_prototype_offset_update_access",
    "phase2_phase1_prototype_member_or_exemplar_access",
    "phase2_phase1_prototype_sample_reconstruction_access",
    "phase2_nonbundle_source_artifact_access",
}
FINAL_MANIFEST_FIELDS = BASE_MANIFEST_FIELDS | {
    "component_npz_sha256",
    "serialized_component_bytes",
    "pre_sign_content_root_sha256",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return normalized


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def v1_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    """Hash the actual aggregate v1 payload rather than trusting caller metadata."""

    if set(payload) != V1_ALLOWED_MEMBERS:
        raise ValueError("v1 aggregate member allowlist mismatch")
    digest = hashlib.sha256()
    for key in sorted(payload):
        encoded_key = key.encode("utf-8")
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
        digest.update(bytes.fromhex(_array_sha256(np.asarray(payload[key]))))
    return digest.hexdigest()


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _scalar_string(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a scalar non-object string")
    return str(array.item())


def _string_registry(value: np.ndarray, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a one-dimensional non-object string array")
    registry = tuple(str(item) for item in array.tolist())
    if not registry or len(set(registry)) != len(registry) or any(not item for item in registry):
        raise ValueError(f"{field} entries must be non-empty and unique")
    return registry


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    if not np.isfinite(rows).all() or bool(np.any(norm <= 1.0e-12)):
        raise ValueError("prototype vectors must be finite and non-zero")
    return rows / norm


def _quantize_vectors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 quantization with one FP16 scale per last-axis vector."""

    vectors = np.asarray(value, dtype=np.float32)
    if vectors.ndim < 1 or not np.isfinite(vectors).all():
        raise ValueError("quantized vectors must be finite")
    max_abs = np.max(np.abs(vectors), axis=-1)
    scale32 = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise ValueError("FP16 quantization scale is not finite and positive")
    q = np.clip(np.rint(vectors / scale32[..., None]), -127, 127).astype(np.int8)
    if bool(np.any(q == -128)):
        raise ValueError("symmetric int8 quantization emitted forbidden -128")
    return q, scale16


def _quantize_radius(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(radius, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all() or bool(np.any(value < 0.0)):
        raise ValueError("P90 cosine-distance radius must be finite non-negative [D,C]")
    max_by_class = np.max(value, axis=0)
    scale32 = np.where(max_by_class > 0.0, max_by_class / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise ValueError("radius FP16 scale is not finite and positive")
    q = np.clip(np.rint(value / scale32[None, :]), 0, 127).astype(np.int8)
    if bool(np.any(q < 0)) or bool(np.any(q == -128)):
        raise ValueError("radius_q must stay in [0,127]")
    return q, scale16


def _validate_v1_dense_payload(
    payload: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    if set(payload) != V1_ALLOWED_MEMBERS:
        raise ValueError("v1 payload member set does not match strict allowlist")
    q = np.asarray(payload["domain_class_q"])
    scale = np.asarray(payload["domain_class_scale"])
    mask = np.asarray(payload["domain_class_mask"])
    if q.dtype != np.int8 or q.ndim != 3 or q.shape[2] != FEATURE_DIM:
        raise ValueError(f"v1 domain_class_q must be int8[D,C,{FEATURE_DIM}]")
    if scale.dtype != np.float16 or scale.shape != q.shape[:2]:
        raise ValueError("v1 domain_class_scale must be float16[D,C]")
    if mask.dtype != np.uint8 or mask.shape != q.shape[:2]:
        raise ValueError("v1 domain_class_mask must be uint8[D,C]")
    if bool(np.any((mask != 0) & (mask != 1))):
        raise ValueError("v1 domain_class_mask must be binary")
    if bool(np.any(q[mask == 0] != 0)) or bool(np.any(q == -128)):
        raise ValueError("v1 int8 payload violates inactive-slot or symmetric range rules")
    if not np.isfinite(scale).all() or bool(np.any(scale <= 0.0)):
        raise ValueError("v1 scales must be finite and positive")
    if _scalar_string(payload["feature_schema"], "feature_schema") != FEATURE_SCHEMA:
        raise ValueError("v1 feature schema mismatch")

    raw_domains = np.asarray(payload["domain_registry"])
    if raw_domains.ndim != 1 or len(raw_domains) != q.shape[0] or raw_domains.dtype.kind == "O":
        raise ValueError("v1 domain_registry shape or dtype mismatch")
    domains = tuple(str(item) for item in raw_domains.tolist())
    classes = _string_registry(payload["class_registry"], "class_registry")
    if len(classes) != q.shape[1] or len(set(domains)) != len(domains):
        raise ValueError("v1 registry does not match tensor shape or uniqueness")

    full_rows = np.all(mask == 1, axis=1)
    empty_rows = np.all(mask == 0, axis=1)
    if not np.all(full_rows | empty_rows):
        raise ValueError("v2 requires complete class coverage for every retained domain")
    retained = np.flatnonzero(full_rows)
    if len(retained) < RESIDUAL_RANK + 1:
        raise ValueError("v2 rank3 compression requires at least four complete domains")
    dense = q[retained].astype(np.float32) * scale[retained, :, None].astype(np.float32)
    _normalize_rows(dense)
    retained_domains = tuple(domains[index] for index in retained)
    return dense, retained_domains, classes, retained


def _global_maximin_center(dense: np.ndarray) -> int:
    normalized = _normalize_rows(dense)
    score: list[float] = []
    for domain_index in range(len(normalized)):
        same_class_cosine = np.sum(
            normalized[domain_index][None, :, :] * normalized, axis=-1
        )
        score.append(float(np.min(same_class_cosine)))
    return int(np.argmax(np.asarray(score, dtype=np.float64)))


def _canonical_rank3_svd(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    domain_count, class_count, feature_dim = residual.shape
    if domain_count < RESIDUAL_RANK or feature_dim < RESIDUAL_RANK:
        raise ValueError("residual matrix is too small for fixed rank3 SVD")
    basis = np.empty((class_count, RESIDUAL_RANK, feature_dim), dtype=np.float32)
    coeff = np.empty((domain_count, class_count, RESIDUAL_RANK), dtype=np.float32)
    for class_index in range(class_count):
        u, singular, vt = np.linalg.svd(
            residual[:, class_index, :].astype(np.float64), full_matrices=False
        )
        selected_basis = vt[:RESIDUAL_RANK].copy()
        selected_coeff = u[:, :RESIDUAL_RANK] * singular[:RESIDUAL_RANK][None, :]
        for rank_index in range(RESIDUAL_RANK):
            pivot = int(np.argmax(np.abs(selected_basis[rank_index])))
            if selected_basis[rank_index, pivot] < 0.0:
                selected_basis[rank_index] *= -1.0
                selected_coeff[:, rank_index] *= -1.0
        basis[class_index] = selected_basis.astype(np.float32)
        coeff[:, class_index] = selected_coeff.astype(np.float32)
    return basis, coeff


def _registry_sha256(
    domains: Sequence[str], classes: Sequence[str], center: str
) -> str:
    return _canonical_sha256(
        {
            "domain_registry": list(domains),
            "class_registry": list(classes),
            "center_domain_handle": str(center),
            "feature_schema": FEATURE_SCHEMA,
            "residual_rank": RESIDUAL_RANK,
        }
    )


def radius_generation_proof_sha256(
    v1_payload: Mapping[str, np.ndarray],
    radius_p90_cosine_distance: np.ndarray,
    *,
    phase1_stream_sha256: str,
    checkpoint_sha256: str,
    class_handle_binding_sha256: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
) -> str:
    """Bind radius generation to actual aggregates and immutable provenance.

    This is a pre-sign content proof.  It is not a signature and becomes
    deployable only when an outer checkpoint+component+registry bundle signs
    the resulting component content root.
    """

    dense, domains, classes, retained = _validate_v1_dense_payload(v1_payload)
    center_index = _global_maximin_center(dense)
    center = domains[center_index]
    raw_radius = np.asarray(radius_p90_cosine_distance, dtype=np.float32)
    original_shape = np.asarray(v1_payload["domain_class_q"]).shape[:2]
    if raw_radius.shape != original_shape:
        raise ValueError("radius matrix must match original v1 [D,C] registry")
    if not np.isfinite(raw_radius).all() or bool(np.any(raw_radius < 0.0)):
        raise ValueError("P90 cosine-distance radius must be finite non-negative [D,C]")
    retained_radius = np.ascontiguousarray(raw_radius[retained], dtype=np.float32)
    return _canonical_sha256(
        {
            "schema": RADIUS_GENERATION_PROOF_SCHEMA,
            "phase1_stream_sha256": _validate_sha256(
                phase1_stream_sha256, "phase1_stream_sha256"
            ),
            "v1_component_sha256": v1_payload_sha256(v1_payload),
            "registry_sha256": _registry_sha256(domains, classes, center),
            "radius_matrix_sha256": _array_sha256(retained_radius),
            "radius_definition": RADIUS_DEFINITION,
            "checkpoint_sha256": _validate_sha256(
                checkpoint_sha256, "checkpoint_sha256"
            ),
            "class_handle_binding_sha256": _validate_sha256(
                class_handle_binding_sha256, "class_handle_binding_sha256"
            ),
            "generation_code_sha256": _validate_sha256(
                generation_code_sha256, "generation_code_sha256"
            ),
            "generation_config_sha256": _validate_sha256(
                generation_config_sha256, "generation_config_sha256"
            ),
        }
    )


def _numeric_resource_audit(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    core_q = int(np.asarray(payload["core_q"]).nbytes)
    core_scale = int(np.asarray(payload["core_scale"]).nbytes)
    basis_q = int(np.asarray(payload["residual_basis_q"]).nbytes)
    basis_scale = int(np.asarray(payload["residual_basis_scale"]).nbytes)
    coeff_q = int(np.asarray(payload["residual_coeff_q"]).nbytes)
    coeff_scale = int(np.asarray(payload["residual_coeff_scale"]).nbytes)
    radius_q = int(np.asarray(payload["radius_q"]).nbytes)
    radius_scale = int(np.asarray(payload["radius_scale"]).nbytes)
    numeric_keys = {
        "core_q",
        "core_scale",
        "residual_basis_q",
        "residual_basis_scale",
        "residual_coeff_q",
        "residual_coeff_scale",
        "radius_q",
        "radius_scale",
    }
    registry_schema = sum(
        int(np.asarray(value).nbytes)
        for key, value in payload.items()
        if key not in numeric_keys
    )
    d, c = np.asarray(payload["radius_q"]).shape
    p = int(np.asarray(payload["core_q"]).shape[1])
    r = RESIDUAL_RANK
    direction = core_q + core_scale + basis_q + basis_scale + coeff_q + coeff_scale
    radius = radius_q + radius_scale
    return {
        "core_q_bytes": core_q,
        "core_scale_bytes": core_scale,
        "residual_basis_q_bytes": basis_q,
        "residual_basis_scale_bytes": basis_scale,
        "residual_coeff_q_bytes": coeff_q,
        "residual_coeff_scale_bytes": coeff_scale,
        "direction_numeric_payload_bytes": direction,
        "radius_q_bytes": radius_q,
        "radius_scale_bytes": radius_scale,
        "radius_numeric_payload_bytes": radius,
        "compressed_numeric_payload_bytes": direction + radius,
        "registry_schema_bytes": registry_schema,
        "logical_deployment_state_bytes": direction + radius + registry_schema,
        "single_class_prototype_reconstruction_macs": r * p,
        "single_domain_all_class_reconstruction_macs": c * r * p,
        "all_residual_domain_enrollment_reconstruction_macs": (d - 1) * c * r * p,
        "center_only_reconstruction_macs": 0,
        "temporary_single_class_reconstruction_peak_bytes": 4 * (2 * p + r * p + r),
        "temporary_single_domain_reconstruction_peak_bytes": 4
        * (2 * c * p + c * r * p + c * r),
        "persistent_dense_float_bank_bytes": 0,
        "target_prototype_state_bytes": 0,
        "adapter_state_bytes": 0,
        "per_query_extra_reconstruction_macs": 0,
    }


def compress_v1_dense_component(
    v1_payload: Mapping[str, np.ndarray],
    *,
    radius_p90_cosine_distance: np.ndarray | None,
    formal_phase2_eligible: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compress an aggregate-only v1 dense component into fixed-rank v2.

    The radius argument must already be the Phase1/offline P90 member-to-
    centroid cosine distance matrix.  This API deliberately accepts no members,
    sample features, sample identifiers, counts, or source paths.
    """

    dense, domains, classes, retained = _validate_v1_dense_payload(v1_payload)
    center_index = _global_maximin_center(dense)
    center_handle = domains[center_index]
    residual_indices = [index for index in range(len(domains)) if index != center_index]
    core = dense[center_index]
    residual = dense[residual_indices] - core[None, :, :]
    basis, coeff = _canonical_rank3_svd(residual)

    core_q, core_scale = _quantize_vectors(core)
    basis_q, basis_scale = _quantize_vectors(basis)
    coeff_q, coeff_scale = _quantize_vectors(coeff)

    if radius_p90_cosine_distance is None:
        if formal_phase2_eligible:
            raise ValueError("formal v2 component requires offline aggregated P90 radius")
        radius = np.zeros((len(domains), len(classes)), dtype=np.float32)
        radius_q = np.zeros(radius.shape, dtype=np.int8)
        radius_scale = np.full(len(classes), ZERO_VECTOR_SCALE, dtype=np.float16)
        radius_provenance = "radius_provenance_missing_direction_only_development"
    else:
        raw_radius = np.asarray(radius_p90_cosine_distance, dtype=np.float32)
        v1_d = np.asarray(v1_payload["domain_class_q"]).shape[0]
        if raw_radius.shape != (v1_d, len(classes)):
            raise ValueError("radius matrix must match original v1 [D,C] registry")
        if not np.isfinite(raw_radius).all() or bool(np.any(raw_radius < 0.0)):
            raise ValueError("P90 cosine-distance radius must be finite non-negative [D,C]")
        radius = raw_radius[retained]
        radius_q, radius_scale = _quantize_radius(radius)
        radius_provenance = "phase1_offline_aggregate_p90_cosine_distance_v1"

    domain_array = np.asarray(domains, dtype=np.str_)
    residual_domains = np.asarray(
        [domains[index] for index in residual_indices], dtype=np.str_
    )
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
        "residual_rank": np.asarray(RESIDUAL_RANK, dtype=np.int16),
        "center_domain_handle": np.asarray(center_handle, dtype=np.str_),
        "domain_registry": domain_array,
        "residual_domain_registry": residual_domains,
        "class_registry": np.asarray(classes, dtype=np.str_),
        "core_q": core_q,
        "core_scale": core_scale,
        "residual_basis_q": basis_q,
        "residual_basis_scale": basis_scale,
        "residual_coeff_q": coeff_q,
        "residual_coeff_scale": coeff_scale,
        "radius_q": radius_q,
        "radius_scale": radius_scale,
    }

    core_hat = core_q.astype(np.float32) * core_scale[:, None].astype(np.float32)
    basis_hat = basis_q.astype(np.float32) * basis_scale[..., None].astype(np.float32)
    coeff_hat = coeff_q.astype(np.float32) * coeff_scale[..., None].astype(np.float32)
    reconstructed_residual = core_hat[None, :, :] + np.einsum(
        "dcr,crp->dcp", coeff_hat, basis_hat, optimize=True
    )
    reconstructed = np.empty_like(dense)
    reconstructed[center_index] = core_hat
    reconstructed[residual_indices] = reconstructed_residual
    reference_unit = _normalize_rows(dense)
    reconstructed_unit = _normalize_rows(reconstructed)
    cosine = np.sum(reference_unit * reconstructed_unit, axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    radius_hat = radius_q.astype(np.float32) * radius_scale[None, :].astype(np.float32)
    resource = _numeric_resource_audit(payload)
    audit: dict[str, Any] = {
        **resource,
        "center_domain_handle": center_handle,
        "center_domain_index_in_retained_registry": center_index,
        "center_domain_policy": "phase1_offline_global_maximin_fixed_before_target_access",
        "residual_rank": RESIDUAL_RANK,
        "svd_sign_canonicalization": SVD_SIGN_SCHEMA,
        "rounding_rule": ROUNDING_SCHEMA,
        "radius_provenance": radius_provenance,
        "mean_reconstruction_cosine": float(np.mean(cosine)),
        "min_reconstruction_cosine": float(np.min(cosine)),
        "mean_reconstruction_angle_deg": float(np.mean(angle)),
        "max_reconstruction_angle_deg": float(np.max(angle)),
        "reconstruction_rmse": float(np.sqrt(np.mean((dense - reconstructed) ** 2))),
        "radius_max_abs_error": float(np.max(np.abs(radius - radius_hat))),
        "radius_mean_abs_error": float(np.mean(np.abs(radius - radius_hat))),
    }
    _validate_payload(payload)
    return payload, audit


def _validate_payload(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(payload) != ALLOWED_NPZ_MEMBERS:
        raise ValueError("v2 payload member set does not match strict allowlist")
    if _scalar_string(payload["schema"], "schema") != SCHEMA:
        raise ValueError("unexpected v2 prototype schema")
    if _scalar_string(payload["feature_schema"], "feature_schema") != FEATURE_SCHEMA:
        raise ValueError("v2 feature schema mismatch")
    rank = np.asarray(payload["residual_rank"])
    if rank.shape != () or rank.dtype != np.int16 or int(rank) != RESIDUAL_RANK:
        raise ValueError("v2 residual rank must be fixed int16 rank3")
    domains = _string_registry(payload["domain_registry"], "domain_registry")
    residual_domains = _string_registry(
        payload["residual_domain_registry"], "residual_domain_registry"
    )
    classes = _string_registry(payload["class_registry"], "class_registry")
    center = _scalar_string(payload["center_domain_handle"], "center_domain_handle")
    if center not in domains:
        raise ValueError("center domain handle is absent from domain registry")
    if tuple(item for item in domains if item != center) != residual_domains:
        raise ValueError("residual domain registry must equal domain registry minus center")
    d, c = len(domains), len(classes)
    core_q = np.asarray(payload["core_q"])
    core_scale = np.asarray(payload["core_scale"])
    basis_q = np.asarray(payload["residual_basis_q"])
    basis_scale = np.asarray(payload["residual_basis_scale"])
    coeff_q = np.asarray(payload["residual_coeff_q"])
    coeff_scale = np.asarray(payload["residual_coeff_scale"])
    radius_q = np.asarray(payload["radius_q"])
    radius_scale = np.asarray(payload["radius_scale"])
    expected = {
        "core_q": (core_q, np.int8, (c, FEATURE_DIM)),
        "core_scale": (core_scale, np.float16, (c,)),
        "residual_basis_q": (
            basis_q,
            np.int8,
            (c, RESIDUAL_RANK, FEATURE_DIM),
        ),
        "residual_basis_scale": (
            basis_scale,
            np.float16,
            (c, RESIDUAL_RANK),
        ),
        "residual_coeff_q": (
            coeff_q,
            np.int8,
            (d - 1, c, RESIDUAL_RANK),
        ),
        "residual_coeff_scale": (
            coeff_scale,
            np.float16,
            (d - 1, c),
        ),
        "radius_q": (radius_q, np.int8, (d, c)),
        "radius_scale": (radius_scale, np.float16, (c,)),
    }
    for field, (array, dtype, shape) in expected.items():
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"{field} dtype or shape mismatch")
    for field, array in (
        ("core_q", core_q),
        ("residual_basis_q", basis_q),
        ("residual_coeff_q", coeff_q),
    ):
        if bool(np.any(array == -128)):
            raise ValueError(f"{field} contains forbidden -128")
    if bool(np.any(radius_q < 0)) or bool(np.any(radius_q > 127)):
        raise ValueError("radius_q must be non-negative int8 [0,127]")
    for field, array in (
        ("core_scale", core_scale),
        ("residual_basis_scale", basis_scale),
        ("residual_coeff_scale", coeff_scale),
        ("radius_scale", radius_scale),
    ):
        if not np.isfinite(array).all() or bool(np.any(array <= 0.0)):
            raise ValueError(f"{field} must be finite positive FP16")
    return {
        "domains": domains,
        "residual_domains": residual_domains,
        "classes": classes,
        "center": center,
        "resource_audit": _numeric_resource_audit(payload),
    }


def build_center_lowrank_component(
    v1_payload: Mapping[str, np.ndarray],
    *,
    radius_p90_cosine_distance: np.ndarray | None,
    phase1_stream_sha256: str,
    radius_generation_proof_sha256_value: str,
    checkpoint_sha256: str,
    class_handle_binding_sha256: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
    provenance_status: str,
    formal_phase2_eligible: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if formal_phase2_eligible:
        raise ValueError(
            "standalone Phase1 component cannot be formally Phase2 eligible before outer joint seal"
        )
    if radius_p90_cosine_distance is None:
        raise ValueError("pending outer-seal component requires aggregated P90 radius")
    stream_sha = _validate_sha256(phase1_stream_sha256, "phase1_stream_sha256")
    checkpoint_sha = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    binding_sha = _validate_sha256(
        class_handle_binding_sha256, "class_handle_binding_sha256"
    )
    code_sha = _validate_sha256(generation_code_sha256, "generation_code_sha256")
    config_sha = _validate_sha256(
        generation_config_sha256, "generation_config_sha256"
    )
    expected_proof = radius_generation_proof_sha256(
        v1_payload,
        radius_p90_cosine_distance,
        phase1_stream_sha256=stream_sha,
        checkpoint_sha256=checkpoint_sha,
        class_handle_binding_sha256=binding_sha,
        generation_code_sha256=code_sha,
        generation_config_sha256=config_sha,
    )
    supplied_proof = _validate_sha256(
        radius_generation_proof_sha256_value,
        "radius_generation_proof_sha256",
    )
    if supplied_proof != expected_proof:
        raise ValueError("radius generation proof does not bind the supplied aggregate radius")
    payload, audit = compress_v1_dense_component(
        v1_payload,
        radius_p90_cosine_distance=radius_p90_cosine_distance,
        formal_phase2_eligible=False,
    )
    details = _validate_payload(payload)
    actual_v1_sha = v1_payload_sha256(v1_payload)
    hashes = {
        "checkpoint_sha256": checkpoint_sha,
        "class_handle_binding_sha256": binding_sha,
        "v1_component_sha256": actual_v1_sha,
        "phase1_stream_sha256": stream_sha,
        "radius_generation_proof_sha256": supplied_proof,
        "generation_code_sha256": code_sha,
        "generation_config_sha256": config_sha,
    }
    radius_provenance = str(audit["radius_provenance"])
    if formal_phase2_eligible and "missing" in radius_provenance:
        raise ValueError("formal v2 component cannot omit radius provenance")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "residual_rank": RESIDUAL_RANK,
        "center_domain_handle": details["center"],
        "domain_count": len(details["domains"]),
        "class_count": len(details["classes"]),
        **hashes,
        "registry_sha256": _registry_sha256(
            details["domains"], details["classes"], details["center"]
        ),
        "provenance_status": str(provenance_status),
        "component_state": PENDING_OUTER_JOINT_SEAL,
        "outer_bundle_signature_required": True,
        "formal_phase2_eligible": False,
        "radius_generation_proof_schema": RADIUS_GENERATION_PROOF_SCHEMA,
        "radius_provenance": radius_provenance,
        "radius_definition": RADIUS_DEFINITION,
        "svd_sign_canonicalization": SVD_SIGN_SCHEMA,
        "rounding_rule": ROUNDING_SCHEMA,
        "member_allowlist": [NPZ_NAME],
        "npz_member_allowlist": sorted(ALLOWED_NPZ_MEMBERS),
        "quantization": {
            "dtype": "int8",
            "scale_dtype": "float16",
            "mode": "symmetric_per_vector",
            "qmin": -127,
            "qmax": 127,
            "radius_qmin": 0,
            "radius_qmax": 127,
            "zero_vector_scale": float(ZERO_VECTOR_SCALE),
        },
        "resource_audit": audit,
        "phase2_authorized_phase1_model_knowledge_policy": SCHEMA,
        "phase2_phase1_prototype_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_prototype_payload": "int8_center_core_plus_int8_lowrank_domain_class_residual_plus_int8_radius_fp16_scales_registry_only",
        "phase2_phase1_prototype_representation": REPRESENTATION,
        "phase2_phase1_prototype_center_domain_policy": "phase1_offline_global_maximin_fixed_before_target_access",
        "phase2_phase1_prototype_center_domain_target_conditioned": False,
        "phase2_phase1_prototype_center_domain_query_conditioned": False,
        "phase2_phase1_prototype_residual_rank": RESIDUAL_RANK,
        "phase2_phase1_prototype_radius_definition": RADIUS_DEFINITION,
        "phase2_phase1_prototype_radius_member_values_exposed": False,
        "phase2_phase1_prototype_dense_bank_persistent": False,
        "phase2_phase1_prototype_dequantized_persistence": False,
        "phase2_phase1_prototype_dequantized_export": False,
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_offset_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "phase2_phase1_prototype_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
    }
    if set(manifest) != BASE_MANIFEST_FIELDS:
        raise AssertionError("internal manifest field drift")
    return payload, manifest


def _pre_sign_content_root(manifest: Mapping[str, Any], npz_sha256: str) -> str:
    return _canonical_sha256(
        {
            "schema": manifest["schema"],
            "component_state": manifest["component_state"],
            "outer_bundle_signature_required": manifest[
                "outer_bundle_signature_required"
            ],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "class_handle_binding_sha256": manifest["class_handle_binding_sha256"],
            "v1_component_sha256": manifest["v1_component_sha256"],
            "phase1_stream_sha256": manifest["phase1_stream_sha256"],
            "radius_generation_proof_sha256": manifest[
                "radius_generation_proof_sha256"
            ],
            "component_npz_sha256": npz_sha256,
            "registry_sha256": manifest["registry_sha256"],
            "generation_code_sha256": manifest["generation_code_sha256"],
            "generation_config_sha256": manifest["generation_config_sha256"],
        }
    )


def save_center_lowrank_component(
    output_dir: str | Path,
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    details = _validate_payload(payload)
    if set(manifest) != BASE_MANIFEST_FIELDS:
        raise ValueError("base manifest field set mismatch")
    if manifest.get("schema") != SCHEMA or int(manifest.get("residual_rank", -1)) != RESIDUAL_RANK:
        raise ValueError("base manifest schema or rank mismatch")
    if manifest.get("center_domain_handle") != details["center"]:
        raise ValueError("manifest center handle mismatch")
    expected_registry = _registry_sha256(
        details["domains"], details["classes"], details["center"]
    )
    if manifest.get("registry_sha256") != expected_registry:
        raise ValueError("manifest registry SHA256 mismatch")
    if bool(manifest.get("formal_phase2_eligible")):
        raise ValueError("standalone component must remain formally Phase2 ineligible")
    if manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL:
        raise ValueError("standalone component state must be pending outer joint seal")
    if manifest.get("outer_bundle_signature_required") is not True:
        raise ValueError("standalone component must require an outer bundle signature")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    npz_sha = sha256_file(npz_path)
    final = dict(manifest)
    final["component_npz_sha256"] = npz_sha
    final["serialized_component_bytes"] = int(npz_path.stat().st_size)
    final["pre_sign_content_root_sha256"] = _pre_sign_content_root(final, npz_sha)
    final["resource_audit"] = dict(final["resource_audit"])
    final["resource_audit"]["serialized_component_bytes"] = int(npz_path.stat().st_size)
    if set(final) != FINAL_MANIFEST_FIELDS:
        raise AssertionError("internal final manifest field drift")
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = sha256_file(manifest_path)
    (root / MANIFEST_SHA_NAME).write_text(
        f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    validate_center_lowrank_component(root)
    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "component_npz_sha256": npz_sha,
        "manifest_sha256": manifest_sha,
        "pre_sign_content_root_sha256": final["pre_sign_content_root_sha256"],
    }


def validate_center_lowrank_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_class_handle_binding_sha256: str | None = None,
    expected_pre_sign_content_root_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(component_dir)
    actual_members = {item.name for item in root.iterdir()} if root.is_dir() else set()
    if actual_members != {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}:
        raise ValueError("component directory member allowlist mismatch")
    manifest_path = root / MANIFEST_NAME
    sha_line = (root / MANIFEST_SHA_NAME).read_text(encoding="ascii")
    expected_line = f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n"
    if sha_line != expected_line:
        raise ValueError("manifest SHA256 sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != FINAL_MANIFEST_FIELDS:
        raise ValueError("final manifest field set mismatch")
    if manifest.get("schema") != SCHEMA or int(manifest.get("residual_rank", -1)) != RESIDUAL_RANK:
        raise ValueError("unexpected v2 schema or residual rank")
    if manifest.get("feature_schema") != FEATURE_SCHEMA or int(
        manifest.get("feature_dim", -1)
    ) != FEATURE_DIM:
        raise ValueError("manifest feature schema mismatch")
    if manifest.get("member_allowlist") != [NPZ_NAME]:
        raise ValueError("component member allowlist mismatch")
    if manifest.get("npz_member_allowlist") != sorted(ALLOWED_NPZ_MEMBERS):
        raise ValueError("NPZ member allowlist mismatch")
    for field in (
        "checkpoint_sha256",
        "class_handle_binding_sha256",
        "v1_component_sha256",
        "phase1_stream_sha256",
        "radius_generation_proof_sha256",
        "generation_code_sha256",
        "generation_config_sha256",
        "registry_sha256",
        "component_npz_sha256",
        "pre_sign_content_root_sha256",
    ):
        _validate_sha256(str(manifest.get(field, "")), field)
    if manifest.get("svd_sign_canonicalization") != SVD_SIGN_SCHEMA:
        raise ValueError("SVD sign canonicalization schema mismatch")
    if manifest.get("rounding_rule") != ROUNDING_SCHEMA:
        raise ValueError("quantization rounding schema mismatch")
    if manifest.get("radius_definition") != RADIUS_DEFINITION:
        raise ValueError("radius definition mismatch")
    expected_quantization = {
        "dtype": "int8",
        "scale_dtype": "float16",
        "mode": "symmetric_per_vector",
        "qmin": -127,
        "qmax": 127,
        "radius_qmin": 0,
        "radius_qmax": 127,
        "zero_vector_scale": float(ZERO_VECTOR_SCALE),
    }
    if manifest.get("quantization") != expected_quantization:
        raise ValueError("quantization schema mismatch")
    expected_protocol = {
        "phase2_authorized_phase1_model_knowledge_policy": SCHEMA,
        "phase2_phase1_prototype_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_prototype_payload": "int8_center_core_plus_int8_lowrank_domain_class_residual_plus_int8_radius_fp16_scales_registry_only",
        "phase2_phase1_prototype_representation": REPRESENTATION,
        "phase2_phase1_prototype_center_domain_policy": "phase1_offline_global_maximin_fixed_before_target_access",
        "phase2_phase1_prototype_center_domain_target_conditioned": False,
        "phase2_phase1_prototype_center_domain_query_conditioned": False,
        "phase2_phase1_prototype_residual_rank": RESIDUAL_RANK,
        "phase2_phase1_prototype_radius_definition": RADIUS_DEFINITION,
        "phase2_phase1_prototype_radius_member_values_exposed": False,
        "phase2_phase1_prototype_dense_bank_persistent": False,
        "phase2_phase1_prototype_dequantized_persistence": False,
        "phase2_phase1_prototype_dequantized_export": False,
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_offset_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "phase2_phase1_prototype_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
    }
    for field, value in expected_protocol.items():
        if manifest.get(field) != value:
            raise ValueError(f"protocol manifest mismatch for {field}")
    if manifest.get("formal_phase2_eligible") is not False:
        raise ValueError("standalone component must be formally Phase2 ineligible")
    if manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL:
        raise ValueError("component is not pending the required outer joint seal")
    if manifest.get("outer_bundle_signature_required") is not True:
        raise ValueError("component does not require an outer bundle signature")
    if manifest.get("radius_generation_proof_schema") != RADIUS_GENERATION_PROOF_SCHEMA:
        raise ValueError("radius generation proof schema mismatch")
    missing_radius = "missing" in str(manifest.get("radius_provenance", ""))
    if missing_radius:
        raise ValueError("pending outer-seal component must contain radius provenance")
    npz_path = root / NPZ_NAME
    npz_sha = sha256_file(npz_path)
    if manifest.get("component_npz_sha256") != npz_sha:
        raise ValueError("component NPZ SHA256 mismatch")
    if manifest.get("pre_sign_content_root_sha256") != _pre_sign_content_root(
        manifest, npz_sha
    ):
        raise ValueError("pre-sign content root SHA256 mismatch")
    for expected, field in (
        (expected_checkpoint_sha256, "checkpoint_sha256"),
        (expected_class_handle_binding_sha256, "class_handle_binding_sha256"),
        (expected_pre_sign_content_root_sha256, "pre_sign_content_root_sha256"),
    ):
        if expected is not None and manifest.get(field) != _validate_sha256(expected, field):
            raise ValueError(f"{field} binding mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != ALLOWED_NPZ_MEMBERS:
            raise ValueError("NPZ contains non-allowlisted members")
        payload = {key: np.array(arrays[key], copy=True) for key in arrays.files}
    details = _validate_payload(payload)
    if manifest.get("center_domain_handle") != details["center"]:
        raise ValueError("manifest center handle mismatch")
    if int(manifest.get("domain_count", -1)) != len(details["domains"]):
        raise ValueError("manifest domain count mismatch")
    if int(manifest.get("class_count", -1)) != len(details["classes"]):
        raise ValueError("manifest class count mismatch")
    if manifest.get("registry_sha256") != _registry_sha256(
        details["domains"], details["classes"], details["center"]
    ):
        raise ValueError("manifest registry SHA256 mismatch")
    logical = details["resource_audit"]
    recorded = manifest.get("resource_audit", {})
    for key, value in logical.items():
        if recorded.get(key) != value:
            raise ValueError(f"resource audit mismatch for {key}")
    if int(manifest.get("serialized_component_bytes", -1)) != npz_path.stat().st_size:
        raise ValueError("serialized component byte count mismatch")
    if int(recorded.get("serialized_component_bytes", -1)) != npz_path.stat().st_size:
        raise ValueError("resource serialized component byte count mismatch")
    return manifest


@dataclass(frozen=True)
class CenterLowRankPrototypeComponent:
    """Immutable compressed Phase2 view with no dense-bank export method."""

    core_q: np.ndarray
    core_scale: np.ndarray
    residual_basis_q: np.ndarray
    residual_basis_scale: np.ndarray
    residual_coeff_q: np.ndarray
    residual_coeff_scale: np.ndarray
    radius_q: np.ndarray
    radius_scale: np.ndarray
    domain_registry: tuple[str, ...]
    residual_domain_registry: tuple[str, ...]
    class_registry: tuple[str, ...]
    center_domain_handle: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload = {
            "schema": np.asarray(SCHEMA, dtype=np.str_),
            "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
            "residual_rank": np.asarray(RESIDUAL_RANK, dtype=np.int16),
            "center_domain_handle": np.asarray(self.center_domain_handle, dtype=np.str_),
            "domain_registry": np.asarray(self.domain_registry, dtype=np.str_),
            "residual_domain_registry": np.asarray(
                self.residual_domain_registry, dtype=np.str_
            ),
            "class_registry": np.asarray(self.class_registry, dtype=np.str_),
            "core_q": np.asarray(self.core_q),
            "core_scale": np.asarray(self.core_scale),
            "residual_basis_q": np.asarray(self.residual_basis_q),
            "residual_basis_scale": np.asarray(self.residual_basis_scale),
            "residual_coeff_q": np.asarray(self.residual_coeff_q),
            "residual_coeff_scale": np.asarray(self.residual_coeff_scale),
            "radius_q": np.asarray(self.radius_q),
            "radius_scale": np.asarray(self.radius_scale),
        }
        details = _validate_payload(payload)
        object.__setattr__(self, "core_q", _readonly(self.core_q, np.int8))
        object.__setattr__(self, "core_scale", _readonly(self.core_scale, np.float16))
        object.__setattr__(
            self, "residual_basis_q", _readonly(self.residual_basis_q, np.int8)
        )
        object.__setattr__(
            self,
            "residual_basis_scale",
            _readonly(self.residual_basis_scale, np.float16),
        )
        object.__setattr__(
            self, "residual_coeff_q", _readonly(self.residual_coeff_q, np.int8)
        )
        object.__setattr__(
            self,
            "residual_coeff_scale",
            _readonly(self.residual_coeff_scale, np.float16),
        )
        object.__setattr__(self, "radius_q", _readonly(self.radius_q, np.int8))
        object.__setattr__(self, "radius_scale", _readonly(self.radius_scale, np.float16))
        object.__setattr__(self, "domain_registry", details["domains"])
        object.__setattr__(self, "residual_domain_registry", details["residual_domains"])
        object.__setattr__(self, "class_registry", details["classes"])
        object.__setattr__(self, "center_domain_handle", details["center"])

    def dequantized_center(self) -> np.ndarray:
        value = self.core_q.astype(np.float32) * self.core_scale[:, None].astype(np.float32)
        return _readonly(value, np.float32)

    def reconstruct_domain(self, domain_handle: str) -> np.ndarray:
        handle = str(domain_handle)
        if handle == self.center_domain_handle:
            return self.dequantized_center()
        try:
            index = self.residual_domain_registry.index(handle)
        except ValueError as exc:
            raise ValueError("unknown pre-registered domain handle") from exc
        core = self.core_q.astype(np.float32) * self.core_scale[:, None].astype(np.float32)
        basis = self.residual_basis_q.astype(np.float32) * self.residual_basis_scale[
            ..., None
        ].astype(np.float32)
        coeff = self.residual_coeff_q[index].astype(np.float32) * self.residual_coeff_scale[
            index, :, None
        ].astype(np.float32)
        value = core + np.einsum("cr,crp->cp", coeff, basis, optimize=True)
        return _readonly(value, np.float32)

    def radius_for_domain(self, domain_handle: str) -> np.ndarray:
        handle = str(domain_handle)
        try:
            index = self.domain_registry.index(handle)
        except ValueError as exc:
            raise ValueError("unknown pre-registered domain handle") from exc
        value = self.radius_q[index].astype(np.float32) * self.radius_scale.astype(np.float32)
        return _readonly(value, np.float32)

    def resource_audit(self) -> dict[str, Any]:
        return dict(self.manifest["resource_audit"])


def load_center_lowrank_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_pre_sign_content_root_sha256: str,
    allow_pending_outer_joint_seal_development: bool = False,
) -> CenterLowRankPrototypeComponent:
    manifest = validate_center_lowrank_component(
        component_dir,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_class_handle_binding_sha256=expected_class_handle_binding_sha256,
        expected_pre_sign_content_root_sha256=expected_pre_sign_content_root_sha256,
    )
    missing_radius = "missing" in str(manifest.get("radius_provenance", ""))
    if not allow_pending_outer_joint_seal_development:
        raise ValueError(
            "component is pending outer joint seal and is not formally Phase2 eligible"
        )
    if missing_radius:
        raise ValueError("formal Phase2 load requires aggregated radius provenance")
    npz_path = Path(component_dir) / NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as arrays:
        kwargs = {
            "core_q": np.array(arrays["core_q"], copy=True),
            "core_scale": np.array(arrays["core_scale"], copy=True),
            "residual_basis_q": np.array(arrays["residual_basis_q"], copy=True),
            "residual_basis_scale": np.array(arrays["residual_basis_scale"], copy=True),
            "residual_coeff_q": np.array(arrays["residual_coeff_q"], copy=True),
            "residual_coeff_scale": np.array(arrays["residual_coeff_scale"], copy=True),
            "radius_q": np.array(arrays["radius_q"], copy=True),
            "radius_scale": np.array(arrays["radius_scale"], copy=True),
            "domain_registry": _string_registry(arrays["domain_registry"], "domain_registry"),
            "residual_domain_registry": _string_registry(
                arrays["residual_domain_registry"], "residual_domain_registry"
            ),
            "class_registry": _string_registry(arrays["class_registry"], "class_registry"),
            "center_domain_handle": _scalar_string(
                arrays["center_domain_handle"], "center_domain_handle"
            ),
            "manifest": manifest,
        }
    return CenterLowRankPrototypeComponent(**kwargs)


__all__ = [
    "ALLOWED_NPZ_MEMBERS",
    "CenterLowRankPrototypeComponent",
    "FEATURE_DIM",
    "NPZ_NAME",
    "RESIDUAL_RANK",
    "PENDING_OUTER_JOINT_SEAL",
    "RADIUS_GENERATION_PROOF_SCHEMA",
    "SCHEMA",
    "build_center_lowrank_component",
    "compress_v1_dense_component",
    "load_center_lowrank_component",
    "radius_generation_proof_sha256",
    "save_center_lowrank_component",
    "sha256_file",
    "validate_center_lowrank_component",
    "v1_payload_sha256",
]
