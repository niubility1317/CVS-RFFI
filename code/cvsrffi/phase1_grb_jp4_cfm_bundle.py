"""Immutable Phase1 knowledge for GRB-JP4-CFM-qKNN-D92/r2.

This is deliberately a new component identity.  It does not accept datasets,
paths, sample features, member identifiers, target material, or a legacy r1
component.  The only feature-bearing inputs are multi-physical aggregate
ground prototypes, aggregate receiver/day means, and the checkpoint projection
weight.  A standalone component always remains pending the existing outer
joint seal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "phase1_grb_jp4_cfm_component_v2"
COMPONENT_PROFILE = "grb_jp4_cfm_q4_multiprototype_int8_v2"
METHOD_LOCK_SCHEMA = "cvs.phase1.grb_jp4_cfm_qknn_d92_method_lock.v2"
METHOD_ID = "GRB-JP4-CFM-qKNN-D92/r2-sharedK1"
PROTOCOL_SCHEMA = "p2_min_v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
RECEIVER_DAY_MEAN_SCHEMA = (
    "ADV3B02:joint_proj.0:pre_relu:160:phase1_receiver_day_mean:v1"
)
AGGREGATION_RECEIPT_SCHEMA = (
    "cvs.phase1.ground_multiprototype_aggregation_receipt.v2"
)
GROUND_QUANTIZATION_CERTIFICATE_SCHEMA = (
    "cvs.phase1.ground_multiprototype_quantization_certificate.v2"
)
MARGIN_RECEIPT_SCHEMA = "cvs.phase1.grb_jp4_cfm_qknn_margin_receipt.v2"
NPZ_NAME = "phase1_grb_jp4_cfm_component_v2.npz"
MANIFEST_NAME = "manifest.json"
MANIFEST_SHA_NAME = "manifest.sha256"
PENDING_OUTER_JOINT_SEAL = "PENDING_OUTER_JOINT_SEAL"
FEATURE_DIM = 160
HIDDEN_DIM = 320
CLASS_COUNT = 6
RANK = 4
MAX_PROTOTYPES_PER_CLASS = 3
MIN_PHYSICAL_SAMPLES_PER_AGGREGATE = 2
DEGENERATE_RELATIVE_GAP = 1.0e-12
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_fp16_rne_v2"
SVD_SCHEMA = (
    "descending_degenerate_standard_basis_projection_mgs_maxabs_positive_v2"
)
JP4_UPDATE_FACTOR_WIRE_LIMIT_BYTES = 4096
ARM_STATE_LIMIT_BYTES = 262_144

_SOURCE_FIELDS = {
    "feature_key",
    "protocol_schema",
    "ground_multiprototypes",
    "receiver_day_mean_schema",
    "receiver_day_means",
    "receiver_day_mask",
    "receiver_day_physical_counts",
    "phase1_qknn_margin_receipt",
}
_CLASS_RECORD_FIELDS = {"class_handle", "prototypes"}
_PROTOTYPE_RECORD_FIELDS = {"vector", "aggregation_receipt"}
_AGGREGATION_RECEIPT_FIELDS = {
    "schema",
    "class_handle",
    "prototype_index",
    "distinct_physical_sample_count",
    "aggregation_radius",
    "physical_sample_commitment_sha256",
    "prototype_sha256",
    "phase1_before_target_access",
    "multi_physical_aggregation",
    "member_ids_included",
    "sample_features_included",
    "source_path_included",
}
_MARGIN_RECEIPT_FIELDS = {
    "schema",
    "target_accessed",
    "receiver_lodo",
    "pseudo_support_query_physical_id_disjoint",
    "correct_predictions_only",
    "target_query_truth_used",
    "margin_definition",
    "margin_evidence_sha256",
    "margins",
}
_METHOD_LOCK_INPUT_FIELDS = {
    "schema",
    "method_id",
    "candidate_id",
    "protocol_schema",
    "feature_schema",
    "checkpoint_sha256",
    "class_handle_binding_sha256",
    "qknn_lock_sha256_by_k",
    "rank",
    "old_class_count",
    "allowed_k",
    "ground_old_multiprototype_enabled",
    "ground_old_multiprototype_max_per_class",
    "ground_old_multiprototype_min_physical_samples",
    "ground_old_multiprototype_old_classes_only",
    "ground_prototypes_enter_qknn_bank",
    "ground_prototypes_generate_logits",
    "ground_prototypes_add_k",
    "ground_component_phase2_mutable",
    "delta_tau_source",
    "active_set_steps",
    "ridge_fraction",
    "theta_box_abs",
    "trust_divisor_squared",
    "g_denominator",
    "target25_release_authorized",
    "query_fit_access",
    "query_rows_used_for_fit",
}
_METHOD_LOCK_FIELDS = _METHOD_LOCK_INPUT_FIELDS | {"delta_q", "tau_q"}

ALLOWED_NPZ_MEMBERS = {
    "p_g_q",
    "p_g_scale",
    "p_g_weight",
    "p_g_radius",
    "p_g_mask",
    "p_g_physical_counts",
    "p_g_receipt_sha256",
    "p_g_source_prototype_sha256",
    "p_g_quantization_max_abs_error",
    "p_g_quantization_certificate_sha256",
    "l_g_q",
    "l_g_scale",
    "r_q",
    "r_scale",
    "direction_energy_a",
    "delta_q",
    "tau_q",
    "class_registry",
    "feature_schema",
    "protocol_schema",
}

BASE_MANIFEST_FIELDS = {
    "schema",
    "component_profile",
    "method_lock_schema",
    "method_lock",
    "method_lock_sha256",
    "protocol_schema",
    "feature_schema",
    "receiver_day_mean_schema",
    "feature_dim",
    "hidden_dim",
    "class_count",
    "rank",
    "max_prototypes_per_class",
    "min_physical_samples_per_aggregate",
    "ground_old_multiprototype_enabled",
    "checkpoint_sha256",
    "class_handle_binding_sha256",
    "registry_sha256",
    "source_aggregate_generation_digest_sha256",
    "margin_receipt_sha256",
    "generation_code_sha256",
    "generation_config_sha256",
    "array_sha256",
    "provenance_status",
    "component_state",
    "outer_bundle_signature_required",
    "formal_phase2_eligible",
    "target25_release_authorized",
    "svd_canonicalization",
    "rounding_rule",
    "member_allowlist",
    "npz_member_allowlist",
    "quantization",
    "resource_audit",
    "phase2_phase1_component_generation_stage",
    "phase2_phase1_component_immutable",
    "phase2_phase1_component_update_access",
    "phase2_phase1_component_member_or_exemplar_access",
    "phase2_phase1_component_sample_reconstruction_access",
    "phase2_nonbundle_source_artifact_access",
    "ground_prototypes_enter_qknn_bank",
    "ground_prototypes_generate_logits",
    "ground_prototypes_add_k",
}
FINAL_MANIFEST_FIELDS = BASE_MANIFEST_FIELDS | {
    "component_npz_sha256",
    "serialized_component_bytes",
    "pre_sign_content_root_sha256",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return result


def canonical_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _class_handle_binding_sha256(class_registry: Sequence[str]) -> str:
    handles = tuple(str(item) for item in class_registry)
    if (
        len(handles) != CLASS_COUNT
        or len(set(handles)) != CLASS_COUNT
        or any(
            not handle or any(char in handle for char in ("/", "\\", ":"))
            for handle in handles
        )
    ):
        raise ValueError("class_registry must contain six unique opaque handles")
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": "phase1_tx_class_handle_binding_v1",
                "class_id_to_handle": [
                    {"class_index": index, "class_handle": handle}
                    for index, handle in enumerate(handles)
                ],
            }
        )
    )


def class_handle_binding_sha256(class_registry: Sequence[str]) -> str:
    """Return the existing outer-bundle class/order binding digest."""

    return _class_handle_binding_sha256(class_registry)


def _as_float64(value: Any, field: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite")
    return result


def _normalize_vector(value: Any, field: str) -> np.ndarray:
    vector = _as_float64(value, field)
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(f"{field} must have shape [{FEATURE_DIM}]")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{field} must be non-zero")
    return np.asarray(vector / norm, dtype=np.float64)


def _canonicalize_sign(row: np.ndarray) -> np.ndarray:
    result = np.asarray(row, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _canonical_projector_basis(vectors: np.ndarray) -> np.ndarray:
    """Choose a deterministic basis of a degenerate right-singular subspace."""

    source = np.asarray(vectors, dtype=np.float64)
    projector = source.T @ source
    selected: list[np.ndarray] = []
    tolerance = max(projector.shape) * np.finfo(np.float64).eps * 32.0
    for coordinate in range(projector.shape[0]):
        candidate = projector[:, coordinate].copy()
        for basis in selected:
            candidate -= float(candidate @ basis) * basis
        norm = float(np.linalg.norm(candidate))
        if norm > tolerance:
            selected.append(candidate / norm)
            if len(selected) == source.shape[0]:
                break
    if len(selected) != source.shape[0]:
        raise ValueError("degenerate singular subspace canonicalization failed")
    canonical: list[np.ndarray] = []
    for row in selected:
        cleaned = _canonicalize_sign(row)
        cleaned[np.abs(cleaned) <= tolerance] = 0.0
        cleaned /= np.linalg.norm(cleaned)
        canonical.append(_canonicalize_sign(cleaned))
    return np.stack(canonical, axis=0)


def _canonical_svd_rows(
    matrix: Any, *, rank: int, columns: int, field: str
) -> tuple[np.ndarray, np.ndarray]:
    value = _as_float64(matrix, field)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{field} must be [rows,{columns}]")
    _unused, singular, vt = np.linalg.svd(value, full_matrices=False)
    tolerance = (
        max(value.shape)
        * np.finfo(np.float64).eps
        * float(singular[0] if len(singular) else 0.0)
    )
    if len(singular) < rank or float(singular[rank - 1]) <= tolerance:
        raise ValueError(f"{field} cannot provide rank {rank}")
    rows: list[np.ndarray] = []
    index = 0
    while index < rank:
        end = index + 1
        while end < len(singular):
            relative_gap = (
                float(singular[end - 1]) - float(singular[end])
            ) / max(float(singular[end - 1]), np.finfo(np.float64).tiny)
            if relative_gap > DEGENERATE_RELATIVE_GAP:
                break
            end += 1
        if end - index == 1:
            cluster = np.stack([_canonicalize_sign(vt[index])], axis=0)
        else:
            cluster = _canonical_projector_basis(vt[index:end])
        rows.extend(cluster[: max(0, rank - len(rows))])
        index = end
    result = np.stack(rows[:rank], axis=0)
    return result, np.asarray(singular[:rank], dtype=np.float64)


def _quantize_vectors(
    value: Any, field: str
) -> tuple[np.ndarray, np.ndarray]:
    vectors = _as_float64(value, field)
    if vectors.ndim != 2:
        raise ValueError(f"{field} must be [vectors,features]")
    maximum = np.max(np.abs(vectors), axis=1)
    if bool(np.any(maximum <= 0.0)):
        raise ValueError(f"{field} vectors must be non-zero")
    scale = np.asarray(maximum / 127.0, dtype=np.float16)
    if not np.isfinite(scale).all() or bool(np.any(scale <= 0.0)):
        raise ValueError(f"{field} FP16 scales must be finite positive")
    codes = np.clip(
        np.rint(vectors / scale.astype(np.float64)[:, None]), -127, 127
    ).astype(np.int8)
    if bool(np.any(codes == -128)):
        raise ValueError(f"{field} emitted forbidden INT8 code -128")
    return codes, scale


def _validate_method_lock_input(
    method_lock: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    class_binding_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(method_lock, Mapping)
        or set(method_lock) != _METHOD_LOCK_INPUT_FIELDS
    ):
        raise ValueError("r2 method lock field allowlist mismatch")
    lock = dict(method_lock)
    expected = {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": METHOD_ID,
        "candidate_id": METHOD_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "class_handle_binding_sha256": class_binding_sha256,
        "qknn_lock_sha256_by_k": {
            key: _validate_sha256(value, f"qknn_lock_sha256_by_k[{key}]")
            for key, value in (
                lock.get("qknn_lock_sha256_by_k", {}).items()
                if isinstance(lock.get("qknn_lock_sha256_by_k"), Mapping)
                else ()
            )
        },
        "rank": RANK,
        "old_class_count": CLASS_COUNT,
        "allowed_k": [1, 5, 10],
        "ground_old_multiprototype_enabled": True,
        "ground_old_multiprototype_max_per_class": MAX_PROTOTYPES_PER_CLASS,
        "ground_old_multiprototype_min_physical_samples": (
            MIN_PHYSICAL_SAMPLES_PER_AGGREGATE
        ),
        "ground_old_multiprototype_old_classes_only": True,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
        "ground_component_phase2_mutable": False,
        "delta_tau_source": (
            "phase1_receiver_lodo_correct_held_pseudoquery_only"
        ),
        "active_set_steps": 2,
        "ridge_fraction": 0.01,
        "theta_box_abs": 1.0,
        "trust_divisor_squared": 160,
        "g_denominator": 4,
        "target25_release_authorized": False,
        "query_fit_access": False,
        "query_rows_used_for_fit": 0,
    }
    qknn_by_k = expected["qknn_lock_sha256_by_k"]
    if set(qknn_by_k) != {"1", "5", "10"} or len(set(qknn_by_k.values())) != 3:
        raise ValueError(
            "qknn_lock_sha256_by_k must bind distinct lowercase SHA256 "
            "values for K=1,5,10"
        )
    if lock != expected:
        raise ValueError("r2 method lock semantic drift")
    return lock


def _validate_final_method_lock(
    method_lock: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    class_binding_sha256: str,
) -> dict[str, Any]:
    if not isinstance(method_lock, Mapping) or set(method_lock) != _METHOD_LOCK_FIELDS:
        raise ValueError("final r2 method lock field allowlist mismatch")
    final = dict(method_lock)
    base = {key: final[key] for key in _METHOD_LOCK_INPUT_FIELDS}
    _validate_method_lock_input(
        base,
        checkpoint_sha256=checkpoint_sha256,
        class_binding_sha256=class_binding_sha256,
    )
    for field in ("delta_q", "tau_q"):
        value = final[field]
        if not isinstance(value, (float, int)) or not np.isfinite(float(value)):
            raise ValueError(f"final method lock {field} must be finite")
    if float(final["delta_q"]) < 0.0 or float(final["tau_q"]) < 2.0**-10:
        raise ValueError("final method lock delta_q/tau_q drift")
    return final


def _receipt_digest(
    receipt: Mapping[str, Any],
    *,
    class_handle: str,
    prototype_index: int,
    prototype: np.ndarray,
) -> tuple[str, int, float]:
    if not isinstance(receipt, Mapping) or set(receipt) != _AGGREGATION_RECEIPT_FIELDS:
        raise ValueError("ground aggregation receipt field allowlist mismatch")
    expected_fixed = {
        "schema": AGGREGATION_RECEIPT_SCHEMA,
        "class_handle": class_handle,
        "prototype_index": prototype_index,
        "phase1_before_target_access": True,
        "multi_physical_aggregation": True,
        "member_ids_included": False,
        "sample_features_included": False,
        "source_path_included": False,
    }
    if any(receipt.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("ground aggregation receipt semantic drift")
    try:
        count = int(receipt["distinct_physical_sample_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ground aggregation physical count drift") from exc
    if count < MIN_PHYSICAL_SAMPLES_PER_AGGREGATE:
        raise ValueError("each ground prototype must aggregate at least two physical samples")
    try:
        radius = float(receipt["aggregation_radius"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ground aggregation radius must be finite non-negative") from exc
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("ground aggregation radius must be finite non-negative")
    _validate_sha256(
        receipt["physical_sample_commitment_sha256"],
        "physical_sample_commitment_sha256",
    )
    expected_prototype_sha = canonical_array_sha256(
        np.asarray(prototype, dtype=np.float64)
    )
    if (
        _validate_sha256(receipt["prototype_sha256"], "prototype_sha256")
        != expected_prototype_sha
    ):
        raise ValueError("ground aggregation receipt does not bind its prototype")
    sanitized = dict(receipt)
    sanitized["distinct_physical_sample_count"] = count
    sanitized["aggregation_radius"] = radius
    return _sha256_bytes(_canonical_json_bytes(sanitized)), count, radius


def _extract_source(
    source_aggregate: Mapping[str, Any],
    class_registry: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(source_aggregate, Mapping) or set(source_aggregate) != _SOURCE_FIELDS:
        raise ValueError("source aggregate field allowlist mismatch")
    if source_aggregate["feature_key"] != "z_id":
        raise ValueError("source aggregate only authorizes z_id")
    if source_aggregate["protocol_schema"] != PROTOCOL_SCHEMA:
        raise ValueError("source aggregate protocol schema drift")
    if source_aggregate["receiver_day_mean_schema"] != RECEIVER_DAY_MEAN_SCHEMA:
        raise ValueError("receiver/day mean representation schema drift")

    records = source_aggregate["ground_multiprototypes"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("ground_multiprototypes must be a six-class sequence")
    if len(records) != CLASS_COUNT:
        raise ValueError("ground multiprototypes must cover every old class")
    prototypes = np.zeros(
        (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS, FEATURE_DIM), dtype=np.float64
    )
    mask = np.zeros((CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS), dtype=np.bool_)
    physical_counts = np.zeros(
        (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS), dtype=np.int16
    )
    prototype_weights = np.zeros(
        (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS), dtype=np.float64
    )
    prototype_radii = np.zeros(
        (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS), dtype=np.float64
    )
    receipt_sha = np.full(
        (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS), b"", dtype="S64"
    )
    sanitized_receipts: list[list[str]] = []
    for class_index, (record, handle) in enumerate(zip(records, class_registry)):
        if not isinstance(record, Mapping) or set(record) != _CLASS_RECORD_FIELDS:
            raise ValueError("ground class record field allowlist mismatch")
        if record["class_handle"] != handle:
            raise ValueError("ground class registry/order binding drift")
        items = record["prototypes"]
        if (
            not isinstance(items, Sequence)
            or isinstance(items, (str, bytes))
            or not 1 <= len(items) <= MAX_PROTOTYPES_PER_CLASS
        ):
            raise ValueError("each old class requires one to three ground prototypes")
        class_receipts: list[str] = []
        for prototype_index, item in enumerate(items):
            if not isinstance(item, Mapping) or set(item) != _PROTOTYPE_RECORD_FIELDS:
                raise ValueError("ground prototype record field allowlist mismatch")
            raw_vector = _as_float64(
                item["vector"], f"ground prototype {class_index}:{prototype_index}"
            )
            vector = _normalize_vector(
                raw_vector, f"ground prototype {class_index}:{prototype_index}"
            )
            digest, count, radius = _receipt_digest(
                item["aggregation_receipt"],
                class_handle=handle,
                prototype_index=prototype_index,
                prototype=raw_vector,
            )
            prototypes[class_index, prototype_index] = vector
            mask[class_index, prototype_index] = True
            physical_counts[class_index, prototype_index] = count
            prototype_weights[class_index, prototype_index] = 1.0 / len(items)
            prototype_radii[class_index, prototype_index] = radius
            receipt_sha[class_index, prototype_index] = digest.encode("ascii")
            class_receipts.append(digest)
        sanitized_receipts.append(class_receipts)

    means = _as_float64(
        source_aggregate["receiver_day_means"], "receiver/day aggregate means"
    )
    domain_mask = np.asarray(source_aggregate["receiver_day_mask"])
    domain_counts = np.asarray(source_aggregate["receiver_day_physical_counts"])
    if (
        means.ndim != 3
        or means.shape[0] != CLASS_COUNT
        or means.shape[2] != FEATURE_DIM
        or domain_mask.shape != means.shape[:2]
        or domain_counts.shape != means.shape[:2]
        or domain_mask.dtype != np.bool_
    ):
        raise ValueError("receiver/day aggregate shape or mask drift")
    if not np.issubdtype(domain_counts.dtype, np.integer):
        raise ValueError("receiver/day physical counts must be integers")
    if bool(np.any(domain_counts[domain_mask] < MIN_PHYSICAL_SAMPLES_PER_AGGREGATE)):
        raise ValueError("each observed receiver/day mean requires two physical samples")
    if bool(np.any(domain_counts[~domain_mask] != 0)):
        raise ValueError("unobserved receiver/day cells must have zero count")
    weighted_rows: list[np.ndarray] = []
    for class_index in range(CLASS_COUNT):
        observed = means[class_index, domain_mask[class_index]]
        if observed.shape[0] < 2:
            raise ValueError("each old class requires two observed receiver/day domains")
        center = observed.mean(axis=0)
        weight = 1.0 / np.sqrt(CLASS_COUNT * observed.shape[0])
        weighted_rows.extend((observed - center) * weight)
    d_matrix = np.stack(weighted_rows, axis=0)

    margin_receipt = source_aggregate["phase1_qknn_margin_receipt"]
    if not isinstance(margin_receipt, Mapping) or set(margin_receipt) != _MARGIN_RECEIPT_FIELDS:
        raise ValueError("Phase1 qKNN margin receipt field allowlist mismatch")
    expected_margin = {
        "schema": MARGIN_RECEIPT_SCHEMA,
        "target_accessed": False,
        "receiver_lodo": True,
        "pseudo_support_query_physical_id_disjoint": True,
        "correct_predictions_only": True,
        "target_query_truth_used": False,
        "margin_definition": "top1_minus_logsumexp_other_raw_qknn_score",
    }
    if any(margin_receipt.get(key) != value for key, value in expected_margin.items()):
        raise ValueError("Phase1 qKNN margin receipt semantic drift")
    _validate_sha256(
        margin_receipt["margin_evidence_sha256"], "margin_evidence_sha256"
    )
    margins = _as_float64(margin_receipt["margins"], "Phase1 qKNN margins")
    if margins.ndim != 1 or margins.size < 2:
        raise ValueError("Phase1 qKNN margin set requires at least two values")
    sanitized_margin = dict(margin_receipt)
    sanitized_margin["margins"] = margins.tolist()
    margin_receipt_sha = _sha256_bytes(_canonical_json_bytes(sanitized_margin))
    return {
        "prototypes": prototypes,
        "prototype_mask": mask,
        "prototype_physical_counts": physical_counts,
        "prototype_weights": prototype_weights,
        "prototype_radii": prototype_radii,
        "prototype_receipt_sha256": receipt_sha,
        "sanitized_receipts": sanitized_receipts,
        "d_matrix": d_matrix,
        "domain_mask": domain_mask,
        "domain_counts": domain_counts.astype(np.int64),
        "margins": margins,
        "margin_receipt_sha256": margin_receipt_sha,
    }


def _type7_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (ordered.size - 1) * probability
    low = int(np.floor(position))
    high = int(np.ceil(position))
    fraction = position - low
    return float(ordered[low] + fraction * (ordered[high] - ordered[low]))


def _resource_audit(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    ground_wire_bytes = int(
        payload["p_g_q"].nbytes
        + payload["p_g_scale"].nbytes
        + payload["p_g_weight"].nbytes
        + payload["p_g_radius"].nbytes
        + payload["p_g_mask"].nbytes
        + payload["p_g_physical_counts"].nbytes
        + payload["p_g_receipt_sha256"].nbytes
        + payload["p_g_source_prototype_sha256"].nbytes
        + payload["p_g_quantization_max_abs_error"].nbytes
        + payload["p_g_quantization_certificate_sha256"].nbytes
    )
    factor_numeric_bytes = int(
        payload["l_g_q"].nbytes
        + payload["l_g_scale"].nbytes
        + payload["r_q"].nbytes
        + payload["r_scale"].nbytes
        + payload["direction_energy_a"].nbytes
        + 6  # Stage2 theta INT8[4] plus one FP16 scale.
    )
    # Reserve four serialized hex SHA256 receipts for L_g, R, a, and the
    # Stage2 theta wire.  The first three are present in this component's
    # per-array receipt; theta is reserved here so the 4096-byte claim remains
    # valid when the Stage2 state is appended.
    factor_receipt_bytes = 4 * 64
    factor_wire_bytes = factor_numeric_bytes + factor_receipt_bytes
    margin_wire_bytes = int(payload["delta_q"].nbytes + payload["tau_q"].nbytes)
    metadata_wire_bytes = int(
        payload["class_registry"].nbytes
        + payload["feature_schema"].nbytes
        + payload["protocol_schema"].nbytes
    )
    total_component_bytes = (
        ground_wire_bytes
        + factor_wire_bytes
        + margin_wire_bytes
        + metadata_wire_bytes
    )
    if factor_wire_bytes > JP4_UPDATE_FACTOR_WIRE_LIMIT_BYTES:
        raise ValueError("JP4 update-factor wire exceeds 4096 bytes")
    if total_component_bytes > ARM_STATE_LIMIT_BYTES:
        raise ValueError("Phase1 component state exceeds 256KiB")
    return {
        "ground_wire_bytes": ground_wire_bytes,
        "jp4_update_factor_numeric_bytes": factor_numeric_bytes,
        "jp4_update_factor_receipt_bytes": factor_receipt_bytes,
        "jp4_update_factor_wire_bytes": factor_wire_bytes,
        "phase1_margin_wire_bytes": margin_wire_bytes,
        "component_metadata_wire_bytes": metadata_wire_bytes,
        "total_component_bytes": total_component_bytes,
        "jp4_update_factor_wire_limit_bytes": JP4_UPDATE_FACTOR_WIRE_LIMIT_BYTES,
        "arm_state_limit_bytes": ARM_STATE_LIMIT_BYTES,
        "persistent_dense_float_bank_bytes": 0,
        "ground_direction_rank": RANK,
    }


def _validate_payload(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(payload) != ALLOWED_NPZ_MEMBERS:
        raise ValueError("r2 payload member allowlist mismatch")
    p_q = np.asarray(payload["p_g_q"])
    p_scale = np.asarray(payload["p_g_scale"])
    p_weight = np.asarray(payload["p_g_weight"])
    p_radius = np.asarray(payload["p_g_radius"])
    p_mask = np.asarray(payload["p_g_mask"])
    p_counts = np.asarray(payload["p_g_physical_counts"])
    p_receipts = np.asarray(payload["p_g_receipt_sha256"])
    p_source_sha = np.asarray(payload["p_g_source_prototype_sha256"])
    p_quant_error = np.asarray(payload["p_g_quantization_max_abs_error"])
    p_quant_cert = np.asarray(payload["p_g_quantization_certificate_sha256"])
    l_q, l_scale = np.asarray(payload["l_g_q"]), np.asarray(payload["l_g_scale"])
    r_q, r_scale = np.asarray(payload["r_q"]), np.asarray(payload["r_scale"])
    energy = np.asarray(payload["direction_energy_a"])
    delta_q, tau_q = np.asarray(payload["delta_q"]), np.asarray(payload["tau_q"])
    if p_q.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS, FEATURE_DIM):
        raise ValueError("ground multiprototype code shape drift")
    if (
        p_scale.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_weight.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_radius.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_mask.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_counts.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_receipts.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_source_sha.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_quant_error.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
        or p_quant_cert.shape != (CLASS_COUNT, MAX_PROTOTYPES_PER_CLASS)
    ):
        raise ValueError("ground multiprototype metadata shape drift")
    if (
        p_q.dtype != np.int8
        or p_scale.dtype != np.float16
        or p_weight.dtype != np.float16
        or p_radius.dtype != np.float16
        or p_quant_error.dtype != np.float16
        or p_mask.dtype != np.bool_
    ):
        raise ValueError("ground multiprototype dtype drift")
    if (
        not np.issubdtype(p_counts.dtype, np.integer)
        or p_receipts.dtype.kind != "S"
        or p_source_sha.dtype.kind != "S"
        or p_quant_cert.dtype.kind != "S"
    ):
        raise ValueError("ground receipt dtype drift")
    if bool(np.any(p_q == -128)):
        raise ValueError("ground multiprototype contains forbidden -128")
    for class_index in range(CLASS_COUNT):
        present = p_mask[class_index]
        count = int(present.sum())
        if not 1 <= count <= MAX_PROTOTYPES_PER_CLASS:
            raise ValueError("ground multiprototype old-class coverage drift")
        if not bool(np.all(present[:count])) or bool(np.any(present[count:])):
            raise ValueError("ground prototype mask must be contiguous per class")
        if (
            bool(np.any(p_scale[class_index, :count] <= 0.0))
            or not np.isfinite(p_scale[class_index, :count]).all()
            or not np.isfinite(p_weight[class_index, :count]).all()
            or not np.isfinite(p_radius[class_index, :count]).all()
            or not np.isfinite(p_quant_error[class_index, :count]).all()
            or bool(np.any(p_radius[class_index, :count] < 0.0))
            or bool(np.any(p_quant_error[class_index, :count] < 0.0))
            or bool(
                np.any(
                    p_counts[class_index, :count]
                    < MIN_PHYSICAL_SAMPLES_PER_AGGREGATE
                )
            )
        ):
            raise ValueError("ground prototype present metadata drift")
        expected_weight = np.float16(1.0 / count)
        if not bool(
            np.all(p_weight[class_index, :count] == expected_weight)
        ):
            raise ValueError("ground prototype persisted weight must equal 1/M")
        if (
            bool(np.any(p_q[class_index, count:] != 0))
            or bool(np.any(p_scale[class_index, count:] != 0.0))
            or bool(np.any(p_weight[class_index, count:] != 0.0))
            or bool(np.any(p_radius[class_index, count:] != 0.0))
            or bool(np.any(p_counts[class_index, count:] != 0))
            or bool(np.any(p_quant_error[class_index, count:] != 0.0))
            or any(bytes(value) for value in p_receipts[class_index, count:])
            or any(bytes(value) for value in p_source_sha[class_index, count:])
            or any(bytes(value) for value in p_quant_cert[class_index, count:])
        ):
            raise ValueError("ground prototype padding must be exact zero")
        for digest in p_receipts[class_index, :count]:
            _validate_sha256(bytes(digest).decode("ascii"), "prototype receipt SHA256")
        for field, digests in (
            ("source prototype SHA256", p_source_sha[class_index, :count]),
            ("quantization certificate SHA256", p_quant_cert[class_index, :count]),
        ):
            for digest in digests:
                _validate_sha256(bytes(digest).decode("ascii"), field)
    if (
        l_q.shape != (RANK, FEATURE_DIM)
        or l_scale.shape != (RANK,)
        or r_q.shape != (RANK, HIDDEN_DIM)
        or r_scale.shape != (RANK,)
        or energy.shape != (RANK,)
        or delta_q.shape != ()
        or tau_q.shape != ()
    ):
        raise ValueError("r2 compact factor/scalar shape drift")
    for field, codes in (("l_g_q", l_q), ("r_q", r_q)):
        if codes.dtype != np.int8 or bool(np.any(codes == -128)):
            raise ValueError(f"{field} INT8 contract drift")
    for field, scales in (
        ("l_g_scale", l_scale),
        ("r_scale", r_scale),
        ("direction_energy_a", energy),
    ):
        if scales.dtype != np.float16 or not np.isfinite(scales).all() or bool(np.any(scales <= 0.0)):
            raise ValueError(f"{field} FP16 contract drift")
    if (
        delta_q.dtype != np.float16
        or tau_q.dtype != np.float16
        or not np.isfinite(delta_q)
        or not np.isfinite(tau_q)
        or float(delta_q) < 0.0
        or float(tau_q) < 2.0**-10
    ):
        raise ValueError("delta_q/tau_q FP16 contract drift")
    l_decoded = l_q.astype(np.float64) * l_scale.astype(np.float64)[:, None]
    r_decoded = r_q.astype(np.float64) * r_scale.astype(np.float64)[:, None]
    for field, matrix in (("quantized L_g", l_decoded), ("quantized R", r_decoded)):
        singular = np.linalg.svd(matrix, compute_uv=False)
        threshold = 1.0e-6 * float(singular[0])
        if int(np.sum(singular > threshold)) != RANK:
            raise ValueError(f"{field} lost rank four")
    registry_array = np.asarray(payload["class_registry"])
    if registry_array.ndim != 1 or registry_array.dtype.kind not in {"U", "S"}:
        raise ValueError("class registry payload drift")
    registry = tuple(str(value) for value in registry_array.tolist())
    _class_handle_binding_sha256(registry)
    for class_index in range(CLASS_COUNT):
        count = int(p_mask[class_index].sum())
        for prototype_index in range(count):
            certificate = {
                "schema": GROUND_QUANTIZATION_CERTIFICATE_SCHEMA,
                "class_handle": registry[class_index],
                "prototype_index": prototype_index,
                "rounding_rule": ROUNDING_SCHEMA,
                "source_prototype_sha256": bytes(
                    p_source_sha[class_index, prototype_index]
                ).decode("ascii"),
                "int8_codes_sha256": canonical_array_sha256(
                    p_q[class_index, prototype_index]
                ),
                "fp16_scale_sha256": canonical_array_sha256(
                    np.asarray(
                        p_scale[class_index, prototype_index], dtype=np.float16
                    )
                ),
                "max_abs_error_fp16_sha256": canonical_array_sha256(
                    np.asarray(
                        p_quant_error[class_index, prototype_index],
                        dtype=np.float16,
                    )
                ),
            }
            expected_certificate = _sha256_bytes(
                _canonical_json_bytes(certificate)
            )
            actual_certificate = bytes(
                p_quant_cert[class_index, prototype_index]
            ).decode("ascii")
            if actual_certificate != expected_certificate:
                raise ValueError("ground quantization certificate mismatch")
    for field, expected in (
        ("feature_schema", FEATURE_SCHEMA),
        ("protocol_schema", PROTOCOL_SCHEMA),
    ):
        value = np.asarray(payload[field])
        if value.shape != () or value.dtype.kind not in {"U", "S"} or str(value.item()) != expected:
            raise ValueError(f"{field} payload drift")
    return {
        "class_registry": registry,
        "resource_audit": _resource_audit(payload),
    }


def build_grb_jp4_cfm_component(
    source_aggregate: Mapping[str, Any],
    *,
    class_registry: Sequence[str],
    checkpoint_joint_proj_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
    class_handle_binding_sha256: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
    method_lock: Mapping[str, Any],
    provenance_status: str,
    formal_phase2_eligible: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the target-inaccessible r2 Phase1 component."""

    if formal_phase2_eligible:
        raise ValueError("standalone r2 component cannot be formally eligible")
    registry = tuple(str(item) for item in class_registry)
    actual_binding = _class_handle_binding_sha256(registry)
    binding = _validate_sha256(
        class_handle_binding_sha256, "class_handle_binding_sha256"
    )
    if binding != actual_binding:
        raise ValueError("class registry binding mismatch")
    checkpoint = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    code_sha = _validate_sha256(generation_code_sha256, "generation_code_sha256")
    config_sha = _validate_sha256(
        generation_config_sha256, "generation_config_sha256"
    )
    lock_input = _validate_method_lock_input(
        method_lock,
        checkpoint_sha256=checkpoint,
        class_binding_sha256=binding,
    )
    extracted = _extract_source(source_aggregate, registry)
    l_g, singular_d = _canonical_svd_rows(
        extracted["d_matrix"], rank=RANK, columns=FEATURE_DIM, field="weighted D"
    )
    r, _singular_w = _canonical_svd_rows(
        checkpoint_joint_proj_weight,
        rank=RANK,
        columns=HIDDEN_DIM,
        field="checkpoint joint_proj.0.weight",
    )
    energy64 = singular_d / np.sqrt(float(np.sum(singular_d**2)))
    if not np.isfinite(energy64).all() or bool(np.any(energy64 <= 0.0)):
        raise ValueError("direction energy a must be finite positive")

    margins = extracted["margins"]
    median = float(np.median(margins))
    tau64 = max(
        2.0**-10,
        1.4826 * float(np.median(np.abs(margins - median))),
    )
    delta64 = max(0.0, _type7_quantile(margins / tau64, 0.10))
    tau_q = np.asarray(tau64, dtype=np.float16)
    delta_q = np.asarray(delta64, dtype=np.float16)
    if not np.isfinite(tau_q) or float(tau_q) < 2.0**-10:
        raise ValueError("tau_q cannot be represented by finite FP16")
    if not np.isfinite(delta_q) or float(delta_q) < 0.0:
        raise ValueError("delta_q cannot be represented by finite FP16")
    lock = {
        **lock_input,
        "delta_q": float(delta_q),
        "tau_q": float(tau_q),
    }
    _validate_final_method_lock(
        lock,
        checkpoint_sha256=checkpoint,
        class_binding_sha256=binding,
    )

    p_q = np.zeros_like(extracted["prototypes"], dtype=np.int8)
    p_scale = np.zeros(extracted["prototype_mask"].shape, dtype=np.float16)
    p_weight = np.asarray(extracted["prototype_weights"], dtype=np.float16)
    p_radius = np.asarray(extracted["prototype_radii"], dtype=np.float16)
    if (
        not np.isfinite(p_weight).all()
        or not np.isfinite(p_radius).all()
        or bool(np.any(p_radius < 0.0))
    ):
        raise ValueError("ground weight/radius cannot be represented by finite FP16")
    p_quantization_error = np.zeros(
        extracted["prototype_mask"].shape, dtype=np.float16
    )
    p_quantization_certificate = np.full(
        extracted["prototype_mask"].shape, b"", dtype="S64"
    )
    p_source_prototype_sha = np.full(
        extracted["prototype_mask"].shape, b"", dtype="S64"
    )
    for class_index in range(CLASS_COUNT):
        count = int(extracted["prototype_mask"][class_index].sum())
        codes, scales = _quantize_vectors(
            extracted["prototypes"][class_index, :count],
            f"ground prototypes class {class_index}",
        )
        p_q[class_index, :count] = codes
        p_scale[class_index, :count] = scales
        for prototype_index in range(count):
            source_vector = extracted["prototypes"][
                class_index, prototype_index
            ]
            decoded = (
                codes[prototype_index].astype(np.float64)
                * float(scales[prototype_index])
            )
            error = float(np.max(np.abs(source_vector - decoded)))
            error_fp16 = np.float16(error)
            if not np.isfinite(error_fp16) or float(error_fp16) < 0.0:
                raise ValueError("ground quantization error certificate overflow")
            p_quantization_error[class_index, prototype_index] = error_fp16
            source_prototype_sha = canonical_array_sha256(source_vector)
            p_source_prototype_sha[
                class_index, prototype_index
            ] = source_prototype_sha.encode("ascii")
            certificate = {
                "schema": GROUND_QUANTIZATION_CERTIFICATE_SCHEMA,
                "class_handle": registry[class_index],
                "prototype_index": prototype_index,
                "rounding_rule": ROUNDING_SCHEMA,
                "source_prototype_sha256": source_prototype_sha,
                "int8_codes_sha256": canonical_array_sha256(
                    codes[prototype_index]
                ),
                "fp16_scale_sha256": canonical_array_sha256(
                    np.asarray(scales[prototype_index], dtype=np.float16)
                ),
                "max_abs_error_fp16_sha256": canonical_array_sha256(
                    np.asarray(error_fp16, dtype=np.float16)
                ),
            }
            p_quantization_certificate[
                class_index, prototype_index
            ] = _sha256_bytes(_canonical_json_bytes(certificate)).encode("ascii")
    l_q, l_scale = _quantize_vectors(l_g, "L_g")
    r_q, r_scale = _quantize_vectors(r, "R")
    payload = {
        "p_g_q": p_q,
        "p_g_scale": p_scale,
        "p_g_weight": p_weight,
        "p_g_radius": p_radius,
        "p_g_mask": extracted["prototype_mask"],
        "p_g_physical_counts": extracted["prototype_physical_counts"],
        "p_g_receipt_sha256": extracted["prototype_receipt_sha256"],
        "p_g_source_prototype_sha256": p_source_prototype_sha,
        "p_g_quantization_max_abs_error": p_quantization_error,
        "p_g_quantization_certificate_sha256": p_quantization_certificate,
        "l_g_q": l_q,
        "l_g_scale": l_scale,
        "r_q": r_q,
        "r_scale": r_scale,
        "direction_energy_a": np.asarray(energy64, dtype=np.float16),
        "delta_q": delta_q,
        "tau_q": tau_q,
        "class_registry": np.asarray(registry, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
        "protocol_schema": np.asarray(PROTOCOL_SCHEMA, dtype=np.str_),
    }
    details = _validate_payload(payload)
    source_digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": "phase1_grb_jp4_cfm_source_aggregate_digest_v2",
                "receiver_day_mean_schema": RECEIVER_DAY_MEAN_SCHEMA,
                "prototype_sha256": canonical_array_sha256(
                    extracted["prototypes"]
                ),
                "prototype_mask_sha256": canonical_array_sha256(
                    extracted["prototype_mask"]
                ),
                "prototype_weight_sha256": canonical_array_sha256(
                    p_weight
                ),
                "prototype_radius_sha256": canonical_array_sha256(
                    p_radius
                ),
                "prototype_receipts": extracted["sanitized_receipts"],
                "weighted_d_sha256": canonical_array_sha256(
                    extracted["d_matrix"]
                ),
                "domain_mask_sha256": canonical_array_sha256(
                    extracted["domain_mask"]
                ),
                "domain_counts_sha256": canonical_array_sha256(
                    extracted["domain_counts"]
                ),
                "margin_receipt_sha256": extracted["margin_receipt_sha256"],
            }
        )
    )
    array_hashes = {
        key: canonical_array_sha256(value) for key, value in sorted(payload.items())
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "component_profile": COMPONENT_PROFILE,
        "method_lock_schema": METHOD_LOCK_SCHEMA,
        "method_lock": lock,
        "method_lock_sha256": _sha256_bytes(_canonical_json_bytes(lock)),
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "receiver_day_mean_schema": RECEIVER_DAY_MEAN_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "class_count": CLASS_COUNT,
        "rank": RANK,
        "max_prototypes_per_class": MAX_PROTOTYPES_PER_CLASS,
        "min_physical_samples_per_aggregate": MIN_PHYSICAL_SAMPLES_PER_AGGREGATE,
        "ground_old_multiprototype_enabled": True,
        "checkpoint_sha256": checkpoint,
        "class_handle_binding_sha256": binding,
        "registry_sha256": _sha256_bytes(_canonical_json_bytes(list(registry))),
        "source_aggregate_generation_digest_sha256": source_digest,
        "margin_receipt_sha256": extracted["margin_receipt_sha256"],
        "generation_code_sha256": code_sha,
        "generation_config_sha256": config_sha,
        "array_sha256": array_hashes,
        "provenance_status": str(provenance_status),
        "component_state": PENDING_OUTER_JOINT_SEAL,
        "outer_bundle_signature_required": True,
        "formal_phase2_eligible": False,
        "target25_release_authorized": False,
        "svd_canonicalization": SVD_SCHEMA,
        "rounding_rule": ROUNDING_SCHEMA,
        "member_allowlist": [NPZ_NAME],
        "npz_member_allowlist": sorted(ALLOWED_NPZ_MEMBERS),
        "quantization": {
            "codes": "symmetric_int8_no_minus128",
            "scales": "float16_round_to_nearest_even",
            "per_vector": True,
            "qmin": -127,
            "qmax": 127,
        },
        "resource_audit": details["resource_audit"],
        "phase2_phase1_component_generation_stage": (
            "phase1_offline_before_target_access"
        ),
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "phase2_phase1_component_member_or_exemplar_access": False,
        "phase2_phase1_component_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
    }
    if set(manifest) != BASE_MANIFEST_FIELDS:
        raise AssertionError("internal r2 base manifest field drift")
    return payload, manifest


def _pre_sign_content_root(
    manifest: Mapping[str, Any], component_npz_sha256: str
) -> str:
    bound = {key: manifest[key] for key in sorted(BASE_MANIFEST_FIELDS)}
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "manifest": bound,
                "component_npz_sha256": component_npz_sha256,
            }
        )
    )


def save_grb_jp4_cfm_component(
    output_dir: str | Path,
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    details = _validate_payload(payload)
    if set(manifest) != BASE_MANIFEST_FIELDS or manifest.get("schema") != SCHEMA:
        raise ValueError("r2 base manifest/schema drift")
    if manifest.get("component_profile") != COMPONENT_PROFILE:
        raise ValueError("r2 component profile drift")
    if manifest.get("resource_audit") != details["resource_audit"]:
        raise ValueError("r2 resource receipt drift before save")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    npz_path = root / NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    npz_sha = sha256_file(npz_path)
    final = dict(manifest)
    final["component_npz_sha256"] = npz_sha
    final["serialized_component_bytes"] = int(npz_path.stat().st_size)
    final["pre_sign_content_root_sha256"] = _pre_sign_content_root(final, npz_sha)
    if set(final) != FINAL_MANIFEST_FIELDS:
        raise AssertionError("internal r2 final manifest field drift")
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = sha256_file(manifest_path)
    (root / MANIFEST_SHA_NAME).write_text(
        f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    validate_grb_jp4_cfm_component(root)
    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "component_npz_sha256": npz_sha,
        "manifest_sha256": manifest_sha,
        "pre_sign_content_root_sha256": final["pre_sign_content_root_sha256"],
    }


def validate_grb_jp4_cfm_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_class_handle_binding_sha256: str | None = None,
    expected_method_lock_sha256: str | None = None,
    expected_pre_sign_content_root_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(component_dir)
    actual = {item.name for item in root.iterdir()} if root.is_dir() else set()
    if actual != {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}:
        raise ValueError("r2 component directory member allowlist mismatch")
    manifest_path = root / MANIFEST_NAME
    expected_sidecar = f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n"
    if (root / MANIFEST_SHA_NAME).read_text(encoding="ascii") != expected_sidecar:
        raise ValueError("r2 manifest SHA256 sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != FINAL_MANIFEST_FIELDS or manifest.get("schema") != SCHEMA:
        raise ValueError("r2 final manifest/schema mismatch")
    if (
        manifest.get("component_profile") != COMPONENT_PROFILE
        or manifest.get("method_lock_schema") != METHOD_LOCK_SCHEMA
        or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("receiver_day_mean_schema") != RECEIVER_DAY_MEAN_SCHEMA
    ):
        raise ValueError("r2 profile/method/protocol/feature drift")
    if (
        int(manifest.get("feature_dim", -1)),
        int(manifest.get("hidden_dim", -1)),
        int(manifest.get("class_count", -1)),
        int(manifest.get("rank", -1)),
    ) != (FEATURE_DIM, HIDDEN_DIM, CLASS_COUNT, RANK):
        raise ValueError("r2 compact shape contract drift")
    if (
        manifest.get("max_prototypes_per_class") != MAX_PROTOTYPES_PER_CLASS
        or manifest.get("min_physical_samples_per_aggregate")
        != MIN_PHYSICAL_SAMPLES_PER_AGGREGATE
        or manifest.get("ground_old_multiprototype_enabled") is not True
        or manifest.get("target25_release_authorized") is not False
    ):
        raise ValueError("r2 multiprototype/release contract drift")
    checkpoint = _validate_sha256(
        manifest.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    binding = _validate_sha256(
        manifest.get("class_handle_binding_sha256"),
        "class_handle_binding_sha256",
    )
    lock = _validate_final_method_lock(
        manifest.get("method_lock"),
        checkpoint_sha256=checkpoint,
        class_binding_sha256=binding,
    )
    lock_sha = _sha256_bytes(_canonical_json_bytes(lock))
    if manifest.get("method_lock_sha256") != lock_sha:
        raise ValueError("r2 method lock SHA256 mismatch")
    for field in (
        "registry_sha256",
        "source_aggregate_generation_digest_sha256",
        "margin_receipt_sha256",
        "generation_code_sha256",
        "generation_config_sha256",
        "component_npz_sha256",
        "pre_sign_content_root_sha256",
    ):
        _validate_sha256(manifest.get(field), field)
    if (
        manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL
        or manifest.get("outer_bundle_signature_required") is not True
        or manifest.get("formal_phase2_eligible") is not False
    ):
        raise ValueError("standalone r2 component must remain pending outer seal")
    if (
        manifest.get("svd_canonicalization") != SVD_SCHEMA
        or manifest.get("rounding_rule") != ROUNDING_SCHEMA
        or manifest.get("member_allowlist") != [NPZ_NAME]
        or manifest.get("npz_member_allowlist") != sorted(ALLOWED_NPZ_MEMBERS)
    ):
        raise ValueError("r2 canonicalization/member allowlist drift")
    if manifest.get("quantization") != {
        "codes": "symmetric_int8_no_minus128",
        "scales": "float16_round_to_nearest_even",
        "per_vector": True,
        "qmin": -127,
        "qmax": 127,
    }:
        raise ValueError("r2 quantization contract drift")
    protocol_expected = {
        "phase2_phase1_component_generation_stage": (
            "phase1_offline_before_target_access"
        ),
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "phase2_phase1_component_member_or_exemplar_access": False,
        "phase2_phase1_component_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
    }
    if any(manifest.get(key) != value for key, value in protocol_expected.items()):
        raise ValueError("r2 protocol manifest drift")
    npz_path = root / NPZ_NAME
    npz_sha = sha256_file(npz_path)
    if manifest["component_npz_sha256"] != npz_sha:
        raise ValueError("r2 NPZ SHA256 mismatch")
    if manifest["pre_sign_content_root_sha256"] != _pre_sign_content_root(
        manifest, npz_sha
    ):
        raise ValueError("r2 pre-sign content root mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != ALLOWED_NPZ_MEMBERS:
            raise ValueError("r2 NPZ member allowlist mismatch")
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    details = _validate_payload(payload)
    if manifest["resource_audit"] != details["resource_audit"]:
        raise ValueError("r2 resource audit mismatch")
    if manifest["array_sha256"] != {
        key: canonical_array_sha256(value) for key, value in sorted(payload.items())
    }:
        raise ValueError("r2 per-array SHA256 mismatch")
    registry = details["class_registry"]
    if binding != _class_handle_binding_sha256(registry):
        raise ValueError("r2 class registry binding drift")
    if manifest["registry_sha256"] != _sha256_bytes(
        _canonical_json_bytes(list(registry))
    ):
        raise ValueError("r2 class registry SHA256 drift")
    if int(manifest["serialized_component_bytes"]) != npz_path.stat().st_size:
        raise ValueError("r2 serialized byte receipt drift")
    for expected, field in (
        (expected_checkpoint_sha256, "checkpoint_sha256"),
        (expected_class_handle_binding_sha256, "class_handle_binding_sha256"),
        (expected_method_lock_sha256, "method_lock_sha256"),
        (expected_pre_sign_content_root_sha256, "pre_sign_content_root_sha256"),
    ):
        if expected is not None and manifest[field] != _validate_sha256(expected, field):
            raise ValueError(f"{field} binding mismatch")
    return manifest


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class GRBJP4CFMPhase1Component:
    p_g_q: np.ndarray
    p_g_scale: np.ndarray
    p_g_weight: np.ndarray
    p_g_radius: np.ndarray
    p_g_mask: np.ndarray
    p_g_physical_counts: np.ndarray
    p_g_receipt_sha256: np.ndarray
    p_g_source_prototype_sha256: np.ndarray
    p_g_quantization_max_abs_error: np.ndarray
    p_g_quantization_certificate_sha256: np.ndarray
    l_g_q: np.ndarray
    l_g_scale: np.ndarray
    r_q: np.ndarray
    r_scale: np.ndarray
    direction_energy_a: np.ndarray
    delta_q: float
    tau_q: float
    class_registry: tuple[str, ...]
    method_lock: Mapping[str, Any]
    manifest: Mapping[str, Any]

    def ground_multiprototypes(self) -> np.ndarray:
        decoded = (
            self.p_g_q.astype(np.float32)
            * self.p_g_scale.astype(np.float32)[:, :, None]
        )
        return _readonly(decoded, np.float32)

    def ground_barycenters(self) -> np.ndarray:
        decoded = self.ground_multiprototypes().astype(np.float64)
        result = np.zeros((CLASS_COUNT, FEATURE_DIM), dtype=np.float64)
        for class_index in range(CLASS_COUNT):
            center = np.sum(
                decoded[class_index]
                * self.p_g_weight[class_index].astype(np.float64)[:, None],
                axis=0,
            )
            norm = float(np.linalg.norm(center))
            if norm <= 1.0e-12:
                raise ValueError("quantized ground barycenter became zero")
            result[class_index] = center / norm
        return _readonly(result, np.float32)

    def ground_left_factors(self) -> np.ndarray:
        return _readonly(
            self.l_g_q.astype(np.float32)
            * self.l_g_scale.astype(np.float32)[:, None],
            np.float32,
        )

    def checkpoint_right_factors(self) -> np.ndarray:
        return _readonly(
            self.r_q.astype(np.float32)
            * self.r_scale.astype(np.float32)[:, None],
            np.float32,
        )


def load_grb_jp4_cfm_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_method_lock_sha256: str,
    expected_pre_sign_content_root_sha256: str,
    allow_pending_outer_joint_seal_development: bool = False,
) -> GRBJP4CFMPhase1Component:
    manifest = validate_grb_jp4_cfm_component(
        component_dir,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_class_handle_binding_sha256=expected_class_handle_binding_sha256,
        expected_method_lock_sha256=expected_method_lock_sha256,
        expected_pre_sign_content_root_sha256=expected_pre_sign_content_root_sha256,
    )
    if not allow_pending_outer_joint_seal_development:
        raise ValueError("formal r2 load requires the existing outer joint seal")
    with np.load(Path(component_dir) / NPZ_NAME, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    details = _validate_payload(payload)
    return GRBJP4CFMPhase1Component(
        p_g_q=_readonly(payload["p_g_q"], np.int8),
        p_g_scale=_readonly(payload["p_g_scale"], np.float16),
        p_g_weight=_readonly(payload["p_g_weight"], np.float16),
        p_g_radius=_readonly(payload["p_g_radius"], np.float16),
        p_g_mask=_readonly(payload["p_g_mask"], np.bool_),
        p_g_physical_counts=_readonly(payload["p_g_physical_counts"], np.int16),
        p_g_receipt_sha256=_readonly(payload["p_g_receipt_sha256"], payload["p_g_receipt_sha256"].dtype),
        p_g_source_prototype_sha256=_readonly(
            payload["p_g_source_prototype_sha256"],
            payload["p_g_source_prototype_sha256"].dtype,
        ),
        p_g_quantization_max_abs_error=_readonly(
            payload["p_g_quantization_max_abs_error"], np.float16
        ),
        p_g_quantization_certificate_sha256=_readonly(
            payload["p_g_quantization_certificate_sha256"],
            payload["p_g_quantization_certificate_sha256"].dtype,
        ),
        l_g_q=_readonly(payload["l_g_q"], np.int8),
        l_g_scale=_readonly(payload["l_g_scale"], np.float16),
        r_q=_readonly(payload["r_q"], np.int8),
        r_scale=_readonly(payload["r_scale"], np.float16),
        direction_energy_a=_readonly(payload["direction_energy_a"], np.float16),
        delta_q=float(payload["delta_q"]),
        tau_q=float(payload["tau_q"]),
        class_registry=details["class_registry"],
        method_lock=_deep_freeze(manifest["method_lock"]),
        manifest=_deep_freeze(manifest),
    )


__all__ = [
    "AGGREGATION_RECEIPT_SCHEMA",
    "ALLOWED_NPZ_MEMBERS",
    "CLASS_COUNT",
    "COMPONENT_PROFILE",
    "FEATURE_DIM",
    "GRBJP4CFMPhase1Component",
    "HIDDEN_DIM",
    "MARGIN_RECEIPT_SCHEMA",
    "MAX_PROTOTYPES_PER_CLASS",
    "METHOD_ID",
    "METHOD_LOCK_SCHEMA",
    "NPZ_NAME",
    "PROTOCOL_SCHEMA",
    "RECEIVER_DAY_MEAN_SCHEMA",
    "RANK",
    "SCHEMA",
    "build_grb_jp4_cfm_component",
    "canonical_array_sha256",
    "class_handle_binding_sha256",
    "load_grb_jp4_cfm_component",
    "save_grb_jp4_cfm_component",
    "sha256_file",
    "validate_grb_jp4_cfm_component",
]
