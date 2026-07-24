"""Immutable GRB-JP4 Phase1 compact component.

The builder accepts only the aggregate Phase1 ``prototypes`` and observed
``domain_shifts`` surface plus the already-bound checkpoint projection weight.
It deliberately has no input for a dataset, source path, member list, sample
feature, or target/query material.  The resulting component is pending the
existing ADV3B02 eight-member outer joint seal; it is not a formal asset by
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "phase1_grb_jp4_compact_component_v1"
COMPONENT_PROFILE = "grb_jp4_q4_int8_v1"
NPZ_NAME = "phase1_grb_jp4_compact_component_v1.npz"
MANIFEST_NAME = "manifest.json"
MANIFEST_SHA_NAME = "manifest.sha256"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
FEATURE_DIM = 160
HIDDEN_DIM = 320
CLASS_COUNT = 6
RANK = 4
PENDING_OUTER_JOINT_SEAL = "PENDING_OUTER_JOINT_SEAL"
SVD_SIGN_SCHEMA = "largest_abs_basis_entry_positive_lowest_index_tie_v1"
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_v1"
# A sign convention alone does not make a repeated-singular-value subspace
# canonical.  This component is sealed once, so reject an ambiguous q4 basis
# instead of silently allowing a BLAS-dependent rotation into the asset.
CANONICAL_SVD_RELATIVE_GAP_MIN = 1.0e-6

ALLOWED_NPZ_MEMBERS = {
    "p_g_q",
    "p_g_scale",
    "l_g_q",
    "l_g_scale",
    "r_q",
    "r_scale",
    "class_registry",
    "feature_schema",
}

BASE_MANIFEST_FIELDS = {
    "schema",
    "component_profile",
    "feature_schema",
    "feature_dim",
    "hidden_dim",
    "class_count",
    "rank",
    "kappa_g",
    "checkpoint_sha256",
    "class_handle_binding_sha256",
    "source_aggregate_generation_digest_sha256",
    "generation_code_sha256",
    "generation_config_sha256",
    "registry_sha256",
    "provenance_status",
    "component_state",
    "outer_bundle_signature_required",
    "formal_phase2_eligible",
    "svd_sign_canonicalization",
    "rounding_rule",
    "member_allowlist",
    "npz_member_allowlist",
    "quantization",
    "resource_audit",
    "phase2_authorized_phase1_model_knowledge_policy",
    "phase2_phase1_component_generation_stage",
    "phase2_phase1_component_payload",
    "phase2_phase1_component_immutable",
    "phase2_phase1_component_update_access",
    "phase2_phase1_component_member_or_exemplar_access",
    "phase2_phase1_component_sample_reconstruction_access",
    "phase2_nonbundle_source_artifact_access",
}
FINAL_MANIFEST_FIELDS = BASE_MANIFEST_FIELDS | {
    "component_npz_sha256",
    "serialized_component_bytes",
    "pre_sign_content_root_sha256",
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _as_float32(value: Any, field: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return array


def _normalize_rows(value: Any, field: str) -> np.ndarray:
    rows = _as_float32(value, field).astype(np.float64)
    if rows.ndim != 2:
        raise ValueError(f"{field} must be two-dimensional")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norm <= 1.0e-12)):
        raise ValueError(f"{field} vectors must be non-zero")
    return np.asarray(rows / norm, dtype=np.float32)


def _quantize_vectors(value: Any, field: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = _as_float32(value, field)
    if vectors.ndim != 2:
        raise ValueError(f"{field} must be [vectors, features]")
    maximum = np.max(np.abs(vectors), axis=1)
    scale32 = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(np.float32)
    scale = scale32.astype(np.float16)
    if not np.isfinite(scale).all() or bool(np.any(scale <= 0.0)):
        raise ValueError(f"{field} FP16 scales must be finite positive")
    codes = np.clip(np.rint(vectors / scale32[:, None]), -127, 127).astype(np.int8)
    if bool(np.any(codes == -128)):
        raise ValueError(f"{field} symmetric INT8 emitted forbidden -128")
    return codes, scale


def _require_canonical_singular_gaps(
    singular: np.ndarray, *, rank: int, field: str
) -> None:
    """Reject q4 bases whose singular-vector orientation is not unique.

    The first ``rank`` directions must be separated from one another and the
    retained q4 subspace must be separated from its next direction when one
    exists.  This preserves the documented sign-canonical convention without
    pretending that it resolves rotations inside a degenerate subspace.
    """

    values = np.asarray(singular, dtype=np.float64)
    if values.ndim != 1 or len(values) < rank:
        raise ValueError(f"{field} singular spectrum cannot provide rank {rank}")
    last = min(len(values) - 1, rank)
    for index in range(last):
        high, low = float(values[index]), float(values[index + 1])
        if high <= 0.0 or low < 0.0:
            raise ValueError(f"{field} singular spectrum is invalid")
        relative_gap = (high - low) / max(high, np.finfo(np.float64).tiny)
        if relative_gap <= CANONICAL_SVD_RELATIVE_GAP_MIN:
            raise ValueError(
                f"{field} has an ambiguous q{rank} singular subspace"
            )


def _canonical_svd_rows(matrix: Any, *, rank: int, field: str) -> tuple[np.ndarray, np.ndarray]:
    value = _as_float32(matrix, field).astype(np.float64)
    if value.ndim != 2 or value.shape[1] != FEATURE_DIM:
        raise ValueError(f"{field} must have {FEATURE_DIM} columns")
    _unused, singular, vt = np.linalg.svd(value, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 0.0:
        raise ValueError(f"{field} cannot provide canonical rank {rank}")
    _require_canonical_singular_gaps(singular, rank=rank, field=field)
    rows = np.asarray(vt[:rank], dtype=np.float64)
    for index, row in enumerate(rows):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            rows[index] *= -1.0
    return np.asarray(rows, dtype=np.float32), np.asarray(singular, dtype=np.float64)


def _canonical_right_svd(weight: Any) -> np.ndarray:
    matrix = _as_float32(weight, "checkpoint_joint_proj_weight")
    if matrix.shape != (FEATURE_DIM, HIDDEN_DIM):
        raise ValueError("checkpoint joint_proj.0 weight must be [160,320]")
    _unused, singular, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
    if len(singular) < RANK or singular[RANK - 1] <= 0.0:
        raise ValueError("checkpoint joint_proj.0 weight cannot provide rank four")
    _require_canonical_singular_gaps(
        singular, rank=RANK, field="checkpoint_joint_proj_weight"
    )
    rows = np.asarray(vt[:RANK], dtype=np.float64)
    for index, row in enumerate(rows):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            rows[index] *= -1.0
    return np.asarray(rows, dtype=np.float32)


def _class_binding_digest(class_registry: Sequence[str]) -> str:
    handles = tuple(str(value) for value in class_registry)
    if len(handles) != CLASS_COUNT or len(set(handles)) != len(handles) or any(
        not value or any(char in value for char in ("/", "\\", ":"))
        for value in handles
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


def _extract_aggregate(source_aggregate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(source_aggregate, Mapping):
        raise ValueError("source_aggregate must be a mapping")
    if str(source_aggregate.get("feature_key", "")) != "z_id":
        raise ValueError("source_aggregate only authorizes z_id")
    prototypes = _normalize_rows(source_aggregate.get("prototypes"), "aggregate prototypes")
    if prototypes.shape != (CLASS_COUNT, FEATURE_DIM):
        raise ValueError("aggregate prototypes must be [6,160]")
    shifts = source_aggregate.get("domain_shifts")
    if not isinstance(shifts, Mapping):
        raise ValueError("source_aggregate requires aggregate domain_shifts")
    domain_shift = _as_float32(shifts.get("domain_shift"), "aggregate domain_shift")
    domain_counts = np.asarray(shifts.get("domain_counts"))
    if domain_shift.ndim != 2 or domain_shift.shape[1] != FEATURE_DIM:
        raise ValueError("aggregate domain_shift must be [domain,160]")
    if domain_counts.shape != (domain_shift.shape[0],):
        raise ValueError("aggregate domain_counts must match domain_shift")
    if (
        not np.issubdtype(domain_counts.dtype, np.number)
        or not np.isfinite(domain_counts).all()
        or bool(np.any(domain_counts < 0))
    ):
        raise ValueError("aggregate domain_counts must be finite non-negative numeric")
    observed = np.asarray(domain_counts > 0, dtype=bool)
    if int(observed.sum()) < RANK:
        raise ValueError("aggregate domain_shifts require at least four observed domains")
    return prototypes, domain_shift, observed


def source_aggregate_generation_digest(source_aggregate: Mapping[str, Any]) -> str:
    """Return a non-reversible digest of the only aggregate inputs used."""

    prototypes, domain_shift, observed = _extract_aggregate(source_aggregate)
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": "phase1_grb_jp4_source_aggregate_generation_digest_v1",
                "feature_schema": FEATURE_SCHEMA,
                "prototypes_sha256": _array_digest(prototypes),
                "observed_domain_shift_sha256": _array_digest(domain_shift[observed]),
            }
        )
    )


def _registry_sha256(class_registry: Sequence[str]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {"class_registry": [str(value) for value in class_registry]}
        )
    )


def _resource_audit(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    vector_bytes = int(
        payload["p_g_q"].nbytes
        + payload["p_g_scale"].nbytes
        + payload["l_g_q"].nbytes
        + payload["l_g_scale"].nbytes
        + payload["r_q"].nbytes
        + payload["r_scale"].nbytes
    )
    return {
        "p_g_numeric_payload_bytes": int(payload["p_g_q"].nbytes + payload["p_g_scale"].nbytes),
        "l_g_numeric_payload_bytes": int(payload["l_g_q"].nbytes + payload["l_g_scale"].nbytes),
        "r_numeric_payload_bytes": int(payload["r_q"].nbytes + payload["r_scale"].nbytes),
        "component_numeric_payload_bytes": int(vector_bytes),
        "stage2_theta_numeric_payload_bytes": 6,
        "component_plus_theta_numeric_payload_bytes": int(vector_bytes + 6),
        "persistent_dense_float_bank_bytes": 0,
        "ground_direction_rank": RANK,
    }


def _validate_payload(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(payload) != ALLOWED_NPZ_MEMBERS:
        raise ValueError("payload member set does not match strict allowlist")
    p_q = np.asarray(payload["p_g_q"])
    p_scale = np.asarray(payload["p_g_scale"])
    l_q = np.asarray(payload["l_g_q"])
    l_scale = np.asarray(payload["l_g_scale"])
    r_q = np.asarray(payload["r_q"])
    r_scale = np.asarray(payload["r_scale"])
    if p_q.shape != (CLASS_COUNT, FEATURE_DIM) or l_q.shape != (RANK, FEATURE_DIM) or r_q.shape != (RANK, HIDDEN_DIM):
        raise ValueError("payload compact factor shape drift")
    if p_scale.shape != (CLASS_COUNT,) or l_scale.shape != (RANK,) or r_scale.shape != (RANK,):
        raise ValueError("payload per-vector scale shape drift")
    for field, array in (("p_g_q", p_q), ("l_g_q", l_q), ("r_q", r_q)):
        if array.dtype != np.int8 or bool(np.any(array == -128)):
            raise ValueError(f"{field} must be symmetric INT8 without -128")
    for field, array in (("p_g_scale", p_scale), ("l_g_scale", l_scale), ("r_scale", r_scale)):
        if array.dtype != np.float16 or not np.isfinite(array).all() or bool(np.any(array <= 0.0)):
            raise ValueError(f"{field} must be finite positive FP16")
    classes = np.asarray(payload["class_registry"])
    if classes.ndim != 1 or classes.dtype.kind not in {"U", "S"}:
        raise ValueError("class_registry must be a non-object string vector")
    registry = tuple(str(value) for value in classes.tolist())
    _class_binding_digest(registry)
    schema = np.asarray(payload["feature_schema"])
    if schema.shape != () or schema.dtype.kind not in {"U", "S"} or str(schema.item()) != FEATURE_SCHEMA:
        raise ValueError("feature schema drift")
    return {"class_registry": registry, "resource_audit": _resource_audit(payload)}


def build_grb_jp4_component(
    source_aggregate: Mapping[str, Any],
    *,
    class_registry: Sequence[str],
    checkpoint_joint_proj_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
    class_handle_binding_sha256: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
    provenance_status: str,
    formal_phase2_eligible: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create the fixed q4 ground/checkpoint component before target access."""

    if formal_phase2_eligible:
        raise ValueError("standalone GRB component cannot be formally Phase2 eligible before outer joint seal")
    registry = tuple(str(value) for value in class_registry)
    actual_binding = _class_binding_digest(registry)
    binding_sha = _validate_sha256(class_handle_binding_sha256, "class_handle_binding_sha256")
    if binding_sha != actual_binding:
        raise ValueError("class_handle_binding_sha256 does not bind class_registry order")
    checkpoint_sha = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    code_sha = _validate_sha256(generation_code_sha256, "generation_code_sha256")
    config_sha = _validate_sha256(generation_config_sha256, "generation_config_sha256")
    prototypes, shifts, observed = _extract_aggregate(source_aggregate)
    centered = shifts[observed].astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    l_g, singular = _canonical_svd_rows(centered, rank=RANK, field="centered aggregate domain_shift")
    r = _canonical_right_svd(checkpoint_joint_proj_weight)
    p_q, p_scale = _quantize_vectors(prototypes, "p_g")
    l_q, l_scale = _quantize_vectors(l_g, "l_g")
    r_q, r_scale = _quantize_vectors(r, "r")
    payload = {
        "p_g_q": p_q,
        "p_g_scale": p_scale,
        "l_g_q": l_q,
        "l_g_scale": l_scale,
        "r_q": r_q,
        "r_scale": r_scale,
        "class_registry": np.asarray(registry, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
    }
    details = _validate_payload(payload)
    generation_digest = source_aggregate_generation_digest(source_aggregate)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "component_profile": COMPONENT_PROFILE,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "class_count": CLASS_COUNT,
        "rank": RANK,
        "kappa_g": float(singular[0] / singular[RANK - 1]),
        "checkpoint_sha256": checkpoint_sha,
        "class_handle_binding_sha256": binding_sha,
        "source_aggregate_generation_digest_sha256": generation_digest,
        "generation_code_sha256": code_sha,
        "generation_config_sha256": config_sha,
        "registry_sha256": _registry_sha256(registry),
        "provenance_status": str(provenance_status),
        "component_state": PENDING_OUTER_JOINT_SEAL,
        "outer_bundle_signature_required": True,
        "formal_phase2_eligible": False,
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
        },
        "resource_audit": details["resource_audit"],
        "phase2_authorized_phase1_model_knowledge_policy": SCHEMA,
        "phase2_phase1_component_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_component_payload": "int8_p_g_l_g_r_fp16_scales_plus_kappa_g_and_class_registry_only",
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "phase2_phase1_component_member_or_exemplar_access": False,
        "phase2_phase1_component_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
    }
    if set(manifest) != BASE_MANIFEST_FIELDS:
        raise AssertionError("internal GRB manifest field drift")
    return payload, manifest


def _pre_sign_content_root(manifest: Mapping[str, Any], npz_sha256: str) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": manifest["schema"],
                "component_profile": manifest["component_profile"],
                "component_state": manifest["component_state"],
                "outer_bundle_signature_required": manifest["outer_bundle_signature_required"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "class_handle_binding_sha256": manifest["class_handle_binding_sha256"],
                "source_aggregate_generation_digest_sha256": manifest["source_aggregate_generation_digest_sha256"],
                "generation_code_sha256": manifest["generation_code_sha256"],
                "generation_config_sha256": manifest["generation_config_sha256"],
                "registry_sha256": manifest["registry_sha256"],
                # The inner filename is distinct from the fixed outer slot.
                # Bind it here so a profile cannot be relabelled after the
                # compact component has been generated and locked.
                "member_allowlist": manifest["member_allowlist"],
                "npz_member_allowlist": manifest["npz_member_allowlist"],
                "component_npz_sha256": npz_sha256,
            }
        )
    )


def save_grb_jp4_component(
    output_dir: str | Path,
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    details = _validate_payload(payload)
    if set(manifest) != BASE_MANIFEST_FIELDS or manifest.get("schema") != SCHEMA:
        raise ValueError("GRB base manifest schema mismatch")
    if manifest.get("component_profile") != COMPONENT_PROFILE:
        raise ValueError("GRB component profile drift")
    if manifest.get("formal_phase2_eligible") is not False or manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL:
        raise ValueError("standalone GRB component formal state drift")
    if manifest.get("class_handle_binding_sha256") != _class_binding_digest(details["class_registry"]):
        raise ValueError("GRB class registry binding drift")
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
        raise AssertionError("internal GRB final manifest field drift")
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = sha256_file(manifest_path)
    (root / MANIFEST_SHA_NAME).write_text(
        f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    validate_grb_jp4_component(root)
    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "component_npz_sha256": npz_sha,
        "manifest_sha256": manifest_sha,
        "pre_sign_content_root_sha256": final["pre_sign_content_root_sha256"],
    }


def validate_grb_jp4_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_class_handle_binding_sha256: str | None = None,
    expected_pre_sign_content_root_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(component_dir)
    actual_members = {item.name for item in root.iterdir()} if root.is_dir() else set()
    if actual_members != {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}:
        raise ValueError("GRB component directory member allowlist mismatch")
    manifest_path = root / MANIFEST_NAME
    if (root / MANIFEST_SHA_NAME).read_text(encoding="ascii") != f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n":
        raise ValueError("GRB manifest SHA256 sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != FINAL_MANIFEST_FIELDS or manifest.get("schema") != SCHEMA:
        raise ValueError("GRB final manifest schema mismatch")
    if manifest.get("component_profile") != COMPONENT_PROFILE or manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("GRB component profile or feature schema drift")
    if (int(manifest.get("feature_dim", -1)), int(manifest.get("hidden_dim", -1)), int(manifest.get("class_count", -1)), int(manifest.get("rank", -1))) != (FEATURE_DIM, HIDDEN_DIM, CLASS_COUNT, RANK):
        raise ValueError("GRB compact shape contract drift")
    try:
        kappa_g = float(manifest.get("kappa_g"))
    except (TypeError, ValueError) as exc:
        raise ValueError("GRB kappa_g scalar drift") from exc
    if not np.isfinite(kappa_g) or kappa_g < 1.0:
        raise ValueError("GRB kappa_g must be finite and at least one")
    if manifest.get("formal_phase2_eligible") is not False or manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL or manifest.get("outer_bundle_signature_required") is not True:
        raise ValueError("GRB standalone component must remain pending outer joint seal")
    if manifest.get("member_allowlist") != [NPZ_NAME] or manifest.get("npz_member_allowlist") != sorted(ALLOWED_NPZ_MEMBERS):
        raise ValueError("GRB component allowlist drift")
    for field in (
        "checkpoint_sha256", "class_handle_binding_sha256", "source_aggregate_generation_digest_sha256",
        "generation_code_sha256", "generation_config_sha256", "registry_sha256",
        "component_npz_sha256", "pre_sign_content_root_sha256",
    ):
        _validate_sha256(manifest.get(field), field)
    if manifest.get("svd_sign_canonicalization") != SVD_SIGN_SCHEMA or manifest.get("rounding_rule") != ROUNDING_SCHEMA:
        raise ValueError("GRB canonical SVD or rounding schema drift")
    if manifest.get("quantization") != {
        "dtype": "int8", "scale_dtype": "float16", "mode": "symmetric_per_vector",
        "qmin": -127, "qmax": 127,
    }:
        raise ValueError("GRB quantization contract drift")
    expected_policy = {
        "phase2_authorized_phase1_model_knowledge_policy": SCHEMA,
        "phase2_phase1_component_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_component_payload": "int8_p_g_l_g_r_fp16_scales_plus_kappa_g_and_class_registry_only",
        "phase2_phase1_component_immutable": True,
        "phase2_phase1_component_update_access": False,
        "phase2_phase1_component_member_or_exemplar_access": False,
        "phase2_phase1_component_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
    }
    if any(manifest.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("GRB protocol manifest drift")
    npz_path = root / NPZ_NAME
    npz_sha = sha256_file(npz_path)
    if manifest["component_npz_sha256"] != npz_sha or manifest["pre_sign_content_root_sha256"] != _pre_sign_content_root(manifest, npz_sha):
        raise ValueError("GRB component digest binding mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != ALLOWED_NPZ_MEMBERS:
            raise ValueError("GRB NPZ contains non-allowlisted members")
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    details = _validate_payload(payload)
    if manifest["class_handle_binding_sha256"] != _class_binding_digest(details["class_registry"]):
        raise ValueError("GRB manifest class registry binding drift")
    if manifest["registry_sha256"] != _registry_sha256(details["class_registry"]):
        raise ValueError("GRB manifest registry digest drift")
    for key, value in details["resource_audit"].items():
        if manifest["resource_audit"].get(key) != value:
            raise ValueError(f"GRB resource audit mismatch for {key}")
    if int(manifest["serialized_component_bytes"]) != npz_path.stat().st_size or int(manifest["resource_audit"].get("serialized_component_bytes", -1)) != npz_path.stat().st_size:
        raise ValueError("GRB serialized byte receipt drift")
    for expected, field in (
        (expected_checkpoint_sha256, "checkpoint_sha256"),
        (expected_class_handle_binding_sha256, "class_handle_binding_sha256"),
        (expected_pre_sign_content_root_sha256, "pre_sign_content_root_sha256"),
    ):
        if expected is not None and manifest[field] != _validate_sha256(expected, field):
            raise ValueError(f"{field} binding mismatch")
    return manifest


@dataclass(frozen=True)
class GRBJP4CompactComponent:
    """Read-only dequantized factor access; no source aggregate is retained."""

    p_g_q: np.ndarray
    p_g_scale: np.ndarray
    l_g_q: np.ndarray
    l_g_scale: np.ndarray
    r_q: np.ndarray
    r_scale: np.ndarray
    kappa_g: float
    class_registry: tuple[str, ...]
    manifest: Mapping[str, Any]

    @staticmethod
    def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
        copied = np.ascontiguousarray(value, dtype=dtype)
        result = np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
        result.setflags(write=False)
        return result

    def ground_prototypes(self) -> np.ndarray:
        return self._readonly(self.p_g_q.astype(np.float32) * self.p_g_scale.astype(np.float32)[:, None], np.float32)

    def ground_left_factors(self) -> np.ndarray:
        return self._readonly(self.l_g_q.astype(np.float32) * self.l_g_scale.astype(np.float32)[:, None], np.float32)

    def checkpoint_right_factors(self) -> np.ndarray:
        return self._readonly(self.r_q.astype(np.float32) * self.r_scale.astype(np.float32)[:, None], np.float32)


def load_grb_jp4_component(
    component_dir: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_pre_sign_content_root_sha256: str,
    allow_pending_outer_joint_seal_development: bool = False,
) -> GRBJP4CompactComponent:
    manifest = validate_grb_jp4_component(
        component_dir,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_class_handle_binding_sha256=expected_class_handle_binding_sha256,
        expected_pre_sign_content_root_sha256=expected_pre_sign_content_root_sha256,
    )
    if not allow_pending_outer_joint_seal_development:
        raise ValueError("formal GRB component load requires the existing outer joint seal")
    with np.load(Path(component_dir) / NPZ_NAME, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    details = _validate_payload(payload)
    return GRBJP4CompactComponent(
        p_g_q=GRBJP4CompactComponent._readonly(payload["p_g_q"], np.int8),
        p_g_scale=GRBJP4CompactComponent._readonly(payload["p_g_scale"], np.float16),
        l_g_q=GRBJP4CompactComponent._readonly(payload["l_g_q"], np.int8),
        l_g_scale=GRBJP4CompactComponent._readonly(payload["l_g_scale"], np.float16),
        r_q=GRBJP4CompactComponent._readonly(payload["r_q"], np.int8),
        r_scale=GRBJP4CompactComponent._readonly(payload["r_scale"], np.float16),
        kappa_g=float(manifest["kappa_g"]),
        class_registry=details["class_registry"],
        manifest=dict(manifest),
    )


__all__ = [
    "ALLOWED_NPZ_MEMBERS", "CLASS_COUNT", "COMPONENT_PROFILE", "FEATURE_DIM",
    "GRBJP4CompactComponent", "HIDDEN_DIM", "NPZ_NAME", "PENDING_OUTER_JOINT_SEAL",
    "RANK", "SCHEMA", "build_grb_jp4_component", "load_grb_jp4_component",
    "save_grb_jp4_component", "source_aggregate_generation_digest", "validate_grb_jp4_component",
]
