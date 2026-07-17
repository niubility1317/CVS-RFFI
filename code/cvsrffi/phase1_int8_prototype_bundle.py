"""Strict Phase1 int8 domain-class prototype deployment component.

The component intentionally retains only many-to-one aggregate centroids. It
must not retain source samples, sample-level features, counts, paths, or other
queryable source state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "phase1_int8_domain_class_centroids_v1"
NPZ_NAME = "int8_domain_class_prototypes.npz"
ALLOWED_NPZ_MEMBERS = {
    "domain_class_q",
    "domain_class_scale",
    "domain_class_mask",
    "domain_registry",
    "class_registry",
    "feature_schema",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return normalized


def quantize_domain_class_centroids(
    prototypes: torch.Tensor | np.ndarray,
    active_mask: torch.Tensor | np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Quantize source layout [class, domain, feature] into [domain, class, feature]."""

    proto = np.asarray(torch.as_tensor(prototypes).detach().float().cpu().numpy(), dtype=np.float32)
    mask = np.asarray(torch.as_tensor(active_mask).detach().cpu().numpy(), dtype=bool)
    if proto.ndim != 3:
        raise ValueError("prototypes must have shape [class, domain, feature]")
    if mask.shape != proto.shape[:2]:
        raise ValueError("active_mask must match [class, domain]")
    if not np.isfinite(proto).all():
        raise ValueError("prototypes must be finite")

    vectors = np.transpose(proto, (1, 0, 2)).copy()
    valid = np.transpose(mask, (1, 0)).copy()
    max_abs = np.max(np.abs(vectors), axis=-1)
    scale32 = np.where(valid & (max_abs > 0.0), max_abs / 127.0, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(vectors / scale32[..., None]), -127, 127).astype(np.int8)
    quantized[~valid] = 0
    scale16 = scale32.astype(np.float16)
    scale16[~valid] = np.float16(1.0)

    reference = vectors[valid]
    restored = quantized[valid].astype(np.float32) * scale16[valid, None].astype(np.float32)
    ref_norm = np.linalg.norm(reference, axis=1)
    out_norm = np.linalg.norm(restored, axis=1)
    denom = np.maximum(ref_norm * out_norm, 1e-12)
    cosine = np.sum(reference * restored, axis=1) / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    payload = {
        "domain_class_q": quantized,
        "domain_class_scale": scale16,
        "domain_class_mask": valid.astype(np.uint8),
    }
    audit = {
        "active_domain_class_cells": int(valid.sum()),
        "float32_centroid_bytes": int(reference.nbytes),
        "active_int8_payload_bytes": int(quantized[valid].nbytes),
        "active_fp16_scale_bytes": int(scale16[valid].nbytes),
        "dense_int8_tensor_bytes": int(quantized.nbytes),
        "dense_fp16_scale_bytes": int(scale16.nbytes),
        "mask_bytes": int(valid.nbytes),
        "logical_dense_state_bytes": int(quantized.nbytes + scale16.nbytes + valid.nbytes),
        "mean_cosine": float(cosine.mean()) if cosine.size else 1.0,
        "min_cosine": float(cosine.min()) if cosine.size else 1.0,
        "mean_angle_error_deg": float(angle.mean()) if angle.size else 0.0,
        "max_angle_error_deg": float(angle.max()) if angle.size else 0.0,
    }
    return payload, audit


def build_int8_component(
    source_package: Mapping[str, Any],
    *,
    class_registry: Sequence[str],
    checkpoint_sha256: str,
    source_prototype_artifact_sha256: str,
    provenance_status: str,
    formal_phase2_eligible: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prototypes = source_package.get("tx_domain_prototypes")
    counts = source_package.get("tx_domain_counts")
    if not torch.is_tensor(prototypes) or prototypes.ndim != 3:
        raise ValueError("source package requires tx_domain_prototypes[class,domain,feature]")
    if not torch.is_tensor(counts) or tuple(counts.shape) != tuple(prototypes.shape[:2]):
        raise ValueError("source package requires matching tx_domain_counts")
    if str(source_package.get("feature_key", "")) != "z_id":
        raise ValueError("only z_id prototypes are authorized")
    if len(class_registry) != int(prototypes.shape[0]):
        raise ValueError("class_registry length must match prototype class dimension")
    if len(set(map(str, class_registry))) != len(class_registry):
        raise ValueError("class_registry entries must be unique")

    checkpoint_hash = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    source_hash = _validate_sha256(source_prototype_artifact_sha256, "source_prototype_artifact_sha256")
    payload, audit = quantize_domain_class_centroids(prototypes, counts > 0)
    domain_count, class_count, feature_dim = payload["domain_class_q"].shape
    payload.update(
        {
            "domain_registry": np.arange(domain_count, dtype=np.int16),
            "class_registry": np.asarray([str(v) for v in class_registry], dtype=np.str_),
            "feature_schema": np.asarray("ADV3B02:z_id:unit_l2:160:v1", dtype=np.str_),
        }
    )

    registry_sha256 = _canonical_sha256(
        {"class_registry": [str(v) for v in class_registry], "domain_registry": list(range(domain_count))}
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "feature_key": "z_id",
        "feature_dim": int(feature_dim),
        "class_count": int(class_count),
        "domain_count": int(domain_count),
        "active_domain_class_cells": int(audit["active_domain_class_cells"]),
        "checkpoint_sha256": checkpoint_hash,
        "source_prototype_artifact_sha256": source_hash,
        "registry_sha256": registry_sha256,
        "provenance_status": str(provenance_status),
        "formal_phase2_eligible": bool(formal_phase2_eligible),
        "phase2_pretrained_artifact_policy": (
            "sealed_phase1_deployment_bundle_with_optional_int8_domain_class_prototypes_v1"
        ),
        "phase2_authorized_phase1_model_knowledge_policy": "int8_domain_class_centroids_v1",
        "phase2_phase1_prototype_generation_stage": "phase1_offline_before_target_access",
        "phase2_phase1_prototype_payload": "int8_centroid_fp16_scale_registry_only",
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "phase2_phase1_prototype_sample_reconstruction_access": False,
        "phase2_nonbundle_source_artifact_access": False,
        "member_allowlist": [NPZ_NAME],
        "npz_member_allowlist": sorted(ALLOWED_NPZ_MEMBERS),
        "quantization": {
            "dtype": "int8",
            "scale_dtype": "float16",
            "mode": "symmetric_per_domain_class_vector",
            "qmin": -127,
            "qmax": 127,
        },
        "resource_audit": audit,
    }
    return payload, manifest


def save_int8_component(
    output_dir: str | Path,
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if set(payload) != ALLOWED_NPZ_MEMBERS:
        raise ValueError("payload member set does not match strict allowlist")
    npz_path = target / NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    npz_sha256 = sha256_file(npz_path)

    final_manifest = dict(manifest)
    final_manifest["component_npz_sha256"] = npz_sha256
    final_manifest["serialized_component_bytes"] = int(npz_path.stat().st_size)
    final_manifest["deployment_bundle_root_sha256"] = _canonical_sha256(
        {
            "schema": final_manifest["schema"],
            "checkpoint_sha256": final_manifest["checkpoint_sha256"],
            "component_npz_sha256": npz_sha256,
            "registry_sha256": final_manifest["registry_sha256"],
        }
    )
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = sha256_file(manifest_path)
    (target / "manifest.sha256").write_text(f"{manifest_sha256}  manifest.json\n", encoding="ascii")
    validate_int8_component(target)
    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": npz_sha256,
        "manifest_sha256": manifest_sha256,
        "deployment_bundle_root_sha256": final_manifest["deployment_bundle_root_sha256"],
    }


def validate_int8_component(component_dir: str | Path) -> dict[str, Any]:
    root = Path(component_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected int8 prototype schema")
    if manifest.get("member_allowlist") != [NPZ_NAME]:
        raise ValueError("component member allowlist mismatch")
    if set(manifest.get("npz_member_allowlist", [])) != ALLOWED_NPZ_MEMBERS:
        raise ValueError("NPZ member allowlist mismatch")
    npz_path = root / NPZ_NAME
    if sha256_file(npz_path) != manifest.get("component_npz_sha256"):
        raise ValueError("component NPZ SHA256 mismatch")

    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != ALLOWED_NPZ_MEMBERS:
            raise ValueError("NPZ contains non-allowlisted members")
        q = arrays["domain_class_q"]
        scale = arrays["domain_class_scale"]
        mask = arrays["domain_class_mask"]
        if q.dtype != np.int8 or q.ndim != 3:
            raise ValueError("domain_class_q must be int8[D,C,P]")
        if scale.dtype != np.float16 or scale.shape != q.shape[:2]:
            raise ValueError("domain_class_scale must be float16[D,C]")
        if mask.dtype != np.uint8 or mask.shape != q.shape[:2]:
            raise ValueError("domain_class_mask must be uint8[D,C]")
        if arrays["class_registry"].dtype.kind not in {"U", "S"}:
            raise ValueError("class_registry must be a non-object string array")
        if arrays["domain_registry"].dtype != np.int16:
            raise ValueError("domain_registry must be int16")
        if bool(np.any((mask != 0) & (mask != 1))):
            raise ValueError("domain_class_mask must be binary")
        if bool(np.any(q[mask == 0] != 0)):
            raise ValueError("inactive prototype slots must be zero")
        if not np.isfinite(scale).all() or bool(np.any(scale <= 0)):
            raise ValueError("all quantization scales must be finite and positive")
    return manifest
