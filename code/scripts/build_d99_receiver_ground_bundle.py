#!/usr/bin/env python3
"""Build a seven-receiver non-formal D99 ground bundle from D19 aggregates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_int8_prototype_bundle import (  # noqa: E402
    ALLOWED_NPZ_MEMBERS,
    NPZ_NAME as SOURCE_NPZ_NAME,
    SCHEMA as SOURCE_SCHEMA,
    validate_int8_component,
)
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99  # noqa: E402
from cvsrffi import stage2_d99_d100_phase1_lodo as lodo  # noqa: E402


SPEC_SCHEMA = "cvs.development.d99_receiver_ground_aggregation.v1"
RESULT_SCHEMA = "cvs.development.d99_receiver_ground_bundle_result.v1"
STATUS = "NONFORMAL_PHASE1_AGGREGATE_DIAGNOSTIC"
OUTPUT_NPZ = "d99_ground_bundle_dev.npz"
OUTPUT_MANIFEST = "d99_ground_bundle_dev.manifest.json"
OUTPUT_SPEC = "d99_ground_aggregation_spec.json"
OUTPUT_LOCK = "d99_base_method_lock_dev.json"
OUTPUT_RESULT = "build_result.json"
DEVELOPMENT_D99_PRIOR_SCHEMA = "cvs.phase1.d99.development_prior_wrapper.v1"
DEVELOPMENT_D99_PRIOR_STATUS = "PREREGISTERED_DEVELOPMENT_PRIORS_NONFORMAL"
DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS = (
    "phase1_receipt_sha256",
    "quantization_margin_audit_sha256",
    "validation_method_lock_sha256",
    "d81_phase1_lock_sha256",
)
SOURCE_DOMAIN_PAIRS = (
    ("1-1", (0, 1)),
    ("1-19", (4, 5)),
    ("14-7", (8, 9)),
    ("18-2", (12, 13)),
    ("19-2", (16, 17)),
    ("2-1", (20, 21)),
    ("2-19", (24, 25)),
)
OUTPUT_MEMBERS = (
    "codes_qint8",
    "scales_fp16",
    "domain_class_mask",
    "physical_sample_count_floor_uint16",
    "domain_ids",
    "ground_old_registry",
)
OLD_TX_REGISTRY = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")


class D99GroundBundleBuildError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    text = str(value)
    if (
        text != text.lower()
        or len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise D99GroundBundleBuildError(f"{name} must be lowercase SHA256")
    return text


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != OUTPUT_MEMBERS:
        raise D99GroundBundleBuildError("D99 output NPZ member order drift")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in OUTPUT_MEMBERS:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100444 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    return buffer.getvalue()


def _read_source(
    component_dir: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], str, str]:
    manifest_path = component_dir / "manifest.json"
    expected = _require_sha(expected_manifest_sha256, "source manifest")
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha256_file(manifest_path) != expected
    ):
        raise D99GroundBundleBuildError("source manifest path/SHA drift")
    try:
        manifest = validate_int8_component(component_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise D99GroundBundleBuildError("source D19 component validation failed") from exc
    npz_path = component_dir / SOURCE_NPZ_NAME
    npz_sha = _require_sha(manifest.get("component_npz_sha256"), "source NPZ")
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("checkpoint_sha256") != BASE_CHECKPOINT_SHA256
        or manifest.get("domain_count") != 26
        or manifest.get("class_count") != 6
        or manifest.get("feature_dim") != d99.Z_DIM
        or set(manifest.get("npz_member_allowlist", ())) != ALLOWED_NPZ_MEMBERS
        or not npz_path.is_file()
        or npz_path.is_symlink()
        or _sha256_file(npz_path) != npz_sha
    ):
        raise D99GroundBundleBuildError("source D19 component contract drift")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != ALLOWED_NPZ_MEMBERS:
            raise D99GroundBundleBuildError("source NPZ member allowlist drift")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return manifest, arrays, expected, npz_sha


def _aggregate_receiver_rows(
    source: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    codes = np.asarray(source["domain_class_q"])
    scales = np.asarray(source["domain_class_scale"])
    mask = np.asarray(source["domain_class_mask"]).astype(bool)
    registry = np.asarray(source["domain_registry"])
    classes = np.asarray(source["class_registry"]).astype(str)
    schema = str(np.asarray(source["feature_schema"]).item())
    active_rows = tuple(np.flatnonzero(np.all(mask, axis=1)).tolist())
    any_active_rows = tuple(np.flatnonzero(np.any(mask, axis=1)).tolist())
    expected_rows = tuple(index for _receiver, pair in SOURCE_DOMAIN_PAIRS for index in pair)
    if (
        codes.dtype != np.int8
        or codes.shape != (26, 6, d99.Z_DIM)
        or scales.dtype != np.float16
        or scales.shape != (26, 6)
        or mask.shape != (26, 6)
        or tuple(registry.tolist()) != tuple(range(26))
        or len(classes) != 6
        or tuple(classes.tolist()) != OLD_TX_REGISTRY
        or schema != "ADV3B02:z_id:unit_l2:160:v1"
        or active_rows != expected_rows
        or any_active_rows != expected_rows
        or not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or np.any(codes == -128)
    ):
        raise D99GroundBundleBuildError("source D19 array layout drift")
    decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, :, None]
    receiver_vectors = []
    for _receiver, pair in SOURCE_DOMAIN_PAIRS:
        rows = decoded[np.asarray(pair, dtype=np.int64)]
        norms = np.linalg.norm(rows, axis=2, keepdims=True)
        if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
            raise D99GroundBundleBuildError("source D19 active vector norm drift")
        rows = rows / norms
        center = np.mean(rows, axis=0)
        center_norm = np.linalg.norm(center, axis=1, keepdims=True)
        if np.any(center_norm <= 1e-12) or not np.isfinite(center_norm).all():
            raise D99GroundBundleBuildError("receiver spherical mean norm drift")
        receiver_vectors.append(center / center_norm)
    vectors = np.asarray(receiver_vectors, dtype=np.float32)
    maximum = np.max(np.abs(vectors), axis=2)
    out_scales = np.maximum(maximum / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    out_codes = np.clip(
        np.rint(vectors / out_scales.astype(np.float32)[:, :, None]), -127, 127
    ).astype(np.int8)
    restored = out_codes.astype(np.float32) * out_scales.astype(np.float32)[:, :, None]
    cosine = np.sum(vectors * restored, axis=2) / np.maximum(
        np.linalg.norm(vectors, axis=2) * np.linalg.norm(restored, axis=2), 1e-12
    )
    output = {
        "codes_qint8": out_codes,
        "scales_fp16": out_scales,
        "domain_class_mask": np.ones((7, 6), dtype=np.bool_),
        "physical_sample_count_floor_uint16": np.full((7, 6), 2, dtype=np.uint16),
        "domain_ids": np.asarray([receiver for receiver, _pair in SOURCE_DOMAIN_PAIRS]),
        "ground_old_registry": classes,
    }
    return output, {
        "mean_requantization_cosine": float(np.mean(cosine)),
        "min_requantization_cosine": float(np.min(cosine)),
    }


def _base_lock(bundle: d99.Phase1GroundAggregateBundle, spec_sha256: str) -> d99.Phase1D99Lock:
    return d99.Phase1D99Lock(
        density_tau=0.2,
        max_ground_rank=2,
        max_target_rank=2,
        coverage_floor=0.01,
        ground_energy_scale=0.01,
        target_energy_scale=0.01,
        shrinkage_prior_strength=2.0,
        ground_weight_max=0.8,
        target_weight_max=0.6,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.5,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        z_weight=0.7,
        fft_weight=0.2,
        rf_weight=0.1,
        eta_k1=0.0,
        eta_k5=0.0,
        eta_k10=0.0,
        eta_k20=0.0,
        eta_k20_lodo_artifact_sha256=None,
        phase1_receipt_sha256=spec_sha256,
        ground_aggregation_receipt_sha256=bundle.aggregation_receipt.receipt_sha256,
        ground_bundle_receipt_sha256=bundle.bundle_sha256,
        quantization_margin_audit_sha256=spec_sha256,
        validation_method_lock_sha256=spec_sha256,
        d81_phase1_lock_sha256=spec_sha256,
        ground_old_registry=bundle.ground_old_registry,
    )


def build_bundle(
    component_dir: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_root = Path(component_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise D99GroundBundleBuildError("output directory already exists")
    _manifest, source, manifest_sha, npz_sha = _read_source(
        source_root, expected_manifest_sha256
    )
    arrays, audit = _aggregate_receiver_rows(source)
    producer_sha = _sha256_file(Path(__file__).resolve())
    spec = {
        "schema": SPEC_SCHEMA,
        "status": STATUS,
        "lifecycle": "phase1_offline_before_target_access",
        "source_component_manifest_sha256": manifest_sha,
        "source_component_npz_sha256": npz_sha,
        "producer_code_sha256": producer_sha,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "source_domain_count": 14,
        "receiver_domain_count": 7,
        "class_count": 6,
        "source_domain_pairs": [
            {"receiver": receiver, "source_domain_rows": list(pair)}
            for receiver, pair in SOURCE_DOMAIN_PAIRS
        ],
        "aggregation_formula": (
            "decode_fp32_then_l2_each_day_then_equal_0.5_spherical_mean_"
            "then_l2_then_symmetric_per_vector_int8"
        ),
        "pair_weights": [0.5, 0.5],
        "density_weighting_deferred_to_d99_receiver_geometry": True,
        "physical_sample_count_floor_semantics": (
            "two_preaggregated_day_domain_centroids_each_created_by_the_"
            "D19_builder_minimum_two_physical_rows_invariant_not_exact_member_count"
        ),
        "source_component_full_validator_used": True,
        "member_ids_present": False,
        "sample_level_features_present": False,
        "raw_or_clean_iq_present": False,
        "target_rows_used": 0,
        "query_rows_used": 0,
        "formal_phase1_eligible": False,
        **audit,
    }
    spec_bytes = _canonical_bytes(spec)
    spec_sha = _sha256_bytes(spec_bytes)
    receipt_payload = {
        "schema": d99.GROUND_AGGREGATION_RECEIPT_SCHEMA,
        "aggregation_manifest_sha256": spec_sha,
        "producer_code_sha256": producer_sha,
        "phase1_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "minimum_physical_sample_count": 2,
        "member_ids_present": False,
        "target_rows_used": 0,
        "cryptographic_external_authority_claimed": False,
    }
    receipt = d99.ExternalGroundAggregationReceipt(
        **receipt_payload,
        receipt_sha256=_sha256_bytes(_canonical_bytes(receipt_payload)),
    )
    bundle = d99.produce_typed_ground_aggregate_bundle(
        **arrays,
        aggregation_receipt=receipt,
    )
    receiver_map = {receiver: receiver for receiver, _pair in SOURCE_DOMAIN_PAIRS}
    release = lodo.ground_release_manifest_payload(
        bundle,
        receiver_map,
        producer_code_sha256=producer_sha,
        release_schema=lodo.GROUND_RELEASE_DEVELOPMENT_SCHEMA,
        release_status=lodo.GROUND_RELEASE_DEVELOPMENT_STATUS,
    )
    release_bytes = lodo._canonical_bytes(release)
    lock = _base_lock(bundle, spec_sha)
    lock_wrapper = {
        "schema": DEVELOPMENT_D99_PRIOR_SCHEMA,
        "status": DEVELOPMENT_D99_PRIOR_STATUS,
        "values": asdict(lock),
        "placeholder_evidence_fields": list(
            DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS
        ),
    }
    lock_bytes = _canonical_bytes(lock_wrapper)
    npz_bytes = _deterministic_npz(arrays)
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "formal_phase1_eligible": False,
        "output_npz_sha256": _sha256_bytes(npz_bytes),
        "output_manifest_sha256": _sha256_bytes(release_bytes),
        "aggregation_spec_sha256": spec_sha,
        "base_d99_lock_sha256": _sha256_bytes(lock_bytes),
        "bundle_sha256": bundle.bundle_sha256,
        "aggregation_receipt_sha256": receipt.receipt_sha256,
        "domain_count": 7,
        "class_count": 6,
        **audit,
    }
    result_bytes = _canonical_bytes(result)
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {
        OUTPUT_NPZ: npz_bytes,
        OUTPUT_MANIFEST: release_bytes,
        OUTPUT_SPEC: spec_bytes,
        OUTPUT_LOCK: lock_bytes,
        OUTPUT_RESULT: result_bytes,
    }
    for name, payload in outputs.items():
        descriptor = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return {**result, "output_dir": str(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-component-dir", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_bundle(
        args.source_component_dir, args.source_manifest_sha256, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
