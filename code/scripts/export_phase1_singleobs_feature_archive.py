#!/usr/bin/env python3
"""Export a temporary Phase1 single-observation archive for D97 LODO locking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache_set,
)
from cvsrffi import phase1_adv3b02_deployment_bundle as deployment_bundle  # noqa: E402
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    rf_statistics,
    spectral_logmag_sketch,
)


SCHEMA = "cvs.phase1.single_leo_feature_archive.v2"
DIAGNOSTIC_SCHEMA = "cvs.test_diagnostic.single_leo_feature_archive.v1"
RUNTIME_MANIFEST_SCHEMA = "cvs.phase1.singleobs_runtime_binding.v1"
RUNTIME_SCHEMA = "adv3b02.torchscript_identity_runtime.v1"
RUNTIME_EXPORT_RECEIPT_SCHEMA = "cvs.phase1.runtime_checkpoint_parity_receipt.v1"
SELECTION_SALT_RECEIPT_SCHEMA = "cvs.phase1.singleobs_selection_salt_receipt.v1"
NPZ_NAME = "phase1_singleobs_feature_archive.npz"
MANIFEST_NAME = "phase1_singleobs_feature_archive.manifest.json"
SELECTION_DOMAIN = b"P1_SINGLE_LEO_V1"
FEATURE_DIM = 160
FFT_DIM = 96
RF_DIM = 32
JOINT_DIM = FEATURE_DIM + FFT_DIM + RF_DIM
SOURCE_VALIDATION_SCOPE = "source_validation"
SOURCE_VALIDATION_ROLES = {"source"}
FORMAL_STATUS = "FORMAL_PHASE1_TEMPORARY_ASSET"
DEVELOPMENT_STATUS = "DEVELOPMENT_PHASE1_TEMPORARY_ASSET"
DIAGNOSTIC_STATUS = "TEST_DIAGNOSTIC_NOT_FORMAL"
DEVELOPMENT_SHA_ONLY_AUTHORITY_MODE = (
    "development_known_adv3b02_runtime_sha_no_parity"
)
KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256 = frozenset(
    {
        "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9",
        "f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a",
    }
)
OUTPUT_MEMBER_ALLOWLIST = (
    "features",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "checkpoint_reference_logits",
)


class Phase1SingleObservationArchiveError(ValueError):
    """Raised when Phase1 selection, lineage, or D97 archive shape drifts."""


ForwardCallback = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray] | Mapping[str, Any]]
CacheLoader = Callable[..., tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise Phase1SingleObservationArchiveError(f"{name} must be lowercase SHA256 hex")
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _read_json_bound(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, name=f"{name} SHA256")
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
        raise Phase1SingleObservationArchiveError(f"{name} path/SHA256 drift")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase1SingleObservationArchiveError(f"{name} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise Phase1SingleObservationArchiveError(f"{name} must be a JSON object")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        raise Phase1SingleObservationArchiveError("object arrays cannot be hashed")
    if array.dtype.kind in {"U", "S"}:
        header = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = _canonical_json_bytes(array.astype(str).tolist())
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        header = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        body = canonical.tobytes(order="C")
    return hashlib.sha256(_canonical_json_bytes(header) + b"\0" + body).hexdigest()


def selection_index(selection_salt_sha256: str, physical_id: str) -> int:
    salt = bytes.fromhex(_require_sha256(selection_salt_sha256, name="selection salt"))
    identifier = str(physical_id)
    if not identifier:
        raise Phase1SingleObservationArchiveError("physical_id must be nonempty")
    digest = hashlib.sha256(SELECTION_DOMAIN + salt + identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % 3


def _resolve_member(manifest_path: Path, raw_path: Any) -> Path:
    candidate = Path(str(raw_path))
    return (candidate if candidate.is_absolute() else manifest_path.parent / candidate).resolve()


def _load_runtime_binding(
    runtime_manifest_path: Path,
    runtime_manifest_sha256: str,
    *,
    require_known_development_runtime: bool = False,
    expected_runtime_sha256: str | None = None,
    expected_parity_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _read_json_bound(
        runtime_manifest_path, runtime_manifest_sha256, name="runtime manifest"
    )
    required = {
        "schema",
        "artifact_stage",
        "bundle_id",
        "phase1_checkpoint_sha256",
        "feature_runtime",
        "runtime_export_receipt",
        "feature_dims",
        "class_ids",
    }
    if set(payload) != required or payload.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        raise Phase1SingleObservationArchiveError("runtime manifest exact schema drift")
    if payload.get("artifact_stage") != "phase1_offline_before_target_access":
        raise Phase1SingleObservationArchiveError("runtime manifest stage drift")
    bundle_id = _require_sha256(payload["bundle_id"], name="runtime bundle_id")
    checkpoint = _require_sha256(
        payload["phase1_checkpoint_sha256"], name="Phase1 checkpoint"
    )
    if checkpoint != BASE_CHECKPOINT_SHA256:
        raise Phase1SingleObservationArchiveError("runtime is not bound to ADV3B02 checkpoint")
    runtime = payload["feature_runtime"]
    export = payload["runtime_export_receipt"]
    dims = payload["feature_dims"]
    classes = tuple(str(value) for value in payload["class_ids"])
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"path", "sha256", "schema"}
        or runtime.get("schema") != RUNTIME_SCHEMA
        or not isinstance(export, dict)
        or set(export) != {"path", "sha256", "schema"}
        or export.get("schema") != RUNTIME_EXPORT_RECEIPT_SCHEMA
        or not isinstance(dims, dict)
        or set(dims) != {
            "input_channels",
            "z160",
            "checkpoint_reference_logits",
            "features",
        }
        or dims.get("input_channels") != 2
        or dims.get("z160") != FEATURE_DIM
        or dims.get("features") != JOINT_DIM
        or dims.get("checkpoint_reference_logits") != len(classes)
        or len(classes) < 2
        or len(set(classes)) != len(classes)
    ):
        raise Phase1SingleObservationArchiveError("runtime feature/schema lineage drift")
    runtime_path = _resolve_member(runtime_manifest_path, runtime["path"])
    runtime_sha = _require_sha256(runtime["sha256"], name="TorchScript runtime")
    if not runtime_path.is_file() or runtime_path.is_symlink() or _sha256_file(runtime_path) != runtime_sha:
        raise Phase1SingleObservationArchiveError("TorchScript runtime path/SHA256 drift")
    receipt_path = _resolve_member(runtime_manifest_path, export["path"])
    receipt_sha = _require_sha256(export["sha256"], name="runtime-export receipt")
    receipt = _read_json_bound(receipt_path, receipt_sha, name="runtime-export receipt")
    receipt_keys = {
        "schema",
        "checkpoint_lineage_sha256",
        "runtime_sha256",
        "parity_status",
        "max_abs_output_delta",
        "parity_vector_root_sha256",
        "runtime_archive_member_root_sha256",
        "runtime_state_schema_root_sha256",
        "runtime_state_bytes",
        "runtime_structure_sha256",
    }
    sha_fields = (
        "parity_vector_root_sha256",
        "runtime_archive_member_root_sha256",
        "runtime_state_schema_root_sha256",
        "runtime_structure_sha256",
    )
    try:
        parity_delta = float(receipt.get("max_abs_output_delta"))
        runtime_state_bytes = int(receipt.get("runtime_state_bytes"))
    except (TypeError, ValueError) as exc:
        raise Phase1SingleObservationArchiveError(
            "runtime-export receipt numeric lineage drift"
        ) from exc
    if (
        set(receipt) != receipt_keys
        or receipt.get("schema") != RUNTIME_EXPORT_RECEIPT_SCHEMA
        or receipt.get("parity_status") != "PASS"
        or receipt.get("checkpoint_lineage_sha256") != checkpoint
        or receipt.get("runtime_sha256") != runtime_sha
        or not np.isfinite(parity_delta)
        or parity_delta < 0.0
        or parity_delta > 1.0e-5
        or runtime_state_bytes < 0
        or any(
            _require_sha256(receipt.get(name), name=f"runtime receipt {name}")
            != receipt.get(name)
            for name in sha_fields
        )
    ):
        raise Phase1SingleObservationArchiveError("runtime-export receipt lineage drift")
    if require_known_development_runtime:
        expected_runtime = _require_sha256(
            expected_runtime_sha256, name="expected development runtime"
        )
        expected_parity = _require_sha256(
            expected_parity_receipt_sha256,
            name="expected development parity receipt",
        )
        if (
            runtime_sha not in KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256
            or runtime_sha != expected_runtime
            or receipt_sha != expected_parity
        ):
            raise Phase1SingleObservationArchiveError(
                "development runtime is not a known ADV3B02 runtime/parity binding"
            )
    return {
        "manifest": payload,
        "manifest_path": runtime_manifest_path,
        "manifest_sha256": _require_sha256(
            runtime_manifest_sha256, name="runtime manifest"
        ),
        "bundle_id": bundle_id,
        "checkpoint_sha256": checkpoint,
        "runtime_path": runtime_path,
        "runtime_sha256": runtime_sha,
        "runtime_receipt_path": receipt_path,
        "runtime_receipt_sha256": receipt_sha,
        "feature_dims": dims,
        "class_ids": classes,
        "authority_mode": (
            "development_known_adv3b02_runtime_sha"
            if require_known_development_runtime
            else "test_diagnostic_self_described_runtime"
        ),
        "authority_binding_sha256": _sha256_file(runtime_manifest_path),
        "formal_outer_content_root_sha256": None,
        "detached_seal_sha256": None,
        "signature_envelope_sha256": None,
    }


def _load_known_runtime_sha_only(
    runtime_path: str | Path,
    expected_runtime_sha256: str,
    class_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Bind a known frozen runtime for a non-formal Phase1 diagnostic export."""

    path = Path(runtime_path).resolve()
    runtime_sha = _require_sha256(
        expected_runtime_sha256, name="expected development runtime"
    )
    classes = tuple(str(value) for value in class_ids)
    if (
        runtime_sha not in KNOWN_DEVELOPMENT_ADV3B02_RUNTIME_SHA256
        or not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != runtime_sha
    ):
        raise Phase1SingleObservationArchiveError(
            "development runtime is not a known SHA-bound ADV3B02 runtime"
        )
    if len(classes) < 2 or any(not value for value in classes) or len(set(classes)) != len(classes):
        raise Phase1SingleObservationArchiveError(
            "development runtime class registry drift"
        )
    binding = {
        "mode": DEVELOPMENT_SHA_ONLY_AUTHORITY_MODE,
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "runtime_sha256": runtime_sha,
        "class_ids": list(classes),
        "feature_dims": {
            "input_channels": 2,
            "z160": FEATURE_DIM,
            "checkpoint_reference_logits": len(classes),
            "features": JOINT_DIM,
        },
    }
    binding_sha = hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()
    return {
        "manifest": None,
        "manifest_path": None,
        "manifest_sha256": None,
        "bundle_id": binding_sha,
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "runtime_path": path,
        "runtime_sha256": runtime_sha,
        "runtime_receipt_path": None,
        "runtime_receipt_sha256": None,
        "feature_dims": binding["feature_dims"],
        "class_ids": classes,
        "authority_mode": DEVELOPMENT_SHA_ONLY_AUTHORITY_MODE,
        "authority_binding_sha256": binding_sha,
        "formal_outer_content_root_sha256": None,
        "detached_seal_sha256": None,
        "signature_envelope_sha256": None,
    }


def _load_formal_runtime_binding(
    package_root: str | Path,
    *,
    detached_seal_path: str | Path,
    expected_detached_seal_sha256: str,
    signature_envelope_path: str | Path,
    expected_signature_envelope_sha256: str,
    expected_checkpoint_lineage_sha256: str,
    expected_runtime_sha256: str,
    expected_component_pre_sign_content_root_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_parity_receipt_sha256: str,
    expected_generation_lock_sha256: str,
    expected_method_lock_sha256: str,
    expected_generation_config_sha256: str,
    expected_generation_code_sha256: str,
    expected_outer_content_root_sha256: str,
) -> dict[str, Any]:
    """Load runtime only through the repository's pinned external authority."""

    expected_checkpoint = _require_sha256(
        expected_checkpoint_lineage_sha256, name="formal checkpoint lineage"
    )
    if expected_checkpoint != BASE_CHECKPOINT_SHA256:
        raise Phase1SingleObservationArchiveError(
            "formal bundle is not bound to the ADV3B02 final checkpoint"
        )
    kwargs = {
        "detached_seal_path": detached_seal_path,
        "expected_detached_seal_sha256": expected_detached_seal_sha256,
        "signature_envelope_path": signature_envelope_path,
        "expected_signature_envelope_sha256": expected_signature_envelope_sha256,
        "expected_checkpoint_lineage_sha256": expected_checkpoint,
        "expected_runtime_sha256": expected_runtime_sha256,
        "expected_component_pre_sign_content_root_sha256": expected_component_pre_sign_content_root_sha256,
        "expected_class_handle_binding_sha256": expected_class_handle_binding_sha256,
        "expected_parity_receipt_sha256": expected_parity_receipt_sha256,
        "expected_generation_lock_sha256": expected_generation_lock_sha256,
        "expected_method_lock_sha256": expected_method_lock_sha256,
        "expected_generation_config_sha256": expected_generation_config_sha256,
        "expected_generation_code_sha256": expected_generation_code_sha256,
        "expected_outer_content_root_sha256": expected_outer_content_root_sha256,
    }
    try:
        verified = deployment_bundle.load_formal_adv3b02_deployment_bundle(
            package_root, **kwargs
        )
    except Exception as exc:
        raise Phase1SingleObservationArchiveError(
            "formal ADV3B02 outer-bundle authority verification failed"
        ) from exc
    context = dict(verified.formal_phase2_context)
    audit = dict(verified.audit)
    if (
        context.get("formal_phase2_eligible") is not True
        or context.get("outer_signature_verified") is not True
        or context.get("detached_seal_verified") is not True
        or context.get("runtime_checkpoint_parity_verified") is not True
        or context.get("checkpoint_lineage_sha256") != expected_checkpoint
        or context.get("runtime_sha256")
        != _require_sha256(expected_runtime_sha256, name="formal runtime")
        or audit.get("status") != "PASS"
    ):
        raise Phase1SingleObservationArchiveError(
            "formal ADV3B02 verifier did not return a complete authority context"
        )
    rows = verified.class_binding.get("class_id_to_handle")
    if not isinstance(rows, list) or not rows:
        raise Phase1SingleObservationArchiveError("formal class registry is missing")
    classes = tuple(str(row.get("class_handle", "")) for row in rows)
    if any(not value for value in classes) or len(set(classes)) != len(classes):
        raise Phase1SingleObservationArchiveError("formal class registry drift")
    parity_sha = _require_sha256(
        expected_parity_receipt_sha256, name="formal parity receipt"
    )
    authority_receipt = {
        "formal_phase2_context": context,
        "load_audit": audit,
        "detached_seal_sha256": _require_sha256(
            expected_detached_seal_sha256, name="formal detached seal"
        ),
        "signature_envelope_sha256": _require_sha256(
            expected_signature_envelope_sha256, name="formal signature envelope"
        ),
    }
    return {
        "model": verified.runtime,
        "bundle_id": _require_sha256(
            expected_outer_content_root_sha256, name="formal outer content root"
        ),
        "checkpoint_sha256": expected_checkpoint,
        "runtime_sha256": _require_sha256(
            expected_runtime_sha256, name="formal runtime"
        ),
        "runtime_receipt_sha256": parity_sha,
        "class_ids": classes,
        "authority_mode": "formal_adv3b02_outer_bundle",
        "authority_binding_sha256": hashlib.sha256(
            _canonical_json_bytes(authority_receipt)
        ).hexdigest(),
        "formal_outer_content_root_sha256": _require_sha256(
            expected_outer_content_root_sha256, name="formal outer content root"
        ),
        "detached_seal_sha256": authority_receipt["detached_seal_sha256"],
        "signature_envelope_sha256": authority_receipt[
            "signature_envelope_sha256"
        ],
    }


def _load_selection_salt(
    receipt_path: Path,
    receipt_sha256: str,
    *,
    runtime_binding: Mapping[str, Any],
) -> dict[str, str]:
    payload = _read_json_bound(receipt_path, receipt_sha256, name="selection-salt receipt")
    expected_keys = {
        "schema",
        "status",
        "artifact_stage",
        "bundle_id",
        "phase1_checkpoint_sha256",
        "selection_salt_sha256",
        "target_access",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SELECTION_SALT_RECEIPT_SCHEMA
        or payload.get("status") != "SEALED_BEFORE_TARGET_ACCESS"
        or payload.get("artifact_stage") != "phase1_offline_before_target_access"
        or payload.get("bundle_id") != runtime_binding["bundle_id"]
        or payload.get("phase1_checkpoint_sha256")
        != runtime_binding["checkpoint_sha256"]
        or payload.get("target_access") is not False
    ):
        raise Phase1SingleObservationArchiveError("selection-salt receipt lineage drift")
    return {
        "path": str(receipt_path),
        "sha256": _require_sha256(receipt_sha256, name="selection-salt receipt"),
        "schema": SELECTION_SALT_RECEIPT_SCHEMA,
        "selection_salt_sha256": _require_sha256(
            payload["selection_salt_sha256"], name="selection salt"
        ),
    }


def _verify_cache_hashes(
    cache_set_path: Path, expected_sha256: str, payload: Mapping[str, Any]
) -> dict[str, str]:
    expected = _require_sha256(expected_sha256, name="cache-set")
    if not cache_set_path.is_file() or cache_set_path.is_symlink() or _sha256_file(cache_set_path) != expected:
        raise Phase1SingleObservationArchiveError("source-validation cache-set SHA256 drift")
    mapping = payload.get("cache_npz_by_scenario")
    hashes = payload.get("cache_sha256_by_scenario")
    if (
        not isinstance(mapping, Mapping)
        or tuple(mapping) != FORMAL_LEO_WEAK_SCENARIOS
        or not isinstance(hashes, Mapping)
        or tuple(hashes) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise Phase1SingleObservationArchiveError("cache-set three-scenario mapping drift")
    result: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        path = _resolve_member(cache_set_path, mapping[scenario])
        digest = _require_sha256(hashes[scenario], name=f"cache NPZ {scenario}")
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
            raise Phase1SingleObservationArchiveError(f"cache NPZ SHA256 drift: {scenario}")
        result[scenario] = digest
    return result


def _validate_and_select(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]], salt_sha256: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if tuple(arrays_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise Phase1SingleObservationArchiveError("all three ordered cache scenarios are required")
    required = {
        "leo_weak_iq",
        "sample_ids",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "dataset_role",
        "sat_scenarios",
    }
    ids: dict[str, list[str]] = {}
    indexes: dict[str, dict[str, int]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        missing = required - set(arrays)
        if missing:
            raise Phase1SingleObservationArchiveError(
                f"cache scenario {scenario} missing fields: {sorted(missing)}"
            )
        current = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
        if (
            not current
            or len(set(current)) != len(current)
            or iq.ndim != 3
            or iq.shape[1] != 2
            or len(iq) != len(current)
            or not np.isfinite(iq).all()
        ):
            raise Phase1SingleObservationArchiveError(f"cache scenario {scenario} row drift")
        for field in required - {"leo_weak_iq"}:
            if len(np.asarray(arrays[field])) != len(current):
                raise Phase1SingleObservationArchiveError(
                    f"cache scenario {scenario} row count drifts for {field}"
                )
        if np.asarray(arrays["sat_scenarios"]).astype(str).tolist() != [scenario] * len(current):
            raise Phase1SingleObservationArchiveError(f"cache scenario identity drift: {scenario}")
        ids[scenario] = current
        indexes[scenario] = {value: index for index, value in enumerate(current)}
    reference = ids[FORMAL_LEO_WEAK_SCENARIOS[0]]
    if any(set(ids[scenario]) != set(reference) for scenario in FORMAL_LEO_WEAK_SCENARIOS[1:]):
        raise Phase1SingleObservationArchiveError(
            "each physical sample must be present in all three scenarios"
        )
    metadata: dict[str, list[str]] = {
        "physical_ids": [],
        "labels": [],
        "receiver_ids": [],
        "day_ids": [],
        "scenario_names": [],
    }
    selected_iq: list[np.ndarray] = []
    for physical_id in reference:
        identities = []
        roles = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[scenario]
            index = indexes[scenario][physical_id]
            identities.append(
                (
                    str(arrays["tx_ids"][index]),
                    str(arrays["rx_ids"][index]),
                    str(arrays["day_ids"][index]),
                )
            )
            roles.append(str(arrays["dataset_role"][index]))
        if len(set(identities)) != 1:
            raise Phase1SingleObservationArchiveError(
                f"TX/RX/day identity drift across scenarios: {physical_id}"
            )
        if set(roles) != {"source"}:
            raise Phase1SingleObservationArchiveError(
                f"source-validation dataset-role drift: {physical_id}"
            )
        selected_scenario = FORMAL_LEO_WEAK_SCENARIOS[
            selection_index(salt_sha256, physical_id)
        ]
        index = indexes[selected_scenario][physical_id]
        metadata["physical_ids"].append(physical_id)
        metadata["labels"].append(identities[0][0])
        metadata["receiver_ids"].append(identities[0][1])
        metadata["day_ids"].append(identities[0][2])
        metadata["scenario_names"].append(selected_scenario)
        selected_iq.append(
            np.asarray(arrays_by_scenario[selected_scenario]["leo_weak_iq"][index], dtype=np.float32)
        )
    return (
        {key: np.asarray(value, dtype=np.str_) for key, value in metadata.items()},
        np.ascontiguousarray(np.stack(selected_iq), dtype=np.float32),
    )


def _resolve_device(requested: str) -> tuple[Any, str]:
    import torch

    try:
        device = torch.device(str(requested))
    except (TypeError, RuntimeError) as exc:
        raise Phase1SingleObservationArchiveError("invalid runtime device") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Phase1SingleObservationArchiveError("CUDA requested but unavailable")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise Phase1SingleObservationArchiveError("requested CUDA device is unavailable")
        device = torch.device(f"cuda:{index}")
    elif device.type != "cpu":
        raise Phase1SingleObservationArchiveError("only CPU or CUDA runtime devices are allowed")
    return device, str(device)


def _forward_torchscript(
    runtime_path: Path,
    runtime_sha256: str,
    rows: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if int(batch_size) < 1:
        raise Phase1SingleObservationArchiveError("batch_size must be positive")
    with runtime_path.open("rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != runtime_sha256:
            raise Phase1SingleObservationArchiveError("TorchScript runtime SHA256 drift")
        handle.seek(0)
        model = torch.jit.load(handle, map_location=device).eval()
    feature_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.ascontiguousarray(rows[start : start + int(batch_size)], dtype=np.float32)
            tensor = torch.frombuffer(chunk, dtype=torch.float32).reshape(chunk.shape).clone().to(device)
            output = model(tensor)
            if isinstance(output, dict):
                features, logits = output.get("features"), output.get("logits")
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                features, logits = output
            else:
                raise Phase1SingleObservationArchiveError("TorchScript output schema drift")
            if not torch.is_tensor(features) or not torch.is_tensor(logits):
                raise Phase1SingleObservationArchiveError("TorchScript outputs must be tensors")
            feature_parts.append(np.asarray(features.detach().float().cpu().tolist(), dtype=np.float32))
            logit_parts.append(np.asarray(logits.detach().float().cpu().tolist(), dtype=np.float32))
    if _sha256_file(runtime_path) != runtime_sha256:
        raise Phase1SingleObservationArchiveError("TorchScript runtime changed during export")
    return np.concatenate(feature_parts), np.concatenate(logit_parts)


def _forward_loaded_runtime(
    model: Any,
    rows: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward a runtime already materialized by the formal bundle verifier."""

    import torch

    if int(batch_size) < 1:
        raise Phase1SingleObservationArchiveError("batch_size must be positive")
    model = model.to(device).eval()
    feature_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.ascontiguousarray(
                rows[start : start + int(batch_size)], dtype=np.float32
            )
            tensor = (
                torch.frombuffer(chunk, dtype=torch.float32)
                .reshape(chunk.shape)
                .clone()
                .to(device)
            )
            output = model(tensor)
            if isinstance(output, dict):
                features, logits = output.get("features"), output.get("logits")
            elif isinstance(output, (tuple, list)) and len(output) == 2:
                features, logits = output
            else:
                raise Phase1SingleObservationArchiveError(
                    "formal TorchScript output schema drift"
                )
            if not torch.is_tensor(features) or not torch.is_tensor(logits):
                raise Phase1SingleObservationArchiveError(
                    "formal TorchScript outputs must be tensors"
                )
            feature_parts.append(
                np.asarray(features.detach().float().cpu().tolist(), dtype=np.float32)
            )
            logit_parts.append(
                np.asarray(logits.detach().float().cpu().tolist(), dtype=np.float32)
            )
    return np.concatenate(feature_parts), np.concatenate(logit_parts)


def _coerce_runtime_outputs(
    value: Any, *, row_count: int, logit_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        features, logits = value.get("features"), value.get("logits")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        features, logits = value
    else:
        raise Phase1SingleObservationArchiveError("runtime must return features and logits")
    z160 = np.asarray(features, dtype=np.float32)
    reference_logits = np.asarray(logits, dtype=np.float32)
    if z160.shape != (row_count, FEATURE_DIM) or not np.isfinite(z160).all():
        raise Phase1SingleObservationArchiveError("runtime z160 output drift")
    if reference_logits.shape != (row_count, logit_dim) or not np.isfinite(reference_logits).all():
        raise Phase1SingleObservationArchiveError("checkpoint-reference logit output drift")
    return z160, reference_logits


def _export_impl(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    runtime: Mapping[str, Any],
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    output_dir: str | Path,
    device: str,
    batch_size: int,
    mode: str,
    forward_callback: ForwardCallback | None,
    cache_loader: CacheLoader,
) -> dict[str, Any]:
    if mode not in {"formal", "development", "diagnostic"}:
        raise Phase1SingleObservationArchiveError("unknown Phase1 export mode")
    cache_path = Path(cache_set_path).resolve()
    salt_receipt_path = Path(selection_salt_receipt_path).resolve()
    salt = _load_selection_salt(
        salt_receipt_path,
        selection_salt_receipt_sha256,
        runtime_binding=runtime,
    )
    expected_cache_sha = _require_sha256(cache_set_sha256, name="cache-set")
    if not cache_path.is_file() or _sha256_file(cache_path) != expected_cache_sha:
        raise Phase1SingleObservationArchiveError("source-validation cache-set SHA256 drift")
    arrays_by_scenario, cache_payload, cache_audit = cache_loader(
        cache_path,
        expected_scope=SOURCE_VALIDATION_SCOPE,
        allowed_roles=SOURCE_VALIDATION_ROLES,
    )
    if cache_payload.get("cache_scope") != SOURCE_VALIDATION_SCOPE:
        raise Phase1SingleObservationArchiveError("cache-set is not source_validation")
    cache_hashes = _verify_cache_hashes(cache_path, expected_cache_sha, cache_payload)
    metadata, selected_iq = _validate_and_select(
        arrays_by_scenario, salt["selection_salt_sha256"]
    )
    labels = metadata["labels"].astype(str)
    if set(labels.tolist()) != set(runtime["class_ids"]):
        raise Phase1SingleObservationArchiveError("cache labels/runtime class registry drift")
    runtime_device, resolved_device = _resolve_device(device)
    if mode == "formal":
        if forward_callback is not None or cache_loader is not load_verified_leo_weak_cache_set:
            raise Phase1SingleObservationArchiveError("formal export forbids injected loaders/forward")
        output = _forward_loaded_runtime(
            runtime["model"],
            selected_iq,
            device=runtime_device,
            batch_size=batch_size,
        )
    elif mode == "development":
        if forward_callback is not None or cache_loader is not load_verified_leo_weak_cache_set:
            raise Phase1SingleObservationArchiveError(
                "development export forbids injected loaders/forward"
            )
        output = _forward_torchscript(
            runtime["runtime_path"],
            runtime["runtime_sha256"],
            selected_iq,
            device=runtime_device,
            batch_size=batch_size,
        )
    else:
        if forward_callback is None:
            raise Phase1SingleObservationArchiveError(
                "TEST_DIAGNOSTIC_NOT_FORMAL requires injected forward"
            )
        output = forward_callback(np.array(selected_iq, copy=True))
    z160, reference_logits = _coerce_runtime_outputs(
        output, row_count=len(selected_iq), logit_dim=len(runtime["class_ids"])
    )
    fft96 = spectral_logmag_sketch(selected_iq)
    rf32 = rf_statistics(selected_iq)
    features = np.ascontiguousarray(
        np.concatenate([z160, fft96, rf32], axis=1), dtype=np.float32
    )
    arrays = {
        "features": features,
        "labels": metadata["labels"],
        "receiver_ids": metadata["receiver_ids"],
        "day_ids": metadata["day_ids"],
        "physical_ids": metadata["physical_ids"],
        "scenario_names": metadata["scenario_names"],
        "class_ids": np.asarray(runtime["class_ids"], dtype=np.str_),
        "checkpoint_reference_logits": reference_logits,
    }
    if tuple(arrays) != OUTPUT_MEMBER_ALLOWLIST:
        raise Phase1SingleObservationArchiveError("D97 output member allowlist drift")
    if features.shape != (len(selected_iq), JOINT_DIM):
        raise Phase1SingleObservationArchiveError("D97 feature dimension drift")
    physical = arrays["physical_ids"].astype(str).tolist()
    if len(physical) != len(set(physical)):
        raise Phase1SingleObservationArchiveError("output must contain one row per physical ID")
    root = Path(output_dir).resolve()
    archive_path = root / NPZ_NAME
    manifest_path = root / MANIFEST_NAME
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite Phase1 single-observation archive")
    root.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    status = {
        "formal": FORMAL_STATUS,
        "development": DEVELOPMENT_STATUS,
        "diagnostic": DIAGNOSTIC_STATUS,
    }[mode]
    dependency_hashes = {
        "leo_weak_cache": _sha256_file(
            Path(load_verified_leo_weak_cache_set.__code__.co_filename).resolve()
        ),
        "feature_descriptors": _sha256_file(
            Path(spectral_logmag_sketch.__code__.co_filename).resolve()
        ),
        "formal_bundle_verifier": _sha256_file(
            Path(deployment_bundle.__file__).resolve()
        ),
    }
    manifest = {
        "schema": SCHEMA if mode != "diagnostic" else DIAGNOSTIC_SCHEMA,
        "status": status,
        "artifact_stage": "phase1_offline_before_target_access",
        "artifact": {"path": NPZ_NAME, "sha256": _sha256_file(archive_path)},
        "exact_member_allowlist": list(OUTPUT_MEMBER_ALLOWLIST),
        "feature_dims": {"z160": 160, "fft96": 96, "rf32": 32, "features": 288},
        "inputs": {
            "cache_set_sha256": expected_cache_sha,
            "cache_npz_sha256_by_scenario": cache_hashes,
            "runtime_authority_mode": runtime["authority_mode"],
            "runtime_authority_binding_sha256": runtime[
                "authority_binding_sha256"
            ],
            "runtime_checkpoint_parity_receipt_sha256": runtime[
                "runtime_receipt_sha256"
            ],
            "runtime_schema": RUNTIME_SCHEMA,
            "runtime_sha256": runtime["runtime_sha256"],
            "phase1_checkpoint_sha256": runtime["checkpoint_sha256"],
            "bundle_id": runtime["bundle_id"],
            "formal_outer_content_root_sha256": runtime[
                "formal_outer_content_root_sha256"
            ],
            "detached_seal_sha256": runtime["detached_seal_sha256"],
            "signature_envelope_sha256": runtime[
                "signature_envelope_sha256"
            ],
            "selection_salt_receipt_sha256": salt["sha256"],
            "selection_salt_receipt_schema": salt["schema"],
            "exporter_code_sha256": _sha256_file(Path(__file__).resolve()),
            "dependency_code_sha256": dependency_hashes,
            "dependency_closure_sha256": hashlib.sha256(
                _canonical_json_bytes(dependency_hashes)
            ).hexdigest(),
        },
        "selection": {
            "selection_salt_sha256": salt["selection_salt_sha256"],
            "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
            "formula": (
                "j=uint64_be(SHA256(b'P1_SINGLE_LEO_V1'||"
                "bytes.fromhex(selection_salt_sha256)||physical_id.utf8)[:8])mod3"
            ),
            "selected_observations_per_physical_id": 1,
            "unselected_observations_forwarded": 0,
        },
        "feature_semantics": {
            "features": (
                "float32_concat(runtime_z160,internally_normalized_fft96,"
                "internally_normalized_rf32)_without_cross_block_weight_or_joint_normalization"
            ),
            "deployment_normalization": "shared_D97_normalize_three_blocks",
            "checkpoint_reference_logits": "sealed_ADV3B02_checkpoint_reference_only_not_D81",
        },
        "requested_device": str(device),
        "resolved_device": resolved_device,
        "row_count": len(physical),
        "physical_id_unique_count": len(set(physical)),
        "one_output_row_per_physical_id": True,
        "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
        "cache_loader_audit_sha256": hashlib.sha256(
            _canonical_json_bytes(cache_audit)
        ).hexdigest(),
        "access_audit": {
            "clean_calls": 0,
            "target_calls": 0,
            "channel_calls": 0,
            "clean_iq_access": False,
            "target_access": False,
            "query_access": False,
            "raw_iq_persisted": False,
            "received_iq_persisted": False,
            "unselected_iq_persisted": False,
        },
        "lifecycle": {
            "phase1_temporary_selection_asset": True,
            "phase2_bundle_ingest_allowed": False,
            "phase2_runtime_access_allowed": False,
            "retention": "archive_or_delete_after_D97_lock_receipt",
        },
        "formal_archive": mode == "formal",
        "development_archive": mode == "development",
    }
    _write_new(manifest_path, _canonical_json_bytes(manifest) + b"\n")
    verify_phase1_singleobs_archive(archive_path, manifest)
    return {
        "archive_path": str(archive_path),
        "archive_sha256": manifest["artifact"]["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "row_count": len(physical),
        "formal_archive": mode == "formal",
        "development_archive": mode == "development",
        "status": status,
    }


def verify_phase1_singleobs_archive(
    archive_path: str | Path, manifest: Mapping[str, Any]
) -> None:
    path = Path(archive_path)
    with np.load(path, allow_pickle=False) as payload:
        members = tuple(payload.files)
        if members != OUTPUT_MEMBER_ALLOWLIST:
            raise Phase1SingleObservationArchiveError("archive has extra/missing D97 members")
        arrays = {name: np.asarray(payload[name]) for name in members}
    if manifest.get("exact_member_allowlist") != list(OUTPUT_MEMBER_ALLOWLIST):
        raise Phase1SingleObservationArchiveError("manifest member allowlist drift")
    if set(manifest.get("array_sha256", {})) != set(OUTPUT_MEMBER_ALLOWLIST):
        raise Phase1SingleObservationArchiveError("manifest array registry drift")
    for name, value in arrays.items():
        if manifest["array_sha256"].get(name) != _array_sha256(value):
            raise Phase1SingleObservationArchiveError(f"archive array SHA drift: {name}")
    if manifest.get("artifact", {}).get("sha256") != _sha256_file(path):
        raise Phase1SingleObservationArchiveError("archive file SHA drift")


def export_phase1_singleobs_feature_archive(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    package_root: str | Path,
    detached_seal_path: str | Path,
    expected_detached_seal_sha256: str,
    signature_envelope_path: str | Path,
    expected_signature_envelope_sha256: str,
    expected_checkpoint_lineage_sha256: str,
    expected_runtime_sha256: str,
    expected_component_pre_sign_content_root_sha256: str,
    expected_class_handle_binding_sha256: str,
    expected_parity_receipt_sha256: str,
    expected_generation_lock_sha256: str,
    expected_method_lock_sha256: str,
    expected_generation_config_sha256: str,
    expected_generation_code_sha256: str,
    expected_outer_content_root_sha256: str,
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Formal API backed only by the pinned ADV3B02 outer-bundle authority."""

    runtime = _load_formal_runtime_binding(
        package_root,
        detached_seal_path=detached_seal_path,
        expected_detached_seal_sha256=expected_detached_seal_sha256,
        signature_envelope_path=signature_envelope_path,
        expected_signature_envelope_sha256=expected_signature_envelope_sha256,
        expected_checkpoint_lineage_sha256=expected_checkpoint_lineage_sha256,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_component_pre_sign_content_root_sha256=expected_component_pre_sign_content_root_sha256,
        expected_class_handle_binding_sha256=expected_class_handle_binding_sha256,
        expected_parity_receipt_sha256=expected_parity_receipt_sha256,
        expected_generation_lock_sha256=expected_generation_lock_sha256,
        expected_method_lock_sha256=expected_method_lock_sha256,
        expected_generation_config_sha256=expected_generation_config_sha256,
        expected_generation_code_sha256=expected_generation_code_sha256,
        expected_outer_content_root_sha256=expected_outer_content_root_sha256,
    )
    return _export_impl(
        cache_set_path=cache_set_path,
        cache_set_sha256=cache_set_sha256,
        runtime=runtime,
        selection_salt_receipt_path=selection_salt_receipt_path,
        selection_salt_receipt_sha256=selection_salt_receipt_sha256,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        mode="formal",
        forward_callback=None,
        cache_loader=load_verified_leo_weak_cache_set,
    )


def export_development_phase1_singleobs_feature_archive(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    runtime_manifest_path: str | Path,
    runtime_manifest_sha256: str,
    expected_runtime_sha256: str,
    expected_parity_receipt_sha256: str,
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Development-only real-runtime export; never claims formal authority."""

    runtime = _load_runtime_binding(
        Path(runtime_manifest_path).resolve(),
        runtime_manifest_sha256,
        require_known_development_runtime=True,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_parity_receipt_sha256=expected_parity_receipt_sha256,
    )
    return _export_impl(
        cache_set_path=cache_set_path,
        cache_set_sha256=cache_set_sha256,
        runtime=runtime,
        selection_salt_receipt_path=selection_salt_receipt_path,
        selection_salt_receipt_sha256=selection_salt_receipt_sha256,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        mode="development",
        forward_callback=None,
        cache_loader=load_verified_leo_weak_cache_set,
    )


def export_development_sha_only_phase1_singleobs_feature_archive(
    *,
    cache_set_path: str | Path,
    cache_set_sha256: str,
    runtime_path: str | Path,
    expected_runtime_sha256: str,
    class_ids: tuple[str, ...] | list[str],
    selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Export Phase1 features from a known runtime without a parity claim.

    This mode is deliberately development-only.  It records the exact runtime
    SHA and class registry, emits no parity receipt, and cannot become a formal
    Phase1 or target result.
    """

    runtime = _load_known_runtime_sha_only(
        runtime_path, expected_runtime_sha256, class_ids
    )
    return _export_impl(
        cache_set_path=cache_set_path,
        cache_set_sha256=cache_set_sha256,
        runtime=runtime,
        selection_salt_receipt_path=selection_salt_receipt_path,
        selection_salt_receipt_sha256=selection_salt_receipt_sha256,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        mode="development",
        forward_callback=None,
        cache_loader=load_verified_leo_weak_cache_set,
    )


def _export_test_diagnostic_not_formal(
    *,
    forward_callback: ForwardCallback,
    cache_loader: CacheLoader,
    **kwargs: Any,
) -> dict[str, Any]:
    """Private injection seam whose receipt is explicitly non-formal."""

    runtime_manifest_path = Path(kwargs.pop("runtime_manifest_path")).resolve()
    runtime_manifest_sha256 = kwargs.pop("runtime_manifest_sha256")
    runtime = _load_runtime_binding(runtime_manifest_path, runtime_manifest_sha256)
    return _export_impl(
        **kwargs,
        runtime=runtime,
        mode="diagnostic",
        forward_callback=forward_callback,
        cache_loader=cache_loader,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "development"), required=True)
    parser.add_argument("--cache-set", type=Path, required=True)
    parser.add_argument("--cache-set-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument(
        "--class-ids",
        help="Comma-separated frozen class handles for development SHA-only export",
    )
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--detached-seal", type=Path)
    parser.add_argument("--signature-envelope", type=Path)
    for name in (
        "expected-detached-seal-sha256",
        "expected-signature-envelope-sha256",
        "expected-checkpoint-lineage-sha256",
        "expected-runtime-sha256",
        "expected-component-pre-sign-content-root-sha256",
        "expected-class-handle-binding-sha256",
        "expected-parity-receipt-sha256",
        "expected-generation-lock-sha256",
        "expected-method-lock-sha256",
        "expected-generation-config-sha256",
        "expected-generation-code-sha256",
        "expected-outer-content-root-sha256",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--selection-salt-receipt", type=Path, required=True)
    parser.add_argument("--selection-salt-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    common = {
        "cache_set_path": args.cache_set,
        "cache_set_sha256": args.cache_set_sha256,
        "selection_salt_receipt_path": args.selection_salt_receipt,
        "selection_salt_receipt_sha256": args.selection_salt_receipt_sha256,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if args.mode == "development":
        if args.runtime is not None:
            if args.runtime_manifest is not None or args.runtime_manifest_sha256 is not None:
                raise Phase1SingleObservationArchiveError(
                    "development runtime and runtime manifest are mutually exclusive"
                )
            class_ids = tuple(
                value.strip() for value in str(args.class_ids or "").split(",") if value.strip()
            )
            result = export_development_sha_only_phase1_singleobs_feature_archive(
                **common,
                runtime_path=args.runtime,
                expected_runtime_sha256=args.expected_runtime_sha256,
                class_ids=class_ids,
            )
        else:
            result = export_development_phase1_singleobs_feature_archive(
                **common,
                runtime_manifest_path=args.runtime_manifest,
                runtime_manifest_sha256=args.runtime_manifest_sha256,
                expected_runtime_sha256=args.expected_runtime_sha256,
                expected_parity_receipt_sha256=args.expected_parity_receipt_sha256,
            )
    else:
        result = export_phase1_singleobs_feature_archive(
            **common,
            package_root=args.package_root,
            detached_seal_path=args.detached_seal,
            expected_detached_seal_sha256=args.expected_detached_seal_sha256,
            signature_envelope_path=args.signature_envelope,
            expected_signature_envelope_sha256=args.expected_signature_envelope_sha256,
            expected_checkpoint_lineage_sha256=args.expected_checkpoint_lineage_sha256,
            expected_runtime_sha256=args.expected_runtime_sha256,
            expected_component_pre_sign_content_root_sha256=(
                args.expected_component_pre_sign_content_root_sha256
            ),
            expected_class_handle_binding_sha256=(
                args.expected_class_handle_binding_sha256
            ),
            expected_parity_receipt_sha256=args.expected_parity_receipt_sha256,
            expected_generation_lock_sha256=args.expected_generation_lock_sha256,
            expected_method_lock_sha256=args.expected_method_lock_sha256,
            expected_generation_config_sha256=args.expected_generation_config_sha256,
            expected_generation_code_sha256=args.expected_generation_code_sha256,
            expected_outer_content_root_sha256=args.expected_outer_content_root_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
