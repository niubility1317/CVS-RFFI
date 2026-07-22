#!/usr/bin/env python3
"""Export a nonformal one-observation Phase1 archive from a sealed dual runtime.

This is deliberately a development-only bridge.  It consumes already verified
source-validation LEO cache rows and never creates a deployment bundle or a
Phase2-eligible artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
import uuid

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    while _value in sys.path:
        sys.path.remove(_value)
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, _value)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA_V1,
    LEO_WEAK_CACHE_SET_SCHEMA_V1,
    load_verified_leo_weak_cache_set,
)
from scripts.export_adv3b02_dual_feature_torchscript import (  # noqa: E402
    EXPORT_SCHEMA,
    FEATURE_DIM,
    RUNTIME_BATCH_CAPACITY,
    RUNTIME_OUTPUT_SCHEMA,
)
from scripts.export_phase1_singleobs_feature_archive import (  # noqa: E402
    BASE_CHECKPOINT_SHA256,
    KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256,
    SELECTION_SALT_RECEIPT_SCHEMA,
    _load_verified_v1_only_source_validation_cache_set,
    selection_index,
)
from scripts.verify_adv3b02_dual_runtime_checkpoint_parity import (  # noqa: E402
    RECEIPT_SCHEMA as PARITY_RECEIPT_SCHEMA,
    RUNTIME_ROLES,
)


SCHEMA = "cvs.phase1.singleobs_dual_feature_archive.v1"
NPZ_NAME = "phase1_singleobs_dual_feature_archive.npz"
MANIFEST_NAME = "phase1_singleobs_dual_feature_archive.manifest.json"
MEMBERS = (
    "z_id",
    "z_dom",
    "tx_logits",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "observation_ids",
)
CACHE_LOADER = _load_verified_v1_only_source_validation_cache_set


class Phase1SingleobsDualArchiveError(ValueError):
    """Raised when a frozen development-only archive input drifts."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise Phase1SingleobsDualArchiveError(f"{name} must be lowercase SHA256 hex")
    return normalized


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _read_bound_json(path: str | Path, expected_sha: str, *, name: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    expected = _sha256(expected_sha, name=f"{name} SHA256")
    if not resolved.is_file() or resolved.is_symlink() or _sha256_file(resolved) != expected:
        raise Phase1SingleobsDualArchiveError(f"{name} path/SHA256 drift")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase1SingleobsDualArchiveError(f"{name} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Phase1SingleobsDualArchiveError(f"{name} must be a JSON object")
    return resolved, value


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype == object:
        raise Phase1SingleobsDualArchiveError("object arrays cannot be persisted")
    if array.dtype.kind in {"U", "S"}:
        header = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = _canonical_json(array.astype(str).tolist())
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        header = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        body = canonical.tobytes(order="C")
    return hashlib.sha256(_canonical_json(header) + b"\0" + body).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _class_registry(class_ids: tuple[str, ...] | list[str], *, logit_width: int) -> tuple[str, ...]:
    registry = tuple(str(value) for value in class_ids)
    if (
        not registry
        or any(not value for value in registry)
        or len(registry) != len(set(registry))
        or len(registry) != int(logit_width)
    ):
        raise Phase1SingleobsDualArchiveError(
            "class_ids must be nonempty, unique, ordered, and match logit width"
        )
    return registry


def _class_registry_sha256(class_ids: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(_canonical_json(list(class_ids))).hexdigest()


def _load_selection_salt(path: str | Path, expected_sha: str, *, checkpoint_sha: str) -> dict[str, str]:
    resolved, receipt = _read_bound_json(path, expected_sha, name="selection-salt receipt")
    required = {
        "schema",
        "status",
        "artifact_stage",
        "bundle_id",
        "phase1_checkpoint_sha256",
        "selection_salt_sha256",
        "target_access",
    }
    if (
        set(receipt) != required
        or receipt.get("schema") != SELECTION_SALT_RECEIPT_SCHEMA
        or receipt.get("status") != "SEALED_BEFORE_TARGET_ACCESS"
        or receipt.get("artifact_stage") != "phase1_offline_before_target_access"
        or not isinstance(receipt.get("bundle_id"), str)
        or _sha256(receipt["bundle_id"], name="selection-salt bundle_id") != receipt["bundle_id"]
        or receipt.get("phase1_checkpoint_sha256") != checkpoint_sha
        or receipt.get("target_access") is not False
    ):
        raise Phase1SingleobsDualArchiveError("selection-salt receipt lineage drift")
    return {
        "path": str(resolved),
        "sha256": _sha256(expected_sha, name="selection-salt receipt"),
        "selection_salt_sha256": _sha256(receipt["selection_salt_sha256"], name="selection salt"),
    }


def _load_runtime_closure(
    *,
    runtime_path: str | Path,
    runtime_sha256: str,
    runtime_role: str,
    export_receipt_path: str | Path,
    export_receipt_sha256: str,
    parity_receipt_path: str | Path,
    parity_receipt_sha256: str,
) -> dict[str, Any]:
    if runtime_role not in RUNTIME_ROLES:
        raise Phase1SingleobsDualArchiveError("runtime role must be base or candidate")
    runtime = Path(runtime_path).resolve()
    runtime_sha = _sha256(runtime_sha256, name="runtime")
    if not runtime.is_file() or runtime.is_symlink() or _sha256_file(runtime) != runtime_sha:
        raise Phase1SingleobsDualArchiveError("runtime path/SHA256 drift")
    _export_path, export = _read_bound_json(
        export_receipt_path, export_receipt_sha256, name="dual runtime export receipt"
    )
    _parity_path, parity = _read_bound_json(
        parity_receipt_path, parity_receipt_sha256, name="dual runtime parity receipt"
    )
    role_runtime_key = f"{runtime_role}_runtime_sha256"
    dimensions = export.get("feature_dimensions")
    if (
        export.get("schema") != EXPORT_SCHEMA
        or export.get("status") != "PASS"
        or export.get("runtime_output_schema") != RUNTIME_OUTPUT_SCHEMA
        or export.get("checkpoint_sha256") != BASE_CHECKPOINT_SHA256
        or not isinstance(export.get("adapter_state_sha256"), str)
        or _sha256(export["adapter_state_sha256"], name="export adapter") != export["adapter_state_sha256"]
        or export.get(role_runtime_key) != runtime_sha
        or export.get("runtime_batch_capacity") != RUNTIME_BATCH_CAPACITY
        or export.get("formal_phase2_eligible") is not False
        or export.get("bundle_created") is not False
        or export.get("bundle_id") is not None
        or not isinstance(export.get("expected_input_len"), int)
        or isinstance(export.get("expected_input_len"), bool)
        or int(export["expected_input_len"]) <= 0
        or not isinstance(dimensions, dict)
        or dimensions.get("z_id") != FEATURE_DIM
        or dimensions.get("z_dom") != FEATURE_DIM
        or not isinstance(dimensions.get("tx_logits"), int)
        or isinstance(dimensions.get("tx_logits"), bool)
        or int(dimensions["tx_logits"]) < 2
    ):
        raise Phase1SingleobsDualArchiveError("role/runtime/export receipt closure drift")
    if (
        parity.get("schema") != PARITY_RECEIPT_SCHEMA
        or parity.get("status") != "PASS"
        or parity.get("runtime_output_schema") != RUNTIME_OUTPUT_SCHEMA
        or parity.get("checkpoint_lineage_sha256") != BASE_CHECKPOINT_SHA256
        or parity.get("adapter_state_sha256") != export["adapter_state_sha256"]
        or parity.get("runtime_sha256") != runtime_sha
        or parity.get("export_receipt_sha256") != _sha256(export_receipt_sha256, name="export receipt")
        or parity.get("runtime_role") != runtime_role
        or parity.get("expected_input_len") != export["expected_input_len"]
        or parity.get("expected_tx_classes") != dimensions["tx_logits"]
        or parity.get("runtime_batch_capacity") != RUNTIME_BATCH_CAPACITY
        or parity.get("runtime_invocations_per_parity_batch") != 1
        or parity.get("formal_phase2_eligible") is not False
        or parity.get("bundle_created") is not False
        or parity.get("bundle_id") is not None
    ):
        raise Phase1SingleobsDualArchiveError("role/runtime/export/parity closure drift")
    try:
        maximum = float(parity.get("max_abs_output_delta"))
    except (TypeError, ValueError) as exc:
        raise Phase1SingleobsDualArchiveError("parity maximum is invalid") from exc
    if not np.isfinite(maximum) or maximum < 0.0 or maximum > 1.0e-5:
        raise Phase1SingleobsDualArchiveError("parity maximum drift")
    return {
        "path": runtime,
        "sha256": runtime_sha,
        "role": runtime_role,
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "adapter_state_sha256": export["adapter_state_sha256"],
        "export_receipt_sha256": _sha256(export_receipt_sha256, name="export receipt"),
        "parity_receipt_sha256": _sha256(parity_receipt_sha256, name="parity receipt"),
        "input_len": int(export["expected_input_len"]),
        "tx_classes": int(dimensions["tx_logits"]),
        "max_abs_output_delta": maximum,
    }


def _select_verified_observations(
    arrays_by_scenario: Mapping[str, Mapping[str, np.ndarray]], salt_sha256: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if tuple(arrays_by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise Phase1SingleobsDualArchiveError("all three ordered scenarios are required")
    required = {
        "leo_weak_iq", "sample_ids", "tx_ids", "rx_ids", "day_ids",
        "dataset_role", "sat_scenarios", "overlay_ids",
    }
    indexes: dict[str, dict[str, int]] = {}
    ids: dict[str, list[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        missing = required - set(arrays)
        if missing:
            raise Phase1SingleobsDualArchiveError(f"verified cache lacks fields: {sorted(missing)}")
        sample_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)) or iq.ndim != 3 or iq.shape[1] != 2 or len(iq) != len(sample_ids) or not np.isfinite(iq).all():
            raise Phase1SingleobsDualArchiveError(f"verified cache row contract drift: {scenario}")
        if any(len(np.asarray(arrays[name])) != len(sample_ids) for name in required - {"leo_weak_iq"}):
            raise Phase1SingleobsDualArchiveError(f"verified cache row count drift: {scenario}")
        if np.asarray(arrays["sat_scenarios"]).astype(str).tolist() != [scenario] * len(sample_ids):
            raise Phase1SingleobsDualArchiveError(f"verified cache scenario drift: {scenario}")
        overlays = np.asarray(arrays["overlay_ids"]).astype(str).tolist()
        if any(not value for value in overlays) or len(overlays) != len(set(overlays)):
            raise Phase1SingleobsDualArchiveError(f"verified overlay_ids drift: {scenario}")
        ids[scenario] = sample_ids
        indexes[scenario] = {value: index for index, value in enumerate(sample_ids)}
    reference = ids[FORMAL_LEO_WEAK_SCENARIOS[0]]
    if any(set(ids[scenario]) != set(reference) for scenario in FORMAL_LEO_WEAK_SCENARIOS[1:]):
        raise Phase1SingleobsDualArchiveError("cache scenarios do not share one selectable physical-ID set")
    metadata = {name: [] for name in ("labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names", "observation_ids")}
    selected_iq: list[np.ndarray] = []
    for physical_id in reference:
        identities = []
        roles = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[scenario]
            index = indexes[scenario][physical_id]
            identities.append((str(arrays["tx_ids"][index]), str(arrays["rx_ids"][index]), str(arrays["day_ids"][index])))
            roles.append(str(arrays["dataset_role"][index]))
        if len(set(identities)) != 1 or set(roles) != {"source"}:
            raise Phase1SingleobsDualArchiveError(f"physical identity/role drift: {physical_id}")
        scenario = FORMAL_LEO_WEAK_SCENARIOS[selection_index(salt_sha256, physical_id)]
        index = indexes[scenario][physical_id]
        arrays = arrays_by_scenario[scenario]
        metadata["labels"].append(identities[0][0])
        metadata["receiver_ids"].append(identities[0][1])
        metadata["day_ids"].append(identities[0][2])
        metadata["physical_ids"].append(physical_id)
        metadata["scenario_names"].append(scenario)
        metadata["observation_ids"].append(str(arrays["overlay_ids"][index]))
        selected_iq.append(np.asarray(arrays["leo_weak_iq"][index], dtype=np.float32))
    if len(metadata["observation_ids"]) != len(set(metadata["observation_ids"])):
        raise Phase1SingleobsDualArchiveError("selected observation IDs are not unique")
    return ({key: np.asarray(value, dtype=np.str_) for key, value in metadata.items()}, np.ascontiguousarray(np.stack(selected_iq), dtype=np.float32))


def _resolve_device(value: str) -> Any:
    import torch

    try:
        device = torch.device(str(value))
    except (TypeError, RuntimeError) as exc:
        raise Phase1SingleobsDualArchiveError("runtime device is invalid") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is not None and (device.index < 0 or device.index >= torch.cuda.device_count()):
            raise Phase1SingleobsDualArchiveError("requested CUDA device is unavailable")
        return torch.device(f"cuda:{torch.cuda.current_device() if device.index is None else device.index}")
    if device.type != "cpu":
        raise Phase1SingleobsDualArchiveError("device must be CPU or CUDA")
    return device


def _forward_once_per_selected_iq_batch(runtime: Mapping[str, Any], rows: np.ndarray, *, device: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import torch

    if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= RUNTIME_BATCH_CAPACITY:
        raise Phase1SingleobsDualArchiveError("batch_size must be in [1,256]")
    with runtime["path"].open("rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != runtime["sha256"]:
            raise Phase1SingleobsDualArchiveError("runtime changed before load")
        handle.seek(0)
        model = torch.jit.load(handle, map_location=device).eval()
    outputs: list[list[np.ndarray]] = [[], [], []]
    invocations = 0
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.ascontiguousarray(rows[start:start + int(batch_size)], dtype=np.float32)
            values = model(torch.from_numpy(chunk).to(device))
            invocations += 1
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise Phase1SingleobsDualArchiveError("dual runtime must return three outputs in one call")
            for output_index, (name, value, width) in enumerate((
                ("z_id", values[0], FEATURE_DIM), ("z_dom", values[1], FEATURE_DIM), ("tx_logits", values[2], runtime["tx_classes"]),
            )):
                if not torch.is_tensor(value) or value.dtype != torch.float32 or tuple(value.shape) != (len(chunk), width) or not bool(torch.isfinite(value).all().item()):
                    raise Phase1SingleobsDualArchiveError(f"dual runtime {name} output drift")
                outputs[output_index].append(value.detach().cpu().numpy().astype(np.float32, copy=False))
    if _sha256_file(runtime["path"]) != runtime["sha256"]:
        raise Phase1SingleobsDualArchiveError("runtime changed during export")
    return tuple(np.ascontiguousarray(np.concatenate(parts), dtype=np.float32) for parts in outputs) + (invocations,)  # type: ignore[return-value]


def verify_phase1_singleobs_dual_feature_archive(archive_path: str | Path, manifest: Mapping[str, Any]) -> None:
    path = Path(archive_path)
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != MEMBERS:
            raise Phase1SingleobsDualArchiveError("archive member order/allowlist drift")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "DEVELOPMENT_ONLY_NOT_FORMAL"
        or manifest.get("artifact_stage") != "phase1_offline_before_target_access"
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("bundle_created") is not False
        or manifest.get("tx_logits_semantics") != "raw_checkpoint_column_index_only_unbound_to_class_ids"
        or manifest.get("held_runner_tx_logits_allowed") is not False
        or manifest.get("artifact", {}).get("path") != NPZ_NAME
        or manifest.get("exact_member_allowlist") != list(MEMBERS)
        or set(manifest.get("array_sha256", {})) != set(MEMBERS)
    ):
        raise Phase1SingleobsDualArchiveError("manifest array registry drift")
    if any(manifest["array_sha256"][name] != _array_sha256(value) for name, value in arrays.items()):
        raise Phase1SingleobsDualArchiveError("archive array SHA drift")
    if manifest.get("artifact", {}).get("sha256") != _sha256_file(path):
        raise Phase1SingleobsDualArchiveError("archive NPZ SHA drift")
    row_count = int(manifest.get("row_count", -1))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("tx_logits_width"), int):
        raise Phase1SingleobsDualArchiveError("manifest runtime shape metadata drift")
    cache_hashes = inputs.get("cache_npz_sha256_by_scenario")
    if (
        not isinstance(cache_hashes, Mapping)
        or tuple(cache_hashes) != FORMAL_LEO_WEAK_SCENARIOS
        or any(
            _sha256(cache_hashes[scenario], name=f"manifest cache NPZ {scenario}")
            != cache_hashes[scenario]
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        )
        or inputs.get("cache_outer_observed_schema") != LEO_WEAK_CACHE_SET_SCHEMA_V1
        or not isinstance(inputs.get("cache_inner_observed_schema_by_scenario"), Mapping)
        or tuple(inputs["cache_inner_observed_schema_by_scenario"])
        != FORMAL_LEO_WEAK_SCENARIOS
        or set(inputs["cache_inner_observed_schema_by_scenario"].values())
        != {LEO_WEAK_CACHE_SCHEMA_V1}
        or inputs.get("cache_legacy_schema_compatibility") is not True
    ):
        raise Phase1SingleobsDualArchiveError("manifest legacy cache lineage drift")
    tx_width = int(inputs["tx_logits_width"])
    for name, width in (("z_id", FEATURE_DIM), ("z_dom", FEATURE_DIM), ("tx_logits", tx_width)):
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != (row_count, width) or not np.isfinite(value).all():
            raise Phase1SingleobsDualArchiveError(f"archive {name} shape/dtype/finite drift")
    string_rows = ("labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names", "observation_ids")
    for name in string_rows:
        value = arrays[name]
        if value.ndim != 1 or len(value) != row_count or any(not item for item in value.astype(str).tolist()):
            raise Phase1SingleobsDualArchiveError(f"archive {name} row/string drift")
    classes = arrays["class_ids"].astype(str).tolist()
    if (
        arrays["class_ids"].ndim != 1
        or tuple(classes) != _class_registry(classes, logit_width=tx_width)
        or manifest.get("class_registry_sha256") != _class_registry_sha256(classes)
        or set(arrays["labels"].astype(str).tolist()) != set(classes)
    ):
        raise Phase1SingleobsDualArchiveError("archive class registry/label semantics drift")
    physical = arrays["physical_ids"].astype(str).tolist()
    observations = arrays["observation_ids"].astype(str).tolist()
    scenarios = arrays["scenario_names"].astype(str).tolist()
    if (
        len(physical) != len(set(physical))
        or len(observations) != len(set(observations))
        or len(set(zip(physical, observations))) != row_count
        or any(value not in FORMAL_LEO_WEAK_SCENARIOS for value in scenarios)
    ):
        raise Phase1SingleobsDualArchiveError("archive physical/observation/scenario semantics drift")


def export_phase1_singleobs_dual_feature_archive(
    *, cache_set_path: str | Path, cache_set_sha256: str, selection_salt_receipt_path: str | Path,
    selection_salt_receipt_sha256: str, runtime_path: str | Path, runtime_sha256: str,
    runtime_role: str, export_receipt_path: str | Path, export_receipt_sha256: str,
    parity_receipt_path: str | Path, parity_receipt_sha256: str, class_ids: tuple[str, ...] | list[str], output_dir: str | Path,
    device: str = "cuda:0", batch_size: int = RUNTIME_BATCH_CAPACITY,
) -> dict[str, Any]:
    """Export a new, immutable development-only archive from verified inputs."""
    runtime = _load_runtime_closure(runtime_path=runtime_path, runtime_sha256=runtime_sha256, runtime_role=runtime_role, export_receipt_path=export_receipt_path, export_receipt_sha256=export_receipt_sha256, parity_receipt_path=parity_receipt_path, parity_receipt_sha256=parity_receipt_sha256)
    salt = _load_selection_salt(selection_salt_receipt_path, selection_salt_receipt_sha256, checkpoint_sha=runtime["checkpoint_sha256"])
    cache_path = Path(cache_set_path).resolve()
    expected_cache = _sha256(cache_set_sha256, name="cache-set")
    if expected_cache not in KNOWN_DEVELOPMENT_SOURCE_VALIDATION_CACHE_SET_SHA256:
        raise Phase1SingleobsDualArchiveError(
            "cache-set is not a known SHA-bound development v1 lineage"
        )
    if not cache_path.is_file() or cache_path.is_symlink() or _sha256_file(cache_path) != expected_cache:
        raise Phase1SingleobsDualArchiveError("cache-set path/SHA256 drift")
    arrays_by_scenario, cache_payload, cache_audit = CACHE_LOADER(
        cache_path, expected_scope="source_validation", allowed_roles={"source"}
    )
    if _sha256_file(cache_path) != expected_cache:
        raise Phase1SingleobsDualArchiveError("cache-set changed during verified load")
    if (
        cache_audit.get("outer_observed_schema") != LEO_WEAK_CACHE_SET_SCHEMA_V1
        or set(cache_audit.get("inner_observed_schema_by_scenario", {}).values())
        != {LEO_WEAK_CACHE_SCHEMA_V1}
        or cache_audit.get("legacy_schema_compatibility") is not True
    ):
        raise Phase1SingleobsDualArchiveError("verified cache must be exact legacy v1")
    if cache_payload.get("cache_scope") != "source_validation":
        raise Phase1SingleobsDualArchiveError("cache-set scope drift")
    scenario_paths = cache_payload.get("cache_npz_by_scenario")
    scenario_hashes = cache_payload.get("cache_sha256_by_scenario")
    if (
        not isinstance(scenario_paths, Mapping)
        or not isinstance(scenario_hashes, Mapping)
        or tuple(scenario_paths) != FORMAL_LEO_WEAK_SCENARIOS
        or tuple(scenario_hashes) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise Phase1SingleobsDualArchiveError("verified cache scenario hash mapping drift")
    cache_npz_hashes: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        expected_member_sha = _sha256(
            scenario_hashes[scenario], name=f"cache NPZ {scenario}"
        )
        raw_member = Path(str(scenario_paths[scenario]))
        member = (raw_member if raw_member.is_absolute() else cache_path.parent / raw_member).resolve()
        if (
            not member.is_file()
            or member.is_symlink()
            or _sha256_file(member) != expected_member_sha
        ):
            raise Phase1SingleobsDualArchiveError(
                f"verified cache NPZ hash/path drift: {scenario}"
            )
        cache_npz_hashes[scenario] = expected_member_sha
    metadata, selected_iq = _select_verified_observations(arrays_by_scenario, salt["selection_salt_sha256"])
    if selected_iq.shape[2] != runtime["input_len"]:
        raise Phase1SingleobsDualArchiveError("received IQ/runtime input length drift")
    registry = _class_registry(class_ids, logit_width=runtime["tx_classes"])
    if set(metadata["labels"].astype(str).tolist()) != set(registry):
        raise Phase1SingleobsDualArchiveError("cache labels do not exactly match explicit class_ids")
    z_id, z_dom, tx_logits, invocations = _forward_once_per_selected_iq_batch(runtime, selected_iq, device=_resolve_device(device), batch_size=batch_size)
    arrays = {
        "z_id": z_id, "z_dom": z_dom, "tx_logits": tx_logits,
        "labels": metadata["labels"], "receiver_ids": metadata["receiver_ids"],
        "day_ids": metadata["day_ids"], "physical_ids": metadata["physical_ids"],
        "scenario_names": metadata["scenario_names"], "class_ids": np.asarray(registry, dtype=np.str_),
        "observation_ids": metadata["observation_ids"],
    }
    if tuple(arrays) != MEMBERS:
        raise Phase1SingleobsDualArchiveError("archive member construction drift")
    root = Path(output_dir).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to overwrite or reuse dual feature archive output directory")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError("dual archive staging collision")
    staging.mkdir()
    archive_path, manifest_path = staging / NPZ_NAME, staging / MANIFEST_NAME
    manifest = {
        "schema": SCHEMA, "status": "DEVELOPMENT_ONLY_NOT_FORMAL", "artifact_stage": "phase1_offline_before_target_access",
        "formal_phase2_eligible": False, "bundle_created": False,
        "exact_member_allowlist": list(MEMBERS), "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
        "class_registry_sha256": _class_registry_sha256(registry),
        "tx_logits_semantics": "raw_checkpoint_column_index_only_unbound_to_class_ids",
        "held_runner_tx_logits_allowed": False,
        "inputs": {"cache_set_sha256": expected_cache, "cache_npz_sha256_by_scenario": cache_npz_hashes, "cache_outer_observed_schema": cache_audit["outer_observed_schema"], "cache_inner_observed_schema_by_scenario": cache_audit["inner_observed_schema_by_scenario"], "cache_legacy_schema_compatibility": cache_audit["legacy_schema_compatibility"], "selection_salt_receipt_sha256": salt["sha256"], "runtime_role": runtime["role"], "runtime_sha256": runtime["sha256"], "export_receipt_sha256": runtime["export_receipt_sha256"], "parity_receipt_sha256": runtime["parity_receipt_sha256"], "checkpoint_sha256": runtime["checkpoint_sha256"], "adapter_state_sha256": runtime["adapter_state_sha256"], "input_len": runtime["input_len"], "tx_logits_width": runtime["tx_classes"], "runtime_output_schema": RUNTIME_OUTPUT_SCHEMA},
        "selection": {"selection_salt_sha256": salt["selection_salt_sha256"], "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS), "selected_observations_per_physical_id": 1, "observation_ids": "verbatim_verified_selected_overlay_ids"},
        "runtime_audit": {"same_iq_outputs": ["z_id", "z_dom", "tx_logits"], "single_runtime_call_per_selected_iq_batch": True, "runtime_invocations": invocations, "batch_size": int(batch_size)},
        "row_count": int(len(selected_iq)), "cache_loader_audit_sha256": hashlib.sha256(_canonical_json(cache_audit)).hexdigest(),
        "access_audit": {"clean_iq_access": False, "target_access": False, "query_access": False, "received_iq_persisted": False, "raw_iq_persisted": False},
    }
    try:
        with archive_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        manifest["artifact"] = {"path": NPZ_NAME, "sha256": _sha256_file(archive_path)}
        _write_new(manifest_path, _canonical_json(manifest) + b"\n")
        verify_phase1_singleobs_dual_feature_archive(archive_path, manifest)
        staging.rename(root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    archive_path, manifest_path = root / NPZ_NAME, root / MANIFEST_NAME
    return {"status": manifest["status"], "formal_phase2_eligible": False, "archive_path": str(archive_path), "archive_sha256": manifest["artifact"]["sha256"], "manifest_path": str(manifest_path), "manifest_sha256": _sha256_file(manifest_path), "row_count": int(len(selected_iq))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-set", type=Path, required=True); parser.add_argument("--cache-set-sha256", required=True)
    parser.add_argument("--selection-salt-receipt", type=Path, required=True); parser.add_argument("--selection-salt-receipt-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True); parser.add_argument("--runtime-sha256", required=True); parser.add_argument("--runtime-role", choices=RUNTIME_ROLES, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True); parser.add_argument("--export-receipt-sha256", required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True); parser.add_argument("--parity-receipt-sha256", required=True)
    parser.add_argument("--class-ids", required=True, help="Ordered comma-separated class registry; not a tx-logit column mapping")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=RUNTIME_BATCH_CAPACITY)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    values = vars(args)
    values["class_ids"] = tuple(value.strip() for value in str(values["class_ids"]).split(",") if value.strip())
    print(json.dumps(export_phase1_singleobs_dual_feature_archive(**values), sort_keys=True))


if __name__ == "__main__":
    main()
