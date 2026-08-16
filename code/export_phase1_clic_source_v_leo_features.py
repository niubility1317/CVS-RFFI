"""Export one frozen checkpoint's held source-V LEO features exactly once.

This is deliberately a file-only post-target completion audit.  It reopens a
training-v5 final checkpoint, clean-v4 identity evidence, one Task1 immutable
source-V received-IQ cache and the already sealed PAIR-v3 source policy state.
It never opens source-L/proxy feature rows, target artifacts, or a fitting,
threshold, adaptation, selection or retry path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import build_phase1_clic_source_v_leo_iq as _cache
import evaluate_phase1_clic_postfreeze_pair as _pair
import export_phase1_clic_features as _clean


SOURCE_V_FEATURE_SCHEMA = "cvs.phase1.clic_source_v_leo_export.v1"
SOURCE_V_FEATURE_BINDING_SCHEMA = "cvs.phase1.clic_source_v_leo_binding.v1"
EXPECTED_CACHE_RUN_ID = _cache.EXPECTED_CACHE_RUN_ID
EXPECTED_PAIR_SCHEMA = _pair.EXPECTED_PAIR_SCHEMA
EXPECTED_SCENARIOS = tuple(_pair.EXPECTED_SCENARIOS)
SOURCE_V_ROLE = _cache.SOURCE_V_ROLE
TECHNICAL_SMOKE_ROOT_NAME = ".smoke_phase1_clic_source_metrics_20260813_v2_F1"
EXPECTED_PAIR_RUN_ID = "phase1_clic_source_pair_20260812_v3"
EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"


class CLICSourceVFeatureExportError(RuntimeError):
    """Raised when a source-V-only feature forward cannot close safely."""


def validate_source_v_execution_roots(
    *,
    training_root: str | Path,
    clean_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    terminal_path: str | Path,
    pair_path: str | Path,
    formal_project_root: str | Path | None,
    fold_index: int,
    candidate_id: str,
    technical_smoke: bool,
) -> None:
    """Close formal roots, or the sole F1 technical-smoke exception, fail closed.

    Formal source-metrics forwards retain the original common ``runs`` parent
    requirement.  The only independent-root form is the pre-registered F1
    technical smoke, whose inputs remain the original canonical formal files
    while its cache and output share one dedicated v2 smoke leaf.
    """

    if type(technical_smoke) is not bool:
        raise CLICSourceVFeatureExportError("source-V technical smoke control must be boolean")
    if fold_index not in range(1, 7):
        raise CLICSourceVFeatureExportError("source-V execution root fold is invalid")
    raw_training = Path(training_root)
    raw_clean = Path(clean_path)
    raw_cache = Path(cache_root)
    raw_output = Path(output_root)
    raw_checkpoint = Path(checkpoint_path)
    raw_terminal = Path(terminal_path)
    raw_pair = Path(pair_path)
    training = raw_training.resolve()
    clean = raw_clean.resolve()
    cache = raw_cache.resolve()
    output = raw_output.resolve()
    checkpoint = raw_checkpoint.resolve()
    terminal = raw_terminal.resolve()
    pair = raw_pair.resolve()
    clean_root = clean.parent.parent
    expected_candidates = {
        f"F{fold_index}C_CLIC12",
        f"F{fold_index}G_CLIC12",
    }
    if candidate_id not in expected_candidates:
        raise CLICSourceVFeatureExportError("source-V candidate/fold root binding drifted")
    if technical_smoke and (
        fold_index != 1 or candidate_id not in {"F1C_CLIC12", "F1G_CLIC12"}
    ):
        raise CLICSourceVFeatureExportError("source-V technical smoke is restricted to F1 C/G")
    if (
        training.name != _clean.EXPECTED_TRAINING_RUN_ID
        or clean_root.name != _cache.EXPECTED_CLEAN_RUN_ID
        or cache.name != EXPECTED_CACHE_RUN_ID
        or output.name != EXPECTED_CACHE_RUN_ID
    ):
        raise CLICSourceVFeatureExportError("source-V training/clean/cache run identity drifted")
    if checkpoint != training / candidate_id / "final_ssdg.pth":
        raise CLICSourceVFeatureExportError("source-V checkpoint path binding drifted")
    if terminal != checkpoint.parent / "phase1_clic_terminal_receipt.json":
        raise CLICSourceVFeatureExportError("source-V terminal path binding drifted")
    if clean != clean_root / candidate_id / "source_clean_proxy.npz":
        raise CLICSourceVFeatureExportError("source-V clean-v4 path binding drifted")

    if not technical_smoke:
        formal_parent = training.parent
        if (
            formal_parent.name != "runs"
            or clean_root.parent != formal_parent
            or cache.parent != formal_parent
            or output.parent != formal_parent
        ):
            raise CLICSourceVFeatureExportError(
                "source-V formal training/clean/cache/output root binding drifted"
            )
        return

    if (
        not isinstance(formal_project_root, str)
        or formal_project_root != EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT
    ):
        raise CLICSourceVFeatureExportError(
            "source-V technical smoke formal project root must equal the frozen canonical root"
        )
    formal_project = Path(formal_project_root)
    formal_runs = formal_project / "runs"
    expected_training = formal_runs / _clean.EXPECTED_TRAINING_RUN_ID
    expected_checkpoint = expected_training / candidate_id / "final_ssdg.pth"
    expected_terminal = expected_checkpoint.parent / "phase1_clic_terminal_receipt.json"
    expected_clean = (
        formal_runs
        / _cache.EXPECTED_CLEAN_RUN_ID
        / candidate_id
        / "source_clean_proxy.npz"
    )
    expected_pair = formal_runs / EXPECTED_PAIR_RUN_ID / "F1_C_vs_G_pair.json"
    expected_smoke_leaf = (
        formal_runs / TECHNICAL_SMOKE_ROOT_NAME / EXPECTED_CACHE_RUN_ID
    )
    anchored_paths = (
        (raw_training, expected_training),
        (raw_checkpoint, expected_checkpoint),
        (raw_terminal, expected_terminal),
        (raw_clean, expected_clean),
        (raw_pair, expected_pair),
        (raw_cache, expected_smoke_leaf),
        (raw_output, expected_smoke_leaf),
    )
    if any(not actual.is_absolute() or actual != expected for actual, expected in anchored_paths):
        raise CLICSourceVFeatureExportError(
            "source-V technical smoke canonical formal input/output root binding drifted"
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CLICSourceVFeatureExportError("source-V feature state cannot be canonicalized") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CLICSourceVFeatureExportError(f"{label} SHA256 is absent or invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CLICSourceVFeatureExportError(f"{label} SHA256 is not hexadecimal") from exc
    return value


def validate_pair_single_leo_common_binding(value: Any) -> dict[str, Any]:
    """Require the PAIR-v3 source-L single-observation identity before reuse."""

    expected_fields = {
        "received_iq_sha256",
        "physical_order_sha256",
        "source_only",
        "single_leo_observation",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CLICSourceVFeatureExportError("PAIR-v3 single-LEO common binding is incomplete")
    binding = dict(value)
    _require_sha256(binding["received_iq_sha256"], label="PAIR-v3 single-LEO received-IQ")
    _require_sha256(binding["physical_order_sha256"], label="PAIR-v3 single-LEO physical order")
    if binding["source_only"] is not True or binding["single_leo_observation"] is not True:
        raise CLICSourceVFeatureExportError("PAIR-v3 single-LEO source-only observation binding drifted")
    return binding


def validate_pair_source_l_policy_binding(
    common_binding: Any, policies: Any
) -> None:
    """Require each sealed source-L scene policy to consume PAIR's common bytes."""

    binding = validate_pair_single_leo_common_binding(common_binding)
    if not isinstance(policies, Mapping) or set(str(scene) for scene in policies) != set(EXPECTED_SCENARIOS):
        raise CLICSourceVFeatureExportError("PAIR-v3 source-L policy scene coverage drifted")
    for scene in EXPECTED_SCENARIOS:
        policy = policies[scene]
        if not isinstance(policy, Mapping):
            raise CLICSourceVFeatureExportError(f"PAIR-v3 source-L {scene} policy is malformed")
        for field in ("received_iq_sha256", "physical_order_sha256"):
            observed = _require_sha256(policy.get(field), label=f"PAIR-v3 source-L {scene} policy {field}")
            if observed != binding[field]:
                raise CLICSourceVFeatureExportError(
                    f"PAIR-v3 source-L {scene} policy/single-LEO {field} binding drifted"
                )


def _parse_source_tx_ids(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        parsed = tuple(str(part) for part in value)
    if len(parsed) != 4 or len(set(parsed)) != 4:
        raise CLICSourceVFeatureExportError("source-V forward requires exactly four source TX IDs")
    return parsed


def _strict_text(values: Any, *, label: str, row_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind not in {"U", "S"}:
        raise CLICSourceVFeatureExportError(f"{label} must be a non-object text array")
    result = np.asarray(array.reshape(-1), dtype=str)
    if result.size != row_count or np.any(result == ""):
        raise CLICSourceVFeatureExportError(f"{label} has invalid row alignment")
    return result


def _safe_float32(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or array.dtype.kind != "f":
        raise CLICSourceVFeatureExportError(f"{label} must be a floating non-object array")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise CLICSourceVFeatureExportError(f"{label} contains non-finite values")
    return result


def numpy_float32_to_tensor(value: np.ndarray) -> torch.Tensor:
    """Use only the buffer protocol; do not call the legacy NumPy C bridge."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.size <= 0 or not np.isfinite(source).all():
        raise CLICSourceVFeatureExportError("source-V received-IQ is empty or non-finite")
    try:
        return torch.frombuffer(memoryview(source), dtype=torch.float32).reshape(source.shape).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLICSourceVFeatureExportError("source-V safe NumPy/Torch bridge failed") from exc


def read_source_v_cache_snapshot(
    *,
    cache_path: str | Path,
    cache_receipt_path: str | Path,
    fold_index: int,
    source_tx_ids: Sequence[str],
    expected_row_count: int = _cache.FROZEN_SOURCE_V_ROW_COUNT,
) -> dict[str, Any]:
    """Reopen only the Task1 V-only cache and its immutable source-only receipt."""

    if fold_index not in range(1, 7):
        raise CLICSourceVFeatureExportError("source-V cache fold must be F1..F6")
    source_order = _parse_source_tx_ids(source_tx_ids)
    cache = Path(cache_path).resolve()
    receipt_path = Path(cache_receipt_path).resolve()
    if not cache.is_file() or not receipt_path.is_file():
        raise CLICSourceVFeatureExportError("source-V cache or receipt is missing")
    receipt_sha_before = _sha256_file(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLICSourceVFeatureExportError("source-V cache receipt is unreadable") from exc
    if not isinstance(receipt, Mapping):
        raise CLICSourceVFeatureExportError("source-V cache receipt must be an object")
    expected = {
        "schema": _cache.SOURCE_V_LEO_CACHE_SCHEMA,
        "method": "P1_CLIC",
        "role": SOURCE_V_ROLE,
        "source_v_only": True,
        "post_target_completion_audit_non_selection": True,
        "fold_index": fold_index,
        "training_run_id": _clean.EXPECTED_TRAINING_RUN_ID,
        "clean_evidence_run_id": _cache.EXPECTED_CLEAN_RUN_ID,
        "same_received_iq_bytes_for_c_and_g": True,
        "single_leo_observation_per_physical_sample": True,
        "cross_scene_physical_sample_reuse": False,
        "clean_source_runtime_access": False,
        "target_access": False,
        "query_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "proxy_forward_rows": 0,
        "source_l_forward_rows": 0,
        "source_v_forward_rows": 0,
        "selection_access": False,
        "retry_access": False,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value or type(receipt.get(field)) is not type(expected_value):
            raise CLICSourceVFeatureExportError(f"source-V cache receipt {field} drifted")
    if tuple(str(item) for item in receipt.get("source_tx_ids", ())) != source_order:
        raise CLICSourceVFeatureExportError("source-V cache source class order drifted")
    if int(receipt.get("source_validation_row_count", -1)) != expected_row_count:
        raise CLICSourceVFeatureExportError("source-V cache V row count drifted")
    for field in (
        "source_validation_indices_sha256",
        "source_validation_physical_order_sha256",
        "physical_order_sha256",
        "received_iq_npz_sha256",
    ):
        _require_sha256(receipt.get(field), label=f"source-V cache receipt {field}")
    if tuple(str(item) for item in receipt.get("formal_scenarios", ())) != EXPECTED_SCENARIOS:
        raise CLICSourceVFeatureExportError("source-V cache formal scene order drifted")
    if Path(str(receipt.get("received_iq_npz_path", ""))).resolve() != cache:
        raise CLICSourceVFeatureExportError("source-V cache receipt path drifted")
    cache_sha_before = _sha256_file(cache)
    if receipt.get("received_iq_npz_sha256") != cache_sha_before:
        raise CLICSourceVFeatureExportError("source-V cache receipt SHA drifted")
    required = {"received_iq", "tx_ids", "rx_ids", "day_ids", "physical_sample_id", "sat_scenarios"}
    try:
        with np.load(cache, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise CLICSourceVFeatureExportError("source-V cache member allowlist drifted")
            arrays = {name: np.array(archive[name], copy=True) for name in required}
    except CLICSourceVFeatureExportError:
        raise
    except (OSError, ValueError) as exc:
        raise CLICSourceVFeatureExportError("source-V cache is unreadable") from exc
    if _sha256_file(cache) != cache_sha_before or _sha256_file(receipt_path) != receipt_sha_before:
        raise CLICSourceVFeatureExportError("source-V cache or receipt changed while opening")
    iq = _safe_float32(arrays["received_iq"], label="source-V received-IQ")
    if iq.ndim != 3 or iq.shape[0] != expected_row_count or iq.shape[1] != 2:
        raise CLICSourceVFeatureExportError("source-V received-IQ shape/row count drifted")
    tx_ids = _strict_text(arrays["tx_ids"], label="source-V TX IDs", row_count=expected_row_count)
    rx_ids = _strict_text(arrays["rx_ids"], label="source-V RX IDs", row_count=expected_row_count)
    day_ids = _strict_text(arrays["day_ids"], label="source-V day IDs", row_count=expected_row_count)
    physical_ids = _strict_text(arrays["physical_sample_id"], label="source-V physical IDs", row_count=expected_row_count)
    scenes = _strict_text(arrays["sat_scenarios"], label="source-V scenes", row_count=expected_row_count)
    if set(tx_ids).difference(source_order) or len(set(physical_ids.tolist())) != expected_row_count:
        raise CLICSourceVFeatureExportError("source-V cache class or physical uniqueness drifted")
    if set(scenes) != set(EXPECTED_SCENARIOS):
        raise CLICSourceVFeatureExportError("source-V cache formal scene coverage drifted")
    if _canonical_sha256(physical_ids.tolist()) != receipt.get("physical_order_sha256"):
        raise CLICSourceVFeatureExportError("source-V cache physical-order receipt drifted")
    return {
        "received_iq": iq,
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "physical_ids": physical_ids,
        "sat_scenarios": scenes,
        "row_count": expected_row_count,
        "cache_sha256": cache_sha_before,
        "cache_receipt_sha256": receipt_sha_before,
        "receipt": dict(receipt),
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
    }


def validate_source_v_clean_v4_binding(
    *, snapshot: Mapping[str, Any], clean_binding: Mapping[str, Any]
) -> None:
    """Bind the V-only received cache to the same clean-v4 held-V rows.

    ``clean_binding`` is the metadata-only projection returned by Task1's
    strict clean-v4 reopener; it never exposes or scores source-L/proxy
    feature rows.
    """

    if not isinstance(snapshot, Mapping) or not isinstance(clean_binding, Mapping):
        raise CLICSourceVFeatureExportError("source-V/clean-v4 binding is malformed")
    receipt = snapshot.get("receipt")
    if not isinstance(receipt, Mapping):
        raise CLICSourceVFeatureExportError("source-V cache receipt is absent for clean-v4 binding")
    cache_index_sha = _require_sha256(
        receipt.get("source_validation_indices_sha256"), label="source-V cache validation index"
    )
    cache_order_sha = _require_sha256(
        receipt.get("source_validation_physical_order_sha256"), label="source-V cache validation order"
    )
    clean_index_sha = _require_sha256(
        clean_binding.get("validation_indices_sha256"), label="clean-v4 validation index"
    )
    clean_order_sha = _require_sha256(
        clean_binding.get("validation_metadata_order_sha256"), label="clean-v4 validation order"
    )
    if cache_index_sha != clean_index_sha or cache_order_sha != clean_order_sha:
        raise CLICSourceVFeatureExportError("source-V cache/clean-v4 validation index/order binding drifted")
    physical = np.asarray(snapshot.get("physical_ids"), dtype=str).reshape(-1)
    row_count = int(physical.size)
    if row_count <= 0 or np.any(physical == "") or len(set(physical.tolist())) != row_count:
        raise CLICSourceVFeatureExportError("source-V cache physical rows are invalid for clean-v4 binding")
    manifest = clean_binding.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CLICSourceVFeatureExportError("clean-v4 manifest is absent for source-V physical binding")
    dataset_sha256 = _require_sha256(
        manifest.get("wisig_pkl_sha256"), label="clean-v4 manifest WiSig dataset"
    )
    clean_tx_ids = _strict_text(
        clean_binding.get("validation_tx_ids"),
        label="clean-v4 validation tx_ids",
        row_count=row_count,
    )
    clean_rx_ids = _strict_text(
        clean_binding.get("validation_rx_ids"),
        label="clean-v4 validation rx_ids",
        row_count=row_count,
    )
    clean_day_ids = _strict_text(
        clean_binding.get("validation_day_ids"),
        label="clean-v4 validation day_ids",
        row_count=row_count,
    )
    clean_eq_ids = _strict_text(
        clean_binding.get("validation_eq_ids"),
        label="clean-v4 validation eq_ids",
        row_count=row_count,
    )
    clean_sig_ids = _strict_text(
        clean_binding.get("validation_sig_ids"),
        label="clean-v4 validation sig_ids",
        row_count=row_count,
    )
    axes = {
        "tx_ids": (snapshot.get("tx_ids"), clean_tx_ids),
        "rx_ids": (snapshot.get("rx_ids"), clean_rx_ids),
        "day_ids": (snapshot.get("day_ids"), clean_day_ids),
    }
    for field, (cache_values, clean_values) in axes.items():
        cache_text = _strict_text(cache_values, label=f"source-V cache {field}", row_count=row_count)
        if not np.array_equal(cache_text, clean_values):
            raise CLICSourceVFeatureExportError(f"source-V cache/clean-v4 {field} row binding drifted")
    expected_physical = np.asarray(
        [
            _cache._physical_sample_id(
                dataset_sha256=dataset_sha256,
                tx_id=str(tx_id),
                rx_id=str(rx_id),
                day_id=str(day_id),
                eq_id=str(eq_id),
                sig_id=str(sig_id),
            )
            for tx_id, rx_id, day_id, eq_id, sig_id in zip(
                clean_tx_ids,
                clean_rx_ids,
                clean_day_ids,
                clean_eq_ids,
                clean_sig_ids,
                strict=True,
            )
        ],
        dtype=str,
    )
    if len(set(expected_physical.tolist())) != row_count:
        raise CLICSourceVFeatureExportError(
            "clean-v4 validation metadata does not map to unique Task1 physical IDs"
        )
    if not np.array_equal(physical, expected_physical):
        raise CLICSourceVFeatureExportError(
            "source-V cache/clean-v4 Task1 physical ID row binding drifted"
        )


def validate_source_v_forward_payload(
    *,
    payload: Mapping[str, Any],
    physical_ids: np.ndarray,
    source_tx_ids: Sequence[str],
    expected_row_count: int,
    expected_tx_ids: np.ndarray | None = None,
    expected_rx_ids: np.ndarray | None = None,
    expected_day_ids: np.ndarray | None = None,
    expected_scenarios: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate the exact one-forward-per-source-V output with no side roles."""

    if not isinstance(payload, Mapping):
        raise CLICSourceVFeatureExportError("source-V forward payload must be a mapping")
    required = {
        "features", "tx_logits", "raw_labels", "domain_labels", "tx_ids", "rx_ids", "day_ids",
        "eq_ids", "sig_ids", "dataset_role", "channel_views", "sat_scenarios",
    }
    if set(payload) != required:
        raise CLICSourceVFeatureExportError("source-V forward payload member set drifted")
    features = _safe_float32(payload["features"], label="source-V features")
    logits = _safe_float32(payload["tx_logits"], label="source-V logits")
    if features.ndim != 2 or features.shape[0] != expected_row_count or logits.shape != (expected_row_count, 4):
        raise CLICSourceVFeatureExportError("source-V forward row count or local4 logits drifted")
    observed_physical = _strict_text(payload["sig_ids"], label="source-V forward physical IDs", row_count=expected_row_count)
    expected_physical = _strict_text(physical_ids, label="source-V expected physical IDs", row_count=expected_row_count)
    if len(set(observed_physical.tolist())) != expected_row_count or not np.array_equal(observed_physical, expected_physical):
        raise CLICSourceVFeatureExportError("source-V forward must contain every physical ID exactly once")
    source_order = _parse_source_tx_ids(source_tx_ids)
    tx_ids = _strict_text(payload["tx_ids"], label="source-V forward TX IDs", row_count=expected_row_count)
    if set(tx_ids).difference(source_order):
        raise CLICSourceVFeatureExportError("source-V forward contains a non-source TX")
    roles = _strict_text(payload["dataset_role"], label="source-V forward roles", row_count=expected_row_count)
    views = _strict_text(payload["channel_views"], label="source-V forward views", row_count=expected_row_count)
    if set(roles) != {SOURCE_V_ROLE} or set(views) != {"received_existing"}:
        raise CLICSourceVFeatureExportError("source-V forward role/view contract drifted")
    rx_ids = _strict_text(payload["rx_ids"], label="source-V forward rx_ids", row_count=expected_row_count)
    day_ids = _strict_text(payload["day_ids"], label="source-V forward day_ids", row_count=expected_row_count)
    _strict_text(payload["eq_ids"], label="source-V forward eq_ids", row_count=expected_row_count)
    scenarios = _strict_text(payload["sat_scenarios"], label="source-V forward sat_scenarios", row_count=expected_row_count)
    expected_metadata = {
        "tx_ids": expected_tx_ids,
        "rx_ids": expected_rx_ids,
        "day_ids": expected_day_ids,
        "sat_scenarios": expected_scenarios,
    }
    if any(value is not None for value in expected_metadata.values()):
        if any(value is None for value in expected_metadata.values()):
            raise CLICSourceVFeatureExportError("source-V cache metadata binding must include TX/RX/day/scene together")
        observed_metadata = {
            "tx_ids": tx_ids,
            "rx_ids": rx_ids,
            "day_ids": day_ids,
            "sat_scenarios": scenarios,
        }
        for field, expected_values in expected_metadata.items():
            expected_text = _strict_text(
                expected_values,
                label=f"source-V expected cache {field}",
                row_count=expected_row_count,
            )
            if not np.array_equal(observed_metadata[field], expected_text):
                raise CLICSourceVFeatureExportError(f"source-V forward {field} cache metadata binding drifted")
    labels = np.asarray(payload["raw_labels"])
    domains = np.asarray(payload["domain_labels"])
    if (
        labels.dtype.hasobject
        or domains.dtype.hasobject
        or labels.dtype.kind not in {"i", "u"}
        or domains.dtype.kind not in {"i", "u"}
        or labels.reshape(-1).size != expected_row_count
        or domains.reshape(-1).size != expected_row_count
    ):
        raise CLICSourceVFeatureExportError("source-V forward label/domain integer rows drifted")
    label_rows = np.asarray(labels, dtype=np.int64).reshape(-1)
    expected_labels = np.asarray([source_order.index(str(tx_id)) for tx_id in tx_ids], dtype=np.int64)
    if not np.array_equal(label_rows, expected_labels):
        raise CLICSourceVFeatureExportError("source-V forward raw label/TX class binding drifted")
    return {
        "features": features,
        "tx_logits": logits,
        "single_leo_forward_count": expected_row_count,
        "source_v_forward_rows": expected_row_count,
        "source_l_forward_rows": 0,
        "proxy_forward_rows": 0,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
    }


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> Any:
    """Reuse Task1's no-replace, pre-sealed immutable publication primitive."""

    try:
        return _cache._atomic_save_npz(path, payload)
    except Exception as exc:
        raise CLICSourceVFeatureExportError(f"source-V feature immutable publish failed: {exc}") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Any:
    """Reuse Task1's no-replace, pre-sealed immutable publication primitive."""

    try:
        return _cache._atomic_write_json(path, payload)
    except Exception as exc:
        raise CLICSourceVFeatureExportError(f"source-V feature binding immutable publish failed: {exc}") from exc


def _load_pair_policy_state(
    *,
    pair_json_path: str | Path,
    fold_index: int,
    arm: str,
    checkpoint_sha256: str,
    terminal_receipt_sha256: str,
    source_tx_ids: tuple[str, ...],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    pair_path = Path(pair_json_path).resolve()
    if not pair_path.is_file():
        raise CLICSourceVFeatureExportError("PAIR-v3 source policy receipt is missing")
    pair_sha = _sha256_file(pair_path)
    try:
        pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLICSourceVFeatureExportError("PAIR-v3 source policy receipt is unreadable") from exc
    if not isinstance(pair_payload, Mapping):
        raise CLICSourceVFeatureExportError("PAIR-v3 source policy receipt is malformed")
    if (
        pair_payload.get("schema") != EXPECTED_PAIR_SCHEMA
        or pair_payload.get("fold_index") != fold_index
        or pair_payload.get("source_only") is not True
        or pair_payload.get("target_artifacts_present") is not False
        or tuple(str(item) for item in pair_payload.get("source_tx_ids", ())) != source_tx_ids
    ):
        raise CLICSourceVFeatureExportError("PAIR-v3 source-only policy binding drifted")
    common_binding = validate_pair_single_leo_common_binding(pair_payload.get("single_leo_common_binding"))
    states = pair_payload.get("clic_source_policy_state")
    proxy = pair_payload.get("proxy_diagnostic")
    if not isinstance(states, Mapping) or not isinstance(proxy, Mapping) or set(states) != {"C", "G"} or set(proxy) != {"C", "G"}:
        raise CLICSourceVFeatureExportError("PAIR-v3 policy/proxy state coverage drifted")
    try:
        state = _pair._validated_clic_source_policy_state(
            states[arm],
            fold_index=fold_index,
            arm=arm,
            checkpoint_sha256=checkpoint_sha256,
            terminal_receipt_sha256=terminal_receipt_sha256,
        )
    except _pair.CLICPostfreezePairError as exc:
        raise CLICSourceVFeatureExportError(f"PAIR-v3 source policy state is invalid: {exc}") from exc
    validate_pair_source_l_policy_binding(common_binding, state["policies"])
    diagnostic = proxy[arm]
    if not isinstance(diagnostic, Mapping) or diagnostic.get("schema") != "cvs.phase1.clic_proxy_diagnostic.v1":
        raise CLICSourceVFeatureExportError("PAIR-v3 proxy diagnostic schema drifted")
    for field in ("AUROC_unknown", "u_gap"):
        value = diagnostic.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise CLICSourceVFeatureExportError(f"PAIR-v3 proxy diagnostic {field} is non-finite")
    for field in ("fit_rows", "threshold_fit_rows", "source_validation_fit_rows", "proxy_fit_rows", "source_validation_threshold_rows", "proxy_threshold_rows"):
        if diagnostic.get(field) != 0:
            raise CLICSourceVFeatureExportError(f"PAIR-v3 proxy diagnostic {field} must remain zero")
    if _sha256_file(pair_path) != pair_sha:
        raise CLICSourceVFeatureExportError("PAIR-v3 policy receipt changed while opening")
    return state, pair_sha, dict(diagnostic)


def export_source_v_leo_features(args: argparse.Namespace) -> dict[str, Any]:
    """Forward each immutable source-V received row once for a C or G final checkpoint."""

    from torch.utils.data import DataLoader, Dataset
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from export_spaceborne_features import extract_features_with_metadata

    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise CLICSourceVFeatureExportError("source-V feature export requires F1..F6 and C/G")
    candidate = f"F{fold}{arm}_CLIC12"
    if str(args.candidate_id) != candidate:
        raise CLICSourceVFeatureExportError("source-V feature candidate/fold/arm binding drifted")
    source_tx_ids = _parse_source_tx_ids(args.source_tx_ids)
    training_root = Path(args.training_run_root).resolve()
    cache_root = Path(args.cache_run_root).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    terminal_path = Path(args.terminal_receipt_json).resolve()
    clean_path = Path(args.clean_npz).resolve()
    cache_path = Path(args.source_v_received_iq_npz).resolve()
    cache_receipt_path = Path(args.source_v_received_iq_receipt_json).resolve()
    pair_path = Path(args.pair_json).resolve()
    output_path = Path(args.out_npz).resolve()
    binding_path = Path(args.binding_json).resolve()
    validate_source_v_execution_roots(
        training_root=args.training_run_root,
        clean_path=args.clean_npz,
        cache_root=args.cache_run_root,
        output_root=args.output_root,
        checkpoint_path=args.ckpt,
        terminal_path=args.terminal_receipt_json,
        pair_path=args.pair_json,
        formal_project_root=getattr(args, "formal_project_root", None),
        fold_index=fold,
        candidate_id=candidate,
        technical_smoke=getattr(args, "technical_smoke", False),
    )
    shared_dir = cache_root / f"F{fold}_SHARED"
    if cache_path != shared_dir / "source_validation_known_leo_weak.npz" or cache_receipt_path != shared_dir / "source_validation_known_leo_weak.receipt.json":
        raise CLICSourceVFeatureExportError("source-V shared cache path binding drifted")
    candidate_dir = output_root / candidate
    if output_path.parent != candidate_dir or binding_path.parent != candidate_dir:
        raise CLICSourceVFeatureExportError("source-V feature output binding drifted")
    if output_path.exists() or binding_path.exists():
        raise CLICSourceVFeatureExportError("refusing to overwrite source-V feature output")
    if not checkpoint_path.is_file() or not terminal_path.is_file() or not clean_path.is_file() or not pair_path.is_file():
        raise CLICSourceVFeatureExportError("source-V checkpoint, terminal, clean-v4, or PAIR-v3 input is missing")
    input_paths = {
        "checkpoint": checkpoint_path,
        "terminal": terminal_path,
        "clean": clean_path,
        "cache": cache_path,
        "cache_receipt": cache_receipt_path,
        "pair": pair_path,
    }
    input_hashes_before = {name: _sha256_file(path) for name, path in input_paths.items()}
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICSourceVFeatureExportError("source-V final checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CLICSourceVFeatureExportError("source-V final checkpoint payload is malformed")
    checkpoint_args = checkpoint["args"]
    try:
        known = _clean._parse_csv(checkpoint_args.get("phase1_source_known_validation_tx_ids", ""), label="source-V held validation TX")
        proxy = _clean._parse_csv(checkpoint_args.get("phase1_source_proxy_unknown_tx_ids", ""), label="source-V proxy TX")
        _, terminal_receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            terminal_receipt_path=terminal_path,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICSourceVFeatureExportError(f"source-V checkpoint/terminal strict reopen failed: {exc}") from exc
    if observed_arm != arm:
        raise CLICSourceVFeatureExportError("source-V terminal arm binding drifted")
    snapshot = read_source_v_cache_snapshot(
        cache_path=cache_path,
        cache_receipt_path=cache_receipt_path,
        fold_index=fold,
        source_tx_ids=source_tx_ids,
    )
    if snapshot["cache_sha256"] != input_hashes_before["cache"] or snapshot["cache_receipt_sha256"] != input_hashes_before["cache_receipt"]:
        raise CLICSourceVFeatureExportError("source-V cache changed before snapshot reopening")
    if snapshot["receipt"].get("checkpoint_sha256_by_arm", {}).get(arm) != input_hashes_before["checkpoint"]:
        raise CLICSourceVFeatureExportError("source-V cache/checkpoint SHA binding drifted")
    if snapshot["receipt"].get("terminal_receipt_sha256_by_arm", {}).get(arm) != input_hashes_before["terminal"]:
        raise CLICSourceVFeatureExportError("source-V cache/terminal SHA binding drifted")
    try:
        clean_binding = _cache._read_clean_validation_binding(
            path=clean_path,
            arm=arm,
            fold_index=fold,
            source_tx_ids=source_tx_ids,
            checkpoint_sha256=input_hashes_before["checkpoint"],
            terminal_sha256=input_hashes_before["terminal"],
        )
    except _cache.CLICSourceVLeoCacheError as exc:
        raise CLICSourceVFeatureExportError(f"source-V clean-v4 strict reopen failed: {exc}") from exc
    if clean_binding["sha256"] != input_hashes_before["clean"]:
        raise CLICSourceVFeatureExportError("source-V clean-v4 changed before V forward")
    validate_source_v_clean_v4_binding(snapshot=snapshot, clean_binding=clean_binding)
    state, pair_sha, proxy_diagnostic = _load_pair_policy_state(
        pair_json_path=pair_path,
        fold_index=fold,
        arm=arm,
        checkpoint_sha256=input_hashes_before["checkpoint"],
        terminal_receipt_sha256=input_hashes_before["terminal"],
        source_tx_ids=source_tx_ids,
    )
    if pair_sha != input_hashes_before["pair"]:
        raise CLICSourceVFeatureExportError("PAIR-v3 receipt changed before forward")
    iq = snapshot["received_iq"]
    if iq.shape[2] != int(checkpoint_args.get("wisig_out_len", 256)):
        raise CLICSourceVFeatureExportError("source-V received-IQ length does not match checkpoint")
    labels = np.asarray([source_tx_ids.index(str(item)) for item in snapshot["tx_ids"]], dtype=np.int64)

    class SourceVReceivedIQDataset(Dataset):
        def __len__(self) -> int:
            return int(snapshot["row_count"])

        def __getitem__(self, index: int):
            return (
                numpy_float32_to_tensor(iq[index]),
                torch.tensor(int(labels[index]), dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                {
                    "tx": str(snapshot["tx_ids"][index]),
                    "rx": str(snapshot["rx_ids"][index]),
                    "day": str(snapshot["day_ids"][index]),
                    "equalized": "existing_received_iq",
                    "sig_i": str(snapshot["physical_ids"][index]),
                },
            )

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model, load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(checkpoint_args.get("wisig_out_len", 256)), device=device
    )
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise CLICSourceVFeatureExportError("source-V feature batch size must be positive")
    loader = DataLoader(SourceVReceivedIQDataset(), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    payload = extract_features_with_metadata(
        model,
        loader,
        device=device,
        feature_name="z_id",
        role=SOURCE_V_ROLE,
        channel_view="received_existing",
        satellite_tta_policy="none",
        safe_numpy_bridge=True,
    )
    payload["sat_scenarios"] = np.asarray(snapshot["sat_scenarios"], dtype=str)
    validated = validate_source_v_forward_payload(
        payload=payload,
        physical_ids=snapshot["physical_ids"],
        source_tx_ids=source_tx_ids,
        expected_row_count=int(snapshot["row_count"]),
        expected_tx_ids=snapshot["tx_ids"],
        expected_rx_ids=snapshot["rx_ids"],
        expected_day_ids=snapshot["day_ids"],
        expected_scenarios=snapshot["sat_scenarios"],
    )
    payload["z_id"] = np.asarray(validated["features"], dtype=np.float32)
    payload["physical_sample_id"] = np.asarray(snapshot["physical_ids"], dtype=str)
    manifest = {
        "schema": SOURCE_V_FEATURE_SCHEMA,
        "method": "P1_CLIC",
        "candidate_id": candidate,
        "fold_index": fold,
        "arm": arm,
        "source_only": True,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "checkpoint_sha256": input_hashes_before["checkpoint"],
        "terminal_receipt_sha256": input_hashes_before["terminal"],
        "clean_v4_sha256": input_hashes_before["clean"],
        "source_validation_indices_sha256": clean_binding["validation_indices_sha256"],
        "source_validation_physical_order_sha256": clean_binding["validation_metadata_order_sha256"],
        "clean_evidence_run_id": _cache.EXPECTED_CLEAN_RUN_ID,
        "source_v_cache_sha256": input_hashes_before["cache"],
        "source_v_cache_receipt_sha256": input_hashes_before["cache_receipt"],
        "pair_v3_sha256": input_hashes_before["pair"],
        "pair_policy_state_sha256": state["state_sha256"],
        "source_tx_ids": list(source_tx_ids),
        "source_validation_role": SOURCE_V_ROLE,
        "single_leo_forward_count": int(snapshot["row_count"]),
        "single_leo_observation": True,
        "single_leo_forward_bound": True,
        "physical_order_sha256": snapshot["receipt"]["physical_order_sha256"],
        "formal_scenarios": list(EXPECTED_SCENARIOS),
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
        "proxy_diagnostic_read_only": {
            "AUROC_unknown": float(proxy_diagnostic["AUROC_unknown"]),
            "u_gap": float(proxy_diagnostic["u_gap"]),
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": load_audit,
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    for name, digest in input_hashes_before.items():
        if _sha256_file(input_paths[name]) != digest:
            raise CLICSourceVFeatureExportError("source-V input changed during model forward")
    output_publication: Any | None = None
    binding_publication: Any | None = None
    try:
        output_publication = _atomic_save_npz(output_path, payload)
        output_sha = output_publication.sha256
        if not isinstance(output_sha, str):
            raise CLICSourceVFeatureExportError("source-V feature publication lacks its pre-publish SHA seal")
        _cache._assert_publication_current(
            output_publication, expected_sha256=output_sha, label="source-V feature export"
        )
        binding = {
            "schema": SOURCE_V_FEATURE_BINDING_SCHEMA,
            "method": "P1_CLIC",
            "candidate_id": candidate,
            "fold_index": fold,
            "arm": arm,
            "source_only": True,
            "post_target_completion_audit_non_selection": True,
            "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
            "checkpoint_sha256": input_hashes_before["checkpoint"],
            "terminal_receipt_sha256": input_hashes_before["terminal"],
            "clean_v4_sha256": input_hashes_before["clean"],
            "source_validation_indices_sha256": clean_binding["validation_indices_sha256"],
            "source_validation_physical_order_sha256": clean_binding["validation_metadata_order_sha256"],
            "source_v_cache_sha256": input_hashes_before["cache"],
            "source_v_cache_receipt_sha256": input_hashes_before["cache_receipt"],
            "pair_v3_sha256": input_hashes_before["pair"],
            "pair_policy_state_sha256": state["state_sha256"],
            "source_tx_ids": list(source_tx_ids),
            "physical_order_sha256": snapshot["receipt"]["physical_order_sha256"],
            "source_v_feature_npz_path": str(output_path),
            "source_v_feature_npz_sha256": output_sha,
            "source_v_feature_manifest_sha256": _canonical_sha256(manifest),
            "single_leo_forward_count": int(snapshot["row_count"]),
            "source_l_forward_rows": 0,
            "proxy_forward_rows": 0,
            "target_access": False,
            "fit_rows": 0,
            "threshold_fit_rows": 0,
            "selection_access": False,
        }
        _cache._assert_publication_current(
            output_publication, expected_sha256=output_sha, label="source-V feature before binding publish"
        )
        binding_publication = _atomic_write_json(binding_path, binding)
        binding_sha = binding_publication.sha256
        if not isinstance(binding_sha, str):
            raise CLICSourceVFeatureExportError("source-V feature binding lacks its pre-publish SHA seal")
        _cache._assert_publication_current(
            binding_publication, expected_sha256=binding_sha, label="source-V feature binding"
        )
        _cache._assert_publication_current(
            output_publication, expected_sha256=output_sha, label="source-V feature after binding publish"
        )
        for name, digest in input_hashes_before.items():
            if _sha256_file(input_paths[name]) != digest:
                raise CLICSourceVFeatureExportError("source-V input changed while sealing output")
    except Exception:
        for publication in (binding_publication, output_publication):
            if publication is not None:
                _cache._unlink_if_owned(publication)
        raise
    _cache._assert_publication_current(
        binding_publication, expected_sha256=binding_sha, label="source-V feature binding before return"
    )
    _cache._assert_publication_current(
        output_publication, expected_sha256=output_sha, label="source-V feature export before return"
    )
    return {"out_npz": str(output_path), "binding_json": str(binding_path), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--terminal-receipt-json", required=True)
    parser.add_argument("--clean-npz", required=True)
    parser.add_argument("--source-v-received-iq-npz", required=True)
    parser.add_argument("--source-v-received-iq-receipt-json", required=True)
    parser.add_argument("--pair-json", required=True)
    parser.add_argument(
        "--formal-project-root",
        help="technical-smoke declaration checked against the frozen canonical N607 project root",
    )
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--cache-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--binding-json", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--technical-smoke",
        action="store_true",
        help="allow only the pre-registered F1 independent technical-smoke root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = export_source_v_leo_features(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLICSourceVFeatureExportError",
    "EXPECTED_SCENARIOS",
    "SOURCE_V_FEATURE_BINDING_SCHEMA",
    "SOURCE_V_FEATURE_SCHEMA",
    "SOURCE_V_ROLE",
    "build_parser",
    "export_source_v_leo_features",
    "main",
    "numpy_float32_to_tensor",
    "read_source_v_cache_snapshot",
    "validate_source_v_clean_v4_binding",
    "validate_source_v_execution_roots",
    "validate_pair_single_leo_common_binding",
    "validate_pair_source_l_policy_binding",
    "validate_source_v_forward_payload",
]
