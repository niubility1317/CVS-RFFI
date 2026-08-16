"""File-only ADV3B02 train-config sealing and blind target prediction.

This entry point intentionally has no truth-sidecar, known-test-config,
reference, metric, adaptation, fitting, retry, or selection input.  A source
train-data configuration is first derived from the final ADV checkpoint, its
same-directory completion receipt, and the matching source-only CLIC clean-v4
authority.  The blind publisher then consumes only that sealed configuration,
the same checkpoint/receipt, and the existing IQ-only package.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import build_phase1_clic_source_v_leo_iq as _clean_v4
import dataset_wisig as _wisig
from cvsrffi import phase1_clic_target_leo as _target
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS, sha256_file
from evaluate_phase1_clic_target_leo import _load_verified_clic_iq_only_package


ADV3B02_TRAIN_CONFIG_SCHEMA = "cvs.phase1.adv3b02_train_data_config.v1"
ADV3B02_PREDICTION_SCHEMA = "cvs.phase1.adv3b02_target_prediction.v1"
_PHYSICAL_AXIS_BINDING_SCHEMA = (
    "cvs.phase1.wisig_source_physical_axis_binding.v1"
)
_ADV_RUN_ID = "phase1_adv3b02_clic6_20260816_v2"
_ADV_BASE_CANDIDATE = "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL"
_COMPLETION_NAME = "phase1_training_completion_receipt.json"
_FINAL_CHECKPOINT_NAME = "final_ssdg.pth"
_BASELINE_TERMINAL_STATUS = "NON_PROMOTABLE_P0_DISABLED"
_BASELINE_EXIT_CODE = 8
_ADV_CANDIDATE_RE = re.compile(r"F([1-6])_ADV3B02_CLIC")
_CLIC_CANDIDATE_RE = re.compile(r"F([1-6])([CG])_CLIC12")
_ROLE_RATIOS = {
    "labeled_ratio": 0.07,
    "unlabeled_ratio": 0.63,
    "source_val_ratio": 0.30,
}


class ADV3B02TargetProtocolError(_target.CLICTargetProtocolError):
    """Raised when an ADV3B02 blind artifact cannot be proven safe."""


def _path(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ADV3B02TargetProtocolError(f"{label} must be a path")
    return Path(value).resolve()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ADV3B02TargetProtocolError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _require_sha(value: Any, *, label: str) -> str:
    try:
        return _target.require_sha256(value, label=label)
    except _target.CLICTargetProtocolError as exc:
        raise ADV3B02TargetProtocolError(str(exc)) from exc


def _canonical_sha(value: Any, *, label: str) -> str:
    try:
        return _target.canonical_sha256(value)
    except _target.CLICTargetProtocolError as exc:
        raise ADV3B02TargetProtocolError(f"{label} cannot be canonicalized") from exc


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return _target.read_json_object(path, label=label)
    except (OSError, _target.CLICTargetProtocolError) as exc:
        raise ADV3B02TargetProtocolError(str(exc)) from exc


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ADV3B02TargetProtocolError(f"{label} must be boolean")
    return bool(value)


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ADV3B02TargetProtocolError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ADV3B02TargetProtocolError(f"{label} is out of range")
    return result


def _require_exact_float(value: Any, *, expected: float, label: str) -> float:
    if isinstance(value, bool):
        raise ADV3B02TargetProtocolError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ADV3B02TargetProtocolError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or not math.isclose(
        result, expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ADV3B02TargetProtocolError(
            f"{label} drifted: {result!r}, expected {expected!r}"
        )
    return expected


def _ids(value: Any, *, label: str, expected_count: int | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ADV3B02TargetProtocolError(f"{label} must be an ordered list")
    result = [str(item) for item in value]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ADV3B02TargetProtocolError(f"{label} must contain unique nonempty IDs")
    if expected_count is not None and len(result) != expected_count:
        raise ADV3B02TargetProtocolError(
            f"{label} must contain exactly {expected_count} IDs"
        )
    return result


def _csv_ids(value: Any, *, label: str, expected_count: int | None = None) -> list[str]:
    if not isinstance(value, str):
        raise ADV3B02TargetProtocolError(f"{label} must be a comma-separated string")
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ADV3B02TargetProtocolError(f"{label} has empty or duplicate IDs")
    if expected_count is not None and len(result) != expected_count:
        raise ADV3B02TargetProtocolError(
            f"{label} must contain exactly {expected_count} IDs"
        )
    return result


def _validate_self_sha(
    payload: Mapping[str, Any], *, sha_field: str, label: str
) -> dict[str, Any]:
    result = _mapping(payload, label=label)
    declared = _require_sha(result.get(sha_field), label=f"{label} {sha_field}")
    base = dict(result)
    base.pop(sha_field, None)
    if _canonical_sha(base, label=label) != declared:
        raise ADV3B02TargetProtocolError(f"{label} {sha_field} drift")
    return result


def _validate_source_split_receipt(
    value: Any, *, wisig_sha: str | None = None
) -> dict[str, Any]:
    receipt = _validate_self_sha(
        _mapping(value, label="ADV source split receipt"),
        sha_field="split_manifest_sha256",
        label="ADV source split receipt",
    )
    if receipt.get("schema") != "cvs.phase1.source_split_receipt.v1":
        raise ADV3B02TargetProtocolError("ADV source split receipt schema drift")
    if _require_int(receipt.get("seed"), label="ADV source split seed") != 392002:
        raise ADV3B02TargetProtocolError("ADV source split seed drift")
    if receipt.get("split_mode") != "tx_rx_day_1_6_3":
        raise ADV3B02TargetProtocolError("ADV source split mode drift")
    receipt_wisig_sha = _require_sha(
        receipt.get("wisig_pkl_sha256"), label="ADV source split WiSig"
    )
    if wisig_sha is not None and receipt_wisig_sha != wisig_sha:
        raise ADV3B02TargetProtocolError("ADV source split WiSig SHA drift")
    source_days = _ids(receipt.get("source_days"), label="ADV source day indices")
    source_receivers = _ids(
        receipt.get("source_receivers"), label="ADV source receiver indices"
    )
    target_days = _ids(receipt.get("target_days"), label="ADV target day indices")
    target_receivers = _ids(
        receipt.get("target_receivers"), label="ADV target receiver indices"
    )
    if set(source_days) & set(target_days) or set(source_receivers) & set(target_receivers):
        raise ADV3B02TargetProtocolError("ADV source/target axis overlap")
    if _require_int(
        receipt.get("source_target_receiver_overlap_count"),
        label="ADV source/target receiver overlap count",
    ) != 0:
        raise ADV3B02TargetProtocolError("ADV source/target receiver overlap drift")
    for field in (
        "labeled_indices_sha256",
        "unlabeled_indices_sha256",
        "source_validation_indices_sha256",
    ):
        _require_sha(receipt.get(field), label=f"ADV source split {field}")
    for field, expected in _ROLE_RATIOS.items():
        _require_exact_float(
            receipt.get(f"requested_{field}"),
            expected=expected,
            label=f"ADV source split requested_{field}",
        )
    labeled = _require_int(receipt.get("labeled_size"), label="ADV labeled size", minimum=1)
    unlabeled = _require_int(
        receipt.get("unlabeled_size"), label="ADV unlabeled size", minimum=1
    )
    validation = _require_int(
        receipt.get("source_validation_size"),
        label="ADV source validation size",
        minimum=1,
    )
    pool = _require_int(receipt.get("source_pool_size"), label="ADV source pool size", minimum=1)
    if labeled + unlabeled + validation != pool:
        raise ADV3B02TargetProtocolError("ADV source split role sizes do not close")
    for field, count, expected in (
        ("requested_labeled_ratio", labeled, _ROLE_RATIOS["labeled_ratio"]),
        ("requested_unlabeled_ratio", unlabeled, _ROLE_RATIOS["unlabeled_ratio"]),
        ("requested_source_val_ratio", validation, _ROLE_RATIOS["source_val_ratio"]),
    ):
        _require_exact_float(receipt.get(field), expected=expected, label=f"ADV source split {field}")
        if not math.isclose(count / pool, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ADV3B02TargetProtocolError(f"ADV source split realized {field} drift")
    if _require_bool(
        receipt.get("realized_rho_within_tolerance"),
        label="ADV source split rho tolerance flag",
    ) is not True or _require_bool(
        receipt.get("realized_source_val_within_tolerance"),
        label="ADV source split validation tolerance flag",
    ) is not True:
        raise ADV3B02TargetProtocolError("ADV source split tolerance receipt drift")
    return receipt


def _validate_tx_partition(value: Any) -> dict[str, Any]:
    receipt = _validate_self_sha(
        _mapping(value, label="ADV TX partition receipt"),
        sha_field="partition_sha256",
        label="ADV TX partition receipt",
    )
    if receipt.get("schema") != "cvs.phase1.tx_partition_receipt.v1":
        raise ADV3B02TargetProtocolError("ADV TX partition schema drift")
    if _require_bool(receipt.get("enabled"), label="ADV TX partition enabled") is not True:
        raise ADV3B02TargetProtocolError("ADV TX partition must be enabled")
    train = _ids(
        receipt.get("source_known_train_tx"),
        label="ADV source local class order",
        expected_count=4,
    )
    held = _ids(
        receipt.get("source_known_validation_tx"),
        label="ADV held validation TX",
        expected_count=1,
    )
    proxy = _ids(
        receipt.get("source_proxy_unknown_tx"),
        label="ADV proxy TX",
        expected_count=1,
    )
    if set(train) & set(held) or set(train) & set(proxy) or set(held) & set(proxy):
        raise ADV3B02TargetProtocolError("ADV TX role overlap")
    expected_order = [*train, *held, *proxy]
    if _ids(receipt.get("dataset_tx_order"), label="ADV dataset TX order") != expected_order:
        raise ADV3B02TargetProtocolError("ADV dataset TX order drift")
    if _require_int(receipt.get("dataset_tx_count"), label="ADV dataset TX count") != len(expected_order):
        raise ADV3B02TargetProtocolError("ADV dataset TX count drift")
    if _require_int(receipt.get("training_tx_count"), label="ADV training TX count") != len(train):
        raise ADV3B02TargetProtocolError("ADV training TX count drift")
    if _require_bool(
        receipt.get("allow_empty_proxy_unknown"), label="ADV empty proxy flag"
    ) is not False or _require_bool(
        receipt.get("held_tx_loaded_by_training"), label="ADV held TX loading flag"
    ) is not False:
        raise ADV3B02TargetProtocolError("ADV TX partition role flag drift")
    contiguous = _mapping(
        receipt.get("training_view_contiguous_reindex"),
        label="ADV contiguous local class order",
    )
    expected_contiguous = {str(index): tx_id for index, tx_id in enumerate(train)}
    if contiguous != expected_contiguous:
        raise ADV3B02TargetProtocolError("ADV contiguous local class order drift")
    return receipt


def _checkpoint_path_binding(path: Path) -> tuple[int, str]:
    if path.name != _FINAL_CHECKPOINT_NAME:
        raise ADV3B02TargetProtocolError("ADV checkpoint must be final_ssdg.pth")
    match = _ADV_CANDIDATE_RE.fullmatch(path.parent.name)
    if match is None or path.parent.parent.name != _ADV_RUN_ID:
        raise ADV3B02TargetProtocolError("ADV checkpoint run/candidate binding drift")
    return int(match.group(1)), path.parent.name


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"ADV checkpoint is missing: {path}")
    initial_sha = sha256_file(path)
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ADV3B02TargetProtocolError("ADV checkpoint cannot be reopened") from exc
    if sha256_file(path) != initial_sha:
        raise ADV3B02TargetProtocolError("ADV checkpoint changed during reopen")
    return _mapping(payload, label="ADV checkpoint"), initial_sha


def _validate_adv_checkpoint(path: Path) -> dict[str, Any]:
    fold_index, candidate_id = _checkpoint_path_binding(path)
    checkpoint, checkpoint_sha = _load_checkpoint(path)
    if checkpoint.get("checkpoint_schema") != "ssdg_phase1_training_state_v2":
        raise ADV3B02TargetProtocolError("ADV checkpoint schema drift")
    if checkpoint.get("checkpoint_role") != "training_final_only" or checkpoint.get(
        "checkpoint_selection"
    ) != "final_only":
        raise ADV3B02TargetProtocolError("ADV checkpoint final-only role drift")
    if checkpoint.get("run_id") != _ADV_RUN_ID or checkpoint.get("candidate_id") != candidate_id:
        raise ADV3B02TargetProtocolError("ADV checkpoint run/candidate identity drift")
    if _require_int(checkpoint.get("final_epoch"), label="ADV checkpoint final epoch", minimum=1) != 200:
        raise ADV3B02TargetProtocolError("ADV checkpoint final epoch drift")
    args = _mapping(checkpoint.get("args"), label="ADV checkpoint args")
    if args.get("run_id") != _ADV_RUN_ID or args.get("candidate_id") != candidate_id:
        raise ADV3B02TargetProtocolError("ADV checkpoint args identity drift")
    if args.get("base_candidate") != _ADV_BASE_CANDIDATE:
        raise ADV3B02TargetProtocolError("ADV checkpoint base candidate drift")
    if str(args.get("dataset", "")).lower() != "wisig":
        raise ADV3B02TargetProtocolError("ADV checkpoint dataset drift")
    wisig_path_value = args.get("wisig_pkl")
    if not isinstance(wisig_path_value, (str, Path)) or not str(
        wisig_path_value
    ).strip():
        raise ADV3B02TargetProtocolError("ADV checkpoint WiSig path is missing")
    wisig_path = _path(wisig_path_value, label="ADV checkpoint WiSig dataset")
    if args.get("split_mode") != "tx_rx_day_1_6_3":
        raise ADV3B02TargetProtocolError("ADV checkpoint split mode drift")
    if _require_int(args.get("seed"), label="ADV checkpoint seed") != 392002:
        raise ADV3B02TargetProtocolError("ADV checkpoint seed drift")
    if args.get("checkpoint_selection") != "final_only":
        raise ADV3B02TargetProtocolError("ADV checkpoint selection drift")
    for field, expected in _ROLE_RATIOS.items():
        _require_exact_float(args.get(field), expected=expected, label=f"ADV checkpoint {field}")
    input_len = _require_int(args.get("wisig_out_len"), label="ADV checkpoint input length", minimum=1)
    split_info = _mapping(checkpoint.get("split_info"), label="ADV checkpoint split_info")
    if split_info.get("mode") != "tx_rx_day_1_6_3":
        raise ADV3B02TargetProtocolError("ADV checkpoint split_info mode drift")
    source_split = _validate_source_split_receipt(
        split_info.get("source_split_receipt")
    )
    wisig_sha = source_split["wisig_pkl_sha256"]
    args_wisig_sha = args.get("wisig_pkl_sha256")
    if args_wisig_sha not in (None, "") and _require_sha(
        args_wisig_sha, label="ADV checkpoint WiSig"
    ) != wisig_sha:
        raise ADV3B02TargetProtocolError("ADV checkpoint WiSig SHA drift")
    partition = _validate_tx_partition(split_info.get("tx_partition_receipt"))
    local_order = _ids(
        split_info.get("class_id_to_tx"),
        label="ADV checkpoint local class order",
        expected_count=4,
    )
    if local_order != partition["source_known_train_tx"]:
        raise ADV3B02TargetProtocolError("ADV checkpoint local class order drift")
    if _csv_ids(
        args.get("phase1_source_train_tx_ids"),
        label="ADV checkpoint source train TX IDs",
        expected_count=4,
    ) != local_order:
        raise ADV3B02TargetProtocolError("ADV checkpoint source train TX drift")
    if _csv_ids(
        args.get("phase1_source_known_validation_tx_ids"),
        label="ADV checkpoint held TX ID",
        expected_count=1,
    ) != partition["source_known_validation_tx"]:
        raise ADV3B02TargetProtocolError("ADV checkpoint held TX drift")
    if _csv_ids(
        args.get("phase1_source_proxy_unknown_tx_ids"),
        label="ADV checkpoint proxy TX ID",
        expected_count=1,
    ) != partition["source_proxy_unknown_tx"]:
        raise ADV3B02TargetProtocolError("ADV checkpoint proxy TX drift")
    if _csv_ids(args.get("wisig_train_rxs"), label="ADV checkpoint receiver indices") != source_split[
        "source_receivers"
    ]:
        raise ADV3B02TargetProtocolError("ADV checkpoint receiver index axis drift")
    if _csv_ids(args.get("wisig_train_days"), label="ADV checkpoint day indices") != source_split[
        "source_days"
    ]:
        raise ADV3B02TargetProtocolError("ADV checkpoint day index axis drift")
    scenes = _csv_ids(
        args.get("sat_train_scenarios"), label="ADV checkpoint training LEO scenes"
    )
    if tuple(scenes) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ADV3B02TargetProtocolError("ADV checkpoint training LEO scene drift")
    eval_scenes = _csv_ids(
        args.get("eval_sat_scenarios"), label="ADV checkpoint evaluation LEO scenes"
    )
    if eval_scenes != scenes:
        raise ADV3B02TargetProtocolError("ADV checkpoint evaluation LEO scene drift")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ADV3B02TargetProtocolError("ADV checkpoint model state is missing")
    return {
        "path": path,
        "sha256": checkpoint_sha,
        "payload": checkpoint,
        "fold_index": fold_index,
        "candidate_id": candidate_id,
        "args": args,
        "input_len": input_len,
        "wisig_path": wisig_path,
        "wisig_sha256": wisig_sha,
        "source_split": source_split,
        "partition": partition,
        "source_class_order": local_order,
        "source_class_order_sha256": _canonical_sha(
            local_order, label="ADV source class order"
        ),
        "scenes": scenes,
    }


def _read_completion(path: Path, *, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    expected = Path(str(checkpoint["path"])).parent / _COMPLETION_NAME
    if path != expected:
        raise ADV3B02TargetProtocolError(
            "ADV completion receipt must be the checkpoint's same-directory receipt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"ADV completion receipt is missing: {path}")
    receipt_sha_before = sha256_file(path)
    payload = _read_json(path, label="ADV completion receipt")
    receipt_sha_after = sha256_file(path)
    if receipt_sha_after != receipt_sha_before:
        raise ADV3B02TargetProtocolError("ADV completion receipt changed during reopen")
    receipt_sha = receipt_sha_before
    if payload.get("schema") != "cvs.phase1.training_completion_receipt.v1":
        raise ADV3B02TargetProtocolError("ADV completion receipt schema drift")
    if payload.get("run_id") != _ADV_RUN_ID:
        raise ADV3B02TargetProtocolError("ADV completion receipt run identity drift")
    if "candidate_id" in payload and payload.get("candidate_id") != checkpoint[
        "candidate_id"
    ]:
        raise ADV3B02TargetProtocolError(
            "ADV completion receipt candidate_id drift"
        )
    completion_wisig_sha = payload.get("wisig_pkl_sha256")
    if completion_wisig_sha not in (None, "") and _require_sha(
        completion_wisig_sha, label="ADV completion receipt WiSig"
    ) != checkpoint["wisig_sha256"]:
        raise ADV3B02TargetProtocolError(
            "ADV completion receipt wisig_pkl_sha256 drift"
        )
    baseline_fields = (
        "terminal_status",
        "exit_code",
        "phase1_training_complete",
        "technical_only",
        "formal_performance_claim",
        "claim",
    )
    missing_baseline_fields = [
        field for field in baseline_fields if field not in payload
    ]
    if missing_baseline_fields:
        raise ADV3B02TargetProtocolError(
            "ADV completion baseline terminal tuple is missing fields: "
            f"{missing_baseline_fields}"
        )
    if (
        type(payload["terminal_status"]) is not str
        or payload["terminal_status"] != _BASELINE_TERMINAL_STATUS
    ):
        raise ADV3B02TargetProtocolError(
            "ADV completion baseline terminal status drift"
        )
    if _require_int(
        payload["exit_code"], label="ADV completion baseline exit code"
    ) != _BASELINE_EXIT_CODE:
        raise ADV3B02TargetProtocolError(
            "ADV completion baseline exit code drift"
        )
    for field in (
        "phase1_training_complete",
        "technical_only",
        "formal_performance_claim",
    ):
        if _require_bool(
            payload[field], label=f"ADV completion baseline {field}"
        ) is not False:
            raise ADV3B02TargetProtocolError(
                f"ADV completion baseline {field} drift"
            )
    if (
        type(payload["claim"]) is not str
        or payload["claim"] != "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT"
    ):
        raise ADV3B02TargetProtocolError("ADV completion baseline claim drift")
    if _require_sha(
        payload.get("selected_checkpoint_sha256"),
        label="ADV completion selected checkpoint",
    ) != checkpoint["sha256"]:
        raise ADV3B02TargetProtocolError("ADV completion selected checkpoint SHA drift")
    completion_split = _validate_source_split_receipt(
        payload.get("source_split_receipt"), wisig_sha=checkpoint["wisig_sha256"]
    )
    if completion_split != checkpoint["source_split"]:
        raise ADV3B02TargetProtocolError("ADV completion source split receipt drift")
    if sha256_file(path) != receipt_sha:
        raise ADV3B02TargetProtocolError("ADV completion receipt changed during validation")
    return {
        "path": path,
        "sha256": receipt_sha,
        "payload": payload,
        "baseline_terminal_status": _BASELINE_TERMINAL_STATUS,
        "baseline_exit_code": _BASELINE_EXIT_CODE,
        "baseline_promotion_ready": False,
        "formal_performance_claim": False,
    }


def _read_clean_manifest_header(path: Path) -> dict[str, Any]:
    """Read only the clean-v4 manifest member before the shared metadata reopener."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "manifest_json" not in archive.files:
                raise ADV3B02TargetProtocolError("clean-v4 manifest member is missing")
            rendered = np.asarray(archive["manifest_json"])
    except (OSError, ValueError) as exc:
        raise ADV3B02TargetProtocolError("clean-v4 manifest cannot be opened") from exc
    if rendered.size != 1:
        raise ADV3B02TargetProtocolError("clean-v4 manifest must contain one value")
    try:
        payload = json.loads(str(rendered.reshape(-1)[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ADV3B02TargetProtocolError("clean-v4 manifest JSON is invalid") from exc
    return _mapping(payload, label="clean-v4 manifest")


def _read_verified_clean_v4_metadata_only(
    *,
    path: Path,
    arm: str,
    fold_index: int,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Reuse the shared clean-v4 metadata authority without reading clean features."""

    if not path.is_file():
        raise FileNotFoundError(f"clean-v4 authority is missing: {path}")
    initial_sha = sha256_file(path)
    manifest = _read_clean_manifest_header(path)
    checkpoint_sha = _require_sha(
        manifest.get("source_checkpoint_sha256"),
        label="clean-v4 source checkpoint",
    )
    terminal_sha = _require_sha(
        manifest.get("terminal_receipt_sha256"),
        label="clean-v4 terminal receipt",
    )
    try:
        binding = _clean_v4._read_clean_validation_binding(
            path=path,
            arm=arm,
            fold_index=fold_index,
            source_tx_ids=tuple(str(item) for item in source_tx_ids),
            checkpoint_sha256=checkpoint_sha,
            terminal_sha256=terminal_sha,
        )
    except _clean_v4.CLICSourceVLeoCacheError as exc:
        raise ADV3B02TargetProtocolError(
            f"clean-v4 metadata authority reopening failed: {exc}"
        ) from exc
    if sha256_file(path) != initial_sha:
        raise ADV3B02TargetProtocolError("clean-v4 authority changed during reopen")
    if binding.get("sha256") != initial_sha:
        raise ADV3B02TargetProtocolError("clean-v4 authority SHA binding drift")
    binding_manifest = _mapping(binding.get("manifest"), label="clean-v4 reopened manifest")
    if binding_manifest != manifest:
        raise ADV3B02TargetProtocolError("clean-v4 manifest changed during metadata reopen")
    return {**binding, "sha256": initial_sha, "manifest": manifest}


def _clean_path_binding(path: Path) -> tuple[int, str]:
    match = _CLIC_CANDIDATE_RE.fullmatch(path.parent.name)
    if match is None:
        raise ADV3B02TargetProtocolError("clean-v4 candidate binding drift")
    if path.name != "source_clean_proxy.npz" or path.parent.parent.name != _clean_v4.EXPECTED_CLEAN_RUN_ID:
        raise ADV3B02TargetProtocolError("clean-v4 run/output binding drift")
    return int(match.group(1)), match.group(2)


def _validate_clean_against_adv(
    path: Path, *, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    fold_index, arm = _clean_path_binding(path)
    if fold_index != checkpoint["fold_index"]:
        raise ADV3B02TargetProtocolError("clean-v4 fold does not match ADV checkpoint")
    binding = _read_verified_clean_v4_metadata_only(
        path=path,
        arm=arm,
        fold_index=fold_index,
        source_tx_ids=checkpoint["source_class_order"],
    )
    manifest = _mapping(binding["manifest"], label="clean-v4 manifest")
    if manifest.get("candidate_id") != f"F{fold_index}{arm}_CLIC12":
        raise ADV3B02TargetProtocolError("clean-v4 candidate identity drift")
    if manifest.get("wisig_pkl_sha256") != checkpoint["wisig_sha256"]:
        raise ADV3B02TargetProtocolError("clean-v4 WiSig SHA drift")
    if _ids(manifest.get("source_tx_ids"), label="clean-v4 source TX IDs", expected_count=4) != checkpoint[
        "source_class_order"
    ]:
        raise ADV3B02TargetProtocolError("clean-v4 source local class order drift")
    clean_split = _validate_source_split_receipt(
        manifest.get("source_split_receipt"), wisig_sha=checkpoint["wisig_sha256"]
    )
    if clean_split != checkpoint["source_split"]:
        raise ADV3B02TargetProtocolError("clean-v4 source split receipt drift")
    if _require_sha(
        manifest.get("source_split_receipt_sha256"),
        label="clean-v4 source split receipt",
    ) != _canonical_sha(clean_split, label="clean-v4 source split receipt"):
        raise ADV3B02TargetProtocolError("clean-v4 source split receipt SHA drift")
    clean_partition = _validate_tx_partition(manifest.get("tx_partition_receipt"))
    if clean_partition != checkpoint["partition"]:
        raise ADV3B02TargetProtocolError("clean-v4 TX partition receipt drift")
    if _require_sha(
        manifest.get("tx_partition_receipt_sha256"),
        label="clean-v4 TX partition receipt",
    ) != _canonical_sha(clean_partition, label="clean-v4 TX partition receipt"):
        raise ADV3B02TargetProtocolError("clean-v4 TX partition receipt SHA drift")
    if _require_sha(
        manifest.get("source_validation_indices_sha256"),
        label="clean-v4 source validation indices",
    ) != checkpoint["source_split"]["source_validation_indices_sha256"]:
        raise ADV3B02TargetProtocolError("clean-v4 validation index receipt drift")
    receiver_ids = _ids(
        manifest.get("source_receiver_ids"), label="clean-v4 source receiver IDs"
    )
    day_ids = _ids(manifest.get("source_day_ids"), label="clean-v4 source day IDs")
    observed_receivers = sorted(set(str(value) for value in binding["validation_rx_ids"]))
    observed_days = sorted(set(str(value) for value in binding["validation_day_ids"]))
    if receiver_ids != observed_receivers or day_ids != observed_days:
        raise ADV3B02TargetProtocolError("clean-v4 physical source axis drift")
    if len(receiver_ids) != len(checkpoint["source_split"]["source_receivers"]):
        raise ADV3B02TargetProtocolError("clean-v4 source receiver index axis drift")
    if len(day_ids) != len(checkpoint["source_split"]["source_days"]):
        raise ADV3B02TargetProtocolError("clean-v4 source day index axis drift")
    if _require_sha(
        manifest.get("source_receiver_ids_sha256"), label="clean-v4 source receiver IDs"
    ) != _canonical_sha(receiver_ids, label="clean-v4 source receiver IDs"):
        raise ADV3B02TargetProtocolError("clean-v4 source receiver axis SHA drift")
    if _require_sha(
        manifest.get("source_day_ids_sha256"), label="clean-v4 source day IDs"
    ) != _canonical_sha(day_ids, label="clean-v4 source day IDs"):
        raise ADV3B02TargetProtocolError("clean-v4 source day axis SHA drift")
    validation_count = len(binding["validation_tx_ids"])
    if validation_count != checkpoint["source_split"]["source_validation_size"]:
        raise ADV3B02TargetProtocolError("clean-v4 validation row count drift")
    if _require_int(
        manifest.get("source_validation_row_count"),
        label="clean-v4 validation row count",
        minimum=1,
    ) != validation_count:
        raise ADV3B02TargetProtocolError("clean-v4 manifest validation row count drift")
    if _require_sha(
        binding.get("validation_indices_sha256"), label="clean-v4 validation indices"
    ) != checkpoint["source_split"]["source_validation_indices_sha256"]:
        raise ADV3B02TargetProtocolError("clean-v4 reopened validation index SHA drift")
    return {
        "path": path,
        "sha256": binding["sha256"],
        "manifest_sha256": _canonical_sha(manifest, label="clean-v4 manifest"),
        "manifest": manifest,
        "fold_index": fold_index,
        "arm": arm,
        "source_receiver_ids": receiver_ids,
        "source_day_ids": day_ids,
    }


def _resolve_wisig_source_axis(
    dataset_axis: Any,
    source_indices: Any,
    *,
    label: str,
) -> tuple[list[str], list[str]]:
    if not isinstance(dataset_axis, (list, tuple)) or any(
        item is None for item in dataset_axis
    ):
        raise ADV3B02TargetProtocolError(
            f"WiSig {label} physical axis must be an explicit ordered list"
        )
    physical_axis = _ids(dataset_axis, label=f"WiSig {label} physical axis")
    index_axis = _ids(source_indices, label=f"ADV source {label} index axis")
    resolved: list[str] = []
    for token in index_axis:
        if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
            raise ADV3B02TargetProtocolError(
                f"ADV source {label} index axis is not canonical"
            )
        index = int(token)
        if index >= len(physical_axis):
            raise ADV3B02TargetProtocolError(
                f"ADV source {label} index is outside the WiSig physical axis"
            )
        resolved.append(physical_axis[index])
    if len(set(resolved)) != len(resolved):
        raise ADV3B02TargetProtocolError(
            f"ADV source {label} physical mapping is not one-to-one"
        )
    return index_axis, resolved


def _build_physical_axis_binding(
    *,
    wisig_path: Path,
    wisig_sha256: str,
    receiver_indices: Sequence[str],
    receiver_ids: Sequence[str],
    day_indices: Sequence[str],
    day_ids: Sequence[str],
) -> dict[str, Any]:
    receiver_index_axis = _ids(
        receiver_indices, label="ADV source receiver index axis"
    )
    receiver_physical_axis = _ids(
        receiver_ids, label="ADV source receiver physical axis"
    )
    day_index_axis = _ids(day_indices, label="ADV source day index axis")
    day_physical_axis = _ids(day_ids, label="ADV source day physical axis")
    if len(receiver_index_axis) != len(receiver_physical_axis) or len(
        day_index_axis
    ) != len(day_physical_axis):
        raise ADV3B02TargetProtocolError(
            "ADV source index-to-physical axis cardinality drift"
        )
    receiver_mapping = [
        {"index": index, "physical_id": physical_id}
        for index, physical_id in zip(
            receiver_index_axis, receiver_physical_axis, strict=True
        )
    ]
    day_mapping = [
        {"index": index, "physical_id": physical_id}
        for index, physical_id in zip(
            day_index_axis, day_physical_axis, strict=True
        )
    ]
    return {
        "schema": _PHYSICAL_AXIS_BINDING_SCHEMA,
        "wisig_pkl_path": str(wisig_path),
        "wisig_pkl_sha256": _require_sha(
            wisig_sha256, label="WiSig physical-axis dataset"
        ),
        "source_receiver_indices": receiver_index_axis,
        "source_receiver_indices_sha256": _canonical_sha(
            receiver_index_axis, label="ADV source receiver indices"
        ),
        "source_receiver_ids": receiver_physical_axis,
        "source_receiver_ids_sha256": _canonical_sha(
            receiver_physical_axis, label="ADV source receiver IDs"
        ),
        "source_receiver_index_to_physical": receiver_mapping,
        "source_receiver_index_to_physical_sha256": _canonical_sha(
            receiver_mapping, label="ADV source receiver index-to-physical binding"
        ),
        "source_day_indices": day_index_axis,
        "source_day_indices_sha256": _canonical_sha(
            day_index_axis, label="ADV source day indices"
        ),
        "source_day_ids": day_physical_axis,
        "source_day_ids_sha256": _canonical_sha(
            day_physical_axis, label="ADV source day IDs"
        ),
        "source_day_index_to_physical": day_mapping,
        "source_day_index_to_physical_sha256": _canonical_sha(
            day_mapping, label="ADV source day index-to-physical binding"
        ),
    }


def _read_verified_wisig_physical_axes(
    *, checkpoint: Mapping[str, Any], clean: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen the checkpoint-bound source dataset only while sealing config."""

    path = Path(str(checkpoint["wisig_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint-bound WiSig dataset is missing: {path}")
    initial_sha = sha256_file(path)
    if initial_sha != checkpoint["wisig_sha256"]:
        raise ADV3B02TargetProtocolError(
            "checkpoint-bound WiSig dataset raw SHA drift"
        )
    clean_manifest = _mapping(clean.get("manifest"), label="clean-v4 manifest")
    if _require_sha(
        clean_manifest.get("wisig_pkl_sha256"), label="clean-v4 WiSig dataset"
    ) != initial_sha:
        raise ADV3B02TargetProtocolError(
            "checkpoint-bound WiSig dataset does not match clean-v4"
        )
    try:
        dataset = _wisig.load_wisig_compact_pkl(str(path))
    except Exception as exc:
        raise ADV3B02TargetProtocolError(
            "checkpoint-bound WiSig dataset cannot be reopened"
        ) from exc
    if sha256_file(path) != initial_sha:
        raise ADV3B02TargetProtocolError(
            "checkpoint-bound WiSig dataset changed during metadata reopen"
        )
    receiver_indices, receiver_ids = _resolve_wisig_source_axis(
        dataset.get("rx_list"),
        checkpoint["source_split"]["source_receivers"],
        label="receiver",
    )
    day_indices, day_ids = _resolve_wisig_source_axis(
        dataset.get("capture_date_list"),
        checkpoint["source_split"]["source_days"],
        label="day",
    )
    if receiver_ids != list(clean["source_receiver_ids"]):
        raise ADV3B02TargetProtocolError(
            "WiSig source receiver index-to-physical mapping drift"
        )
    if day_ids != list(clean["source_day_ids"]):
        raise ADV3B02TargetProtocolError(
            "WiSig source day index-to-physical mapping drift"
        )
    binding = _build_physical_axis_binding(
        wisig_path=path,
        wisig_sha256=initial_sha,
        receiver_indices=receiver_indices,
        receiver_ids=receiver_ids,
        day_indices=day_indices,
        day_ids=day_ids,
    )
    return {
        "path": path,
        "sha256": initial_sha,
        "source_receiver_ids": receiver_ids,
        "source_day_ids": day_ids,
        "binding": binding,
        "binding_sha256": _canonical_sha(
            binding, label="ADV source physical-axis binding"
        ),
    }


def _validate_sealed_physical_axis_binding(
    *,
    value: Any,
    declared_sha256: Any,
    checkpoint: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every sealed field without reopening source authorities."""

    binding = _mapping(value, label="ADV train-data physical-axis binding")
    expected = _build_physical_axis_binding(
        wisig_path=Path(str(checkpoint["wisig_path"])),
        wisig_sha256=checkpoint["wisig_sha256"],
        receiver_indices=checkpoint["source_split"]["source_receivers"],
        receiver_ids=normalized.get("source_receiver_ids"),
        day_indices=checkpoint["source_split"]["source_days"],
        day_ids=normalized.get("source_day_ids"),
    )
    if binding != expected:
        raise ADV3B02TargetProtocolError(
            "ADV train-data physical-axis binding field drift"
        )
    if _require_sha(
        declared_sha256, label="ADV train-data physical-axis binding"
    ) != _canonical_sha(binding, label="ADV train-data physical-axis binding"):
        raise ADV3B02TargetProtocolError(
            "ADV train-data physical-axis binding SHA drift"
        )
    return binding


def _verify_unchanged(inputs: Mapping[str, tuple[Path, str]]) -> None:
    for label, (path, expected_sha) in inputs.items():
        if sha256_file(path) != expected_sha:
            raise ADV3B02TargetProtocolError(f"{label} changed during validation or forward")


def _write_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    if path.exists():
        raise ADV3B02TargetProtocolError(f"{label} already exists and is immutable: {path}")
    if not path.parent.is_dir():
        raise ADV3B02TargetProtocolError(f"{label} parent directory is missing: {path.parent}")
    encoded = _target.canonical_json_bytes(dict(payload)) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
    except FileExistsError as exc:
        raise ADV3B02TargetProtocolError(
            f"{label} already exists and is immutable: {path}"
        ) from exc
    return path


def seal_adv3b02_train_data_config(
    *,
    checkpoint_path: str | Path,
    completion_receipt_path: str | Path,
    clean_v4_npz_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Seal one file-only ADV train-data configuration without target access."""

    checkpoint_file = _path(checkpoint_path, label="ADV checkpoint")
    completion_file = _path(completion_receipt_path, label="ADV completion receipt")
    clean_file = _path(clean_v4_npz_path, label="clean-v4 authority")
    output_file = _path(output_path, label="ADV train-data config output")
    if output_file.exists():
        raise ADV3B02TargetProtocolError(
            f"ADV train-data config output already exists and is immutable: {output_file}"
        )
    checkpoint = _validate_adv_checkpoint(checkpoint_file)
    completion = _read_completion(completion_file, checkpoint=checkpoint)
    clean = _validate_clean_against_adv(clean_file, checkpoint=checkpoint)
    wisig = _read_verified_wisig_physical_axes(
        checkpoint=checkpoint, clean=clean
    )
    input_hashes = {
        "checkpoint": (checkpoint_file, checkpoint["sha256"]),
        "completion_receipt": (completion_file, completion["sha256"]),
        "clean_v4": (clean_file, clean["sha256"]),
        "checkpoint-bound WiSig dataset": (wisig["path"], wisig["sha256"]),
    }
    _verify_unchanged(input_hashes)
    normalized = _target.normalize_train_data_config(
        {
            "dataset_provenance": {
                "dataset_schema": "WiSig",
                "wisig_pkl_sha256": checkpoint["wisig_sha256"],
            },
            "source_train_tx_ids": list(checkpoint["source_class_order"]),
            "source_validation_tx_ids": list(
                checkpoint["partition"]["source_known_validation_tx"]
            ),
            "source_proxy_tx_ids": list(
                checkpoint["partition"]["source_proxy_unknown_tx"]
            ),
            "source_receiver_ids": list(wisig["source_receiver_ids"]),
            "source_day_ids": list(wisig["source_day_ids"]),
            "split_mode": "tx_rx_day_1_6_3",
            "role_construction": {
                "split_mode": "tx_rx_day_1_6_3",
                **_ROLE_RATIOS,
            },
            "physical_row_selection": {
                "selection_policy": "pre_registered_tx_rx_day_eq_split_by_sig_i",
                "group_axes": ["tx_id", "rx_id", "day_id", "eq_id"],
            },
            "preprocessing": {
                "input_len": checkpoint["input_len"],
                "iq_dtype": "float32",
            },
            "single_leo_training_scenes": list(checkpoint["scenes"]),
        }
    )
    normalized_sha = _canonical_sha(normalized, label="ADV normalized train-data config")
    payload = {
        "schema": ADV3B02_TRAIN_CONFIG_SCHEMA,
        "immutable": True,
        "checkpoint_sha256": checkpoint["sha256"],
        "completion_receipt_sha256": completion["sha256"],
        "clean_v4_npz_sha256": clean["sha256"],
        "clean_v4_manifest_sha256": clean["manifest_sha256"],
        "wisig_pkl_path": str(wisig["path"]),
        "wisig_pkl_sha256": wisig["sha256"],
        "physical_axis_binding": wisig["binding"],
        "physical_axis_binding_sha256": wisig["binding_sha256"],
        "fold_index": checkpoint["fold_index"],
        "clic_clean_arm": clean["arm"],
        "source_class_order": list(checkpoint["source_class_order"]),
        "source_class_order_sha256": checkpoint["source_class_order_sha256"],
        "baseline_terminal_status": completion["baseline_terminal_status"],
        "baseline_exit_code": completion["baseline_exit_code"],
        "baseline_promotion_ready": completion["baseline_promotion_ready"],
        "formal_performance_claim": completion["formal_performance_claim"],
        "source_split_receipt_sha256": _canonical_sha(
            checkpoint["source_split"], label="ADV source split receipt"
        ),
        "tx_partition_receipt_sha256": _canonical_sha(
            checkpoint["partition"], label="ADV TX partition receipt"
        ),
        "normalized": normalized,
        "normalized_sha256": normalized_sha,
    }
    _verify_unchanged(input_hashes)
    return _write_immutable_json(
        output_file, payload, label="ADV train-data config output"
    ).resolve()


def _read_verified_train_config(
    path: Path,
    *,
    checkpoint: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ADV train-data config is missing: {path}")
    raw_sha = sha256_file(path)
    payload = _read_json(path, label="ADV train-data config")
    if sha256_file(path) != raw_sha:
        raise ADV3B02TargetProtocolError("ADV train-data config changed during reopen")
    if payload.get("schema") != ADV3B02_TRAIN_CONFIG_SCHEMA:
        raise ADV3B02TargetProtocolError("ADV train-data config schema drift")
    if _require_bool(payload.get("immutable"), label="ADV train-data config immutable") is not True:
        raise ADV3B02TargetProtocolError("ADV train-data config is not immutable")
    required_baseline = {
        "baseline_terminal_status": _BASELINE_TERMINAL_STATUS,
        "baseline_exit_code": _BASELINE_EXIT_CODE,
        "baseline_promotion_ready": False,
        "formal_performance_claim": False,
    }
    missing_baseline = [
        field for field in required_baseline if field not in payload
    ]
    if missing_baseline:
        raise ADV3B02TargetProtocolError(
            f"ADV train-data config baseline binding is missing fields: {missing_baseline}"
        )
    for field, expected_value in required_baseline.items():
        observed = payload[field]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ADV3B02TargetProtocolError(
                f"ADV train-data config baseline {field} drift"
            )
    if _require_sha(
        payload.get("checkpoint_sha256"), label="ADV train-data config checkpoint"
    ) != checkpoint["sha256"]:
        raise ADV3B02TargetProtocolError("ADV train-data config checkpoint SHA drift")
    if _require_sha(
        payload.get("completion_receipt_sha256"),
        label="ADV train-data config completion receipt",
    ) != completion["sha256"]:
        raise ADV3B02TargetProtocolError("ADV train-data config completion SHA drift")
    if (
        type(payload.get("wisig_pkl_path")) is not str
        or payload.get("wisig_pkl_path") != str(checkpoint["wisig_path"])
    ):
        raise ADV3B02TargetProtocolError(
            "ADV train-data config checkpoint-bound WiSig path drift"
        )
    if _require_sha(
        payload.get("wisig_pkl_sha256"), label="ADV train-data config WiSig dataset"
    ) != checkpoint["wisig_sha256"]:
        raise ADV3B02TargetProtocolError(
            "ADV train-data config checkpoint-bound WiSig SHA drift"
        )
    if _require_int(payload.get("fold_index"), label="ADV train-data config fold") != checkpoint[
        "fold_index"
    ]:
        raise ADV3B02TargetProtocolError("ADV train-data config fold drift")
    source_order = _ids(
        payload.get("source_class_order"),
        label="ADV train-data config source class order",
        expected_count=4,
    )
    if source_order != checkpoint["source_class_order"]:
        raise ADV3B02TargetProtocolError("ADV train-data config source class order drift")
    if _require_sha(
        payload.get("source_class_order_sha256"),
        label="ADV train-data config source class order",
    ) != checkpoint["source_class_order_sha256"]:
        raise ADV3B02TargetProtocolError("ADV train-data config source class order SHA drift")
    normalized = _mapping(payload.get("normalized"), label="ADV train-data config normalized")
    normalized_sha = _require_sha(
        payload.get("normalized_sha256"), label="ADV train-data config normalized"
    )
    if _canonical_sha(normalized, label="ADV train-data config normalized") != normalized_sha:
        raise ADV3B02TargetProtocolError("ADV train-data config normalized SHA drift")
    try:
        normalized = _target.normalize_train_data_config(normalized)
    except _target.CLICTargetProtocolError as exc:
        raise ADV3B02TargetProtocolError(
            f"ADV train-data config normalized contract drift: {exc}"
        ) from exc
    dataset_provenance = _mapping(
        normalized.get("dataset_provenance"),
        label="ADV train-data config dataset provenance",
    )
    if _require_sha(
        dataset_provenance.get("wisig_pkl_sha256"),
        label="ADV train-data config normalized WiSig dataset",
    ) != checkpoint["wisig_sha256"]:
        raise ADV3B02TargetProtocolError(
            "ADV train-data config normalized WiSig SHA drift"
        )
    physical_axis_binding = _validate_sealed_physical_axis_binding(
        value=payload.get("physical_axis_binding"),
        declared_sha256=payload.get("physical_axis_binding_sha256"),
        checkpoint=checkpoint,
        normalized=normalized,
    )
    if normalized["source_train_tx_ids"] != source_order:
        raise ADV3B02TargetProtocolError("ADV train-data config normalized class order drift")
    if normalized["preprocessing"].get("input_len") != checkpoint["input_len"]:
        raise ADV3B02TargetProtocolError("ADV train-data config input length drift")
    if normalized["single_leo_training_scenes"] != checkpoint["scenes"]:
        raise ADV3B02TargetProtocolError("ADV train-data config LEO scene drift")
    if sha256_file(path) != raw_sha:
        raise ADV3B02TargetProtocolError("ADV train-data config changed during validation")
    return {
        "path": path,
        "sha256": raw_sha,
        "payload": payload,
        "normalized": normalized,
        "normalized_sha256": normalized_sha,
        "physical_axis_binding": physical_axis_binding,
        "physical_axis_binding_sha256": _require_sha(
            payload.get("physical_axis_binding_sha256"),
            label="ADV train-data physical-axis binding",
        ),
        "source_class_order": source_order,
        "source_class_order_sha256": checkpoint["source_class_order_sha256"],
        **required_baseline,
    }


def safe_received_iq_tensor(received_iq: Any, *, input_len: int):
    """Use the audited safe bridge; neither legacy NumPy entry point is used."""

    try:
        return _target._strict_received_iq(received_iq, input_len=int(input_len))
    except _target.CLICTargetProtocolError as exc:
        raise ADV3B02TargetProtocolError(str(exc)) from exc


def _safe_logits(value: Any, *, source_class_count: int) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ADV3B02TargetProtocolError("ADV target forward logits are invalid") from exc
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (source_class_count,) or not np.isfinite(values).all():
        raise ADV3B02TargetProtocolError("ADV target forward logits shape/non-finite drift")
    return np.ascontiguousarray(values, dtype=np.float64)


class _ADV3B02Runtime:
    """One fixed source model with a no-update, one-row forward boundary."""

    def __init__(
        self,
        *,
        model: Any,
        device: Any,
        input_len: int,
        source_class_order: Sequence[str],
        source_class_order_sha256: str,
    ) -> None:
        self._model = model
        self._device = device
        self._input_len = int(input_len)
        self.source_class_order = [str(item) for item in source_class_order]
        self.source_class_order_sha256 = str(source_class_order_sha256)

    def forward_once(self, received_iq: Any, *, scene: str) -> dict[str, Any]:
        if str(scene) not in FORMAL_LEO_WEAK_SCENARIOS:
            raise ADV3B02TargetProtocolError("ADV target forward scene drift")
        try:
            import torch

            tensor = safe_received_iq_tensor(
                received_iq, input_len=self._input_len
            ).to(self._device)
            with torch.no_grad():
                output = self._model(
                    tensor,
                    y_tx=None,
                    grl_lambda=1.0,
                    return_aux=True,
                )
        except ADV3B02TargetProtocolError:
            raise
        except Exception as exc:
            raise ADV3B02TargetProtocolError("ADV target model forward failed") from exc
        if not isinstance(output, Mapping):
            raise ADV3B02TargetProtocolError("ADV target model output is not a mapping")
        logits = output.get("tx_logits", output.get("logits"))
        if logits is None:
            raise ADV3B02TargetProtocolError("ADV target model output lacks tx logits")
        try:
            values = _target._tensor_to_numpy_float64(logits, label="ADV target logits")
        except _target.CLICTargetProtocolError as exc:
            raise ADV3B02TargetProtocolError(str(exc)) from exc
        return {"tx_logits": _safe_logits(values, source_class_count=len(self.source_class_order))}


def load_verified_adv3b02_runtime(
    *,
    checkpoint_path: str | Path,
    completion_receipt_path: str | Path,
    train_config_manifest_path: str | Path,
) -> _ADV3B02Runtime:
    """Strictly reconstruct a model only from the three verified blind inputs."""

    checkpoint_file = _path(checkpoint_path, label="ADV checkpoint")
    completion_file = _path(completion_receipt_path, label="ADV completion receipt")
    config_file = _path(train_config_manifest_path, label="ADV train-data config")
    checkpoint = _validate_adv_checkpoint(checkpoint_file)
    completion = _read_completion(completion_file, checkpoint=checkpoint)
    config = _read_verified_train_config(
        config_file, checkpoint=checkpoint, completion=completion
    )
    _verify_unchanged(
        {
            "checkpoint": (checkpoint_file, checkpoint["sha256"]),
            "completion receipt": (completion_file, completion["sha256"]),
            "train-data config": (config_file, config["sha256"]),
        }
    )
    try:
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, audit = build_exact_ssdg_model_from_checkpoint(
            checkpoint["payload"], input_len=checkpoint["input_len"], device=device
        )
        if audit.get("checkpoint_load_strict") is not True:
            raise ADV3B02TargetProtocolError("ADV strict model reconstruction audit drift")
        model.eval()
    except ADV3B02TargetProtocolError:
        raise
    except Exception as exc:
        raise ADV3B02TargetProtocolError(
            "ADV strict model reconstruction failed"
        ) from exc
    _verify_unchanged(
        {
            "checkpoint": (checkpoint_file, checkpoint["sha256"]),
            "completion receipt": (completion_file, completion["sha256"]),
            "train-data config": (config_file, config["sha256"]),
        }
    )
    return _ADV3B02Runtime(
        model=model,
        device=device,
        input_len=checkpoint["input_len"],
        source_class_order=config["source_class_order"],
        source_class_order_sha256=config["source_class_order_sha256"],
    )


def _require_runtime_source_order(runtime: Any, *, config: Mapping[str, Any]) -> None:
    order = _ids(
        getattr(runtime, "source_class_order", None),
        label="ADV predictor source class order",
        expected_count=4,
    )
    digest = _require_sha(
        getattr(runtime, "source_class_order_sha256", None),
        label="ADV predictor source class order",
    )
    if _canonical_sha(order, label="ADV predictor source class order") != digest:
        raise ADV3B02TargetProtocolError("ADV predictor source class order SHA drift")
    if order != config["source_class_order"] or digest != config["source_class_order_sha256"]:
        raise ADV3B02TargetProtocolError("ADV predictor/train-data source class binding drift")


def _prediction_row(
    *,
    opaque_token: str,
    scene: str,
    received_iq_sha256: str,
    output: Mapping[str, Any],
    source_class_count: int,
) -> dict[str, Any]:
    token = _require_sha(opaque_token, label="ADV prediction opaque token")
    received_sha = _require_sha(
        received_iq_sha256, label="ADV prediction received-IQ"
    )
    if scene not in FORMAL_LEO_WEAK_SCENARIOS:
        raise ADV3B02TargetProtocolError("ADV prediction scene drift")
    logits = _safe_logits(
        output.get("tx_logits"), source_class_count=source_class_count
    )
    maximum = float(np.max(logits))
    winners = np.flatnonzero(logits == maximum)
    if winners.size != 1:
        raise ADV3B02TargetProtocolError("ADV prediction exact-head tie")
    return {
        "opaque_token": token,
        "scene": scene,
        "received_iq_sha256": received_sha,
        "predicted_index": int(winners[0]),
    }


def _require_exact_package_shape(package: Mapping[str, Any], *, input_len: int) -> None:
    manifest = _mapping(package.get("manifest"), label="IQ-only package manifest")
    if _require_int(manifest.get("row_count"), label="IQ-only package row count") != 3120:
        raise ADV3B02TargetProtocolError("IQ-only package must contain exactly 3120 rows")
    expected_counts = {scene: 1040 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    if manifest.get("row_count_by_scene") != expected_counts:
        raise ADV3B02TargetProtocolError("IQ-only package formal scene row counts drift")
    received = np.asarray(package.get("received_iq"))
    if received.shape != (3120, 2, int(input_len)):
        raise ADV3B02TargetProtocolError("IQ-only package received-IQ shape drift")
    scenes = np.asarray(package.get("scenes")).astype(str)
    if scenes.shape != (3120,) or [
        str(item) for item in scenes.tolist()
    ] != [
        scene
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for _ in range(1040)
    ]:
        raise ADV3B02TargetProtocolError("IQ-only package formal scene order drift")


def publish_adv3b02_target_prediction(
    *,
    checkpoint_path: str | Path,
    completion_receipt_path: str | Path,
    train_config_manifest_path: str | Path,
    iq_only_package_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Forward each sealed IQ-only row once and publish an immutable opaque artifact."""

    checkpoint_file = _path(checkpoint_path, label="ADV checkpoint")
    completion_file = _path(completion_receipt_path, label="ADV completion receipt")
    config_file = _path(train_config_manifest_path, label="ADV train-data config")
    package_dir = _path(iq_only_package_path, label="IQ-only package")
    output_file = _path(output_path, label="ADV target prediction output")
    if output_file.exists():
        raise ADV3B02TargetProtocolError(
            f"ADV target prediction output already exists and is immutable: {output_file}"
        )
    checkpoint = _validate_adv_checkpoint(checkpoint_file)
    completion = _read_completion(completion_file, checkpoint=checkpoint)
    config = _read_verified_train_config(
        config_file, checkpoint=checkpoint, completion=completion
    )
    try:
        package = _load_verified_clic_iq_only_package(package_dir)
    except _target.CLICTargetProtocolError as exc:
        raise ADV3B02TargetProtocolError(f"IQ-only package validation failed: {exc}") from exc
    _require_exact_package_shape(
        package, input_len=int(config["normalized"]["preprocessing"]["input_len"])
    )
    input_hashes = {
        "checkpoint": (checkpoint_file, checkpoint["sha256"]),
        "completion receipt": (completion_file, completion["sha256"]),
        "train-data config": (config_file, config["sha256"]),
        "IQ-only package manifest": (
            Path(package["manifest_path"]),
            str(package["manifest_raw_sha256"]),
        ),
        "IQ-only received-IQ data": (
            Path(package["data_path"]),
            str(package["data_raw_sha256"]),
        ),
    }
    _verify_unchanged(input_hashes)
    runtime = load_verified_adv3b02_runtime(
        checkpoint_path=checkpoint_file,
        completion_receipt_path=completion_file,
        train_config_manifest_path=config_file,
    )
    _require_runtime_source_order(runtime, config=config)
    _verify_unchanged(input_hashes)
    rows: list[dict[str, Any]] = []
    received = np.asarray(package["received_iq"], dtype=np.float32)
    tokens = np.asarray(package["opaque_tokens"]).astype(str)
    scenes = np.asarray(package["scenes"]).astype(str)
    iq_hashes = np.asarray(package["received_iq_sha256"]).astype(str)
    for index in range(3120):
        output = runtime.forward_once(received[index], scene=str(scenes[index]))
        if not isinstance(output, Mapping):
            raise ADV3B02TargetProtocolError("ADV predictor forward did not return a mapping")
        rows.append(
            _prediction_row(
                opaque_token=str(tokens[index]),
                scene=str(scenes[index]),
                received_iq_sha256=str(iq_hashes[index]),
                output=output,
                source_class_count=len(config["source_class_order"]),
            )
        )
    if len(rows) != 3120 or len({row["opaque_token"] for row in rows}) != 3120:
        raise ADV3B02TargetProtocolError("ADV target prediction forward closure failed")
    _verify_unchanged(input_hashes)
    payload = {
        "schema": ADV3B02_PREDICTION_SCHEMA,
        "sealed": True,
        "truth_sidecar_opened": False,
        "checkpoint_sha256": checkpoint["sha256"],
        "completion_receipt_sha256": completion["sha256"],
        "train_config_manifest_sha256": config["sha256"],
        "train_config_normalized_sha256": config["normalized_sha256"],
        "train_config_physical_axis_binding_sha256": config[
            "physical_axis_binding_sha256"
        ],
        "package_manifest_sha256": package["manifest_raw_sha256"],
        "received_iq_data_sha256": package["data_raw_sha256"],
        "package_sha256": package["manifest"]["package_sha256"],
        "input_artifact_sha256": {
            "checkpoint": checkpoint["sha256"],
            "completion_receipt": completion["sha256"],
            "train_config_manifest": config["sha256"],
            "iq_only_package_manifest": package["manifest_raw_sha256"],
            "received_iq_data": package["data_raw_sha256"],
        },
        "source_class_order": list(config["source_class_order"]),
        "source_class_order_sha256": config["source_class_order_sha256"],
        "baseline_terminal_status": completion["baseline_terminal_status"],
        "baseline_exit_code": completion["baseline_exit_code"],
        "baseline_promotion_ready": completion["baseline_promotion_ready"],
        "formal_performance_claim": completion["formal_performance_claim"],
        "row_count": 3120,
        "forward_count": 3120,
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "rows": rows,
    }
    return _write_immutable_json(
        output_file, payload, label="ADV target prediction output"
    ).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the intentionally narrow source sealer and four-input blind CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--seal-train-data-config", action="store_true")
    modes.add_argument("--publish-target-prediction", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--completion-receipt-json")
    parser.add_argument("--clean-v4-npz")
    parser.add_argument("--train-config-manifest")
    parser.add_argument("--iq-only-package")
    parser.add_argument("--output")
    return parser


def _require_cli_paths(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *fields: str
) -> None:
    missing = [
        f"--{field.replace('_', '-')}"
        for field in fields
        if not isinstance(getattr(args, field, None), str)
        or not str(getattr(args, field)).strip()
    ]
    if missing:
        parser.error(f"selected mode requires {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.seal_train_data_config:
        _require_cli_paths(
            parser,
            args,
            "checkpoint",
            "completion_receipt_json",
            "clean_v4_npz",
            "output",
        )
        result = seal_adv3b02_train_data_config(
            checkpoint_path=args.checkpoint,
            completion_receipt_path=args.completion_receipt_json,
            clean_v4_npz_path=args.clean_v4_npz,
            output_path=args.output,
        )
    else:
        _require_cli_paths(
            parser,
            args,
            "checkpoint",
            "completion_receipt_json",
            "train_config_manifest",
            "iq_only_package",
            "output",
        )
        result = publish_adv3b02_target_prediction(
            checkpoint_path=args.checkpoint,
            completion_receipt_path=args.completion_receipt_json,
            train_config_manifest_path=args.train_config_manifest,
            iq_only_package_path=args.iq_only_package,
            output_path=args.output,
        )
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
