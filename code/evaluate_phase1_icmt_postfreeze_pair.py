#!/usr/bin/env python
"""Final-only source-paired closure for frozen P1-ICMT checkpoints.

The evaluator reuses the established pure-NumPy CB/CP export validators.  An
ICMT-only split export supplies the checkpoint-identical labelled L rows and
local4 source-validation V rows.  The frozen Gaussian fits L only, scores V as
registered-known and scores the TX-exclusive proxy from the common clean
export.  No U row is forwarded or fitted, and no threshold, calibration,
sweep, checkpoint selection, or checkpoint-weight loading occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SHARED_SCRIPT_DIR = SCRIPT_DIR / "scripts"
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))

import eval_phase1_cb_sfce_pair as _cb  # noqa: E402
import export_phase1_icmt_leo_features as _icmt_leo  # noqa: E402


EXPECTED_SCENARIOS = _cb.EXPECTED_SCENARIOS
EXPECTED_SOURCE_DAYS = _cb.EXPECTED_SOURCE_DAYS
EXPECTED_SOURCE_RXS = _cb.EXPECTED_SOURCE_RXS
EXPECTED_LEO_RUNTIME_VIEW = _cb.EXPECTED_LEO_RUNTIME_VIEW
FROZEN_FOLD_SOURCE_TX = _cb.FROZEN_FOLD_SOURCE_TX
METADATA_FIELDS = _cb.METADATA_FIELDS
CLASSIFICATION_METRICS = _cb.CLASSIFICATION_METRICS
FLOOR_DELTA_LIMIT_PP = _cb.FLOOR_DELTA_LIMIT_PP

EXPECTED_CLASSIFICATION_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_TRAINING_RUN_LEAF = "phase1_icmt12_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.icmt_lv_export.v1"
EXPECTED_PAIR_SCHEMA = "cvs.phase1.icmt_postfreeze_pair.v2"
EXPECTED_ICMT_RECEIPT_SCHEMA = "cvs.phase1.icmt_receipt.v1"
FROZEN_FOLD_KNOWN_HELDOUT_TX = {
    1: "14-7",
    2: "20-15",
    3: "20-19",
    4: "6-15",
    5: "8-20",
    6: "14-10",
}
FROZEN_FOLD_PROXY_TX = {
    1: "14-10",
    2: "14-7",
    3: "20-15",
    4: "20-19",
    5: "6-15",
    6: "8-20",
}
GEOMETRY_DDOF = 1
GEOMETRY_VARIANCE_FLOOR = 1e-6
GEOMETRY_SHRINKAGE = 0.10
FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400

FROZEN_POSTFREEZE_CONTRACT = {
    "ICMT-PF-01": "final-only z_id; diagonal Gaussian fit from checkpoint-identical labelled L rows only",
    "ICMT-PF-02": "totalized float64 row L2: positive norm maps to z/norm and exact zero norm maps to zero",
    "ICMT-PF-03": "continuous u=log4-logsumexp(-NLL); no threshold, calibration, sweep, or selection",
    "ICMT-PF-04": "local4 source-validation V is registered-known; proxy_unknown is source-proxy; neither enters fit",
    "ICMT-PF-05": "strict C/G L/V+clean+LEO NPZ, proxy, head, manifest, checkpoint, arm, root, pair, TX, and matrix binding",
    "ICMT-PF-06": "clean/LEO classification gates plus two strictly positive proxy deltas, all non-compensating",
    "ICMT-PF-07": "retain every L/V/proxy row and seal C/G role plus labelled-fit per-class norm counts",
    "ICMT-PF-08": "source proxy is fixed to signed GD-v2 days/RXs/seed/max-per-TX/total=400 and bound across NPZ, JSON, and CSV",
}


class ICMTPostfreezePairError(RuntimeError):
    """Raised when frozen P1-ICMT postfreeze evidence cannot close."""


def _translate_cb_error(error: BaseException) -> ICMTPostfreezePairError:
    return ICMTPostfreezePairError(str(error))


def _canonical_training_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF:
        raise ICMTPostfreezePairError(
            f"training run root leaf must be {EXPECTED_TRAINING_RUN_LEAF}: {root}"
        )
    if not root.is_dir():
        raise ICMTPostfreezePairError(f"training run root must already exist: {root}")
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if arm not in {"C", "G"}:
        raise ICMTPostfreezePairError(f"unsupported frozen arm: {arm}")
    candidate = f"F{fold_index}{arm}_ICMT12"
    return candidate, (training_root / candidate / "final_ssdg.pth").resolve()


def _require_exact_final_checkpoint(value: str | Path, expected: Path, *, label: str) -> Path:
    observed = Path(value).resolve()
    if observed != expected:
        raise ICMTPostfreezePairError(
            f"{label} final checkpoint path does not match frozen candidate path: "
            f"expected={expected} observed={observed}"
        )
    return observed


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    if str(manifest.get("classification_head_contract", "")) != EXPECTED_CLASSIFICATION_HEAD_CONTRACT:
        raise ICMTPostfreezePairError(
            f"{label} classification_head_contract must be {EXPECTED_CLASSIFICATION_HEAD_CONTRACT}"
        )
    checkpoint_value = str(manifest.get("checkpoint", "")).strip()
    if not checkpoint_value:
        raise ICMTPostfreezePairError(f"{label} lacks manifest checkpoint path")
    observed_checkpoint = Path(checkpoint_value).resolve()
    if observed_checkpoint != expected_checkpoint:
        raise ICMTPostfreezePairError(
            f"{label} manifest checkpoint path does not bind frozen {expected_candidate}: "
            f"expected={expected_checkpoint} observed={observed_checkpoint}"
        )


def _validate_payload_identity(
    payload: Mapping[str, Any],
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks manifest")
    _validate_manifest_identity(
        manifest,
        expected_checkpoint=expected_checkpoint,
        expected_candidate=expected_candidate,
        label=label,
    )


def _load_icmt_leo_binding(
    path: str | Path,
    leo_payload: Mapping[str, Any],
    clean_payload: Mapping[str, Any],
    *,
    expected_npz: Path,
    expected_checkpoint: Path,
    expected_candidate: str,
    fold_index: int,
    arm: str,
    source_tx_ids: Sequence[str],
    training_root: Path,
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    expected_path = (output_root / expected_candidate / "source_leo_binding.json").resolve()
    if source != expected_path or not source.is_file():
        raise ICMTPostfreezePairError(
            f"{label} ICMT LEO binding path does not match frozen candidate layout"
        )
    try:
        binding = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ICMTPostfreezePairError(f"{label} ICMT LEO binding JSON is invalid") from exc
    if not isinstance(binding, Mapping):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO binding must encode an object")
    exact = {
        "schema": _icmt_leo.EXPECTED_BINDING_SCHEMA,
        "candidate_id": expected_candidate,
        "fold_index": int(fold_index),
        "arm": str(arm),
        "training_run_root": str(training_root),
        "postfreeze_output_root": str(output_root),
        "checkpoint_path": str(expected_checkpoint),
        "checkpoint_role": "training_final_only",
        "training_run_id": EXPECTED_TRAINING_RUN_LEAF,
        "classification_head_contract": EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
        "leo_npz_path": str(expected_npz),
        "dataset_sha256": FROZEN_WISIG_SHA256,
        "all_source_rows_reconstructed": True,
        "all_scenarios_complete": True,
    }
    for field, expected in exact.items():
        observed = binding.get(field)
        if observed != expected or type(observed) is not type(expected):
            raise ICMTPostfreezePairError(
                f"{label} ICMT LEO binding {field} drifted"
            )
    clean_manifest = clean_payload.get("manifest")
    leo_manifest = leo_payload.get("manifest")
    if not isinstance(clean_manifest, Mapping) or not isinstance(leo_manifest, Mapping):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO binding lacks manifests")
    clean_dataset = Path(str(clean_manifest.get("dataset_path", ""))).resolve()
    binding_dataset = Path(str(binding.get("dataset_path", ""))).resolve()
    if binding_dataset != clean_dataset:
        raise ICMTPostfreezePairError(
            f"{label} ICMT LEO dataset path does not bind clean export"
        )
    if str(clean_manifest.get("wisig_pkl_sha256", "")) != FROZEN_WISIG_SHA256:
        raise ICMTPostfreezePairError(f"{label} clean dataset SHA256 is not frozen")
    current_npz_sha = _cb._sha256_file(expected_npz)
    if str(binding.get("leo_npz_sha256", "")) != current_npz_sha:
        raise ICMTPostfreezePairError(
            f"{label} ICMT LEO NPZ current SHA256 does not match sidecar"
        )
    current_checkpoint_sha = _cb._sha256_file(expected_checkpoint)
    if str(binding.get("checkpoint_sha256", "")) != current_checkpoint_sha:
        raise ICMTPostfreezePairError(
            f"{label} ICMT LEO checkpoint SHA256 does not match current checkpoint"
        )
    if str(binding.get("leo_manifest_sha256", "")) != _icmt_leo._canonical_json_sha256(
        leo_manifest
    ):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO manifest SHA256 drifted")
    selection = binding.get("source_selection")
    if not isinstance(selection, Mapping):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO binding lacks source selection")
    selection_without_hash = dict(selection)
    recorded_selection_sha = str(selection_without_hash.pop("selection_sha256", ""))
    if recorded_selection_sha != _icmt_leo._canonical_json_sha256(selection_without_hash):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO source selection SHA256 drifted")
    frozen_selection = {
        "source_tx_ids": list(source_tx_ids),
        "source_rx_ids": list(EXPECTED_SOURCE_RXS),
        "source_day_ids": list(EXPECTED_SOURCE_DAYS),
        "equalized": _icmt_leo.EXPECTED_EQUALIZED,
        "domain": _icmt_leo.EXPECTED_DOMAIN,
        "out_len": _icmt_leo.EXPECTED_OUT_LEN,
        "max_samples_per_combo": _icmt_leo.EXPECTED_MAX_PER_COMBO,
        "max_samples_per_tx": _icmt_leo.EXPECTED_MAX_PER_TX,
        "export_seed": _icmt_leo.EXPECTED_EXPORT_SEED,
        "batch_size": _icmt_leo.EXPECTED_BATCH_SIZE,
        "channel_view": _icmt_leo.EXPECTED_CHANNEL_VIEW,
        "runtime_view": _icmt_leo.EXPECTED_RUNTIME_VIEW,
        "satellite_scenarios": list(EXPECTED_SCENARIOS),
        "source_sat_seed": _icmt_leo.EXPECTED_SOURCE_SAT_SEED,
        "satellite_tta_policy": _icmt_leo.EXPECTED_TTA_POLICY,
        "star_ground_channel_impl": _icmt_leo.EXPECTED_STAR_GROUND_IMPL,
    }
    for field, expected in frozen_selection.items():
        if selection.get(field) != expected or type(selection.get(field)) is not type(expected):
            raise ICMTPostfreezePairError(
                f"{label} ICMT LEO source selection {field} drifted"
            )
    row_count = int(np.asarray(leo_payload["tx_ids"]).size)
    if int(selection.get("reconstructed_size", -1)) != row_count:
        raise ICMTPostfreezePairError(f"{label} ICMT LEO reconstructed size drifted")
    try:
        physical_receipt = _icmt_leo._physical_key_receipt(
            _icmt_leo._physical_keys_from_payload(leo_payload)
        )
        coverage_receipt = _icmt_leo._scenario_coverage_receipt(
            leo_payload, source_tx_ids=source_tx_ids
        )
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise ICMTPostfreezePairError(f"{label} {exc}") from exc
    if binding.get("physical_keys") != physical_receipt:
        raise ICMTPostfreezePairError(f"{label} ICMT LEO physical-key receipt drifted")
    if binding.get("scenario_coverage") != coverage_receipt:
        raise ICMTPostfreezePairError(f"{label} ICMT LEO scenario coverage receipt drifted")
    scenarios = np.asarray(leo_payload["sat_scenarios"]).astype(str).tolist()
    if str(binding.get("scenario_assignment_sha256", "")) != _icmt_leo._canonical_json_sha256(
        scenarios
    ):
        raise ICMTPostfreezePairError(f"{label} ICMT LEO scenario assignment SHA256 drifted")
    return {
        "path": str(source),
        "sha256": _cb._sha256_file(source),
        "leo_npz_sha256": current_npz_sha,
        "dataset_path": str(binding_dataset),
        "dataset_sha256": FROZEN_WISIG_SHA256,
        "selection_sha256": recorded_selection_sha,
        "physical_key_set_sha256": physical_receipt["set_sha256"],
        "scenario_coverage": coverage_receipt,
    }


def _validate_proxy_manifest_identity(
    path: str | Path,
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ICMTPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("manifest"), Mapping):
        raise ICMTPostfreezePairError(f"{label} proxy diagnostic lacks manifest")
    _validate_manifest_identity(
        raw["manifest"],
        expected_checkpoint=expected_checkpoint,
        expected_candidate=expected_candidate,
        label=f"{label} proxy manifest",
    )


def _load_proxy_scores_csv(
    path: str | Path,
    clean_payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_known_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ICMTPostfreezePairError(f"missing {label} proxy score CSV: {source}")
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "row",
                "role",
                "tx_id",
                "rx_id",
                "day_id",
                "channel_view",
                "sat_scenario",
                "is_known_query",
                "is_unknown_query",
            }
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ICMTPostfreezePairError(
                    f"{label} proxy score CSV lacks frozen metadata columns"
                )
            rows = list(reader)
    except ICMTPostfreezePairError:
        raise
    except Exception as exc:
        raise ICMTPostfreezePairError(f"{label} proxy score CSV is invalid") from exc
    total_rows = int(np.asarray(clean_payload["features"]).shape[0])
    if len(rows) != total_rows:
        raise ICMTPostfreezePairError(f"{label} proxy score CSV/clean row count mismatch")
    known_count = 0
    proxy_count = 0
    source_tx_set = set(str(value) for value in source_tx_ids)
    metadata = {
        "role": "dataset_role",
        "tx_id": "tx_ids",
        "rx_id": "rx_ids",
        "day_id": "day_ids",
        "channel_view": "channel_views",
        "sat_scenario": "sat_scenarios",
    }
    for index, row in enumerate(rows):
        try:
            observed_index = int(row["row"])
            known_flag = int(row["is_known_query"])
            unknown_flag = int(row["is_unknown_query"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ICMTPostfreezePairError(
                f"{label} proxy score CSV has malformed row/role flags"
            ) from exc
        if observed_index != index or known_flag not in {0, 1} or unknown_flag not in {0, 1}:
            raise ICMTPostfreezePairError(
                f"{label} proxy score CSV row order or role flags drifted"
            )
        for csv_field, payload_field in metadata.items():
            expected = str(np.asarray(clean_payload[payload_field]).reshape(-1)[index])
            if str(row[csv_field]) != expected:
                raise ICMTPostfreezePairError(
                    f"{label} proxy score CSV {csv_field} does not bind clean row {index}"
                )
        role = str(row["role"])
        tx_id = str(row["tx_id"])
        expected_known = role == "source_validation_known" and tx_id in source_tx_set
        expected_unknown = role == "proxy_unknown"
        if known_flag != int(expected_known) or unknown_flag != int(expected_unknown):
            raise ICMTPostfreezePairError(
                f"{label} proxy score CSV query-role flags do not bind clean roles"
            )
        known_count += int(expected_known)
        proxy_count += int(expected_unknown)
    if known_count != int(expected_known_count):
        raise ICMTPostfreezePairError(f"{label} proxy score CSV known row count mismatch")
    if proxy_count != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"{label} proxy score CSV unknown row count must equal frozen "
            f"{FROZEN_PROXY_TOTAL_COUNT}"
        )
    return {
        "path": str(source),
        "sha256": _cb._sha256_file(source),
        "total_rows": len(rows),
        "known_query_count": known_count,
        "unknown_query_count": proxy_count,
    }


def _load_proxy_binding(
    path: str | Path,
    scores_csv_path: str | Path,
    clean_payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    expected_known_count: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    if int(expected_proxy_count) != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"{label} expected proxy count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
        )
    source = Path(path)
    if not source.is_file():
        raise ICMTPostfreezePairError(f"missing {label} proxy diagnostic JSON: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ICMTPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(raw, Mapping):
        raise ICMTPostfreezePairError(f"{label} proxy diagnostic JSON must encode an object")
    if tuple(str(item) for item in raw.get("source_tx_ids", [])) != tuple(source_tx_ids):
        raise ICMTPostfreezePairError(f"{label} proxy source TX order does not bind clean NPZ")
    if list(raw.get("known_query_roles", [])) != ["source_validation_known"]:
        raise ICMTPostfreezePairError(f"{label} proxy known role is not source_validation_known")
    if list(raw.get("unknown_query_roles", [])) != ["proxy_unknown"]:
        raise ICMTPostfreezePairError(f"{label} proxy unknown role is not proxy_unknown")
    if str(raw.get("threshold_scope", "")) != "source_calibrated_only_no_target_support_no_unknown_query_tuning":
        raise ICMTPostfreezePairError(f"{label} proxy threshold scope drifted")
    if int(raw.get("known_query_count", -1)) != int(expected_known_count):
        raise ICMTPostfreezePairError(f"{label} proxy known row count mismatch")
    if int(raw.get("unknown_query_count", -1)) != int(expected_proxy_count):
        raise ICMTPostfreezePairError(f"{label} proxy unknown row count mismatch")
    proxy_manifest = raw.get("manifest")
    if not isinstance(proxy_manifest, Mapping) or _cb._manifest_sha256(proxy_manifest) != _cb._manifest_sha256(
        clean_payload["manifest"]
    ):
        raise ICMTPostfreezePairError(f"{label} proxy manifest does not bind clean NPZ")
    feature_path = Path(str(raw.get("feature_npz", "")))
    if feature_path.resolve() != Path(str(clean_payload["path"])).resolve():
        raise ICMTPostfreezePairError(f"{label} proxy feature NPZ path does not bind clean NPZ")
    values: dict[str, float] = {}
    for field in ("AUROC_unknown", "unknown_FAR"):
        try:
            value = float(raw[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ICMTPostfreezePairError(f"{label} proxy diagnostic lacks finite {field}") from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ICMTPostfreezePairError(f"{label} proxy diagnostic has invalid {field}")
        values[field] = value
    csv_binding = _load_proxy_scores_csv(
        scores_csv_path,
        clean_payload,
        source_tx_ids,
        expected_known_count,
        label=label,
    )
    if int(raw.get("unknown_query_count", -1)) != int(csv_binding["unknown_query_count"]):
        raise ICMTPostfreezePairError(f"{label} proxy JSON/CSV unknown count mismatch")
    if int(raw.get("known_query_count", -1)) != int(csv_binding["known_query_count"]):
        raise ICMTPostfreezePairError(f"{label} proxy JSON/CSV known count mismatch")
    return {
        "path": str(source.resolve()),
        "sha256": _cb._sha256_file(source),
        "scores_csv_path": csv_binding["path"],
        "scores_csv_sha256": csv_binding["sha256"],
        **values,
    }


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise ICMTPostfreezePairError(f"refusing to overwrite final-only pair output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise ICMTPostfreezePairError(f"refusing to overwrite temporary pair output: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _role_mask(payload: Mapping[str, Any], role: str) -> np.ndarray:
    return np.asarray(payload["dataset_role"] == str(role), dtype=bool)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_sha256(values: Sequence[Any]) -> str:
    return _canonical_json_sha256([int(value) for value in values])


def _load_lv_npz(path: str | Path) -> dict[str, Any]:
    payload = _cb._load_npz(path)
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if "source_base_indices" not in data.files:
            raise ICMTPostfreezePairError(
                f"{source} is missing source_base_indices"
            )
        indices = np.asarray(data["source_base_indices"], dtype=np.int64).reshape(-1)
    if indices.size != int(payload["features"].shape[0]):
        raise ICMTPostfreezePairError(
            f"{source} source_base_indices/features row mismatch"
        )
    payload["source_base_indices"] = indices
    return payload


def _validate_lv_payload(
    payload: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    fold_index: int,
    expected_proxy_count: int,
    *,
    label: str,
) -> dict[str, Any]:
    if int(expected_proxy_count) != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"{label} expected proxy count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
        )
    _cb._validate_logit_contract(payload, source_tx_ids, label=label)
    manifest = payload["manifest"]
    if str(manifest.get("schema", "")) != EXPECTED_LV_EXPORT_SCHEMA:
        raise ICMTPostfreezePairError(f"{label} L/V export schema mismatch")
    if str(manifest.get("checkpoint_role", "")) != "training_final_only":
        raise ICMTPostfreezePairError(f"{label} L/V checkpoint role is not training_final_only")
    if str(manifest.get("checkpoint_selection", "")) != "final_only":
        raise ICMTPostfreezePairError(f"{label} L/V checkpoint selection is not final_only")
    if str(manifest.get("split_mode", "")) != "tx_rx_day_1_6_3":
        raise ICMTPostfreezePairError(f"{label} L/V split mode mismatch")
    if str(manifest.get("feature_name", "")) != "z_id" or str(
        manifest.get("z_id_source_key", "")
    ) != "feat_joint":
        raise ICMTPostfreezePairError(f"{label} L/V z_id=feat_joint path binding mismatch")
    if str(manifest.get("postfreeze_geometry_path", "")) != "checkpoint_model.feat_joint_as_z_id":
        raise ICMTPostfreezePairError(f"{label} L/V postfreeze geometry path drifted")
    arm = "C" if label.startswith("C") else "G" if label.startswith("G") else ""
    expected_candidate = f"F{fold_index}{arm}_ICMT12"
    if not arm or str(manifest.get("candidate_id", "")) != expected_candidate:
        raise ICMTPostfreezePairError(f"{label} L/V candidate arm binding mismatch")
    if str(manifest.get("run_id", "")) != EXPECTED_TRAINING_RUN_LEAF:
        raise ICMTPostfreezePairError(f"{label} L/V training run binding mismatch")
    if str(manifest.get("training_run_contract", "")) != EXPECTED_TRAINING_RUN_LEAF:
        raise ICMTPostfreezePairError(f"{label} L/V training root contract mismatch")
    if str(manifest.get("icmt_receipt_schema", "")) != EXPECTED_ICMT_RECEIPT_SCHEMA:
        raise ICMTPostfreezePairError(f"{label} L/V ICMT receipt schema mismatch")
    if manifest.get("icmt_enabled") is not (arm == "G"):
        raise ICMTPostfreezePairError(f"{label} L/V ICMT enabled flag mismatches arm")
    frozen_scalars = {
        "seed": 7281105,
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
    }
    for field, expected in frozen_scalars.items():
        try:
            observed = float(manifest[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ICMTPostfreezePairError(f"{label} L/V manifest lacks {field}") from exc
        if not math.isfinite(observed) or observed != float(expected):
            raise ICMTPostfreezePairError(
                f"{label} L/V manifest {field} mismatch: expected={expected} observed={observed}"
            )
    if tuple(str(item) for item in manifest.get("source_tx_ids", [])) != tuple(source_tx_ids):
        raise ICMTPostfreezePairError(f"{label} L/V source TX order mismatch")
    if tuple(str(item) for item in manifest.get("known_validation_outer_tx_ids", [])) != (
        FROZEN_FOLD_KNOWN_HELDOUT_TX[fold_index],
    ):
        raise ICMTPostfreezePairError(f"{label} L/V outer held TX binding mismatch")
    if tuple(str(item) for item in manifest.get("proxy_unknown_tx_ids", [])) != (
        FROZEN_FOLD_PROXY_TX[fold_index],
    ):
        raise ICMTPostfreezePairError(f"{label} L/V proxy TX binding mismatch")
    frozen_proxy_text = {
        "proxy_days": ",".join(FROZEN_PROXY_DAYS),
        "proxy_rxs": ",".join(FROZEN_PROXY_RXS),
    }
    for field, expected in frozen_proxy_text.items():
        if str(manifest.get(field, "")) != expected:
            raise ICMTPostfreezePairError(
                f"{label} L/V manifest {field} does not equal frozen proxy selection"
            )
    frozen_proxy_counts = {
        "proxy_seed": FROZEN_PROXY_SELECTION_SEED,
        "proxy_max_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        "proxy_expected_total_count": FROZEN_PROXY_TOTAL_COUNT,
        "proxy_row_count": FROZEN_PROXY_TOTAL_COUNT,
    }
    for field, expected in frozen_proxy_counts.items():
        if _strict_count(manifest, field, label=f"{label} L/V manifest") != expected:
            raise ICMTPostfreezePairError(
                f"{label} L/V manifest {field} does not equal frozen {expected}"
            )
    proxy_selection = manifest.get("proxy_selection")
    if not isinstance(proxy_selection, Mapping):
        raise ICMTPostfreezePairError(f"{label} L/V manifest lacks frozen proxy selection")
    selection_without_hash = dict(proxy_selection)
    selection_sha = str(selection_without_hash.pop("selection_sha256", ""))
    expected_selection = {
        "days": list(FROZEN_PROXY_DAYS),
        "rxs": list(FROZEN_PROXY_RXS),
        "selection_seed": FROZEN_PROXY_SELECTION_SEED,
        "max_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        "expected_total_count": FROZEN_PROXY_TOTAL_COUNT,
    }
    if selection_without_hash != expected_selection or selection_sha != _canonical_json_sha256(
        expected_selection
    ):
        raise ICMTPostfreezePairError(f"{label} L/V frozen proxy selection receipt drifted")
    proxy_export_info = manifest.get("proxy_export_info")
    if not isinstance(proxy_export_info, Mapping):
        raise ICMTPostfreezePairError(f"{label} L/V manifest lacks proxy export info")
    expected_export_info = {
        "role": "proxy_unknown",
        "days": ",".join(FROZEN_PROXY_DAYS),
        "rxs": ",".join(FROZEN_PROXY_RXS),
        "size": FROZEN_PROXY_TOTAL_COUNT,
    }
    for field, expected in expected_export_info.items():
        if proxy_export_info.get(field) != expected or type(proxy_export_info.get(field)) is not type(
            expected
        ):
            raise ICMTPostfreezePairError(
                f"{label} L/V proxy export info {field} drifted"
            )
    if manifest.get("source_split_receipt_checkpoint_equal") is not True:
        raise ICMTPostfreezePairError(f"{label} L/V split receipt is not checkpoint-equal")
    if manifest.get("tx_partition_receipt_checkpoint_equal") is not True:
        raise ICMTPostfreezePairError(f"{label} L/V TX partition receipt is not checkpoint-equal")
    if manifest.get("labeled_source_validation_physical_disjoint") is not True:
        raise ICMTPostfreezePairError(f"{label} L/V physical disjoint receipt is not true")
    if manifest.get("labeled_validation_proxy_physical_disjoint") is not True:
        raise ICMTPostfreezePairError(f"{label} L/V/proxy physical disjoint receipt is not true")
    if int(manifest.get("unlabeled_forward_rows", -1)) != 0:
        raise ICMTPostfreezePairError(f"{label} forwarded U rows")
    if manifest.get("unlabeled_loader_constructed") is not False:
        raise ICMTPostfreezePairError(f"{label} constructed a U loader")
    if int(manifest.get("unlabeled_loader_rows", -1)) != 0:
        raise ICMTPostfreezePairError(f"{label} iterated U loader rows")
    if manifest.get("unlabeled_features_persisted") is not False:
        raise ICMTPostfreezePairError(f"{label} persisted U features")
    if tuple(str(item) for item in manifest.get("forwarded_roles", [])) != (
        "labeled_fit",
        "source_validation_known",
        "proxy_unknown",
    ):
        raise ICMTPostfreezePairError(f"{label} forwarded role contract mismatch")
    roles = set(payload["dataset_role"].tolist())
    if roles != {"labeled_fit", "source_validation_known", "proxy_unknown"}:
        raise ICMTPostfreezePairError(f"{label} L/V/proxy payload roles mismatch")
    if set(payload["channel_views"].tolist()) != {"clean"}:
        raise ICMTPostfreezePairError(f"{label} L/V payload must be clean-only")
    if any(str(value) for value in payload["sat_scenarios"].tolist()):
        raise ICMTPostfreezePairError(f"{label} L/V payload assigns satellite scenarios")
    labeled_mask = _role_mask(payload, "labeled_fit")
    validation_mask = _role_mask(payload, "source_validation_known")
    proxy_mask = _role_mask(payload, "proxy_unknown")
    for role, mask in (
        ("labeled_fit", labeled_mask),
        ("source_validation_known", validation_mask),
    ):
        if set(payload["tx_ids"][mask].tolist()) != set(source_tx_ids):
            raise ICMTPostfreezePairError(f"{label} {role} lacks exact local4 TX coverage")
    if set(payload["tx_ids"][proxy_mask].tolist()) != {FROZEN_FOLD_PROXY_TX[fold_index]}:
        raise ICMTPostfreezePairError(f"{label} proxy_unknown TX set mismatch")
    if set(payload["day_ids"][proxy_mask].tolist()) != set(FROZEN_PROXY_DAYS):
        raise ICMTPostfreezePairError(f"{label} proxy_unknown day coverage drifted")
    if set(payload["rx_ids"][proxy_mask].tolist()) != set(FROZEN_PROXY_RXS):
        raise ICMTPostfreezePairError(f"{label} proxy_unknown RX coverage drifted")
    indices = np.asarray(payload["source_base_indices"], dtype=np.int64)
    labeled_indices = indices[labeled_mask]
    validation_indices = indices[validation_mask]
    if len(set(np.concatenate((labeled_indices, validation_indices)).tolist())) != int(
        labeled_indices.size + validation_indices.size
    ):
        raise ICMTPostfreezePairError(f"{label} L/V source_base_indices contain duplicates")
    if set(labeled_indices.tolist()) & set(validation_indices.tolist()):
        raise ICMTPostfreezePairError(f"{label} L/V source_base_indices overlap")
    receipt = manifest.get("source_split_receipt")
    if not isinstance(receipt, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks source split receipt")
    if manifest.get("dataset_path_checkpoint_equal") is not True:
        raise ICMTPostfreezePairError(f"{label} dataset path is not checkpoint-equal")
    actual_dataset_sha256 = str(manifest.get("wisig_pkl_sha256", "")).lower()
    expected_dataset_sha256 = str(manifest.get("expected_wisig_pkl_sha256", "")).lower()
    if actual_dataset_sha256 != FROZEN_WISIG_SHA256 or expected_dataset_sha256 != FROZEN_WISIG_SHA256:
        raise ICMTPostfreezePairError(f"{label} WiSig bytes do not bind the frozen SHA256")
    declared_dataset_sha256 = str(
        manifest.get("checkpoint_declared_wisig_pkl_sha256", "")
    ).lower()
    if declared_dataset_sha256 and declared_dataset_sha256 != FROZEN_WISIG_SHA256:
        raise ICMTPostfreezePairError(
            f"{label} checkpoint-declared WiSig SHA256 conflicts with frozen bytes"
        )
    if manifest.get("checkpoint_declared_wisig_pkl_sha256_empty_caveat") is not (
        not bool(declared_dataset_sha256)
    ):
        raise ICMTPostfreezePairError(f"{label} checkpoint dataset-hash caveat is not bound")
    if str(receipt.get("wisig_pkl_sha256", "")).lower() != declared_dataset_sha256:
        raise ICMTPostfreezePairError(
            f"{label} checkpoint-declared WiSig SHA256 does not bind source split receipt"
        )
    expected_hashes = {
        "labeled_indices_sha256": _index_sha256(labeled_indices.tolist()),
        "source_validation_indices_sha256": _index_sha256(validation_indices.tolist()),
    }
    for field, observed in expected_hashes.items():
        if str(manifest.get(field, "")) != observed or str(receipt.get(field, "")) != observed:
            raise ICMTPostfreezePairError(f"{label} {field} does not bind exported rows")
    if str(manifest.get("icmt_source_labeled_indices_sha256", "")) != expected_hashes[
        "labeled_indices_sha256"
    ]:
        raise ICMTPostfreezePairError(f"{label} P1-ICMT labeled-index SHA256 drifted")
    split_manifest_sha = str(receipt.get("split_manifest_sha256", ""))
    if len(split_manifest_sha) != 64 or str(
        manifest.get("icmt_source_split_manifest_sha256", "")
    ) != split_manifest_sha:
        raise ICMTPostfreezePairError(f"{label} P1-ICMT source-split manifest SHA256 drifted")
    unlabeled_hash = str(manifest.get("unlabeled_indices_sha256", ""))
    if len(unlabeled_hash) != 64 or str(receipt.get("unlabeled_indices_sha256", "")) != unlabeled_hash:
        raise ICMTPostfreezePairError(f"{label} U index hash is not receipt-bound")
    counts = {
        "labeled_fit": int(labeled_mask.sum()),
        "source_validation_known": int(validation_mask.sum()),
        "proxy_unknown": int(proxy_mask.sum()),
    }
    if counts["labeled_fit"] != int(manifest.get("labeled_row_count", -1)):
        raise ICMTPostfreezePairError(f"{label} labeled row count mismatch")
    if counts["source_validation_known"] != int(manifest.get("source_validation_row_count", -1)):
        raise ICMTPostfreezePairError(f"{label} validation row count mismatch")
    if int(manifest.get("unlabeled_row_count", 0)) <= 0:
        raise ICMTPostfreezePairError(f"{label} reconstructed U count must be positive")
    if counts["proxy_unknown"] != int(expected_proxy_count) or counts["proxy_unknown"] != int(
        manifest.get("proxy_row_count", -1)
    ):
        raise ICMTPostfreezePairError(f"{label} proxy row count mismatch")
    for tx_id in source_tx_ids:
        if int(np.sum(payload["tx_ids"][labeled_mask] == tx_id)) <= 1:
            raise ICMTPostfreezePairError(f"{label} labeled_fit requires n_c>1 for {tx_id}")
        if int(np.sum(payload["tx_ids"][validation_mask] == tx_id)) <= 0:
            raise ICMTPostfreezePairError(f"{label} source_validation_known is empty for {tx_id}")
    physical = _cb._require_unique_physical_keys(payload, label=label)
    labeled_physical = physical[labeled_mask]
    validation_physical = physical[validation_mask]
    proxy_physical = physical[proxy_mask]
    if (
        set(labeled_physical.tolist()) & set(validation_physical.tolist())
        or set(labeled_physical.tolist()) & set(proxy_physical.tolist())
        or set(validation_physical.tolist()) & set(proxy_physical.tolist())
    ):
        raise ICMTPostfreezePairError(f"{label} L/V/proxy physical keys overlap")
    physical_hashes = {
        "labeled_physical_keys_sha256": _canonical_json_sha256(labeled_physical.tolist()),
        "source_validation_physical_keys_sha256": _canonical_json_sha256(validation_physical.tolist()),
        "proxy_physical_keys_sha256": _canonical_json_sha256(proxy_physical.tolist()),
    }
    for field, observed in physical_hashes.items():
        if str(manifest.get(field, "")) != observed:
            raise ICMTPostfreezePairError(f"{label} {field} does not bind exported rows")
    proxy_physical_receipt = {
        "row_count": int(proxy_physical.size),
        "unique_count": len(set(proxy_physical.tolist())),
        "ordered_sha256": _canonical_json_sha256(proxy_physical.tolist()),
        "set_sha256": _canonical_json_sha256(sorted(proxy_physical.tolist())),
    }
    if proxy_physical_receipt["row_count"] != FROZEN_PROXY_TOTAL_COUNT or proxy_physical_receipt[
        "unique_count"
    ] != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"{label} proxy physical count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
        )
    if manifest.get("proxy_physical_key_receipt") != proxy_physical_receipt:
        raise ICMTPostfreezePairError(f"{label} proxy physical-key receipt drifted")
    return {
        "labeled_mask": labeled_mask,
        "validation_mask": validation_mask,
        "proxy_mask": proxy_mask,
        "labeled_indices": labeled_indices,
        "validation_indices": validation_indices,
        "labeled_physical": labeled_physical,
        "validation_physical": validation_physical,
        "proxy_physical": proxy_physical,
    }


def _normalize_float64(features: Any, *, label: str) -> np.ndarray:
    """Apply the exact totalized row-L2 map without eps or row deletion."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ICMTPostfreezePairError(f"{label} features must be non-empty rank-2")
    if not np.isfinite(values).all():
        raise ICMTPostfreezePairError(f"{label} features contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all():
        raise ICMTPostfreezePairError(f"{label} features produce a non-finite L2 norm")
    normalized = np.zeros_like(values, dtype=np.float64)
    positive = norms > 0.0
    normalized[positive] = values[positive] / norms[positive, None]
    return normalized


def _row_norm_stats(features: Any, *, label: str) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ICMTPostfreezePairError(f"{label} features must be non-empty rank-2")
    finite_rows = np.all(np.isfinite(values), axis=1)
    norms = np.full(values.shape[0], np.nan, dtype=np.float64)
    norms[finite_rows] = np.linalg.norm(values[finite_rows], axis=1)
    finite_norm_rows = finite_rows & np.isfinite(norms)
    nonfinite_rows = ~finite_norm_rows
    zero_rows = finite_norm_rows & (norms == 0.0)
    positive_rows = finite_norm_rows & (norms > 0.0)
    total = int(values.shape[0])
    nonfinite = int(nonfinite_rows.sum())
    zero = int(zero_rows.sum())
    positive = int(positive_rows.sum())
    if positive + zero + nonfinite != total:
        raise ICMTPostfreezePairError(f"{label} norm counts do not close")
    if nonfinite != 0:
        raise ICMTPostfreezePairError(f"{label} contains non-finite feature rows or norms")
    return {
        "total_rows": total,
        "positive_norm_rows": positive,
        "zero_norm_rows": zero,
        "nonfinite_rows": nonfinite,
        "retained_rows": total,
        "dropped_rows": 0,
        "count_closed": True,
    }


def _feature_norm_receipt(
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    role_masks = {
        "labeled_fit": np.asarray(binding["labeled_mask"], dtype=bool),
        "source_validation_known": np.asarray(binding["validation_mask"], dtype=bool),
        "proxy_unknown": np.asarray(binding["proxy_mask"], dtype=bool),
    }
    row_membership = np.stack(list(role_masks.values()), axis=0).sum(axis=0)
    if row_membership.shape[0] != int(payload["features"].shape[0]) or not np.all(row_membership == 1):
        raise ICMTPostfreezePairError(f"{label} feature norm role masks do not partition all rows")
    roles = {
        role: _row_norm_stats(payload["features"][mask], label=f"{label} {role}")
        for role, mask in role_masks.items()
    }
    per_class: dict[str, dict[str, Any]] = {}
    labeled_mask = role_masks["labeled_fit"]
    for tx_id in source_tx_ids:
        mask = labeled_mask & np.asarray(payload["tx_ids"] == tx_id, dtype=bool)
        stats = _row_norm_stats(payload["features"][mask], label=f"{label} labeled_fit {tx_id}")
        if int(stats["total_rows"]) <= GEOMETRY_DDOF:
            raise ICMTPostfreezePairError(
                f"{label} labeled_fit per-class total must exceed ddof=1 for {tx_id}"
            )
        per_class[str(tx_id)] = stats
    clean_total = int(payload["features"].shape[0])
    role_total = sum(int(stats["total_rows"]) for stats in roles.values())
    total_zero = sum(int(stats["zero_norm_rows"]) for stats in roles.values())
    total_nonfinite = sum(int(stats["nonfinite_rows"]) for stats in roles.values())
    labeled = roles["labeled_fit"]
    if (
        role_total != clean_total
        or sum(int(stats["total_rows"]) for stats in per_class.values()) != int(labeled["total_rows"])
        or sum(int(stats["zero_norm_rows"]) for stats in per_class.values())
        != int(labeled["zero_norm_rows"])
    ):
        raise ICMTPostfreezePairError(f"{label} feature norm receipt counts do not close")
    return {
        "normalization_rule": "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS",
        "zero_map_continuous_at_origin": False,
        "epsilon_used": False,
        "threshold_used": False,
        "topk_used": False,
        "fixed_zero_penalty_used": False,
        "rows_deleted": False,
        "roles": roles,
        "labeled_fit_per_class": per_class,
        "clean_total_rows": clean_total,
        "role_total_rows_sum": role_total,
        "total_zero_norm_rows": total_zero,
        "role_zero_norm_rows_sum": total_zero,
        "total_nonfinite_rows": total_nonfinite,
        "role_nonfinite_rows_sum": total_nonfinite,
        "retained_rows": clean_total,
        "dropped_rows": 0,
        "counts_closed": True,
    }


def fit_frozen_diagonal_gaussian(
    features: Any,
    tx_ids: Sequence[Any],
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit the frozen local4 float64 diagonal Gaussian from labelled L rows."""

    normalized = _normalize_float64(features, label="geometry fit")
    labels = np.asarray([str(value) for value in tx_ids], dtype=object).reshape(-1)
    class_order = tuple(str(value) for value in source_tx_ids)
    if len(class_order) != 4 or len(set(class_order)) != 4:
        raise ICMTPostfreezePairError("geometry fit requires a duplicate-free local4 class order")
    if labels.size != normalized.shape[0] or set(labels.tolist()) != set(class_order):
        raise ICMTPostfreezePairError("geometry fit labels must contain all and only local4 source TX")
    means: list[np.ndarray] = []
    raw_variances: list[np.ndarray] = []
    class_counts: dict[str, int] = {}
    for tx_id in class_order:
        rows = normalized[labels == tx_id]
        count = int(rows.shape[0])
        if count <= GEOMETRY_DDOF:
            raise ICMTPostfreezePairError("geometry fit requires n_c>ddof=1 for every local4 class")
        mean = np.mean(rows, axis=0, dtype=np.float64)
        variance = np.sum(np.square(rows - mean), axis=0, dtype=np.float64) / float(count - 1)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ICMTPostfreezePairError("geometry fit statistics are non-finite")
        means.append(mean)
        raw_variances.append(variance)
        class_counts[tx_id] = count
    mean_array = np.stack(means, axis=0).astype(np.float64, copy=False)
    raw_variance_array = np.stack(raw_variances, axis=0).astype(np.float64, copy=False)
    pooled_variance = np.mean(raw_variance_array, axis=0, dtype=np.float64)
    variances = np.maximum(
        GEOMETRY_VARIANCE_FLOOR,
        (1.0 - GEOMETRY_SHRINKAGE) * raw_variance_array
        + GEOMETRY_SHRINKAGE * pooled_variance[None, :],
    )
    if not np.isfinite(variances).all() or np.any(variances < GEOMETRY_VARIANCE_FLOOR):
        raise ICMTPostfreezePairError("geometry shrunk variances are invalid")
    return {
        "class_order": class_order,
        "class_counts": class_counts,
        "means": mean_array,
        "raw_variances": raw_variance_array,
        "pooled_variance": pooled_variance,
        "variances": variances,
    }


def score_frozen_icmt_nll(features: Any, geometry: Mapping[str, Any]) -> np.ndarray:
    """Return continuous u=log(4)-logsumexp(-NLL) with no threshold."""

    normalized = _normalize_float64(features, label="geometry score")
    means = np.asarray(geometry.get("means"), dtype=np.float64)
    variances = np.asarray(geometry.get("variances"), dtype=np.float64)
    if means.ndim != 2 or means.shape[0] != 4 or variances.shape != means.shape:
        raise ICMTPostfreezePairError("geometry score received an invalid local4 geometry shape")
    if means.shape[1] != normalized.shape[1]:
        raise ICMTPostfreezePairError("geometry score feature dimension does not match fit")
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ICMTPostfreezePairError("geometry score received non-finite parameters")
    if np.any(variances < GEOMETRY_VARIANCE_FLOOR):
        raise ICMTPostfreezePairError("geometry score variance is below the frozen floor")
    difference = normalized[:, None, :] - means[None, :, :]
    nll = 0.5 * np.sum(
        np.square(difference) / variances[None, :, :]
        + np.log(2.0 * math.pi * variances)[None, :, :],
        axis=2,
        dtype=np.float64,
    )
    neg_nll = -nll
    maximum = np.max(neg_nll, axis=1)
    logsumexp = maximum + np.log(
        np.sum(np.exp(neg_nll - maximum[:, None]), axis=1, dtype=np.float64)
    )
    score = math.log(4.0) - logsumexp
    if not np.isfinite(nll).all() or not np.isfinite(score).all():
        raise ICMTPostfreezePairError("geometry NLL or continuous u is non-finite")
    return np.asarray(score, dtype=np.float64)


def _auroc_unknown(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    known = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    if known.size <= 0 or unknown.size <= 0:
        raise ICMTPostfreezePairError("proxy AUROC requires non-empty known-heldout and proxy rows")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ICMTPostfreezePairError("proxy AUROC scores are non-finite")
    greater = unknown[:, None] > known[None, :]
    equal = unknown[:, None] == known[None, :]
    return float(np.mean(greater, dtype=np.float64) + 0.5 * np.mean(equal, dtype=np.float64))


def _geometry_sha256(geometry: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(geometry["class_order"]), separators=(",", ":")).encode("utf-8"))
    for field in ("means", "raw_variances", "pooled_variance", "variances"):
        values = np.ascontiguousarray(np.asarray(geometry[field], dtype="<f8"))
        digest.update(field.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _continuous_proxy_diagnostic(
    clean_payload: Mapping[str, Any],
    clean_binding: Mapping[str, Any],
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    labeled_mask = np.asarray(clean_binding["labeled_mask"], dtype=bool)
    validation_mask = np.asarray(clean_binding["validation_mask"], dtype=bool)
    proxy_mask = np.asarray(clean_binding["proxy_mask"], dtype=bool)
    geometry = fit_frozen_diagonal_gaussian(
        clean_payload["features"][labeled_mask],
        clean_payload["tx_ids"][labeled_mask],
        source_tx_ids,
    )
    held_u = score_frozen_icmt_nll(clean_payload["features"][validation_mask], geometry)
    proxy_u = score_frozen_icmt_nll(clean_payload["features"][proxy_mask], geometry)
    held_mean = float(np.mean(held_u, dtype=np.float64))
    proxy_mean = float(np.mean(proxy_u, dtype=np.float64))
    gap = proxy_mean - held_mean
    auroc = _auroc_unknown(held_u, proxy_u)
    if not all(math.isfinite(value) for value in (held_mean, proxy_mean, gap, auroc)):
        raise ICMTPostfreezePairError("continuous proxy diagnostic is non-finite")
    return {
        "fit": {
            "role": "labeled_fit",
            "row_count": int(labeled_mask.sum()),
            "class_counts": dict(geometry["class_counts"]),
            "feature_dimension": int(np.asarray(geometry["means"]).shape[1]),
            "normalization": "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS",
            "ddof": GEOMETRY_DDOF,
            "variance_shrinkage": GEOMETRY_SHRINKAGE,
            "variance_floor": GEOMETRY_VARIANCE_FLOOR,
            "geometry_sha256": _geometry_sha256(geometry),
            "source_validation_fit_rows": 0,
            "proxy_unknown_fit_rows": 0,
            "leo_fit_rows": 0,
            "unlabeled_fit_rows": 0,
        },
        "known_heldout": {
            "role": "source_validation_known",
            "count": int(held_u.size),
            "mean_u": held_mean,
            "min_u": float(np.min(held_u)),
            "max_u": float(np.max(held_u)),
        },
        "proxy_unknown": {
            "role": "proxy_unknown",
            "count": int(proxy_u.size),
            "mean_u": proxy_mean,
            "min_u": float(np.min(proxy_u)),
            "max_u": float(np.max(proxy_u)),
        },
        "AUROC_unknown": auroc,
        "proxy_minus_known_heldout_mean_u": gap,
        "score_rule": "LOG4_MINUS_LOGSUMEXP_NEGATIVE_DIAGONAL_GAUSSIAN_NLL",
        "threshold_used": False,
    }


def _continuous_proxy_guardrail(
    c_diagnostic: Mapping[str, Any], g_diagnostic: Mapping[str, Any]
) -> dict[str, Any]:
    c_auroc = float(c_diagnostic["AUROC_unknown"])
    g_auroc = float(g_diagnostic["AUROC_unknown"])
    c_gap = float(c_diagnostic["proxy_minus_known_heldout_mean_u"])
    g_gap = float(g_diagnostic["proxy_minus_known_heldout_mean_u"])
    auroc_delta = g_auroc - c_auroc
    gap_delta = g_gap - c_gap
    auroc_passed = auroc_delta > 0.0
    gap_passed = gap_delta > 0.0
    return {
        "C": dict(c_diagnostic),
        "G": dict(g_diagnostic),
        "G_minus_C": {
            "AUROC_unknown": auroc_delta,
            "proxy_minus_known_heldout_mean_u": gap_delta,
        },
        "strict_AUROC_improvement": auroc_passed,
        "strict_proxy_known_gap_improvement": gap_passed,
        "passed": bool(auroc_passed and gap_passed),
        "diagnostic_only_non_compensating": True,
    }


def _fold_gates(
    clean_delta: Mapping[str, Any],
    leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any],
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    clean = _cb._floor_gate(clean_delta)
    scenario_floor = {
        scenario: _cb._floor_gate(leo_scenarios[scenario]["G_minus_C_pp"])
        for scenario in expected_scenarios
    }
    fold_equal_overall = float(
        np.mean(
            np.asarray(
                [
                    float(leo_scenarios[scenario]["G_minus_C_pp"]["overall_accuracy"])
                    for scenario in expected_scenarios
                ],
                dtype=np.float64,
            )
        )
    )
    leo_floor_passed = bool(all(gate["passed"] for gate in scenario_floor.values()))
    fold_overall_passed = fold_equal_overall >= 0.0
    passed = bool(clean["passed"] and leo_floor_passed and fold_overall_passed and proxy_guardrail["passed"])
    return {
        "technical_binding": {"passed": True},
        "clean_four_floors_ge_minus2pp": clean,
        "leo_scenario_four_floors_ge_minus2pp": {
            "by_scenario": scenario_floor,
            "passed": leo_floor_passed,
        },
        "fold_three_scenario_equal_weight_overall_delta_pp": {
            "value": fold_equal_overall,
            "passed": fold_overall_passed,
        },
        "proxy_continuous_two_strict_improvements": dict(proxy_guardrail),
        "fold_verdict": "PENDING_GLOBAL_18_GRID" if passed else "REJECT_P1_ICMT_PERMANENT",
    }


def _as_fold_index(record: Mapping[str, Any], *, label: str) -> int:
    try:
        fold_index = int(record["fold_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ICMTPostfreezePairError(f"{label} lacks a valid fold_index") from exc
    if fold_index not in range(1, 7):
        raise ICMTPostfreezePairError(f"{label} fold_index must be in [1,6]")
    return fold_index


def _validate_proxy_receipt(proxy: Mapping[str, Any], *, label: str) -> None:
    try:
        c_auroc = float(proxy["C"]["AUROC_unknown"])
        g_auroc = float(proxy["G"]["AUROC_unknown"])
        c_gap = float(proxy["C"]["proxy_minus_known_heldout_mean_u"])
        g_gap = float(proxy["G"]["proxy_minus_known_heldout_mean_u"])
        recorded_auroc_delta = float(proxy["G_minus_C"]["AUROC_unknown"])
        recorded_gap_delta = float(proxy["G_minus_C"]["proxy_minus_known_heldout_mean_u"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ICMTPostfreezePairError(f"{label} continuous proxy receipt is malformed") from exc
    values = (c_auroc, g_auroc, c_gap, g_gap, recorded_auroc_delta, recorded_gap_delta)
    if not all(math.isfinite(value) for value in values) or not (0.0 <= c_auroc <= 1.0 and 0.0 <= g_auroc <= 1.0):
        raise ICMTPostfreezePairError(f"{label} continuous proxy receipt has invalid values")
    auroc_delta = g_auroc - c_auroc
    gap_delta = g_gap - c_gap
    if recorded_auroc_delta != auroc_delta or recorded_gap_delta != gap_delta:
        raise ICMTPostfreezePairError(f"{label} continuous proxy deltas are not exactly bound")
    auroc_passed = auroc_delta > 0.0
    gap_passed = gap_delta > 0.0
    if proxy.get("strict_AUROC_improvement") is not auroc_passed:
        raise ICMTPostfreezePairError(f"{label} strict AUROC receipt is not bound")
    if proxy.get("strict_proxy_known_gap_improvement") is not gap_passed:
        raise ICMTPostfreezePairError(f"{label} strict proxy-known gap receipt is not bound")
    if proxy.get("passed") is not bool(auroc_passed and gap_passed):
        raise ICMTPostfreezePairError(f"{label} continuous proxy passed receipt is not bound")
    for arm in ("C", "G"):
        diagnostic = proxy.get(arm)
        fit = diagnostic.get("fit") if isinstance(diagnostic, Mapping) else None
        if not isinstance(fit, Mapping) or fit.get("role") != "labeled_fit":
            raise ICMTPostfreezePairError(f"{label} {arm} geometry fit role is not labeled_fit")
        if fit.get("normalization") != "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS":
            raise ICMTPostfreezePairError(
                f"{label} {arm} geometry normalization is not frozen totalized L2"
            )
        for field in (
            "source_validation_fit_rows",
            "proxy_unknown_fit_rows",
            "leo_fit_rows",
            "unlabeled_fit_rows",
        ):
            if type(fit.get(field)) is not int or int(fit[field]) != 0:
                raise ICMTPostfreezePairError(f"{label} {arm} geometry {field} is not zero")
        if diagnostic.get("threshold_used") is not False:
            raise ICMTPostfreezePairError(f"{label} {arm} geometry used a threshold")
        known = diagnostic.get("known_heldout")
        if not isinstance(known, Mapping) or known.get("role") != "source_validation_known":
            raise ICMTPostfreezePairError(f"{label} {arm} known role is not source_validation_known")
        proxy_unknown = diagnostic.get("proxy_unknown")
        if not isinstance(proxy_unknown, Mapping) or proxy_unknown.get("role") != "proxy_unknown":
            raise ICMTPostfreezePairError(f"{label} {arm} proxy role is not proxy_unknown")
        if _strict_count(
            proxy_unknown, "count", label=f"{label} {arm} proxy diagnostic"
        ) != FROZEN_PROXY_TOTAL_COUNT:
            raise ICMTPostfreezePairError(
                f"{label} {arm} proxy diagnostic count must equal frozen "
                f"{FROZEN_PROXY_TOTAL_COUNT}"
            )


def _strict_count(mapping: Mapping[str, Any], field: str, *, label: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or int(value) < 0:
        raise ICMTPostfreezePairError(f"{label} {field} is not a non-negative integer")
    return int(value)


def _validate_norm_stats(
    stats: Mapping[str, Any], expected_total: int, *, label: str
) -> tuple[int, int]:
    total = _strict_count(stats, "total_rows", label=label)
    positive = _strict_count(stats, "positive_norm_rows", label=label)
    zero = _strict_count(stats, "zero_norm_rows", label=label)
    nonfinite = _strict_count(stats, "nonfinite_rows", label=label)
    retained = _strict_count(stats, "retained_rows", label=label)
    dropped = _strict_count(stats, "dropped_rows", label=label)
    if total != int(expected_total):
        raise ICMTPostfreezePairError(f"{label} total_rows does not bind expected role count")
    if positive + zero + nonfinite != total:
        raise ICMTPostfreezePairError(f"{label} positive/zero/nonfinite counts do not close")
    if nonfinite != 0 or retained != total or dropped != 0 or stats.get("count_closed") is not True:
        raise ICMTPostfreezePairError(f"{label} retention or finite-count contract drifted")
    return zero, nonfinite


def _validate_feature_norm_receipt(record: Mapping[str, Any], *, label: str) -> None:
    receipt = record.get("feature_norm_receipt")
    expected_counts = record.get("expected_role_counts")
    source_tx_ids = tuple(str(item) for item in record.get("source_tx_ids", []))
    if not isinstance(receipt, Mapping) or set(receipt) != {"C", "G"}:
        raise ICMTPostfreezePairError(f"{label} feature norm receipt lacks exact C/G arms")
    if not isinstance(expected_counts, Mapping) or len(source_tx_ids) != 4:
        raise ICMTPostfreezePairError(f"{label} feature norm receipt lacks role/class expectations")
    role_expected = {
        role: _strict_count(expected_counts, role, label=f"{label} expected_role_counts")
        for role in ("labeled_fit", "source_validation_known", "proxy_unknown")
    }
    expected_clean_total = sum(role_expected.values())
    for arm in ("C", "G"):
        arm_receipt = receipt.get(arm)
        if not isinstance(arm_receipt, Mapping):
            raise ICMTPostfreezePairError(f"{label} feature norm receipt {arm} is malformed")
        exact_policy = {
            "normalization_rule": "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS",
            "zero_map_continuous_at_origin": False,
            "epsilon_used": False,
            "threshold_used": False,
            "topk_used": False,
            "fixed_zero_penalty_used": False,
            "rows_deleted": False,
            "counts_closed": True,
        }
        for field, expected in exact_policy.items():
            if arm_receipt.get(field) != expected or type(arm_receipt.get(field)) is not type(expected):
                raise ICMTPostfreezePairError(
                    f"{label} feature norm receipt {arm} policy {field} drifted"
                )
        roles = arm_receipt.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != set(role_expected):
            raise ICMTPostfreezePairError(f"{label} feature norm receipt {arm} roles drifted")
        role_zero_sum = 0
        role_nonfinite_sum = 0
        for role, expected_total in role_expected.items():
            stats = roles.get(role)
            if not isinstance(stats, Mapping):
                raise ICMTPostfreezePairError(
                    f"{label} feature norm receipt {arm}/{role} is malformed"
                )
            zero, nonfinite = _validate_norm_stats(
                stats, expected_total, label=f"{label} feature norm receipt {arm}/{role}"
            )
            role_zero_sum += zero
            role_nonfinite_sum += nonfinite
        per_class = arm_receipt.get("labeled_fit_per_class")
        if not isinstance(per_class, Mapping) or set(per_class) != set(source_tx_ids):
            raise ICMTPostfreezePairError(
                f"{label} feature norm receipt {arm} labelled-fit class order drifted"
            )
        class_total_sum = 0
        class_zero_sum = 0
        for tx_id in source_tx_ids:
            stats = per_class.get(tx_id)
            if not isinstance(stats, Mapping):
                raise ICMTPostfreezePairError(
                    f"{label} feature norm receipt {arm} labelled-fit {tx_id} is malformed"
                )
            class_total = _strict_count(
                stats, "total_rows", label=f"{label} feature norm receipt {arm}/labeled_fit/{tx_id}"
            )
            if class_total <= GEOMETRY_DDOF:
                raise ICMTPostfreezePairError(
                    f"{label} feature norm receipt {arm} labelled-fit {tx_id} total must exceed one"
                )
            class_zero, _ = _validate_norm_stats(
                stats,
                class_total,
                label=f"{label} feature norm receipt {arm}/labeled_fit/{tx_id}",
            )
            class_total_sum += class_total
            class_zero_sum += class_zero
        if class_total_sum != role_expected["labeled_fit"] or class_zero_sum != int(
            roles["labeled_fit"]["zero_norm_rows"]
        ):
            raise ICMTPostfreezePairError(
                f"{label} feature norm receipt {arm} labelled-fit per-class counts do not close"
            )
        top_level_counts = {
            "clean_total_rows": expected_clean_total,
            "role_total_rows_sum": expected_clean_total,
            "total_zero_norm_rows": role_zero_sum,
            "role_zero_norm_rows_sum": role_zero_sum,
            "total_nonfinite_rows": role_nonfinite_sum,
            "role_nonfinite_rows_sum": role_nonfinite_sum,
            "retained_rows": expected_clean_total,
            "dropped_rows": 0,
        }
        for field, expected in top_level_counts.items():
            if _strict_count(arm_receipt, field, label=f"{label} feature norm receipt {arm}") != expected:
                raise ICMTPostfreezePairError(
                    f"{label} feature norm receipt {arm} top-level {field} does not close"
                )


def _validate_pair_record_contract(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    label: str,
) -> int:
    fold_index = _as_fold_index(record, label=label)
    if record.get("schema") != EXPECTED_PAIR_SCHEMA:
        raise ICMTPostfreezePairError(f"{label} schema mismatch")
    if str(record.get("candidate_pair", "")) != f"F{fold_index}_C_vs_G":
        raise ICMTPostfreezePairError(f"{label} candidate_pair does not match frozen fold {fold_index}")
    if tuple(str(item) for item in record.get("source_tx_ids", [])) != FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise ICMTPostfreezePairError(f"{label} source TX order does not match frozen fold {fold_index}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise ICMTPostfreezePairError(f"{label} matrix_id mismatch")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise ICMTPostfreezePairError(f"{label} output root mismatch")
    if str(record.get("training_run_root", "")) != str(training_root):
        raise ICMTPostfreezePairError(f"{label} training root mismatch")
    expected_counts = record.get("expected_role_counts")
    if not isinstance(expected_counts, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks expected role counts")
    if _strict_count(
        expected_counts, "proxy_unknown", label=f"{label} expected_role_counts"
    ) != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"{label} expected proxy count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
        )
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks checkpoint bindings")
    if bindings.get("classification_head_contract") != EXPECTED_CLASSIFICATION_HEAD_CONTRACT:
        raise ICMTPostfreezePairError(f"{label} classification head contract mismatch")
    for arm in ("C", "G"):
        expected_candidate, expected_checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        candidate_field = f"{arm.lower()}_candidate"
        checkpoint_field = f"{arm.lower()}_final_checkpoint_path"
        if bindings.get(candidate_field) != expected_candidate:
            raise ICMTPostfreezePairError(
                f"{label} {candidate_field} does not match frozen fold {fold_index}"
            )
        checkpoint_value = str(bindings.get(checkpoint_field, "")).strip()
        if not checkpoint_value or Path(checkpoint_value).resolve() != expected_checkpoint:
            raise ICMTPostfreezePairError(
                f"{label} {checkpoint_field} does not match frozen {expected_candidate}"
            )
        candidate_dir = (output_root / expected_candidate).resolve()
        artifact_layout = {
            "clean_npz": candidate_dir / "icmt_clean_l_v_proxy_final_only.npz",
            "leo_npz": candidate_dir / "source_leo_final_only.npz",
            "leo_binding_json": candidate_dir / "source_leo_binding.json",
            "proxy_metrics_json": candidate_dir / "proxy_logits_open_set_metrics.json",
            "proxy_scores_csv": candidate_dir / "proxy_logits_open_set_scores.csv",
        }
        for kind, expected_path in artifact_layout.items():
            path_field = f"{arm.lower()}_{kind}_path"
            sha_field = f"{arm.lower()}_{kind}_sha256"
            observed_path = Path(str(bindings.get(path_field, ""))).resolve()
            if observed_path != expected_path.resolve() or not observed_path.is_file():
                raise ICMTPostfreezePairError(
                    f"{label} {path_field} does not match immutable candidate artifact"
                )
            if str(bindings.get(sha_field, "")) != _cb._sha256_file(observed_path):
                raise ICMTPostfreezePairError(
                    f"{label} {sha_field} does not match current artifact"
                )
        dataset_sha_field = f"{arm.lower()}_leo_dataset_sha256"
        if str(bindings.get(dataset_sha_field, "")) != FROZEN_WISIG_SHA256:
            raise ICMTPostfreezePairError(f"{label} {dataset_sha_field} is not frozen")
    policy = record.get("policy")
    if not isinstance(policy, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks policy receipt")
    if policy.get("geometry_fit_performed") is not True or policy.get("geometry_fit_role") != "labeled_fit":
        raise ICMTPostfreezePairError(f"{label} geometry fit policy is not L-only")
    if policy.get("normalization_rule") != "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS":
        raise ICMTPostfreezePairError(f"{label} totalized normalization policy drifted")
    if type(policy.get("zero_norm_rows_dropped")) is not int or int(
        policy["zero_norm_rows_dropped"]
    ) != 0:
        raise ICMTPostfreezePairError(f"{label} zero-norm row drop count is not zero")
    if policy.get("fixed_zero_penalty_used") is not False:
        raise ICMTPostfreezePairError(f"{label} fixed zero penalty must remain disabled")
    for field in ("calibration_performed", "threshold_used", "model_selection_performed", "checkpoint_weights_loaded"):
        if policy.get(field) is not False:
            raise ICMTPostfreezePairError(f"{label} policy {field} is not strictly false")
    for field in (
        "source_validation_fit_rows",
        "proxy_unknown_fit_rows",
        "leo_fit_rows",
        "unlabeled_fit_rows",
    ):
        if type(policy.get(field)) is not int or int(policy[field]) != 0:
            raise ICMTPostfreezePairError(f"{label} policy {field} is not strictly zero")
    gates = record.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or not isinstance(gates.get("technical_binding"), Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks technical binding receipt")
    if gates["technical_binding"].get("passed") is not True:
        raise ICMTPostfreezePairError(f"{label} technical binding is not strictly true")
    _validate_feature_norm_receipt(record, label=label)
    proxy = record.get("proxy_continuous_guardrail")
    if not isinstance(proxy, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks continuous proxy guardrail")
    _validate_proxy_receipt(proxy, label=label)
    return fold_index


def _load_prior_pair(
    path: str | Path,
    *,
    expected_scenarios: Sequence[str],
    source_sat_seed: int,
    matrix_id: str,
    output_root: Path,
    training_root: Path,
) -> dict[str, Any]:
    try:
        source = _cb._require_under_root(path, output_root, label="prior pair metrics JSON")
    except _cb.CBSFCEPostfreezePairError as exc:
        raise _translate_cb_error(exc) from exc
    if not source.is_file():
        raise ICMTPostfreezePairError(f"missing prior pair metrics JSON: {source}")
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ICMTPostfreezePairError(f"prior pair metrics JSON is invalid: {source}") from exc
    if not isinstance(record, dict):
        raise ICMTPostfreezePairError(f"prior pair metrics JSON must encode an object: {source}")
    if tuple(str(item) for item in record.get("expected_scenarios", [])) != tuple(expected_scenarios):
        raise ICMTPostfreezePairError(f"prior pair scenario contract mismatch: {source}")
    if int(record.get("source_sat_seed", -1)) != int(source_sat_seed):
        raise ICMTPostfreezePairError(f"prior pair satellite seed mismatch: {source}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise ICMTPostfreezePairError(f"prior pair matrix_id mismatch: {source}")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise ICMTPostfreezePairError(f"prior pair output root mismatch: {source}")
    if str(record.get("training_run_root", "")) != str(training_root):
        raise ICMTPostfreezePairError(f"prior pair training root mismatch: {source}")
    if record.get("matrix_aggregate") is not None:
        raise ICMTPostfreezePairError(f"prior pair must be a per-fold record, not an aggregate: {source}")
    record["_input_path"] = str(source)
    record["_input_sha256"] = _cb._sha256_file(source)
    return record


def _record_deltas(
    record: Mapping[str, Any], scenarios: Sequence[str], *, label: str
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    try:
        clean = record["clean_source_validation"]["G_minus_C_pp"]
        leo = record["leo_scenarios"]
    except (KeyError, TypeError) as exc:
        raise ICMTPostfreezePairError(f"{label} lacks classifier deltas") from exc
    clean_out: dict[str, float] = {}
    leo_out: dict[str, dict[str, float]] = {}
    for metric in CLASSIFICATION_METRICS:
        try:
            value = float(clean[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ICMTPostfreezePairError(f"{label} clean delta lacks {metric}") from exc
        if not math.isfinite(value):
            raise ICMTPostfreezePairError(f"{label} clean delta is non-finite for {metric}")
        clean_out[metric] = value
    for scenario in scenarios:
        try:
            delta = leo[scenario]["G_minus_C_pp"]
        except (KeyError, TypeError) as exc:
            raise ICMTPostfreezePairError(f"{label} lacks LEO scenario {scenario}") from exc
        leo_out[scenario] = {}
        for metric in CLASSIFICATION_METRICS:
            try:
                value = float(delta[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ICMTPostfreezePairError(
                    f"{label} LEO delta lacks {scenario}/{metric}"
                ) from exc
            if not math.isfinite(value):
                raise ICMTPostfreezePairError(
                    f"{label} LEO delta is non-finite for {scenario}/{metric}"
                )
            leo_out[scenario][metric] = value
    return clean_out, leo_out


def _require_prior_artifact(
    record: Mapping[str, Any],
    *,
    arm: str,
    kind: str,
    expected_path: Path,
    label: str,
) -> Path:
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ICMTPostfreezePairError(f"{label} lacks artifact bindings")
    prefix = arm.lower()
    path_field = f"{prefix}_{kind}_path"
    sha_field = f"{prefix}_{kind}_sha256"
    observed_path = Path(str(bindings.get(path_field, ""))).resolve()
    if observed_path != expected_path.resolve() or not observed_path.is_file():
        raise ICMTPostfreezePairError(f"{label} {path_field} does not bind immutable artifact")
    recorded_sha = str(bindings.get(sha_field, ""))
    current_sha = _cb._sha256_file(observed_path)
    if recorded_sha != current_sha:
        raise ICMTPostfreezePairError(f"{label} {sha_field} does not match current artifact")
    return observed_path


def _recompute_prior_pair_artifacts(
    record: Mapping[str, Any],
    *,
    output_root: Path,
    matrix_id: str,
    training_root: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    """Re-read one prior fold's sealed artifacts and recompute every verdict field."""

    fold_index = _validate_pair_record_contract(
        record,
        output_root=output_root,
        matrix_id=matrix_id,
        training_root=training_root,
        label="prior pair",
    )
    source_tx_ids = FROZEN_FOLD_SOURCE_TX[fold_index]
    expected_counts = record.get("expected_role_counts")
    if not isinstance(expected_counts, Mapping):
        raise ICMTPostfreezePairError("prior pair lacks expected role counts")
    expected_source_count = _strict_count(
        expected_counts, "leo_source", label="prior pair expected_role_counts"
    )
    recorded_proxy_count = _strict_count(
        expected_counts, "proxy_unknown", label="prior pair expected_role_counts"
    )
    if recorded_proxy_count != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTPostfreezePairError(
            f"prior pair expected proxy count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
        )
    expected_proxy_count = FROZEN_PROXY_TOTAL_COUNT
    expected_days = tuple(str(item) for item in record.get("expected_source_days", []))
    expected_rxs = tuple(str(item) for item in record.get("expected_source_rxs", []))
    if expected_days != EXPECTED_SOURCE_DAYS or expected_rxs != EXPECTED_SOURCE_RXS:
        raise ICMTPostfreezePairError("prior pair source day/RX contract drifted")
    source_sat_seed = int(record.get("source_sat_seed", -1))

    paths: dict[str, dict[str, Path]] = {"C": {}, "G": {}}
    for arm in ("C", "G"):
        candidate, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        candidate_dir = (output_root / candidate).resolve()
        paths[arm]["clean"] = _require_prior_artifact(
            record,
            arm=arm,
            kind="clean_npz",
            expected_path=candidate_dir / "icmt_clean_l_v_proxy_final_only.npz",
            label=f"prior F{fold_index}{arm}",
        )
        paths[arm]["leo"] = _require_prior_artifact(
            record,
            arm=arm,
            kind="leo_npz",
            expected_path=candidate_dir / "source_leo_final_only.npz",
            label=f"prior F{fold_index}{arm}",
        )
        paths[arm]["leo_binding_json"] = _require_prior_artifact(
            record,
            arm=arm,
            kind="leo_binding_json",
            expected_path=candidate_dir / "source_leo_binding.json",
            label=f"prior F{fold_index}{arm}",
        )
        paths[arm]["proxy_metrics_json"] = _require_prior_artifact(
            record,
            arm=arm,
            kind="proxy_metrics_json",
            expected_path=candidate_dir / "proxy_logits_open_set_metrics.json",
            label=f"prior F{fold_index}{arm}",
        )
        paths[arm]["proxy_scores_csv"] = _require_prior_artifact(
            record,
            arm=arm,
            kind="proxy_scores_csv",
            expected_path=candidate_dir / "proxy_logits_open_set_scores.csv",
            label=f"prior F{fold_index}{arm}",
        )
        bindings = record["bindings"]
        checkpoint_sha_field = f"{arm.lower()}_final_checkpoint_sha256"
        if str(bindings.get(checkpoint_sha_field, "")) != _cb._sha256_file(checkpoint):
            raise ICMTPostfreezePairError(
                f"prior F{fold_index}{arm} final checkpoint SHA256 drifted"
            )
        paths[arm]["checkpoint"] = checkpoint

    c_clean = _load_lv_npz(paths["C"]["clean"])
    g_clean = _load_lv_npz(paths["G"]["clean"])
    c_leo = _cb._load_npz(paths["C"]["leo"])
    g_leo = _cb._load_npz(paths["G"]["leo"])
    for arm, clean, leo in (("C", c_clean, c_leo), ("G", g_clean, g_leo)):
        candidate, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        _validate_payload_identity(
            clean,
            expected_checkpoint=checkpoint,
            expected_candidate=candidate,
            label=f"prior F{fold_index}{arm} clean",
        )
        _validate_payload_identity(
            leo,
            expected_checkpoint=checkpoint,
            expected_candidate=candidate,
            label=f"prior F{fold_index}{arm} LEO",
        )
    _cb._assert_pair_metadata(c_clean, g_clean, label=f"prior F{fold_index} clean")
    if not np.array_equal(c_clean["source_base_indices"], g_clean["source_base_indices"]):
        raise ICMTPostfreezePairError("prior C/G clean L/V/proxy index binding differs")
    _cb._assert_pair_metadata(c_leo, g_leo, label=f"prior F{fold_index} LEO")
    if int(c_clean["features"].shape[1]) != int(c_leo["features"].shape[1]):
        raise ICMTPostfreezePairError("prior C clean/LEO z_id dimension mismatch")
    if int(g_clean["features"].shape[1]) != int(g_leo["features"].shape[1]):
        raise ICMTPostfreezePairError("prior G clean/LEO z_id dimension mismatch")
    c_role = _validate_lv_payload(
        c_clean, source_tx_ids, fold_index, expected_proxy_count, label="C prior clean"
    )
    g_role = _validate_lv_payload(
        g_clean, source_tx_ids, fold_index, expected_proxy_count, label="G prior clean"
    )
    feature_norm_receipt = {
        "C": _feature_norm_receipt(c_clean, c_role, source_tx_ids, label="prior C clean"),
        "G": _feature_norm_receipt(g_clean, g_role, source_tx_ids, label="prior G clean"),
    }
    c_leo_keys = _cb._validate_leo_payload(
        c_leo,
        source_tx_ids,
        expected_source_count,
        expected_scenarios,
        expected_days,
        expected_rxs,
        source_sat_seed,
        label="prior C LEO",
    )
    g_leo_keys = _cb._validate_leo_payload(
        g_leo,
        source_tx_ids,
        expected_source_count,
        expected_scenarios,
        expected_days,
        expected_rxs,
        source_sat_seed,
        label="prior G LEO",
    )
    if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
        raise ICMTPostfreezePairError("prior C/G LEO physical key sets differ")
    for arm, clean, leo in (("C", c_clean, c_leo), ("G", g_clean, g_leo)):
        candidate, checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        _load_icmt_leo_binding(
            paths[arm]["leo_binding_json"],
            leo,
            clean,
            expected_npz=paths[arm]["leo"],
            expected_checkpoint=checkpoint,
            expected_candidate=candidate,
            fold_index=fold_index,
            arm=arm,
            source_tx_ids=source_tx_ids,
            training_root=training_root,
            output_root=output_root,
            label=f"prior F{fold_index}{arm} LEO",
        )
        clean_checkpoint_sha = _cb._checkpoint_sha256_from_manifest(
            clean, label=f"prior F{fold_index}{arm} clean"
        )
        if clean_checkpoint_sha != _cb._checkpoint_sha256_from_manifest(
            leo, label=f"prior F{fold_index}{arm} LEO"
        ):
            raise ICMTPostfreezePairError(
                f"prior F{fold_index}{arm} clean/LEO checkpoint SHA256 differs"
            )
        _cb._bind_final_checkpoint(checkpoint, clean_checkpoint_sha, label=f"prior {arm}")
        _validate_proxy_manifest_identity(
            paths[arm]["proxy_metrics_json"],
            expected_checkpoint=checkpoint,
            expected_candidate=candidate,
            label=f"prior F{fold_index}{arm}",
        )

    c_proxy = _load_proxy_binding(
        paths["C"]["proxy_metrics_json"],
        paths["C"]["proxy_scores_csv"],
        c_clean,
        source_tx_ids,
        int(c_role["validation_mask"].sum()),
        expected_proxy_count,
        label="prior C",
    )
    g_proxy = _load_proxy_binding(
        paths["G"]["proxy_metrics_json"],
        paths["G"]["proxy_scores_csv"],
        g_clean,
        source_tx_ids,
        int(g_role["validation_mask"].sum()),
        expected_proxy_count,
        label="prior G",
    )
    c_clean_summary = _cb._classification_summary(
        c_clean, np.asarray(c_role["validation_mask"], dtype=bool), source_tx_ids
    )
    g_clean_summary = _cb._classification_summary(
        g_clean, np.asarray(g_role["validation_mask"], dtype=bool), source_tx_ids
    )
    clean_summary = {
        "C": c_clean_summary,
        "G": g_clean_summary,
        "G_minus_C_pp": _cb._delta_pp(c_clean_summary, g_clean_summary),
    }
    scenario_metrics: dict[str, Any] = {}
    for scenario in expected_scenarios:
        c_summary = _cb._classification_summary(
            c_leo, np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool), source_tx_ids
        )
        g_summary = _cb._classification_summary(
            g_leo, np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool), source_tx_ids
        )
        scenario_metrics[scenario] = {
            "C": c_summary,
            "G": g_summary,
            "G_minus_C_pp": _cb._delta_pp(c_summary, g_summary),
        }
    proxy_guardrail = _continuous_proxy_guardrail(
        _continuous_proxy_diagnostic(c_clean, c_role, source_tx_ids),
        _continuous_proxy_diagnostic(g_clean, g_role, source_tx_ids),
    )
    gates = _fold_gates(
        clean_summary["G_minus_C_pp"], scenario_metrics, proxy_guardrail, expected_scenarios
    )
    recomputed = {
        "feature_norm_receipt": feature_norm_receipt,
        "clean_source_validation": clean_summary,
        "leo_scenarios": scenario_metrics,
        "proxy_continuous_guardrail": proxy_guardrail,
        "postfreeze_gates": gates,
        "legacy_logits_proxy_binding_only": {
            "C": {
                "AUROC_unknown": float(c_proxy["AUROC_unknown"]),
                "unknown_FAR": float(c_proxy["unknown_FAR"]),
            },
            "G": {
                "AUROC_unknown": float(g_proxy["AUROC_unknown"]),
                "unknown_FAR": float(g_proxy["unknown_FAR"]),
            },
            "used_for_verdict": False,
        },
    }
    for field, expected in recomputed.items():
        if _canonical_json_sha256(record.get(field)) != _canonical_json_sha256(expected):
            raise ICMTPostfreezePairError(
                f"prior F{fold_index} {field} does not match raw-artifact recomputation"
            )
    clean_delta, leo_delta = _record_deltas(
        recomputed, expected_scenarios, label=f"recomputed fold{fold_index}"
    )
    return {
        "fold_index": fold_index,
        "clean_delta": clean_delta,
        "leo_delta": leo_delta,
        "proxy": proxy_guardrail,
        "postfreeze_gates": gates,
        "raw_artifacts_recomputed": True,
    }


def _matrix_aggregate(
    current: Mapping[str, Any],
    prior_paths: Sequence[str],
    *,
    expected_scenarios: Sequence[str],
    output_root: Path,
    matrix_id: str,
    training_root: Path,
) -> dict[str, Any]:
    fold_index = _validate_pair_record_contract(
        current,
        output_root=output_root,
        matrix_id=matrix_id,
        training_root=training_root,
        label="current pair",
    )
    if fold_index != 6:
        raise ICMTPostfreezePairError("matrix aggregate is frozen to the sixth and final pair")
    if len(prior_paths) != 5:
        raise ICMTPostfreezePairError("sixth pair requires exactly five prior per-fold metrics JSONs")
    records = [
        _load_prior_pair(
            path,
            expected_scenarios=expected_scenarios,
            source_sat_seed=int(current["source_sat_seed"]),
            matrix_id=matrix_id,
            output_root=output_root,
            training_root=training_root,
        )
        for path in prior_paths
    ]
    records.append(dict(current))
    fold_indices = [
        _validate_pair_record_contract(
            record,
            output_root=output_root,
            matrix_id=matrix_id,
            training_root=training_root,
            label="pair record",
        )
        for record in records
    ]
    if set(fold_indices) != set(range(1, 7)) or len(set(fold_indices)) != len(fold_indices):
        raise ICMTPostfreezePairError("matrix aggregate must contain exactly folds 1..6 once")
    records.sort(key=lambda record: _as_fold_index(record, label="pair record"))
    prior_recomputed = {
        int(record["fold_index"]): _recompute_prior_pair_artifacts(
            record,
            output_root=output_root,
            matrix_id=matrix_id,
            training_root=training_root,
            expected_scenarios=expected_scenarios,
        )
        for record in records
        if "_input_path" in record
    }

    clean_passes: list[bool] = []
    leo_passes: list[bool] = []
    fold_equal_overall: dict[str, float] = {}
    technical_passes: list[bool] = []
    proxy_passes: list[bool] = []
    proxy_deltas: dict[str, dict[str, float]] = {}
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in CLASSIFICATION_METRICS}
    for record in records:
        fold = int(record["fold_index"])
        if fold in prior_recomputed:
            clean_delta = prior_recomputed[fold]["clean_delta"]
            leo_delta = prior_recomputed[fold]["leo_delta"]
            proxy = prior_recomputed[fold]["proxy"]
            gates = prior_recomputed[fold]["postfreeze_gates"]
        else:
            clean_delta, leo_delta = _record_deltas(
                record, expected_scenarios, label=f"fold{fold}"
            )
            proxy = record["proxy_continuous_guardrail"]
            gates = record["postfreeze_gates"]
        clean_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in clean_delta.values()))
        leo_values = [
            value for scenario in expected_scenarios for value in leo_delta[scenario].values()
        ]
        leo_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in leo_values))
        fold_value = float(
            np.mean(
                np.asarray(
                    [leo_delta[scenario]["overall_accuracy"] for scenario in expected_scenarios],
                    dtype=np.float64,
                )
            )
        )
        fold_key = f"F{fold}"
        fold_equal_overall[fold_key] = fold_value
        for scenario in expected_scenarios:
            for metric in CLASSIFICATION_METRICS:
                deltas_by_metric[metric].append(leo_delta[scenario][metric])
        technical_passes.append(gates["technical_binding"]["passed"] is True)
        auroc_delta = float(proxy["G_minus_C"]["AUROC_unknown"])
        gap_delta = float(proxy["G_minus_C"]["proxy_minus_known_heldout_mean_u"])
        proxy_deltas[fold_key] = {
            "AUROC_unknown": auroc_delta,
            "proxy_minus_known_heldout_mean_u": gap_delta,
        }
        proxy_passes.append(bool(auroc_delta > 0.0 and gap_delta > 0.0))

    global_18 = {
        metric: float(np.mean(np.asarray(values, dtype=np.float64)))
        for metric, values in deltas_by_metric.items()
    }
    technical_passed = bool(all(technical_passes))
    clean_passed = bool(all(clean_passes))
    leo_passed = bool(all(leo_passes))
    fold_overall_passed = bool(all(value >= 0.0 for value in fold_equal_overall.values()))
    global_overall_passed = global_18["overall_accuracy"] >= 0.0
    proxy_passed = bool(all(proxy_passes))
    passed = bool(
        technical_passed
        and clean_passed
        and leo_passed
        and fold_overall_passed
        and global_overall_passed
        and proxy_passed
    )
    prior_bindings = [
        {
            "fold_index": int(record["fold_index"]),
            "metrics_json": record["_input_path"],
            "sha256": record["_input_sha256"],
            "raw_artifacts_recomputed": True,
            "artifact_sha256": {
                field: value
                for field, value in record["bindings"].items()
                if field.endswith("_sha256")
            },
        }
        for record in records
        if "_input_path" in record
    ]
    return {
        "fold_indices": [int(record["fold_index"]) for record in records],
        "prior_pair_metrics_bindings": prior_bindings,
        "global_18_cell_equal_weight_G_minus_C_pp": global_18,
        "proxy_continuous_G_minus_C_by_fold": proxy_deltas,
        "gates": {
            "technical_binding": {"passed": technical_passed},
            "clean_6of6_four_floors_ge_minus2pp": {
                "passed": clean_passed,
                "by_fold": clean_passes,
            },
            "leo_18of18_four_floors_ge_minus2pp": {
                "passed": leo_passed,
                "by_fold": leo_passes,
            },
            "fold_three_scenario_equal_weight_overall_delta_pp": {
                "values": fold_equal_overall,
                "passed": fold_overall_passed,
            },
            "global_18_cell_equal_weight_overall_delta_pp": {
                "value": global_18["overall_accuracy"],
                "passed": global_overall_passed,
            },
            "proxy_continuous_6of6_two_strict_improvements": {
                "passed": proxy_passed,
                "by_fold": proxy_passes,
            },
        },
        "verdict": (
            "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
            if passed
            else "REJECT_P1_ICMT_PERMANENT"
        ),
        "phase3_unknown_capability_claim": "NOT_EVALUATED",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one immutable ICMT C/G fold; F6 also seals the six-fold matrix."""

    try:
        source_tx_ids = _cb._parse_items(args.source_tx_ids, field="source_tx_ids")
        if len(source_tx_ids) != 4:
            raise ICMTPostfreezePairError("P1-ICMT postfreeze is frozen to local4 source TX classes")
        fold_index = int(args.fold_index)
        if fold_index not in range(1, 7):
            raise ICMTPostfreezePairError("fold_index must be in [1,6]")
        if source_tx_ids != FROZEN_FOLD_SOURCE_TX[fold_index]:
            raise ICMTPostfreezePairError(f"source_tx_ids do not match frozen fold {fold_index}")
        if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
            raise ICMTPostfreezePairError(f"candidate_pair does not match frozen fold {fold_index}")
        matrix_id = str(args.postfreeze_matrix_id).strip()
        if not matrix_id:
            raise ICMTPostfreezePairError("postfreeze_matrix_id must be non-empty")
        output_root = _cb._canonical_root(args.postfreeze_output_root)
        training_root = _canonical_training_root(args.training_run_root)
        if training_root == output_root:
            raise ICMTPostfreezePairError("training run root must differ from postfreeze output root")
        c_candidate, expected_c_checkpoint = _expected_final_checkpoint(training_root, fold_index, "C")
        g_candidate, expected_g_checkpoint = _expected_final_checkpoint(training_root, fold_index, "G")
        c_final_checkpoint = _require_exact_final_checkpoint(
            args.c_final_checkpoint, expected_c_checkpoint, label="C"
        )
        g_final_checkpoint = _require_exact_final_checkpoint(
            args.g_final_checkpoint, expected_g_checkpoint, label="G"
        )
        for path, label in (
            (args.c_clean_npz, "C clean NPZ"),
            (args.g_clean_npz, "G clean NPZ"),
            (args.c_leo_npz, "C LEO NPZ"),
            (args.g_leo_npz, "G LEO NPZ"),
            (args.c_leo_binding_json, "C ICMT LEO binding JSON"),
            (args.g_leo_binding_json, "G ICMT LEO binding JSON"),
            (args.c_proxy_metrics_json, "C proxy metrics JSON"),
            (args.g_proxy_metrics_json, "G proxy metrics JSON"),
            (args.c_proxy_scores_csv, "C proxy scores CSV"),
            (args.g_proxy_scores_csv, "G proxy scores CSV"),
            (args.output_metrics_json, "pair output JSON"),
        ):
            _cb._require_under_root(path, output_root, label=label)
        expected_scenarios = _cb._parse_items(args.expected_scenarios, field="expected_scenarios")
        if expected_scenarios != EXPECTED_SCENARIOS:
            raise ICMTPostfreezePairError("expected_scenarios are frozen to the three leo_*_weak scenarios")
        expected_days = _cb._parse_items(args.expected_source_days, field="expected_source_days")
        expected_rxs = _cb._parse_items(args.expected_source_rxs, field="expected_source_rxs")
        if expected_days != EXPECTED_SOURCE_DAYS:
            raise ICMTPostfreezePairError("expected_source_days do not match the frozen WRC LEO v2 slice")
        if expected_rxs != EXPECTED_SOURCE_RXS:
            raise ICMTPostfreezePairError("expected_source_rxs do not match the frozen WRC LEO v2 slice")
        expected_source_count = int(args.expected_source_count)
        expected_proxy_count = int(args.expected_proxy_count)
        if expected_source_count <= 0:
            raise ICMTPostfreezePairError("expected source row count must be positive")
        if expected_proxy_count != FROZEN_PROXY_TOTAL_COUNT:
            raise ICMTPostfreezePairError(
                f"expected_proxy_count must equal frozen {FROZEN_PROXY_TOTAL_COUNT}"
            )
        prior_paths = (
            _cb._parse_items(
                args.aggregate_prior_pair_metrics_json,
                field="aggregate_prior_pair_metrics_json",
            )
            if args.aggregate_prior_pair_metrics_json
            else ()
        )
        if fold_index < 6 and prior_paths:
            raise ICMTPostfreezePairError("only the sixth pair may aggregate prior pair metrics")
        if fold_index == 6 and len(prior_paths) != 5:
            raise ICMTPostfreezePairError("sixth pair requires five prior pair metrics JSONs for the 18-cell gate")

        c_clean = _load_lv_npz(args.c_clean_npz)
        g_clean = _load_lv_npz(args.g_clean_npz)
        c_leo = _cb._load_npz(args.c_leo_npz)
        g_leo = _cb._load_npz(args.g_leo_npz)
        _validate_payload_identity(
            c_clean,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C clean",
        )
        _validate_payload_identity(
            c_leo,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C LEO",
        )
        _validate_payload_identity(
            g_clean,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G clean",
        )
        _validate_payload_identity(
            g_leo,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G LEO",
        )
        _cb._assert_pair_metadata(c_clean, g_clean, label="clean")
        if not np.array_equal(c_clean["source_base_indices"], g_clean["source_base_indices"]):
            raise ICMTPostfreezePairError("C/G clean L/V/proxy index binding differs")
        _cb._assert_pair_metadata(c_leo, g_leo, label="LEO")
        if int(c_clean["features"].shape[1]) != int(c_leo["features"].shape[1]):
            raise ICMTPostfreezePairError("C clean/LEO z_id dimension mismatch")
        if int(g_clean["features"].shape[1]) != int(g_leo["features"].shape[1]):
            raise ICMTPostfreezePairError("G clean/LEO z_id dimension mismatch")
        c_role_binding = _validate_lv_payload(
            c_clean,
            source_tx_ids,
            fold_index,
            expected_proxy_count,
            label="C clean",
        )
        g_role_binding = _validate_lv_payload(
            g_clean,
            source_tx_ids,
            fold_index,
            expected_proxy_count,
            label="G clean",
        )
        feature_norm_receipt = {
            "C": _feature_norm_receipt(
                c_clean, c_role_binding, source_tx_ids, label="C clean"
            ),
            "G": _feature_norm_receipt(
                g_clean, g_role_binding, source_tx_ids, label="G clean"
            ),
        }
        c_leo_keys = _cb._validate_leo_payload(
            c_leo,
            source_tx_ids,
            expected_source_count,
            expected_scenarios,
            expected_days,
            expected_rxs,
            int(args.source_sat_seed),
            label="C LEO",
        )
        g_leo_keys = _cb._validate_leo_payload(
            g_leo,
            source_tx_ids,
            expected_source_count,
            expected_scenarios,
            expected_days,
            expected_rxs,
            int(args.source_sat_seed),
            label="G LEO",
        )
        c_leo_binding = _load_icmt_leo_binding(
            args.c_leo_binding_json,
            c_leo,
            c_clean,
            expected_npz=Path(args.c_leo_npz).resolve(),
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            fold_index=fold_index,
            arm="C",
            source_tx_ids=source_tx_ids,
            training_root=training_root,
            output_root=output_root,
            label="C LEO",
        )
        g_leo_binding = _load_icmt_leo_binding(
            args.g_leo_binding_json,
            g_leo,
            g_clean,
            expected_npz=Path(args.g_leo_npz).resolve(),
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            fold_index=fold_index,
            arm="G",
            source_tx_ids=source_tx_ids,
            training_root=training_root,
            output_root=output_root,
            label="G LEO",
        )
        if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
            raise ICMTPostfreezePairError("C/G LEO source physical key sets differ")
        c_checkpoint_sha256 = _cb._checkpoint_sha256_from_manifest(c_clean, label="C clean")
        g_checkpoint_sha256 = _cb._checkpoint_sha256_from_manifest(g_clean, label="G clean")
        if c_checkpoint_sha256 != _cb._checkpoint_sha256_from_manifest(c_leo, label="C LEO"):
            raise ICMTPostfreezePairError("C clean/LEO source checkpoint SHA256 differs")
        if g_checkpoint_sha256 != _cb._checkpoint_sha256_from_manifest(g_leo, label="G LEO"):
            raise ICMTPostfreezePairError("G clean/LEO source checkpoint SHA256 differs")
        c_final_checkpoint_sha256 = _cb._bind_final_checkpoint(
            c_final_checkpoint, c_checkpoint_sha256, label="C"
        )
        g_final_checkpoint_sha256 = _cb._bind_final_checkpoint(
            g_final_checkpoint, g_checkpoint_sha256, label="G"
        )
        _validate_proxy_manifest_identity(
            args.c_proxy_metrics_json,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C",
        )
        _validate_proxy_manifest_identity(
            args.g_proxy_metrics_json,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G",
        )
        c_proxy_binding = _load_proxy_binding(
            args.c_proxy_metrics_json,
            args.c_proxy_scores_csv,
            c_clean,
            source_tx_ids,
            int(c_role_binding["validation_mask"].sum()),
            expected_proxy_count,
            label="C",
        )
        g_proxy_binding = _load_proxy_binding(
            args.g_proxy_metrics_json,
            args.g_proxy_scores_csv,
            g_clean,
            source_tx_ids,
            int(g_role_binding["validation_mask"].sum()),
            expected_proxy_count,
            label="G",
        )
        c_clean_summary = _cb._classification_summary(
            c_clean, np.asarray(c_role_binding["validation_mask"], dtype=bool), source_tx_ids
        )
        g_clean_summary = _cb._classification_summary(
            g_clean, np.asarray(g_role_binding["validation_mask"], dtype=bool), source_tx_ids
        )
        scenario_metrics: dict[str, Any] = {}
        for scenario in expected_scenarios:
            c_mask = np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool)
            g_mask = np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool)
            c_summary = _cb._classification_summary(c_leo, c_mask, source_tx_ids)
            g_summary = _cb._classification_summary(g_leo, g_mask, source_tx_ids)
            scenario_metrics[scenario] = {
                "C": c_summary,
                "G": g_summary,
                "G_minus_C_pp": _cb._delta_pp(c_summary, g_summary),
            }
        c_continuous = _continuous_proxy_diagnostic(c_clean, c_role_binding, source_tx_ids)
        g_continuous = _continuous_proxy_diagnostic(g_clean, g_role_binding, source_tx_ids)
        proxy_guardrail = _continuous_proxy_guardrail(c_continuous, g_continuous)
        clean_delta = _cb._delta_pp(c_clean_summary, g_clean_summary)
        postfreeze_gates = _fold_gates(
            clean_delta, scenario_metrics, proxy_guardrail, expected_scenarios
        )
        metrics: dict[str, Any] = {
            "schema": EXPECTED_PAIR_SCHEMA,
            "candidate_pair": str(args.candidate_pair),
            "fold_index": fold_index,
            "postfreeze_matrix_id": matrix_id,
            "postfreeze_output_root": str(output_root),
            "training_run_root": str(training_root),
            "evidence_boundary": "PHASE1_SOURCE_ONLY_FINAL_ONLY_CONTINUOUS_GEOMETRY_DIAGNOSTIC",
            "phase3_unknown_capability_claim": "NOT_EVALUATED",
            "frozen_contract": dict(FROZEN_POSTFREEZE_CONTRACT),
            "policy": {
                "geometry_fit_performed": True,
                "geometry_fit_role": "labeled_fit",
                "source_validation_fit_rows": 0,
                "proxy_unknown_fit_rows": 0,
                "leo_fit_rows": 0,
                "unlabeled_fit_rows": 0,
                "calibration_performed": False,
                "threshold_used": False,
                "model_selection_performed": False,
                "checkpoint_weights_loaded": False,
                "legacy_logits_proxy_metrics_used_for_verdict": False,
                "proxy_guardrail_non_compensating": True,
                "normalization_rule": "TOTALIZED_EXACT_ROW_L2_FLOAT64_POSITIVE_ELSE_ZERO_NO_EPS",
                "zero_norm_rows_dropped": 0,
                "fixed_zero_penalty_used": False,
            },
            "source_tx_ids": list(source_tx_ids),
            "outer_known_validation_tx_id": FROZEN_FOLD_KNOWN_HELDOUT_TX[fold_index],
            "proxy_unknown_tx_id": FROZEN_FOLD_PROXY_TX[fold_index],
            "expected_source_days": list(expected_days),
            "expected_source_rxs": list(expected_rxs),
            "expected_role_counts": {
                "leo_source": expected_source_count,
                "labeled_fit": int(c_role_binding["labeled_mask"].sum()),
                "source_validation_known": int(c_role_binding["validation_mask"].sum()),
                "proxy_unknown": expected_proxy_count,
            },
            "expected_scenarios": list(expected_scenarios),
            "source_sat_seed": int(args.source_sat_seed),
            "bindings": {
                "c_clean_npz_path": str(Path(args.c_clean_npz).resolve()),
                "g_clean_npz_path": str(Path(args.g_clean_npz).resolve()),
                "c_leo_npz_path": str(Path(args.c_leo_npz).resolve()),
                "g_leo_npz_path": str(Path(args.g_leo_npz).resolve()),
                "c_clean_npz_sha256": _cb._sha256_file(args.c_clean_npz),
                "g_clean_npz_sha256": _cb._sha256_file(args.g_clean_npz),
                "c_leo_npz_sha256": _cb._sha256_file(args.c_leo_npz),
                "g_leo_npz_sha256": _cb._sha256_file(args.g_leo_npz),
                "c_leo_binding_json_path": c_leo_binding["path"],
                "g_leo_binding_json_path": g_leo_binding["path"],
                "c_leo_binding_json_sha256": c_leo_binding["sha256"],
                "g_leo_binding_json_sha256": g_leo_binding["sha256"],
                "c_leo_dataset_sha256": c_leo_binding["dataset_sha256"],
                "g_leo_dataset_sha256": g_leo_binding["dataset_sha256"],
                "c_leo_selection_sha256": c_leo_binding["selection_sha256"],
                "g_leo_selection_sha256": g_leo_binding["selection_sha256"],
                "c_leo_physical_key_set_sha256": c_leo_binding[
                    "physical_key_set_sha256"
                ],
                "g_leo_physical_key_set_sha256": g_leo_binding[
                    "physical_key_set_sha256"
                ],
                "c_source_checkpoint_sha256": c_checkpoint_sha256,
                "g_source_checkpoint_sha256": g_checkpoint_sha256,
                "classification_head_contract": EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
                "c_candidate": c_candidate,
                "g_candidate": g_candidate,
                "c_final_checkpoint_path": str(c_final_checkpoint),
                "g_final_checkpoint_path": str(g_final_checkpoint),
                "c_final_checkpoint_sha256": c_final_checkpoint_sha256,
                "g_final_checkpoint_sha256": g_final_checkpoint_sha256,
                "c_proxy_metrics_json_path": str(Path(args.c_proxy_metrics_json).resolve()),
                "g_proxy_metrics_json_path": str(Path(args.g_proxy_metrics_json).resolve()),
                "c_proxy_metrics_json_sha256": c_proxy_binding["sha256"],
                "g_proxy_metrics_json_sha256": g_proxy_binding["sha256"],
                "c_proxy_scores_csv_path": c_proxy_binding["scores_csv_path"],
                "g_proxy_scores_csv_path": g_proxy_binding["scores_csv_path"],
                "c_proxy_scores_csv_sha256": c_proxy_binding["scores_csv_sha256"],
                "g_proxy_scores_csv_sha256": g_proxy_binding["scores_csv_sha256"],
                "checkpoint_weight_reading": "DISALLOWED",
            },
            "legacy_logits_proxy_binding_only": {
                "C": {
                    "AUROC_unknown": float(c_proxy_binding["AUROC_unknown"]),
                    "unknown_FAR": float(c_proxy_binding["unknown_FAR"]),
                },
                "G": {
                    "AUROC_unknown": float(g_proxy_binding["AUROC_unknown"]),
                    "unknown_FAR": float(g_proxy_binding["unknown_FAR"]),
                },
                "used_for_verdict": False,
            },
            "feature_norm_receipt": feature_norm_receipt,
            "clean_source_validation": {
                "C": c_clean_summary,
                "G": g_clean_summary,
                "G_minus_C_pp": clean_delta,
            },
            "leo_scenarios": scenario_metrics,
            "proxy_continuous_guardrail": proxy_guardrail,
            "postfreeze_gates": postfreeze_gates,
            "matrix_aggregate": None,
        }
        _validate_feature_norm_receipt(metrics, label="current pair")
        if fold_index == 6:
            metrics["matrix_aggregate"] = _matrix_aggregate(
                metrics,
                prior_paths,
                expected_scenarios=expected_scenarios,
                output_root=output_root,
                matrix_id=matrix_id,
                training_root=training_root,
            )
        _atomic_write_json(args.output_metrics_json, metrics)
        return metrics
    except _cb.CBSFCEPostfreezePairError as exc:
        raise _translate_cb_error(exc) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--c-leo-binding-json", required=True)
    parser.add_argument("--g-leo-binding-json", required=True)
    parser.add_argument("--c-final-checkpoint", required=True)
    parser.add_argument("--g-final-checkpoint", required=True)
    parser.add_argument("--c-proxy-metrics-json", required=True)
    parser.add_argument("--g-proxy-metrics-json", required=True)
    parser.add_argument("--c-proxy-scores-csv", required=True)
    parser.add_argument("--g-proxy-scores-csv", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--postfreeze-matrix-id", required=True)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument(
        "--expected-proxy-count", type=int, default=FROZEN_PROXY_TOTAL_COUNT
    )
    parser.add_argument("--aggregate-prior-pair-metrics-json", default="")
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
