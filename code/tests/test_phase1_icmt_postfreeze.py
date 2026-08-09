from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "evaluate_phase1_icmt_postfreeze_pair.py"
EXPORTER_PATH = CODE_ROOT / "export_phase1_icmt_features.py"
LEO_EXPORTER_PATH = CODE_ROOT / "export_phase1_icmt_leo_features.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_icmt_postfreeze_20260810.sh"
_SPEC = importlib.util.spec_from_file_location("icmt_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS
TEST_MATRIX_ID = "test_phase1_icmt_postfreeze_matrix_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_hot(index: int) -> np.ndarray:
    value = np.zeros(4, dtype=np.float32)
    value[int(index)] = 1.0
    return value


def _logits(tx: str, tx_ids: tuple[str, ...]) -> np.ndarray:
    values = np.full(4, -2.0, dtype=np.float32)
    values[tx_ids.index(tx)] = 4.0
    return values


def _physical_keys(rows: list[dict[str, str]]) -> list[str]:
    return [
        "\x1f".join(row[field] for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"))
        for row in rows
    ]


def _strict_load_manifest(checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "feature_name": "z_id",
        "checkpoint": str(checkpoint.resolve()),
        "classification_head_contract": PAIR.EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
        "class_id_to_tx": list(tx_ids),
        "logit_class_order": list(range(4)),
        "source_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": {
            "checkpoint_load_strict": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skipped_mismatch": 0,
        },
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "satellite_tta_policy": "none",
    }


def _clean_rows(tx_ids: tuple[str, ...], proxy_tx: str) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    indices: list[int] = []
    source_index = 0
    for role, per_class, offset in (("labeled_fit", 3, 0), ("source_validation_known", 2, 100)):
        for tx_index, tx in enumerate(tx_ids):
            for sample in range(per_class):
                rows.append(
                    {
                        "dataset_role": role,
                        "tx_ids": tx,
                        "rx_ids": RXS[(tx_index + sample) % len(RXS)],
                        "day_ids": DAYS[(tx_index + sample) % len(DAYS)],
                        "eq_ids": "1",
                        "sig_ids": f"{role}-{tx_index}-{sample}",
                        "channel_views": "clean",
                        "sat_scenarios": "",
                    }
                )
                indices.append(source_index if role == "labeled_fit" else offset + source_index - 12)
                source_index += 1
    for sample in range(PAIR.FROZEN_PROXY_TOTAL_COUNT):
        rows.append(
            {
                "dataset_role": "proxy_unknown",
                "tx_ids": proxy_tx,
                "rx_ids": RXS[sample % len(RXS)],
                "day_ids": DAYS[sample % len(DAYS)],
                "eq_ids": "1",
                "sig_ids": f"proxy-{sample}",
                "channel_views": "clean",
                "sat_scenarios": "",
            }
        )
        indices.append(-1 - sample)
    return rows, np.asarray(indices, dtype=np.int64)


def _clean_manifest(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    tx_ids: tuple[str, ...],
    fold: int,
    rows: list[dict[str, str]],
    source_base_indices: np.ndarray,
) -> dict[str, object]:
    labeled = np.asarray([row["dataset_role"] == "labeled_fit" for row in rows], dtype=bool)
    validation = np.asarray(
        [row["dataset_role"] == "source_validation_known" for row in rows], dtype=bool
    )
    proxy = np.asarray([row["dataset_role"] == "proxy_unknown" for row in rows], dtype=bool)
    labeled_indices = source_base_indices[labeled].tolist()
    validation_indices = source_base_indices[validation].tolist()
    unlabeled_indices = list(range(200, 220))
    physical = np.asarray(_physical_keys(rows), dtype=object)
    proxy_physical = physical[proxy].tolist()
    proxy_selection = {
        "days": list(PAIR.FROZEN_PROXY_DAYS),
        "rxs": list(PAIR.FROZEN_PROXY_RXS),
        "selection_seed": PAIR.FROZEN_PROXY_SELECTION_SEED,
        "max_samples_per_tx": PAIR.FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        "expected_total_count": PAIR.FROZEN_PROXY_TOTAL_COUNT,
    }
    receipt = {
        "labeled_indices_sha256": PAIR._index_sha256(labeled_indices),
        "unlabeled_indices_sha256": PAIR._index_sha256(unlabeled_indices),
        "source_validation_indices_sha256": PAIR._index_sha256(validation_indices),
        "wisig_pkl_sha256": "",
    }
    receipt["split_manifest_sha256"] = PAIR._canonical_json_sha256(receipt)
    manifest = _strict_load_manifest(checkpoint, checkpoint_sha256, tx_ids)
    manifest.update(
        {
            "schema": "cvs.phase1.icmt_lv_export.v1",
            "z_id_source_key": "feat_joint",
            "postfreeze_geometry_path": "checkpoint_model.feat_joint_as_z_id",
            "checkpoint_role": "training_final_only",
            "checkpoint_selection": "final_only",
            "candidate_id": checkpoint.parent.name,
            "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
            "training_run_contract": PAIR.EXPECTED_TRAINING_RUN_LEAF,
            "icmt_receipt_schema": PAIR.EXPECTED_ICMT_RECEIPT_SCHEMA,
            "icmt_enabled": checkpoint.parent.name.endswith("G_ICMT12"),
            "icmt_source_labeled_indices_sha256": receipt["labeled_indices_sha256"],
            "icmt_source_split_manifest_sha256": receipt["split_manifest_sha256"],
            "source_only_export": True,
            "channel_profile": {
                "labeled_fit": {"view": "clean", "scenarios": []},
                "source_validation_known": {"view": "clean", "scenarios": []},
                "proxy_unknown": {"view": "clean", "scenarios": []},
            },
            "split_mode": "tx_rx_day_1_6_3",
            "seed": 7281105,
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "source_tx_ids": list(tx_ids),
            "dataset_path": str((checkpoint.parents[2] / "ManySig.pkl").resolve()),
            "wisig_pkl_sha256": PAIR.FROZEN_WISIG_SHA256,
            "expected_wisig_pkl_sha256": PAIR.FROZEN_WISIG_SHA256,
            "checkpoint_declared_wisig_pkl_sha256": "",
            "checkpoint_declared_wisig_pkl_sha256_empty_caveat": True,
            "dataset_path_checkpoint_equal": True,
            "known_validation_outer_tx_ids": [PAIR.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold]],
            "proxy_unknown_tx_ids": [PAIR.FROZEN_FOLD_PROXY_TX[fold]],
            "source_split_receipt": receipt,
            "source_split_receipt_checkpoint_equal": True,
            "tx_partition_receipt_checkpoint_equal": True,
            "labeled_indices_sha256": receipt["labeled_indices_sha256"],
            "unlabeled_indices_sha256": receipt["unlabeled_indices_sha256"],
            "source_validation_indices_sha256": receipt["source_validation_indices_sha256"],
            "labeled_physical_keys_sha256": PAIR._canonical_json_sha256(physical[labeled].tolist()),
            "source_validation_physical_keys_sha256": PAIR._canonical_json_sha256(
                physical[validation].tolist()
            ),
            "proxy_physical_keys_sha256": PAIR._canonical_json_sha256(physical[proxy].tolist()),
            "labeled_source_validation_physical_disjoint": True,
            "labeled_validation_proxy_physical_disjoint": True,
            "labeled_row_count": int(labeled.sum()),
            "unlabeled_row_count": len(unlabeled_indices),
            "source_validation_row_count": int(validation.sum()),
            "proxy_row_count": int(proxy.sum()),
            "proxy_export_info": {
                "pkl": str((checkpoint.parents[2] / "ManySig.pkl").resolve()),
                "role": "proxy_unknown",
                "tx_idx": [4],
                "tx_labels": [PAIR.FROZEN_FOLD_PROXY_TX[fold]],
                "days": ",".join(PAIR.FROZEN_PROXY_DAYS),
                "rxs": ",".join(PAIR.FROZEN_PROXY_RXS),
                "size": PAIR.FROZEN_PROXY_TOTAL_COUNT,
                "excluded_source_record_count": 0,
            },
            "proxy_days": ",".join(PAIR.FROZEN_PROXY_DAYS),
            "proxy_rxs": ",".join(PAIR.FROZEN_PROXY_RXS),
            "proxy_seed": PAIR.FROZEN_PROXY_SELECTION_SEED,
            "proxy_max_samples_per_tx": PAIR.FROZEN_PROXY_MAX_SAMPLES_PER_TX,
            "proxy_expected_total_count": PAIR.FROZEN_PROXY_TOTAL_COUNT,
            "proxy_selection": {
                **proxy_selection,
                "selection_sha256": PAIR._canonical_json_sha256(proxy_selection),
            },
            "proxy_physical_key_receipt": {
                "row_count": len(proxy_physical),
                "unique_count": len(set(proxy_physical)),
                "ordered_sha256": PAIR._canonical_json_sha256(proxy_physical),
                "set_sha256": PAIR._canonical_json_sha256(sorted(proxy_physical)),
            },
            "forwarded_roles": ["labeled_fit", "source_validation_known", "proxy_unknown"],
            "unlabeled_loader_constructed": False,
            "unlabeled_loader_rows": 0,
            "unlabeled_forward_rows": 0,
            "unlabeled_features_persisted": False,
        }
    )
    return manifest


def _clean_payload(
    *, arm: str, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...], fold: int
) -> dict[str, np.ndarray]:
    rows, source_base_indices = _clean_rows(tx_ids, PAIR.FROZEN_FOLD_PROXY_TX[fold])
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for row in rows:
        role = row["dataset_role"]
        if role in {"labeled_fit", "source_validation_known"}:
            features.append(_one_hot(tx_ids.index(row["tx_ids"])))
            logits.append(_logits(row["tx_ids"], tx_ids))
        else:
            features.append(_one_hot(0) if arm == "C" else -np.ones(4, dtype=np.float32))
            logits.append(np.zeros(4, dtype=np.float32))
    manifest = _clean_manifest(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        tx_ids=tx_ids,
        fold=fold,
        rows=rows,
        source_base_indices=source_base_indices,
    )
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "source_base_indices": source_base_indices,
        "manifest_json": np.asarray(json.dumps(manifest)),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _leo_payload(
    *, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]
) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for tx_index, tx in enumerate(tx_ids):
            for rx_index, rx in enumerate(RXS):
                rows.append(
                    {
                        "dataset_role": "source",
                        "tx_ids": tx,
                        "rx_ids": rx,
                        "day_ids": DAYS[(scenario_index + tx_index + rx_index) % len(DAYS)],
                        "eq_ids": "1",
                        "sig_ids": f"{scenario}-{tx_index}-{rx_index}",
                        "channel_views": PAIR.EXPECTED_LEO_RUNTIME_VIEW,
                        "sat_scenarios": scenario,
                    }
                )
                features.append(_one_hot(tx_index))
                logits.append(_logits(tx, tx_ids))
    manifest = _strict_load_manifest(checkpoint, checkpoint_sha256, tx_ids)
    manifest.update(
        {
            "source_only_export": True,
            "star_ground_channel_impl": "simplified_leo_residual",
            "channel_profile": {
                "source": {
                    "view": "satellite",
                    "scenarios": list(SCENARIOS),
                    "sat_seed": 7281718,
                }
            },
        }
    )
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "manifest_json": np.asarray(json.dumps(manifest)),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _rewrite(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    mutate(payload)
    np.savez(path, **payload)


def _set_manifest(path: Path, mutate) -> None:
    def apply(payload: dict[str, np.ndarray]) -> None:
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        mutate(manifest)
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(path, apply)


def _shrink_clean_proxy_to_one(path: Path) -> None:
    def apply(payload: dict[str, np.ndarray]) -> None:
        roles = np.asarray(payload["dataset_role"]).astype(str)
        proxy_indices = np.flatnonzero(roles == "proxy_unknown")
        assert proxy_indices.size == PAIR.FROZEN_PROXY_TOTAL_COUNT
        keep = roles != "proxy_unknown"
        keep[int(proxy_indices[0])] = True
        original_rows = int(roles.size)
        for field, value in list(payload.items()):
            array = np.asarray(value)
            if field != "manifest_json" and array.ndim >= 1 and array.shape[0] == original_rows:
                payload[field] = array[keep]
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        physical = [
            "\x1f".join(
                str(np.asarray(payload[field]).reshape(-1)[index])
                for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
            )
            for index in np.flatnonzero(
                np.asarray(payload["dataset_role"]).astype(str) == "proxy_unknown"
            )
        ]
        assert len(physical) == 1
        manifest["proxy_row_count"] = 1
        manifest["proxy_expected_total_count"] = 1
        manifest["proxy_export_info"]["size"] = 1
        selection = dict(manifest["proxy_selection"])
        selection.pop("selection_sha256", None)
        selection["expected_total_count"] = 1
        manifest["proxy_selection"] = {
            **selection,
            "selection_sha256": PAIR._canonical_json_sha256(selection),
        }
        manifest["proxy_physical_keys_sha256"] = PAIR._canonical_json_sha256(physical)
        manifest["proxy_physical_key_receipt"] = {
            "row_count": 1,
            "unique_count": 1,
            "ordered_sha256": PAIR._canonical_json_sha256(physical),
            "set_sha256": PAIR._canonical_json_sha256(sorted(physical)),
        }
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(path, apply)


def _synchronize_prior_proxy_one(record: dict[str, object], paths: dict[str, object]) -> None:
    record["expected_role_counts"]["proxy_unknown"] = 1
    for arm in ("C", "G"):
        prefix = arm.lower()
        clean_path = Path(paths[f"{prefix}_clean"])
        proxy_json = Path(paths[f"{prefix}_proxy"])
        proxy_csv = Path(paths[f"{prefix}_proxy_scores"])
        _shrink_clean_proxy_to_one(clean_path)
        _proxy_metrics(proxy_json, proxy_csv, clean_path)
        bindings = record["bindings"]
        bindings[f"{prefix}_clean_npz_sha256"] = _sha(clean_path)
        bindings[f"{prefix}_proxy_metrics_json_sha256"] = _sha(proxy_json)
        bindings[f"{prefix}_proxy_scores_csv_sha256"] = _sha(proxy_csv)
        receipt = record["feature_norm_receipt"][arm]
        proxy_stats = receipt["roles"]["proxy_unknown"]
        removed = int(proxy_stats["total_rows"]) - 1
        proxy_stats.update(
            {
                "total_rows": 1,
                "positive_norm_rows": 1,
                "zero_norm_rows": 0,
                "nonfinite_rows": 0,
                "retained_rows": 1,
                "dropped_rows": 0,
                "count_closed": True,
            }
        )
        for field in ("clean_total_rows", "role_total_rows_sum", "retained_rows"):
            receipt[field] = int(receipt[field]) - removed
        record["proxy_continuous_guardrail"][arm]["proxy_unknown"]["count"] = 1


def _proxy_metrics(path: Path, score_path: Path, clean_path: Path) -> None:
    with np.load(clean_path, allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        roles = np.asarray(data["dataset_role"])
        tx_ids = np.asarray(data["tx_ids"]).astype(str)
        rx_ids = np.asarray(data["rx_ids"]).astype(str)
        day_ids = np.asarray(data["day_ids"]).astype(str)
        channel_views = np.asarray(data["channel_views"]).astype(str)
        sat_scenarios = np.asarray(data["sat_scenarios"]).astype(str)
    path.write_text(
        json.dumps(
            {
                "phase": "phase1_only_logits_open_set_reject",
                "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
                "feature_npz": str(clean_path.resolve()),
                "source_tx_ids": manifest["source_tx_ids"],
                "known_query_roles": ["source_validation_known"],
                "unknown_query_roles": ["proxy_unknown"],
                "known_query_count": int(np.sum(roles == "source_validation_known")),
                "unknown_query_count": int(np.sum(roles == "proxy_unknown")),
                "AUROC_unknown": 0.5,
                "unknown_FAR": 0.5,
                "manifest": manifest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "row",
        "role",
        "tx_id",
        "rx_id",
        "day_id",
        "channel_view",
        "sat_scenario",
        "is_known_query",
        "is_unknown_query",
        "pred_class",
        "pred_tx_id",
        "accepted",
        "closed_correct_known",
        "accepted_correct_known",
        "confidence",
        "logit_margin",
        "energy",
    ]
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, role_value in enumerate(roles.astype(str).tolist()):
            known = role_value == "source_validation_known"
            unknown = role_value == "proxy_unknown"
            writer.writerow(
                {
                    "row": index,
                    "role": role_value,
                    "tx_id": tx_ids[index],
                    "rx_id": rx_ids[index],
                    "day_id": day_ids[index],
                    "channel_view": channel_views[index],
                    "sat_scenario": sat_scenarios[index],
                    "is_known_query": int(known),
                    "is_unknown_query": int(unknown),
                    "pred_class": 0,
                    "pred_tx_id": manifest["source_tx_ids"][0],
                    "accepted": 1,
                    "closed_correct_known": 0,
                    "accepted_correct_known": 0,
                    "confidence": "0.50000000",
                    "logit_margin": "0.00000000",
                    "energy": "0.00000000",
                }
            )


def _write_leo_binding(
    path: Path,
    *,
    leo_path: Path,
    clean_path: Path,
    checkpoint: Path,
    candidate: str,
    fold: int,
    arm: str,
    training_root: Path,
    output_root: Path,
) -> None:
    with np.load(leo_path, allow_pickle=False) as data:
        payload = {
            field: np.asarray(data[field]).reshape(-1)
            for field in (
                "tx_ids",
                "rx_ids",
                "day_ids",
                "eq_ids",
                "sig_ids",
                "dataset_role",
                "channel_views",
                "sat_scenarios",
            )
        }
        leo_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
    with np.load(clean_path, allow_pickle=False) as data:
        clean_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
    source_tx_ids = PAIR.FROZEN_FOLD_SOURCE_TX[fold]
    selection = {
        "source_tx_ids": list(source_tx_ids),
        "source_rx_ids": list(PAIR.EXPECTED_SOURCE_RXS),
        "source_day_ids": list(PAIR.EXPECTED_SOURCE_DAYS),
        "equalized": PAIR._icmt_leo.EXPECTED_EQUALIZED,
        "domain": PAIR._icmt_leo.EXPECTED_DOMAIN,
        "out_len": PAIR._icmt_leo.EXPECTED_OUT_LEN,
        "max_samples_per_combo": PAIR._icmt_leo.EXPECTED_MAX_PER_COMBO,
        "max_samples_per_tx": PAIR._icmt_leo.EXPECTED_MAX_PER_TX,
        "export_seed": PAIR._icmt_leo.EXPECTED_EXPORT_SEED,
        "batch_size": PAIR._icmt_leo.EXPECTED_BATCH_SIZE,
        "channel_view": PAIR._icmt_leo.EXPECTED_CHANNEL_VIEW,
        "runtime_view": PAIR._icmt_leo.EXPECTED_RUNTIME_VIEW,
        "satellite_scenarios": list(PAIR.EXPECTED_SCENARIOS),
        "source_sat_seed": PAIR._icmt_leo.EXPECTED_SOURCE_SAT_SEED,
        "satellite_tta_policy": PAIR._icmt_leo.EXPECTED_TTA_POLICY,
        "star_ground_channel_impl": PAIR._icmt_leo.EXPECTED_STAR_GROUND_IMPL,
        "reconstructed_size": int(payload["tx_ids"].size),
        "generic_source_info": {"test_fixture": True},
    }
    selection["selection_sha256"] = PAIR._icmt_leo._canonical_json_sha256(selection)
    physical = PAIR._icmt_leo._physical_key_receipt(
        PAIR._icmt_leo._physical_keys_from_payload(payload)
    )
    coverage = PAIR._icmt_leo._scenario_coverage_receipt(
        payload, source_tx_ids=source_tx_ids
    )
    binding = {
        "schema": PAIR._icmt_leo.EXPECTED_BINDING_SCHEMA,
        "candidate_id": candidate,
        "fold_index": fold,
        "arm": arm,
        "training_run_root": str(training_root.resolve()),
        "postfreeze_output_root": str(output_root.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_role": "training_final_only",
        "training_run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        "classification_head_contract": PAIR.EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
        "leo_npz_path": str(leo_path.resolve()),
        "leo_npz_sha256": _sha(leo_path),
        "leo_manifest_sha256": PAIR._icmt_leo._canonical_json_sha256(leo_manifest),
        "dataset_path": clean_manifest["dataset_path"],
        "dataset_sha256": PAIR.FROZEN_WISIG_SHA256,
        "source_selection": selection,
        "physical_keys": physical,
        "scenario_assignment_sha256": PAIR._icmt_leo._canonical_json_sha256(
            payload["sat_scenarios"].astype(str).tolist()
        ),
        "scenario_coverage": coverage,
        "all_source_rows_reconstructed": True,
        "all_scenarios_complete": True,
    }
    path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")


def _write_pair(root: Path, *, fold: int = 1) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    tx_ids = PAIR.FROZEN_FOLD_SOURCE_TX[fold]
    training_root = root.parent / PAIR.EXPECTED_TRAINING_RUN_LEAF
    c_checkpoint = training_root / f"F{fold}C_ICMT12" / "final_ssdg.pth"
    g_checkpoint = training_root / f"F{fold}G_ICMT12" / "final_ssdg.pth"
    c_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    g_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    c_checkpoint.write_bytes(f"ICMT-C-{fold}".encode("ascii"))
    g_checkpoint.write_bytes(f"ICMT-G-{fold}".encode("ascii"))
    (root.parent / "ManySig.pkl").write_bytes(b"ICMT synthetic ManySig binding fixture")
    c_directory = root / f"F{fold}C_ICMT12"
    g_directory = root / f"F{fold}G_ICMT12"
    c_directory.mkdir(parents=True, exist_ok=True)
    g_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, object] = {
        "root": root.resolve(),
        "training_root": training_root.resolve(),
        "source_tx_ids": tx_ids,
        "c_checkpoint": c_checkpoint,
        "g_checkpoint": g_checkpoint,
        "c_clean": c_directory / "icmt_clean_l_v_proxy_final_only.npz",
        "g_clean": g_directory / "icmt_clean_l_v_proxy_final_only.npz",
        "c_leo": c_directory / "source_leo_final_only.npz",
        "g_leo": g_directory / "source_leo_final_only.npz",
        "c_leo_binding": c_directory / "source_leo_binding.json",
        "g_leo_binding": g_directory / "source_leo_binding.json",
        "c_proxy": c_directory / "proxy_logits_open_set_metrics.json",
        "g_proxy": g_directory / "proxy_logits_open_set_metrics.json",
        "c_proxy_scores": c_directory / "proxy_logits_open_set_scores.csv",
        "g_proxy_scores": g_directory / "proxy_logits_open_set_scores.csv",
    }
    _save(
        paths["c_clean"],
        _clean_payload(
            arm="C", checkpoint=c_checkpoint, checkpoint_sha256=_sha(c_checkpoint), tx_ids=tx_ids, fold=fold
        ),
    )
    _save(
        paths["g_clean"],
        _clean_payload(
            arm="G", checkpoint=g_checkpoint, checkpoint_sha256=_sha(g_checkpoint), tx_ids=tx_ids, fold=fold
        ),
    )
    _save(
        paths["c_leo"],
        _leo_payload(checkpoint=c_checkpoint, checkpoint_sha256=_sha(c_checkpoint), tx_ids=tx_ids),
    )
    _save(
        paths["g_leo"],
        _leo_payload(checkpoint=g_checkpoint, checkpoint_sha256=_sha(g_checkpoint), tx_ids=tx_ids),
    )
    _proxy_metrics(paths["c_proxy"], paths["c_proxy_scores"], paths["c_clean"])
    _proxy_metrics(paths["g_proxy"], paths["g_proxy_scores"], paths["g_clean"])
    _write_leo_binding(
        paths["c_leo_binding"],
        leo_path=paths["c_leo"],
        clean_path=paths["c_clean"],
        checkpoint=c_checkpoint,
        candidate=f"F{fold}C_ICMT12",
        fold=fold,
        arm="C",
        training_root=training_root,
        output_root=root,
    )
    _write_leo_binding(
        paths["g_leo_binding"],
        leo_path=paths["g_leo"],
        clean_path=paths["g_clean"],
        checkpoint=g_checkpoint,
        candidate=f"F{fold}G_ICMT12",
        fold=fold,
        arm="G",
        training_root=training_root,
        output_root=root,
    )
    return paths


def _args(
    paths: dict[str, object], output: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()
) -> object:
    command = [
        "--c-clean-npz", str(paths["c_clean"]),
        "--g-clean-npz", str(paths["g_clean"]),
        "--c-leo-npz", str(paths["c_leo"]),
        "--g-leo-npz", str(paths["g_leo"]),
        "--c-leo-binding-json", str(paths["c_leo_binding"]),
        "--g-leo-binding-json", str(paths["g_leo_binding"]),
        "--c-final-checkpoint", str(paths["c_checkpoint"]),
        "--g-final-checkpoint", str(paths["g_checkpoint"]),
        "--c-proxy-metrics-json", str(paths["c_proxy"]),
        "--g-proxy-metrics-json", str(paths["g_proxy"]),
        "--c-proxy-scores-csv", str(paths["c_proxy_scores"]),
        "--g-proxy-scores-csv", str(paths["g_proxy_scores"]),
        "--candidate-pair", f"F{fold}_C_vs_G",
        "--fold-index", str(fold),
        "--postfreeze-matrix-id", TEST_MATRIX_ID,
        "--postfreeze-output-root", str(paths["root"]),
        "--training-run-root", str(paths["training_root"]),
        "--source-tx-ids", ",".join(paths["source_tx_ids"]),
        "--expected-source-count", "72",
        "--expected-proxy-count", str(PAIR.FROZEN_PROXY_TOTAL_COUNT),
        "--output-metrics-json", str(output),
    ]
    if priors:
        command.extend(["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)])
    return PAIR.build_parser().parse_args(command)


def test_float64_geometry_matches_formula_totalizes_zero_and_rejects_nonfinite():
    tx_ids = ("a", "b", "c", "d")
    features = np.asarray(
        [
            [1.0, 0.1], [1.0, -0.1],
            [0.1, 1.0], [-0.1, 1.0],
            [-1.0, 0.1], [-1.0, -0.1],
            [0.1, -1.0], [-0.1, -1.0],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.asarray(tx_ids, dtype=object), 2)
    geometry = PAIR.fit_frozen_diagonal_gaussian(features, labels, tx_ids)
    normalized = features.astype(np.float64) / np.linalg.norm(features.astype(np.float64), axis=1)[:, None]
    means = np.stack([normalized[labels == tx].mean(axis=0) for tx in tx_ids])
    raw_variances = np.stack([normalized[labels == tx].var(axis=0, ddof=1) for tx in tx_ids])
    pooled = raw_variances.mean(axis=0)
    variances = np.maximum(1e-6, 0.9 * raw_variances + 0.1 * pooled[None, :])
    np.testing.assert_allclose(geometry["means"], means, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(geometry["variances"], variances, rtol=0.0, atol=1e-15)
    probe = np.asarray([[0.8, 0.2], [-0.2, -0.8]], dtype=np.float32)
    probe64 = probe.astype(np.float64) / np.linalg.norm(probe.astype(np.float64), axis=1)[:, None]
    nll = 0.5 * np.sum(
        (probe64[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
        + np.log(2.0 * math.pi * variances)[None, :, :],
        axis=2,
    )
    maximum = np.max(-nll, axis=1)
    expected = math.log(4.0) - (
        maximum + np.log(np.sum(np.exp(-nll - maximum[:, None]), axis=1))
    )
    observed = PAIR.score_frozen_icmt_nll(probe, geometry)
    assert observed.dtype == np.float64
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
    piecewise_input = np.concatenate((probe, np.zeros((1, 2), dtype=np.float32)), axis=0)
    normalized = PAIR._normalize_float64(piecewise_input, label="piecewise test")
    np.testing.assert_allclose(normalized[:2], probe64, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(normalized[2], np.zeros(2, dtype=np.float64))
    totalized_scores = PAIR.score_frozen_icmt_nll(piecewise_input, geometry)
    assert totalized_scores.shape == (3,)
    assert np.isfinite(totalized_scores).all()
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="non-finite"):
        PAIR.score_frozen_icmt_nll(np.asarray([[np.nan, 0.0]], dtype=np.float32), geometry)


def test_pair_closes_l_only_v_known_proxy_strict_and_classifier_gates(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == metrics["schema"]
    assert metrics["schema"] == "cvs.phase1.icmt_postfreeze_pair.v2"
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    for role_field in (
        "source_validation_fit_rows",
        "proxy_unknown_fit_rows",
        "leo_fit_rows",
        "unlabeled_fit_rows",
    ):
        assert metrics["policy"][role_field] == 0
    assert metrics["clean_source_validation"]["C"]["count"] == 8
    assert metrics["proxy_continuous_guardrail"]["strict_AUROC_improvement"] is True
    assert metrics["proxy_continuous_guardrail"]["strict_proxy_known_gap_improvement"] is True
    assert metrics["postfreeze_gates"]["fold_verdict"] == "PENDING_GLOBAL_18_GRID"
    assert metrics["phase3_unknown_capability_claim"] == "NOT_EVALUATED"
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


def test_zero_v_row_is_retained_and_counted_for_both_arms(tmp_path):
    paths = _write_pair(tmp_path / "matrix")

    def zero_first_validation(payload: dict[str, np.ndarray]) -> None:
        index = int(np.flatnonzero(payload["dataset_role"] == "source_validation_known")[0])
        payload["features"][index] = 0.0

    for key in ("c_clean", "g_clean"):
        _rewrite(paths[key], zero_first_validation)
    metrics = PAIR.evaluate(_args(paths, Path(paths["root"]) / "zero_v_retained.json"))
    assert metrics["clean_source_validation"]["C"]["count"] == 8
    assert metrics["clean_source_validation"]["G"]["count"] == 8
    for arm in ("C", "G"):
        role = metrics["feature_norm_receipt"][arm]["roles"]["source_validation_known"]
        assert role == {
            "total_rows": 8,
            "positive_norm_rows": 7,
            "zero_norm_rows": 1,
            "nonfinite_rows": 0,
            "retained_rows": 8,
            "dropped_rows": 0,
            "count_closed": True,
        }
        assert metrics["proxy_continuous_guardrail"][arm]["known_heldout"]["count"] == 8


def test_zero_l_v_and_proxy_rows_are_all_retained_and_scored(tmp_path):
    paths = _write_pair(tmp_path / "matrix")

    def zero_one_per_role(payload: dict[str, np.ndarray]) -> None:
        for role in ("labeled_fit", "source_validation_known", "proxy_unknown"):
            index = int(np.flatnonzero(payload["dataset_role"] == role)[0])
            payload["features"][index] = 0.0

    for key in ("c_clean", "g_clean"):
        _rewrite(paths[key], zero_one_per_role)
    metrics = PAIR.evaluate(_args(paths, Path(paths["root"]) / "zero_all_roles.json"))
    for arm in ("C", "G"):
        receipt = metrics["feature_norm_receipt"][arm]
        for role in ("labeled_fit", "source_validation_known", "proxy_unknown"):
            assert receipt["roles"][role]["zero_norm_rows"] == 1
            assert receipt["roles"][role]["dropped_rows"] == 0
        diagnostic = metrics["proxy_continuous_guardrail"][arm]
        assert diagnostic["fit"]["row_count"] == 12
        assert diagnostic["known_heldout"]["count"] == 8
        assert diagnostic["proxy_unknown"]["count"] == PAIR.FROZEN_PROXY_TOTAL_COUNT


def test_nonfinite_payload_feature_is_fatal_before_any_receipt(tmp_path):
    paths = _write_pair(tmp_path / "matrix")

    def make_nonfinite(payload: dict[str, np.ndarray]) -> None:
        index = int(np.flatnonzero(payload["dataset_role"] == "proxy_unknown")[0])
        payload["features"][index, 0] = np.nan

    _rewrite(paths["g_clean"], make_nonfinite)
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="non-finite"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "nonfinite.json"))


def test_pair_permanently_rejects_without_both_strict_continuous_improvements(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    with np.load(paths["c_clean"], allow_pickle=False) as data:
        c_features = np.asarray(data["features"]).copy()
    _rewrite(paths["g_clean"], lambda payload: payload.__setitem__("features", c_features))
    metrics = PAIR.evaluate(_args(paths, Path(paths["root"]) / "no_improvement.json"))
    guardrail = metrics["proxy_continuous_guardrail"]
    assert guardrail["G_minus_C"]["AUROC_unknown"] == 0.0
    assert guardrail["G_minus_C"]["proxy_minus_known_heldout_mean_u"] == 0.0
    assert guardrail["passed"] is False
    assert metrics["postfreeze_gates"]["fold_verdict"] == "REJECT_P1_ICMT_PERMANENT"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest.__setitem__("unlabeled_forward_rows", 1))
                for key in ("c_clean", "g_clean")
            ],
            "forwarded U rows",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["source_base_indices"].__setitem__(12, 0))
                for key in ("c_clean", "g_clean")
            ],
            "source_base_indices contain duplicates",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__("classification_head_contract", "FORGED_HEAD"),
            ),
            "classification_head_contract",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__("schema", "cvs.phase1.foreign_export.v1"),
            ),
            "schema mismatch",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__("candidate_id", "F1G_ICMT12"),
            ),
            "candidate arm binding mismatch",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__(
                    "icmt_source_labeled_indices_sha256", "0" * 64
                ),
            ),
            "labeled-index SHA256 drifted",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__("unlabeled_loader_constructed", True),
            ),
            "constructed a U loader",
        ),
    ],
)
def test_pair_fails_closed_on_u_index_or_head_tamper(tmp_path, mutate, message):
    paths = _write_pair(tmp_path / "matrix")
    mutate(paths)
    with pytest.raises(PAIR.ICMTPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "tampered.json"))


def test_pair_rejects_complete_arm_swap_and_non_v1_training_root(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    swapped = _args(paths, Path(paths["root"]) / "swapped.json")
    swapped.c_clean_npz, swapped.g_clean_npz = str(paths["g_clean"]), str(paths["c_clean"])
    swapped.c_leo_npz, swapped.g_leo_npz = str(paths["g_leo"]), str(paths["c_leo"])
    swapped.c_final_checkpoint, swapped.g_final_checkpoint = str(paths["g_checkpoint"]), str(paths["c_checkpoint"])
    swapped.c_proxy_metrics_json, swapped.g_proxy_metrics_json = str(paths["g_proxy"]), str(paths["c_proxy"])
    swapped.c_proxy_scores_csv, swapped.g_proxy_scores_csv = str(paths["g_proxy_scores"]), str(paths["c_proxy_scores"])
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="C final checkpoint path"):
        PAIR.evaluate(swapped)

    wrong_root = Path(paths["root"]).parent / "phase1_icmt12_20260810_v0"
    wrong_root.mkdir(parents=True, exist_ok=True)
    wrong = _args(paths, Path(paths["root"]) / "wrong_root.json")
    wrong.training_run_root = str(wrong_root)
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="training run root leaf must be"):
        PAIR.evaluate(wrong)


def test_pair_rejects_wrong_leo_dataset_binding_and_replaced_source_artifact(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    binding_path = Path(paths["c_leo_binding"])
    original_binding = binding_path.read_text(encoding="utf-8")
    binding = json.loads(original_binding)
    binding["dataset_path"] = str((tmp_path / "wrong_ManySig.pkl").resolve())
    binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="dataset path does not bind clean export"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "wrong_dataset.json"))

    binding_path.write_text(original_binding, encoding="utf-8")
    _rewrite(
        Path(paths["c_leo"]),
        lambda payload: payload["features"].__setitem__((0, 0), 0.5),
    )
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="LEO NPZ current SHA256"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "replaced_source.json"))


def test_icmt_leo_export_binding_rejects_nonfrozen_dataset_bytes(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    args = PAIR._icmt_leo.build_parser().parse_args(
        [
            "--ckpt", str(paths["c_checkpoint"]),
            "--wisig-pkl", str(Path(paths["root"]).parent / "ManySig.pkl"),
            "--out-npz", str(paths["c_leo"]),
            "--binding-json", str(paths["c_leo_binding"]),
            "--training-run-root", str(paths["training_root"]),
            "--postfreeze-output-root", str(paths["root"]),
            "--candidate-id", "F1C_ICMT12",
            "--fold-index", "1",
            "--arm", "C",
            "--source-tx-ids", ",".join(paths["source_tx_ids"]),
            "--source_only_export",
            "--expected-wisig-sha256", PAIR.FROZEN_WISIG_SHA256,
        ]
    )
    with pytest.raises(PAIR._icmt_leo.ICMTLEOBindingError, match="input bytes"):
        PAIR._icmt_leo._validate_frozen_args(args)


def test_pair_rejects_one_scenario_missing_day_even_when_global_days_remain_complete(tmp_path):
    paths = _write_pair(tmp_path / "matrix")

    def remove_one_day_from_first_scenario(payload: dict[str, np.ndarray]) -> None:
        scenarios = np.asarray(payload["sat_scenarios"]).astype(str)
        days = np.asarray(payload["day_ids"]).astype(str)
        targeted = np.logical_and(scenarios == SCENARIOS[0], days == DAYS[1])
        assert targeted.any()
        days[targeted] = DAYS[0]
        payload["day_ids"] = days
        assert set(days.tolist()) == set(DAYS)

    for arm in ("c", "g"):
        leo_path = Path(paths[f"{arm}_leo"])
        binding_path = Path(paths[f"{arm}_leo_binding"])
        _rewrite(leo_path, remove_one_day_from_first_scenario)
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["leo_npz_sha256"] = _sha(leo_path)
        binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")

    with pytest.raises(PAIR.ICMTPostfreezePairError, match="scenario lacks complete day coverage"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "missing_scenario_day.json"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("proxy_days", "2021_03_01", "proxy_days must equal frozen value"),
        ("proxy_rxs", "1-1,1-19", "proxy_rxs must equal frozen value"),
        (
            "max_proxy_samples_per_tx",
            PAIR.FROZEN_PROXY_MAX_SAMPLES_PER_TX - 1,
            "max_proxy_samples_per_tx must equal frozen value",
        ),
    ],
)
def test_clean_exporter_rejects_proxy_cli_selection_drift(field, value, message):
    module_name = "icmt_split_export_proxy_contract"
    exporter = sys.modules.get(module_name)
    if exporter is None:
        export_spec = importlib.util.spec_from_file_location(module_name, EXPORTER_PATH)
        assert export_spec is not None and export_spec.loader is not None
        exporter = importlib.util.module_from_spec(export_spec)
        sys.modules[module_name] = exporter
        export_spec.loader.exec_module(exporter)
    args = exporter.build_parser().parse_args(
        [
            "--ckpt", "unused.pth",
            "--wisig_pkl", "unused.pkl",
            "--out_npz", "unused.npz",
            "--source_tx_ids", "a,b,c,d",
            "--known_validation_tx_ids", "e",
            "--proxy_unknown_tx_ids", "f",
            "--expected-wisig-sha256", PAIR.FROZEN_WISIG_SHA256,
        ]
    )
    setattr(args, field, value)
    with pytest.raises(exporter.ICMTSplitExportError, match=message):
        exporter._require_frozen_proxy_selection(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proxy_days", "2021_03_01"),
        ("proxy_rxs", "1-1,1-19"),
        ("proxy_seed", PAIR.FROZEN_PROXY_SELECTION_SEED + 1),
        ("proxy_max_samples_per_tx", PAIR.FROZEN_PROXY_MAX_SAMPLES_PER_TX - 1),
    ],
)
def test_pair_rejects_each_frozen_proxy_selection_manifest_drift(tmp_path, field, value):
    paths = _write_pair(tmp_path / "matrix")
    _set_manifest(paths["c_clean"], lambda manifest: manifest.__setitem__(field, value))
    with pytest.raises(PAIR.ICMTPostfreezePairError, match=field):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / f"drift_{field}.json"))


def test_pair_rejects_proxy_csv_count_drift(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    score_path = Path(paths["c_proxy_scores"])
    rows = score_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) > PAIR.FROZEN_PROXY_TOTAL_COUNT
    score_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="CSV/clean row count mismatch"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "proxy_csv_short.json"))


@pytest.mark.parametrize("surface", ["json", "physical_receipt"])
def test_pair_rejects_proxy_json_or_physical_count_drift(tmp_path, surface):
    paths = _write_pair(tmp_path / "matrix")
    if surface == "json":
        proxy_path = Path(paths["c_proxy"])
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        proxy["unknown_query_count"] = PAIR.FROZEN_PROXY_TOTAL_COUNT - 1
        proxy_path.write_text(json.dumps(proxy, sort_keys=True), encoding="utf-8")
        message = "proxy unknown row count mismatch"
    else:
        _set_manifest(
            paths["c_clean"],
            lambda manifest: manifest["proxy_physical_key_receipt"].__setitem__(
                "row_count", PAIR.FROZEN_PROXY_TOTAL_COUNT - 1
            ),
        )
        message = "proxy physical-key receipt drifted"
    with pytest.raises(PAIR.ICMTPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / f"proxy_{surface}_drift.json"))


def test_current_pair_rejects_fully_synchronized_one_row_proxy_cli_count(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    for prefix in ("c", "g"):
        clean_path = Path(paths[f"{prefix}_clean"])
        _shrink_clean_proxy_to_one(clean_path)
        _proxy_metrics(
            Path(paths[f"{prefix}_proxy"]),
            Path(paths[f"{prefix}_proxy_scores"]),
            clean_path,
        )
    args = _args(paths, Path(paths["root"]) / "one_proxy_current.json")
    args.expected_proxy_count = 1
    with pytest.raises(
        PAIR.ICMTPostfreezePairError,
        match=f"expected_proxy_count must equal frozen {PAIR.FROZEN_PROXY_TOTAL_COUNT}",
    ):
        PAIR.evaluate(args)


def test_f6_rejects_fully_synchronized_one_row_proxy_prior_at_frozen_count_gate(tmp_path):
    root = tmp_path / "matrix"
    priors: list[Path] = []
    prior_paths_by_fold: dict[int, dict[str, object]] = {}
    for fold in range(1, 6):
        paths = _write_pair(root, fold=fold)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        priors.append(output)
        prior_paths_by_fold[fold] = paths
    final_paths = _write_pair(root, fold=6)
    record = json.loads(priors[0].read_text(encoding="utf-8"))
    _synchronize_prior_proxy_one(record, prior_paths_by_fold[1])
    priors[0].write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        PAIR.ICMTPostfreezePairError,
        match=f"expected proxy count must equal frozen {PAIR.FROZEN_PROXY_TOTAL_COUNT}",
    ):
        PAIR.evaluate(
            _args(
                final_paths,
                root / "F6_one_proxy_attack.json",
                fold=6,
                priors=tuple(priors),
            )
        )


def test_six_fold_aggregate_closes_same_matrix_root_training_root_and_prior_receipts(tmp_path):
    root = tmp_path / "matrix"
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(root, fold=fold)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        priors.append(output)
    final_paths = _write_pair(root, fold=6)
    final = PAIR.evaluate(
        _args(final_paths, root / "F6_C_vs_G_pair_metrics.json", fold=6, priors=tuple(priors))
    )
    aggregate = final["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert aggregate["verdict"] == "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
    assert aggregate["global_18_cell_equal_weight_G_minus_C_pp"] == {
        metric: 0.0 for metric in PAIR.CLASSIFICATION_METRICS
    }
    original = priors[0].read_text(encoding="utf-8")
    receipt = json.loads(original)
    receipt["postfreeze_matrix_id"] = "foreign_postfreeze_run"
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="matrix_id mismatch"):
        PAIR.evaluate(_args(final_paths, root / "F6_cross_run.json", fold=6, priors=tuple(priors)))
    priors[0].write_text(original, encoding="utf-8")

    receipt = json.loads(original)
    receipt["clean_source_validation"]["G_minus_C_pp"]["overall_accuracy"] += 0.25
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="raw-artifact recomputation"):
        PAIR.evaluate(_args(final_paths, root / "F6_clean_delta_tamper.json", fold=6, priors=tuple(priors)))
    priors[0].write_text(original, encoding="utf-8")

    receipt = json.loads(original)
    receipt["leo_scenarios"][SCENARIOS[0]]["G_minus_C_pp"]["overall_accuracy"] += 0.25
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="raw-artifact recomputation"):
        PAIR.evaluate(_args(final_paths, root / "F6_leo_delta_tamper.json", fold=6, priors=tuple(priors)))
    priors[0].write_text(original, encoding="utf-8")

    receipt = json.loads(original)
    receipt["leo_scenarios"][SCENARIOS[0]]["G"]["overall_accuracy"] -= 0.01
    receipt["leo_scenarios"][SCENARIOS[0]]["G_minus_C_pp"]["overall_accuracy"] -= 1.0
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="raw-artifact recomputation"):
        PAIR.evaluate(_args(final_paths, root / "F6_synced_summary_tamper.json", fold=6, priors=tuple(priors)))
    priors[0].write_text(original, encoding="utf-8")

    prior_clean_path = Path(root / "F1C_ICMT12" / "icmt_clean_l_v_proxy_final_only.npz")
    prior_clean_bytes = prior_clean_path.read_bytes()
    _rewrite(
        prior_clean_path,
        lambda payload: payload["features"].__setitem__((0, 0), 0.75),
    )
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="does not match current artifact"):
        PAIR.evaluate(_args(final_paths, root / "F6_prior_artifact_tamper.json", fold=6, priors=tuple(priors)))
    prior_clean_path.write_bytes(prior_clean_bytes)

    receipt = json.loads(original)
    receipt["feature_norm_receipt"]["C"]["roles"]["labeled_fit"]["zero_norm_rows"] += 1
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.ICMTPostfreezePairError, match="positive/zero/nonfinite counts do not close"):
        PAIR.evaluate(_args(final_paths, root / "F6_retry.json", fold=6, priors=tuple(priors)))


def test_launcher_is_exactly_42_steps_and_exporter_never_forwards_u():
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    exporter = EXPORTER_PATH.read_text(encoding="utf-8")
    evaluator = EVALUATOR_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_icmt_postfreeze_20260810_v2",
        "phase1_icmt12_20260810_v1",
        "export_phase1_icmt_features.py",
        "export_phase1_icmt_leo_features.py",
        "F${fold}${arm}_ICMT12",
        "--known_query_roles source_validation_known",
        "--calibration_roles source_validation_known",
        "--unknown_query_roles proxy_unknown",
        "--c-proxy-scores-csv",
        "--g-proxy-scores-csv",
        "--expected-proxy-count 400",
        "--expected-wisig-sha256 \"${WISIG_SHA256}\"",
        "--source_only_export",
        "--source_channel_view satellite",
        "--satellite_tta_policy none",
        "CUDA_VISIBLE_DEVICES=\"\"",
    ):
        assert required in launcher
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", launcher, flags=re.MULTILINE) == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_icmt_postfreeze_20260810.sh", "--dry-run"],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][ICMT_CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert len(lines) == 42
    assert all("phase1_icmt_postfreeze_20260810_v2" in line for line in lines)
    assert "phase1_icmt_postfreeze_20260810_v1" not in launcher
    invalid = subprocess.run(
        [
            "bash", "-c",
            "TRAIN_RUN_ROOT='/tmp/phase1_icmt12_20260810_v0' "
            "bash scripts/launch_phase1_icmt_postfreeze_20260810.sh --dry-run",
        ],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=False,
    )
    assert invalid.returncode == 3
    assert "TRAIN_RUN_ROOT leaf must be phase1_icmt12_20260810_v1" in invalid.stderr
    assert "unlabeled_ds =" not in exporter
    assert "DataLoader(\n        unlabeled" not in exporter
    assert '"unlabeled_loader_constructed": False' in exporter
    assert '"unlabeled_loader_rows": 0' in exporter
    assert '"unlabeled_forward_rows": 0' in exporter
    assert '"unlabeled_features_persisted": False' in exporter
    assert "FROZEN_PROXY_SELECTION_SEED = 7281148" in exporter
    assert "FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400" in exporter
    assert "FROZEN_PROXY_TOTAL_COUNT = 400" in exporter
    assert "proxy_physical_key_receipt" in exporter
    assert "_require_split_receipts_match(checkpoint, reconstructed)" in exporter
    assert "import torch" not in evaluator
    assert "gd_proto" not in exporter.lower()
    assert "gd_proto" not in evaluator.lower()
    assert "gd_proto" not in launcher.lower()
    leo_exporter = LEO_EXPORTER_PATH.read_text(encoding="utf-8")
    assert "export_spaceborne_features.py" in leo_exporter
    assert "all_scenarios_complete" in leo_exporter
    assert "ManySig input bytes do not match frozen SHA256" in leo_exporter
    export_spec = importlib.util.spec_from_file_location("icmt_split_export", EXPORTER_PATH)
    assert export_spec is not None and export_spec.loader is not None
    export_module = importlib.util.module_from_spec(export_spec)
    sys.modules[export_spec.name] = export_module
    export_spec.loader.exec_module(export_module)
    with pytest.raises(export_module.ICMTSplitExportError, match="frozen value"):
        export_module._require_frozen_dataset_sha256(
            actual=PAIR.FROZEN_WISIG_SHA256,
            expected="0" * 64,
            checkpoint_declared="",
        )
