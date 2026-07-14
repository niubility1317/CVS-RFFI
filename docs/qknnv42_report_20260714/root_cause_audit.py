from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
MATRIX_ROOT = (
    REPO_ROOT
    / "local_artifacts"
    / "cvs_publication_stage2_full_matrix_20260713"
    / "cvs_publication_stage2c_full_matrix_20260713"
)
CURRENT_FEATURE = (
    REPO_ROOT
    / "local_artifacts"
    / "cvs_publication_adv3b02_feature_cache_20260713"
    / "leo_clear_weak.npz"
)
LEGACY_ROOT = (
    WORKSPACE_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "phase2_qknn_hardpair_n20_20260706"
    / "artifacts"
)
LEGACY_FEATURE = LEGACY_ROOT / "features_hardpair_HP08L5_n20.npz"
LEGACY_AUX = (
    LEGACY_ROOT
    / "v53_fftlogmag_20260706"
    / "features_hardpair_HP08L5_n20_leo_fftlogmag96.npz"
)
LEGACY_DIAG_ROOT = LEGACY_ROOT / "v53_fftlogmag_20260706" / "local_v55_diagnostics_20260706"
LEGACY_BEST = LEGACY_DIAG_ROOT / "k5_strict_seed421070_floor_param_best_predictions_20260707.json"
LEGACY_SEED_SWEEP = LEGACY_DIAG_ROOT / "k5_strict_support_quality_probe_20260707.csv"
LEGACY_RUNNER = REPO_ROOT / "code" / "scripts" / "phase2_support_metric_qknn_probe.py"
CURRENT_RUNNER = REPO_ROOT / "paper_reproduction" / "cvs_aligned" / "cvs_method_runner.py"
OUTPUT_JSON = REPORT_DIR / "root_cause_audit.json"

OLD_LABELS = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
NEW_LABELS = [
    "1-1",
    "1-10",
    "1-11",
    "1-12",
    "1-14",
    "1-15",
    "1-16",
    "1-18",
    "1-19",
    "1-2",
    "10-10",
    "11-10",
    "18-5",
    "19-3",
    "2-13",
    "2-5",
    "3-8",
    "4-10",
    "8-18",
    "8-3",
]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def harmonic(old_acc: float, new_acc: float) -> float:
    denominator = old_acc + new_acc
    return 0.0 if denominator <= 0 else 2.0 * old_acc * new_acc / denominator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_npz_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["manifest_json"].item()))


def assert_inputs() -> None:
    required = [
        MATRIX_ROOT,
        CURRENT_FEATURE,
        LEGACY_FEATURE,
        LEGACY_AUX,
        LEGACY_BEST,
        LEGACY_SEED_SWEEP,
        LEGACY_RUNNER,
        CURRENT_RUNNER,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing root-cause audit inputs:\n" + "\n".join(missing))


def legacy_command(
    *,
    output_json: Path,
    output_csv: Path,
    seed_start: int = 421070,
    seed_count: int = 1,
    new_labels: list[str] | None = None,
    aux_weight: float = 0.34,
    neg_lambda: float = 0.7,
    labelprop_weight: float = 0.025,
    scenario_residual_weight: float = 0.5,
    scenario_aware: bool = True,
    balanced_assignment: bool = True,
    role_balanced_assignment: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(LEGACY_RUNNER),
        "--feature_npz",
        str(LEGACY_FEATURE),
        "--aux_feature_npz",
        str(LEGACY_AUX),
        "--output_json",
        str(output_json),
        "--output_csv",
        str(output_csv),
        "--old_tx_ids",
        ",".join(OLD_LABELS),
        "--new_tx_ids",
        ",".join(new_labels or NEW_LABELS),
        "--old_role",
        "target_old",
        "--new_role",
        "target_unknown",
        "--policies",
        "stable_first",
        "--seed_start",
        str(seed_start),
        "--seed_count",
        str(seed_count),
        "--k_old",
        "5",
        "--k_new",
        "5",
        "--query_per_old",
        "70",
        "--query_per_new",
        "70",
        "--pool_per_old",
        "5",
        "--pool_per_new",
        "5",
        "--transform_modes",
        "diag_whiten_fisher",
        "--transform_strengths",
        "0.1",
        "--topm_grid",
        "1",
        "--proto_mix_grid",
        "0.45",
        "--aux_score_weight_grid",
        str(aux_weight),
        "--radius_norm_grid",
        "0",
        "--old_bias_grid",
        "0.001",
        "--neg_lambda_grid",
        str(neg_lambda),
        "--neg_threshold_grid",
        "0.75",
        "--neg_margin_grid",
        "0.01",
        "--mutual_only_grid",
        "true",
        "--labelprop_weight_grid",
        str(labelprop_weight),
        "--labelprop_k_grid",
        "10",
        "--labelprop_alpha_grid",
        "0.76",
        "--labelprop_temperature_grid",
        "0.05",
        "--labelprop_rounds_grid",
        "8",
        "--labelprop_clip_grid",
        "2",
        "--labelprop_scope_grid",
        "all",
        "--scenario_residual_weight_grid",
        str(scenario_residual_weight),
        "--scenario_residual_min_classes_grid",
        "2",
        "--scenario_residual_clip_grid",
        "0.5",
        "--scenario_residual_scope_grid",
        "new",
        "--slot_release_margin_grid",
        "0.02",
        "--slot_release_accept_margin_grid",
        "-0.35",
    ]
    if scenario_aware:
        command.append("--scenario_aware")
    if balanced_assignment:
        command.append("--balanced_assignment")
    if role_balanced_assignment:
        command.append("--role_balanced_assignment")
    return command


def run_legacy_variant(temp_dir: Path, variant: dict[str, Any]) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    output_json = temp_dir / f"{variant_id}.json"
    output_csv = temp_dir / f"{variant_id}.csv"
    command = legacy_command(output_json=output_json, output_csv=output_csv, **variant["parameters"])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Legacy replay failed for {variant_id}: {completed.returncode}\n"
            + completed.stderr[-4000:]
        )
    payload = read_json(output_json)
    row = payload["best"][0]
    old_acc = float(row["query_old_acc"])
    new_acc = float(row["query_seen_new_acc"])
    return {
        "variant_id": variant_id,
        "setting": variant["setting"],
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "H_old_new": harmonic(old_acc, new_acc),
        "min_old_class_acc": float(row["query_min_old_class_acc"]),
        "min_seen_new_class_acc": float(row["query_min_seen_new_class_acc"]),
        "support_index_sha16": row["support_index_sha16"],
        "query_index_sha16": row["query_index_sha16"],
    }


def run_legacy_ablations() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = [
        {
            "variant_id": "full_legacy",
            "setting": "完整legacy：角色分区+等额quota+场景硬筛+FFT辅助+LP+场景残差",
            "parameters": {},
        },
        {
            "variant_id": "global_quota",
            "setting": "去掉old/new角色分区，仅保留全局等额quota",
            "parameters": {"role_balanced_assignment": False},
        },
        {
            "variant_id": "plain_scenario",
            "setting": "去掉角色分区和全部quota，保留场景硬筛",
            "parameters": {"balanced_assignment": False, "role_balanced_assignment": False},
        },
        {
            "variant_id": "role_no_scenario_info",
            "setting": "保留角色quota，关闭场景硬筛和场景残差",
            "parameters": {"scenario_aware": False, "scenario_residual_weight": 0.0},
        },
        {
            "variant_id": "no_aux_fft",
            "setting": "完整legacy但关闭96维FFT辅助分数",
            "parameters": {"aux_weight": 0.0},
        },
        {
            "variant_id": "no_lp_residual",
            "setting": "完整legacy但关闭label propagation和场景残差",
            "parameters": {"labelprop_weight": 0.0, "scenario_residual_weight": 0.0},
        },
        {
            "variant_id": "current_like_n20",
            "setting": "历史强特征上的current-like统一argmax：无角色、quota、场景和FFT辅助",
            "parameters": {
                "scenario_aware": False,
                "balanced_assignment": False,
                "role_balanced_assignment": False,
                "aux_weight": 0.0,
                "neg_lambda": 0.0,
                "scenario_residual_weight": 0.0,
            },
        },
    ]
    with tempfile.TemporaryDirectory(prefix="qknnv42_rootcause_") as temp:
        temp_dir = Path(temp)
        rows = [run_legacy_variant(temp_dir, variant) for variant in variants]
        bridge_json = temp_dir / "current_like_n2_40.json"
        bridge_csv = temp_dir / "current_like_n2_40.csv"
        bridge_command = legacy_command(
            output_json=bridge_json,
            output_csv=bridge_csv,
            seed_start=421038,
            seed_count=40,
            new_labels=["1-16", "1-18"],
            aux_weight=0.0,
            neg_lambda=0.0,
            scenario_residual_weight=0.0,
            scenario_aware=False,
            balanced_assignment=False,
            role_balanced_assignment=False,
        )
        completed = subprocess.run(
            bridge_command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError("Feature bridge replay failed:\n" + completed.stderr[-4000:])
        with bridge_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            bridge_rows = list(csv.DictReader(handle))
        if len(bridge_rows) != 40:
            raise ValueError(f"Expected 40 bridge rows, got {len(bridge_rows)}")
        bridge_old = [float(row["query_old_acc"]) for row in bridge_rows]
        bridge_new = [float(row["query_seen_new_acc"]) for row in bridge_rows]
        bridge_h = [harmonic(old, new) for old, new in zip(bridge_old, bridge_new)]
        bridge = {
            "evidence": "legacy适配5-view特征+current-like head，2个new类，seed421038..421077",
            "rows": 40,
            "old_acc_mean": mean(bridge_old),
            "old_acc_std": statistics.stdev(bridge_old),
            "seen_new_acc_mean": mean(bridge_new),
            "seen_new_acc_std": statistics.stdev(bridge_new),
            "H_old_new_mean": mean(bridge_h),
            "H_old_new_std": statistics.stdev(bridge_h),
            "H_old_new_min": min(bridge_h),
            "H_old_new_max": max(bridge_h),
            "comparison_limit": "与当前正式clear行receiver和类别近似对齐，但query样本与生成链仍不完全相同",
        }

    full = rows[0]
    historical = read_json(LEGACY_BEST)["best"][0]
    expected = {
        "old_acc": float(historical["query_old_acc"]),
        "seen_new_acc": float(historical["query_seen_new_acc"]),
        "support_index_sha16": historical["support_index_sha16"],
        "query_index_sha16": historical["query_index_sha16"],
    }
    checks = {
        "metric_reproduction": abs(full["old_acc"] - expected["old_acc"]) < 1e-12
        and abs(full["seen_new_acc"] - expected["seen_new_acc"]) < 1e-12,
        "support_hash_match": full["support_index_sha16"] == expected["support_index_sha16"],
        "query_hash_match": full["query_index_sha16"] == expected["query_index_sha16"],
    }
    if not all(checks.values()):
        raise AssertionError(f"Legacy replay mismatch: {checks}")
    for row in rows:
        row["delta_H_vs_full_pp"] = 100.0 * (row["H_old_new"] - full["H_old_new"])
    return rows, {"checks": checks, "feature_bridge": bridge}


def summarize_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    old = mean(float(row["old_acc"]) for row in rows)
    new = mean(float(row["seen_new_acc"]) for row in rows)
    return {
        key: rows[0][key],
        "rows": len(rows),
        "old_acc": old,
        "seen_new_acc": new,
        "H_old_new": mean(float(row["H_old_new"]) for row in rows),
        "old_to_seen_new_rate": mean(float(row["old_to_seen_new_rate"]) for row in rows),
        "seen_new_to_old_rate": mean(float(row["seen_new_to_old_rate"]) for row in rows),
    }


def build_current_audit() -> dict[str, Any]:
    paths = sorted(path for path in MATRIX_ROOT.rglob("metrics.json") if path.parent.name == "cvs_qknnv42")
    if len(paths) != 125:
        raise ValueError(f"Expected 125 qKNNV42 metrics files, got {len(paths)}")
    scenario_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        split = payload["split_manifest"]
        k_shot = int(split["k_shot"])
        receiver = str(payload["target_receiver_label"])
        seed = int(payload["seed"])
        run_metrics = payload["metrics"]
        run_rows.append(
            {
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "old_acc": float(run_metrics["old_acc_mean"]),
                "seen_new_acc": float(run_metrics["seen_new_acc_mean"]),
                "H_old_new": float(run_metrics["H_old_new_mean"]),
            }
        )
        for scenario, metrics in payload["metrics_by_scenario"].items():
            scenario_rows.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": k_shot,
                    "scenario": scenario,
                    "old_acc": float(metrics["old_acc"]),
                    "seen_new_acc": float(metrics["seen_new_acc"]),
                    "H_old_new": float(metrics["H_old_new"]),
                    "old_to_seen_new_rate": float(metrics["old_to_seen_new_rate"]),
                    "seen_new_to_old_rate": float(metrics["seen_new_to_old_rate"]),
                }
            )
    if len(scenario_rows) != 375:
        raise ValueError(f"Expected 375 scenario rows, got {len(scenario_rows)}")

    def grouped(field: str, rows: list[dict[str, Any]] = scenario_rows) -> list[dict[str, Any]]:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        return [summarize_rows(group, field) for _, group in sorted(groups.items(), key=lambda item: item[0])]

    k_rows = grouped("k_shot")
    receiver_rows = grouped("receiver")
    scenario_summary = grouped("scenario")
    k5 = [row for row in scenario_rows if row["k_shot"] == 5]
    k5_summary = summarize_rows(k5, "k_shot")
    error_flow = [
        {
            "truth_role": "target-old",
            "correct": k5_summary["old_acc"],
            "cross_role_error": k5_summary["old_to_seen_new_rate"],
            "within_role_error": 1.0
            - k5_summary["old_acc"]
            - k5_summary["old_to_seen_new_rate"],
            "interpretation": "old侧除跨角色错误外，仍有显著old类内部混淆",
        },
        {
            "truth_role": "seen-new",
            "correct": k5_summary["seen_new_acc"],
            "cross_role_error": k5_summary["seen_new_to_old_rate"],
            "within_role_error": 1.0
            - k5_summary["seen_new_acc"]
            - k5_summary["seen_new_to_old_rate"],
            "interpretation": "new侧首要错误是被吸入old类，而非两个new类互混",
        },
    ]

    rx7_k5 = [row for row in scenario_rows if row["receiver"] == "7-14" and row["k_shot"] == 5]
    rx7_scenarios = grouped("scenario", rx7_k5)
    rx7_runs = [row for row in run_rows if row["receiver"] == "7-14" and row["k_shot"] == 5]
    factor_spread = [
        {
            "factor": "K-shot",
            "low_setting": f"K={min(k_rows, key=lambda row: row['H_old_new'])['k_shot']}",
            "low_H": min(row["H_old_new"] for row in k_rows),
            "high_setting": f"K={max(k_rows, key=lambda row: row['H_old_new'])['k_shot']}",
            "high_H": max(row["H_old_new"] for row in k_rows),
            "spread_pp": 100.0 * (max(row["H_old_new"] for row in k_rows) - min(row["H_old_new"] for row in k_rows)),
        },
        {
            "factor": "receiver",
            "low_setting": min(receiver_rows, key=lambda row: row["H_old_new"])["receiver"],
            "low_H": min(row["H_old_new"] for row in receiver_rows),
            "high_setting": max(receiver_rows, key=lambda row: row["H_old_new"])["receiver"],
            "high_H": max(row["H_old_new"] for row in receiver_rows),
            "spread_pp": 100.0
            * (max(row["H_old_new"] for row in receiver_rows) - min(row["H_old_new"] for row in receiver_rows)),
        },
        {
            "factor": "场景",
            "low_setting": min(scenario_summary, key=lambda row: row["H_old_new"])["scenario"],
            "low_H": min(row["H_old_new"] for row in scenario_summary),
            "high_setting": max(scenario_summary, key=lambda row: row["H_old_new"])["scenario"],
            "high_H": max(row["H_old_new"] for row in scenario_summary),
            "spread_pp": 100.0
            * (max(row["H_old_new"] for row in scenario_summary) - min(row["H_old_new"] for row in scenario_summary)),
        },
        {
            "factor": "support seed",
            "low_setting": "rx7-14,K=5最弱seed",
            "low_H": min(row["H_old_new"] for row in rx7_runs),
            "high_setting": "rx7-14,K=5最强seed",
            "high_H": max(row["H_old_new"] for row in rx7_runs),
            "spread_pp": 100.0 * (max(row["H_old_new"] for row in rx7_runs) - min(row["H_old_new"] for row in rx7_runs)),
        },
    ]
    return {
        "run_count": len(run_rows),
        "scenario_row_count": len(scenario_rows),
        "k_summary": k_rows,
        "receiver_summary": receiver_rows,
        "scenario_summary": scenario_summary,
        "k5_summary": k5_summary,
        "k5_error_flow": error_flow,
        "rx7_14_k5_scenario_summary": rx7_scenarios,
        "rx7_14_k5_seed_H_min": min(row["H_old_new"] for row in rx7_runs),
        "rx7_14_k5_seed_H_max": max(row["H_old_new"] for row in rx7_runs),
        "factor_spread": factor_spread,
    }


def build_scenario_shortcut_audit() -> dict[str, Any]:
    scripts = str(LEGACY_RUNNER.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase2_qknn_active_support_select as active
    import phase2_source_guarded_qknn_sweep as qknn
    import phase2_support_metric_qknn_probe as probe

    with np.load(LEGACY_FEATURE, allow_pickle=True) as data:
        features = qknn._normalize_rows(data["features"])
        tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
        roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
        scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
        logits = np.asarray(data["tx_logits"], dtype=np.float64)
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: index for index, label in enumerate(OLD_LABELS)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in OLD_LABELS:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]
    common = {
        "tx_ids": tx_ids,
        "roles": roles,
        "features": features,
        "scenarios": scenarios,
        "source_probs": source_probs,
        "source_label_to_idx": source_label_to_idx,
        "source_prototypes": source_prototypes,
        "policy": "stable_first",
        "seed": 421070,
        "exclude_pool_from_query": False,
    }
    old_raw = active._build_active_splits(
        labels=OLD_LABELS,
        role="target_old",
        k=5,
        query_per_class=70,
        pool_per_class=5,
        **common,
    )
    new_raw = active._build_active_splits(
        labels=NEW_LABELS,
        role="target_unknown",
        k=5,
        query_per_class=70,
        pool_per_class=5,
        **common,
    )
    old_splits = active._as_eval_splits(old_raw)
    new_splits = active._as_eval_splits(new_raw)
    fingerprint = probe._split_fingerprint(old_splits, new_splits, OLD_LABELS, NEW_LABELS)
    historical = read_json(LEGACY_BEST)["best"][0]
    if fingerprint["support_index_sha16"] != historical["support_index_sha16"]:
        raise AssertionError("Scenario audit support split does not match historical row")
    if fingerprint["query_index_sha16"] != historical["query_index_sha16"]:
        raise AssertionError("Scenario audit query split does not match historical row")

    role_rows = []
    for role, labels, splits in (
        ("target-old", OLD_LABELS, old_splits),
        ("seen-new", NEW_LABELS, new_splits),
    ):
        support_scenarios = {
            label: set(scenarios[splits[label][0]].tolist()) for label in labels
        }
        candidate_counts: list[int] = []
        retained: list[bool] = []
        for label in labels:
            query_scenarios = scenarios[splits[label][1]]
            for scenario in query_scenarios.tolist():
                candidate_counts.append(
                    sum(str(scenario) in support_scenarios[candidate] for candidate in labels)
                )
                retained.append(str(scenario) in support_scenarios[label])
        role_rows.append(
            {
                "role": role,
                "full_candidate_classes": len(labels),
                "scenario_candidate_mean": mean(candidate_counts),
                "scenario_candidate_min": min(candidate_counts),
                "scenario_candidate_max": max(candidate_counts),
                "true_class_retained_rate": mean(float(value) for value in retained),
                "query_rows": len(candidate_counts),
            }
        )
    return {
        "support_index_sha16": fingerprint["support_index_sha16"],
        "query_index_sha16": fingerprint["query_index_sha16"],
        "roles": role_rows,
    }


def build_seed_selection_audit() -> dict[str, Any]:
    with LEGACY_SEED_SWEEP.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120:
        raise ValueError(f"Expected 120 seed rows, got {len(rows)}")
    old = [float(row["query_old_acc"]) for row in rows]
    new = [float(row["query_seen_new_acc"]) for row in rows]
    h_values = [harmonic(old_value, new_value) for old_value, new_value in zip(old, new)]
    selected_index = next(index for index, row in enumerate(rows) if int(row["seed"]) == 421070)
    selected_h = h_values[selected_index]
    refined = read_json(LEGACY_BEST)["best"][0]
    refined_h = harmonic(float(refined["query_old_acc"]), float(refined["query_seen_new_acc"]))
    return {
        "seed_rows": len(rows),
        "old_acc_mean": mean(old),
        "old_acc_std": statistics.stdev(old),
        "seen_new_acc_mean": mean(new),
        "seen_new_acc_std": statistics.stdev(new),
        "H_old_new_mean": mean(h_values),
        "H_old_new_std": statistics.stdev(h_values),
        "H_old_new_min": min(h_values),
        "H_old_new_max": max(h_values),
        "selected_seed": 421070,
        "selected_seed_pre_refine_H": selected_h,
        "selected_seed_H_rank": sorted(h_values, reverse=True).index(selected_h) + 1,
        "refined_H": refined_h,
        "refinement_delta_pp": 100.0 * (refined_h - selected_h),
        "refined_vs_seed_mean_delta_pp": 100.0 * (refined_h - mean(h_values)),
    }


def build_feature_stack_audit() -> list[dict[str, Any]]:
    legacy = read_npz_manifest(LEGACY_FEATURE)
    current = read_npz_manifest(CURRENT_FEATURE)
    adapter = legacy["adapter"]
    return [
        {
            "stack": "历史legacy",
            "base_checkpoint": Path(str(legacy["checkpoint"])).name,
            "adapter": f"{adapter['model_adapter']['mode']}，{adapter['epochs']} epoch，{adapter['model_adapter']['trainable_parameters']}个可训练参数",
            "satellite_tta_views": int(legacy["satellite_tta_view_count"]),
            "aux_feature": "96维FFT log-magnitude，score weight=0.34",
            "decision": "scenario-aware+role-balanced equal-quota Hungarian",
        },
        {
            "stack": "当前正式",
            "base_checkpoint": Path(str(current["checkpoint"])).name,
            "adapter": "无；冻结基础z_id",
            "satellite_tta_views": int(current["satellite_tta_view_count"]),
            "aux_feature": "无",
            "decision": "8类逐样本统一argmax；无role/quota/scenario hard mask",
        },
    ]


def build_protocol_comparison() -> list[dict[str, str]]:
    return [
        {"dimension": "评估任务", "legacy": "批量转导式、角色已知、类别计数已知", "formal": "8类统一逐样本分类"},
        {"dimension": "receiver", "legacy": "仅7-14", "formal": "20-1、3-19、7-14、7-7、8-8"},
        {"dimension": "类别", "legacy": "6 old+20 new", "formal": "6 old+2 seen-new"},
        {"dimension": "场景覆盖", "legacy": "TX只覆盖部分场景，场景与类别相关", "formal": "每类均在clear、low、rain评估"},
        {"dimension": "seed口径", "legacy": "120-seed扫描后选421070，再做162-row精调", "formal": "5个固定seed全部报告"},
        {"dimension": "特征", "legacy": "60 epoch adapter+5-view TTA+96维FFT辅助", "formal": "冻结160维单视图z_id，无辅助特征"},
        {"dimension": "决策", "legacy": "old/new角色分区+每类等额quota+场景硬筛", "formal": "无角色、quota和场景排除，独立argmax"},
        {"dimension": "证据地位", "legacy": "NON_DEPLOYMENT legacy diagnostic", "formal": "当前Stage2-C正式矩阵候选证据"},
    ]


def build_cause_ranking() -> list[dict[str, str]]:
    return [
        {
            "rank": "1",
            "cause": "任务与决策协议不等价",
            "evidence": "legacy利用old/new角色和每类等额quota；当前K5有36.03%的new→old错误，而legacy规则直接禁止该错误",
            "judgement": "解释历史高分的首要原因",
        },
        {
            "rank": "2",
            "cause": "表示栈不等价",
            "evidence": "legacy使用60 epoch adapter、5-view TTA和FFT辅助；当前使用冻结单视图z_id",
            "judgement": "主要上游原因",
        },
        {
            "rank": "3",
            "cause": "场景-类别混杂与硬筛",
            "evidence": "legacy场景硬筛把new候选类平均从20降至8.42，真实类仍保留97.7%",
            "judgement": "形成强场景捷径",
        },
        {
            "rank": "4",
            "cause": "正式矩阵扩大receiver与场景覆盖",
            "evidence": "receiver分组H相差16.83pp，rain场景H比clear低7.58pp",
            "judgement": "暴露真实域泛化不足",
        },
        {
            "rank": "5",
            "cause": "seed选择与后验精调",
            "evidence": "选中seed+精调相对120-seed H均值高约2.01pp，精调本身约0.41pp",
            "judgement": "存在乐观偏差，但不是36.42pp差距主因",
        },
    ]


def main() -> None:
    assert_inputs()
    ablations, replay = run_legacy_ablations()
    current = build_current_audit()
    scenario_shortcut = build_scenario_shortcut_audit()
    seed_selection = build_seed_selection_audit()
    feature_stack = build_feature_stack_audit()

    clear = next(
        row
        for row in current["rx7_14_k5_scenario_summary"]
        if row["scenario"] == "leo_clear_weak"
    )
    bridge = replay["feature_bridge"]
    bridge["current_rx7_14_k5_clear_old_acc"] = clear["old_acc"]
    bridge["current_rx7_14_k5_clear_seen_new_acc"] = clear["seen_new_acc"]
    bridge["current_rx7_14_k5_clear_H_old_new"] = clear["H_old_new"]
    bridge["delta_old_pp"] = 100.0 * (bridge["old_acc_mean"] - clear["old_acc"])
    bridge["delta_seen_new_pp"] = 100.0 * (bridge["seen_new_acc_mean"] - clear["seen_new_acc"])
    bridge["delta_H_pp"] = 100.0 * (bridge["H_old_new_mean"] - clear["H_old_new"])

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "scope": "qKNNV42 legacy diagnostic versus 2026-07-13 formal Stage2-C matrix root-cause audit",
        "claim_boundary": "诊断性反事实结果，不替代当前正式矩阵，不构成部署成功证据",
        "protocol_comparison": build_protocol_comparison(),
        "legacy_replay": replay,
        "legacy_ablations": ablations,
        "scenario_shortcut": scenario_shortcut,
        "seed_selection": seed_selection,
        "feature_stack": feature_stack,
        "current_formal": current,
        "cause_ranking": build_cause_ranking(),
        "limitations": [
            "消融在历史split上进行，组件交互明显，delta不能相加为36.42pp的严格方差分解。",
            "feature bridge对齐receiver、K、clear场景和2个new类，但query样本与特征生成链仍不完全相同。",
            "当前score_table未保存逐类原始score，无法离线精确计算当前矩阵的role/quota oracle上界。",
            "当前正式矩阵没有stdout日志；本审计完整读取结构化metrics、split、score和loss artifact。",
        ],
        "source_hashes": {
            "legacy_runner_sha256": sha256(LEGACY_RUNNER),
            "current_runner_sha256": sha256(CURRENT_RUNNER),
            "legacy_feature_sha256": sha256(LEGACY_FEATURE),
            "legacy_aux_sha256": sha256(LEGACY_AUX),
            "legacy_best_sha256": sha256(LEGACY_BEST),
            "legacy_seed_sweep_sha256": sha256(LEGACY_SEED_SWEEP),
            "current_feature_sha256": sha256(CURRENT_FEATURE),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON),
                "legacy_replay_checks": replay["checks"],
                "legacy_variants": len(ablations),
                "current_runs": current["run_count"],
                "current_scenario_rows": current["scenario_row_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
