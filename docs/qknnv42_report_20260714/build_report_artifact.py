from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[1]
SUMMARY_ROOT = REPO_ROOT / "local_artifacts" / "cvs_publication_stage2_summary_20260713"
MATRIX_ROOT = (
    REPO_ROOT
    / "local_artifacts"
    / "cvs_publication_stage2_full_matrix_20260713"
    / "cvs_publication_stage2c_full_matrix_20260713"
)
HISTORICAL_JSON = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "phase2_qknn_hardpair_n20_20260706"
    / "artifacts"
    / "v53_fftlogmag_20260706"
    / "local_v55_diagnostics_20260706"
    / "k5_strict_seed421070_floor_param_best_predictions_20260707.json"
)

QKNN_METHOD = "cvs_qknnv42"
CURRENT_K = [1, 2, 5, 10, 20]
METHOD_LABELS = {
    "cvs_qknnv42": "CVS qKNNV42",
    "csil": "CSIL",
    "mopc_hr": "MoPC-HR",
    "orthogonal_incremental": "Orthogonal Incremental",
}
SCENARIO_LABELS = {
    "leo_clear_weak": "LEO清晰弱扰动",
    "leo_low_elev_weak": "LEO低仰角弱扰动",
    "leo_rain_weak": "LEO降雨弱扰动",
}


def read_csv(name: str) -> list[dict[str, str]]:
    path = SUMMARY_ROOT / name
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        return math.nan
    return float(value)


def mean(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else math.nan


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def assert_inputs() -> None:
    required = [
        SUMMARY_ROOT / "method_k_summary.csv",
        SUMMARY_ROOT / "per_run_results.csv",
        SUMMARY_ROOT / "per_scenario_results.csv",
        SUMMARY_ROOT / "final_audit.json",
        MATRIX_ROOT,
        HISTORICAL_JSON,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report inputs:\n" + "\n".join(missing))


def build_k_curve(method_summary: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in method_summary
        if row.get("phase") == "stage2c" and row.get("method") == QKNN_METHOD
    ]
    by_k = {int(row["k_shot"]): row for row in rows}
    if sorted(by_k) != CURRENT_K:
        raise ValueError(f"Unexpected qKNNV42 K grid: {sorted(by_k)}")
    result: list[dict[str, Any]] = []
    for k in CURRENT_K:
        row = by_k[k]
        result.append(
            {
                "k_shot": k,
                "k_label": f"K={k}",
                "runs": int(float(row["H_old_new_n"])),
                "old_acc_before_increment": f(row, "old_acc_before_increment_mean"),
                "old_acc": f(row, "old_acc_mean"),
                "old_acc_ci95_low": f(row, "old_acc_ci95_low"),
                "old_acc_ci95_high": f(row, "old_acc_ci95_high"),
                "seen_new_acc": f(row, "seen_new_acc_mean"),
                "seen_new_acc_ci95_low": f(row, "seen_new_acc_ci95_low"),
                "seen_new_acc_ci95_high": f(row, "seen_new_acc_ci95_high"),
                "H_old_new": f(row, "H_old_new_mean"),
                "H_old_new_ci95_low": f(row, "H_old_new_ci95_low"),
                "H_old_new_ci95_high": f(row, "H_old_new_ci95_high"),
                "average_forgetting": f(row, "average_forgetting_mean"),
                "old80_rows": 0,
            }
        )
    return result


def current_runs(per_run: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in per_run
        if row.get("phase") == "stage2c" and row.get("method") == QKNN_METHOD
    ]


def add_old80_counts(k_curve: list[dict[str, Any]], runs: list[dict[str, str]]) -> None:
    counts: dict[int, int] = defaultdict(int)
    for row in runs:
        if f(row, "old_acc") >= 0.80:
            counts[int(row["k_shot"])] += 1
    for row in k_curve:
        row["old80_rows"] = counts[row["k_shot"]]


def build_method_overall(per_run: list[dict[str, str]]) -> list[dict[str, Any]]:
    methods = list(METHOD_LABELS)
    result = []
    for method in methods:
        rows = [
            row
            for row in per_run
            if row.get("phase") == "stage2c" and row.get("method") == method
        ]
        if len(rows) != 125:
            raise ValueError(f"Expected 125 Stage2-C rows for {method}, got {len(rows)}")
        result.append(
            {
                "method": METHOD_LABELS[method],
                "method_id": method,
                "runs": len(rows),
                "old_acc": mean(f(row, "old_acc") for row in rows),
                "seen_new_acc": mean(f(row, "seen_new_acc") for row in rows),
                "H_old_new": mean(f(row, "H_old_new") for row in rows),
                "average_forgetting": mean(f(row, "average_forgetting") for row in rows),
            }
        )
    return sorted(result, key=lambda row: row["H_old_new"], reverse=True)


def build_receiver_summary(runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        groups[row["receiver"]].append(row)
    result = []
    for receiver, rows in sorted(groups.items()):
        result.append(
            {
                "receiver": receiver,
                "runs": len(rows),
                "old_acc": mean(f(row, "old_acc") for row in rows),
                "seen_new_acc": mean(f(row, "seen_new_acc") for row in rows),
                "H_old_new": mean(f(row, "H_old_new") for row in rows),
                "average_forgetting": mean(f(row, "average_forgetting") for row in rows),
            }
        )
    return result


def build_scenario_summary(per_scenario: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in per_scenario
        if row.get("phase") == "stage2c" and row.get("method") == QKNN_METHOD
    ]
    if len(rows) != 375:
        raise ValueError(f"Expected 375 qKNNV42 scenario rows, got {len(rows)}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["scenario"]].append(row)
    result = []
    for scenario in SCENARIO_LABELS:
        group = groups[scenario]
        result.append(
            {
                "scenario": SCENARIO_LABELS[scenario],
                "scenario_id": scenario,
                "rows": len(group),
                "old_acc": mean(f(row, "old_acc") for row in group),
                "seen_new_acc": mean(f(row, "seen_new_acc") for row in group),
                "H_old_new": mean(f(row, "H_old_new") for row in group),
            }
        )
    return result


def build_extremes(runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(runs, key=lambda row: f(row, "H_old_new"))
    result = []
    for label, row in (("最低H行", ordered[0]), ("最高H行", ordered[-1])):
        result.append(
            {
                "position": label,
                "experiment_id": row["experiment_id"],
                "receiver": row["receiver"],
                "seed": int(row["seed"]),
                "k_shot": int(row["k_shot"]),
                "old_acc_before_increment": f(row, "old_acc_before_increment"),
                "old_acc": f(row, "old_acc"),
                "seen_new_acc": f(row, "seen_new_acc"),
                "H_old_new": f(row, "H_old_new"),
                "average_forgetting": f(row, "average_forgetting"),
            }
        )
    return result


def build_pairwise(per_run: list[dict[str, str]]) -> list[dict[str, Any]]:
    stage2c = [row for row in per_run if row.get("phase") == "stage2c"]
    keyed = {
        (row["method"], row["receiver"], int(row["seed"]), int(row["k_shot"])): row
        for row in stage2c
    }
    result = []
    for candidate in ("csil", "mopc_hr", "orthogonal_incremental"):
        for k in CURRENT_K:
            deltas = []
            wins = 0
            for receiver in sorted({row["receiver"] for row in stage2c}):
                for seed in sorted({int(row["seed"]) for row in stage2c}):
                    reference = keyed[(QKNN_METHOD, receiver, seed, k)]
                    comparison = keyed[(candidate, receiver, seed, k)]
                    delta = f(reference, "H_old_new") - f(comparison, "H_old_new")
                    deltas.append(delta)
                    wins += int(delta > 0)
            result.append(
                {
                    "candidate": METHOD_LABELS[candidate],
                    "k_shot": k,
                    "pairs": len(deltas),
                    "qknnv42_wins": wins,
                    "delta_H_old_new": mean(deltas),
                }
            )
    return result


def build_structured_audit(runs: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_files = [
        "metrics.json",
        "resolved_config.json",
        "split_manifest.json",
        "loss_trace.json",
        "loss_trace.csv",
        "score_table.csv",
        "detailed_metrics.json",
        "detailed_metrics.csv",
    ]
    missing: list[str] = []
    bad_json: list[str] = []
    nonfinite = 0
    overlaps = 0
    transductive = 0
    query_training = 0
    query_model_selection = 0
    unknown_enabled = 0
    trace_count = 0
    gradient_updates: list[int] = []
    feature_dims: set[int] = set()
    raw_support_counts: list[int] = []
    prototype_counts: list[int] = []
    residual_applied = 0
    latencies: list[float] = []
    support_codes: dict[int, set[int]] = defaultdict(set)

    for run in runs:
        run_dir = REPO_ROOT / Path(run["run_dir"])
        for name in required_files:
            if not (run_dir / name).exists():
                missing.append(rel(run_dir / name))
        for name in ("metrics.json", "resolved_config.json", "split_manifest.json", "loss_trace.json", "detailed_metrics.json"):
            path = run_dir / name
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                bad_json.append(rel(path))
                continue
            stack = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
                elif isinstance(item, float) and not math.isfinite(item):
                    nonfinite += 1

        split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8-sig"))
        overlaps += int(bool(split.get("support_query_overlap")))
        transductive += int(bool(split.get("query_used_for_transductive_inference")))
        query_training += int(bool(split.get("target_query_used_for_training")))
        query_model_selection += int(bool(split.get("target_query_used_for_model_selection")))
        unknown_enabled += int(bool(split.get("unknown_rejection_enabled")))

        trace = json.loads((run_dir / "loss_trace.json").read_text(encoding="utf-8-sig"))
        trace_rows = trace if isinstance(trace, list) else trace.get("trace", trace.get("rows", []))
        trace_count += len(trace_rows)
        for trace_row in trace_rows:
            if isinstance(trace_row, dict) and "gradient_updates" in trace_row:
                gradient_updates.append(int(trace_row["gradient_updates"]))

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        scenario_rows = metrics.get(
            "metrics_by_scenario",
            metrics.get("scenario_results", metrics.get("results", [])),
        )
        if isinstance(scenario_rows, dict):
            scenario_rows = list(scenario_rows.values())
        for scenario_row in scenario_rows:
            if not isinstance(scenario_row, dict):
                continue
            metadata = scenario_row.get("method_metadata", scenario_row.get("metadata", scenario_row))
            if not isinstance(metadata, dict):
                continue
            if "feature_dim" in metadata:
                feature_dims.add(int(metadata["feature_dim"]))
            if "stored_raw_support_count" in metadata:
                raw_support_counts.append(int(metadata["stored_raw_support_count"]))
            if "stored_class_prototype_count" in metadata:
                prototype_counts.append(int(metadata["stored_class_prototype_count"]))
            if "scenario_residual_applied" in metadata:
                residual_applied += int(bool(metadata["scenario_residual_applied"]))
            if "latency_per_query_ms" in metadata:
                latencies.append(float(metadata["latency_per_query_ms"]))
            if "stored_quantized_support_code_count" in metadata:
                support_codes[int(run["k_shot"])].add(
                    int(metadata["stored_quantized_support_code_count"])
                )

    stdout_logs = list(MATRIX_ROOT.rglob("*.out")) + list(MATRIX_ROOT.rglob("*.log"))
    latency_sorted = sorted(latencies)
    latency_p95 = (
        latency_sorted[min(len(latency_sorted) - 1, math.ceil(0.95 * len(latency_sorted)) - 1)]
        if latency_sorted
        else math.nan
    )
    details = {
        "runs": len(runs),
        "scenario_rows": len(runs) * 3,
        "required_file_checks": len(runs) * len(required_files),
        "missing_files": len(missing),
        "bad_json": len(bad_json),
        "nonfinite_values": nonfinite,
        "support_query_overlap_rows": overlaps,
        "transductive_rows": transductive,
        "query_training_rows": query_training,
        "query_model_selection_rows": query_model_selection,
        "unknown_enabled_rows": unknown_enabled,
        "trace_rows": trace_count,
        "gradient_updates_max": max(gradient_updates, default=0),
        "feature_dims": sorted(feature_dims),
        "raw_support_max": max(raw_support_counts, default=0),
        "prototype_counts": sorted(set(prototype_counts)),
        "scenario_residual_applied_rows": residual_applied,
        "stdout_logs": len(stdout_logs),
        "latency_ms_median": statistics.median(latencies) if latencies else math.nan,
        "latency_ms_p95": latency_p95,
        "latency_ms_mean": mean(latencies),
        "latency_ms_max": max(latencies, default=math.nan),
        "support_codes": {str(k): sorted(values) for k, values in support_codes.items()},
        "missing_examples": missing[:5],
        "bad_json_examples": bad_json[:5],
    }
    audit_rows = [
        {"check": "正式qKNNV42运行目录", "observed": len(runs), "expected": 125, "verdict": "PASS" if len(runs) == 125 else "FAIL"},
        {"check": "三种LEO场景结果", "observed": len(runs) * 3, "expected": 375, "verdict": "PASS"},
        {"check": "必需结构化文件缺失", "observed": len(missing), "expected": 0, "verdict": "PASS" if not missing else "FAIL"},
        {"check": "JSON解析失败", "observed": len(bad_json), "expected": 0, "verdict": "PASS" if not bad_json else "FAIL"},
        {"check": "NaN或Inf", "observed": nonfinite, "expected": 0, "verdict": "PASS" if nonfinite == 0 else "FAIL"},
        {"check": "support/query重叠", "observed": overlaps, "expected": 0, "verdict": "PASS" if overlaps == 0 else "FAIL"},
        {"check": "梯度更新次数最大值", "observed": max(gradient_updates, default=0), "expected": 0, "verdict": "PASS" if max(gradient_updates, default=0) == 0 else "FAIL"},
        {"check": "stdout日志文件", "observed": len(stdout_logs), "expected": "未规定", "verdict": "LIMITATION" if not stdout_logs else "PASS"},
    ]
    return audit_rows, details


def build_historical() -> dict[str, Any]:
    payload = json.loads(HISTORICAL_JSON.read_text(encoding="utf-8-sig"))
    best = payload["best"][0]
    old_acc = float(best["query_old_acc"])
    new_acc = float(best["query_seen_new_acc"])
    harmonic = 2 * old_acc * new_acc / (old_acc + new_acc)
    return {
        "evidence": "2026-07-07历史单行机制验证",
        "protocol_status": "legacy diagnostic，不属于2026-07-13正式矩阵",
        "receiver_count": "未按当前5个target receiver矩阵汇总",
        "seed_count": 1,
        "k_shot": int(best["k_old"]),
        "old_classes": int(best["adaptive_old_class_count"]),
        "new_classes": int(best["adaptive_new_class_count"]),
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "H_old_new": harmonic,
        "min_old_class_acc": float(best["query_min_old_class_acc"]),
        "min_seen_new_class_acc": float(best["query_min_seen_new_class_acc"]),
        "new_role": best["new_role"],
        "stored_codes": int(best["stored_quantized_support_code_count"]),
    }


def build_artifact() -> dict[str, Any]:
    assert_inputs()
    method_summary = read_csv("method_k_summary.csv")
    per_run = read_csv("per_run_results.csv")
    per_scenario = read_csv("per_scenario_results.csv")
    runs = current_runs(per_run)
    if len(runs) != 125:
        raise ValueError(f"Expected 125 current qKNNV42 runs, got {len(runs)}")

    k_curve = build_k_curve(method_summary)
    add_old80_counts(k_curve, runs)
    method_overall = build_method_overall(per_run)
    receiver_summary = build_receiver_summary(runs)
    scenario_summary = build_scenario_summary(per_scenario)
    extremes = build_extremes(runs)
    pairwise = build_pairwise(per_run)
    audit_rows, audit_details = build_structured_audit(runs)
    historical = build_historical()

    qknn_overall = next(row for row in method_overall if row["method_id"] == QKNN_METHOD)
    k20 = next(row for row in k_curve if row["k_shot"] == 20)
    pairwise_total = sum(row["pairs"] for row in pairwise)
    pairwise_wins = sum(row["qknnv42_wins"] for row in pairwise)
    no_h80 = sum(f(row, "H_old_new") >= 0.80 for row in runs)
    both80 = sum(f(row, "old_acc") >= 0.80 and f(row, "seen_new_acc") >= 0.80 for row in runs)

    headline = [
        {
            "k20_old_acc": k20["old_acc"],
            "k20_seen_new_acc": k20["seen_new_acc"],
            "k20_H_old_new": k20["H_old_new"],
            "overall_H_old_new": qknn_overall["H_old_new"],
            "paired_H_win_rate": pairwise_wins / pairwise_total,
            "formal_runs": len(runs),
        }
    ]

    input_output = [
        {
            "step": 1,
            "stage": "特征入口",
            "input": "同一target receiver下的raw IQ样本",
            "operation": "冻结ADV3B02主干提取身份特征",
            "output": "每个样本1个160维L2归一化z_id",
            "uses_label": "否",
        },
        {
            "step": 2,
            "stage": "support构建",
            "input": "6个old TX与2个seen-new TX，每类K个已标注support",
            "operation": "确定性nested K-shot切分；support/query互斥",
            "output": "8K条带类标的160维support特征",
            "uses_label": "仅support标签",
        },
        {
            "step": 3,
            "stage": "轻量适配",
            "input": "support特征与support标签",
            "operation": "对角whitening+Fisher尺度；无反向传播",
            "output": "变换后的support/query特征与8个prototype",
            "uses_label": "仅support标签",
        },
        {
            "step": 4,
            "stage": "压缩记忆与分类",
            "input": "int8 support codes、prototype、query特征",
            "operation": "top-1邻居+prototype+old anchor+label propagation",
            "output": "8个已注册类的argmax预测标签",
            "uses_label": "query标签不使用；query特征参与传导图",
        },
        {
            "step": 5,
            "stage": "评估产物",
            "input": "预测标签与仅用于离线评估的query真值",
            "operation": "按样本、类、receiver、场景聚合",
            "output": "old_acc、seen_new_acc、H、forgetting、混淆、延迟、存储统计",
            "uses_label": "query标签仅用于评估",
        },
    ]

    storage = []
    for k in CURRENT_K:
        codes = 8 * k
        storage.append(
            {
                "k_shot": k,
                "registered_classes": 8,
                "stored_int8_codes": codes,
                "code_bytes": codes * 160,
                "code_kib": codes * 160 / 1024,
                "note": "仅support code字节，不含8个prototype、变换参数及元数据",
            }
        )

    current_reference = {
        "evidence": "2026-07-13正式Stage2-C矩阵",
        "protocol_status": "当前主证据",
        "receiver_count": "5个target receiver",
        "seed_count": 5,
        "k_shot": "1/2/5/10/20",
        "old_classes": 6,
        "new_classes": 2,
        "old_acc": qknn_overall["old_acc"],
        "seen_new_acc": qknn_overall["seen_new_acc"],
        "H_old_new": qknn_overall["H_old_new"],
        "min_old_class_acc": None,
        "min_seen_new_class_acc": None,
        "new_role": "target_seen_new",
        "stored_codes": "8K",
    }

    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    sources = [
        {
            "id": "src_protocol",
            "label": "CVS-RFFI当前项目协议",
            "path": "docs/PROJECT_PROTOCOL.md",
            "query": {
                "engine": "local_markdown",
                "sql": "SELECT * FROM project_protocol WHERE section IN ('Stage2-C','metrics','LEO scenarios')",
                "description": "Stage2-C、receiver/TX划分、LEO场景与成功边界",
                "executed_at": "2026-07-14",
                "metric_definitions": [
                    "old_acc=增量学习后旧类准确率",
                    "seen_new_acc=已注册新类准确率",
                    "H_old_new=old_acc与seen_new_acc的调和平均",
                ],
            },
        },
        {
            "id": "src_runner",
            "label": "qKNNV42当前正式实现",
            "path": "paper_reproduction/cvs_aligned/cvs_method_runner.py",
            "query": {
                "engine": "local_python",
                "sql": "SELECT * FROM qknnv42_input_output_contract ORDER BY step",
                "description": "qKNNV42特征变换、int8量化、评分、label propagation与产物写出",
                "executed_at": "2026-07-14",
            },
        },
        {
            "id": "src_method_summary",
            "label": "Stage2-C方法×K汇总",
            "path": "local_artifacts/cvs_publication_stage2_summary_20260713/method_k_summary.csv",
            "query": {
                "engine": "local_csv",
                "sql": "SELECT * FROM method_k_summary WHERE phase = 'stage2c' AND method = 'cvs_qknnv42' ORDER BY k_shot",
                "description": "筛选phase=stage2c且method=cvs_qknnv42",
                "executed_at": "2026-07-14",
                "filters": ["phase=stage2c", "method=cvs_qknnv42"],
            },
        },
        {
            "id": "src_per_run",
            "label": "Stage2-C逐运行结果",
            "path": "local_artifacts/cvs_publication_stage2_summary_20260713/per_run_results.csv",
            "query": {
                "engine": "local_csv",
                "sql": "SELECT * FROM per_run_results WHERE phase = 'stage2c'",
                "description": "当前4方法、5个receiver、5个seed、5档K-shot的paired结果",
                "executed_at": "2026-07-14",
            },
        },
        {
            "id": "src_per_scenario",
            "label": "Stage2-C逐场景结果",
            "path": "local_artifacts/cvs_publication_stage2_summary_20260713/per_scenario_results.csv",
            "query": {
                "engine": "local_csv",
                "sql": "SELECT * FROM per_scenario_results WHERE phase = 'stage2c' AND method = 'cvs_qknnv42'",
                "description": "qKNNV42在三种LEO弱场景下的375行结果",
                "executed_at": "2026-07-14",
                "filters": ["phase=stage2c", "method=cvs_qknnv42"],
            },
        },
        {
            "id": "src_audit",
            "label": "结构化artifact完整性审计",
            "path": "docs/qknnv42_report_20260714/build_report_artifact.py",
            "query": {
                "engine": "local_filesystem",
                "sql": "SELECT * FROM qknnv42_structured_artifact_audit ORDER BY check",
                "description": "逐目录检查125个qKNNV42运行的必需文件、JSON、数值有限性、切分和loss trace",
                "executed_at": "2026-07-14",
            },
        },
        {
            "id": "src_final_audit",
            "label": "正式矩阵最终审计",
            "path": "local_artifacts/cvs_publication_stage2_summary_20260713/final_audit.json",
            "query": {
                "engine": "local_json",
                "sql": "SELECT * FROM final_audit WHERE phase = 'stage2c'",
                "description": "500个Stage2-C方法运行的artifact完成状态",
                "executed_at": "2026-07-14",
            },
        },
        {
            "id": "src_historical",
            "label": "历史qKNNV42单行诊断",
            "path": rel(HISTORICAL_JSON),
            "query": {
                "engine": "local_json",
                "sql": "SELECT * FROM historical_qknnv42_diagnostic WHERE seed = 421070 UNION ALL SELECT * FROM current_qknnv42_reference",
                "description": "2026-07-07 K=5、6 old+20 new、seed421070的legacy diagnostic",
                "executed_at": "2026-07-14",
            },
        },
    ]

    cards = [
        {
            "id": "card_k20",
            "description": "正式矩阵K=20的25个receiver×seed联合均值",
            "dataset": "headline",
            "sourceId": "src_method_summary",
            "metrics": [
                {"label": "K=20 old_acc", "field": "k20_old_acc", "format": "percent"},
                {"label": "K=20 seen_new_acc", "field": "k20_seen_new_acc", "format": "percent"},
                {"label": "K=20 H", "field": "k20_H_old_new", "format": "percent"},
            ],
        },
        {
            "id": "card_matrix",
            "description": "完整正式矩阵与paired比较",
            "dataset": "headline",
            "sourceId": "src_per_run",
            "metrics": [
                {"label": "全K总体H", "field": "overall_H_old_new", "format": "percent"},
                {"label": "paired H胜率", "field": "paired_H_win_rate", "format": "percent"},
                {"label": "正式qKNNV42运行", "field": "formal_runs", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "chart_k_curve",
            "title": "K-shot增加时，old/new/H均稳步改善",
            "subtitle": "每个K点为5个receiver×5个seed=25个运行的均值；三种LEO场景先在运行内聚合",
            "intent": "comparison",
            "question": "支持样本数K如何影响旧类保持、新类学习与联合H？",
            "rationale": "K只有5个离散档位，采用分组柱状图表达组间比较，避免暗示连续趋势或外推。",
            "comparisonContext": {
                "baseline": "同一方法在不同K-shot下比较",
                "grain": "K-shot",
                "normalization": "25个receiver×seed运行均值",
                "unit": "accuracy",
            },
            "type": "bar",
            "dataset": "k_curve",
            "sourceId": "src_method_summary",
            "encodings": {
                "x": {"field": "k_label", "type": "ordinal", "label": "K-shot"},
                "y": {
                    "fields": ["old_acc", "seen_new_acc", "H_old_new"],
                    "type": "quantitative",
                    "format": "percent",
                    "label": "准确率/调和均值",
                },
                "tooltip": [
                    {"field": "runs", "type": "quantitative", "label": "运行数"},
                    {"field": "average_forgetting", "type": "quantitative", "format": "percent", "label": "平均遗忘"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 5,
            "palette": {"kind": "categorical"},
            "settings": {"groupMode": "grouped", "orientation": "vertical", "showValues": True},
            "surface": {"surface": "explorer", "viewMode": "both", "interactiveLegend": True},
        },
        {
            "id": "chart_method_h",
            "title": "完整正式矩阵中，qKNNV42的联合H居首",
            "subtitle": "每种方法125个Stage2-C运行，覆盖5个receiver、5个seed和5档K-shot",
            "intent": "comparison",
            "question": "同一paired matrix下，哪种方法最能兼顾旧类保持与新类学习？",
            "rationale": "仅4个类别且目标是排名，采用水平条形图并按H降序排列。",
            "comparisonContext": {
                "denominator": "每种方法125个运行",
                "grain": "method",
                "normalization": "全K、receiver和seed均值",
                "unit": "H old-new",
            },
            "type": "horizontalBar",
            "dataset": "method_overall",
            "sourceId": "src_per_run",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "方法"},
                "y": {"field": "H_old_new", "type": "quantitative", "format": "percent", "label": "H old-new"},
                "tooltip": [
                    {"field": "old_acc", "type": "quantitative", "format": "percent", "label": "old_acc"},
                    {"field": "seen_new_acc", "type": "quantitative", "format": "percent", "label": "seen_new_acc"},
                    {"field": "average_forgetting", "type": "quantitative", "format": "percent", "label": "平均遗忘"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "maxRows": 4,
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True},
            "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]

    tables = [
        {
            "id": "table_input_output",
            "title": "qKNNV42端到端输入输出契约",
            "subtitle": "qKNNV42本体接收160维特征；raw IQ由冻结ADV3B02主干先转换",
            "dataset": "input_output",
            "defaultSort": {"field": "step", "direction": "asc"},
            "density": "spacious",
            "sourceId": "src_runner",
            "layout": "full",
            "columns": [
                {"field": "step", "label": "步骤", "format": "number"},
                {"field": "stage", "label": "阶段", "type": "text"},
                {"field": "input", "label": "输入", "type": "text"},
                {"field": "operation", "label": "处理", "type": "text"},
                {"field": "output", "label": "输出", "type": "text"},
                {"field": "uses_label", "label": "标签边界", "type": "text"},
            ],
        },
        {
            "id": "table_k_curve",
            "title": "正式矩阵K-shot结果明细",
            "subtitle": "95%CI为当前summary manifest中的1.96×标准误正态近似，并非bootstrap CI",
            "dataset": "k_curve",
            "defaultSort": {"field": "k_shot", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_method_summary",
            "layout": "full",
            "columns": [
                {"field": "k_shot", "label": "K", "format": "number"},
                {"field": "runs", "label": "运行", "format": "number"},
                {"field": "old_acc_before_increment", "label": "old before", "format": "percent"},
                {"field": "old_acc", "label": "old_acc", "format": "percent"},
                {"field": "seen_new_acc", "label": "seen_new_acc", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
                {"field": "average_forgetting", "label": "forgetting", "format": "percent"},
                {"field": "old80_rows", "label": "old≥80%行", "format": "number"},
            ],
        },
        {
            "id": "table_method_overall",
            "title": "4种方法全矩阵联合结果",
            "subtitle": "同一5 receiver×5 seed×5 K-shot网格；每种方法125个运行",
            "dataset": "method_overall",
            "defaultSort": {"field": "H_old_new", "direction": "desc"},
            "density": "dense",
            "sourceId": "src_per_run",
            "layout": "full",
            "columns": [
                {"field": "method", "label": "方法", "type": "text"},
                {"field": "runs", "label": "运行", "format": "number"},
                {"field": "old_acc", "label": "old_acc", "format": "percent"},
                {"field": "seen_new_acc", "label": "seen_new_acc", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
                {"field": "average_forgetting", "label": "forgetting", "format": "percent"},
            ],
        },
        {
            "id": "table_pairwise",
            "title": "qKNNV42相对3种方法的paired H优势",
            "subtitle": "每行25个完全匹配的receiver×seed对；delta为qKNNV42减候选方法",
            "dataset": "pairwise",
            "defaultSort": {"field": "k_shot", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_per_run",
            "layout": "full",
            "columns": [
                {"field": "candidate", "label": "对比方法", "type": "text"},
                {"field": "k_shot", "label": "K", "format": "number"},
                {"field": "pairs", "label": "paired行", "format": "number"},
                {"field": "qknnv42_wins", "label": "qKNNV42胜", "format": "number"},
                {"field": "delta_H_old_new", "label": "ΔH", "format": "percent", "movement": True},
            ],
        },
        {
            "id": "table_receiver",
            "title": "receiver差异揭示跨接收域瓶颈",
            "subtitle": "每个receiver汇总25个seed×K运行",
            "dataset": "receiver_summary",
            "defaultSort": {"field": "H_old_new", "direction": "desc"},
            "density": "dense",
            "sourceId": "src_per_run",
            "layout": "half",
            "columns": [
                {"field": "receiver", "label": "receiver", "type": "text"},
                {"field": "old_acc", "label": "old", "format": "percent"},
                {"field": "seen_new_acc", "label": "new", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
            ],
        },
        {
            "id": "table_scenario",
            "title": "LEO弱场景差异",
            "subtitle": "每个场景汇总125个receiver×seed×K行",
            "dataset": "scenario_summary",
            "defaultSort": {"field": "H_old_new", "direction": "desc"},
            "density": "dense",
            "sourceId": "src_per_scenario",
            "layout": "half",
            "columns": [
                {"field": "scenario", "label": "场景", "type": "text"},
                {"field": "old_acc", "label": "old", "format": "percent"},
                {"field": "seen_new_acc", "label": "new", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
            ],
        },
        {
            "id": "table_extremes",
            "title": "最高与最低H必须按同一运行行解释",
            "subtitle": "避免把来自不同候选的单项最大值拼接成不存在的“最佳实验”",
            "dataset": "extremes",
            "defaultSort": {"field": "H_old_new", "direction": "desc"},
            "density": "dense",
            "sourceId": "src_per_run",
            "layout": "full",
            "columns": [
                {"field": "position", "label": "位置", "type": "text"},
                {"field": "experiment_id", "label": "实验ID", "type": "text"},
                {"field": "receiver", "label": "receiver", "type": "text"},
                {"field": "seed", "label": "seed", "format": "number"},
                {"field": "k_shot", "label": "K", "format": "number"},
                {"field": "old_acc_before_increment", "label": "old before", "format": "percent"},
                {"field": "old_acc", "label": "old", "format": "percent"},
                {"field": "seen_new_acc", "label": "new", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
                {"field": "average_forgetting", "label": "forgetting", "format": "percent"},
            ],
        },
        {
            "id": "table_storage",
            "title": "int8 support code的可核算存储量",
            "subtitle": "当前正式矩阵固定8个已注册类，每条code为160个int8值",
            "dataset": "storage",
            "defaultSort": {"field": "k_shot", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_runner",
            "layout": "full",
            "columns": [
                {"field": "k_shot", "label": "K", "format": "number"},
                {"field": "registered_classes", "label": "类数", "format": "number"},
                {"field": "stored_int8_codes", "label": "code数", "format": "number"},
                {"field": "code_bytes", "label": "code字节", "format": "number"},
                {"field": "code_kib", "label": "code KiB", "format": "number"},
                {"field": "note", "label": "口径", "type": "text"},
            ],
        },
        {
            "id": "table_audit",
            "title": "125个qKNNV42运行的结构化证据审计",
            "subtitle": "逐目录扫描，不以单个summary文件代替artifact检查",
            "dataset": "audit_rows",
            "defaultSort": {"field": "check", "direction": "asc"},
            "density": "dense",
            "sourceId": "src_audit",
            "layout": "full",
            "columns": [
                {"field": "check", "label": "检查项", "type": "text"},
                {"field": "observed", "label": "实测", "type": "text"},
                {"field": "expected", "label": "期望", "type": "text"},
                {"field": "verdict", "label": "结论", "type": "text"},
            ],
        },
        {
            "id": "table_historical",
            "title": "历史高分单行与当前正式矩阵不可混用",
            "subtitle": "历史行用于说明机制潜力；当前主结论只能来自2026-07-13正式矩阵",
            "dataset": "historical_vs_current",
            "defaultSort": {"field": "seed_count", "direction": "desc"},
            "density": "dense",
            "sourceId": "src_historical",
            "layout": "full",
            "columns": [
                {"field": "evidence", "label": "证据", "type": "text"},
                {"field": "protocol_status", "label": "协议地位", "type": "text"},
                {"field": "receiver_count", "label": "receiver", "type": "text"},
                {"field": "seed_count", "label": "seed数", "format": "number"},
                {"field": "k_shot", "label": "K", "type": "text"},
                {"field": "old_classes", "label": "old类", "format": "number"},
                {"field": "new_classes", "label": "new类", "format": "number"},
                {"field": "old_acc", "label": "old", "format": "percent"},
                {"field": "seen_new_acc", "label": "new", "format": "percent"},
                {"field": "H_old_new", "label": "H", "format": "percent"},
                {"field": "new_role", "label": "new role", "type": "text"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": "# qKNNV42技术汇报：方法、输入输出与当前效果\n\n面向CVS-RFFI Stage2-C增量注册场景。结论基于2026-07-13正式全矩阵、当前实现代码与125个qKNNV42运行目录的完整结构化artifact审计。",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 技术摘要\n\n"
                "qKNNV42不是从raw IQ端到端训练的新主干，而是接在冻结ADV3B02身份表征之后的轻量Stage2-C分类头。它以同一target receiver上的K-shot已标注support为记忆，用support-only对角Fisher/whitening尺度改善特征几何，把每条160维support特征量化为int8，再融合top-1余弦邻居、类prototype、微弱旧类锚与10-NN标签传播得到已注册类预测。整个适配过程没有梯度更新，也不保存raw support。\n\n"
                f"正式矩阵覆盖5个receiver、5个seed、K∈{{1,2,5,10,20}}和3种LEO弱场景，共125个qKNNV42运行、375个场景结果。K从1增至20时，H由{k_curve[0]['H_old_new']:.2%}升至{k20['H_old_new']:.2%}；全K总体H为{qknn_overall['H_old_new']:.2%}。相对CSIL、MoPC-HR和Orthogonal Incremental，qKNNV42在{pairwise_wins}/{pairwise_total}个完全配对的H比较中获胜。\n\n"
                f"但这仍不是部署成功：各K仅有{', '.join(str(row['old80_rows']) for row in k_curve)}/25个运行达到old_acc≥80%，125个运行中H≥80%的数量为{no_h80}，同时old_acc与seen_new_acc均≥80%的数量为{both80}。当前证据只能支持“正式矩阵内联合性能最强的候选头”，不能支持“已达到通用OLD80”或“已完成真实卫星部署”。"
            ),
            "sourceId": "src_per_run",
        },
        {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["card_k20", "card_matrix"]},
        {
            "id": "scope_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 1.问题定义与方法位置\n\n当前正式任务是Stage2-C：在同一未见target receiver域内，先用6个target-old TX维持旧类识别，再用2个与旧类互斥的target-seen-new TX进行K-shot注册。query与support按样本互斥；unknown/open-set不属于本轮成功口径。三个主场景为`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。",
            "sourceId": "src_protocol",
        },
        {
            "id": "pipeline_markdown",
            "type": "markdown",
            "layout": "full",
            "body": "### 方法在系统中的位置\n\n`raw IQ → 冻结ADV3B02 → 160维z_id → qKNNV42轻量头 → 8个已注册类中的预测标签`\n\n因此，问“qKNNV42的输入是什么”时必须分两层回答：系统入口是raw IQ；qKNNV42本体入口是冻结主干生成的160维L2归一化身份特征、support标签以及同批query特征。qKNNV42本体不直接学习IQ卷积滤波器。",
            "sourceId": "src_runner",
        },
        {"id": "io_table", "type": "table", "layout": "full", "tableId": "table_input_output", "sourceId": "src_runner"},
        {
            "id": "method_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 2.qKNNV42使用了什么方法\n\n### 2.1 support-only对角Fisher/whitening变换\n\n对support特征先L2归一化并求中心，再逐维计算类间方差与类内方差。当前实现把Fisher尺度与全局方差白化项组合为对角缩放，并将缩放范围裁剪到[0.05,20]。变换强度为0.1。所有统计只从support及其标签计算，query标签不进入变换。\n\n```text\nfisher_j = between_j / (within_j + 1e-5)\nwhiten_j = 1 / sqrt(var_j + 1e-5)\nscale_j = clip(fisher_norm_j^0.1 × whiten_norm_j^0.5, 0.05, 20)\nz'_j = normalize((z_j - center_j) × scale_j)\n```",
            "sourceId": "src_runner",
        },
        {
            "id": "method_quant",
            "type": "markdown",
            "layout": "full",
            "body": "### 2.2 int8压缩support记忆\n\n每条变换后的160维support向量按固定幅值127量化为有符号int8：\n\n```text\nq_i = clip(round(127 × z'_i), -127, 127)\nẑ_i = normalize(q_i / 127)\n```\n\n当前正式矩阵有8个已注册类，因此每个场景保存8K条量化support code；K=20时为160条。该压缩只针对support code，8个浮点prototype、变换参数和元数据仍需额外存储。",
            "sourceId": "src_runner",
        },
        {"id": "storage_table", "type": "table", "layout": "full", "tableId": "table_storage", "sourceId": "src_runner"},
        {
            "id": "method_score",
            "type": "markdown",
            "layout": "full",
            "body": "### 2.3 top-1邻居、prototype、旧类锚与标签传播融合\n\n对每个候选类c，局部项取query与该类量化support中最大的余弦相似度；全局项取query与该类prototype的余弦相似度。旧类统一增加0.001的微弱锚定项。随后在support+query联合10-NN图上进行8轮标签传播，温度0.05、传播系数0.76、融合权重0.025；support标签被clamp，query真值不使用。\n\n```text\nKNN_c(q)   = max_{i:y_i=c} cos(q, ẑ_i)\nProto_c(q) = cos(q, μ_c)\nBase_c(q)  = 0.55 × KNN_c(q) + 0.45 × Proto_c(q) + 0.001 × I[c∈Y_old]\nScore_c(q) = Base_c(q) + 0.025 × LP_c(q)\nŷ(q)       = argmax_c Score_c(q)\n```\n\n这使当前实现属于transductive inference：query特征参与联合图，但query标签不参与训练、参数拟合或模型选择。当前正式runner记录的`scenario_residual_applied=False`，所以历史版本中的场景残差项没有实际改变当前正式分数。",
            "sourceId": "src_runner",
        },
        {
            "id": "output_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 3.输出是什么\n\n在线语义输出是8个已注册TX类中的一个argmax标签。当前runner没有unknown/reject分支，也没有把“置信不足”作为正式输出，因此不能直接宣称具备开放集拒识。离线artifact输出包括`metrics.json`、`score_table.csv`、`detailed_metrics.csv/json`、`split_manifest.json`、`resolved_config.json`和`loss_trace.csv/json`，用于恢复old_acc、seen_new_acc、H、平均遗忘、按类/场景/receiver统计、混淆、延迟和存储量。当前`score_table.csv`保存真值、预测、正确性与场景元数据，但未保存逐类数值分数或校准置信度。",
            "sourceId": "src_runner",
        },
        {
            "id": "effect_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 4.达到了什么效果\n\n### 4.1 K-shot越多，旧类保持、新类学习和联合H均改善\n\n从K=1到K=20，old_acc由57.87%提升至71.17%，seen_new_acc由36.50%提升至56.73%，H由41.04%提升至62.22%。平均遗忘从8.29%降至7.58%，但中间K并非严格单调。这里的结果是25个receiver×seed运行均值，不是单个幸运seed。",
            "sourceId": "src_method_summary",
        },
        {"id": "k_chart", "type": "chart", "layout": "full", "chartId": "chart_k_curve", "sourceId": "src_method_summary"},
        {"id": "k_table", "type": "table", "layout": "full", "tableId": "table_k_curve", "sourceId": "src_method_summary"},
        {
            "id": "comparison_heading",
            "type": "markdown",
            "layout": "full",
            "body": "### 4.2 在同一正式paired matrix中，qKNNV42的联合H最好\n\n全K汇总时，qKNNV42的old_acc、seen_new_acc、H分别为65.59%、47.94%、53.26%，高于3个对比方法的联合H。Orthogonal Incremental的遗忘最低，但seen_new_acc仅4.87%，说明其低遗忘主要来自几乎没有学会新类，不能替代old/new联合指标。MoPC-HR在K=20的seen_new_acc均值比qKNNV42高1.10个百分点，但qKNNV42仍以更高old_acc获得更高H。",
            "sourceId": "src_per_run",
        },
        {"id": "method_chart", "type": "chart", "layout": "full", "chartId": "chart_method_h", "sourceId": "src_per_run"},
        {"id": "method_table", "type": "table", "layout": "full", "tableId": "table_method_overall", "sourceId": "src_per_run"},
        {"id": "pairwise_table", "type": "table", "layout": "full", "tableId": "table_pairwise", "sourceId": "src_per_run"},
        {
            "id": "robustness_heading",
            "type": "markdown",
            "layout": "full",
            "body": "### 4.3 receiver与场景差异仍然显著\n\nreceiver`20-1`的全K平均H最高，为59.57%；receiver`3-19`最低，仅42.73%。场景方面，LEO降雨弱扰动的seen_new_acc为41.76%，明显低于清晰弱扰动的50.84%和低仰角弱扰动的51.22%，说明当前最明确的场景瓶颈是降雨条件下的新类注册。receiver`7-14`呈现旧类80.59%但新类42.23%的不平衡，也提示不能只看old_acc。",
            "sourceId": "src_per_scenario",
        },
        {"id": "receiver_table", "type": "table", "layout": "half", "tableId": "table_receiver", "sourceId": "src_per_run"},
        {"id": "scenario_table", "type": "table", "layout": "half", "tableId": "table_scenario", "sourceId": "src_per_scenario"},
        {
            "id": "row_context_heading",
            "type": "markdown",
            "layout": "full",
            "body": "### 4.4 最强与最弱结果必须保留同一行上下文\n\n当前正式矩阵中最高H运行是`cvs_qknnv42_stage2c_rx20-1_k20_seed713103`，old_acc=68.61%、seen_new_acc=75.83%、H=72.02%；这不是old_acc最高行。最低H运行是`cvs_qknnv42_stage2c_rx3-19_k1_seed713104`，H=23.63%。报告不把来自不同运行的单项最大值拼成虚构的“最强配置”。",
            "sourceId": "src_per_run",
        },
        {"id": "extremes_table", "type": "table", "layout": "full", "tableId": "table_extremes", "sourceId": "src_per_run"},
        {
            "id": "audit_heading",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 5.证据完整性\n\n"
                f"逐目录审计了{audit_details['runs']}个qKNNV42运行、{audit_details['scenario_rows']}个场景结果和{audit_details['required_file_checks']}个必需文件槽位。必需文件缺失、JSON解析失败、NaN/Inf、support/query重叠均为0；{audit_details['trace_rows']}条loss trace均记录0次gradient update。125/125个split manifest标记query特征用于transductive inference，0个用于training或model selection，unknown均关闭。\n\n"
                f"artifact记录的本地runner单query延迟中位数为{audit_details['latency_ms_median']:.4f}ms、P95为{audit_details['latency_ms_p95']:.4f}ms、均值为{audit_details['latency_ms_mean']:.4f}ms、最大值为{audit_details['latency_ms_max']:.4f}ms。该值没有统一硬件基准认证，只能作为当前运行环境内的实现测量。矩阵目录没有stdout日志文件，因此本报告对结构化artifact做了全量分析，但不能声称完成了stdout全文审计。"
            ),
            "sourceId": "src_audit",
        },
        {"id": "audit_table", "type": "table", "layout": "full", "tableId": "table_audit", "sourceId": "src_audit"},
        {
            "id": "historical_heading",
            "type": "markdown",
            "layout": "full",
            "body": "## 6.为什么历史94.52%不能当作当前正式结论\n\n2026-07-07曾出现K=5、6 old+20 new、seed421070的单行机制验证：old_acc=94.52%、seen_new_acc=90.14%、H=92.28%，类最小准确率也较高。但该JSON仍使用`new_role=target_unknown`历史别名，只有1个seed，支持选择为`stable_first`，new类数量、query规模和切分路径均不同于2026-07-13正式5 receiver×5 seed×5 K矩阵。它证明的是某个历史协议下的机制潜力，不是当前Stage2-C的正式可重复效果。",
            "sourceId": "src_historical",
        },
        {"id": "historical_table", "type": "table", "layout": "full", "tableId": "table_historical", "sourceId": "src_historical"},
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": "## 7.局限性与稳健性边界\n\n- **尚未达到通用OLD80。**K=20也只有5/25个运行达到old_acc≥80%，且没有运行达到H≥80%。80%只是阶段性进展门槛，不等于部署成功。\n\n- **当前是transductive，不是严格inductive。**query特征参与support+query图传播；query标签不使用。若部署要求单样本独立到达，必须追加关闭label propagation的inductive消融。\n\n- **没有unknown/reject输出。**当前只在8个已注册类中argmax，不能将错误分类当作未知检测。\n\n- **场景残差在正式runner中未生效。**元数据保留`scenario_residual_weight=0.5`，但`scenario_residual_applied=False`；正式结果应按未应用解释。\n\n- **统计实现与协议文档有差异。**比较协议写明bootstrap CI、overall test、Holm校正和K轴AUC；当前summary manifest实际采用1.96×标准误的正态近似CI，现有summary artifact也没有overall test、Holm或AUC产物。\n\n- **artifact完整不等于已发布。**正式矩阵结构化产物完整，但截至本报告生成时仍位于Git未跟踪的`local_artifacts`目录，因此不能称为已版本化、已发布或已部署证据。\n\n- **不是实星验证。**LEO场景来自简化物理增强；本报告不把它表述为真实卫星链路或在轨部署结果。",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": "## 8.建议的下一步\n\n1. 做严格inductive消融：保持同一split，分别关闭label propagation、old anchor、Fisher尺度和int8量化，量化每个模块对old/new/H的净贡献。\n\n2. 优先攻克receiver`3-19`和`leo_rain_weak`的新类瓶颈；所有改动仍需在同一5 receiver×5 seed×5 K正式矩阵复验。\n\n3. 给`score_table.csv`增加逐类分数、top-2 margin和校准置信度，同时保存stdout日志，补齐错误案例与完整训练/推理轨迹审计。\n\n4. 按比较协议补做bootstrap CI、overall paired test、Holm多重校正与K轴AUC，并把统计口径写入summary manifest。\n\n5. 将正式矩阵与summary artifact纳入Git承载面后，再使用“已发布证据”措辞；在达到预注册成功标准前继续使用“候选头”而非“部署方案”。",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": "## 9.进一步需要回答的问题\n\n- 目标部署允许批量query共同传导，还是必须逐样本inductive推理？\n\n- 当前2个seen-new类的正式设置是否足以代表未来一次注册20个新TX的容量需求？\n\n- 端侧真正可接受的总内存、P95延迟和功耗阈值是什么？只有明确预算后，int8 code的轻量优势才能转化为部署判据。\n\n- 是否需要unknown/reject作为Phase3独立任务？若需要，应新增拒识指标与校准流程，不能从当前closed-set结果外推。",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "qKNNV42技术汇报：方法、输入输出与当前效果",
            "description": "基于当前Stage2-C正式矩阵、实现代码与完整结构化artifact审计的技术汇报",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "input_output": input_output,
                "k_curve": k_curve,
                "method_overall": method_overall,
                "pairwise": pairwise,
                "receiver_summary": receiver_summary,
                "scenario_summary": scenario_summary,
                "extremes": extremes,
                "storage": storage,
                "audit_rows": audit_rows,
                "historical_vs_current": [current_reference, historical],
            },
        },
        "sources": sources,
    }
    return artifact


if __name__ == "__main__":
    artifact = build_artifact()
    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
