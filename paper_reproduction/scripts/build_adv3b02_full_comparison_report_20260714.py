#!/usr/bin/env python3
"""Build the ADV3B02/qKNN/domain-adaptation/class-incremental comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean
from typing import Any


METHOD_LABELS = {
    "cvs_opgac": "CVS-OPGAC",
    "protonet_cda": "ProtoNet CDA",
    "mrior_sda": "MRIOR-SDA",
    "dadda_sda": "DADDA-SDA",
    "cvs_qknnv42": "CVS-qKNNV42",
    "csil": "CSIL",
    "mopc_hr": "MoPC-HR",
    "orthogonal_incremental": "Orthogonal Incremental",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def harmonic(old: float, new: float) -> float:
    return 0.0 if old + new <= 0 else 2.0 * old * new / (old + new)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("|" + "|".join(row) + "|" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")

    strict_root = repo / "local_artifacts/qknnv42_strict_dual125_20260714_183556/analysis"
    direct_root = repo / "local_artifacts/adv3b02_direct_old_strict_20260714_181100"
    publication_root = repo / "local_artifacts/cvs_publication_stage2_summary_20260713"
    cache_root = repo / "local_artifacts/cvs_publication_adv3b02_feature_cache_20260713"

    strict_arm_rows = read_csv(strict_root / "summary_by_arm.csv")
    strict_k_rows = read_csv(strict_root / "summary_by_k.csv")
    direct_summary_rows = read_csv(direct_root / "query125_summary.csv")
    publication_rows = read_csv(publication_root / "per_run_results.csv")
    publication_k_rows = read_csv(publication_root / "method_k_summary.csv")

    direct_overall = next(row for row in direct_summary_rows if row["level"] == "overall")
    direct_old = float(direct_overall["accuracy"])
    strict_by_arm = {row["arm"]: row for row in strict_arm_rows}
    light = strict_by_arm["singleview_fft96_strict"]
    full = strict_by_arm["full_legacy_oracle_strict"]

    legacy_qknn_runs = [row for row in publication_rows if row["method"] == "cvs_qknnv42"]
    if len(legacy_qknn_runs) != 125:
        raise RuntimeError(f"expected 125 historical qKNN rows, got {len(legacy_qknn_runs)}")
    legacy_old = mean(float(row["old_acc"]) for row in legacy_qknn_runs)
    legacy_new = mean(float(row["seen_new_acc"]) for row in legacy_qknn_runs)
    legacy_h = mean(float(row["H_old_new"]) for row in legacy_qknn_runs)

    cache_validation = json.loads((cache_root / "validation.json").read_text(encoding="utf-8-sig"))
    npz_path = cache_root / "leo_clear_weak.npz"
    cache_load_note = "manifest字段缺失"
    try:
        import numpy as np

        with np.load(npz_path, allow_pickle=True) as payload:
            manifest = json.loads(str(payload["manifest_json"].item()))
        cache_load_note = (
            f"missing={manifest.get('missing_keys')},unexpected={manifest.get('unexpected_keys')},"
            f"mismatch={manifest.get('skipped_mismatch')}"
        )
    except Exception as exc:  # report the evidence gap rather than silently assuming strictness
        cache_load_note = f"manifest读取失败:{type(exc).__name__}"

    core_rows = [
        {
            "method": "直接地面ADV3B02分类头",
            "old_pct": 100 * direct_old,
            "new_pct": None,
            "H_pct": None,
            "runs": 125,
            "input": "单LEO视图raw IQ→严格ADV3B02 tx_logits",
            "adaptation": "无support、无域适应、无新类头",
            "decision": "六旧类分类头argmax",
            "strictness": "严格加载；old-query与125矩阵对齐",
            "evidence_tier": "A-严格旧类基线",
            "verdict": "只回答旧类；new/H不可定义",
        },
        {
            "method": "单qKNNV42（历史无FFT）",
            "old_pct": 100 * legacy_old,
            "new_pct": 100 * legacy_new,
            "H_pct": 100 * legacy_h,
            "runs": 125,
            "input": "单LEO视图z_id160",
            "adaptation": "support-only Fisher对角白化、int8类内top-1、prototype与标签传播",
            "decision": "逐样本argmax",
            "strictness": f"兼容加载诊断；{cache_load_note}",
            "evidence_tier": "B-完整但非严格",
            "verdict": "保留为历史基线，不能写作严格ADV3B02结果",
        },
        {
            "method": "qKNNV42+单视图FFT96",
            "old_pct": 100 * float(light["old_acc"]),
            "new_pct": 100 * float(light["seen_new_acc"]),
            "H_pct": 100 * float(light["H_old_new"]),
            "runs": int(light["count"]),
            "input": "单LEO视图z_id160+FFT96",
            "adaptation": "无训练adapter；qKNN support-only拟合；主/辅分数融合",
            "decision": "逐样本argmax",
            "strictness": "严格加载0/0/0",
            "evidence_tier": "A-严格125矩阵",
            "verdict": "当前最接近轻量卫星侧路径的严格诊断",
        },
        {
            "method": "完整qKNN legacy Oracle",
            "old_pct": 100 * float(full["old_acc"]),
            "new_pct": 100 * float(full["seen_new_acc"]),
            "H_pct": 100 * float(full["H_old_new"]),
            "runs": int(full["count"]),
            "input": "严格ADV3B02+60epoch adapter+5-view TTA+FFT96",
            "adaptation": "id_norm_late_feature适配、TTA融合、support-only qKNN",
            "decision": "角色Oracle+类别配额Hungarian",
            "strictness": "严格加载0/0/0；但含Oracle约束",
            "evidence_tier": "A-严格Oracle上限",
            "verdict": "NON_DEPLOYMENT_ORACLE_DIAGNOSTIC",
        },
    ]

    core_long: list[dict[str, Any]] = []
    for row in core_rows:
        for field, label in [("old_pct", "old_acc"), ("new_pct", "new_acc"), ("H_pct", "H")]:
            if row[field] is not None:
                core_long.append({
                    "method": row["method"],
                    "metric": label,
                    "value_pct": row[field],
                    "runs": row["runs"],
                    "evidence_tier": row["evidence_tier"],
                    "strictness": row["strictness"],
                })

    legacy_by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in legacy_qknn_runs:
        legacy_by_k[int(row["k_shot"])].append(row)
    k_compare: list[dict[str, Any]] = []
    for k, rows in sorted(legacy_by_k.items()):
        k_compare.append({
            "method": "单qKNNV42（历史无FFT）",
            "k_shot": k,
            "old_pct": 100 * mean(float(r["old_acc"]) for r in rows),
            "new_pct": 100 * mean(float(r["seen_new_acc"]) for r in rows),
            "H_pct": 100 * mean(float(r["H_old_new"]) for r in rows),
            "strictness": "兼容加载诊断",
            "run_count": len(rows),
        })
    for row in strict_k_rows:
        label = "qKNNV42+单视图FFT96" if row["arm"] == "singleview_fft96_strict" else "完整qKNN legacy Oracle"
        k_compare.append({
            "method": label,
            "k_shot": int(row["k_shot"]),
            "old_pct": 100 * float(row["old_acc"]),
            "new_pct": 100 * float(row["seen_new_acc"]),
            "H_pct": 100 * float(row["H_old_new"]),
            "strictness": "严格ADV3B02" + ("+Oracle" if row["arm"] == "full_legacy_oracle_strict" else ""),
            "run_count": int(row["count"]),
        })

    stage2b: list[dict[str, Any]] = []
    stage2c: list[dict[str, Any]] = []
    for row in publication_k_rows:
        method = row["method"]
        k = int(row["k_shot"])
        if row["phase"] == "stage2b":
            stage2b.append({
                "method": METHOD_LABELS[method],
                "k_shot": k,
                "target_old_pct": 100 * float(row["target_old_accuracy_mean"]),
                "before_pct": 100 * float(row["target_old_accuracy_before_adaptation_mean"]),
                "delta_pp": 100 * float(row["target_old_accuracy_delta_mean"]),
                "run_count": int(row["target_old_accuracy_n"]),
                "protocol": "Stage2-B target-old support/query",
                "evidence_boundary": "20260713完整矩阵；不等同严格ADV3B02适配器消融",
            })
        else:
            stage2c.append({
                "method": METHOD_LABELS[method],
                "k_shot": k,
                "old_pct": 100 * float(row["old_acc_mean"]),
                "new_pct": 100 * float(row["seen_new_acc_mean"]),
                "H_pct": 100 * float(row["H_old_new_mean"]),
                "forgetting_pct": 100 * float(row["average_forgetting_mean"]),
                "run_count": int(row["H_old_new_n"]),
                "protocol": "Stage2-C old+seen-new support/query",
                "evidence_boundary": "20260713完整矩阵；CVS-qKNN缓存为兼容加载，其他方法为各自训练管线",
            })

    method_notes = [
        {"family": "域适应", "method": "CVS-OPGAC", "input": "冻结特征+target-old K-shot support", "methodology": "support-only旧类原型与高斯收缩校准", "output": "旧类后验/预测", "effect": "完整矩阵中K=1→20为72.52%→78.21%", "boundary": "旧类-only；历史特征缓存非严格"},
        {"family": "域适应", "method": "ProtoNet CDA", "input": "source IQ+target-old support", "methodology": "源侧ProtoNet训练后以目标support类原型最近邻适应", "output": "旧类预测", "effect": "K=1→20为20.54%→27.42%", "boundary": "方法管线基座不同，不能归因为ADV3B02 adapter"},
        {"family": "域适应", "method": "MRIOR-SDA", "input": "source IQ+target-old support", "methodology": "ReceiverImpactGADNet，Adam下源训练与目标有监督适应", "output": "旧类logits", "effect": "K=1→20为20.77%→24.00%", "boundary": "方法管线基座不同"},
        {"family": "域适应", "method": "DADDA-SDA", "input": "source IQ+target-old support", "methodology": "DADDANet，SGD+逆衰减完成源训练与目标有监督适应", "output": "旧类logits", "effect": "K=1→20为16.98%→17.22%", "boundary": "方法管线基座不同"},
        {"family": "类增量", "method": "CVS-qKNNV42", "input": "冻结z_id160+old/new K-shot support", "methodology": "Fisher对角白化、int8类内top-1、prototype、old anchor、标签传播", "output": "8类old/new分数与预测", "effect": "H从K=1的41.04%升至K=20的62.22%", "boundary": "历史缓存兼容加载；严格无FFT尚未重跑"},
        {"family": "类增量", "method": "CSIL", "input": "source IQ+old/new support", "methodology": "零偏置余弦头、通道分离、旧块梯度mask、KD/EWC", "output": "8类logits", "effect": "H在K=1/2/5/10/20为16.23/19.34/18.05/17.69/9.45%", "boundary": "CVS extension，不是原论文原生数据结果"},
        {"family": "类增量", "method": "MoPC-HR", "input": "source IQ+old/new support", "methodology": "原型增强、层次正则、动量原型修正", "output": "8类logits", "effect": "H从14.70%升至37.30%", "boundary": "CVS extension"},
        {"family": "类增量", "method": "Orthogonal Incremental", "input": "source IQ+old/new support", "methodology": "正交simplex伪目标、冻结编码器、新类增量校准", "output": "余弦分类预测", "effect": "K=20 old=71.11%，但new=3.47%、H=6.00%", "boundary": "旧类保持不能替代old/new联合结论"},
    ]

    receiver_rows = []
    for row in direct_summary_rows:
        if row["level"] == "receiver":
            receiver_rows.append({
                "receiver": row["receiver"],
                "direct_old_pct": 100 * float(row["accuracy"]),
                "query_count": int(row["total"]),
                "task_rows": int(row["row_count"]),
            })

    write_csv(out / "core_comparison.csv", core_rows)
    write_csv(out / "kshot_comparison.csv", k_compare)
    write_csv(out / "stage2b_domain_adaptation.csv", sorted(stage2b, key=lambda r: (r["method"], r["k_shot"])))
    write_csv(out / "stage2c_class_incremental.csv", sorted(stage2c, key=lambda r: (r["method"], r["k_shot"])))
    write_csv(out / "method_input_output_effect.csv", method_notes)
    write_csv(out / "direct_ground_by_receiver.csv", receiver_rows)

    sources = [
        {"id": "strict_qknn", "label": "qKNNV42严格双125重跑", "path": "local_artifacts/qknnv42_strict_dual125_20260714_183556/analysis/summary.json", "query": {"engine": "DuckDB", "language": "sql", "sql": "SELECT * FROM read_csv_auto('local_artifacts/qknnv42_strict_dual125_20260714_183556/analysis/summary_by_arm.csv', header=true);", "description": "读取严格双125的arm与K-shot汇总；保留同一run内old/new/H联合指标。", "tables_used": ["summary_by_arm.csv", "summary_by_k.csv"], "metric_definitions": ["old_acc=旧类query准确率", "new_acc=seen-new query准确率", "H=每个run的old/new harmonic mean后求宏平均"]}},
        {"id": "direct_ground", "label": "ADV3B02严格直接地面模型评测", "path": "local_artifacts/adv3b02_direct_old_strict_20260714_181100/query125_metrics.json", "query": {"engine": "DuckDB", "language": "sql", "sql": "SELECT * FROM read_csv_auto('local_artifacts/adv3b02_direct_old_strict_20260714_181100/query125_summary.csv', header=true) WHERE level='overall';", "description": "读取与125任务old-query对齐的严格地面分类头汇总。", "tables_used": ["query125_summary.csv"], "metric_definitions": ["old_acc=严格ADV3B02六旧类分类头在对齐query上的准确率", "new_acc与H不适用"]}},
        {"id": "publication_matrix", "label": "CVS论文级Stage2-B/C完整矩阵汇总", "path": "local_artifacts/cvs_publication_stage2_summary_20260713/final_audit.json", "query": {"engine": "DuckDB", "language": "sql", "sql": "SELECT * FROM read_csv_auto('local_artifacts/cvs_publication_stage2_summary_20260713/method_k_summary.csv', header=true);", "description": "读取20260713 Stage2-B/C各500行完整矩阵的method×K汇总及125行历史qKNN。", "tables_used": ["method_k_summary.csv", "per_run_results.csv", "final_audit.json"], "metric_definitions": ["Stage2-B=target-old accuracy", "Stage2-C=old/new harmonic mean", "统计单位=receiver-seed run的三场景均值"]}},
        {"id": "compat_cache", "label": "20260713 ADV3B02历史特征缓存与manifest", "path": "local_artifacts/cvs_publication_adv3b02_feature_cache_20260713/validation.json", "query": {"engine": "DuckDB", "language": "sql", "sql": "SELECT * FROM read_json_auto('local_artifacts/cvs_publication_adv3b02_feature_cache_20260713/validation.json');", "description": "读取历史特征缓存验证文件；严格性字段另从NPZ内嵌manifest解析。", "tables_used": ["leo_clear_weak.npz", "validation.json"], "metric_definitions": ["strictness由missing_keys、unexpected_keys、skipped_mismatch共同判定"]}},
        {"id": "protocol", "label": "CVS-RFFI项目科学协议", "path": "docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md", "query": {"language": "text", "query": "read_text('docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md', encoding='utf-8')", "description": "读取Stage2-B/C、receiver、K-shot和场景定义。", "tables_used": ["CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md"]}},
        {"id": "core_synthesis", "label": "四条主路径证据分层汇总", "path": f"local_artifacts/{args.run_id}/core_comparison.csv", "query": {"engine": "DuckDB", "language": "sql", "sql": f"SELECT * FROM read_csv_auto('local_artifacts/{args.run_id}/core_comparison.csv', header=true);", "description": "由严格双125、严格直接地面评测和历史完整矩阵合并生成；每行保留严格性与结论边界。", "tables_used": ["summary_by_arm.csv", "query125_summary.csv", "per_run_results.csv", "leo_clear_weak.npz"], "metric_definitions": ["跨证据层差值仅为诊断，不解释为因果消融"]}},
        {"id": "k_synthesis", "label": "qKNN三路径K-shot汇总", "path": f"local_artifacts/{args.run_id}/kshot_comparison.csv", "query": {"engine": "DuckDB", "language": "sql", "sql": f"SELECT * FROM read_csv_auto('local_artifacts/{args.run_id}/kshot_comparison.csv', header=true);", "description": "合并历史无FFT qKNN与两条严格qKNN路径的逐K汇总。", "tables_used": ["method_k_summary.csv", "summary_by_k.csv"], "metric_definitions": ["每个K点为5接收机×5seed的25行均值"]}},
    ]

    cards = [
        {"id": "card_direct_old", "dataset": "headline", "filter": {"key": "direct_old"}, "metrics": [{"label": "直接地面old_acc", "field": "value_pct", "format": "number"}], "sourceId": "direct_ground"},
        {"id": "card_light_h", "dataset": "headline", "filter": {"key": "light_h"}, "metrics": [{"label": "严格单视图FFT96 H", "field": "value_pct", "format": "number"}], "sourceId": "strict_qknn"},
        {"id": "card_full_h", "dataset": "headline", "filter": {"key": "full_h"}, "metrics": [{"label": "严格完整Oracle H", "field": "value_pct", "format": "number"}], "sourceId": "strict_qknn"},
        {"id": "card_legacy_h", "dataset": "headline", "filter": {"key": "legacy_h"}, "metrics": [{"label": "历史单qKNN H", "field": "value_pct", "format": "number"}], "sourceId": "publication_matrix"},
    ]
    charts = [
        {"id": "core_chart", "title": "四条主路径old/new/H对比", "subtitle": "百分比；缺失值表示指标不适用", "type": "bar", "dataset": "core_chart", "encodings": {"x": {"field": "method", "type": "nominal", "label": "方法"}, "y": {"fields": ["old_pct", "new_pct", "H_pct"], "type": "quantitative", "label": "准确率/H", "unit": "%"}, "tooltip": [{"field": "strictness", "type": "text"}, {"field": "runs", "type": "quantitative"}]}, "valueFormat": "number", "unit": "%", "layout": "full", "sourceId": "core_synthesis"},
        {"id": "k_chart", "title": "qKNN路径随K-shot变化的H", "subtitle": "每个点为5接收机×5seed的25行均值", "type": "line", "dataset": "k_compare", "encodings": {"x": {"field": "k_shot", "type": "ordinal", "label": "K-shot"}, "y": {"field": "H_pct", "type": "quantitative", "label": "H", "unit": "%"}, "color": {"field": "method", "type": "nominal", "label": "方法"}, "tooltip": [{"field": "old_pct", "type": "quantitative", "unit": "%"}, {"field": "new_pct", "type": "quantitative", "unit": "%"}, {"field": "strictness", "type": "text"}]}, "valueFormat": "number", "unit": "%", "layout": "full", "sourceId": "k_synthesis"},
        {"id": "stage2b_chart", "title": "Stage2-B域适应方法target-old accuracy", "subtitle": "20260713完整矩阵；各点25个receiver-seed运行", "type": "line", "dataset": "stage2b", "encodings": {"x": {"field": "k_shot", "type": "ordinal", "label": "K-shot"}, "y": {"field": "target_old_pct", "type": "quantitative", "label": "target-old accuracy", "unit": "%"}, "color": {"field": "method", "type": "nominal", "label": "方法"}, "tooltip": [{"field": "before_pct", "type": "quantitative", "unit": "%"}, {"field": "delta_pp", "type": "quantitative", "unit": "pp"}]}, "valueFormat": "number", "unit": "%", "layout": "full", "sourceId": "publication_matrix"},
        {"id": "stage2c_chart", "title": "Stage2-C类增量方法old/new harmonic mean", "subtitle": "20260713完整矩阵；各点25个receiver-seed运行", "type": "line", "dataset": "stage2c", "encodings": {"x": {"field": "k_shot", "type": "ordinal", "label": "K-shot"}, "y": {"field": "H_pct", "type": "quantitative", "label": "H", "unit": "%"}, "color": {"field": "method", "type": "nominal", "label": "方法"}, "tooltip": [{"field": "old_pct", "type": "quantitative", "unit": "%"}, {"field": "new_pct", "type": "quantitative", "unit": "%"}]}, "valueFormat": "number", "unit": "%", "layout": "full", "sourceId": "publication_matrix"},
    ]
    tables = [
        {"id": "core_table", "title": "主路径完整对比与证据边界", "dataset": "core_table", "columns": [
            {"field": "method", "label": "方法", "type": "text"}, {"field": "old_pct", "label": "old_acc(%)", "type": "number"}, {"field": "new_pct", "label": "new_acc(%)", "type": "number"}, {"field": "H_pct", "label": "H(%)", "type": "number"}, {"field": "runs", "label": "运行数", "type": "number"}, {"field": "strictness", "label": "加载/证据", "type": "text"}, {"field": "verdict", "label": "结论", "type": "text"}], "density": "dense", "layout": "full", "sourceId": "core_synthesis"},
        {"id": "method_table", "title": "方法、输入、输出与效果", "dataset": "method_notes", "columns": [
            {"field": "family", "label": "类别", "type": "text"}, {"field": "method", "label": "方法", "type": "text"}, {"field": "input", "label": "输入", "type": "text"}, {"field": "methodology", "label": "主要机制", "type": "text"}, {"field": "output", "label": "输出", "type": "text"}, {"field": "effect", "label": "达到效果", "type": "text"}, {"field": "boundary", "label": "边界", "type": "text"}], "density": "dense", "layout": "full", "sourceId": "publication_matrix"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ADV3B02、qKNN、域适应与类增量方法完整对比",
            "description": "严格结果、历史兼容诊断和不同协议方法分层汇报",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# ADV3B02、qKNN、域适应与类增量方法完整对比"},
                {"id": "summary", "type": "markdown", "body": "## 技术摘要\n严格可比的核心结论是：直接地面ADV3B02在对齐125任务的旧类query上为73.87%；加入单视图FFT96后的严格qKNN达到old 75.12%、new 64.64%、H 68.56%；完整体达到84.07%、93.24%、88.23%，但它同时使用60epoch特征适配、5-view TTA和Oracle角色/类别配额，因此只能作为非部署上限。历史单qKNN为65.59%、47.94%、53.26%，其缓存manifest存在7个missing key、31个unexpected key和3个shape mismatch，不能当作严格ADV3B02结果。"},
                {"id": "metric_strip", "type": "metric-strip", "cardIds": ["card_direct_old", "card_light_h", "card_full_h", "card_legacy_h"]},
                {"id": "core_chart_block", "type": "chart", "chartId": "core_chart"},
                {"id": "core_interpret", "type": "markdown", "body": "## 核心判断\nFFT96使严格轻量qKNN相对历史单qKNN提高old约9.53pp、new约16.70pp、H约15.31pp，但由于历史单qKNN不是严格加载，这一差值只能作为跨证据层诊断，不能作为纯FFT因果消融。严格完整体相对严格单视图FFT96提高old 8.94pp、new 28.60pp、H 19.66pp；这项差值混合了adapter、TTA与Oracle约束。"},
                {"id": "core_table_block", "type": "table", "tableId": "core_table"},
                {"id": "k_heading", "type": "markdown", "body": "## K-shot敏感性\n三条qKNN路径均随K增加而改善。严格单视图FFT96的H由K=1的52.70%升至K=20的79.89%；完整Oracle由81.10%升至92.81%。低K和困难接收机仍是轻量路径的主要瓶颈。"},
                {"id": "k_chart_block", "type": "chart", "chartId": "k_chart"},
                {"id": "da_heading", "type": "markdown", "body": "## Stage2-B域适应方法\n该组比较只评估target-old。CVS-OPGAC在完整矩阵中显著高于ProtoNet CDA、MRIOR-SDA和DADDA-SDA，但这些行比较的是完整管线而非同一严格ADV3B02特征上的适配模块消融；不同方法继承的表示与训练入口不同。"},
                {"id": "stage2b_chart_block", "type": "chart", "chartId": "stage2b_chart"},
                {"id": "cil_heading", "type": "markdown", "body": "## Stage2-C类增量方法\n历史CVS-qKNNV42在五档K上均高于CSIL、MoPC-HR和Orthogonal Incremental。Orthogonal在K=20保留71.11%的旧类，却只有3.47%的新类，说明不能用old_acc单独替代old/new联合指标。该历史CVS-qKNNV42仍受兼容加载问题约束；严格无FFT单qKNN尚未重跑。"},
                {"id": "stage2c_chart_block", "type": "chart", "chartId": "stage2c_chart"},
                {"id": "method_table_block", "type": "table", "tableId": "method_table"},
                {"id": "quality", "type": "markdown", "body": "## 数据质量与可比性\n证据分三层：A层为20260714严格checkpoint重建后的直接地面、单视图FFT96和完整Oracle；B层为20260713 artifact完整但ADV3B02缓存兼容加载的历史qKNN/OPGAC；C层为各自训练管线的DA/CIL对比。只能在同层、同阶段、同指标内排序。直接地面模型没有新类头，因此new_acc与H必须留空；完整Oracle使用角色和类别配额真值约束，因此不得作为自主卫星部署性能。"},
                {"id": "next", "type": "markdown", "body": "## 建议\n下一项最有价值的实验是用已修复的严格ADV3B02导出器重跑无FFT单qKNN 125矩阵，并在完全相同的strict cache上做FFT96开/关配对；随后分别做adapter、TTA和Oracle约束的逐项消融。只有这样才能把15.31pp和19.66pp拆解为可归因的增益。"},
                {"id": "limitations", "type": "markdown", "body": "## 限制\n所有LEO场景均为简化星地信道仿真，不是真实在轨测量；125行由5接收机×5seed×5K构成，但直接地面分支不读取K，实际只有25个独立receiver-seed query集合。历史94.52/90.14/92.28属于不同切分、20新类、单seed legacy diagnostic，不进入本报告数值排名。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {"key": "direct_old", "value_pct": 100 * direct_old},
                    {"key": "light_h", "value_pct": 100 * float(light["H_old_new"])},
                    {"key": "full_h", "value_pct": 100 * float(full["H_old_new"])},
                    {"key": "legacy_h", "value_pct": 100 * legacy_h},
                ],
                "core_chart": core_rows,
                "core_table": core_rows,
                "k_compare": k_compare,
                "stage2b": stage2b,
                "stage2c": stage2c,
                "method_notes": method_notes,
            },
        },
    }
    (out / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    core_md_rows = []
    for row in core_rows:
        core_md_rows.append([
            row["method"], f"{row['old_pct']:.2f}%", "—" if row["new_pct"] is None else f"{row['new_pct']:.2f}%",
            "—" if row["H_pct"] is None else f"{row['H_pct']:.2f}%", str(row["runs"]), row["strictness"], row["verdict"],
        ])
    stage2b_pivot: dict[str, dict[int, float]] = defaultdict(dict)
    for row in stage2b:
        stage2b_pivot[row["method"]][row["k_shot"]] = row["target_old_pct"]
    stage2c_pivot: dict[str, dict[int, float]] = defaultdict(dict)
    for row in stage2c:
        stage2c_pivot[row["method"]][row["k_shot"]] = row["H_pct"]
    ks = [1, 2, 5, 10, 20]

    report = f"""# ADV3B02、qKNN、域适应与类增量方法完整对比报告

## 实验与报告信息

|字段|内容|
|---|---|
|报告ID|`{args.run_id}`|
|生成时间|{generated_at}|
|操作者|Codex|
|目标|统一汇报直接地面模型、单qKNN、qKNN+单视图FFT96、严格完整qKNN Oracle，以及此前域适应和类增量方法|
|协议|5个target receiver×5个seed×K={{1,2,5,10,20}}；旧类6个、新类2个；三种简化LEO场景|
|声明边界|严格结果、兼容加载历史诊断和不同训练管线结果分层，不做跨层因果归因|

## 一、执行结论

严格可引用的轻量路径是`qKNNV42+单视图FFT96`：old_acc={pct(float(light['old_acc']))}、new_acc={pct(float(light['seen_new_acc']))}、H={pct(float(light['H_old_new']))}。完整体为old_acc={pct(float(full['old_acc']))}、new_acc={pct(float(full['seen_new_acc']))}、H={pct(float(full['H_old_new']))}，但其同时使用60epoch特征适配、5-view TTA和Oracle角色/类别配额，必须标记`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`。

直接地面ADV3B02在对齐原125任务的旧类query上old_acc={pct(direct_old)}；它没有新类头，所以new_acc和H不可定义。历史单qKNNV42无FFT为old_acc={pct(legacy_old)}、new_acc={pct(legacy_new)}、H={pct(legacy_h)}，但其特征缓存manifest为`{cache_load_note}`，只能保留为兼容加载诊断。

## 二、四条主路径完整对比

{md_table(['方法','old_acc','new_acc','H','运行数','加载/证据','结论'], core_md_rows)}

### 输入、方法与输出

1.直接地面模型输入单个LEO视图raw IQ，严格重建ADV3B02后直接读取六旧类`tx_logits`并argmax；输出只有旧类预测。
2.单qKNN输入单视图`z_id160`与old/new K-shot support；输出8类分数与预测。历史数值完整，但checkpoint为兼容加载。
3.qKNN+FFT96输入`z_id160+FFT96`，主特征与FFT辅助分别做support-only qKNN后融合分数；无60epoch训练adapter、无TTA、无Oracle。
4.完整体输入严格ADV3B02的5-view特征与FFT96，先训练60epoch `id_norm_late_feature` adapter，再做TTA融合和qKNN，最后用角色Oracle与类别配额Hungarian约束输出。

### 达到的效果与不能下的结论

- 严格单视图FFT96相对历史单qKNN提高old {100*(float(light['old_acc'])-legacy_old):.2f}pp、new {100*(float(light['seen_new_acc'])-legacy_new):.2f}pp、H {100*(float(light['H_old_new'])-legacy_h):.2f}pp；由于两者严格加载状态不同，这不是纯FFT因果增益。
- 严格完整体相对严格单视图FFT96提高old {100*(float(full['old_acc'])-float(light['old_acc'])):.2f}pp、new {100*(float(full['seen_new_acc'])-float(light['seen_new_acc'])):.2f}pp、H {100*(float(full['H_old_new'])-float(light['H_old_new'])):.2f}pp；该差值混合adapter、TTA与Oracle。
- 完整体的88.23%H是上限诊断，不是卫星自主部署性能；轻量FFT96才更接近当前星上约束，但H=68.56%仍未达到部署成功。

## 三、K-shot分解

{md_table(['方法','K=1 H','K=2 H','K=5 H','K=10 H','K=20 H'], [[m] + [f"{next(r['H_pct'] for r in k_compare if r['method']==m and r['k_shot']==k):.2f}%" for k in ks] for m in ['单qKNNV42（历史无FFT）','qKNNV42+单视图FFT96','完整qKNN legacy Oracle']])}

严格单视图FFT96的H由K=1的52.70%升至K=20的79.89%；完整Oracle由81.10%升至92.81%。低K时新类support不足是轻量路径的主要瓶颈。

## 四、此前Stage2-B域适应方法

下表每个单元格是25个receiver-seed运行的target-old accuracy均值。该组只回答旧类域适应，不回答新类学习。

{md_table(['方法','K=1','K=2','K=5','K=10','K=20'], [[m] + [f"{stage2b_pivot[m][k]:.2f}%" for k in ks] for m in ['CVS-OPGAC','ProtoNet CDA','MRIOR-SDA','DADDA-SDA']])}

- `CVS-OPGAC`：输入冻结特征与target-old support，做support-only原型/高斯收缩校准，输出旧类预测；K=20达到78.21%。
- `ProtoNet CDA`：输入source IQ和target-old support，先训练ProtoNet，再以目标原型最近邻分类；K=20为27.42%。
- `MRIOR-SDA`：ReceiverImpactGADNet使用Adam完成源训练与目标有监督适应；K=20为24.00%。
- `DADDA-SDA`：DADDANet使用SGD与逆衰减完成源训练和目标适应；K=20为17.22%。

重要边界：该表比较的是各自完整方法管线，不是“同一个严格ADV3B02特征+不同adapter”的纯模块消融。CVS-OPGAC使用的20260713缓存同样存在`{cache_load_note}`，不能写成严格ADV3B02结果。

## 五、此前Stage2-C类增量方法

主指标为old/seen-new harmonic mean；每个单元格是25个receiver-seed运行均值。

{md_table(['方法','K=1 H','K=2 H','K=5 H','K=10 H','K=20 H'], [[m] + [f"{stage2c_pivot[m][k]:.2f}%" for k in ks] for m in ['CVS-qKNNV42','CSIL','MoPC-HR','Orthogonal Incremental']])}

- `CVS-qKNNV42`：Fisher对角白化、int8类内top-1、prototype、old anchor和标签传播；H从41.04%升至62.22%。该行就是本报告的历史单qKNN，无FFT且非严格加载。
- `CSIL`：零偏置余弦、通道分离、旧块梯度mask、知识蒸馏/EWC；K=2最高H约19.34%，K=20降至9.45%。
- `MoPC-HR`：原型增强、层次正则、动量原型修正；H从14.70%升至37.30%。
- `Orthogonal Incremental`：正交simplex伪目标、冻结编码器和新类权重校准；K=20旧类71.11%，但新类仅3.47%、H仅6.00%，说明旧类保持不能替代联合性能。

## 六、数据质量与问题定位

|证据层|包含结果|可回答的问题|不可回答的问题|
|---|---|---|---|
|A：严格checkpoint|直接地面、FFT96、完整Oracle|严格ADV3B02下的当前性能|完整Oracle不能代表部署；直接地面不能回答新类|
|B：完整artifact但兼容加载|历史单qKNN、CVS-OPGAC|历史管线与K趋势|不能称为严格ADV3B02；不能和A层做纯因果消融|
|C：各自训练管线|ProtoNet/MRIOR/DADDA、CSIL/MoPC/Orthogonal|同协议下完整管线比较|不能归因于单一adapter或ADV3B02特征|

此前125次差距大的主要问题不是单一“随机波动”，而是四项叠加：checkpoint重建是否严格、是否加入FFT96、是否训练60epoch特征adapter并使用5-view TTA、以及是否使用Oracle角色/类别配额约束。困难接收机`3-19`和低K进一步放大差距。

## 七、建议的下一步严格消融

1.用当前严格ADV3B02导出器重跑“无FFT单qKNN”125矩阵，建立真正的strict no-FFT基线。
2.在完全相同的strict cache、split、query上仅切换FFT96开/关，得到FFT的配对因果增益。
3.依次加入60epoch adapter、5-view TTA、场景筛选、角色Oracle、类别配额，逐项报告Δold/Δnew/ΔH和资源开销。
4.主表只报告可部署逐样本决策；Oracle结果移入上限附表。

## 八、限制与声明边界

- 所有LEO场景均为简化星地信道仿真，不是真实在轨信道测量。
- 直接地面分支不读取K；125行实际上只有25个独立receiver-seed query集合，K只是对齐索引。
- 历史94.52% old_acc、90.14% new_acc、92.28%H属于不同切分、20个新类、单seed legacy diagnostic，不进入本报告排名。
- 20260713完整矩阵artifact审计为Stage2-B 500/500、Stage2-C 500/500，但artifact完整不等于checkpoint严格。

## 九、机器可读产物

|文件|用途|
|---|---|
|`core_comparison.csv`|四条主路径及证据层|
|`kshot_comparison.csv`|qKNN三条路径K-shot分解|
|`stage2b_domain_adaptation.csv`|域适应方法逐K结果|
|`stage2c_class_incremental.csv`|类增量方法逐K结果|
|`method_input_output_effect.csv`|各方法输入、方法、输出、效果与边界|
|`artifact.json`|HTML报告的规范数据/叙事/来源输入|
|`report.html`|自包含可交互技术报告|
"""
    (out / "report.md").write_text(report, encoding="utf-8")

    quality = {
        "run_id": args.run_id,
        "generated_at": generated_at,
        "source_row_counts": {
            "strict_arm_rows": len(strict_arm_rows),
            "strict_k_rows": len(strict_k_rows),
            "direct_summary_rows": len(direct_summary_rows),
            "publication_run_rows": len(publication_rows),
            "publication_k_rows": len(publication_k_rows),
            "historical_qknn_rows": len(legacy_qknn_runs),
        },
        "completeness": {
            "strict_qknn_arms": sorted(strict_by_arm),
            "stage2b_methods": sorted({row["method"] for row in stage2b}),
            "stage2c_methods": sorted({row["method"] for row in stage2c}),
            "k_grid": ks,
            "cache_validation_status": cache_validation.get("status"),
        },
        "cache_checkpoint_load_audit": cache_load_note,
        "metric_rules": {
            "H": "per-run harmonic mean aggregated as supplied; not recomputed from marginal means",
            "direct_ground_new_and_H": "not applicable",
            "cross_tier_deltas": "diagnostic only, not causal",
        },
        "status": "PASS_WITH_COMPARABILITY_TIERS",
    }
    (out / "data_quality_audit.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
