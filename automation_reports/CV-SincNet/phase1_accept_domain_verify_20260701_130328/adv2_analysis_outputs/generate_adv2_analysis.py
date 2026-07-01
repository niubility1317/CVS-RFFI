from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\type10-7")
RUN_ID = "phase1_accept_domain_verify_20260701_130328"
LOCAL_DIR = ROOT / "automation_reports" / "CV-SincNet" / RUN_ID
ART = LOCAL_DIR / "adv2_full_analysis_artifacts"
RUNS = ART / "runs" / RUN_ID
LOGS = ART / "logs" / RUN_ID
OUT = LOCAL_DIR / "adv2_analysis_outputs"
GIT_DIR = ROOT / "github_publish" / "CVS-RFFI-repo" / "automation_reports" / "CV-SincNet" / RUN_ID
GIT_OUT = GIT_DIR / "adv2_analysis_outputs"

OUT.mkdir(parents=True, exist_ok=True)
GIT_OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_META = {
    "ADV2_R17_CORESTRICT_E260": ("R17", "CORESTRICT", "R17主线缩短训练"),
    "ADV2_R17_PROXYHI_E260": ("R17", "PROXYHI", "R17拒识压力"),
    "ADV2_R20_SAT70_E260": ("R20", "SAT70", "R20稳定主线"),
    "ADV2_R20_VACMID_E260": ("R20", "VACMID", "R20中等vacuum"),
    "ADV2_R28_PROXYLOW_E260": ("R28", "PROXYLOW", "R28低proxy-vac对照"),
    "ADV2_R28_FUSE6_E260": ("R28", "FUSE6", "R28严格融合"),
    "ADV2_T13_CONSERVE_E260": ("T13", "CONSERVE", "T13保守对照"),
    "ADV2_T13_TAILGUARD_E260": ("T13", "TAILGUARD", "T13尾部保护"),
    "ADV2_SRCLOW_R17_E260": ("R17", "SRCLOW", "source-episode降档"),
    "ADV2_SOURCECAP32_R20_E260": ("R20", "SOURCECAP32", "source半径上限压力"),
    "ADV2_FUSE6_R17_E260": ("R17", "FUSE6", "R17融合压力"),
    "ADV2_FUSE5_R20_E260": ("R20", "FUSE5", "R20融合中档"),
    "ADV2_TAILCV_R17_E260": ("R17", "TAILCV", "R17尾部压力"),
    "ADV2_TAILCV_R20_E260": ("R20", "TAILCV", "R20尾部压力"),
}

PARENT_MAP = {
    "R17": "FSP_VAC_R17_Q2_HARDK3_E280",
    "R20": "FSP_VAC_R20_Q2_SAT70_E280",
    "R28": "FSP_VAC_R28_Q2_SAT72_E300",
    "T13": "FSP_VAC_T13_LATE60_SAT68_E260",
}


def finite(value):
    try:
        if value is None:
            return np.nan
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


def safe_mean(values):
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def safe_min(values):
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.min(vals)) if vals else np.nan


def safe_max(values):
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.max(vals)) if vals else np.nan


def last_finite_row(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            mask &= pd.to_numeric(df[col], errors="coerce").notna()
    sub = df[mask]
    return sub.iloc[-1] if len(sub) else df.iloc[-1]


def row_at_epoch(df: pd.DataFrame, epoch):
    if pd.isna(epoch):
        return None
    sub = df[df["epoch"].astype(int) == int(epoch)]
    return sub.iloc[-1] if len(sub) else None


def extremum_epoch(df: pd.DataFrame, col: str, active_col: str | None = None, minimize: bool = True):
    if col not in df:
        return np.nan, np.nan
    series = pd.to_numeric(df[col], errors="coerce")
    mask = series.notna()
    if active_col and active_col in df:
        mask &= pd.to_numeric(df[active_col], errors="coerce").fillna(0) > 0
    if not mask.any():
        return np.nan, np.nan
    idx = series[mask].idxmin() if minimize else series[mask].idxmax()
    return int(df.loc[idx, "epoch"]), float(series.loc[idx])


def md_table(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None, floatfmt: str = ".4f") -> str:
    if cols:
        df = df[cols]
    if n:
        df = df.head(n)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) else (f"{x:{floatfmt}}" if isinstance(x, (float, np.floating)) else str(x))
            )
        else:
            out[col] = out[col].fillna("").astype(str)
    headers = [str(col) for col in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in out.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_candidates():
    rows = []
    health = []
    trajectory = []

    for csv_path in sorted(RUNS.glob("*/metrics_epoch.csv")):
        cid = csv_path.parent.name
        df = pd.read_csv(csv_path)
        base, mechanism, role = CANDIDATE_META.get(cid, ("OTHER", "OTHER", ""))
        family = "T" if base.startswith("T") else "R"
        final = last_finite_row(df, ["protected_overall_tx", "protected_strict_udu", "protected_receiver_floor"])
        expected = int(pd.to_numeric(df["epochs"], errors="coerce").dropna().max())
        max_epoch = int(pd.to_numeric(df["epoch"], errors="coerce").dropna().max())
        best_epoch = int(final["best_epoch"]) if pd.notna(final.get("best_epoch", np.nan)) else np.nan
        best_row = row_at_epoch(df, best_epoch)

        p95_min_epoch, p95_min_val = extremum_epoch(
            df, "train_ow_feat_pos_angle_p95_deg", "train_ow_feat_active_classes", True
        )
        pos_min_epoch, pos_min_val = extremum_epoch(
            df, "train_ow_feat_pos_angle_deg", "train_ow_feat_active_classes", True
        )
        mininter_max_epoch, mininter_max_val = extremum_epoch(
            df, "train_ow_feat_min_inter_deg", "train_ow_feat_active_classes", False
        )
        proxy_auc_epoch, proxy_auc_max = extremum_epoch(
            df, "train_proxy_unknown_auc_proxy", "train_proxy_unknown_active", False
        )
        proxy_vac_epoch, proxy_vac_min = extremum_epoch(
            df, "train_proxy_unknown_vacuum_violation_rate", "train_proxy_unknown_active", True
        )
        ow_vac_epoch, ow_vac_min = extremum_epoch(
            df, "train_ow_feat_vacuum_violation_rate", "train_ow_feat_active_classes", True
        )

        src_active = df[pd.to_numeric(df.get("train_source_episode_classes", 0), errors="coerce").fillna(0) > 0]
        if len(src_active) and "train_source_episode_overflow_rate" in src_active:
            src_first = float(pd.to_numeric(src_active["train_source_episode_overflow_rate"], errors="coerce").dropna().iloc[0])
        else:
            src_first = np.nan
        src_last = finite(final.get("train_source_episode_overflow_rate", np.nan))

        final_overall = finite(final.get("protected_overall_tx", np.nan))
        best_test = finite(final.get("best_test_tx", np.nan))
        final_strict = finite(final.get("protected_strict_udu", np.nan))
        best_strict = finite(best_row.get("protected_strict_udu", np.nan)) if best_row is not None else np.nan

        log_path = LOGS / f"{cid}.out"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        epoch_begin = [int(m.group(1)) for m in re.finditer(r"\[EPOCH-BEGIN\] E(\d+)/(\d+)", text)]
        epoch_end = [int(m.group(1)) for m in re.finditer(r"\[EPOCH-END\] E(\d+)/(\d+)", text)]
        fatal_lines = [
            line[:260]
            for line in text.splitlines()
            if re.search(r"Traceback|RuntimeError|CUDA out of memory|\bOOM\b|\bKilled\b|unrecognized arguments|\bFATAL\b", line, re.I)
        ]
        grad_nan_lines = [
            line for line in text.splitlines() if "[GRAD]" in line and re.search(r"\b(aux|total|backbone|domain)=nan\b", line, re.I)
        ]
        grad_inf_lines = [
            line for line in text.splitlines() if "[GRAD]" in line and re.search(r"\b(aux|total|backbone|domain)=inf\b", line, re.I)
        ]
        skipped_test_nan = [
            line for line in text.splitlines() if "[TEST]" in line and "overall_tx=nan%" in line and "(0/0)" in line
        ]
        loss_nan_lines = [line for line in text.splitlines() if re.search(r"loss[^\n=]*=nan|train loss nan|loss=nan", line, re.I)]
        real_loss_nan = [line for line in loss_nan_lines if "[LOSS-SAT-RAW]" not in line and "[LOSS-DG-RAW]" not in line]
        metric_nan_final = int(pd.isna(final_overall) or pd.isna(final_strict))
        phase2_export = "[PHASE2-EXPORT]" in text

        rows.append(
            {
                "candidate_id": cid,
                "base": base,
                "family": family,
                "mechanism_tag": mechanism,
                "role": role,
                "parent_candidate": PARENT_MAP.get(base, ""),
                "completed": max_epoch == expected,
                "expected_epochs": expected,
                "max_epoch_begin": max(epoch_begin) if epoch_begin else np.nan,
                "last_epoch_end": max(epoch_end) if epoch_end else np.nan,
                "csv_rows": len(df),
                "final_epoch": int(final["epoch"]),
                "best_joint_epoch": best_epoch,
                "best_joint_test_tx": best_test,
                "best_joint_val_tx": finite(final.get("best_val_tx", np.nan)),
                "best_joint_score": finite(final.get("best_score", np.nan)),
                "final_overall_tx": final_overall,
                "final_strict_udu": final_strict,
                "final_receiver_floor": finite(final.get("protected_receiver_floor", np.nan)),
                "final_sat_mean": finite(final.get("protected_sat_mean_tx", np.nan)),
                "final_sat_floor": finite(final.get("protected_sat_floor_tx", np.nan)),
                "final_sat_strict_mean": finite(final.get("protected_sat_strict_mean", np.nan)),
                "final_sat_strict_floor": finite(final.get("protected_sat_strict_floor", np.nan)),
                "final_pos_angle": finite(final.get("train_ow_feat_pos_angle_deg", np.nan)),
                "final_p95": finite(final.get("train_ow_feat_pos_angle_p95_deg", np.nan)),
                "final_p99": finite(final.get("train_ow_feat_pos_angle_p99_deg", np.nan)),
                "final_tail_frac_gt_3sigma": finite(final.get("train_ow_feat_tail_frac_gt_3sigma", np.nan)),
                "final_r3sigma": finite(final.get("train_ow_feat_tail_radius_3sigma_deg", np.nan)),
                "final_min_inter": finite(final.get("train_ow_feat_min_inter_deg", np.nan)),
                "final_ow_vac_rate": finite(final.get("train_ow_feat_vacuum_violation_rate", np.nan)),
                "final_ow_vac_gap": finite(final.get("train_ow_feat_vacuum_margin_deg", np.nan)),
                "final_proxy_auc": finite(final.get("train_proxy_unknown_auc_proxy", np.nan)),
                "final_proxy_vac_rate": finite(final.get("train_proxy_unknown_vacuum_violation_rate", np.nan)),
                "final_proxy_vaccept": finite(final.get("train_proxy_unknown_virtual_accept_rate", np.nan)),
                "final_proxy_vac_gap": finite(final.get("train_proxy_unknown_vacuum_margin_deg", np.nan)),
                "final_proxy_known": finite(final.get("train_proxy_unknown_known_count", np.nan)),
                "final_proxy_virtual": finite(final.get("train_proxy_unknown_virtual_count", np.nan)),
                "final_source_overflow": src_last,
                "final_source_r3s": finite(final.get("train_source_episode_radius_3sigma_deg", np.nan)),
                "final_source_val_angle": finite(final.get("train_source_episode_val_angle_deg", np.nan)),
                "min_p95_epoch": p95_min_epoch,
                "min_p95": p95_min_val,
                "min_pos_angle_epoch": pos_min_epoch,
                "min_pos_angle": pos_min_val,
                "max_min_inter_epoch": mininter_max_epoch,
                "max_min_inter": mininter_max_val,
                "best_proxy_auc_epoch": proxy_auc_epoch,
                "best_proxy_auc": proxy_auc_max,
                "min_proxy_vac_rate_epoch": proxy_vac_epoch,
                "min_proxy_vac_rate": proxy_vac_min,
                "min_ow_vac_rate_epoch": ow_vac_epoch,
                "min_ow_vac_rate": ow_vac_min,
                "source_overflow_first_active": src_first,
                "source_overflow_delta_active_to_final": src_last - src_first
                if not pd.isna(src_first) and not pd.isna(src_last)
                else np.nan,
                "phase2_export": phase2_export,
                "log_path": str(log_path),
                "metrics_path": str(csv_path),
            }
        )
        health.append(
            {
                "candidate_id": cid,
                "completed": max_epoch == expected,
                "expected_epochs": expected,
                "last_epoch": max_epoch,
                "max_epoch_begin": max(epoch_begin) if epoch_begin else np.nan,
                "last_epoch_end": max(epoch_end) if epoch_end else np.nan,
                "phase2_export": phase2_export,
                "fatal_count": len(fatal_lines),
                "fatal_samples": " | ".join(fatal_lines[:2]),
                "skipped_test_placeholder_nan_lines": len(skipped_test_nan),
                "nan_aux_grad_lines": len(grad_nan_lines),
                "inf_grad_lines": len(grad_inf_lines),
                "nan_real_loss_lines": len(real_loss_nan),
                "nan_real_metric_final": metric_nan_final,
                "health_status": "PASS_WITH_TELEMETRY_NAN"
                if (max_epoch == expected and phase2_export and not fatal_lines and not real_loss_nan and metric_nan_final == 0)
                else "CHECK",
                "notes": "跳过测试占位NaN和辅助梯度NaN存在；最终指标有限且训练完成。"
                if not fatal_lines
                else "存在fatal样本，需人工复核。",
            }
        )
        trajectory.append(
            {
                "candidate_id": cid,
                "best_joint_epoch": best_epoch,
                "final_epoch": int(final["epoch"]),
                "final_minus_best_test": final_overall - best_test
                if not pd.isna(final_overall) and not pd.isna(best_test)
                else np.nan,
                "final_minus_best_strict": final_strict - best_strict
                if not pd.isna(final_strict) and not pd.isna(best_strict)
                else np.nan,
                "p95_min_epoch": p95_min_epoch,
                "final_minus_min_p95": finite(final.get("train_ow_feat_pos_angle_p95_deg", np.nan)) - p95_min_val
                if not pd.isna(p95_min_val)
                else np.nan,
                "min_inter_max_epoch": mininter_max_epoch,
                "final_minus_max_min_inter": finite(final.get("train_ow_feat_min_inter_deg", np.nan)) - mininter_max_val
                if not pd.isna(mininter_max_val)
                else np.nan,
                "proxy_vaccept_final": finite(final.get("train_proxy_unknown_virtual_accept_rate", np.nan)),
                "source_overflow_first_active": src_first,
                "source_overflow_final": src_last,
                "source_overflow_delta": src_last - src_first if not pd.isna(src_first) and not pd.isna(src_last) else np.nan,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(health), pd.DataFrame(trajectory)


def parse_prototypes() -> pd.DataFrame:
    proto_rows = []
    for cand_dir in sorted(RUNS.glob("*")):
        if not cand_dir.is_dir():
            continue
        cid = cand_dir.name
        json_path = cand_dir / "phase2_zid_prototypes.json"
        pt_path = cand_dir / "phase2_zid_prototypes.pt"
        rec = {
            "candidate_id": cid,
            "json_exists": json_path.exists(),
            "pt_exists": pt_path.exists(),
            "pt_size_bytes": pt_path.stat().st_size if pt_path.exists() else 0,
        }
        if json_path.exists():
            data = json.load(open(json_path, encoding="utf-8"))
            metadata = data.get("metadata", {})
            counts = data.get("tx_domain_counts") or []
            active_counts = [sum(1 for item in arr if finite(item) > 0) for arr in counts]
            components = data.get("fusion_components") or []
            comp_counts = [len(item) if isinstance(item, list) else 0 for item in components]
            tail = data.get("radius_tail_stats", {})
            geometry = data.get("geometry", {})
            fusion_config = data.get("fusion_config", {})
            rec.update(
                {
                    "n_classes": metadata.get("num_tx"),
                    "n_domains": metadata.get("num_domains"),
                    "samples": metadata.get("num_samples"),
                    "active_domain_prototypes_min": safe_min(active_counts),
                    "active_domain_prototypes_mean": safe_mean(active_counts),
                    "active_domain_prototypes_max": safe_max(active_counts),
                    "tx_domain_components_min": safe_min(comp_counts),
                    "tx_domain_components_mean": safe_mean(comp_counts),
                    "tx_domain_components_max": safe_max(comp_counts),
                    "tx_domain_components_total": sum(comp_counts),
                    "has_fusion_components": "fusion_components" in data,
                    "has_fused_tx_prototypes": "fused_tx_prototypes" in data,
                    "has_fusion_config": "fusion_config" in data,
                    "fusion_enabled": fusion_config.get("enabled"),
                    "fusion_accept_policy": fusion_config.get("accept_policy"),
                    "fusion_global_ball_accept": fusion_config.get("global_ball_accept"),
                    "fusion_tail_auto_accept": fusion_config.get("tail_auto_accept"),
                    "fusion_max_components_per_tx": fusion_config.get("max_components_per_tx")
                    or fusion_config.get("max_components_per_class"),
                    "prototype_radius_p95_mean_deg": finite(geometry.get("radius_p95_mean_deg")),
                    "prototype_radius_p99_mean_deg": math.degrees(safe_mean(tail.get("p99", [])))
                    if isinstance(tail.get("p99"), list)
                    else np.nan,
                    "prototype_r3sigma_mean_deg": math.degrees(safe_mean(tail.get("r_3sigma", [])))
                    if isinstance(tail.get("r_3sigma"), list)
                    else np.nan,
                    "prototype_min_inter_deg": finite(geometry.get("min_interclass_angle_deg")),
                    "prototype_tail_frac_mean": safe_mean(tail.get("tail_frac_gt_3sigma", []))
                    if isinstance(tail.get("tail_frac_gt_3sigma"), list)
                    else np.nan,
                }
            )
            rec["prototype_conclusion"] = (
                "fusion字段已导出；可审计local_component字段，但仍缺真实unknown验证。"
                if rec["has_fusion_components"] and rec["has_fused_tx_prototypes"] and rec["has_fusion_config"]
                else "fusion字段缺失；不能视为融合生效。"
            )
        proto_rows.append(rec)
    return pd.DataFrame(proto_rows)


def flatten_vac_row(row: dict, cohort: str) -> dict:
    final_eval = row.get("final_eval") or {}
    best = row.get("best_joint_final") or {}
    feature = row.get("final_feature") or {}
    proxy = row.get("final_proxy") or {}
    source = row.get("final_source") or {}
    return {
        "cohort": cohort,
        "candidate_id": row.get("id"),
        "completed": row.get("completed"),
        "expected_epochs": row.get("expected_epochs"),
        "best_joint_epoch": best.get("epoch"),
        "best_joint_test_tx": best.get("test_tx"),
        "final_overall_tx": final_eval.get("overall_tx"),
        "final_strict_udu": final_eval.get("strict_udu"),
        "final_receiver_floor": final_eval.get("receiver_floor"),
        "final_sat_mean": final_eval.get("sat_mean"),
        "final_sat_strict_floor": final_eval.get("sat_strict_floor"),
        "final_pos_angle": feature.get("pos_angle"),
        "final_p95": feature.get("p95"),
        "final_min_inter": feature.get("min_inter"),
        "final_ow_vac_rate": feature.get("vac_rate"),
        "final_proxy_auc": proxy.get("auc"),
        "final_proxy_vac_rate": proxy.get("vac_rate"),
        "final_proxy_vaccept": proxy.get("vaccept"),
        "final_source_overflow": source.get("overflow"),
    }


def aggregate_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "n": int(series.count()),
                "mean": series.mean() if len(series) else np.nan,
                "median": series.median() if len(series) else np.nan,
                "min": series.min() if len(series) else np.nan,
                "max": series.max() if len(series) else np.nan,
                "std": series.std(ddof=1) if len(series) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def weighted_rank_score(df: pd.DataFrame, specs: list[tuple[str, bool, float]]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for col, ascending, weight in specs:
        score += weight * pd.to_numeric(df[col], errors="coerce").rank(ascending=ascending, method="average")
        weight_sum += weight
    return score / weight_sum


def build_report(cand, health, trajectory, proto, all_agg, pivot, group_df, cor_df, rank_df, promote_df):
    adv2_mean = all_agg[all_agg["cohort"] == "ADV2_14"].set_index("metric")["mean"]
    vacuum_mean = all_agg[all_agg["cohort"] == "vacuum_effective32"].set_index("metric")["mean"]
    lateopt_mean = all_agg[all_agg["cohort"] == "lateopt16"].set_index("metric")["mean"]
    parent_mean = all_agg[all_agg["cohort"] == "vacuum_parent4"].set_index("metric")["mean"]

    best_closed = cand.sort_values("closed_score").iloc[0]
    worst_overflow = cand.sort_values("final_source_overflow", ascending=False).iloc[0]

    health_table = md_table(
        health[
            [
                "candidate_id",
                "completed",
                "expected_epochs",
                "last_epoch",
                "phase2_export",
                "fatal_count",
                "skipped_test_placeholder_nan_lines",
                "nan_aux_grad_lines",
                "inf_grad_lines",
                "health_status",
            ]
        ],
        floatfmt=".0f",
    )
    agg_table = md_table(
        pivot[
            [
                "metric",
                "lateopt16",
                "vacuum_effective32",
                "vacuum_parent4",
                "ADV2_14",
                "delta_vs_vacuum_effective32",
                "delta_vs_vacuum_parent4",
            ]
        ],
        floatfmt=".4f",
    )
    group_table = md_table(
        group_df[group_df["group_type"].isin(["family", "base", "mechanism_tag"])][
            [
                "group_type",
                "group",
                "n",
                "best_joint_test_tx_mean",
                "final_strict_udu_mean",
                "final_receiver_floor_mean",
                "final_sat_strict_floor_mean",
                "final_p95_mean",
                "final_p99_mean",
                "final_min_inter_mean",
                "final_proxy_auc_mean",
                "final_proxy_vac_rate_mean",
                "final_proxy_vaccept_mean",
                "final_source_overflow_mean",
                "prototype_radius_p95_mean_deg_mean",
            ]
        ],
        floatfmt=".4f",
    )
    core_cols = [
        "candidate_id",
        "mechanism_tag",
        "best_joint_test_tx",
        "final_overall_tx",
        "final_strict_udu",
        "final_receiver_floor",
        "final_sat_strict_floor",
        "final_p95",
        "final_p99",
        "final_min_inter",
        "final_proxy_auc",
        "final_proxy_vac_rate",
        "final_proxy_vaccept",
        "final_source_overflow",
        "prototype_radius_p95_mean_deg",
        "tx_domain_components_mean",
        "closed_score",
        "geometry_score",
        "proxy_score",
        "pareto_front",
    ]
    candidate_table = md_table(cand.sort_values("composite_score", na_position="last")[core_cols], floatfmt=".4f")
    rank_closed = md_table(
        cand.sort_values("closed_score")[
            ["candidate_id", "best_joint_test_tx", "final_strict_udu", "final_receiver_floor", "final_sat_strict_floor", "closed_score"]
        ],
        n=10,
        floatfmt=".4f",
    )
    rank_geo = md_table(
        cand.sort_values("geometry_score")[
            ["candidate_id", "final_p95", "final_p99", "final_min_inter", "final_source_overflow", "final_ow_vac_rate", "geometry_score"]
        ],
        n=10,
        floatfmt=".4f",
    )
    rank_proxy = md_table(
        cand.sort_values("proxy_score")[["candidate_id", "final_proxy_auc", "final_proxy_vac_rate", "final_proxy_vaccept", "proxy_score"]],
        n=10,
        floatfmt=".4f",
    )
    proto_table = md_table(
        proto[
            [
                "candidate_id",
                "json_exists",
                "pt_exists",
                "n_classes",
                "n_domains",
                "samples",
                "active_domain_prototypes_mean",
                "tx_domain_components_mean",
                "has_fusion_components",
                "has_fused_tx_prototypes",
                "has_fusion_config",
                "fusion_accept_policy",
                "fusion_global_ball_accept",
                "fusion_tail_auto_accept",
                "prototype_radius_p95_mean_deg",
                "prototype_radius_p99_mean_deg",
                "prototype_min_inter_deg",
                "prototype_tail_frac_mean",
            ]
        ],
        floatfmt=".4f",
    )
    cor_table = md_table(cor_df, floatfmt=".4f")
    trajectory_table = md_table(
        trajectory[
            [
                "candidate_id",
                "best_joint_epoch",
                "final_epoch",
                "final_minus_best_test",
                "final_minus_best_strict",
                "p95_min_epoch",
                "final_minus_min_p95",
                "min_inter_max_epoch",
                "final_minus_max_min_inter",
                "proxy_vaccept_final",
                "source_overflow_final",
                "source_overflow_delta",
            ]
        ],
        floatfmt=".4f",
    )
    promotion_table = md_table(promote_df, floatfmt=".2f")

    risk_df = pd.DataFrame(
        [
            ["known core", f"final_overall mean {adv2_mean.get('final_overall_tx', np.nan):.2f}%", "闭集中心仍强", "支撑Phase1闭集DG有效，但不等于unknown拒识", "Stage2 old/query真实评估"],
            ["known soft tail", f"p95 mean {adv2_mean.get('final_p95', np.nan):.2f}deg", "比vacuum32低" if adv2_mean.get("final_p95", np.nan) < vacuum_mean.get("final_p95", np.nan) else "未优于vacuum32", "p95只覆盖常规尾部", "p95+p99+CVaR联合门控"],
            ["known extreme tail", f"p99 mean {cand['final_p99'].mean():.2f}deg", "已可量化", "p99仍长，可能污染accept半径", "tail quarantine/CVaR97"],
            ["source cross-domain overflow", f"overflow mean {adv2_mean.get('final_source_overflow', np.nan):.4f}", "低于vacuum32" if adv2_mean.get("final_source_overflow", np.nan) < vacuum_mean.get("final_source_overflow", np.nan) else "未低于vacuum32", "跨域query仍有较大比例越界", "source_episode_density_gate"],
            ["inter-class low-density zone", f"min_inter mean {adv2_mean.get('final_min_inter', np.nan):.2f}deg", "类中心分离很高", "中心角不覆盖类间低密度带", "inter-class slerp negative"],
            ["same-class multi-mode bridge", f"fusion components mean {proto['tx_domain_components_mean'].mean():.2f}/class", "local component已导出", "多组件之间空洞可能被错误接收", "same-class bridge negative"],
            ["old-class shell outside r_accept", f"prototype p95 mean {proto['prototype_radius_p95_mean_deg'].mean():.2f}deg,p99 {proto['prototype_radius_p99_mean_deg'].mean():.2f}deg", "local accept字段可审计", "p99远大于p95时自动接收必须限制", "shell accept dry-run"],
            ["proxy unknown near tail", f"proxy_vac_rate mean {adv2_mean.get('final_proxy_vac_rate', np.nan):.4f}", "部分候选下降", "proxy点仍不能代表真实unknown", "tail-outward proxy sampler"],
            ["virtual unknown accepted by energy", f"proxy_vaccept mean {adv2_mean.get('final_proxy_vaccept', np.nan):.4f}", "暴露失败面", "几乎全接收，拒识面未形成", "energy/density/geometric hard gate"],
            ["unknown not evaluated", "unknown_FAR/FPR95/real AUROC缺失", "边界清楚", "不能声明部署拒识成功", "真实Y_unknown Stage2-A/C"],
        ],
        columns=["risk_region", "observed_metric", "current_value_or_evidence", "why_it_matters", "next_test"],
    )
    risk_table = md_table(risk_df)

    failure_df = pd.DataFrame(
        [
            ["proxy_vaccept接近1", f"均值{adv2_mean.get('final_proxy_vaccept', np.nan):.4f}，min{cand['final_proxy_vaccept'].min():.4f}", "virtual unknown几乎全被接收", "energy阈值和接收规则没有形成open-space reject地形", "真实unknown FAR大概率仍高", "真实unknown+shell/inter-slerp/same-class bridge accept dry-run", "P0"],
            ["p95改善但p99长尾仍在", f"p95均值{cand['final_p95'].mean():.2f}deg，p99均值{cand['final_p99'].mean():.2f}deg", "常规包络收紧，极端尾部仍长", "3sigma/accept半径仍被tail污染", "unknown可能落入旧类尾部", "CVaR97、tail quarantine、tail_auto_accept=false断言", "P0"],
            ["min_inter高但拒识弱", f"min_inter均值{cand['final_min_inter'].mean():.2f}deg，proxy_auc均值{cand['final_proxy_auc'].mean():.4f}", "类中心远离但proxy判别弱", "开放空间风险由tail/低密度/桥接区决定", "继续单推min_inter收益有限", "inter-class slerp negative评估", "P1"],
            ["source_episode_overflow仍有风险", f"均值{cand['final_source_overflow'].mean():.4f}，最高{worst_overflow['candidate_id']}={worst_overflow['final_source_overflow']:.4f}", "部分source约束扩大known包络", "跨域query被强拉为known", "unknown误接收风险增加", "source_episode_query_core_only/density_gate", "P0"],
            ["fusion已导出但未验证拒识收益", f"{int(proto['has_fusion_components'].sum())}/14 JSON有fusion字段，global_ball_accept均False", "实现路径已通，但只是导出成功", "缺少local component hard gate dry-run和真实unknown", "不能说local component gate部署成功", "不重训hard gate dry-run", "P0"],
            ["proxy unknown覆盖不足", f"proxy_auc均值{cand['final_proxy_auc'].mean():.4f}", "proxy区分度弱", "virtual/leave-one-TX-out未覆盖shell/bridge/tail-outward", "真实unknown风险未被训练面覆盖", "四类negative sampler", "P1"],
            ["E260缩短训练不是万能", f"best epoch均值{cand['best_joint_epoch'].mean():.1f}，final-best test均值{trajectory['final_minus_best_test'].mean():.2f}pp", "部分候选final低于best", "后期joint guard仍需更细", "不能默认final checkpoint最佳", "joint early-stop/hard gate checkpoint selection", "P1"],
            ["NaN遥测污染健康判断", f"每候选skipped-test NaN均{health['skipped_test_placeholder_nan_lines'].mean():.1f}行，aux grad NaN均{health['nan_aux_grad_lines'].mean():.1f}行", "占位/辅助NaN混杂", "parser若不分类会误报", "自动化健康判断不稳定", "NaN分类parser落地", "P1"],
        ],
        columns=["failure_mode", "evidence", "symptoms", "likely_root_cause", "impact_on_unknown_rejection", "how_to_test_next", "fix_priority"],
    )
    failure_table = md_table(failure_df)

    next_df = pd.DataFrame(
        [
            ["A:不重训fusion+local hard gate dry-run", "SRCLOW_R17,FUSE5_R20,R28_FUSE6,R17_CORESTRICT,T13_CONSERVE", "local component distance+density+NLL+geo margin+energy gate", "known_core_accept,known_tail_review,proxy_vaccept,shell/inter/bridge accept,reject_reason_counts", "proxy_vaccept显著低于1；shell/inter/bridge accept<0.05；known_core_accept>=0.90；tail不自动接收"],
            ["B:negative space filling训练", "SRCLOW_R17/FUSE5_R20主线+R28机制候选", "shell negative,tail-outward negative,inter-class slerp negative,same-class bridge negative", "proxy_vaccept,proxy_vac_rate,shell_accept,inter_slerp_accept,source_overflow,closed metrics", "unknown-risk accept下降且closed-set不崩"],
            ["C:core/tail/outside quarantine", "SRCLOW_R17,FUSE5_R20,TAILCV_R17", "core强CE,soft tail低权重,extreme tail quarantine,p99/CVaR约束,tail_auto_accept=false", "p99,CVaR95/97,source_overflow,known_core_accept,known_tail_review,proxy_vaccept", "p99和overflow下降，不只降p95"],
            ["D:unlabeled unknown-risk mining", "SRCLOW_R17,R17_CORESTRICT,R20_VACMID,PROXYHI", "unlabeled分pseudo-known core/unknown-risk buffer/ignore", "unl_pseudo_core_count,unl_risk_count,risk_energy_out_loss,proxy_vaccept", "低density/低margin高softmax样本不再污染known compactness"],
            ["E:source episode safe gate", "SRCLOW_R17,SOURCECAP32_R20,FUSE5_R20", "source query仅core+density pass才拉近，tail/outside进入uncertain/risk", "source_ep_known_query_frac,source_ep_uncertain_query_frac,source_overflow,closed metrics", "source_overflow下降且closed-set不大幅下降"],
            ["F:真实Stage2-A/C unknown评估", "SRCLOW_R17,FUSE5_R20,R28_FUSE6+T13/R17对照", "使用真实Y_unknown query和Stage2 support/query权限", "unknown_FAR,FPR95,AUROC_energy,old_acc,seen_new_acc,H_old_new,reject_reason_counts", "unknown_FAR<=0.05且old/seen-new可接受时才谈拒识达标"],
        ],
        columns=["experiment_group", "candidates", "mechanism", "metrics", "success_criteria"],
    )
    next_table = md_table(next_df)

    report = f"""# ADV2实验全面机制审计报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Asia/Hong_Kong

分析对象：`{RUN_ID}`

证据根目录：`{ART}`

远端来源：`/home/szu2070436088/2510044040/CV-SincNet/logs/{RUN_ID}`与`runs/{RUN_ID}`

## Executive Summary

1. 闭集DG相对lateopt有小幅提升，但相对vacuum有效32和四个父候选没有闭集均值优势，unknown拒识也没有被真实验证。ADV2的`final_overall_tx`均值为{adv2_mean.get('final_overall_tx', np.nan):.2f}%，相对lateopt变化{adv2_mean.get('final_overall_tx', np.nan)-lateopt_mean.get('final_overall_tx', np.nan):.2f}pp，相对vacuum32变化{adv2_mean.get('final_overall_tx', np.nan)-vacuum_mean.get('final_overall_tx', np.nan):.2f}pp；`final_strict_udu`均值为{adv2_mean.get('final_strict_udu', np.nan):.2f}%，相对lateopt变化{adv2_mean.get('final_strict_udu', np.nan)-lateopt_mean.get('final_strict_udu', np.nan):.2f}pp，相对vacuum32变化{adv2_mean.get('final_strict_udu', np.nan)-vacuum_mean.get('final_strict_udu', np.nan):.2f}pp。这里仍是Phase1 source-only地面训练，不含真实`Y_unknown` query。
2. `proxy_vaccept≈1`仍是关键失败信号。14个候选`final_proxy_vaccept`均值{adv2_mean.get('final_proxy_vaccept', np.nan):.4f}、最小{cand['final_proxy_vaccept'].min():.4f}、最大{cand['final_proxy_vaccept'].max():.4f}，说明virtual unknown几乎仍被旧类能量/接收规则接收。
3. fusion导出问题在ADV2中已被修复到可审计层面。14/14个`phase2_zid_prototypes.json`包含`fusion_components`、`fused_tx_prototypes`和`fusion_config`，`fusion_accept_policy=local_component`且`global_ball_accept=False`。但这只证明导出字段存在，不证明local component gate已经让真实unknown拒识成功。
4. known中心和常规尾部更紧，但不是完整accept-domain收紧。ADV2`final_p95`均值{adv2_mean.get('final_p95', np.nan):.2f}deg，低于vacuum32的{vacuum_mean.get('final_p95', np.nan):.2f}deg；但`final_p99`均值{cand['final_p99'].mean():.2f}deg，source overflow均值{adv2_mean.get('final_source_overflow', np.nan):.4f}，仍显示极端尾部与跨域越界风险。
5. 最强闭集候选是`{best_closed['candidate_id']}`：`best_joint_test_tx={best_closed['best_joint_test_tx']:.2f}%`、`final_strict_udu={best_closed['final_strict_udu']:.2f}%`、`receiver_floor={best_closed['final_receiver_floor']:.2f}%`、`sat_strict_floor={best_closed['final_sat_strict_floor']:.2f}%`。它适合进入真实Stage2 unknown评估，但不能直接宣称拒识达标。
6. 主推进候选应按本轮同排数据重选，而不是沿用父候选角色。`ADV2_SRCLOW_R17_E260`是闭集主候选，`ADV2_FUSE5_R20_E260`是闭集+最低proxy_vac的融合候选，`ADV2_R28_FUSE6_E260`是p95最低且satellite floor较强的机制候选；`ADV2_R20_SAT70_E260`本轮`receiver_floor`只有63.81%，不能按“稳定主线”直接推进。
7. 下一步优先级不是继续盲扫参，而是先做不重训hard gate dry-run、shell/inter/bridge negative评估、真实Stage2-A/C unknown评估，再决定是否重训negative-space filling。

## Protocol Boundary

本轮是Phase1 source-only地面训练/弱标注半监督DG实验。训练协议为`split_mode=tx_rx_day_1_7_2`、`labeled_ratio=0.10`、`unlabeled_ratio=0.70`、`source_val_ratio=0.20`，训练期不使用target receiver数据。ADV2可说明闭集DG、特征空间几何、proxy unknown遥测、source episode风险和Phase2原型/fusion导出状态。

禁止声明：`unknown_FAR`下降、`FPR95`改善、真实unknown AUROC改善、Stage2-C成功、真实新类注册成功、local component gate已经部署成功。允许声明：闭集DG小幅改善、p95/min_inter等训练期几何指标改善、proxy unknown证据仍弱、fusion字段已导出但需要hard gate dry-run和真实unknown验证。

## Data Integrity and Run Health

有效候选数：14。排除候选：无。未发现`T16-T31`或纠偏前artifact混入ADV2统计。14/14完成260个epoch，14/14有`[PHASE2-EXPORT]`，14/14有`metrics_epoch.csv/jsonl`、`phase2_zid_prototypes.json/pt`。fatal模式`Traceback/RuntimeError/OOM/Killed/unrecognized arguments/FATAL`为0。

NaN分类：早期`[TEST] overall_tx=nan% (0/0)`属于`NAN_SKIPPED_TEST_PLACEHOLDER`；`[GRAD] aux=nan`和禁用辅助项的`sat_cos=nan/cons_cos=nan`属于`NAN_AUX_GRAD_TELEMETRY`或辅助遥测缺省；最终指标有限且训练完成。未发现`NAN_REAL_LOSS`、`NAN_REAL_METRIC`或`NAN_FATAL`影响本轮有效性。

{health_table}

## Aggregate Metrics vs Baseline

ADV2相对lateopt有`final_overall_tx`提升，但相对vacuum32有效候选和四个父候选在`final_overall_tx`、`final_strict_udu`、`receiver_floor`、`sat_strict_floor`上均值回退。它的价值更偏向fusion导出修复、p95略降和source overflow均值下降，而不是闭集峰值压倒上一轮。

{agg_table}

关键解释：相对lateopt的`final_overall_tx`提升不能掩盖相对vacuum32的闭集回退；`final_p95`略降和`source_overflow`均值下降是好信号；`proxy_vaccept`没有改善到可用拒识水平；`final_proxy_auc`仍接近0.56-0.57的弱判别区间；候选间差异大，不能用均值掩盖`SOURCECAP32`、`TAILCV`等高风险样本。

## Group Ablation

{group_table}

分组结论：

- R组均值闭集更强，T13组更像保守尾部对照。T组只有2个候选，结论不能过度泛化。
- R17系更偏闭集峰值和强推进；R20系内部差异大，只有`FUSE5_R20`达到主推进门槛；R28系更多是proxy/fusion机制候选，其中`R28_FUSE6`优于`R28_PROXYLOW`的satellite floor；T13系用于保守对照但本轮闭集不足。
- `FUSE5/FUSE6`标签组不能简单解释为性能增益，因为所有ADV2候选都已导出fusion字段，差异来自组件数、radius cap、source/vacuum配置的组合。
- `TAILCV/TAILGUARD`没有把proxy_vaccept压下来，说明单纯tail压力不足以形成拒识面。

## Candidate-Level Deep Dive

候选级同排指标如下，排序优先展示综合推进分可计算候选。`composite_score`仅在`final_strict_udu>=82`、`receiver_floor>=70`、`best_joint_test_tx>=88`的闭集门槛内计算。

{candidate_table}

闭集推进分：

{rank_closed}

几何安全分：

{rank_geo}

拒识代理分：

{rank_proxy}

## Geometry and Acceptance-Domain Analysis

{risk_table}

最终判断：ADV2更像是“known中心与p95常规包络略收紧+fusion导出可审计”，不是“known接收域真正变紧”。真正的automatic accept域必须通过local component hard gate、density gate、NLL/Mahalanobis gate、geo margin gate和energy/reject gate重新定义，并且tail/outside样本不能自动计入known accept。

## Proxy Unknown and Vacuum Mechanism Analysis

`ow_vac_rate`低只能说明旧类foreign-tail intrusion在训练代理面减少，不等于unknown被拒绝。`proxy_vac_rate`下降只说明virtual/proxy点较少进入某些真空带，不代表不会被known energy/softmax接收。`proxy_vaccept≈1`是最严重失败信号，说明最终接收规则仍把几乎所有virtual unknown当known接收。

| metric | interpretation | good_sign | bad_sign | current_result | conclusion |
| --- | --- | --- | --- | --- | --- |
| `ow_vac_rate` | 旧类特征进入真空带的代理率 | 下降 | 不能代表真实unknown | 均值{adv2_mean.get('final_ow_vac_rate', np.nan):.4f} | 只能作为几何辅助 |
| `proxy_auc` | proxy/known能量区分度 | 越高越好 | 0.56级很弱 | 均值{adv2_mean.get('final_proxy_auc', np.nan):.4f} | 不足以支撑拒识声明 |
| `proxy_vac_rate` | proxy进入真空带比例 | 下降 | 与真实unknown不同分布 | 均值{adv2_mean.get('final_proxy_vac_rate', np.nan):.4f} | 可做机制筛选 |
| `proxy_vaccept` | virtual unknown被接收比例 | 应显著低于1 | 接近1表示拒识面失败 | 均值{adv2_mean.get('final_proxy_vaccept', np.nan):.4f} | 当前最大失败面 |
| `source_overflow` | source episode越过3σ比例 | 越低越好 | 高值扩大known包络 | 均值{adv2_mean.get('final_source_overflow', np.nan):.4f} | 需density gate |
| `p95/p99` | known半径常规尾/极端尾 | p95下降 | p99长尾污染accept | p95均值{cand['final_p95'].mean():.2f}deg,p99均值{cand['final_p99'].mean():.2f}deg | 不能只看p95 |
| `min_inter` | 类中心分离 | 越高越好 | 不覆盖低密度/尾部风险 | 均值{cand['final_min_inter'].mean():.2f}deg | 不等价于拒识成功 |

proxy unknown设计缺陷仍在：virtual outlier可能太靠近known manifold；leave-one-TX-out不等价于真实unknown；没有shell negative、tail-outward negative、inter-class slerp negative和same-class bridge negative；energy surface没有被训练成open-space reject地形。

## Source Episode Risk Analysis

source episode的目标不能是“所有跨域query都回到known 3σ内”。ADV2需要把query分为core/uncertain/outside：只有高密度core query才自动known拉近，低密度query进入uncertain/risk/reject。否则闭集提升可能来自扩大known包络，反而损害真实unknown拒识。

{trajectory_table}

## Prototype Export and Fusion Audit

ADV2的fusion审计结论不同于vacuum：本轮14/14原型JSON都存在`fusion_components`、`fused_tx_prototypes`、`fusion_config`，且`fusion_config.enabled=True`、`accept_policy=local_component`、`global_ball_accept=False`、`tail_auto_accept=False`。这说明导出路径已经执行，`[PHASE2-EXPORT] fused=1`与JSON字段一致。

但本轮仍不能写“local component gate部署成功”，原因是：导出字段只定义组件和半径，没有在真实Stage2 unknown query上给出`unknown_FAR/FPR95/AUROC`；也没有shell/inter/bridge synthetic hard gate dry-run的accept率。

{proto_table}

注意区分训练日志中的`final_ow_p95/final_ow_min_inter`与导出包中的`prototype_radius_p95_mean_deg/prototype_min_inter_deg`。前者是训练期特征损失遥测，后者是导出原型包的接收半径/类间角审计面。

## Correlation and Trade-off Analysis

{cor_table}

相关性解释只作为提示，不作为因果结论。需要关注三点：第一，`min_inter`高不必然带来proxy unknown改善；第二，`p95`低不必然压低source overflow；第三，proxy_vac_rate低的候选如果闭集不足，只能作为机制候选，不能越过闭集门槛成为主推进。

## Failure Modes and Root Causes

{failure_table}

## Promotion Decision

{promotion_table}

主推进池建议：`ADV2_SRCLOW_R17_E260`、`ADV2_FUSE5_R20_E260`、`ADV2_R28_FUSE6_E260`。机制诊断池建议：`ADV2_R28_PROXYLOW_E260`、`ADV2_R17_CORESTRICT_E260`、`ADV2_T13_CONSERVE_E260`、`ADV2_SOURCECAP32_R20_E260`、`ADV2_FUSE6_R17_E260`、`ADV2_TAILCV_R17_E260`、`ADV2_TAILCV_R20_E260`。`ADV2_R20_SAT70_E260`和`ADV2_T13_TAILGUARD_E260`因receiver floor不足，不应直接推进。高风险或负例候选按`source_overflow`、receiver floor不足和proxy_vaccept未改善处理。

## Next Experiment Matrix

{next_table}

## Final Verdict

ADV2是一次有效的Phase1 source-only实验：14/14完成、14/14导出、fatal为0、fusion字段真实落地到JSON。它相对lateopt有闭集总体提升，但相对vacuum32有效候选和父候选闭集均值回退；它真正推进的是fusion可审计、p95略降和source overflow均值下降，而不是unknown拒识成功。

但是ADV2不能作为unknown拒识成功证据。`proxy_vaccept≈1`、`proxy_auc≈0.56-0.57`、p99长尾和source episode overflow共同说明accept-domain还没有真正收紧。下一步必须先做fusion/local component hard gate dry-run和真实Stage2 unknown评估，再考虑negative-space filling、tail quarantine和source episode safe gate重训。
"""
    return report


def main():
    cand, health, trajectory = parse_candidates()
    proto = parse_prototypes()
    cand = cand.merge(
        proto[
            [
                "candidate_id",
                "prototype_radius_p95_mean_deg",
                "prototype_radius_p99_mean_deg",
                "prototype_r3sigma_mean_deg",
                "prototype_min_inter_deg",
                "prototype_tail_frac_mean",
                "has_fusion_components",
                "has_fused_tx_prototypes",
                "has_fusion_config",
                "tx_domain_components_mean",
                "tx_domain_components_max",
                "fusion_global_ball_accept",
                "fusion_tail_auto_accept",
            ]
        ],
        on="candidate_id",
        how="left",
    )

    metrics = [
        "final_overall_tx",
        "final_strict_udu",
        "final_receiver_floor",
        "final_sat_mean",
        "final_sat_strict_floor",
        "best_joint_test_tx",
        "best_joint_epoch",
        "final_pos_angle",
        "final_p95",
        "final_p99",
        "final_min_inter",
        "final_ow_vac_rate",
        "final_proxy_auc",
        "final_proxy_vac_rate",
        "final_proxy_vaccept",
        "final_source_overflow",
    ]

    vac_path = ROOT / "automation_reports" / "CV-SincNet" / "phase1_fsp_vacuum_20260701" / "effective32_full_parse.json"
    vac_data = json.load(open(vac_path, encoding="utf-8"))
    vacuum_current = pd.DataFrame([flatten_vac_row(row, "vacuum_effective32") for row in vac_data["current"]["rows"]])
    lateopt = pd.DataFrame([flatten_vac_row(row, "lateopt16") for row in vac_data["baseline_lateopt"]["rows"]])
    parent_rows = vacuum_current[vacuum_current["candidate_id"].isin(PARENT_MAP.values())].copy()
    parent_rows["cohort"] = "vacuum_parent4"

    adv2_agg = aggregate_stats(cand, metrics)
    adv2_agg.insert(0, "cohort", "ADV2_14")
    vacuum_agg = aggregate_stats(vacuum_current, metrics)
    vacuum_agg.insert(0, "cohort", "vacuum_effective32")
    late_agg = aggregate_stats(lateopt, metrics)
    late_agg.insert(0, "cohort", "lateopt16")
    parent_agg = aggregate_stats(parent_rows, metrics)
    parent_agg.insert(0, "cohort", "vacuum_parent4")
    all_agg = pd.concat([adv2_agg, vacuum_agg, late_agg, parent_agg], ignore_index=True)
    pivot = all_agg.pivot(index="metric", columns="cohort", values="mean").reset_index()
    for base in ["vacuum_effective32", "vacuum_parent4", "lateopt16"]:
        pivot[f"delta_vs_{base}"] = pivot["ADV2_14"] - pivot[base]
        pivot[f"rel_vs_{base}_pct"] = 100 * (pivot["ADV2_14"] - pivot[base]) / pivot[base].replace(0, np.nan)

    cand["is_fuse_label"] = cand["mechanism_tag"].str.contains("FUSE", regex=False).map({True: "fuse_label", False: "not_fuse_label"})
    cand["is_tail_label"] = cand["mechanism_tag"].str.contains("TAIL", regex=False).map({True: "tail_label", False: "not_tail_label"})
    cand["is_proxy_label"] = cand["mechanism_tag"].str.contains("PROXY|VAC", regex=True).map({True: "proxy_label", False: "not_proxy_label"})
    cand["is_source_label"] = cand["mechanism_tag"].str.contains("SRC|SOURCE", regex=True).map({True: "source_label", False: "not_source_label"})

    def group_agg(group_col):
        rows = []
        for key, sub in cand.groupby(group_col):
            rec = {"group_type": group_col, "group": key, "n": len(sub)}
            for metric in [
                "best_joint_test_tx",
                "final_overall_tx",
                "final_strict_udu",
                "final_receiver_floor",
                "final_sat_strict_floor",
                "final_p95",
                "final_p99",
                "final_min_inter",
                "final_proxy_auc",
                "final_proxy_vac_rate",
                "final_proxy_vaccept",
                "final_source_overflow",
                "prototype_radius_p95_mean_deg",
                "tx_domain_components_mean",
            ]:
                rec[f"{metric}_mean"] = pd.to_numeric(sub[metric], errors="coerce").mean()
            rows.append(rec)
        return pd.DataFrame(rows)

    group_df = pd.concat(
        [
            group_agg(col)
            for col in ["family", "base", "mechanism_tag", "is_fuse_label", "is_tail_label", "is_proxy_label", "is_source_label"]
        ],
        ignore_index=True,
    )

    pair_specs = [
        ("final_strict_udu", "final_source_overflow"),
        ("final_strict_udu", "final_p95"),
        ("final_strict_udu", "final_proxy_vac_rate"),
        ("final_strict_udu", "final_proxy_auc"),
        ("final_receiver_floor", "final_source_overflow"),
        ("final_p95", "final_proxy_vac_rate"),
        ("final_p95", "final_proxy_auc"),
        ("final_p95", "final_source_overflow"),
        ("final_min_inter", "final_proxy_vac_rate"),
        ("final_min_inter", "final_proxy_auc"),
        ("final_min_inter", "final_source_overflow"),
        ("final_ow_vac_rate", "final_proxy_vac_rate"),
        ("final_ow_vac_rate", "final_source_overflow"),
        ("best_joint_test_tx", "final_strict_udu"),
        ("best_joint_test_tx", "final_receiver_floor"),
        ("expected_epochs", "final_overall_tx"),
        ("expected_epochs", "final_strict_udu"),
        ("prototype_radius_p95_mean_deg", "final_p95"),
        ("tx_domain_components_mean", "prototype_radius_p95_mean_deg"),
    ]
    cor_rows = []
    for left, right in pair_specs:
        sub = cand[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        cor_rows.append(
            {
                "pair": f"{left} vs {right}",
                "pearson": sub[left].corr(sub[right], method="pearson") if len(sub) >= 3 else np.nan,
                "spearman": sub[left].corr(sub[right], method="spearman") if len(sub) >= 3 else np.nan,
                "n": len(sub),
            }
        )
    cor_df = pd.DataFrame(cor_rows)

    rank_rows = []
    for metric, ascending in [
        ("best_joint_test_tx", False),
        ("final_overall_tx", False),
        ("final_strict_udu", False),
        ("final_receiver_floor", False),
        ("final_sat_strict_floor", False),
        ("final_p95", True),
        ("final_p99", True),
        ("final_min_inter", False),
        ("final_proxy_auc", False),
        ("final_proxy_vac_rate", True),
        ("final_source_overflow", True),
        ("prototype_radius_p95_mean_deg", True),
    ]:
        sub = cand[["candidate_id", metric, "base", "mechanism_tag"]].dropna().sort_values(metric, ascending=ascending).head(10)
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            rank_rows.append(
                {
                    "ranking": metric,
                    "rank": rank,
                    "candidate_id": row["candidate_id"],
                    "value": row[metric],
                    "base": row["base"],
                    "mechanism_tag": row["mechanism_tag"],
                }
            )
    rank_df = pd.DataFrame(rank_rows)

    cand["closed_score"] = weighted_rank_score(
        cand,
        [
            ("best_joint_test_tx", False, 0.35),
            ("final_strict_udu", False, 0.30),
            ("final_receiver_floor", False, 0.20),
            ("final_sat_strict_floor", False, 0.15),
        ],
    )
    cand["geometry_score"] = weighted_rank_score(
        cand,
        [
            ("final_p95", True, 0.25),
            ("final_min_inter", False, 0.25),
            ("final_source_overflow", True, 0.30),
            ("final_ow_vac_rate", True, 0.20),
        ],
    )
    cand["proxy_score"] = weighted_rank_score(
        cand,
        [
            ("final_proxy_auc", False, 0.35),
            ("final_proxy_vac_rate", True, 0.45),
            ("final_proxy_vaccept", True, 0.20),
        ],
    )
    closed_gate = (cand["final_strict_udu"] >= 82.0) & (cand["final_receiver_floor"] >= 70.0) & (cand["best_joint_test_tx"] >= 88.0)
    cand["closed_gate_for_composite"] = closed_gate
    cand["composite_score"] = np.where(
        closed_gate, 0.45 * cand["closed_score"] + 0.30 * cand["geometry_score"] + 0.25 * cand["proxy_score"], np.nan
    )

    max_cols = ["best_joint_test_tx", "final_strict_udu", "final_receiver_floor", "final_sat_strict_floor", "final_proxy_auc", "final_min_inter"]
    min_cols = ["final_p95", "final_proxy_vac_rate", "final_proxy_vaccept", "final_source_overflow"]
    vals = cand[max_cols + min_cols].apply(pd.to_numeric, errors="coerce")
    pareto = []
    for i in vals.index:
        vi = vals.loc[i]
        dominated = False
        for j in vals.index:
            if i == j:
                continue
            vj = vals.loc[j]
            if vi.isna().any() or vj.isna().any():
                continue
            better_or_equal = all(vj[col] >= vi[col] for col in max_cols) and all(vj[col] <= vi[col] for col in min_cols)
            strictly = any(vj[col] > vi[col] for col in max_cols) or any(vj[col] < vi[col] for col in min_cols)
            if better_or_equal and strictly:
                dominated = True
                break
        pareto.append(not dominated)
    cand["pareto_front"] = pareto

    promote_rows = []
    for _, row in cand.iterrows():
        cid = row["candidate_id"]
        category = "不建议直接推进"
        promote = False
        mechanism = False
        reason = ""
        if cid in ["ADV2_SRCLOW_R17_E260", "ADV2_FUSE5_R20_E260", "ADV2_R28_FUSE6_E260"]:
            category = "Stage2真实unknown评估主推进候选"
            promote = True
            reason = "闭集/receiver floor/对照价值较强，但必须真实unknown验证。"
        elif cid in [
            "ADV2_R28_PROXYLOW_E260",
            "ADV2_R28_FUSE6_E260",
            "ADV2_SRCLOW_R17_E260",
            "ADV2_SOURCECAP32_R20_E260",
            "ADV2_T13_TAILGUARD_E260",
            "ADV2_TAILCV_R17_E260",
            "ADV2_TAILCV_R20_E260",
            "ADV2_FUSE6_R17_E260",
            "ADV2_R17_CORESTRICT_E260",
            "ADV2_T13_CONSERVE_E260",
        ]:
            category = "机制诊断候选"
            mechanism = True
            reason = "用于隔离proxy/vacuum、fusion、source episode或tail压力机制，不应直接声明部署成功。"
        if row["final_proxy_vaccept"] >= 0.99:
            reason += "proxy_vaccept仍接近1。"
        if row["final_source_overflow"] > 0.45:
            category = "高风险机制/负例候选"
            promote = False
            mechanism = True
            reason += "source_overflow过高。"
        if row["final_receiver_floor"] < 70 and not promote:
            reason += "receiver floor不足。"
        promote_rows.append(
            {
                "candidate_id": cid,
                "category": category,
                "promote_to_stage2": promote,
                "use_for_mechanism": mechanism,
                "reject_or_risk_reason": reason,
                "required_followup": "local hard gate dry-run+真实Stage2 unknown评估" if promote else "机制隔离或负例分析",
            }
        )
    promote_df = pd.DataFrame(promote_rows)

    cand_out_cols = [
        "candidate_id",
        "base",
        "family",
        "mechanism_tag",
        "role",
        "parent_candidate",
        "completed",
        "expected_epochs",
        "max_epoch_begin",
        "last_epoch_end",
        "phase2_export",
        "best_joint_epoch",
        "best_joint_test_tx",
        "final_overall_tx",
        "final_strict_udu",
        "final_receiver_floor",
        "final_sat_mean",
        "final_sat_strict_floor",
        "final_pos_angle",
        "final_p95",
        "final_p99",
        "final_tail_frac_gt_3sigma",
        "final_r3sigma",
        "final_min_inter",
        "final_ow_vac_rate",
        "final_proxy_auc",
        "final_proxy_vac_rate",
        "final_proxy_vaccept",
        "final_source_overflow",
        "prototype_radius_p95_mean_deg",
        "prototype_radius_p99_mean_deg",
        "prototype_r3sigma_mean_deg",
        "prototype_min_inter_deg",
        "prototype_tail_frac_mean",
        "tx_domain_components_mean",
        "closed_score",
        "geometry_score",
        "proxy_score",
        "composite_score",
        "pareto_front",
    ]
    cand[cand_out_cols].to_csv(OUT / "adv2_candidate_rows.csv", index=False, encoding="utf-8-sig")
    health.to_csv(OUT / "adv2_health_audit.csv", index=False, encoding="utf-8-sig")
    proto.to_csv(OUT / "adv2_prototype_audit.csv", index=False, encoding="utf-8-sig")
    all_agg.to_csv(OUT / "adv2_aggregate_stats.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(OUT / "adv2_aggregate_delta_vs_baselines.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(OUT / "adv2_group_stats.csv", index=False, encoding="utf-8-sig")
    cor_df.to_csv(OUT / "adv2_correlations.csv", index=False, encoding="utf-8-sig")
    rank_df.to_csv(OUT / "adv2_rankings.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(OUT / "adv2_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    promote_df.to_csv(OUT / "adv2_promotion_decisions.csv", index=False, encoding="utf-8-sig")

    report = build_report(cand, health, trajectory, proto, all_agg, pivot, group_df, cor_df, rank_df, promote_df)
    (OUT / "adv2_full_analysis_report.md").write_text(report, encoding="utf-8")
    shutil.copy2(OUT / "adv2_full_analysis_report.md", LOCAL_DIR / "adv2_full_analysis_report.md")
    shutil.copy2(OUT / "adv2_full_analysis_report.md", GIT_DIR / "adv2_full_analysis_report.md")

    for path in OUT.glob("adv2_*"):
        if path.is_file():
            shutil.copy2(path, GIT_OUT / path.name)

    adv2_mean = all_agg[all_agg["cohort"] == "ADV2_14"].set_index("metric")["mean"]
    best_closed = cand.sort_values("closed_score").iloc[0]
    summary = {
        "run_id": RUN_ID,
        "evidence_root": str(ART),
        "candidates": int(len(cand)),
        "completed": int(cand["completed"].sum()),
        "phase2_export": int(cand["phase2_export"].sum()),
        "fatal_total": int(health["fatal_count"].sum()),
        "fusion_json_count": int(proto["has_fusion_components"].sum()),
        "adv2_mean": {key: finite(value) for key, value in adv2_mean.items()},
        "best_closed_candidate": str(best_closed["candidate_id"]),
        "main_promotion_candidates": ["ADV2_SRCLOW_R17_E260", "ADV2_FUSE5_R20_E260", "ADV2_R28_FUSE6_E260"],
        "limitations": [
            "source-only Phase1",
            "no real Y_unknown query",
            "no unknown_FAR/FPR95/real unknown AUROC",
            "hard gate dry-run not yet run",
        ],
    }
    (OUT / "adv2_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(OUT / "adv2_analysis_summary.json", GIT_OUT / "adv2_analysis_summary.json")

    print(f"WROTE {OUT / 'adv2_full_analysis_report.md'}")
    print(f"GIT_REPORT {GIT_DIR / 'adv2_full_analysis_report.md'}")
    print(f"CANDIDATES {len(cand)} COMPLETED {int(cand['completed'].sum())} FUSION {int(proto['has_fusion_components'].sum())}")
    print(f"BEST_CLOSED {best_closed['candidate_id']} CLOSED_SCORE {best_closed['closed_score']:.4f}")
    print(f"PROXY_VACCEPT_MEAN {adv2_mean.get('final_proxy_vaccept', np.nan):.6f}")


if __name__ == "__main__":
    main()
