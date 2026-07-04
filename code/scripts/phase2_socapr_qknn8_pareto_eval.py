#!/usr/bin/env python
"""Run repeatable SO-CAPR qknn8 route diagnostics and Pareto re-evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence


ROUTE_CONFIGS: dict[str, list[str]] = {
    "known_route": [
        "--collab_counts",
        "all",
        "--collab_group_policy",
        "available_up_to_k",
        "--partial_collab_min_receivers",
        "1",
        "--k_shot",
        "8",
        "--query_per_class",
        "20",
        "--qknn_k",
        "8",
        "--seed",
        "4070303",
        "--candidate_class_top_m",
        "2",
        "--class_evidence_top_m",
        "3",
        "--prototype_score_blend",
        "2.0",
        "--mahalanobis_score_blend",
        "1.0",
        "--support_calibration_mode",
        "leave_one_out",
        "--unknown_gate_mode",
        "score",
        "--score_threshold_combine",
        "qknn_only",
        "--scenario_aware",
        "--radius_norm",
        "0.3",
        "--fusion_policy",
        "risk_margin",
        "--collaboration_policy",
        "fixed_k",
        "--label_fusion_policy",
        "weighted_vote_margin",
        "--event_alignment_policy",
        "receiver_domain_ranked",
        "--support_selection_policy",
        "stable_first",
        "--unknown_risk_threshold",
        "0.99",
        "--accept_margin_threshold",
        "0.02",
        "--evidence_packet_bytes",
        "40",
    ],
    "safety_route": [
        "--collab_counts",
        "all",
        "--collab_group_policy",
        "available_up_to_k",
        "--partial_collab_min_receivers",
        "1",
        "--k_shot",
        "8",
        "--query_per_class",
        "20",
        "--qknn_k",
        "8",
        "--seed",
        "4070303",
        "--event_alignment_policy",
        "receiver_domain_ranked",
        "--support_selection_policy",
        "stable_first",
        "--support_calibration_mode",
        "leave_one_out",
        "--unknown_gate_mode",
        "support_envelope_consensus",
        "--score_threshold_combine",
        "max",
        "--support_quantile",
        "0.10",
        "--score_quantile",
        "0.10",
        "--margin_quantile",
        "0.10",
        "--radius_quantile",
        "0.90",
        "--mahalanobis_quantile",
        "0.90",
        "--evt_tail_quantile",
        "0.75",
        "--risk_temperature",
        "0.035",
        "--radius_temperature",
        "0.02",
        "--margin_temperature",
        "0.02",
        "--mahalanobis_temperature",
        "0.12",
        "--evt_temperature",
        "0.04",
        "--mahalanobis_score_blend",
        "0.20",
        "--mahalanobis_score_temperature",
        "0.20",
        "--class_conformal_enabled",
        "--class_conformal_min_support",
        "2",
        "--receiver_class_reliability_policy",
        "support_calibrated",
        "--class_reliability_policy",
        "conformal_margin_risk",
        "--class_evidence_top_m",
        "2",
        "--class_score_threshold_enabled",
        "--class_score_threshold_quantile",
        "0.10",
        "--virtual_unknown_calibration_enabled",
        "--virtual_unknown_samples_per_class",
        "4",
        "--virtual_unknown_risk_enabled",
        "--virtual_unknown_risk_samples_per_class",
        "4",
        "--class_negative_risk_enabled",
        "--class_shell_unknown_risk_enabled",
        "--unknown_risk_threshold",
        "0.65",
        "--accept_margin_threshold",
        "0.02",
        "--fusion_policy",
        "candidate_set_cvs",
        "--label_fusion_policy",
        "weighted_vote_margin",
        "--receiver_selection_policy",
        "support_quality_prior",
        "--candidate_set_unknown_reject_risk",
        "0.65",
        "--candidate_set_max_event_unknown_risk",
        "0.85",
        "--candidate_set_max_label_unknown_risk",
        "0.75",
        "--candidate_set_min_conformal_pvalue",
        "0.05",
        "--candidate_set_min_label_receiver_class_reliability",
        "0.05",
        "--candidate_set_min_score_gap",
        "0.01",
        "--evidence_packet_bytes",
        "128",
    ],
}

PARETO_FUSIONS = ["risk_margin", "scg_qknn_cvs", "support_router_cvs", "candidate_set_cvs"]
PARETO_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _joint_score(metrics: Mapping[str, Any]) -> float:
    return float(metrics.get("old_acc", 0.0) or 0.0) + float(metrics.get("seen_new_acc", 0.0) or 0.0) + float(
        metrics.get("unknown_reject_rate", 0.0) or 0.0
    ) - float(metrics.get("unknown_FAR", 0.0) or 0.0) - float(metrics.get("defer_rate", 0.0) or 0.0)


def _pareto_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    route_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fusion in PARETO_FUSIONS:
        for threshold in PARETO_THRESHOLDS:
            result = evaluate_collaborative_open_set_evidence(
                evidence_rows,
                collab_counts="all",
                threshold_selection_label_scope=str(metadata["threshold_scope"]),
                unknown_query_eval_only=True,
                protocol_metadata=metadata,
                strict_protocol_metadata=True,
                collab_group_policy="available_up_to_k",
                partial_collab_min_receivers=1,
                unknown_risk_threshold=float(threshold),
                accept_margin_threshold=0.02,
                fusion_policy=fusion,
                label_fusion_policy="weighted_vote_margin",
                receiver_selection_policy="support_quality_prior",
                class_reliability_policy="conformal_margin_risk",
                receiver_class_reliability_policy=str(metadata.get("receiver_class_reliability_policy", "none")),
                candidate_set_unknown_reject_risk=float(threshold),
                candidate_set_max_event_unknown_risk=min(1.0, float(threshold) + 0.15),
                candidate_set_max_label_unknown_risk=min(1.0, float(threshold) + 0.10),
                candidate_set_min_conformal_pvalue=0.05,
                candidate_set_min_label_receiver_class_reliability=0.05,
                candidate_set_min_score_gap=0.01,
            )
            for count, metrics in result["counts"].items():
                row = {
                    "route": route_name,
                    "fusion_policy": fusion,
                    "unknown_risk_threshold": float(threshold),
                    "collab_count": int(count),
                    "joint_score": _joint_score(metrics),
                    "old_acc": metrics.get("old_acc", 0.0),
                    "min_old_class_acc": metrics.get("min_old_class_acc", 0.0),
                    "seen_new_acc": metrics.get("seen_new_acc", 0.0),
                    "min_seen_new_class_acc": metrics.get("min_seen_new_class_acc", 0.0),
                    "unknown_reject_rate": metrics.get("unknown_reject_rate", 0.0),
                    "unknown_FAR": metrics.get("unknown_FAR", 0.0),
                    "known_coverage": metrics.get("known_coverage", 0.0),
                    "defer_rate": metrics.get("defer_rate", 0.0),
                    "participating_receivers_p95": metrics.get("participating_receivers_p95", 0.0),
                    "bytes_per_event": metrics.get("bytes_per_event", 0.0),
                    "latency_ms_p95": metrics.get("latency_ms_p95", 0.0),
                }
                out.append(row)
    return sorted(out, key=lambda row: float(row["joint_score"]), reverse=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _run_route(*, route: str, feature_npz: Path, output_dir: Path, force: bool) -> tuple[Path, Path]:
    result_json = output_dir / f"{route}.json"
    evidence_csv = output_dir / f"{route}_evidence.csv"
    if result_json.exists() and evidence_csv.exists() and not force:
        return result_json, evidence_csv
    script = Path(__file__).with_name("phase2_collaborative_open_set_qknn_eval.py")
    cmd = [
        sys.executable,
        str(script),
        "--feature_npz",
        str(feature_npz),
        "--output_json",
        str(result_json),
        "--output_evidence_csv",
        str(evidence_csv),
        *ROUTE_CONFIGS[route],
    ]
    subprocess.run(cmd, check=True)
    return result_json, evidence_csv


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--routes", default="known_route,safety_route")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    route_names = [part.strip() for part in str(args.routes).split(",") if part.strip()]
    for route in route_names:
        if route not in ROUTE_CONFIGS:
            raise ValueError(f"unknown route {route!r}; expected {sorted(ROUTE_CONFIGS)}")
        result_json, evidence_csv = _run_route(
            route=route,
            feature_npz=args.feature_npz,
            output_dir=args.output_dir,
            force=bool(args.force),
        )
        result = json.loads(result_json.read_text(encoding="utf-8"))
        metadata = result["qknn_metadata"]
        all_rows.extend(_pareto_rows(_read_csv_rows(evidence_csv), metadata, route_name=route))

    summary_csv = args.output_dir / "socapr_pareto_summary.csv"
    summary_json = args.output_dir / "socapr_pareto_summary.json"
    _write_csv(summary_csv, all_rows)
    summary_json.write_text(json.dumps({"rows": all_rows}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "summary_csv": str(summary_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
