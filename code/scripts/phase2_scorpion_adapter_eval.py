#!/usr/bin/env python
"""Run SCORPION-CVS on support-only virtual-negative adapter evidence.

This script keeps the Phase1/ADV3B02 feature extractor frozen. It fits only a
receiver-local ridge head and a virtual-negative known boundary from target old
and target seen-new support rows. Target unknown rows remain query-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import load_feature_npz
from phase2_scorpion_cvs_eval import _parse_ints, _parse_weighted_components, evaluate_scorpion
from phase2_virtual_negative_adapter_eval import build_virtual_negative_evidence


def _write_evidence_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "receiver_count",
        "event_id",
        "role",
        "true_label",
        "predicted_label",
        "accepted_label",
        "reject",
        "old_shield",
        "event_unknown_score",
        "event_unknown_risk",
        "high_risk_fraction",
        "vote_fraction",
        "local_accepts",
        "selected_receivers",
        "bytes_per_event",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("selected_receivers"), list):
                out["selected_receivers"] = "|".join(str(v) for v in out["selected_receivers"])
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def run_scorpion_adapter(
    *,
    feature_npz: Path,
    collab_counts: Sequence[int] | None,
    k_shot: int,
    query_per_class: int,
    seed: int,
    ridge_lambda: float,
    boundary_ridge_lambda: float,
    class_temperature: float,
    boundary_temperature: float,
    support_threshold_quantile: float,
    virtual_negative_policy: str,
    virtual_negative_shell_scale: float,
    virtual_negative_mix_pairs_per_class: int,
    event_alignment_policy: str,
    support_selection_policy: str,
    evidence_packet_bytes: float,
    risk_components: Sequence[tuple[str, float]],
    unknown_gate: float,
    old_shield_gate: float,
    min_margin: float,
    min_pvalue: float,
    min_quality: float,
) -> dict[str, Any]:
    evidence, metadata = build_virtual_negative_evidence(
        load_feature_npz(feature_npz),
        k_shot=int(k_shot),
        query_per_class=int(query_per_class),
        seed=int(seed),
        ridge_lambda=float(ridge_lambda),
        boundary_ridge_lambda=float(boundary_ridge_lambda),
        class_temperature=float(class_temperature),
        boundary_temperature=float(boundary_temperature),
        support_threshold_quantile=float(support_threshold_quantile),
        virtual_negative_policy=str(virtual_negative_policy),
        virtual_negative_shell_scale=float(virtual_negative_shell_scale),
        virtual_negative_mix_pairs_per_class=int(virtual_negative_mix_pairs_per_class),
        event_alignment_policy=str(event_alignment_policy),
        support_selection_policy=str(support_selection_policy),
        evidence_packet_bytes=float(evidence_packet_bytes),
    )
    result = evaluate_scorpion(
        evidence,
        collab_counts=collab_counts,
        risk_components=risk_components,
        unknown_gate=float(unknown_gate),
        old_shield_gate=float(old_shield_gate),
        min_margin=float(min_margin),
        min_pvalue=float(min_pvalue),
        min_quality=float(min_quality),
        evidence_packet_bytes=float(evidence_packet_bytes),
    )
    result["algorithm"] = "SCORPION-CVS-support-virtual-negative-adapter"
    result["adapter_metadata"] = metadata
    result["evidence_row_count"] = len(evidence)
    result["non_deployment_diagnostic"] = True
    result["unknown_query_used_for_threshold_fit"] = False
    result["_evidence_rows"] = evidence
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_rows_csv", type=Path)
    parser.add_argument("--output_evidence_csv", type=Path)
    parser.add_argument("--collab_counts", default="all")
    parser.add_argument("--k_shot", type=int, default=8)
    parser.add_argument("--query_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4070303)
    parser.add_argument("--ridge_lambda", type=float, default=0.1)
    parser.add_argument("--boundary_ridge_lambda", type=float, default=0.1)
    parser.add_argument("--class_temperature", type=float, default=0.05)
    parser.add_argument("--boundary_temperature", type=float, default=0.25)
    parser.add_argument("--support_threshold_quantile", type=float, default=0.05)
    parser.add_argument("--virtual_negative_policy", choices=["shell", "midpoint", "mix", "shell_mix"], default="shell_mix")
    parser.add_argument("--virtual_negative_shell_scale", type=float, default=1.5)
    parser.add_argument("--virtual_negative_mix_pairs_per_class", type=int, default=4)
    parser.add_argument("--event_alignment_policy", choices=["strict_event_key", "receiver_domain_ranked"], default="receiver_domain_ranked")
    parser.add_argument("--support_selection_policy", choices=["stable_first", "centroid", "scenario_diverse"], default="stable_first")
    parser.add_argument("--evidence_packet_bytes", type=float, default=112.0)
    parser.add_argument(
        "--risk_components",
        default="virtual_unknown_risk:0.45,class_negative_risk:0.25,score_risk:0.20,margin_risk:0.10",
    )
    parser.add_argument("--unknown_gate", type=float, default=0.52)
    parser.add_argument("--old_shield_gate", type=float, default=0.68)
    parser.add_argument("--min_margin", type=float, default=0.02)
    parser.add_argument("--min_pvalue", type=float, default=0.0)
    parser.add_argument("--min_quality", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_scorpion_adapter(
        feature_npz=args.feature_npz,
        collab_counts=_parse_ints(args.collab_counts),
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        boundary_ridge_lambda=float(args.boundary_ridge_lambda),
        class_temperature=float(args.class_temperature),
        boundary_temperature=float(args.boundary_temperature),
        support_threshold_quantile=float(args.support_threshold_quantile),
        virtual_negative_policy=str(args.virtual_negative_policy),
        virtual_negative_shell_scale=float(args.virtual_negative_shell_scale),
        virtual_negative_mix_pairs_per_class=int(args.virtual_negative_mix_pairs_per_class),
        event_alignment_policy=str(args.event_alignment_policy),
        support_selection_policy=str(args.support_selection_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        risk_components=_parse_weighted_components(args.risk_components),
        unknown_gate=float(args.unknown_gate),
        old_shield_gate=float(args.old_shield_gate),
        min_margin=float(args.min_margin),
        min_pvalue=float(args.min_pvalue),
        min_quality=float(args.min_quality),
    )
    evidence_rows = result.pop("_evidence_rows")
    result["feature_npz"] = str(args.feature_npz)
    result["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    result["python_executable"] = sys.executable
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_rows_csv:
        _write_rows_csv(args.output_rows_csv, result["rows"])
    if args.output_evidence_csv:
        _write_evidence_csv(args.output_evidence_csv, evidence_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
