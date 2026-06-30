#!/usr/bin/env python
"""Analyze where Stage2 unknown-query features fall relative to old-class gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "1.0", "true", "yes", "y"}


def finite(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def pct(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) / float(den)


def stats(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    def q(pos: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * pos)))
        return vals[int(idx)]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "min": vals[0],
        "max": vals[-1],
    }


def row_position(row: dict[str, str]) -> tuple[str, list[str]]:
    """Return a primary location bucket and non-exclusive evidence flags."""

    flags: list[str] = []
    failure_count = finite(row.get("class_envelope_failure_count")) or 0.0
    margin = finite(row.get("margin"))
    support_knn_margin = finite(row.get("support_knn_margin"))
    old_anchor_margin = finite(row.get("old_support_anchor_margin"))
    density_delta = finite(row.get("anchor_density_delta"))
    density_margin_delta = finite(row.get("anchor_density_margin_delta"))
    soft_residual = finite(row.get("soft_mixture_residual"))
    old_background_margin = finite(row.get("old_primary_background_margin"))
    void_margin = finite(row.get("void_background_margin"))

    if failure_count >= 1 or is_true(row.get("class_envelope_reject")):
        flags.append("outside_or_tail_of_class_envelope")
    if (density_delta is not None and density_delta < 0.0) or (
        density_margin_delta is not None and density_margin_delta < 0.0
    ):
        flags.append("low_local_support_density")
    if (margin is not None and margin <= 0.05) or (
        support_knn_margin is not None and support_knn_margin <= 0.05
    ) or (old_anchor_margin is not None and old_anchor_margin <= 0.0):
        flags.append("low_unknown_or_interclass_margin")
    if is_true(row.get("soft_mixture_consistency_pass")) or (
        failure_count == 0 and (soft_residual is None or soft_residual <= 1.20)
    ):
        flags.append("inside_fused_old_acceptance_ball")
    if (old_background_margin is not None and old_background_margin >= 0.0) or (
        void_margin is not None and void_margin >= 0.0
    ):
        flags.append("closer_to_pseudo_unknown_background_than_known_margin")

    if "outside_or_tail_of_class_envelope" in flags:
        primary = "old_tail_or_radius_escape"
    elif "low_local_support_density" in flags:
        primary = "density_hole_near_old"
    elif "low_unknown_or_interclass_margin" in flags:
        primary = "interclass_or_weak_margin"
    elif "inside_fused_old_acceptance_ball" in flags:
        primary = "inside_fused_old_ball"
    else:
        primary = "nearest_old_absorbed_other"
    return primary, flags


def summarize_score_table(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    false_accept_positions: Counter[str] = Counter()
    false_accept_flags: Counter[str] = Counter()
    all_unknown_positions: Counter[str] = Counter()
    all_unknown_flags: Counter[str] = Counter()
    by_tx: dict[str, Counter[str]] = defaultdict(Counter)
    values: dict[str, list[float]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            true_group = row.get("true_group") or ""
            if true_group != "unknown":
                continue
            counts["unknown_total"] += 1
            accepted = is_true(row.get("accepted"))
            if accepted:
                counts["unknown_false_accept"] += 1
            else:
                counts["unknown_rejected"] += 1
            primary, flags = row_position(row)
            all_unknown_positions[primary] += 1
            for flag in flags:
                all_unknown_flags[flag] += 1
            if accepted:
                false_accept_positions[primary] += 1
                by_tx[row.get("query_tx_id") or "missing"][primary] += 1
                for flag in flags:
                    false_accept_flags[flag] += 1
                if len(sample_rows) < 8:
                    sample_rows.append({
                        "row": row.get("row"),
                        "query_tx_id": row.get("query_tx_id"),
                        "scenario": row.get("query_sat_scenario"),
                        "predicted_label": row.get("predicted_label"),
                        "primary_position": primary,
                        "flags": flags,
                        "score": finite(row.get("score")),
                        "margin": finite(row.get("margin")),
                        "class_envelope_failure_count": finite(row.get("class_envelope_failure_count")),
                        "anchor_density_delta": finite(row.get("anchor_density_delta")),
                        "anchor_density_margin_delta": finite(row.get("anchor_density_margin_delta")),
                        "old_support_anchor_margin": finite(row.get("old_support_anchor_margin")),
                    })
            for key in (
                "score",
                "margin",
                "old_support_anchor_margin",
                "anchor_density_delta",
                "anchor_density_margin_delta",
                "class_envelope_failure_count",
                "class_envelope_residual",
                "class_envelope_margin",
                "soft_mixture_residual",
                "soft_mixture_score_margin",
                "old_primary_background_margin",
                "support_knn_margin",
            ):
                value = finite(row.get(key))
                if value is not None:
                    prefix = "false_accept" if accepted else "rejected"
                    values[f"{prefix}.{key}"].append(value)

    total = counts["unknown_total"]
    false_accept = counts["unknown_false_accept"]
    return {
        "score_table": str(path),
        "unknown_total": int(total),
        "unknown_false_accept": int(false_accept),
        "unknown_far_from_score_table": pct(false_accept, total),
        "unknown_rejected": int(counts["unknown_rejected"]),
        "false_accept_positions": dict(false_accept_positions),
        "false_accept_position_rates": {k: pct(v, false_accept) for k, v in false_accept_positions.items()},
        "false_accept_flags": dict(false_accept_flags),
        "false_accept_flag_rates": {k: pct(v, false_accept) for k, v in false_accept_flags.items()},
        "all_unknown_positions": dict(all_unknown_positions),
        "all_unknown_flags": dict(all_unknown_flags),
        "false_accept_by_tx": {k: dict(v) for k, v in sorted(by_tx.items())},
        "numeric_stats": {k: stats(v) for k, v in sorted(values.items())},
        "sample_false_accepts": sample_rows,
    }


def metrics_row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
    rollback = data.get("rollback", {}) if isinstance(data.get("rollback"), dict) else {}
    return {
        "old_acc": finite(metrics.get("old_acc") or metrics.get("old_class_accuracy")),
        "coverage": finite(metrics.get("coverage")),
        "unknown_far": finite(metrics.get("unknown_FAR") or metrics.get("unknown_false_accept_rate")),
        "full_accuracy": finite(metrics.get("full_accuracy")),
        "accepted_accuracy": finite(metrics.get("accepted_accuracy")),
        "rollback_triggered": rollback.get("rollback_triggered"),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_rows = []
    for domain in args.domains:
        slug = domain.replace("-", "_")
        for candidate in args.candidates:
            run_dir = root / "runs" / f"{args.run_prefix}_{slug}" / candidate
            score_path = run_dir / args.score_table_name
            metrics_path = run_dir / args.metrics_name
            if not score_path.exists():
                out_rows.append({"domain": domain, "candidate": candidate, "status": "missing_score_table", "path": str(score_path)})
                continue
            row = {
                "domain": domain,
                "candidate": candidate,
                "status": "ok",
                "metrics": metrics_row(metrics_path) if metrics_path.exists() else {},
                "unknown_geometry": summarize_score_table(score_path),
            }
            out_rows.append(row)
    return {
        "schema": "stage2_unknown_geometry_v1",
        "run_prefix": args.run_prefix,
        "domains": args.domains,
        "candidates": args.candidates,
        "position_definitions": {
            "old_tail_or_radius_escape": "unknown rows that fail class-envelope/radius tests but are still accepted",
            "density_hole_near_old": "unknown rows near an old candidate but below calibrated local support density",
            "interclass_or_weak_margin": "unknown rows with weak top-vs-competitor or support margin",
            "inside_fused_old_ball": "unknown rows passing class envelope and/or soft mixture consistency, consistent with oversized fused old acceptance region",
            "nearest_old_absorbed_other": "accepted unknown rows not explained by the previous diagnostic flags",
        },
        "rows": out_rows,
    }


def write_markdown(result: dict[str, Any], output: Path) -> None:
    lines = [
        "# Stage2 Unknown Geometry Diagnostic",
        "",
        "This diagnostic uses score-table side channels only. Unknown-query labels are used for post-hoc evaluation, not for threshold fitting.",
        "",
        "| Domain | Candidate | Old acc | Coverage | Unknown FAR | Main false-accept location | False-accepted unknowns | Key flags |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in result["rows"]:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('domain')} | {row.get('candidate')} |  |  |  | {row.get('status')} |  |  |")
            continue
        metrics = row["metrics"]
        geom = row["unknown_geometry"]
        positions = geom.get("false_accept_positions", {})
        main = max(positions.items(), key=lambda item: item[1])[0] if positions else "none"
        flags = geom.get("false_accept_flags", {})
        flag_text = ", ".join(f"{k}:{v}" for k, v in sorted(flags.items(), key=lambda item: item[1], reverse=True)[:3])
        def fmt(v: Any) -> str:
            return "" if v is None else f"{float(v) * 100:.2f}%"
        lines.append(
            "| {domain} | {cand} | {old} | {cov} | {far} | {main} | {fa}/{total} | {flags} |".format(
                domain=row["domain"],
                cand=row["candidate"],
                old=fmt(metrics.get("old_acc")),
                cov=fmt(metrics.get("coverage")),
                far=fmt(metrics.get("unknown_far")),
                main=main,
                fa=geom.get("unknown_false_accept"),
                total=geom.get("unknown_total"),
                flags=flag_text,
            )
        )
    lines.append("")
    lines.append("## Position Definitions")
    for key, value in result["position_definitions"].items():
        lines.append(f"- `{key}`: {value}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--domains", nargs="+", default=["3-19", "7-14", "7-7", "8-8"])
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--metrics-name", default="metrics.json")
    parser.add_argument("--score-table-name", default="score_table.csv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.output_md)
    print(json.dumps({
        "rows": len(result["rows"]),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
