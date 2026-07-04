#!/usr/bin/env python3
"""Sweep Phase2-C qKNN scalability across new-class counts and small K.

The script keeps the existing confusion-aware qKNN probe as the scoring
authority, then orchestrates comparable episodes:

* strict target-domain episodes use ``target_unknown`` only;
* proxy-domain capacity diagnostics may append ``proxy_unknown`` labels and
  are not Stage2-C deployment evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROBE = SCRIPT_DIR / "phase2_confusion_aware_qknn_probe.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_source_guarded_qknn_sweep as qknn


DEFAULT_OLD = "14-10,14-7,20-15,20-19,6-15,8-20"
DEFAULT_BASE_NEW = "10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3"


def _role_counts(feature_npz: Path) -> dict[str, Counter[str]]:
    data = np.load(feature_npz, allow_pickle=True)
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    out: dict[str, Counter[str]] = {}
    for role in sorted({str(value) for value in roles.tolist()}):
        out[role] = Counter(tx_ids[roles == role].tolist())
    return out


def _ordered_labels(
    *,
    counts: Counter[str],
    preferred: list[str],
    exclude: set[str],
    min_count: int,
) -> list[str]:
    labels: list[str] = []
    for label in preferred:
        if label not in exclude and counts.get(label, 0) >= min_count and label not in labels:
            labels.append(label)
    for label, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        if label in exclude or label in labels or count < min_count:
            continue
        labels.append(str(label))
    return labels


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load_probe_best(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    support_diversity = summary.get("best_by_support_diversity") or []
    support_loo = summary.get("best_by_support_loo") or []
    query = summary.get("best_by_query") or []
    if support_diversity:
        selected = dict(support_diversity[0])
        selected["selection_rule"] = "best_by_support_diversity"
    elif support_loo:
        selected = dict(support_loo[0])
        selected["selection_rule"] = "best_by_support_loo"
    elif query:
        selected = dict(query[0])
        selected["selection_rule"] = "best_by_query_AUDIT_ONLY"
    else:
        selected = {"selection_rule": "NO_VALID_ROW"}
    oracle = dict(query[0]) if query else {}
    return {"selected": selected, "oracle_query_best": oracle, "probe_rows": int(summary.get("rows_count", 0))}


def _row_from_probe(
    *,
    scope: str,
    new_count: int,
    k: int,
    new_role: str,
    new_labels: list[str],
    query_per_old: int,
    query_per_new: int,
    output_json: Path,
) -> dict[str, Any]:
    best = _load_probe_best(output_json)
    row = best["selected"]
    oracle = best["oracle_query_best"]
    return {
        "scope": scope,
        "new_count": int(new_count),
        "k": int(k),
        "new_role": new_role,
        "query_per_old": int(query_per_old),
        "query_per_new": int(query_per_new),
        "new_tx_ids": new_labels,
        "probe_rows": int(best["probe_rows"]),
        "selection_rule": row.get("selection_rule", ""),
        "seed": row.get("seed", ""),
        "support_selection_policy": row.get("support_selection_policy", ""),
        "topm": row.get("topm", ""),
        "proto_mix": row.get("proto_mix", ""),
        "radius_norm": row.get("radius_norm", ""),
        "old_bias": row.get("old_bias", ""),
        "query_old_acc": row.get("query_old_acc", 0.0),
        "query_min_old_class_acc": row.get("query_min_old_class_acc", 0.0),
        "query_seen_new_acc": row.get("query_seen_new_acc", 0.0),
        "query_min_seen_new_class_acc": row.get("query_min_seen_new_class_acc", 0.0),
        "query_passes_joint_target": row.get("query_passes_joint_target", False),
        "query_per_old_acc": row.get("query_per_old_acc", {}),
        "query_per_new_acc": row.get("query_per_new_acc", {}),
        "support_diversity_mean": row.get("support_diversity_mean", 0.0),
        "support_scenario_coverage_mean": row.get("support_scenario_coverage_mean", 0.0),
        "stored_quantized_support_code_count": row.get("stored_quantized_support_code_count", 0),
        "oracle_query_old_acc": oracle.get("query_old_acc", 0.0),
        "oracle_query_seen_new_acc": oracle.get("query_seen_new_acc", 0.0),
        "oracle_query_min_seen_new_class_acc": oracle.get("query_min_seen_new_class_acc", 0.0),
        "output_json": str(output_json),
    }


def _write_report(report_path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase2 Many-New Count/K Sweep")
    lines.append("")
    lines.append(f"- objective: {summary['objective']}")
    lines.append(f"- feature_npz: `{summary['feature_npz']}`")
    lines.append(f"- old_tx_ids: `{','.join(summary['old_tx_ids'])}`")
    lines.append(f"- K grid: `{','.join(str(k) for k in summary['k_grid'])}`")
    lines.append(f"- strict counts: `{','.join(str(n) for n in summary['strict_counts'])}`")
    lines.append(f"- proxy diagnostic counts: `{','.join(str(n) for n in summary['proxy_counts'])}`")
    lines.append(f"- strict target-domain candidate new classes: `{summary['strict_candidate_count']}`")
    lines.append(f"- proxy-domain candidate new classes: `{summary['proxy_candidate_count']}`")
    lines.append("- claim boundary: strict rows are target-domain Stage2-C probes; proxy rows are many-new capacity diagnostics because `proxy_unknown` is not the same target receiver domain.")
    lines.append("- selection rule: main rows select by support-only diversity over the seed/policy grid; query-oracle rows are audit-only upper bounds and are not deployable selection evidence.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| scope | new_count | K | query/class old/new | old_acc | min_old | seen_new_acc | min_new | pass_joint | support_codes | selection | seed/policy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|")
    for row in summary["rows"]:
        lines.append(
                "| {scope} | {new_count} | {k} | {qold}/{qnew} | {old} | {minold} | {new} | {minnew} | {passjoint} | {codes} | {selection} | {seed}/{policy} |".format(
                scope=row["scope"],
                new_count=row["new_count"],
                k=row["k"],
                qold=row["query_per_old"],
                qnew=row["query_per_new"],
                old=_pct(row["query_old_acc"]),
                minold=_pct(row["query_min_old_class_acc"]),
                new=_pct(row["query_seen_new_acc"]),
                minnew=_pct(row["query_min_seen_new_class_acc"]),
                passjoint=str(bool(row["query_passes_joint_target"])),
                codes=row["stored_quantized_support_code_count"],
                selection=row["selection_rule"],
                seed=row["seed"],
                policy=row["support_selection_policy"],
            )
        )
    lines.append("")
    lines.append("## Query-Oracle Audit")
    lines.append("")
    lines.append("| scope | new_count | K | oracle_old_acc | oracle_seen_new_acc | oracle_min_new |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        lines.append(
            "| {scope} | {new_count} | {k} | {old} | {new} | {minnew} |".format(
                scope=row["scope"],
                new_count=row["new_count"],
                k=row["k"],
                old=_pct(row["oracle_query_old_acc"]),
                new=_pct(row["oracle_query_seen_new_acc"]),
                minnew=_pct(row["oracle_query_min_seen_new_class_acc"]),
            )
        )
    lines.append("")
    lines.append("## Per-Class Accuracy")
    for row in summary["rows"]:
        lines.append("")
        lines.append(f"### {row['scope']} new_count={row['new_count']} K={row['k']}")
        lines.append("")
        lines.append("Old classes:")
        lines.append("")
        lines.append("| tx | acc |")
        lines.append("|---|---:|")
        for label, acc in sorted(row["query_per_old_acc"].items()):
            lines.append(f"| {label} | {_pct(acc)} |")
        lines.append("")
        lines.append("New classes:")
        lines.append("")
        lines.append("| tx | acc |")
        lines.append("|---|---:|")
        for label, acc in sorted(row["query_per_new_acc"].items()):
            lines.append(f"| {label} | {_pct(acc)} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- summary_json: `{summary['output_json']}`")
    lines.append(f"- summary_csv: `{summary['output_csv']}`")
    for row in summary["rows"]:
        lines.append(f"- probe {row['scope']} N={row['new_count']} K={row['k']}: `{row['output_json']}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--old_tx_ids", default=DEFAULT_OLD)
    parser.add_argument("--base_new_tx_ids", default=DEFAULT_BASE_NEW)
    parser.add_argument("--strict_counts", default="2,5,10")
    parser.add_argument("--proxy_counts", default="15,20,30")
    parser.add_argument("--k_grid", default="5,10")
    parser.add_argument("--policies", default="stable_first,scenario_diverse")
    parser.add_argument("--seed_start", type=int, default=421000)
    parser.add_argument("--seed_count", type=int, default=120)
    parser.add_argument("--topm", type=int, default=4)
    parser.add_argument("--proto_mix", type=float, default=0.15)
    parser.add_argument("--radius_norm", type=float, default=0.1)
    parser.add_argument("--old_bias", type=float, default=0.001)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--strict_new_role", default="target_unknown")
    parser.add_argument("--proxy_new_role", default="proxy_unknown")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    feature_npz = Path(args.feature_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_labels = qknn._parse_csv(args.old_tx_ids)
    base_new = qknn._parse_csv(args.base_new_tx_ids)
    strict_counts = qknn._parse_int_csv(args.strict_counts)
    proxy_counts = qknn._parse_int_csv(args.proxy_counts)
    k_grid = qknn._parse_int_csv(args.k_grid)
    max_k = max(k_grid)

    counts_by_role = _role_counts(feature_npz)
    old_counts = counts_by_role.get(str(args.old_role), Counter())
    strict_counts_by_label = counts_by_role.get(str(args.strict_new_role), Counter())
    proxy_counts_by_label = counts_by_role.get(str(args.proxy_new_role), Counter())
    exclude = set(old_labels)
    strict_labels_all = _ordered_labels(
        counts=strict_counts_by_label,
        preferred=base_new,
        exclude=exclude,
        min_count=max_k + 1,
    )
    proxy_labels_all = _ordered_labels(
        counts=proxy_counts_by_label,
        preferred=[],
        exclude=exclude,
        min_count=max_k + 1,
    )
    rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, int, str, list[str]]] = []
    for count in strict_counts:
        if count <= len(strict_labels_all):
            tasks.append(("STRICT_TARGET_DOMAIN", int(count), str(args.strict_new_role), strict_labels_all[:count]))
    for count in proxy_counts:
        if count <= len(proxy_labels_all):
            tasks.append(("PROXY_DOMAIN_CAPACITY_DIAGNOSTIC", int(count), str(args.proxy_new_role), proxy_labels_all[:count]))

    for scope, new_count, new_role, new_labels in tasks:
        for k in k_grid:
            min_old = min(old_counts.get(label, 0) for label in old_labels)
            role_counts = strict_counts_by_label if new_role == str(args.strict_new_role) else proxy_counts_by_label
            min_new = min(role_counts.get(label, 0) for label in new_labels)
            query_per_old = int(min_old) - int(k)
            query_per_new = int(min_new) - int(k)
            if query_per_old <= 0 or query_per_new <= 0:
                continue
            stem = f"{scope.lower()}_n{new_count}_k{k}"
            probe_json = output_dir / f"{stem}.json"
            probe_csv = output_dir / f"{stem}.csv"
            cmd = [
                sys.executable,
                str(PROBE),
                "--feature_npz",
                str(feature_npz),
                "--output_json",
                str(probe_json),
                "--output_csv",
                str(probe_csv),
                "--old_tx_ids",
                ",".join(old_labels),
                "--new_tx_ids",
                ",".join(new_labels),
                "--old_role",
                str(args.old_role),
                "--new_role",
                new_role,
                "--k_old",
                str(k),
                "--k_new",
                str(k),
                "--pool_per_old",
                str(k),
                "--pool_per_new",
                str(k),
                "--query_per_old",
                str(query_per_old),
                "--query_per_new",
                str(query_per_new),
                "--policies",
                str(args.policies),
                "--seed_start",
                str(args.seed_start),
                "--seed_count",
                str(args.seed_count),
                "--topm_grid",
                str(args.topm),
                "--proto_mix_grid",
                str(args.proto_mix),
                "--radius_norm_grid",
                str(args.radius_norm),
                "--old_bias_grid",
                str(args.old_bias),
                "--neg_lambda_grid",
                "0",
                "--neg_threshold_grid",
                "0.75",
                "--neg_margin_grid",
                "0",
                "--mutual_only_grid",
                "true",
                "--scenario_aware",
                "--balanced_assignment",
                "--skip_support_loo",
            ]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            subprocess.run(cmd, check=True)
            rows.append(
                _row_from_probe(
                    scope=scope,
                    new_count=new_count,
                    k=k,
                    new_role=new_role,
                    new_labels=new_labels,
                    query_per_old=query_per_old,
                    query_per_new=query_per_new,
                    output_json=probe_json,
                )
            )

    summary_json = output_dir / "manynew_count_k_sweep_summary.json"
    summary_csv = output_dir / "manynew_count_k_sweep_summary.csv"
    report_path = output_dir / "report.md"
    summary = {
        "objective": "Assess qKNN behavior when new-class count exceeds the prior 10-class setting, using K=5/10 as the main low-shot grid.",
        "feature_npz": str(feature_npz),
        "old_tx_ids": old_labels,
        "strict_new_role": str(args.strict_new_role),
        "proxy_new_role": str(args.proxy_new_role),
        "strict_candidate_count": len(strict_labels_all),
        "proxy_candidate_count": len(proxy_labels_all),
        "strict_counts": strict_counts,
        "proxy_counts": proxy_counts,
        "k_grid": k_grid,
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "fixed_hyperparameters": {
            "topm": int(args.topm),
            "proto_mix": float(args.proto_mix),
            "radius_norm": float(args.radius_norm),
            "old_bias": float(args.old_bias),
            "neg_lambda": 0.0,
            "balanced_assignment": True,
            "scenario_aware": True,
        },
        "rows": rows,
        "output_json": str(summary_json),
        "output_csv": str(summary_csv),
        "report_path": str(report_path),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "scope",
        "new_count",
        "k",
        "new_role",
        "query_per_old",
        "query_per_new",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_joint_target",
        "support_diversity_mean",
        "stored_quantized_support_code_count",
        "seed",
        "support_selection_policy",
        "query_per_old_acc",
        "query_per_new_acc",
        "new_tx_ids",
        "output_json",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            for key in ("query_per_old_acc", "query_per_new_acc", "new_tx_ids"):
                out[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    _write_report(report_path, summary)
    print(json.dumps({"rows": len(rows), "summary_json": str(summary_json), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
