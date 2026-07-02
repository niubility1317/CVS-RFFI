#!/usr/bin/env python
"""Fuse existing source-calibrated branch decisions for sat-only rejection.

Each branch keeps its own source/proxy calibrated threshold from its existing
metrics/score_table. Target query labels are used only for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


CELL_NAMES = [
    "rx20_1_u10",
    "rx20_1_u1",
    "rx3_19_u10",
    "rx3_19_u1",
    "rx7_14_u10",
    "rx7_14_u1",
    "rx7_7_u10",
    "rx7_7_u1",
    "rx8_8_u10",
    "rx8_8_u1",
]

BRANCH_TEMPLATES = {
    "v3_lcos_ret": "phase1_adv3b02_multiview_keepold_{cell}_20260702/LEOADAPT3_LINR_COS/ADAPT3_LIN_SRC9999",
    "v3_lcos_strict": "phase1_adv3b02_multiview_keepold_{cell}_20260702/LEOADAPT3_LINR_COS/ADAPT3_MLP64_MIN05",
    "v3_mlp_ret": "phase1_adv3b02_multiview_keepold_{cell}_20260702/LEOADAPT3_MLP_ID/ADAPT3_LIN_SRC9999",
    "v3_mlp_strict": "phase1_adv3b02_multiview_keepold_{cell}_20260702/LEOADAPT3_MLP_ID/ADAPT3_MLP64_MIN05",
    "v11a_ret": "phase1_adv3b02_iqpre_v11a_{cell}_20260703/IQPRE_LIN_SRC1000",
    "v11a_strict": "phase1_adv3b02_iqpre_v11a_{cell}_20260703/IQPRE_MLP64_MIN05",
    "v11b_ret": "phase1_adv3b02_iqpre_v11b_{cell}_20260703/IQPRE_LIN_SRC1000",
    "v11b_strict": "phase1_adv3b02_iqpre_v11b_{cell}_20260703/IQPRE_MLP64_MIN05",
}

BRANCH_SETS = {
    "retention4": ["v3_lcos_ret", "v3_mlp_ret", "v11a_ret", "v11b_ret"],
    "strict4": ["v3_lcos_strict", "v3_mlp_strict", "v11a_strict", "v11b_strict"],
    "v3_mixed4": ["v3_lcos_ret", "v3_lcos_strict", "v3_mlp_ret", "v3_mlp_strict"],
    "iqpre_mixed4": ["v11a_ret", "v11a_strict", "v11b_ret", "v11b_strict"],
    "hetero_mixed4a": ["v3_lcos_ret", "v3_lcos_strict", "v11a_ret", "v11a_strict"],
    "hetero_mixed4b": ["v3_mlp_ret", "v3_mlp_strict", "v11b_ret", "v11b_strict"],
    "all8": [
        "v3_lcos_ret",
        "v3_lcos_strict",
        "v3_mlp_ret",
        "v3_mlp_strict",
        "v11a_ret",
        "v11a_strict",
        "v11b_ret",
        "v11b_strict",
    ],
}

KEY_FIELDS = ["role", "tx_id", "rx_id", "day_id", "sig_id"]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def _load_branch(branch_dir: Path) -> tuple[dict[tuple[str, ...], dict], dict]:
    score_path = branch_dir / "score_table.csv"
    metrics_path = branch_dir / "metrics.json"
    if not score_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(str(branch_dir))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, ...], dict] = {}
    with score_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _row_key(row)
            rows[key] = {
                **{field: str(row.get(field, "")) for field in KEY_FIELDS},
                "accepted": _truthy(row.get("accepted", "0")),
                "is_known_query": _truthy(row.get("is_known_query", "0")),
                "is_unknown_query": _truthy(row.get("is_unknown_query", "0")),
                "closed_correct_known": _truthy(row.get("closed_correct_known", "0")),
                "unknown_score": float(row.get("unknown_score", "nan")),
            }
    return rows, metrics


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _candidate_thresholds(n: int, spec: str) -> list[int]:
    if str(spec).strip().lower() == "all":
        return list(range(1, int(n) + 1))
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(max(1, min(int(n), int(item))))
    return sorted(set(out))


def _evaluate_vote(
    *,
    cell: str,
    branch_set: str,
    branch_names: list[str],
    branches: dict[str, dict[tuple[str, ...], dict]],
    branch_metrics: dict[str, dict],
    primary: str,
    min_accepts: int,
) -> dict:
    common = None
    for name in branch_names:
        keys = set(branches[name])
        common = keys if common is None else common & keys
    if not common:
        raise ValueError(f"empty branch intersection for {cell} {branch_set}")
    primary_rows = branches[primary]
    known_keys = [k for k in sorted(common) if primary_rows[k]["is_known_query"]]
    unknown_keys = [k for k in sorted(common) if primary_rows[k]["is_unknown_query"]]
    eval_keys = known_keys + unknown_keys
    accept_count = {}
    for key in eval_keys:
        accept_count[key] = sum(1 for name in branch_names if branches[name][key]["accepted"])
    accepted = {key: int(accept_count[key]) >= int(min_accepts) for key in eval_keys}
    known_total = len(known_keys)
    unknown_total = len(unknown_keys)
    known_closed_correct = sum(1 for key in known_keys if primary_rows[key]["closed_correct_known"])
    known_correct_after = sum(1 for key in known_keys if primary_rows[key]["closed_correct_known"] and accepted[key])
    known_accepted = sum(1 for key in known_keys if accepted[key])
    unknown_accepted = sum(1 for key in unknown_keys if accepted[key])
    closed_acc = _safe_rate(known_closed_correct, known_total)
    full_acc = _safe_rate(known_correct_after, known_total)
    old_drop = 100.0 * (float(closed_acc) - float(full_acc))
    far = _safe_rate(unknown_accepted, unknown_total)
    return {
        "cell": cell,
        "branch_set": branch_set,
        "branches": ",".join(branch_names),
        "primary_branch": primary,
        "fusion_rule": f"min_accepts_{int(min_accepts)}_of_{len(branch_names)}",
        "min_accepts": int(min_accepts),
        "branch_count": len(branch_names),
        "common_group_count": len(common),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
        "known_closed_correct": known_closed_correct,
        "known_correct_after_reject": known_correct_after,
        "known_accepted_count": known_accepted,
        "unknown_accepted_count": unknown_accepted,
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
        "primary_branch_closed_old_reference": branch_metrics.get(primary, {}).get("known_closed_accuracy_no_reject", ""),
        "primary_branch_unknown_far_reference": branch_metrics.get(primary, {}).get("unknown_FAR", ""),
        "uses_target_labels_for_threshold": False,
        "uses_unknown_query_for_threshold": False,
    }


def _parse_branch_sets(text: str) -> dict[str, list[str]]:
    if not str(text).strip():
        return BRANCH_SETS
    out: dict[str, list[str]] = {}
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        name, branches = item.split(":", 1)
        out[name.strip()] = [x.strip() for x in branches.split(",") if x.strip()]
    return out


def _parse_cells(text: str) -> list[str]:
    if not str(text).strip():
        return CELL_NAMES
    return [x.strip() for x in str(text).split(",") if x.strip()]


def evaluate(args: argparse.Namespace) -> list[dict]:
    branch_sets = _parse_branch_sets(args.branch_sets)
    cells = _parse_cells(args.cells)
    rows: list[dict] = []
    for cell in cells:
        loaded: dict[str, dict[tuple[str, ...], dict]] = {}
        metrics: dict[str, dict] = {}
        for branch_name, template in BRANCH_TEMPLATES.items():
            branch_dir = Path(args.runs_root) / template.format(cell=cell)
            if not branch_dir.is_dir():
                continue
            loaded[branch_name], metrics[branch_name] = _load_branch(branch_dir)
        for set_name, branch_names in branch_sets.items():
            missing = [b for b in branch_names if b not in loaded]
            if missing:
                continue
            thresholds = _candidate_thresholds(len(branch_names), str(args.min_accepts))
            primary_names = branch_names if str(args.primary_branches).strip().lower() == "all" else [
                x.strip() for x in str(args.primary_branches).split(",") if x.strip()
            ]
            for primary in primary_names:
                if primary not in branch_names:
                    continue
                for min_accepts in thresholds:
                    rows.append(
                        _evaluate_vote(
                            cell=cell,
                            branch_set=set_name,
                            branch_names=branch_names,
                            branches=loaded,
                            branch_metrics=metrics,
                            primary=primary,
                            min_accepts=min_accepts,
                        )
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--cells", default="")
    parser.add_argument("--branch_sets", default="")
    parser.add_argument("--min_accepts", default="all")
    parser.add_argument("--primary_branches", default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = evaluate(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["cell", "branch_set"]
    leading = ["cell", "branch_set", "primary_branch", "fusion_rule"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "dual_pass": sum(1 for r in rows if r.get("passes_dual_target")), "out_csv": str(args.out_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
