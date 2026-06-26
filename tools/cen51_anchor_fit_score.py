#!/usr/bin/env python
"""Score completed CEN51-SAFD anchor-fit runs against per-shot anchors."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TARGETS: dict[int, dict[str, float]] = {
    5: {"strict": 74.52, "overall": 79.70, "rx_floor": 61.0, "sat_floor": 29.0},
    10: {"strict": 76.27, "overall": 81.54, "rx_floor": 62.0, "sat_floor": 29.0},
    20: {"strict": 77.34, "overall": 83.58, "rx_floor": 61.0, "sat_floor": 32.0},
    30: {"strict": 78.72, "overall": 85.53, "rx_floor": 65.0, "sat_floor": 34.0},
    50: {"strict": 82.31, "overall": 88.58, "rx_floor": 68.0, "sat_floor": 35.0},
    100: {"strict": 84.05, "overall": 88.45, "rx_floor": 80.0, "sat_floor": 39.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def shot(row: dict[str, str]) -> int | None:
    value = f(row, "shot")
    if value is None:
        return None
    return int(value)


def pick_overall(row: dict[str, str]) -> float | None:
    for key in ("best_test_overall_finished", "final_overall", "overall", "test_overall"):
        value = f(row, key)
        if value is not None:
            return value
    return None


def action(row: dict[str, str], target: dict[str, float]) -> str:
    strict = f(row, "final_strict") or f(row, "best_strict") or -999.0
    latest_drop = f(row, "latest_strict_drop") or 0.0
    rx_floor = f(row, "rx_floor") or -999.0
    sat_floor = f(row, "sat_floor") or -999.0
    final_val = f(row, "final_val") or 0.0

    if latest_drop > 2.0 and strict < target["strict"]:
        return "pressure_clamp_or_swad"
    if rx_floor < target["rx_floor"] - 1.0:
        return "rx_floor_repair"
    if sat_floor < target["sat_floor"] - 1.0:
        return "sat_gate_repair"
    if final_val < 90.0 and strict < target["strict"] - 1.0:
        return "identity_fit_repair"
    if strict >= target["strict"] and latest_drop <= 1.5:
        return "promote_or_seed_confirm"
    return "seed_confirm_before_controller"


def score_row(row: dict[str, str]) -> dict[str, object] | None:
    k = shot(row)
    if k not in TARGETS:
        return None
    target = TARGETS[k]
    strict = f(row, "final_strict") or f(row, "best_strict")
    if strict is None:
        return None
    overall = pick_overall(row)
    rx_floor = f(row, "rx_floor")
    sat_floor = f(row, "sat_floor")
    latest_drop = f(row, "latest_strict_drop") or 0.0

    strict_gap = strict - target["strict"]
    overall_gap = None if overall is None else overall - target["overall"]
    rx_gap = None if rx_floor is None else rx_floor - target["rx_floor"]
    sat_gap = None if sat_floor is None else sat_floor - target["sat_floor"]

    penalty = 0.0
    penalty += max(0.0, -strict_gap) * 8.0
    if overall_gap is not None:
        penalty += max(0.0, -overall_gap) * 2.0
    if rx_gap is not None:
        penalty += max(0.0, -rx_gap) * 1.5
    if sat_gap is not None:
        penalty += max(0.0, -sat_gap) * 1.0
    penalty += max(0.0, latest_drop - 1.5) * 3.0
    if row.get("stability_status") and row.get("stability_status") != "PASS":
        penalty += 6.0

    fit_score = 100.0 - penalty + max(0.0, strict_gap) * 4.0
    return {
        "run": row.get("run") or Path(row.get("path") or "").stem,
        "shot": k,
        "stability_status": row.get("stability_status") or "",
        "fit_score": round(fit_score, 3),
        "strict": round(strict, 3),
        "target_strict": target["strict"],
        "strict_gap": round(strict_gap, 3),
        "overall": "" if overall is None else round(overall, 3),
        "target_overall": target["overall"],
        "overall_gap": "" if overall_gap is None else round(overall_gap, 3),
        "rx_floor": "" if rx_floor is None else round(rx_floor, 3),
        "rx_gap": "" if rx_gap is None else round(rx_gap, 3),
        "sat_floor": "" if sat_floor is None else round(sat_floor, 3),
        "sat_gap": "" if sat_gap is None else round(sat_gap, 3),
        "latest_strict_drop": round(latest_drop, 3),
        "recommended_action": action(row, target),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CEN51-SAFD Anchor Fit Score",
        "",
        "This report scores runs against the screenshot anchors. A metric is useful only if its recommended action improves strict/overall fit without hiding rx-floor, sat-floor, or late-rollback damage.",
        "",
        "| rank | run | K | score | strict gap | rx gap | sat gap | action |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, 1):
        lines.append(
            f"| {idx} | `{row['run']}` | {row['shot']} | {row['fit_score']} | "
            f"{row['strict_gap']} | {row['rx_gap']} | {row['sat_gap']} | {row['recommended_action']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    with args.summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [item for item in (score_row(row) for row in csv.DictReader(handle)) if item]
    rows.sort(key=lambda item: (-float(item["fit_score"]), int(item["shot"]), str(item["run"])))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "anchor_fit_scores.csv", rows)
    write_report(args.out_dir / "anchor_fit_report.md", rows)
    print(f"loaded_runs={len(rows)}")
    print(f"scores={args.out_dir / 'anchor_fit_scores.csv'}")
    print(f"report={args.out_dir / 'anchor_fit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
