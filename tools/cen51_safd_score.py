#!/usr/bin/env python3
"""Score CEN51 few-shot runs with a diagnostic SAFD metric.

This tool consumes the CSV emitted by ``cen51_fewshot_stability_validator.py``
and turns the hard promotion gates into a continuous diagnostic score. The
score is meant for controller design: it says which deficit should be repaired
next, not just whether a run passed a fixed threshold.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


THRESHOLDS: dict[int, dict[str, float]] = {
    5: {
        "val_drop": 1.50,
        "latest_strict_drop": 2.50,
        "val_late_std": 1.00,
        "strict_late_span": 2.50,
        "swad_gap": 1.70,
        "ema_gap": 3.00,
        "rx_floor": 55.00,
        "sat_floor": 28.00,
        "target_val": 90.00,
        "target_strict": 74.60,
    },
    10: {
        "val_drop": 1.00,
        "latest_strict_drop": 2.00,
        "val_late_std": 0.80,
        "strict_late_span": 2.00,
        "swad_gap": 1.50,
        "ema_gap": 2.50,
        "rx_floor": 60.00,
        "sat_floor": 30.00,
        "target_val": 93.00,
        "target_strict": 76.50,
    },
    20: {
        "val_drop": 1.00,
        "latest_strict_drop": 2.00,
        "val_late_std": 0.70,
        "strict_late_span": 1.80,
        "swad_gap": 1.20,
        "ema_gap": 2.00,
        "rx_floor": 62.00,
        "sat_floor": 31.00,
        "target_val": 94.00,
        "target_strict": 77.00,
    },
    30: {
        "val_drop": 0.80,
        "latest_strict_drop": 1.80,
        "val_late_std": 0.55,
        "strict_late_span": 1.50,
        "swad_gap": 1.20,
        "ema_gap": 2.00,
        "rx_floor": 65.00,
        "sat_floor": 33.00,
        "target_val": 96.50,
        "target_strict": 79.00,
    },
    50: {
        "val_drop": 0.80,
        "latest_strict_drop": 1.50,
        "val_late_std": 0.45,
        "strict_late_span": 1.30,
        "swad_gap": 1.00,
        "ema_gap": 1.80,
        "rx_floor": 70.00,
        "sat_floor": 34.00,
        "target_val": 97.50,
        "target_strict": 83.00,
    },
}

FAMILY_STRICT_STD_MAX = {5: 2.50, 10: 0.80, 20: 1.50, 30: 1.50, 50: 1.20}

QUALITY_BANDS = {
    "val": 3.0,
    "strict": 5.0,
    "rx": 8.0,
    "sat": 6.0,
}

SCORE_WEIGHTS = {
    "val": 0.15,
    "strict": 0.30,
    "rx": 0.20,
    "sat": 0.15,
    "late": 0.10,
    "seed": 0.10,
}

LATE_WEIGHTS = {
    "val_drop": 0.30,
    "latest_strict_drop": 0.25,
    "late_val_std": 0.15,
    "late_strict_span": 0.15,
    "swad_gap": 0.10,
    "ema_gap": 0.05,
}

LATE_THRESHOLD_KEYS = {
    "val_drop": "val_drop",
    "latest_strict_drop": "latest_strict_drop",
    "late_val_std": "val_late_std",
    "late_strict_span": "strict_late_span",
    "swad_gap": "swad_gap",
    "ema_gap": "ema_gap",
}

ACTION_TEXT = {
    "strict": "clean_strict_repair: keep full-DG satellite, but tune identity separation/feature decoupling/prototype-SupCon pressure; if val is also weak, first relax early regularization.",
    "rx": "receiver_floor_repair: raise receiver-floor pressure through GroupCE/DRO cap, modest dom/adv increase, and worst-RX-aware checkpointing; do not blindly increase all DG losses.",
    "sat": "satellite_floor_repair: increase or advance light full-DG satellite exposure and scenario coverage; keep CE-only disabled and monitor clean strict regression.",
    "val": "validation_capacity_repair: reduce early bottleneck, delay satellite/augmentation, relax norm/Fishr/proto pressure, and preserve identity-first warmup.",
    "late": "late_stability_repair: tighten checkpoint/SWAD selection, reduce late perturbation, and avoid high-variance schedules after the best strict epoch.",
    "seed": "seed_stability_repair: run confirmation seeds before changing mechanism; only tune if the same deficit repeats across seeds.",
}


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_int(value: str | None) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def clip(value: float, low: float = 0.0, high: float = 1.05) -> float:
    return max(low, min(high, value))


def quality(value: float | None, target: float, band: float) -> float:
    if value is None:
        return 0.0
    return clip(1.0 + (value - target) / band)


def late_quality(row: dict[str, str], shot: int) -> float:
    thresholds = THRESHOLDS[shot]
    penalty = 0.0
    for field, weight in LATE_WEIGHTS.items():
        value = safe_float(row.get(field))
        limit = thresholds[LATE_THRESHOLD_KEYS[field]]
        if value is None or limit <= 0:
            continue
        penalty += weight * max(0.0, value / limit)
    return 1.0 / (1.0 + penalty)


def seed_quality(rows: list[dict[str, str]], shot: int) -> float:
    if len(rows) < 2:
        return 0.60

    strict_values = values(rows, "final_strict")
    rx_values = values(rows, "rx_floor")
    sat_values = values(rows, "sat_floor")
    strict_std = pstdev(strict_values) if len(strict_values) >= 2 else 0.0
    rx_std = pstdev(rx_values) if len(rx_values) >= 2 else 0.0
    sat_std = pstdev(sat_values) if len(sat_values) >= 2 else 0.0

    strict_limit = FAMILY_STRICT_STD_MAX.get(shot, 1.50)
    penalty = (
        0.50 * strict_std / strict_limit
        + 0.25 * rx_std / 6.0
        + 0.25 * sat_std / 3.0
    )
    return clip(1.0 - penalty, high=1.0)


def values(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            out.append(value)
    return out


def run_qualities(row: dict[str, str]) -> dict[str, float]:
    shot = safe_int(row.get("shot"))
    if shot not in THRESHOLDS:
        raise ValueError(f"unsupported shot value: {row.get('shot')!r}")
    thresholds = THRESHOLDS[shot]
    return {
        "val": quality(safe_float(row.get("final_val")), thresholds["target_val"], QUALITY_BANDS["val"]),
        "strict": quality(safe_float(row.get("final_strict")), thresholds["target_strict"], QUALITY_BANDS["strict"]),
        "rx": quality(safe_float(row.get("rx_floor")), thresholds["rx_floor"], QUALITY_BANDS["rx"]),
        "sat": quality(safe_float(row.get("sat_floor")), thresholds["sat_floor"], QUALITY_BANDS["sat"]),
        "late": late_quality(row, shot),
    }


def deficit_vector(qualities: dict[str, float], q_seed: float | None = None) -> dict[str, float]:
    deficits = {
        key: SCORE_WEIGHTS[key] * max(0.0, 1.0 - qualities[key])
        for key in ("val", "strict", "rx", "sat", "late")
    }
    if q_seed is not None:
        deficits["seed"] = SCORE_WEIGHTS["seed"] * max(0.0, 1.0 - q_seed)
    return deficits


def top_deficits(deficits: dict[str, float], n: int = 2) -> list[str]:
    return [
        key
        for key, value in sorted(deficits.items(), key=lambda item: (-item[1], item[0]))[:n]
        if value > 0.001
    ]


def score_family(rows: list[dict[str, str]]) -> dict[str, object]:
    shot = safe_int(rows[0].get("shot"))
    assert shot is not None
    thresholds = THRESHOLDS[shot]

    q_rows = [run_qualities(row) for row in rows]
    q_val = mean(item["val"] for item in q_rows)
    q_strict = mean(item["strict"] for item in q_rows)
    q_rx = quality(min(values(rows, "rx_floor") or [None]), thresholds["rx_floor"], QUALITY_BANDS["rx"])
    q_sat = quality(min(values(rows, "sat_floor") or [None]), thresholds["sat_floor"], QUALITY_BANDS["sat"])
    q_late = min(item["late"] for item in q_rows)
    q_seed = seed_quality(rows, shot)

    fail_count = sum(1 for row in rows if row.get("stability_status") != "PASS")
    score = 100.0 * (
        SCORE_WEIGHTS["val"] * q_val
        + SCORE_WEIGHTS["strict"] * q_strict
        + SCORE_WEIGHTS["rx"] * q_rx
        + SCORE_WEIGHTS["sat"] * q_sat
        + SCORE_WEIGHTS["late"] * q_late
        + SCORE_WEIGHTS["seed"] * q_seed
    )
    score -= 10.0 * fail_count / max(1, len(rows))

    deficits = deficit_vector(
        {
            "val": q_val,
            "strict": q_strict,
            "rx": q_rx,
            "sat": q_sat,
            "late": q_late,
        },
        q_seed,
    )
    actions = top_deficits(deficits, n=3)

    return {
        "family": rows[0].get("family") or "",
        "shot": shot,
        "n_runs": len(rows),
        "stable_runs": len(rows) - fail_count,
        "safd_score": round(score, 3),
        "q_val": round(q_val, 4),
        "q_strict": round(q_strict, 4),
        "q_rx_floor": round(q_rx, 4),
        "q_sat_floor": round(q_sat, 4),
        "q_late": round(q_late, 4),
        "q_seed": round(q_seed, 4),
        "deficit_val": round(deficits["val"], 5),
        "deficit_strict": round(deficits["strict"], 5),
        "deficit_rx": round(deficits["rx"], 5),
        "deficit_sat": round(deficits["sat"], 5),
        "deficit_late": round(deficits["late"], 5),
        "deficit_seed": round(deficits["seed"], 5),
        "dominant_deficits": ";".join(actions),
        "recommended_actions": " | ".join(ACTION_TEXT[action] for action in actions),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, family_rows: list[dict[str, object]], run_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# CEN51-SAFD Diagnostic Score Report")
    lines.append("")
    lines.append("This report ranks few-shot candidates with a continuous diagnostic score. It is not a replacement for the hard stability validator; it explains which training actuator should be changed next.")
    lines.append("")
    lines.append("## Family Ranking")
    lines.append("")
    lines.append("| rank | shot | family | SAFD | stable/runs | dominant deficits |")
    lines.append("|---:|---:|---|---:|---:|---|")
    for index, row in enumerate(family_rows, 1):
        lines.append(
            f"| {index} | {row['shot']} | `{row['family']}` | {row['safd_score']:.3f} | "
            f"{row['stable_runs']}/{row['n_runs']} | {row['dominant_deficits']} |"
        )

    lines.append("")
    lines.append("## Action Dictionary")
    lines.append("")
    for key, text in ACTION_TEXT.items():
        lines.append(f"- `{key}`: {text}")

    lines.append("")
    lines.append("## Highest Run-Level Deficits")
    lines.append("")
    lines.append("| run | shot | dominant deficits | val | strict | rx | sat | late |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in run_rows[:12]:
        lines.append(
            f"| `{row['run']}` | {row['shot']} | {row['dominant_deficits']} | "
            f"{row['q_val']:.3f} | {row['q_strict']:.3f} | {row['q_rx']:.3f} | "
            f"{row['q_sat']:.3f} | {row['q_late']:.3f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if safe_int(row.get("shot")) in THRESHOLDS]

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("family") or "", safe_int(row.get("shot")) or -1)].append(row)

    family_rows = [score_family(items) for _, items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0]))]
    family_rows.sort(key=lambda row: (-float(row["safd_score"]), int(row["shot"]), str(row["family"])))

    run_rows: list[dict[str, object]] = []
    for row in rows:
        qualities = run_qualities(row)
        deficits = deficit_vector(qualities)
        actions = top_deficits(deficits, n=3)
        run_rows.append(
            {
                "run": row.get("run") or "",
                "family": row.get("family") or "",
                "shot": safe_int(row.get("shot")) or "",
                "stability_status": row.get("stability_status") or "",
                "promotion_status": row.get("promotion_status") or "",
                "q_val": round(qualities["val"], 4),
                "q_strict": round(qualities["strict"], 4),
                "q_rx": round(qualities["rx"], 4),
                "q_sat": round(qualities["sat"], 4),
                "q_late": round(qualities["late"], 4),
                "deficit_val": round(deficits["val"], 5),
                "deficit_strict": round(deficits["strict"], 5),
                "deficit_rx": round(deficits["rx"], 5),
                "deficit_sat": round(deficits["sat"], 5),
                "deficit_late": round(deficits["late"], 5),
                "dominant_deficits": ";".join(actions),
                "recommended_actions": " | ".join(ACTION_TEXT[action] for action in actions),
            }
        )
    run_rows.sort(
        key=lambda row: (
            -sum(float(row[key]) for key in ("deficit_val", "deficit_strict", "deficit_rx", "deficit_sat", "deficit_late")),
            int(row["shot"]),
            str(row["run"]),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "safd_family_scores.csv", family_rows)
    write_csv(args.out_dir / "safd_run_deficits.csv", run_rows)
    write_markdown(args.out_dir / "safd_score_report.md", family_rows, run_rows)

    print(f"loaded_runs={len(rows)} families={len(family_rows)}")
    print(f"family_scores={args.out_dir / 'safd_family_scores.csv'}")
    print(f"run_deficits={args.out_dir / 'safd_run_deficits.csv'}")
    print(f"report={args.out_dir / 'safd_score_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
