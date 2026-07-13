from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean_ci(values: Iterable[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(samples), array.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_sign_flip_pvalue(deltas: Iterable[float]) -> float:
    array = np.asarray(list(deltas), dtype=np.float64)
    if array.size == 0:
        raise ValueError("paired test requires at least one delta")
    observed = abs(float(array.mean()))
    null_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=int(array.size)):
        null_means.append(abs(float((array * np.asarray(signs)).mean())))
    extreme = sum(value >= observed - 1e-15 for value in null_means)
    return float(extreme / len(null_means))


def paired_effect_size(deltas: Iterable[float]) -> float:
    array = np.asarray(list(deltas), dtype=np.float64)
    if array.size < 2:
        return float("nan")
    std = float(array.std(ddof=1))
    if std == 0.0:
        return math.copysign(float("inf"), float(array.mean())) if float(array.mean()) else 0.0
    return float(array.mean() / std)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        current = min(1.0, float(value) * float(total - rank))
        running = max(running, current)
        adjusted[name] = running
    return adjusted


def analyze_rows(
    rows: list[dict[str, object]],
    *,
    reference_method: str,
    bootstrap_seed: int = 20260713,
    bootstrap_samples: int = 10000,
) -> dict[str, object]:
    required = {"method", "seed", "metric", "value"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")

    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        metric = str(row["metric"])
        seed = str(row["seed"])
        key = (method, metric)
        if seed in grouped[key]:
            raise ValueError(f"duplicate method/metric/seed row: {method}/{metric}/{seed}")
        grouped[key][seed] = float(row["value"])

    summary = []
    for (method, metric), seed_values in sorted(grouped.items()):
        values = np.asarray(list(seed_values.values()), dtype=np.float64)
        low, high = bootstrap_mean_ci(
            values,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        summary.append(
            {
                "method": method,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "seeds": sorted(seed_values),
            }
        )

    comparisons = []
    raw_pvalues: dict[str, float] = {}
    for metric in sorted({metric for _, metric in grouped}):
        reference = grouped.get((reference_method, metric))
        if not reference:
            continue
        for (method, current_metric), seed_values in sorted(grouped.items()):
            if current_metric != metric or method == reference_method:
                continue
            common_seeds = sorted(set(reference) & set(seed_values))
            if not common_seeds:
                continue
            deltas = np.asarray(
                [reference[seed] - seed_values[seed] for seed in common_seeds],
                dtype=np.float64,
            )
            low, high = bootstrap_mean_ci(
                deltas,
                seed=bootstrap_seed + len(comparisons) + 1,
                samples=bootstrap_samples,
            )
            comparison_id = f"{metric}:{reference_method}-vs-{method}"
            pvalue = paired_sign_flip_pvalue(deltas)
            raw_pvalues[comparison_id] = pvalue
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "reference_method": reference_method,
                    "comparison_method": method,
                    "paired_n": len(common_seeds),
                    "paired_seeds": common_seeds,
                    "mean_delta_reference_minus_comparison": float(deltas.mean()),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "paired_effect_size_dz": paired_effect_size(deltas),
                    "sign_flip_p_raw": pvalue,
                }
            )
    adjusted = holm_adjust(raw_pvalues)
    for row in comparisons:
        row["holm_p_adjusted"] = adjusted[row["comparison_id"]]

    return {
        "schema": "cvs_publication_comparison_stats_v1",
        "reference_method": reference_method,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": int(bootstrap_samples),
        "summary": summary,
        "paired_comparisons": comparisons,
        "claim_boundary": "statistics_only_requires_protocol_valid_completed_runs",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate paired multi-seed CVS publication comparisons.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    result = analyze_rows(
        _read_rows(args.input),
        reference_method=args.reference_method,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary_rows": len(result["summary"]), "comparisons": len(result["paired_comparisons"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
