"""Benchmark dense and streaming qKNNV42 heads on identical cached Stage2-C splits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, run


MODES = ("dense_transductive", "support_prototype", "disabled")
METRICS = ("old_acc_mean", "seen_new_acc_mean", "H_old_new_mean")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _aggregate_mode(rows: list[dict[str, Any]], *, baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        deltas = [float(row[metric]) - float(baseline[str(row["run_key"])][metric]) for row in rows]
        result[metric] = _mean(values)
        result[f"{metric}_median"] = float(statistics.median(values))
        result[f"{metric}_delta_pp"] = 100.0 * _mean(deltas)
        result[f"{metric}_worst_paired_delta_pp"] = 100.0 * min(deltas)
        result[f"{metric}_paired_drop_gt_3pp_count"] = sum(delta < -0.03 for delta in deltas)
    for key in (
        "latency_per_query_ms",
        "adaptation_latency_sec",
        "estimated_head_macs",
        "dense_graph_bytes_lower_bound",
    ):
        result[key] = _mean([float(row[key]) for row in rows])
    return result


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config_paths = sorted(args.baseline_root.glob("rx_*/seed_*/k_*/cvs_qknnv42/resolved_config.json"))
    specs: list[tuple[Path, int | None]]
    if args.seed_grid:
        templates: dict[tuple[str, int], Path] = {}
        for path in config_paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            key = (str(config["target_receiver_labels"][0]), int(config["k_shot"]))
            templates.setdefault(key, path)
        specs = [(path, int(seed)) for path in templates.values() for seed in args.seed_grid]
    else:
        specs = [(path, None) for path in config_paths]
    if len(specs) != int(args.expected_runs):
        raise ValueError(f"expected {args.expected_runs} benchmark configs, found {len(specs)}")
    feature_paths = {scenario: str(args.feature_cache / f"{scenario}.npz") for scenario in SCENARIOS}
    missing = [path for path in feature_paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing feature caches: {missing}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        for config_path, seed_override in specs:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if seed_override is not None:
                config["seed"] = int(seed_override)
                config["split_seed"] = int(seed_override)
            config["feature_npz_by_scenario"] = feature_paths
            config["qknnv42_labelprop_mode"] = mode
            config["qknnv42_old_anchor_bias"] = (
                0.001 if mode == "dense_transductive" else float(args.light_old_anchor_bias)
            )
            receiver = str(config["target_receiver_labels"][0])
            seed = int(config["seed"])
            k_shot = int(config["k_shot"])
            config["experiment_id"] = f"cvs_qknnv42_stage2c_rx{receiver}_k{k_shot}_seed{seed}_{mode}"
            relative_run = Path(f"rx_{receiver}") / f"seed_{seed}" / f"k_{k_shot}" / "cvs_qknnv42"
            result = run(config, args.out_root / mode / relative_run)
            metrics = result["metrics"]
            scenario_rows = list(result["metrics_by_scenario"].values())
            run_key = "/".join(relative_run.parts[:-1])
            row = {
                "run_key": run_key,
                "receiver": result["target_receiver_label"],
                "seed": int(result["seed"]),
                "k_shot": int(config["k_shot"]),
                "mode": mode,
                **{metric: float(metrics[metric]) for metric in METRICS},
                "latency_per_query_ms": _mean(
                    [float(item["latency_per_query_ms"]) for item in scenario_rows]
                ),
                "adaptation_latency_sec": _mean(
                    [float(item["adaptation_latency_sec"]) for item in scenario_rows]
                ),
                "estimated_head_macs": _mean(
                    [float(item["estimated_head_macs"]) for item in scenario_rows]
                ),
                "dense_graph_bytes_lower_bound": _mean(
                    [float(item["dense_graph_bytes_lower_bound"]) for item in scenario_rows]
                ),
            }
            all_rows.append(row)
            if mode == "dense_transductive":
                baseline[run_key] = row

    if len(baseline) != int(args.expected_runs):
        raise ValueError(f"dense baseline has {len(baseline)} unique rows, expected {args.expected_runs}")
    summary = {
        "schema": "cvs_qknnv42_labelprop_compression_benchmark_v1",
        "baseline_mode": "dense_transductive",
        "performance_gate": "aggregate old_acc, seen_new_acc, and H_old_new drop <= 3 pp",
        "seed_grid": [int(value) for value in args.seed_grid] if args.seed_grid else [],
        "light_old_anchor_bias": float(args.light_old_anchor_bias),
        "modes": {
            mode: _aggregate_mode(
                [row for row in all_rows if row["mode"] == mode], baseline=baseline
            )
            for mode in args.modes
        },
    }
    for mode, mode_summary in summary["modes"].items():
        mode_summary["performance_gate_pass"] = all(
            float(mode_summary[f"{metric}_delta_pp"]) >= -3.0 for metric in METRICS
        )

    with (args.out_root / "paired_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--expected-runs", type=int, default=125)
    parser.add_argument("--seed-grid", nargs="+", type=int, default=None)
    parser.add_argument("--light-old-anchor-bias", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    summary = benchmark(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
