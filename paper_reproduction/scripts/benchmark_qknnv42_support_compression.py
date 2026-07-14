"""Benchmark qKNNV42 full, diverse-exemplar, medoid, and prototype-only states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, run
from paper_reproduction.scripts.benchmark_qknnv42_tta_policies import (
    HISTORICAL_METRIC_DEFAULTS_PP,
    _load_historical_reference,
    _validate_historical_reference_metrics,
)


MODE_CONFIG = {
    "dense_all_support": {
        "labelprop": "dense_transductive",
        "support": "all_support",
        "old_bias": 0.001,
    },
    "light_all_support": {
        "labelprop": "disabled",
        "support": "all_support",
        "old_bias": -0.001,
    },
    "light_class_medoid": {
        "labelprop": "disabled",
        "support": "class_medoid",
        "old_bias": -0.001,
    },
    "light_class_diverse2": {
        "labelprop": "disabled",
        "support": "class_diverse2",
        "old_bias": -0.001,
    },
    "light_class_diverse4": {
        "labelprop": "disabled",
        "support": "class_diverse4",
        "old_bias": -0.001,
    },
    "light_prototype_only": {
        "labelprop": "disabled",
        "support": "prototype_only",
        "old_bias": -0.001,
    },
    "oracle_stream_all_support": {
        "labelprop": "support_prototype",
        "support": "all_support",
        "old_bias": 0.001,
    },
    "oracle_disabled_all_support": {
        "labelprop": "disabled",
        "support": "all_support",
        "old_bias": 0.001,
    },
    "oracle_stream_class_medoid": {
        "labelprop": "support_prototype",
        "support": "class_medoid",
        "old_bias": 0.001,
    },
    "oracle_stream_class_diverse2": {
        "labelprop": "support_prototype",
        "support": "class_diverse2",
        "old_bias": 0.001,
    },
    "oracle_stream_class_diverse4": {
        "labelprop": "support_prototype",
        "support": "class_diverse4",
        "old_bias": 0.001,
    },
    "oracle_stream_prototype_only": {
        "labelprop": "support_prototype",
        "support": "prototype_only",
        "old_bias": 0.001,
    },
}
DEPLOYABLE_MODES = (
    "dense_all_support",
    "light_all_support",
    "light_class_medoid",
    "light_class_diverse2",
    "light_class_diverse4",
    "light_prototype_only",
)
FULL_HISTORY_MODES = (
    "dense_all_support",
    "oracle_stream_all_support",
    "oracle_disabled_all_support",
    "oracle_stream_class_diverse4",
    "oracle_stream_class_diverse2",
    "oracle_stream_class_medoid",
    "oracle_stream_prototype_only",
)
METRICS = ("old_acc_mean", "seen_new_acc_mean", "H_old_new_mean")
RESOURCE_KEYS = (
    "latency_per_query_ms",
    "onboard_scoring_latency_per_query_ms",
    "enrollment_latency_sec",
    "estimated_head_macs",
    "estimated_support_score_macs",
    "estimated_prototype_score_macs",
    "estimated_labelprop_macs",
    "persistent_state_bytes",
    "support_code_bytes",
    "stored_quantized_support_code_count_total",
    "dense_graph_bytes_lower_bound",
    "decision_workspace_bytes_lower_bound",
    "estimated_decision_cubic_work_units",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _aggregate(
    rows: list[dict[str, Any]], *, baseline: dict[str, dict[str, Any]],
    historical: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        deltas = [float(row[metric]) - float(baseline[str(row["run_key"])][metric]) for row in rows]
        result[metric] = _mean(values)
        result[f"{metric}_median"] = float(statistics.median(values))
        result[f"{metric}_delta_vs_dense_pp"] = 100.0 * _mean(deltas)
        result[f"{metric}_worst_paired_delta_vs_dense_pp"] = 100.0 * min(deltas)
        result[f"{metric}_paired_drop_gt_3pp_count"] = sum(delta < -0.03 for delta in deltas)
        if historical is not None:
            historical_deltas = [
                float(row[metric]) - float(historical[str(row["run_key"])][metric])
                for row in rows
            ]
            result[f"{metric}_delta_vs_historical_pp"] = 100.0 * _mean(historical_deltas)
            result[f"{metric}_worst_paired_delta_vs_historical_pp"] = 100.0 * min(
                historical_deltas
            )
    for key in RESOURCE_KEYS:
        result[key] = _mean([float(row[key]) for row in rows])
    dense_macs = _mean([float(baseline[str(row["run_key"])]["estimated_head_macs"]) for row in rows])
    dense_state = _mean([float(baseline[str(row["run_key"])]["persistent_state_bytes"]) for row in rows])
    result["head_macs_reduction_vs_dense_pct"] = 100.0 * (
        1.0 - float(result["estimated_head_macs"]) / dense_macs
    )
    result["persistent_state_reduction_vs_dense_pct"] = 100.0 * (
        1.0 - float(result["persistent_state_bytes"]) / dense_state
    )
    result["performance_gate_pass"] = all(
        float(result[f"{metric}_delta_vs_dense_pp"]) >= -3.0 - 1e-9 for metric in METRICS
    )
    if historical is not None:
        result["historical_performance_gate_pass"] = all(
            float(result[f"{metric}_delta_vs_historical_pp"]) >= -3.0 - 1e-9
            for metric in METRICS
        )
    return result


def _feature_paths(args: argparse.Namespace, receiver: str) -> dict[str, str]:
    if args.feature_cache_root is not None:
        path = (
            args.feature_cache_root
            / f"FULL_RX_{receiver}"
            / str(args.feature_subdir)
            / str(args.feature_name)
        )
        mapping = {scenario: str(path) for scenario in SCENARIOS}
    else:
        mapping = {
            scenario: str(args.feature_cache / f"{scenario}.npz")
            for scenario in SCENARIOS
        }
    missing = [path for path in mapping.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing feature caches: {sorted(set(missing))}")
    return mapping


def _validate_mode_counts(
    rows: list[dict[str, Any]], modes: list[str], expected_per_mode: int
) -> dict[str, int]:
    counts = {
        mode: sum(str(row.get("mode")) == mode for row in rows)
        for mode in modes
    }
    invalid = {mode: count for mode, count in counts.items() if count != expected_per_mode}
    if invalid:
        raise ValueError(
            f"support compression matrix is incomplete: {invalid}; "
            f"expected_per_mode={expected_per_mode}"
        )
    return counts


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    profile = str(args.profile)
    allowed_modes = FULL_HISTORY_MODES if profile == "full_legacy_oracle" else DEPLOYABLE_MODES
    modes = list(allowed_modes) if args.modes is None else [str(value) for value in args.modes]
    unknown_modes = [mode for mode in modes if mode not in allowed_modes]
    if unknown_modes:
        raise ValueError(f"modes are incompatible with profile={profile}: {unknown_modes}")
    if "dense_all_support" not in modes:
        raise ValueError("modes must include dense_all_support as the original baseline")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.out_root}")
    config_paths = sorted(
        args.baseline_root.glob("rx_*/seed_*/k_*/cvs_qknnv42/resolved_config.json")
    )
    templates: dict[tuple[str, int], Path] = {}
    for path in config_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        templates.setdefault(
            (str(config["target_receiver_labels"][0]), int(config["k_shot"])), path
        )
    specs = [(path, int(seed)) for path in templates.values() for seed in args.seed_grid]
    if len(specs) != int(args.expected_runs_per_mode):
        raise ValueError(
            f"expected {args.expected_runs_per_mode} configs per mode, found {len(specs)}"
        )
    historical = None
    historical_metrics_pp = None
    if profile == "full_legacy_oracle":
        historical = _load_historical_reference(args.baseline_root)
        if len(historical) != int(args.expected_runs_per_mode):
            raise ValueError(
                f"historical reference has {len(historical)} rows; "
                f"expected {args.expected_runs_per_mode}"
            )
        historical_metrics_pp = _validate_historical_reference_metrics(
            historical, HISTORICAL_METRIC_DEFAULTS_PP
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_config = dict(MODE_CONFIG[mode])
        if mode == "light_class_medoid":
            mode_config["old_bias"] = float(args.medoid_old_anchor_bias)
        elif mode == "light_prototype_only":
            mode_config["old_bias"] = float(args.prototype_old_anchor_bias)
        for config_path, seed in specs:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            receiver = str(config["target_receiver_labels"][0])
            config.update(
                {
                    "seed": int(seed),
                    "split_seed": int(seed),
                    "feature_npz_by_scenario": _feature_paths(args, receiver),
                    "feature_npz_by_receiver_scenario": {},
                    "qknnv42_decision_mode": (
                        "legacy_role_quota_oracle"
                        if profile == "full_legacy_oracle"
                        else "per_sample_argmax"
                    ),
                    "qknnv42_labelprop_mode": str(mode_config["labelprop"]),
                    "qknnv42_support_representation": str(mode_config["support"]),
                    "qknnv42_old_anchor_bias": float(mode_config["old_bias"]),
                }
            )
            k_shot = int(config["k_shot"])
            config["experiment_id"] = (
                f"qknnv42_support_{mode}_rx{receiver}_k{k_shot}_seed{seed}"
            )
            relative = Path(f"rx_{receiver}") / f"seed_{seed}" / f"k_{k_shot}" / "cvs_qknnv42"
            result = run(config, args.out_root / mode / relative)
            manifest = result["split_manifest"]
            expected_decision = (
                "legacy_role_quota_oracle"
                if profile == "full_legacy_oracle"
                else "per_sample_argmax"
            )
            if manifest["qknnv42_decision_mode"] != expected_decision:
                raise AssertionError("support compression benchmark lost its decision profile")
            if profile == "full_legacy_oracle" and not manifest["non_deployment_oracle_diagnostic"]:
                raise AssertionError("full-history compression must remain non-deployment Oracle")
            split_payload = json.dumps(
                manifest["splits_by_scenario"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            scenario_rows = list(result["metrics_by_scenario"].values())
            run_key = "/".join(relative.parts[:-1])
            row = {
                "run_key": run_key,
                "mode": mode,
                "receiver": receiver,
                "seed": int(seed),
                "k_shot": k_shot,
                "split_manifest_sha256": hashlib.sha256(split_payload).hexdigest(),
                **{metric: float(result["metrics"][metric]) for metric in METRICS},
                **{
                    key: _mean([float(item[key]) for item in scenario_rows])
                    for key in RESOURCE_KEYS
                },
            }
            rows.append(row)
            if mode == "dense_all_support":
                baseline[run_key] = row
    mode_counts = _validate_mode_counts(rows, modes, int(args.expected_runs_per_mode))
    if len(baseline) != int(args.expected_runs_per_mode):
        raise ValueError(
            f"dense baseline has {len(baseline)} rows; expected {args.expected_runs_per_mode}"
        )
    split_mismatches = [
        str(row["run_key"])
        for row in rows
        if str(row["split_manifest_sha256"])
        != str(baseline[str(row["run_key"])]["split_manifest_sha256"])
    ]
    if split_mismatches:
        raise ValueError(f"support modes use different splits: {split_mismatches[:5]}")
    if historical is not None:
        historical_split_mismatches = [
            str(row["run_key"])
            for row in rows
            if str(row["split_manifest_sha256"])
            != str(historical[str(row["run_key"])]["split_manifest_sha256"])
        ]
        if historical_split_mismatches:
            raise ValueError(
                "head candidates and historical runs use different splits: "
                f"{historical_split_mismatches[:5]}"
            )
    summaries = {
        mode: _aggregate(
            [row for row in rows if row["mode"] == mode],
            baseline=baseline,
            historical=historical,
        )
        for mode in modes
    }
    passing = [
        mode for mode in modes
        if summaries[mode]["performance_gate_pass"]
        and summaries[mode].get("historical_performance_gate_pass", True)
    ]
    summary = {
        "schema": "cvs_qknnv42_support_compression_benchmark_v2",
        "profile": profile,
        "baseline_mode": "dense_all_support",
        "performance_gate": "matrix-mean old_acc, seen_new_acc, H_old_new drop vs original <= 3 pp",
        "historical_reference_metrics_pp": historical_metrics_pp,
        "mode_run_counts": mode_counts,
        "seed_grid": [int(value) for value in args.seed_grid],
        "medoid_old_anchor_bias": float(args.medoid_old_anchor_bias),
        "prototype_old_anchor_bias": float(args.prototype_old_anchor_bias),
        "modes": summaries,
        "passing_modes": passing,
    }
    with (args.out_root / "paired_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--feature-cache", type=Path)
    feature_group.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--feature-subdir", default="ADV3B02_FULL_ADAPTER5_FFT96")
    parser.add_argument("--feature-name", default="features_full_adapter5_fft96.npz")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("deployable_light", "full_legacy_oracle"),
        default="deployable_light",
    )
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_CONFIG), default=None)
    parser.add_argument("--seed-grid", nargs="+", type=int, default=[713101, 713102, 713103, 713104, 713105])
    parser.add_argument("--expected-runs-per-mode", type=int, default=125)
    parser.add_argument("--medoid-old-anchor-bias", type=float, default=-0.001)
    parser.add_argument("--prototype-old-anchor-bias", type=float, default=-0.001)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(benchmark(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
