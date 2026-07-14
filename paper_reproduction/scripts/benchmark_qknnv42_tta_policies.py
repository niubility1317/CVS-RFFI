"""Compare 1/3/5-view TTA with one fixed adapter and deployable qKNNV42 head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, run


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from export_spaceborne_features import (  # noqa: E402
    SATELLITE_TTA_POLICIES,
    _satellite_tta_view_count,
)


METRICS = ("old_acc_mean", "seen_new_acc_mean", "H_old_new_mean")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _feature_mapping(args: argparse.Namespace, policy: str, receivers: list[str]) -> dict[str, Any]:
    view_count = _satellite_tta_view_count(policy)
    subdir = str(args.feature_subdir_template).format(
        base=str(args.feature_subdir_base), policy=policy, view_count=view_count
    )
    mapping: dict[str, Any] = {}
    missing: list[str] = []
    for receiver in receivers:
        path = args.feature_root / f"FULL_RX_{receiver}" / subdir / str(args.feature_name)
        if not path.is_file():
            missing.append(str(path))
        mapping[receiver] = {scenario: str(path) for scenario in SCENARIOS}
    if missing:
        raise FileNotFoundError(f"missing {policy} feature caches: {missing}")
    return mapping


def _aggregate(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        deltas = [float(row[metric]) - float(baseline[str(row["run_key"])][metric]) for row in rows]
        result[metric] = _mean(values)
        result[f"{metric}_median"] = float(statistics.median(values))
        result[f"{metric}_delta_vs_5view_pp"] = 100.0 * _mean(deltas)
        result[f"{metric}_worst_paired_delta_vs_5view_pp"] = 100.0 * min(deltas)
        result[f"{metric}_paired_drop_gt_3pp_count"] = sum(delta < -0.03 for delta in deltas)
    for key in ("latency_per_query_ms", "estimated_head_macs", "persistent_state_bytes"):
        result[key] = _mean([float(row[key]) for row in rows])
    view_count = int(rows[0]["tta_view_count"])
    result.update(
        {
            "tta_view_count": view_count,
            "backbone_forwards_per_physical_sample": view_count,
            "fft_sketches_per_physical_sample": view_count,
            "front_end_compute_vs_5view": view_count / 5.0,
            "front_end_compute_reduction_vs_5view_pct": 100.0 * (1.0 - view_count / 5.0),
        }
    )
    result["performance_gate_pass"] = all(
        float(result[f"{metric}_delta_vs_5view_pp"]) >= -3.0 for metric in METRICS
    )
    return result


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    policies = [str(value).strip().lower() for value in args.policies]
    if len(set(policies)) != len(policies):
        raise ValueError("policies must not contain duplicates")
    unknown = [policy for policy in policies if policy not in SATELLITE_TTA_POLICIES]
    if unknown:
        raise ValueError(f"unknown TTA policies: {unknown}")
    if "rx_light5" not in policies:
        raise ValueError("policies must include rx_light5 as the paired baseline")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.out_root}")
    template = json.loads(args.template_config.read_text(encoding="utf-8"))
    receivers = [str(value) for value in template["publication_target_receiver_grid"]]
    expected = len(policies) * len(receivers) * len(args.seed_grid) * len(args.k_grid)
    if int(args.expected_runs) != expected:
        raise ValueError(f"expected_runs={args.expected_runs}, but policy matrix contains {expected} runs")
    feature_mappings = {
        policy: _feature_mapping(args, policy, receivers) for policy in policies
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for policy in policies:
        view_count = _satellite_tta_view_count(policy)
        for receiver in receivers:
            for seed in args.seed_grid:
                for k_shot in args.k_grid:
                    config = dict(template)
                    config.update(
                        {
                            "experiment_id": f"cvs_qknnv42_tta_{policy}_rx{receiver}_k{k_shot}_seed{seed}",
                            "feature_npz_by_receiver_scenario": feature_mappings[policy],
                            "target_receiver_labels": [receiver],
                            "seed": int(seed),
                            "split_seed": int(seed),
                            "k_shot": int(k_shot),
                            "qknnv42_expected_tta_view_count": int(view_count),
                            "qknnv42_decision_mode": "per_sample_argmax",
                            "qknnv42_labelprop_mode": "disabled",
                            "qknnv42_old_anchor_bias": float(args.old_anchor_bias),
                            "non_deployment_oracle_diagnostic": False,
                            "publication_protocol": "fixed_adapter_paired_tta_policy_deployment_ablation",
                        }
                    )
                    relative = Path(f"rx_{receiver}") / f"seed_{seed}" / f"k_{k_shot}" / "cvs_qknnv42"
                    result = run(config, args.out_root / policy / relative)
                    scenario_rows = list(result["metrics_by_scenario"].values())
                    run_key = "/".join(relative.parts[:-1])
                    split_payload = json.dumps(
                        result["split_manifest"]["splits_by_scenario"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    row = {
                        "run_key": run_key,
                        "policy": policy,
                        "tta_view_count": view_count,
                        "receiver": receiver,
                        "seed": int(seed),
                        "k_shot": int(k_shot),
                        "split_manifest_sha256": hashlib.sha256(split_payload).hexdigest(),
                        **{metric: float(result["metrics"][metric]) for metric in METRICS},
                        "latency_per_query_ms": _mean(
                            [float(item["latency_per_query_ms"]) for item in scenario_rows]
                        ),
                        "estimated_head_macs": _mean(
                            [float(item["estimated_head_macs"]) for item in scenario_rows]
                        ),
                        "persistent_state_bytes": _mean(
                            [float(item["persistent_state_bytes"]) for item in scenario_rows]
                        ),
                    }
                    rows.append(row)
                    if policy == "rx_light5":
                        baseline[run_key] = row
    expected_baseline = len(receivers) * len(args.seed_grid) * len(args.k_grid)
    if len(baseline) != expected_baseline:
        raise ValueError(f"5-view baseline has {len(baseline)} rows; expected {expected_baseline}")
    split_mismatches = [
        str(row["run_key"])
        for row in rows
        if str(row["split_manifest_sha256"])
        != str(baseline[str(row["run_key"])]["split_manifest_sha256"])
    ]
    if split_mismatches:
        raise ValueError(f"TTA policies do not share identical support/query splits: {split_mismatches[:5]}")
    summaries = {
        policy: _aggregate([row for row in rows if row["policy"] == policy], baseline)
        for policy in policies
    }
    passing = sorted(
        (value for value in summaries.values() if value["performance_gate_pass"]),
        key=lambda value: int(value["tta_view_count"]),
    )
    summary = {
        "schema": "cvs_qknnv42_fixed_adapter_tta_ablation_v1",
        "baseline_policy": "rx_light5",
        "head": "fft96+labelprop_disabled+per_sample_argmax",
        "old_anchor_bias": float(args.old_anchor_bias),
        "performance_gate": "matrix-mean old_acc, seen_new_acc, H_old_new drop vs 5-view <= 3 pp",
        "policies": summaries,
        "lightest_passing_view_count": int(passing[0]["tta_view_count"]) if passing else None,
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
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--policies", nargs="+", default=["none", "rx_shift3", "rx_cfo3", "rx_light5"]
    )
    parser.add_argument("--feature-subdir-base", default="ADV3B02_ADAPTER60_FFT96")
    parser.add_argument("--feature-subdir-template", default="{base}_{policy}")
    parser.add_argument("--feature-name", default="features_adapter60_fft96.npz")
    parser.add_argument("--seed-grid", nargs="+", type=int, default=[713101, 713102, 713103, 713104, 713105])
    parser.add_argument("--k-grid", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--old-anchor-bias", type=float, default=-0.001)
    parser.add_argument("--expected-runs", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(benchmark(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
