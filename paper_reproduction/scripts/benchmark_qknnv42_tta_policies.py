"""Compare frozen-ADV3B02 1/3/5-view features with qKNN-side adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
HISTORICAL_METRIC_DEFAULTS_PP = {
    "old_acc_mean": 84.07,
    "seen_new_acc_mean": 93.24,
    "H_old_new_mean": 88.23,
}


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


def _load_historical_reference(root: Path) -> dict[str, dict[str, Any]]:
    reference: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        receiver = str(payload["target_receiver_label"])
        seed = int(payload["seed"])
        k_shot = int(path.parents[1].name.removeprefix("k_"))
        run_key = f"rx_{receiver}/seed_{seed}/k_{k_shot}"
        if run_key in reference:
            raise ValueError(f"duplicate historical run key: {run_key}")
        split_path = path.with_name("split_manifest.json")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split_payload = json.dumps(
            split["splits_by_scenario"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        reference[run_key] = {
            **{metric: float(payload["metrics"][metric]) for metric in METRICS},
            "split_manifest_sha256": hashlib.sha256(split_payload).hexdigest(),
        }
    return reference


def _validate_historical_reference_metrics(
    reference: dict[str, dict[str, Any]], expected_pp: dict[str, float]
) -> dict[str, float]:
    actual_pp = {
        metric: 100.0 * _mean([float(row[metric]) for row in reference.values()])
        for metric in METRICS
    }
    mismatches = {
        metric: {"actual_pp": actual_pp[metric], "expected_pp": float(expected_pp[metric])}
        for metric in METRICS
        if abs(actual_pp[metric] - float(expected_pp[metric])) > 0.0051
    }
    if mismatches:
        raise ValueError(
            "historical reference metrics do not match the locked 125-run baseline: "
            f"{mismatches}"
        )
    return actual_pp


def _load_feature_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise ValueError(f"feature cache has no manifest_json: {path}")
        raw = np.asarray(data["manifest_json"]).item()
    return json.loads(str(raw))


def _validate_frozen_feature_caches(
    mappings: dict[str, dict[str, Any]], expected_checkpoint_sha256: str
) -> dict[str, Any]:
    expected_hash = str(expected_checkpoint_sha256).strip().lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("expected_checkpoint_sha256 must contain 64 hex characters")
    manifests: dict[str, dict[str, Any]] = {}
    for receiver_mapping in mappings.values():
        for raw_path in receiver_mapping.values():
            path = Path(str(raw_path))
            if str(path) in manifests:
                continue
            manifest = _load_feature_manifest(path)
            adapter = dict(manifest.get("adapter", {}))
            load_audit = dict(manifest.get("checkpoint_load_audit", {}))
            checks = {
                "payload_source": manifest.get("payload_source")
                == "qknnv42_frozen_adv3b02_identity_only_features_v1",
                "checkpoint_sha256": str(manifest.get("source_checkpoint_sha256", "")).lower()
                == expected_hash,
                "skip_adapter_training": adapter.get("skip_adapter_training") is True,
                "adv3b02_gradient_updates": int(adapter.get("adv3b02_gradient_updates", -1)) == 0,
                "identity_only_forward": manifest.get("identity_only_forward") is True,
                "domain_branch_not_executed": manifest.get("domain_branch_executed_for_qknn")
                is False,
                "feature_name": str(manifest.get("feature_name", "")) == "z_id",
                "checkpoint_load_strict": manifest.get("checkpoint_load_strict") is True,
                "checkpoint_load_audit": all(
                    int(load_audit.get(key, -1)) == 0
                    for key in ("missing_keys", "unexpected_keys", "skipped_mismatch")
                ),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"feature cache is not a frozen qKNN export ({failed}): {path}")
            manifests[str(path)] = manifest
    return {
        "validated_cache_count": len(manifests),
        "source_checkpoint_sha256": expected_hash,
    }


def _apply_head_profile(
    config: dict[str, Any], *, profile: str, old_anchor_bias: float
) -> None:
    config["qknnv42_feature_adapter_mode"] = "support_diag_whiten_fisher"
    if profile == "deployable_light":
        config.update({
            "qknnv42_decision_mode": "per_sample_argmax",
            "qknnv42_labelprop_mode": "disabled",
            "qknnv42_old_anchor_bias": float(old_anchor_bias),
            "non_deployment_oracle_diagnostic": False,
            "publication_protocol": "fixed_adapter_paired_tta_policy_deployment_ablation",
        })
    elif profile in {"full_legacy_oracle", "full_legacy_oracle_prototype"}:
        config.update({
            "qknnv42_decision_mode": "legacy_role_quota_oracle",
            "qknnv42_labelprop_mode": (
                "dense_transductive"
                if profile == "full_legacy_oracle"
                else "support_prototype"
            ),
            "qknnv42_support_representation": (
                "all_support"
                if profile == "full_legacy_oracle"
                else "prototype_only"
            ),
            "qknnv42_old_anchor_bias": 0.001,
            "non_deployment_oracle_diagnostic": True,
            "publication_protocol": (
                "frozen_adv3b02_qknn_feature_adapter_full_history_tta_ablation"
                if profile == "full_legacy_oracle"
                else "frozen_adv3b02_qknn_prototype_full_history_tta_ablation"
            ),
        })
    else:
        raise ValueError(f"unsupported head profile: {profile}")


def _aggregate(
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    historical: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        deltas = [float(row[metric]) - float(baseline[str(row["run_key"])][metric]) for row in rows]
        result[metric] = _mean(values)
        result[f"{metric}_median"] = float(statistics.median(values))
        result[f"{metric}_delta_vs_5view_pp"] = 100.0 * _mean(deltas)
        result[f"{metric}_worst_paired_delta_vs_5view_pp"] = 100.0 * min(deltas)
        result[f"{metric}_paired_drop_gt_3pp_count"] = sum(delta < -0.03 for delta in deltas)
        if historical is not None:
            historical_deltas = [
                float(row[metric]) - float(historical[str(row["run_key"])][metric])
                for row in rows
            ]
            result[f"{metric}_delta_vs_historical_pp"] = 100.0 * _mean(
                historical_deltas
            )
            result[f"{metric}_worst_paired_delta_vs_historical_pp"] = 100.0 * min(
                historical_deltas
            )
            result[f"{metric}_historical_drop_gt_3pp_count"] = sum(
                delta < -0.03 for delta in historical_deltas
            )
    for key in (
        "latency_per_query_ms",
        "estimated_head_macs",
        "persistent_state_bytes",
        "decision_workspace_bytes_lower_bound",
        "estimated_decision_cubic_work_units",
    ):
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
    gate_suffix = "historical" if historical is not None else "5view"
    result["performance_gate_reference"] = gate_suffix
    result["performance_gate_pass"] = all(
        float(result[f"{metric}_delta_vs_{gate_suffix}_pp"]) >= -3.0 - 1e-9
        for metric in METRICS
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
    feature_evidence = {
        policy: _validate_frozen_feature_caches(
            mapping, str(args.expected_checkpoint_sha256)
        )
        for policy, mapping in feature_mappings.items()
    }
    historical = None
    if args.head_profile in {
        "full_legacy_oracle",
        "full_legacy_oracle_prototype",
    } and args.historical_reference_root is None:
        raise ValueError(
            "full_legacy_oracle requires --historical-reference-root with the locked "
            "84.07/93.24/88.23 baseline"
        )
    if args.historical_reference_root is not None:
        historical = _load_historical_reference(args.historical_reference_root)
        expected_reference = len(receivers) * len(args.seed_grid) * len(args.k_grid)
        if len(historical) != expected_reference:
            raise ValueError(
                f"historical reference has {len(historical)} rows; expected {expected_reference}"
            )
        historical_metrics_pp = _validate_historical_reference_metrics(
            historical,
            HISTORICAL_METRIC_DEFAULTS_PP,
        )
    else:
        historical_metrics_pp = None
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    baseline: dict[str, dict[str, Any]] = {}
    for policy in policies:
        view_count = _satellite_tta_view_count(policy)
        for receiver in receivers:
            for seed in args.seed_grid:
                for k_shot in args.k_grid:
                    config = dict(template)
                    config.update({
                        "experiment_id": f"cvs_qknnv42_tta_{policy}_rx{receiver}_k{k_shot}_seed{seed}",
                        "feature_npz_by_receiver_scenario": feature_mappings[policy],
                        "target_receiver_labels": [receiver],
                        "seed": int(seed),
                        "split_seed": int(seed),
                        "k_shot": int(k_shot),
                        "qknnv42_expected_tta_view_count": int(view_count),
                        "backbone_id": (
                            "ADV3B02_CORE90_SOFT_E200_STRICT_LOAD_FROZEN_ZID_"
                            "QKNN_SUPPORT_DIAG_WHITEN_FISHER"
                        ),
                    })
                    _apply_head_profile(
                        config,
                        profile=str(args.head_profile),
                        old_anchor_bias=float(args.old_anchor_bias),
                    )
                    relative = Path(f"rx_{receiver}") / f"seed_{seed}" / f"k_{k_shot}" / "cvs_qknnv42"
                    result = run(config, args.out_root / policy / relative)
                    manifest = result["split_manifest"]
                    if args.head_profile == "full_legacy_oracle":
                        if manifest["qknnv42_decision_mode"] != "legacy_role_quota_oracle":
                            raise AssertionError("full-history profile lost the role/quota oracle")
                        if manifest["qknnv42_labelprop_mode"] != "dense_transductive":
                            raise AssertionError("full-history profile lost dense label propagation")
                    elif args.head_profile == "full_legacy_oracle_prototype":
                        if manifest["qknnv42_decision_mode"] != "legacy_role_quota_oracle":
                            raise AssertionError("prototype profile lost the role/quota oracle")
                        if manifest["qknnv42_labelprop_mode"] != "support_prototype":
                            raise AssertionError("prototype profile lost streaming residual scoring")
                        if manifest["qknnv42_support_representation"] != "prototype_only":
                            raise AssertionError("prototype profile retained support codes")
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
                        "feature_adapter_mode": str(
                            scenario_rows[0]["feature_adapter_mode"]
                        ),
                        "decision_mode": str(scenario_rows[0]["decision_mode"]),
                        "labelprop_mode": str(scenario_rows[0]["labelprop_mode"]),
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
                        "decision_workspace_bytes_lower_bound": _mean(
                            [
                                float(item["decision_workspace_bytes_lower_bound"])
                                for item in scenario_rows
                            ]
                        ),
                        "estimated_decision_cubic_work_units": _mean(
                            [
                                float(item["estimated_decision_cubic_work_units"])
                                for item in scenario_rows
                            ]
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
    if historical is not None:
        historical_split_mismatches = [
            str(row["run_key"])
            for row in rows
            if str(row["split_manifest_sha256"])
            != str(historical[str(row["run_key"])]["split_manifest_sha256"])
        ]
        if historical_split_mismatches:
            raise ValueError(
                "candidate and historical runs do not share identical support/query splits: "
                f"{historical_split_mismatches[:5]}"
            )
    summaries = {
        policy: _aggregate(
            [row for row in rows if row["policy"] == policy], baseline, historical
        )
        for policy in policies
    }
    passing = sorted(
        (value for value in summaries.values() if value["performance_gate_pass"]),
        key=lambda value: int(value["tta_view_count"]),
    )
    summary = {
        "schema": "cvs_qknnv42_frozen_adv3b02_qknn_adapter_tta_ablation_v2",
        "baseline_policy": "rx_light5",
        "historical_reference_root": (
            str(args.historical_reference_root)
            if args.historical_reference_root is not None
            else ""
        ),
        "historical_reference_metrics_pp": historical_metrics_pp,
        "feature_cache_evidence": feature_evidence,
        "head_profile": str(args.head_profile),
        "head": (
            "fft96+dense_transductive+legacy_role_quota_oracle"
            if args.head_profile == "full_legacy_oracle"
            else (
                "fft96+support_prototype+prototype_only+legacy_role_quota_oracle"
                if args.head_profile == "full_legacy_oracle_prototype"
                else "fft96+labelprop_disabled+per_sample_argmax"
            )
        ),
        "adv3b02_gradient_updates": 0,
        "qknn_feature_adapter": "support_diag_whiten_fisher",
        "old_anchor_bias": (
            0.001
            if args.head_profile in {
                "full_legacy_oracle",
                "full_legacy_oracle_prototype",
            }
            else float(args.old_anchor_bias)
        ),
        "performance_gate": (
            "matrix-mean old_acc, seen_new_acc, H_old_new drop vs full historical "
            "60epoch+5view reference <= 3 pp"
            if historical is not None
            else "matrix-mean old_acc, seen_new_acc, H_old_new drop vs candidate 5-view <= 3 pp"
        ),
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
        "--historical-reference-root",
        type=Path,
        default=None,
        help="optional completed full historical 60epoch+5-view run used for the <=3pp gate",
    )
    parser.add_argument(
        "--head-profile",
        choices=(
            "deployable_light",
            "full_legacy_oracle",
            "full_legacy_oracle_prototype",
        ),
        default="deployable_light",
    )
    parser.add_argument(
        "--policies", nargs="+", default=["none", "rx_shift3", "rx_cfo3", "rx_light5"]
    )
    parser.add_argument("--feature-subdir-base", default="ADV3B02_ADAPTER60_FFT96")
    parser.add_argument("--feature-subdir-template", default="{base}_{policy}")
    parser.add_argument("--feature-name", default="features_adapter60_fft96.npz")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
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
