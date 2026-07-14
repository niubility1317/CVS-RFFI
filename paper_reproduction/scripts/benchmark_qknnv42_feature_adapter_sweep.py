"""Sweep qKNN-side support transforms and FFT weights on frozen feature caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, run
from paper_reproduction.scripts.benchmark_qknnv42_tta_policies import (
    HISTORICAL_METRIC_DEFAULTS_PP,
    METRICS,
    _load_historical_reference,
    _load_feature_manifest,
    _validate_historical_reference_metrics,
)

RESOURCE_FIELDS = (
    "estimated_head_macs",
    "persistent_state_bytes",
    "decision_workspace_bytes_lower_bound",
    "estimated_decision_cubic_work_units",
    "post_feature_adapter_parameter_count",
    "post_feature_adapter_macs_per_sample",
    "post_feature_adapter_support_macs",
    "post_feature_adapter_query_macs",
    "post_feature_adapter_total_macs",
    "post_feature_adapter_state_bytes",
    "estimated_head_macs_with_post_adapter",
    "persistent_state_bytes_with_post_adapter",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _feature_mapping(args: argparse.Namespace, receivers: list[str]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for receiver in receivers:
        path = (
            args.feature_root
            / f"FULL_RX_{receiver}"
            / f"{args.feature_subdir_base}_{args.policy}"
            / str(args.feature_name)
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping[receiver] = {scenario: str(path) for scenario in SCENARIOS}
    return mapping


def _validate_feature_caches(
    mapping: dict[str, Any], expected_checkpoint_sha256: str, *,
    expected_policy: str, expected_tta_view_count: int,
) -> dict[str, Any]:
    expected = str(expected_checkpoint_sha256).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected_checkpoint_sha256 must contain 64 hex characters")
    manifests: dict[str, dict[str, Any]] = {}
    post_count = 0
    for receiver_mapping in mapping.values():
        for raw_path in receiver_mapping.values():
            path = Path(str(raw_path))
            if str(path) in manifests:
                continue
            manifest = _load_feature_manifest(path)
            with np.load(path, allow_pickle=False) as cache:
                if "features" not in cache.files:
                    raise ValueError(f"feature cache has no features array: {path}")
                finite_fields = ["features"]
                if "fft_logmag_features" in cache.files:
                    finite_fields.append("fft_logmag_features")
                nonfinite = [
                    field
                    for field in finite_fields
                    if not np.all(np.isfinite(np.asarray(cache[field])))
                ]
                if nonfinite:
                    raise ValueError(
                        f"feature cache contains non-finite arrays {nonfinite}: {path}"
                    )
            adapter = dict(manifest.get("adapter", {}))
            load_audit = dict(manifest.get("checkpoint_load_audit", {}))
            post = dict(manifest.get("qknn_post_feature_adapter", {}))
            is_post = manifest.get("payload_source") == "qknnv42_post_feature_adapter_v1"
            checks = {
                "payload_source": (
                    is_post
                    and manifest.get("parent_payload_source")
                    == "qknnv42_frozen_adv3b02_identity_only_features_v1"
                    and manifest.get("parent_feature_name") == "z_id"
                    and manifest.get("parent_identity_only_forward") is True
                    and manifest.get("feature_name") == "qknn_post_adapter_z_id"
                    and manifest.get("identity_only_forward") is False
                    and manifest.get("post_feature_adapter_applied") is True
                )
                or (
                    not is_post
                    and manifest.get("payload_source")
                    == "qknnv42_frozen_adv3b02_identity_only_features_v1"
                    and manifest.get("feature_name") == "z_id"
                    and manifest.get("identity_only_forward") is True
                    and not post
                    and manifest.get("post_feature_adapter_applied") is not True
                ),
                "checkpoint_sha256": str(
                    manifest.get("source_checkpoint_sha256", "")
                ).lower()
                == expected,
                "tta_policy": manifest.get("satellite_tta_policy")
                == str(expected_policy),
                "tta_view_count": int(manifest.get("satellite_tta_view_count", -1))
                == int(expected_tta_view_count),
                "skip_adapter_training": adapter.get("skip_adapter_training") is True,
                "adv3b02_gradient_updates": int(
                    adapter.get("adv3b02_gradient_updates", -1)
                )
                == 0,
                "domain_branch_not_executed": manifest.get(
                    "domain_branch_executed_for_qknn"
                )
                is False,
                "qknn_registered_roles_only": manifest.get("export_role_scope")
                == "qknn_registered_only"
                and set(manifest.get("omitted_unused_qknn_roles", []))
                == {"source", "proxy_unknown", "target_unknown"},
                "checkpoint_load_strict": manifest.get("checkpoint_load_strict") is True,
                "checkpoint_load_audit": all(
                    int(load_audit.get(key, -1)) == 0
                    for key in ("missing_keys", "unexpected_keys", "skipped_mismatch")
                ),
                "post_adapter_provenance": not is_post
                or (
                    post.get("uses_target_rows_for_fit") is False
                    and post.get("uses_target_labels_for_fit") is False
                    and post.get("uses_target_query_for_fit") is False
                    and post.get("updates_adv3b02") is False
                    and int(post.get("gradient_updates_adv3b02", -1)) == 0
                    and int(post.get("parameter_count", -1)) >= 0
                    and int(post.get("estimated_macs_per_sample", -1)) >= 0
                    and post.get("policy") == str(expected_policy)
                ),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"feature cache failed strict qKNN validation ({failed}): {path}")
            manifests[str(path)] = manifest
            post_count += int(is_post)
    return {
        "validated_cache_count": len(manifests),
        "post_adapter_cache_count": post_count,
        "source_checkpoint_sha256": expected,
    }


def _apply_head_profile(
    config: dict[str, Any], profile: str, *, deployable_old_anchor_bias: float = -0.001
) -> None:
    if profile == "legacy_oracle_dense":
        config.update(
            {
                "qknnv42_decision_mode": "legacy_role_quota_oracle",
                "qknnv42_labelprop_mode": "dense_transductive",
                "qknnv42_support_representation": "all_support",
                "qknnv42_old_anchor_bias": 0.001,
                "non_deployment_oracle_diagnostic": True,
            }
        )
    elif profile == "deployable_single_qknn":
        config.update(
            {
                "qknnv42_decision_mode": "per_sample_argmax",
                "qknnv42_labelprop_mode": "disabled",
                "qknnv42_support_representation": "all_support",
                "qknnv42_old_anchor_bias": float(deployable_old_anchor_bias),
                "non_deployment_oracle_diagnostic": False,
            }
        )
    else:
        raise ValueError(f"unsupported head profile: {profile}")


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {args.out_root}")
    template = json.loads(args.template_config.read_text(encoding="utf-8"))
    receivers = [str(value) for value in template["publication_target_receiver_grid"]]
    if not receivers or any(not value.strip() for value in receivers):
        raise ValueError("publication_target_receiver_grid must contain non-empty receivers")
    if len(set(receivers)) != len(receivers):
        raise ValueError("publication_target_receiver_grid must not contain duplicates")
    if not args.seed_grid or len(set(int(value) for value in args.seed_grid)) != len(
        args.seed_grid
    ):
        raise ValueError("seed_grid must be non-empty and contain no duplicates")
    if (
        not args.k_grid
        or any(int(value) <= 0 for value in args.k_grid)
        or len(set(int(value) for value in args.k_grid)) != len(args.k_grid)
    ):
        raise ValueError("k_grid must be positive, non-empty, and contain no duplicates")
    mapping = _feature_mapping(args, receivers)
    feature_evidence = _validate_feature_caches(
        mapping,
        str(args.expected_checkpoint_sha256),
        expected_policy=str(args.policy),
        expected_tta_view_count=int(args.tta_view_count),
    )
    expected_per_candidate = len(receivers) * len(args.seed_grid) * len(args.k_grid)
    historical = None
    historical_metrics_pp = None
    if args.historical_reference_root is not None:
        historical = _load_historical_reference(args.historical_reference_root)
        if len(historical) != expected_per_candidate:
            raise ValueError(
                f"historical reference has {len(historical)} rows; expected {expected_per_candidate}"
            )
        historical_metrics_pp = _validate_historical_reference_metrics(
            historical, HISTORICAL_METRIC_DEFAULTS_PP
        )
    if len(set(args.adapter_modes)) != len(args.adapter_modes):
        raise ValueError("adapter_modes must not contain duplicates")
    if not args.aux_weights or any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in args.aux_weights
    ):
        raise ValueError("aux_weights must be finite and in [0,1]")
    if len(set(float(value) for value in args.aux_weights)) != len(args.aux_weights):
        raise ValueError("aux_weights must not contain duplicates")
    if not args.old_anchor_biases or any(
        not math.isfinite(float(value)) or abs(float(value)) > 0.1
        for value in args.old_anchor_biases
    ):
        raise ValueError("old_anchor_biases must be finite and within [-0.1,0.1]")
    if len(set(float(value) for value in args.old_anchor_biases)) != len(
        args.old_anchor_biases
    ):
        raise ValueError("old_anchor_biases must not contain duplicates")
    bias_grid = (
        [0.001]
        if str(args.head_profile) == "legacy_oracle_dense"
        else [float(value) for value in args.old_anchor_biases]
    )
    candidates = [
        (str(mode), float(weight), float(bias))
        for mode in args.adapter_modes
        for weight in args.aux_weights
        for bias in bias_grid
    ]
    expected_total = expected_per_candidate * len(candidates)
    if int(args.expected_runs) != expected_total:
        raise ValueError(
            f"expected_runs={args.expected_runs}, but sweep contains {expected_total} rows"
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for adapter_mode, aux_weight, old_anchor_bias in candidates:
        identity = json.dumps(
            [adapter_mode, aux_weight, old_anchor_bias],
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:10]
        candidate = (
            f"{adapter_mode}__fft{aux_weight:.2f}__bias{old_anchor_bias:+.4f}"
            .replace(".", "p")
            .replace("+", "pos")
            .replace("-", "neg")
            + f"__{digest}"
        )
        for receiver in receivers:
            for seed in args.seed_grid:
                for k_shot in args.k_grid:
                    config = dict(template)
                    config.update(
                        {
                            "experiment_id": f"qknnv42_{candidate}_rx{receiver}_k{k_shot}_s{seed}",
                            "feature_npz_by_receiver_scenario": mapping,
                            "target_receiver_labels": [receiver],
                            "seed": int(seed),
                            "split_seed": int(seed),
                            "k_shot": int(k_shot),
                            "qknnv42_expected_tta_view_count": int(args.tta_view_count),
                            "qknnv42_feature_adapter_mode": adapter_mode,
                            "qknnv42_aux_score_weight": aux_weight,
                        }
                    )
                    _apply_head_profile(
                        config,
                        str(args.head_profile),
                        deployable_old_anchor_bias=old_anchor_bias,
                    )
                    relative = (
                        Path(f"rx_{receiver}")
                        / f"seed_{seed}"
                        / f"k_{k_shot}"
                        / "cvs_qknnv42"
                    )
                    result = run(config, args.out_root / candidate / relative)
                    split_payload = json.dumps(
                        result["split_manifest"]["splits_by_scenario"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    run_key = "/".join(relative.parts[:-1])
                    split_hash = hashlib.sha256(split_payload).hexdigest()
                    if historical is not None and split_hash != historical[run_key]["split_manifest_sha256"]:
                        raise ValueError(f"candidate/historical split mismatch: {run_key}")
                    scenario_rows = list(result["metrics_by_scenario"].values())
                    rows.append(
                        {
                            "candidate": candidate,
                            "head_profile": str(args.head_profile),
                            "adapter_mode": adapter_mode,
                            "aux_weight": aux_weight,
                            "old_anchor_bias": old_anchor_bias,
                            "policy": str(args.policy),
                            "run_key": run_key,
                            "receiver": receiver,
                            "seed": int(seed),
                            "k_shot": int(k_shot),
                            "split_manifest_sha256": split_hash,
                            "decision_mode": str(scenario_rows[0]["decision_mode"]),
                            "labelprop_mode": str(scenario_rows[0]["labelprop_mode"]),
                            "support_representation": str(
                                scenario_rows[0]["support_representation"]
                            ),
                            **{
                                metric: float(result["metrics"][metric])
                                for metric in METRICS
                            },
                            **{
                                field: _mean([float(item[field]) for item in scenario_rows])
                                for field in RESOURCE_FIELDS
                            },
                        }
                    )
    summaries: dict[str, Any] = {}
    for candidate, group_rows in {
        name: [row for row in rows if row["candidate"] == name]
        for name in sorted({str(row["candidate"]) for row in rows})
    }.items():
        if len(group_rows) != expected_per_candidate:
            raise ValueError(f"candidate {candidate} has {len(group_rows)} rows")
        summary: dict[str, Any] = {
            "run_count": len(group_rows),
            "adapter_mode": str(group_rows[0]["adapter_mode"]),
            "aux_weight": float(group_rows[0]["aux_weight"]),
            "old_anchor_bias": float(group_rows[0]["old_anchor_bias"]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary[metric] = _mean(values)
            if historical is not None:
                deltas = [
                    float(row[metric]) - float(historical[str(row["run_key"])][metric])
                    for row in group_rows
                ]
                summary[f"{metric}_delta_vs_historical_pp"] = 100.0 * _mean(deltas)
        for field in RESOURCE_FIELDS:
            summary[field] = _mean([float(row[field]) for row in group_rows])
        summary["performance_gate_pass"] = (
            all(
                float(summary[f"{metric}_delta_vs_historical_pp"]) >= -3.0 - 1.0e-9
                for metric in METRICS
            )
            if historical is not None
            else None
        )
        summaries[candidate] = summary
    ranked = sorted(
        summaries,
        key=lambda name: (
            -float(summaries[name]["H_old_new_mean"]),
            -float(summaries[name]["seen_new_acc_mean"]),
        ),
    )
    summary = {
        "schema": "qknnv42_feature_adapter_sweep_v2",
        "head_profile": str(args.head_profile),
        "policy": str(args.policy),
        "tta_view_count": int(args.tta_view_count),
        "historical_reference_metrics_pp": historical_metrics_pp,
        "feature_cache_evidence": feature_evidence,
        "candidate_count": len(candidates),
        "row_count": len(rows),
        "ranked_candidates": ranked,
        "candidates": summaries,
    }
    with (args.out_root / "paired_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--historical-reference-root", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--policy", choices=("none", "rx_shift3", "rx_cfo3", "rx_light5"), default="none")
    parser.add_argument("--tta-view-count", type=int, default=1)
    parser.add_argument(
        "--head-profile",
        choices=("legacy_oracle_dense", "deployable_single_qknn"),
        default="legacy_oracle_dense",
    )
    parser.add_argument("--feature-subdir-base", default="ADV3B02_FROZEN_QKNN_FFT96")
    parser.add_argument("--feature-name", default="features_frozen_adv3b02_fft96.npz")
    parser.add_argument(
        "--adapter-modes",
        nargs="+",
        default=["none", "support_center", "support_diag_whiten", "support_diag_whiten_fisher"],
    )
    parser.add_argument("--aux-weights", nargs="+", type=float, default=[0.0, 0.34, 0.7, 1.0])
    parser.add_argument("--old-anchor-biases", nargs="+", type=float, default=[-0.001])
    parser.add_argument("--seed-grid", nargs="+", type=int, default=[713101, 713102, 713103, 713104, 713105])
    parser.add_argument("--k-grid", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--expected-runs", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    print(
        json.dumps(
            benchmark(parse_args()), ensure_ascii=False, indent=2, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
