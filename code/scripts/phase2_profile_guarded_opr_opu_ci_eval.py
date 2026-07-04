#!/usr/bin/env python
"""Profile-guarded OPR adapter with OPU-CI rollback.

This evaluator treats ADV3B02/qknn8 as the protected base route.  It may train
several lightweight OPR feature-adapter profiles, but profile selection is based
only on source-old and target-support known-class checks.  target_unknown rows
remain evaluation-only and are never used for adapter fitting, threshold
selection, or profile selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import load_feature_npz  # noqa: E402
from phase2_old_protected_unknown_confirm_ci_eval import (  # noqa: E402
    _evaluate_policy,
    _parse_policy_names,
    _policy_by_name,
    _summary_row,
)
from phase2_proxy_adapter_ci_eval import (  # noqa: E402
    apply_adapter,
    build_training_plan,
    save_adapted_npz,
    train_adapter,
)
from phase2_socapr_dual_route_veto_eval import (  # noqa: E402
    _read_csv_rows,
    _run_route,
    _write_csv,
    build_dual_route_evidence,
)


@dataclass(frozen=True)
class AdapterProfile:
    name: str
    adapter_alpha: float
    old_preserve_weight: float
    proxy_open_weight: float
    support_compact_weight: float
    residual_weight: float
    proxy_open_margin: float


PROFILES: tuple[AdapterProfile, ...] = (
    AdapterProfile(
        name="conservative",
        adapter_alpha=0.10,
        old_preserve_weight=8.0,
        proxy_open_weight=0.35,
        support_compact_weight=0.25,
        residual_weight=0.30,
        proxy_open_margin=0.20,
    ),
    AdapterProfile(
        name="known_tight",
        adapter_alpha=0.14,
        old_preserve_weight=10.0,
        proxy_open_weight=0.55,
        support_compact_weight=0.35,
        residual_weight=0.25,
        proxy_open_margin=0.24,
    ),
    AdapterProfile(
        name="open_light",
        adapter_alpha=0.16,
        old_preserve_weight=6.0,
        proxy_open_weight=0.75,
        support_compact_weight=0.40,
        residual_weight=0.22,
        proxy_open_margin=0.28,
    ),
)


def _profile_names(value: str) -> list[str]:
    if str(value).strip().lower() in {"", "all", "*"}:
        return ["base", *[profile.name for profile in PROFILES]]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _profile_by_name(name: str) -> AdapterProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown adapter profile {name!r}; expected base or {[p.name for p in PROFILES]}")


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _base_metrics() -> dict[str, Any]:
    return {
        "profile_name": "base",
        "target_unknown_training_count": 0,
        "source_proto_acc_before": 1.0,
        "source_proto_acc_after": 1.0,
        "support_proto_acc_before": 1.0,
        "support_proto_acc_after": 1.0,
        "proxy_max_logit_before_mean": 0.0,
        "proxy_max_logit_after_mean": 0.0,
        "mean_source_residual_norm": 0.0,
        "mean_support_residual_norm": 0.0,
        "mean_proxy_residual_norm": 0.0,
        "state_bytes": {"total_fp16_state_bytes": 0},
        "training_counts": {"target_unknown_training_count": 0},
    }


def _guard_report(metrics: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_before = float(metrics.get("source_proto_acc_before", 0.0))
    source_after = float(metrics.get("source_proto_acc_after", 0.0))
    support_before = float(metrics.get("support_proto_acc_before", 0.0))
    support_after = float(metrics.get("support_proto_acc_after", 0.0))
    source_drop = source_before - source_after
    support_drop = support_before - support_after
    proxy_before = float(metrics.get("proxy_max_logit_before_mean", 0.0))
    proxy_after = float(metrics.get("proxy_max_logit_after_mean", 0.0))
    proxy_reduction = proxy_before - proxy_after
    source_residual = float(metrics.get("mean_source_residual_norm", 0.0))
    support_residual = float(metrics.get("mean_support_residual_norm", 0.0))
    target_unknown_training = int(
        metrics.get("training_counts", {}).get("target_unknown_training_count", 0)
        if isinstance(metrics.get("training_counts", {}), Mapping)
        else 0
    )
    pass_guard = (
        source_after >= float(args.min_source_proto_acc)
        and support_after >= float(args.min_support_proto_acc)
        and source_drop <= float(args.max_source_proto_drop)
        and support_drop <= float(args.max_support_proto_drop)
        and proxy_reduction >= float(args.min_proxy_logit_reduction)
        and proxy_after <= float(args.max_proxy_logit_after_mean)
        and source_residual <= float(args.max_source_residual_norm)
        and support_residual <= float(args.max_support_residual_norm)
        and target_unknown_training == 0
    )
    score = source_after + support_after + float(args.proxy_reduction_weight) * max(0.0, proxy_reduction)
    score -= float(args.residual_penalty_weight) * (
        float(metrics.get("mean_source_residual_norm", 0.0))
        + float(metrics.get("mean_support_residual_norm", 0.0))
    )
    return {
        "guard_pass": bool(pass_guard),
        "guard_score": float(score),
        "source_proto_drop": float(source_drop),
        "support_proto_drop": float(support_drop),
        "proxy_max_logit_reduction": float(proxy_reduction),
        "proxy_unknown_surrogate_pass": bool(
            proxy_reduction >= float(args.min_proxy_logit_reduction)
            and proxy_after <= float(args.max_proxy_logit_after_mean)
        ),
        "known_drift_pass": bool(
            source_residual <= float(args.max_source_residual_norm)
            and support_residual <= float(args.max_support_residual_norm)
        ),
        "target_unknown_training_count": int(target_unknown_training),
    }


def _select_profile(profile_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passing = [dict(row) for row in profile_reports if bool(row.get("guard_pass"))]
    if not passing:
        return {"selected_profile": "base", "selection_reason": "no_adapter_profile_passed_known_guard"}
    passing.sort(
        key=lambda row: (
            float(row.get("guard_score", 0.0)),
            row.get("profile_name") != "base",
            -float(row.get("total_fp16_state_bytes", 0.0)),
        ),
        reverse=True,
    )
    selected = passing[0]
    return {
        "selected_profile": str(selected["profile_name"]),
        "selection_reason": "known_and_proxy_surrogate_guard_score",
        "selected_guard_score": float(selected.get("guard_score", 0.0)),
    }


def _adapter_args(args: argparse.Namespace, profile: AdapterProfile) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=args.seed,
        adapter_rank=args.adapter_rank,
        adapter_alpha=profile.adapter_alpha,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        proto_temperature=args.proto_temperature,
        proxy_open_margin=profile.proxy_open_margin,
        source_cls_weight=args.source_cls_weight,
        support_cls_weight=args.support_cls_weight,
        old_preserve_weight=profile.old_preserve_weight,
        proxy_open_weight=profile.proxy_open_weight,
        support_compact_weight=profile.support_compact_weight,
        residual_weight=profile.residual_weight,
        grad_clip=args.grad_clip,
    )


def _evaluate_opu(
    *,
    feature_npz: Path,
    output_dir: Path,
    args: argparse.Namespace,
    profile_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=feature_npz,
        output_dir=output_dir,
        force=True,
    )
    _, safety_csv = _run_route(
        route="safety_route",
        feature_npz=feature_npz,
        output_dir=output_dir,
        force=True,
    )
    known_result = json.loads(known_json.read_text(encoding="utf-8"))
    metadata = dict(known_result["qknn_metadata"])
    metadata["adapter_type"] = f"profile_guarded_opr_opu_ci:{profile_name}"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0
    metadata["target_unknown_eval_only"] = True
    metadata["target_unknown_training_count"] = 0

    evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    _write_csv(output_dir / f"{profile_name}_opu_evidence.csv", evidence)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name in _parse_policy_names(args.policies):
        policy = _policy_by_name(name)
        result = _evaluate_policy(
            evidence,
            metadata,
            policy,
            max_event_bytes=float(args.max_event_bytes),
            max_event_latency_ms=float(args.max_event_latency_ms),
        )
        results[policy.name] = result
        for count, metrics in result["counts"].items():
            row = _summary_row(policy=policy, count=count, metrics=metrics)
            row["profile_name"] = profile_name
            rows.append(row)
    rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    return rows, {"metadata": metadata, "results": results}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--profiles", default="base,conservative,known_tight,open_light")
    p.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--seed", type=int, default=4070412)
    p.add_argument("--support_selection_policy", default="stable_first")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=20)
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.0)
    p.add_argument("--max_source_proto_drop", type=float, default=0.0)
    p.add_argument("--max_support_proto_drop", type=float, default=0.0)
    p.add_argument("--min_source_proto_acc", type=float, default=0.995)
    p.add_argument("--min_support_proto_acc", type=float, default=0.995)
    p.add_argument("--min_proxy_logit_reduction", type=float, default=0.0)
    p.add_argument("--max_proxy_logit_after_mean", type=float, default=1.0e9)
    p.add_argument("--max_source_residual_norm", type=float, default=0.15)
    p.add_argument("--max_support_residual_norm", type=float, default=0.15)
    p.add_argument("--proxy_reduction_weight", type=float, default=0.05)
    p.add_argument("--residual_penalty_weight", type=float, default=0.10)
    p.add_argument("--score_anchor", type=float, default=0.70)
    p.add_argument("--margin_anchor", type=float, default=0.40)
    p.add_argument("--safety_weight", type=float, default=0.35)
    p.add_argument("--discount_mode", choices=["prod", "mean", "max"], default="mean")
    p.add_argument("--max_event_bytes", type=float, default=900.0)
    p.add_argument("--max_event_latency_ms", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_feature_npz(args.feature_npz)
    plan = build_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )

    profile_reports: list[dict[str, Any]] = []
    opu_rows: list[dict[str, Any]] = []
    opu_results: dict[str, Any] = {}
    adapted_paths: dict[str, str] = {"base": str(args.feature_npz)}
    profile_train_metrics: dict[str, Any] = {}

    for profile_name in _profile_names(args.profiles):
        profile_dir = args.output_dir / profile_name
        if profile_name == "base":
            train_metrics = _base_metrics()
            feature_npz = args.feature_npz
        else:
            profile = _profile_by_name(profile_name)
            adapter, train_metrics = train_adapter(payload, plan, _adapter_args(args, profile))
            adapted = apply_adapter(payload, adapter, str(args.device))
            feature_npz = profile_dir / f"{profile_name}_adapted_features.npz"
            metadata = {
                "algorithm": "profile_guarded_OPR-OPU-CI",
                "profile": asdict(profile),
                "target_unknown_eval_only": True,
                "target_unknown_training_count": 0,
                "profile_selection_uses_unknown": False,
                "training_roles": ["source", "proxy_unknown", "target_old_support", "target_new_support"],
                "forbidden_roles": ["target_unknown"],
                "plan": asdict(plan),
                "train_metrics": train_metrics,
            }
            save_adapted_npz(args.feature_npz, feature_npz, adapted, metadata)
        guard = _guard_report(train_metrics, args)
        state_bytes = train_metrics.get("state_bytes", {})
        row = {
            "profile_name": profile_name,
            **guard,
            "source_proto_acc_before": train_metrics.get("source_proto_acc_before", 0.0),
            "source_proto_acc_after": train_metrics.get("source_proto_acc_after", 0.0),
            "support_proto_acc_before": train_metrics.get("support_proto_acc_before", 0.0),
            "support_proto_acc_after": train_metrics.get("support_proto_acc_after", 0.0),
            "proxy_max_logit_before_mean": train_metrics.get("proxy_max_logit_before_mean", 0.0),
            "proxy_max_logit_after_mean": train_metrics.get("proxy_max_logit_after_mean", 0.0),
            "mean_source_residual_norm": train_metrics.get("mean_source_residual_norm", 0.0),
            "mean_support_residual_norm": train_metrics.get("mean_support_residual_norm", 0.0),
            "total_fp16_state_bytes": state_bytes.get("total_fp16_state_bytes", 0)
            if isinstance(state_bytes, Mapping)
            else 0,
        }
        profile_reports.append(row)
        adapted_paths[profile_name] = str(feature_npz)
        profile_train_metrics[profile_name] = train_metrics

        rows, result = _evaluate_opu(
            feature_npz=feature_npz,
            output_dir=profile_dir,
            args=args,
            profile_name=profile_name,
        )
        opu_rows.extend(rows)
        opu_results[profile_name] = result

    selection = _select_profile(profile_reports)
    selected_profile = str(selection["selected_profile"])
    for row in opu_rows:
        row["selected_profile"] = row["profile_name"] == selected_profile

    profile_reports.sort(key=lambda row: (bool(row["guard_pass"]), float(row["guard_score"])), reverse=True)
    opu_rows.sort(key=lambda row: (row["profile_name"] == selected_profile, float(row["joint_score"])), reverse=True)
    _write_summary(args.output_dir / "profile_guard_report.csv", profile_reports)
    _write_summary(args.output_dir / "profile_guarded_opr_opu_ci_summary.csv", opu_rows)
    (args.output_dir / "profile_guarded_opr_opu_ci_summary.json").write_text(
        json.dumps(
            {
                "algorithm": "profile_guarded_OPR-OPU-CI",
                "feature_npz": str(args.feature_npz),
                "target_unknown_eval_only": True,
                "target_unknown_training_count": 0,
                "profile_selection_uses_unknown": False,
                "selection_metric_scope": "source_support_known_plus_source_proxy_unknown_surrogate",
                "target_unknown_selection_count": 0,
                "selection": selection,
                "adapted_paths": adapted_paths,
                "profile_guard_rows": profile_reports,
                "opu_summary_rows": opu_rows,
                "profile_train_metrics": profile_train_metrics,
                "opu_results": opu_results,
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_profile": selected_profile,
                "selection_reason": selection["selection_reason"],
                "profile_guard_rows": len(profile_reports),
                "opu_summary_rows": len(opu_rows),
                "summary_csv": str(args.output_dir / "profile_guarded_opr_opu_ci_summary.csv"),
                "target_unknown_training_count": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
