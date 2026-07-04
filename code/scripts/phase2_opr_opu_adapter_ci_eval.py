#!/usr/bin/env python
"""Evaluate OPR adapter plus OPU old-protected collaborative confirmation.

The script freezes ADV3B02 features, fits only a low-rank feature adapter from
source old, source-side proxy_unknown, and target old/seen-new support, then
runs the OPU-CI old-protected unknown-confirmation backend. target_unknown rows
remain evaluation-only and are never used by the adapter or threshold fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
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


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--adapter_npz", type=Path, default=None)
    p.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--seed", type=int, default=4070411)
    p.add_argument("--support_selection_policy", default="stable_first")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=30)
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--adapter_alpha", type=float, default=0.20)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--proxy_open_margin", type=float, default=0.20)
    p.add_argument("--source_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.0)
    p.add_argument("--old_preserve_weight", type=float, default=4.0)
    p.add_argument("--proxy_open_weight", type=float, default=1.0)
    p.add_argument("--support_compact_weight", type=float, default=0.5)
    p.add_argument("--residual_weight", type=float, default=0.15)
    p.add_argument("--score_anchor", type=float, default=0.70)
    p.add_argument("--margin_anchor", type=float, default=0.40)
    p.add_argument("--safety_weight", type=float, default=0.35)
    p.add_argument("--discount_mode", choices=["prod", "mean", "max"], default="mean")
    p.add_argument("--max_event_bytes", type=float, default=900.0)
    p.add_argument("--max_event_latency_ms", type=float, default=2.0)
    return p.parse_args(argv)


def _adapter_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=args.seed,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        proto_temperature=args.proto_temperature,
        proxy_open_margin=args.proxy_open_margin,
        source_cls_weight=args.source_cls_weight,
        support_cls_weight=args.support_cls_weight,
        old_preserve_weight=args.old_preserve_weight,
        proxy_open_weight=args.proxy_open_weight,
        support_compact_weight=args.support_compact_weight,
        residual_weight=args.residual_weight,
        grad_clip=args.grad_clip,
    )


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
    adapter, train_metrics = train_adapter(payload, plan, _adapter_args(args))
    adapted = apply_adapter(payload, adapter, str(args.device))
    adapted_npz = args.adapter_npz or (args.output_dir / "opr_opu_adapted_features.npz")
    adapter_metadata = {
        "algorithm": "OPR-OPU-CI",
        "adapter": "low_rank_residual_feature_adapter",
        "collaborative_backend": "OPU-CI",
        "target_unknown_eval_only": True,
        "target_unknown_training_count": 0,
        "training_roles": ["source", "proxy_unknown", "target_old_support", "target_new_support"],
        "forbidden_roles": ["target_unknown"],
        "plan": asdict(plan),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_metrics": train_metrics,
    }
    save_adapted_npz(args.feature_npz, adapted_npz, adapted, adapter_metadata)

    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=adapted_npz,
        output_dir=args.output_dir,
        force=True,
    )
    _, safety_csv = _run_route(
        route="safety_route",
        feature_npz=adapted_npz,
        output_dir=args.output_dir,
        force=True,
    )
    known_result = json.loads(known_json.read_text(encoding="utf-8"))
    metadata = dict(known_result["qknn_metadata"])
    metadata["adapter_type"] = "opr_opu_ci_low_rank_proxy_open_adapter"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0
    metadata["target_unknown_eval_only"] = True
    metadata["target_unknown_training_count"] = 0
    metadata["adapter_state_bytes"] = train_metrics.get("state_bytes", {})

    evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    evidence_csv = args.output_dir / "opr_opu_ci_evidence.csv"
    _write_csv(evidence_csv, evidence)

    summary_rows: list[dict[str, Any]] = []
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
            row["adapter_train_seconds"] = train_metrics.get("adapter_train_seconds", 0.0)
            row["adapter_total_fp16_state_bytes"] = train_metrics.get("state_bytes", {}).get(
                "total_fp16_state_bytes",
                0,
            )
            summary_rows.append(row)

    summary_rows.sort(key=lambda row: float(row["joint_score"]), reverse=True)
    summary_csv = args.output_dir / "opr_opu_ci_summary.csv"
    summary_json = args.output_dir / "opr_opu_ci_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "adapted_feature_npz": str(adapted_npz),
                "adapter_metadata": adapter_metadata,
                "metadata": metadata,
                "results": results,
                "summary_rows": summary_rows,
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
                "adapted_feature_npz": str(adapted_npz),
                "summary_rows": len(summary_rows),
                "summary_csv": str(summary_csv),
                "target_unknown_training_count": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
