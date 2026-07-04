#!/usr/bin/env python
"""Old-protected adaptive receiver consensus for collaborative RFFI.

OPC-ARC-CI keeps target-unknown rows evaluation-only. It builds a known
candidate set from support-confirmed old/seen-new evidence, applies a hard
old-class protection cap when an old candidate is strong, and only raises
unknown risk when the known candidate evidence is weak.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import evaluate_collaborative_open_set_evidence  # noqa: E402
from phase2_old_protected_unknown_confirm_ci_eval import (  # noqa: E402
    OpuPolicy,
    _parse_policy_names,
    _policy_by_name,
    _summary_row,
)
from phase2_socapr_dual_route_veto_eval import (  # noqa: E402
    _read_csv_rows,
    _run_route,
    _write_csv,
    build_dual_route_evidence,
)


@dataclass(frozen=True)
class ArcProfile:
    name: str
    description: str
    old_min_score: float
    old_min_margin: float
    old_min_support: float
    old_min_reliability: float
    old_max_tail_risk: float
    old_risk_cap: float
    weak_score_anchor: float
    weak_margin_anchor: float
    weak_support_anchor: float
    weak_reliability_anchor: float
    empty_set_risk_floor: float
    tail_weight: float
    safety_weight: float


PROFILES: tuple[ArcProfile, ...] = (
    ArcProfile(
        name="arc_old_floor",
        description="hard old-candidate floor; unknown only when known evidence is weak",
        old_min_score=0.30,
        old_min_margin=0.025,
        old_min_support=0.25,
        old_min_reliability=0.50,
        old_max_tail_risk=0.45,
        old_risk_cap=0.30,
        weak_score_anchor=0.52,
        weak_margin_anchor=0.05,
        weak_support_anchor=0.30,
        weak_reliability_anchor=0.55,
        empty_set_risk_floor=0.58,
        tail_weight=0.10,
        safety_weight=0.12,
    ),
    ArcProfile(
        name="arc_balanced",
        description="candidate-set empty rejection with moderate old protection",
        old_min_score=0.38,
        old_min_margin=0.045,
        old_min_support=0.35,
        old_min_reliability=0.60,
        old_max_tail_risk=0.55,
        old_risk_cap=0.38,
        weak_score_anchor=0.62,
        weak_margin_anchor=0.10,
        weak_support_anchor=0.45,
        weak_reliability_anchor=0.70,
        empty_set_risk_floor=0.66,
        tail_weight=0.16,
        safety_weight=0.18,
    ),
    ArcProfile(
        name="arc_unknown_safe",
        description="diagnostic stronger empty-candidate unknown rejection",
        old_min_score=0.42,
        old_min_margin=0.06,
        old_min_support=0.40,
        old_min_reliability=0.65,
        old_max_tail_risk=0.60,
        old_risk_cap=0.42,
        weak_score_anchor=0.68,
        weak_margin_anchor=0.14,
        weak_support_anchor=0.52,
        weak_reliability_anchor=0.78,
        empty_set_risk_floor=0.74,
        tail_weight=0.22,
        safety_weight=0.24,
    ),
)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    return float(value) if math.isfinite(float(value)) else float(default)


def _items(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}
    try:
        return {str(part).strip() for part in value if str(part).strip()}
    except TypeError:
        return {str(value).strip()}


def _profile_names(value: str) -> list[str]:
    if str(value).strip().lower() in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _profile_by_name(name: str) -> ArcProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"unknown ARC profile {name!r}; expected one of {[p.name for p in PROFILES]}")


def _weakness(value: float, anchor: float) -> float:
    if anchor <= 0:
        return 0.0
    return _clip01((float(anchor) - float(value)) / float(anchor))


def _tail_risk(row: Mapping[str, Any]) -> float:
    return max(
        _float(row, "socapr_safety_class_negative_risk", _float(row, "class_negative_risk", 0.0)),
        _float(row, "socapr_safety_class_shell_risk", _float(row, "class_shell_risk", 0.0)),
        _float(row, "socapr_safety_evt_risk", _float(row, "evt_risk", 0.0)),
        _float(row, "socapr_safety_mahalanobis_risk", _float(row, "mahalanobis_risk", 0.0)),
        _float(row, "radius_risk", 0.0),
    )


def _top_label(row: Mapping[str, Any]) -> str:
    return str(row.get("class_evidence_top1_label") or row.get("predicted_label") or "")


def _strong_old_candidate(row: Mapping[str, Any], profile: ArcProfile, old_labels: set[str]) -> bool:
    label = _top_label(row)
    tail = _tail_risk(row)
    return (
        label in old_labels
        and _float(row, "known_score", 0.0) >= profile.old_min_score
        and _float(row, "known_margin", 0.0) >= profile.old_min_margin
        and _float(row, "support_density", 0.0) >= profile.old_min_support
        and _float(row, "receiver_class_reliability", 0.0) >= profile.old_min_reliability
        and tail <= profile.old_max_tail_risk
    )


def augment_arc_evidence(
    rows: Sequence[Mapping[str, Any]], profile: ArcProfile, *, old_labels: set[str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        base_risk = _clip01(_float(row, "unknown_risk", 0.0))
        safety_risk = _clip01(_float(row, "socapr_safety_route_unknown_risk", 0.0))
        tail = _clip01(_tail_risk(row))
        weak_known = max(
            _weakness(_float(row, "known_score", 0.0), profile.weak_score_anchor),
            _weakness(_float(row, "known_margin", 0.0), profile.weak_margin_anchor),
            _weakness(_float(row, "support_density", 0.0), profile.weak_support_anchor),
            _weakness(_float(row, "receiver_class_reliability", 0.0), profile.weak_reliability_anchor),
        )
        strong_old = _strong_old_candidate(row, profile, old_labels)
        empty_candidate_risk = _clip01(
            profile.empty_set_risk_floor * weak_known
            + profile.tail_weight * tail
            + profile.safety_weight * safety_risk
        )
        if strong_old:
            unknown_risk = min(base_risk, profile.old_risk_cap)
        else:
            unknown_risk = max(base_risk, empty_candidate_risk)

        q_score = (
            _float(row, "known_score", 0.0)
            + 0.35 * _float(row, "known_margin", 0.0)
            + 0.20 * _float(row, "class_conformal_pvalue", _float(row, "class_evidence_top1_conformal_pvalue", 0.0))
            - 0.45 * unknown_risk
        ) * max(0.0, _float(row, "receiver_class_reliability", 1.0))
        row["arc_profile"] = profile.name
        row["arc_top_label"] = _top_label(row)
        row["arc_strong_old_candidate"] = bool(strong_old)
        row["arc_weak_known_gate"] = float(weak_known)
        row["arc_tail_risk"] = float(tail)
        row["arc_safety_unknown_risk"] = float(safety_risk)
        row["arc_empty_candidate_risk"] = float(empty_candidate_risk)
        row["arc_candidate_q_score"] = float(q_score)
        row["unknown_risk"] = _clip01(unknown_risk)
        row["class_evidence_top1_unknown_risk"] = row["unknown_risk"]
        out.append(row)
    return out


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _evaluate_arc_policy(
    evidence: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    policy: OpuPolicy,
    *,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    return evaluate_collaborative_open_set_evidence(
        evidence,
        collab_counts="all",
        threshold_selection_label_scope=str(metadata["threshold_scope"]),
        unknown_query_eval_only=True,
        protocol_metadata=metadata,
        strict_protocol_metadata=True,
        collab_group_policy="available_up_to_k",
        partial_collab_min_receivers=1,
        fusion_policy="old_protected_unknown_confirm_cvs",
        label_fusion_policy="weighted_vote_margin",
        receiver_selection_policy="support_quality_prior",
        collaboration_policy="dual_route_cvs",
        latency_budget_ms=float(max_event_latency_ms),
        max_event_bytes=float(max_event_bytes),
        max_event_latency_ms=float(max_event_latency_ms),
        unknown_risk_threshold=policy.unknown_risk_threshold,
        accept_margin_threshold=policy.accept_margin_threshold,
        unknown_quantile=0.75,
        consensus_gap_threshold=policy.consensus_gap_threshold,
        consensus_score_threshold=policy.consensus_score_threshold,
        scorer_component_vote_threshold=policy.scorer_component_vote_threshold,
        scorer_risk_components="score,radius,margin,mahalanobis,evt,class_negative,class_shell",
        class_reliability_policy="conformal_margin_risk",
        receiver_class_reliability_policy="support_calibrated",
        class_set_gate_enabled=True,
        old_gate_min_receivers=1,
        old_gate_max_effective_unknown_risk=0.82,
        old_gate_max_component_agreement=0.88,
        old_gate_min_support_density=policy.old_gate_min_support_density,
        seen_new_gate_min_receivers=1,
        seen_new_gate_max_effective_unknown_risk=0.88,
        seen_new_gate_max_component_agreement=0.88,
        seen_new_gate_min_support_density=policy.seen_new_gate_min_support_density,
        candidate_set_min_receivers=policy.candidate_set_min_receivers,
        candidate_set_min_top1_receivers=policy.candidate_set_min_top1_receivers,
        candidate_set_min_conformal_pvalue=policy.candidate_set_min_conformal_pvalue,
        candidate_set_max_label_unknown_risk=policy.candidate_set_max_label_unknown_risk,
        candidate_set_max_event_unknown_risk=policy.candidate_set_max_event_unknown_risk,
        candidate_set_max_label_risk_component_agreement=policy.candidate_set_max_label_risk_component_agreement,
        candidate_set_max_label_shell_risk=policy.candidate_set_max_label_shell_risk,
        candidate_set_shell_reject_risk=policy.candidate_set_shell_reject_risk,
        candidate_set_event_high_unknown_risk_veto=policy.candidate_set_unknown_reject_risk,
        candidate_set_max_label_high_unknown_risk_fraction=0.50,
        candidate_set_high_unknown_risk_threshold=policy.candidate_set_unknown_reject_risk,
        candidate_set_unknown_reject_risk=policy.candidate_set_unknown_reject_risk,
        candidate_set_max_receiver_pair_label_disagreement=policy.candidate_set_max_receiver_pair_label_disagreement,
        candidate_set_max_receiver_pair_unknown_risk_range=policy.candidate_set_max_receiver_pair_unknown_risk_range,
        candidate_set_min_label_receiver_class_reliability=policy.candidate_set_min_label_receiver_class_reliability,
        candidate_set_pairguard_mode="support_calibrated",
        candidate_set_pairguard_action="request_more",
        candidate_set_pairguard_soft_penalty=0.18,
        candidate_set_pairguard_soft_floor=0.28,
        candidate_set_pairguard_soft_min_margin=0.025,
        candidate_set_pairguard_soft_min_agreement=0.45,
        candidate_set_pairguard_soft_min_pvalue=0.0,
        candidate_set_pairguard_soft_min_reliability=0.35,
    )


def _summary_order(row: Mapping[str, Any], profile_names: Sequence[str], policy_names: Sequence[str]) -> tuple[int, int, int]:
    return (
        profile_names.index(str(row["profile"])) if str(row["profile"]) in profile_names else len(profile_names),
        policy_names.index(str(row["policy"])) if str(row["policy"]) in policy_names else len(policy_names),
        int(row["collab_count"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--profiles", default="all")
    p.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    p.add_argument("--force", action="store_true")
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
    known_json, known_csv = _run_route(
        route="known_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    _, safety_csv = _run_route(
        route="safety_route",
        feature_npz=args.feature_npz,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    known_result = json.loads(known_json.read_text(encoding="utf-8"))
    metadata = dict(known_result["qknn_metadata"])
    metadata["adapter_type"] = "old_protected_arc_ci"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0
    metadata["target_unknown_eval_only"] = True
    metadata["target_unknown_training_count"] = 0
    metadata["target_unknown_selection_count"] = 0
    old_labels = _items(metadata.get("old_tx_ids"))

    base_evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    _write_csv(args.output_dir / "arc_base_dual_route_evidence.csv", base_evidence)

    profile_names = _profile_names(args.profiles)
    policy_names = _parse_policy_names(args.policies)
    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for profile_name in profile_names:
        profile = _profile_by_name(profile_name)
        evidence = augment_arc_evidence(base_evidence, profile, old_labels=old_labels)
        _write_csv(args.output_dir / f"{profile.name}_arc_evidence.csv", evidence)
        for policy_name in policy_names:
            policy = _policy_by_name(policy_name)
            result = _evaluate_arc_policy(
                evidence,
                metadata,
                policy,
                max_event_bytes=float(args.max_event_bytes),
                max_event_latency_ms=float(args.max_event_latency_ms),
            )
            results[f"{profile.name}:{policy.name}"] = result
            for count, metrics in result["counts"].items():
                row = _summary_row(policy=policy, count=count, metrics=metrics)
                row["profile"] = profile.name
                row["profile_description"] = profile.description
                row["summary_order"] = "pre_registered_profile_policy_collab_count"
                row["profile_selection_uses_target_unknown"] = False
                row["unknown_query_eval_only"] = True
                row["target_unknown_training_count"] = 0
                row["target_unknown_selection_count"] = 0
                summary_rows.append(row)
    summary_rows.sort(key=lambda row: _summary_order(row, profile_names, policy_names))
    summary_csv = args.output_dir / "arc_ci_summary.csv"
    summary_json = args.output_dir / "arc_ci_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "algorithm": "OPC-ARC-CI",
                "feature_npz": str(args.feature_npz),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "metadata": metadata,
                "profiles": [asdict(_profile_by_name(name)) for name in profile_names],
                "summary_order": "pre_registered_profile_policy_collab_count",
                "profile_selection_uses_target_unknown": False,
                "target_unknown_training_count": 0,
                "target_unknown_selection_count": 0,
                "results": results,
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "summary_csv": str(summary_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
