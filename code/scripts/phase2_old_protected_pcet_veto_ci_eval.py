#!/usr/bin/env python
"""Old-protected PCET veto on top of OPU-CI dual-route evidence.

OPV-CI keeps the qknn8/OPU known route intact, then adds a query-free PCET-style
unknown-risk veto only when known evidence is weak.  Strong support-confirmed
known evidence caps the extra risk so unknown rejection cannot be improved by
silently rejecting old/seen-new samples.  target_unknown rows remain
evaluation-only and are never used for threshold fitting or profile selection.
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

from phase2_old_protected_unknown_confirm_ci_eval import (  # noqa: E402
    _evaluate_policy,
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
class OpvProfile:
    name: str
    description: str
    weak_score_anchor: float
    weak_margin_anchor: float
    weak_support_anchor: float
    weak_reliability_anchor: float
    safety_weight: float
    tail_weight: float
    proto_weight: float
    old_safe_score: float
    old_safe_margin: float
    old_safe_support: float
    old_safe_reliability: float
    old_safe_risk_cap: float


PROFILES: tuple[OpvProfile, ...] = (
    OpvProfile(
        name="opv_ultra_preserve",
        description="very light veto; accepts only no-regression unknown gains over OPU old-preserve",
        weak_score_anchor=0.46,
        weak_margin_anchor=0.045,
        weak_support_anchor=0.30,
        weak_reliability_anchor=0.55,
        safety_weight=0.06,
        tail_weight=0.03,
        proto_weight=0.03,
        old_safe_score=0.28,
        old_safe_margin=0.025,
        old_safe_support=0.25,
        old_safe_reliability=0.50,
        old_safe_risk_cap=0.35,
    ),
    OpvProfile(
        name="opv_preserve",
        description="minimal PCET veto; preserve OPU known acceptance first",
        weak_score_anchor=0.58,
        weak_margin_anchor=0.10,
        weak_support_anchor=0.45,
        weak_reliability_anchor=0.70,
        safety_weight=0.20,
        tail_weight=0.10,
        proto_weight=0.10,
        old_safe_score=0.45,
        old_safe_margin=0.06,
        old_safe_support=0.45,
        old_safe_reliability=0.70,
        old_safe_risk_cap=0.45,
    ),
    OpvProfile(
        name="opv_balanced",
        description="balanced old-protected risk veto for weak-known events",
        weak_score_anchor=0.65,
        weak_margin_anchor=0.16,
        weak_support_anchor=0.55,
        weak_reliability_anchor=0.80,
        safety_weight=0.35,
        tail_weight=0.20,
        proto_weight=0.15,
        old_safe_score=0.50,
        old_safe_margin=0.08,
        old_safe_support=0.50,
        old_safe_reliability=0.75,
        old_safe_risk_cap=0.52,
    ),
    OpvProfile(
        name="opv_unknown_push",
        description="stronger veto; diagnostic unless old/seen retention remains non-degraded",
        weak_score_anchor=0.72,
        weak_margin_anchor=0.22,
        weak_support_anchor=0.60,
        weak_reliability_anchor=0.85,
        safety_weight=0.50,
        tail_weight=0.28,
        proto_weight=0.18,
        old_safe_score=0.55,
        old_safe_margin=0.10,
        old_safe_support=0.55,
        old_safe_reliability=0.80,
        old_safe_risk_cap=0.58,
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
        return float(default)
    return value if math.isfinite(value) else float(default)


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    unknown = sorted(set(names) - known)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown OPV profile(s): {', '.join(unknown)}")
    return names


def _profile_by_name(name: str) -> OpvProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(name)


def _weakness(value: float, anchor: float) -> float:
    return _clip01((float(anchor) - float(value)) / max(float(anchor), 1e-6))


def _tail_risk(row: Mapping[str, Any]) -> float:
    return max(
        _clip01(_float(row, "socapr_safety_class_negative_risk", 0.0)),
        _clip01(_float(row, "socapr_safety_class_shell_risk", 0.0)),
        _clip01(_float(row, "socapr_safety_evt_risk", 0.0)),
        _clip01(_float(row, "socapr_safety_mahalanobis_risk", 0.0)),
        _clip01(_float(row, "class_evidence_top1_class_shell_risk", 0.0)),
        _clip01(_float(row, "class_evidence_top1_evt_risk", 0.0)),
        _clip01(_float(row, "class_evidence_top1_mahalanobis_risk", 0.0)),
    )


def _proto_instability(row: Mapping[str, Any], profile: OpvProfile) -> float:
    score = _float(row, "known_score", _float(row, "class_evidence_top1_score", 0.0))
    margin = _float(row, "known_margin", _float(row, "class_evidence_top1_margin", 0.0))
    support = _float(row, "support_density", 0.0)
    reliability = _float(row, "receiver_class_reliability", _float(row, "reliability", 0.0))
    return _clip01(
        0.35 * _weakness(score, profile.weak_score_anchor)
        + 0.30 * _weakness(margin, profile.weak_margin_anchor)
        + 0.20 * _weakness(support, profile.weak_support_anchor)
        + 0.15 * _weakness(reliability, profile.weak_reliability_anchor)
    )


def _strong_known(row: Mapping[str, Any], profile: OpvProfile) -> bool:
    return bool(
        _float(row, "known_score", 0.0) >= profile.old_safe_score
        and _float(row, "known_margin", 0.0) >= profile.old_safe_margin
        and _float(row, "support_density", 0.0) >= profile.old_safe_support
        and _float(row, "receiver_class_reliability", _float(row, "reliability", 0.0)) >= profile.old_safe_reliability
    )


def augment_opv_evidence(
    evidence: Sequence[Mapping[str, Any]],
    profile: OpvProfile,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in evidence:
        row = dict(source)
        base_risk = _clip01(_float(row, "unknown_risk", 0.0))
        safety_risk = _clip01(_float(row, "socapr_safety_route_unknown_risk", 0.0))
        tail = _tail_risk(row)
        proto = _proto_instability(row, profile)
        weak_gate = max(
            _weakness(_float(row, "known_score", 0.0), profile.weak_score_anchor),
            _weakness(_float(row, "known_margin", 0.0), profile.weak_margin_anchor),
            _weakness(_float(row, "support_density", 0.0), profile.weak_support_anchor),
        )
        veto_risk = _clip01(
            base_risk
            + weak_gate
            * (
                float(profile.safety_weight) * safety_risk
                + float(profile.tail_weight) * tail
                + float(profile.proto_weight) * proto
            )
        )
        strong_known = _strong_known(row, profile)
        if strong_known:
            veto_risk = min(veto_risk, float(profile.old_safe_risk_cap))
        row["opv_profile"] = profile.name
        row["opv_base_unknown_risk"] = base_risk
        row["opv_safety_unknown_risk"] = safety_risk
        row["opv_tail_risk"] = tail
        row["opv_proto_instability"] = proto
        row["opv_weak_known_gate"] = weak_gate
        row["opv_strong_known_cap_applied"] = bool(strong_known)
        row["unknown_risk"] = _clip01(veto_risk)
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


def _joint_score(metrics: Mapping[str, Any]) -> float:
    return (
        _float(metrics, "old_acc")
        + _float(metrics, "seen_new_acc")
        + _float(metrics, "unknown_reject_rate")
        - _float(metrics, "unknown_FAR")
        - 0.5 * _float(metrics, "defer_rate")
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
    metadata["adapter_type"] = "old_protected_pcet_veto_ci"
    metadata["safety_route_scope"] = "support_only_virtual_unknown_no_unknown_query_calibration"
    metadata["evidence_bytes_per_receiver_event"] = 168.0
    metadata["target_unknown_eval_only"] = True
    metadata["target_unknown_training_count"] = 0

    base_evidence = build_dual_route_evidence(
        _read_csv_rows(known_csv),
        _read_csv_rows(safety_csv),
        score_anchor=float(args.score_anchor),
        margin_anchor=float(args.margin_anchor),
        safety_weight=float(args.safety_weight),
        discount_mode=str(args.discount_mode),
    )
    _write_csv(args.output_dir / "opv_base_dual_route_evidence.csv", base_evidence)

    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for profile_name in _profile_names(args.profiles):
        profile = _profile_by_name(profile_name)
        evidence = augment_opv_evidence(base_evidence, profile)
        _write_csv(args.output_dir / f"{profile.name}_opv_evidence.csv", evidence)
        for policy_name in _parse_policy_names(args.policies):
            policy = _policy_by_name(policy_name)
            result = _evaluate_policy(
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
                row["joint_score"] = _joint_score(metrics)
                row["target_unknown_training_count"] = 0
                row["target_unknown_selection_count"] = 0
                summary_rows.append(row)

    summary_rows.sort(
        key=lambda row: (
            float(row["old_acc"]) >= 0.80,
            float(row["unknown_reject_rate"]),
            float(row["old_acc"]),
            float(row["seen_new_acc"]),
            -float(row["unknown_FAR"]),
        ),
        reverse=True,
    )
    summary_csv = args.output_dir / "opv_ci_summary.csv"
    summary_json = args.output_dir / "opv_ci_summary.json"
    _write_summary(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "algorithm": "OPV-CI",
                "feature_npz": str(args.feature_npz),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "metadata": metadata,
                "profiles": [asdict(_profile_by_name(name)) for name in _profile_names(args.profiles)],
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
