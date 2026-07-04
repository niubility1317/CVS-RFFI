#!/usr/bin/env python
"""Evaluate disagreement-aware rejection confirmation for satellite CI.

DARC-CI uses only base qKNN evidence plus same-event receiver disagreement. It
does not use unknown query labels for thresholds and does not change labels.
The auxiliary risk is raised only when known evidence is weak and receiver
views disagree; strong old-class candidates are capped to protect old accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from scripts.phase2_old_protected_unknown_confirm_ci_eval import (  # noqa: E402
    POLICIES,
    OpuPolicy,
    _evaluate_policy,
    _summary_row,
)


@dataclass(frozen=True)
class DarcProfile:
    name: str
    disagreement_weight: float
    weak_weight: float
    tail_weight: float
    strong_old_score: float
    strong_old_margin: float
    strong_old_support: float
    strong_old_agreement: float
    old_risk_cap: float
    weak_score_anchor: float
    weak_margin_anchor: float
    weak_support_anchor: float
    weak_reliability_anchor: float


PROFILES: tuple[DarcProfile, ...] = (
    DarcProfile(
        name="darc_light",
        disagreement_weight=0.35,
        weak_weight=0.10,
        tail_weight=0.04,
        strong_old_score=0.55,
        strong_old_margin=0.05,
        strong_old_support=0.30,
        strong_old_agreement=0.60,
        old_risk_cap=0.30,
        weak_score_anchor=0.65,
        weak_margin_anchor=0.12,
        weak_support_anchor=0.40,
        weak_reliability_anchor=0.70,
    ),
    DarcProfile(
        name="darc_balanced",
        disagreement_weight=0.55,
        weak_weight=0.18,
        tail_weight=0.06,
        strong_old_score=0.58,
        strong_old_margin=0.06,
        strong_old_support=0.35,
        strong_old_agreement=0.70,
        old_risk_cap=0.35,
        weak_score_anchor=0.68,
        weak_margin_anchor=0.14,
        weak_support_anchor=0.45,
        weak_reliability_anchor=0.72,
    ),
    DarcProfile(
        name="darc_unknown_push",
        disagreement_weight=0.80,
        weak_weight=0.25,
        tail_weight=0.08,
        strong_old_score=0.62,
        strong_old_margin=0.08,
        strong_old_support=0.40,
        strong_old_agreement=0.80,
        old_risk_cap=0.45,
        weak_score_anchor=0.72,
        weak_margin_anchor=0.16,
        weak_support_anchor=0.50,
        weak_reliability_anchor=0.75,
    ),
)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _deficit(value: float, anchor: float) -> float:
    return _unit((float(anchor) - float(value)) / max(float(anchor), 1e-6))


def _top_label(row: Mapping[str, Any]) -> str:
    return str(row.get("class_evidence_top1_label") or row.get("predicted_label") or "")


def _tail_risk(row: Mapping[str, Any]) -> float:
    return max(
        _float(row, "socapr_safety_route_unknown_risk"),
        _float(row, "socapr_safety_class_negative_risk"),
        _float(row, "socapr_safety_class_shell_risk"),
        _float(row, "socapr_safety_evt_risk"),
        _float(row, "socapr_safety_mahalanobis_risk"),
        _float(row, "class_negative_risk"),
        _float(row, "class_shell_risk"),
        _float(row, "evt_risk"),
        _float(row, "mahalanobis_risk"),
        _float(row, "radius_risk"),
    )


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _event_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("event_id", ""))].append(row)
    stats: dict[str, dict[str, float]] = {}
    for event_id, group in grouped.items():
        labels = [_top_label(row) for row in group]
        counts = Counter(labels)
        n = max(len(group), 1)
        top_count = counts.most_common(1)[0][1] if counts else 0
        agreement = float(top_count) / float(n)
        margins = [_float(row, "known_margin") for row in group]
        scores = [_float(row, "known_score") for row in group]
        risks = [_float(row, "unknown_risk") for row in group]
        stats[event_id] = {
            "receiver_count": float(n),
            "label_agreement": agreement,
            "label_disagreement": 1.0 - agreement,
            "mean_margin": sum(margins) / float(n),
            "min_margin": min(margins) if margins else 0.0,
            "mean_score": sum(scores) / float(n),
            "unknown_risk_range": (max(risks) - min(risks)) if risks else 0.0,
        }
    return stats


def _old_labels_from_rows(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("true_label")) for row in rows if str(row.get("role")) == "old"}


def _strong_old(row: Mapping[str, Any], stats: Mapping[str, float], profile: DarcProfile, old_labels: set[str]) -> bool:
    return (
        _top_label(row) in old_labels
        and _float(row, "known_score") >= profile.strong_old_score
        and _float(row, "known_margin") >= profile.strong_old_margin
        and _float(row, "support_density") >= profile.strong_old_support
        and float(stats.get("label_agreement", 0.0)) >= profile.strong_old_agreement
    )


def augment_darc_evidence(
    rows: Sequence[Mapping[str, Any]],
    profile: DarcProfile,
    *,
    old_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    labels = set(old_labels or _old_labels_from_rows(rows))
    stats_by_event = _event_stats(rows)
    out: list[dict[str, Any]] = []
    for source in rows:
        event_id = str(source.get("event_id", ""))
        stats = stats_by_event[event_id]
        weak_known = max(
            _deficit(_float(source, "known_score"), profile.weak_score_anchor),
            _deficit(_float(source, "known_margin"), profile.weak_margin_anchor),
            _deficit(_float(source, "support_density"), profile.weak_support_anchor),
            _deficit(_float(source, "receiver_class_reliability"), profile.weak_reliability_anchor),
        )
        disagreement = _unit(float(stats.get("label_disagreement", 0.0)))
        tail = _unit(_tail_risk(source))
        aux = _unit(
            profile.disagreement_weight * disagreement
            + profile.weak_weight * weak_known * max(disagreement, 0.25)
            + profile.tail_weight * tail * max(disagreement, 0.25)
        )
        base_risk = _unit(_float(source, "unknown_risk"))
        strong_old = _strong_old(source, stats, profile, labels)
        fused = max(base_risk, aux)
        if strong_old:
            fused = min(fused, profile.old_risk_cap)
        row = dict(source)
        row["unknown_risk"] = fused
        row["class_evidence_top1_unknown_risk"] = fused
        row["darc_profile"] = profile.name
        row["darc_top_label"] = _top_label(row)
        row["darc_label_agreement"] = float(stats.get("label_agreement", 0.0))
        row["darc_label_disagreement"] = disagreement
        row["darc_weak_known"] = weak_known
        row["darc_tail_risk"] = tail
        row["darc_aux_unknown_risk"] = aux
        row["darc_strong_old_candidate"] = int(strong_old)
        row["darc_label_authority"] = "base_qknn_only"
        out.append(row)
    return out


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("metadata"), dict):
        return dict(data["metadata"])
    if isinstance(data.get("qknn_metadata"), dict):
        return dict(data["qknn_metadata"])
    return dict(data)


def _select_profiles(spec: str) -> list[DarcProfile]:
    if str(spec).strip().lower() in {"", "all", "*"}:
        return list(PROFILES)
    wanted = [part.strip() for part in str(spec).split(",") if part.strip()]
    out = []
    for name in wanted:
        for profile in PROFILES:
            if profile.name == name:
                out.append(profile)
                break
        else:
            raise ValueError(f"unknown DARC profile {name!r}")
    return out


def _select_policies(spec: str) -> list[OpuPolicy]:
    if str(spec).strip().lower() in {"all", "*"}:
        return list(POLICIES)
    wanted = {part.strip() for part in str(spec).split(",") if part.strip()}
    selected = [policy for policy in POLICIES if policy.name in wanted]
    missing = sorted(wanted - {policy.name for policy in selected})
    if missing:
        raise ValueError(f"unknown policies: {missing}")
    return selected


def _flatten_counts(
    *,
    algorithm: str,
    profile: str,
    policy: OpuPolicy,
    metrics: Mapping[str, Any],
    base_counts: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count, count_metrics in sorted(metrics.get("counts", {}).items(), key=lambda item: int(item[0])):
        row = _summary_row(policy=policy, count=str(count), metrics=count_metrics)
        row.update(
            {
                "algorithm": algorithm,
                "darc_profile": profile,
                "threshold_fit_scope": metrics.get("threshold_selection_label_scope", ""),
                "unknown_query_used_for_threshold": "false",
                "old_to_reject": json.dumps(
                    count_metrics.get("per_old_class_decision_counts", {}),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "unknown_output_counts": json.dumps(
                    count_metrics.get("unknown_output_counts", {}),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )
        if base_counts is not None:
            base = base_counts[str(count)]
            row["delta_old_acc"] = float(row["old_acc"]) - float(base.get("old_acc", 0.0))
            row["delta_seen_new_acc"] = float(row["seen_new_acc"]) - float(base.get("seen_new_acc", 0.0))
            row["delta_unknown_reject_rate"] = float(row["unknown_reject_rate"]) - float(
                base.get("unknown_reject_rate", 0.0)
            )
            row["delta_unknown_FAR"] = float(row["unknown_FAR"]) - float(base.get("unknown_FAR", 0.0))
            row["old_not_drop_pass"] = float(row["delta_old_acc"]) >= -1e-12
            row["verdict"] = (
                "candidate"
                if row["old_not_drop_pass"]
                and float(row["old_acc"]) >= 0.80
                and float(row["delta_unknown_reject_rate"]) > 0.0
                and float(row["delta_unknown_FAR"]) <= 0.0
                else "diagnostic_only"
            )
        rows.append(row)
    return rows


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_evidence_csv", type=Path, required=True)
    parser.add_argument("--metadata_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    parser.add_argument("--max_event_bytes", type=float, default=900.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--write_evidence", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv_rows(args.base_evidence_csv)
    metadata = _load_metadata(args.metadata_json)
    policies = _select_policies(args.policies)
    profiles = _select_profiles(args.profiles)

    base_metrics: dict[str, dict[str, Any]] = {
        policy.name: _evaluate_policy(
            base_rows,
            metadata,
            policy,
            max_event_bytes=args.max_event_bytes,
            max_event_latency_ms=args.max_event_latency_ms,
        )
        for policy in policies
    }
    summary_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for policy in policies:
        summary_rows.extend(
            _flatten_counts(
                algorithm="base_paired",
                profile="base",
                policy=policy,
                metrics=base_metrics[policy.name],
            )
        )
    for profile in profiles:
        evidence = augment_darc_evidence(base_rows, profile)
        if args.write_evidence:
            write_csv_rows(args.output_dir / f"darc_evidence_{profile.name}.csv", evidence)
        for policy in policies:
            metrics = _evaluate_policy(
                evidence,
                metadata,
                policy,
                max_event_bytes=args.max_event_bytes,
                max_event_latency_ms=args.max_event_latency_ms,
            )
            rows = _flatten_counts(
                algorithm="darc_ci",
                profile=profile.name,
                policy=policy,
                metrics=metrics,
                base_counts=base_metrics[policy.name]["counts"],
            )
            summary_rows.extend(rows)
            best_rows.extend(rows)

    best_rows = sorted(
        best_rows,
        key=lambda row: (
            row.get("verdict") == "candidate",
            float(row.get("old_acc", 0.0)),
            float(row.get("delta_unknown_reject_rate", 0.0)),
            float(row.get("unknown_reject_rate", 0.0)),
            -float(row.get("unknown_FAR", 1.0)),
        ),
        reverse=True,
    )
    write_csv_rows(args.output_dir / "darc_ci_summary.csv", summary_rows)
    write_csv_rows(args.output_dir / "darc_ci_best_rows.csv", best_rows[:50])
    (args.output_dir / "darc_ci_audit.json").write_text(
        json.dumps(
            {
                "algorithm": "DARC-CI",
                "base_evidence_csv": str(args.base_evidence_csv),
                "metadata_json": str(args.metadata_json),
                "profiles": [profile.__dict__ for profile in profiles],
                "candidate_count": sum(1 for row in best_rows if row.get("verdict") == "candidate"),
                "same_event_receiver_disagreement_only": True,
                "unknown_query_used_for_threshold": False,
                "label_authority": "base_qknn_only",
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "candidate_count": sum(1 for row in best_rows if row.get("verdict") == "candidate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
