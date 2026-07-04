#!/usr/bin/env python
"""Evaluate cross-feature auxiliary unknown-risk collaborative inference.

XFA-CI keeps the frozen base qKNN label route as the only label authority.
Auxiliary adapted-feature evidence may only raise unknown/defer risk, and only
on rows where the base route is weak. Base-only and base+aux are evaluated on
the exact same matched event rows to avoid subset-driven gains.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


KEY_FIELDS = ("event_id", "receiver_id", "role", "true_label")
AUX_RISK_FIELDS = (
    "unknown_risk",
    "socapr_safety_route_unknown_risk",
    "class_negative_risk",
    "class_shell_risk",
    "evt_risk",
    "mahalanobis_risk",
)


@dataclass(frozen=True)
class XfaConfig:
    aux_weight: float
    strong_score: float
    strong_margin: float
    strong_support_density: float
    strong_reliability: float
    weak_score_anchor: float
    weak_margin_anchor: float
    weak_support_anchor: float
    weak_reliability_anchor: float
    strong_aux_cap: float
    aux_bytes_per_receiver: float


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)  # type: ignore[return-value]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate evidence key {key}")
        indexed[key] = row
    return indexed


def _deficit(value: float, anchor: float) -> float:
    anchor = max(float(anchor), 1e-6)
    return _unit((anchor - float(value)) / anchor)


def _base_strength(row: Mapping[str, Any], config: XfaConfig) -> tuple[bool, float]:
    score = max(_float(row, "known_score"), _float(row, "class_evidence_top1_score"))
    margin = max(_float(row, "known_margin"), _float(row, "class_evidence_top1_margin"))
    support = _float(row, "support_density")
    reliability = max(_float(row, "reliability"), _float(row, "receiver_class_reliability"))
    strong = (
        score >= config.strong_score
        and margin >= config.strong_margin
        and support >= config.strong_support_density
        and reliability >= config.strong_reliability
    )
    weakness = max(
        _deficit(score, config.weak_score_anchor),
        _deficit(margin, config.weak_margin_anchor),
        _deficit(support, config.weak_support_anchor),
        _deficit(reliability, config.weak_reliability_anchor),
    )
    return strong, weakness


def _aux_unknown_risk(row: Mapping[str, Any]) -> tuple[float, str]:
    values = [(field, _unit(_float(row, field))) for field in AUX_RISK_FIELDS if field in row]
    if not values:
        return 0.0, "missing"
    field, value = max(values, key=lambda item: item[1])
    return value, field


def build_xfa_evidence(
    base_rows: Sequence[Mapping[str, Any]],
    aux_rows: Sequence[Mapping[str, Any]],
    config: XfaConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aux_index = _index_rows(aux_rows)
    paired_base: list[dict[str, Any]] = []
    xfa_rows: list[dict[str, Any]] = []
    missing = 0
    strong_count = 0
    aux_lift_count = 0
    for base in base_rows:
        key = _key(base)
        aux = aux_index.get(key)
        if aux is None:
            missing += 1
            continue
        base_out = dict(base)
        xfa = dict(base)
        base_risk = _unit(_float(base, "unknown_risk"))
        aux_risk, aux_source = _aux_unknown_risk(aux)
        strong, weakness = _base_strength(base, config)
        aux_component = _unit(config.aux_weight * aux_risk * weakness)
        if strong:
            strong_count += 1
            aux_component = min(aux_component, config.strong_aux_cap)
        fused_risk = max(base_risk, aux_component)
        if fused_risk > base_risk + 1e-12:
            aux_lift_count += 1
        xfa["unknown_risk"] = fused_risk
        xfa["xfa_base_unknown_risk"] = base_risk
        xfa["xfa_aux_unknown_risk"] = aux_risk
        xfa["xfa_aux_risk_source"] = aux_source
        xfa["xfa_aux_component"] = aux_component
        xfa["xfa_base_strong_known"] = int(strong)
        xfa["xfa_base_weakness"] = weakness
        xfa["xfa_aux_weight"] = float(config.aux_weight)
        xfa["xfa_strong_aux_cap"] = float(config.strong_aux_cap)
        xfa["xfa_same_subset"] = "true"
        xfa["xfa_label_authority"] = "base_qknn_only"
        xfa["bytes"] = _float(base, "bytes") + max(0.0, float(config.aux_bytes_per_receiver))
        xfa["latency_ms"] = max(_float(base, "latency_ms"), _float(aux, "latency_ms"))
        xfa["reliability_source"] = "xfa_ci_base_qknn_aux_unknown_risk"
        paired_base.append(base_out)
        xfa_rows.append(xfa)
    audit = {
        "base_row_count": len(base_rows),
        "aux_row_count": len(aux_rows),
        "matched_row_count": len(paired_base),
        "missing_aux_row_count": missing,
        "same_subset": True,
        "strong_known_row_count": strong_count,
        "aux_lift_row_count": aux_lift_count,
        "aux_lift_rate": aux_lift_count / max(len(paired_base), 1),
    }
    return paired_base, xfa_rows, audit


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metadata" in data and isinstance(data["metadata"], dict):
        return dict(data["metadata"])
    if "qknn_metadata" in data and isinstance(data["qknn_metadata"], dict):
        return dict(data["qknn_metadata"])
    return dict(data)


def _parse_csv_floats(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty numeric list {spec!r}")
    return values


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            out[key] = value
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


def _iter_aux_specs(values: Iterable[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for raw in values:
        if "=" in raw:
            name, path = raw.split("=", 1)
            specs.append((name.strip(), Path(path.strip())))
        else:
            path = Path(raw)
            specs.append((path.stem, path))
    if not specs:
        raise ValueError("at least one --aux_evidence_csv is required")
    return specs


def _flatten_counts(
    *,
    algorithm: str,
    aux_name: str,
    policy: OpuPolicy,
    metrics: Mapping[str, Any],
    audit: Mapping[str, Any],
    config: XfaConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = metrics.get("counts", {})
    for count, count_metrics in sorted(counts.items(), key=lambda item: int(item[0])):
        row = _summary_row(policy=policy, count=str(count), metrics=count_metrics)
        row.update(
            {
                "algorithm": algorithm,
                "aux_name": aux_name,
                "same_subset": str(bool(audit["same_subset"])).lower(),
                "sample_count_matched": int(audit["matched_row_count"]),
                "missing_aux_row_count": int(audit["missing_aux_row_count"]),
                "aux_lift_rate": float(audit["aux_lift_rate"]),
                "strong_known_row_count": int(audit["strong_known_row_count"]),
                "aux_weight": float(config.aux_weight),
                "strong_aux_cap": float(config.strong_aux_cap),
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
        rows.append(row)
    return rows


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_evidence_csv", type=Path, required=True)
    parser.add_argument("--aux_evidence_csv", action="append", default=[])
    parser.add_argument("--metadata_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--policies", default="opu_old_preserve,opu_old_guarded")
    parser.add_argument("--aux_weight", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--strong_aux_cap", default="0.15,0.25,0.35")
    parser.add_argument("--strong_score", type=float, default=0.55)
    parser.add_argument("--strong_margin", type=float, default=0.08)
    parser.add_argument("--strong_support_density", type=float, default=0.35)
    parser.add_argument("--strong_reliability", type=float, default=0.60)
    parser.add_argument("--weak_score_anchor", type=float, default=0.70)
    parser.add_argument("--weak_margin_anchor", type=float, default=0.25)
    parser.add_argument("--weak_support_anchor", type=float, default=0.45)
    parser.add_argument("--weak_reliability_anchor", type=float, default=0.70)
    parser.add_argument("--aux_bytes_per_receiver", type=float, default=16.0)
    parser.add_argument("--max_event_bytes", type=float, default=900.0)
    parser.add_argument("--max_event_latency_ms", type=float, default=20.0)
    parser.add_argument("--old_floor", type=float, default=0.80)
    parser.add_argument("--write_evidence", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv_rows(args.base_evidence_csv)
    metadata = _load_metadata(args.metadata_json)
    policies = _select_policies(args.policies)
    aux_specs = _iter_aux_specs(args.aux_evidence_csv)
    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    base_metric_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for aux_name, aux_path in aux_specs:
        aux_rows = read_csv_rows(aux_path)
        for aux_weight in _parse_csv_floats(args.aux_weight):
            for strong_aux_cap in _parse_csv_floats(args.strong_aux_cap):
                config = XfaConfig(
                    aux_weight=aux_weight,
                    strong_score=args.strong_score,
                    strong_margin=args.strong_margin,
                    strong_support_density=args.strong_support_density,
                    strong_reliability=args.strong_reliability,
                    weak_score_anchor=args.weak_score_anchor,
                    weak_margin_anchor=args.weak_margin_anchor,
                    weak_support_anchor=args.weak_support_anchor,
                    weak_reliability_anchor=args.weak_reliability_anchor,
                    strong_aux_cap=strong_aux_cap,
                    aux_bytes_per_receiver=args.aux_bytes_per_receiver,
                )
                paired_base, xfa_rows, audit = build_xfa_evidence(base_rows, aux_rows, config)
                tag = f"{aux_name}_w{aux_weight:g}_cap{strong_aux_cap:g}".replace(".", "p")
                audit = dict(audit)
                audit.update({"aux_name": aux_name, "aux_path": str(aux_path), "tag": tag})
                audits.append(audit)
                if args.write_evidence:
                    write_csv_rows(args.output_dir / f"paired_base_{tag}.csv", paired_base)
                    write_csv_rows(args.output_dir / f"xfa_evidence_{tag}.csv", xfa_rows)
                for policy in policies:
                    cache_key = (aux_name, policy.name)
                    if cache_key not in base_metric_cache:
                        base_metric_cache[cache_key] = _evaluate_policy(
                            paired_base,
                            metadata,
                            policy,
                            max_event_bytes=args.max_event_bytes,
                            max_event_latency_ms=args.max_event_latency_ms,
                        )
                    base_metrics = base_metric_cache[cache_key]
                    xfa_metrics = _evaluate_policy(
                        xfa_rows,
                        metadata,
                        policy,
                        max_event_bytes=args.max_event_bytes,
                        max_event_latency_ms=args.max_event_latency_ms,
                    )
                    base_flat = _flatten_counts(
                        algorithm="base_paired",
                        aux_name=aux_name,
                        policy=policy,
                        metrics=base_metrics,
                        audit=audit,
                        config=config,
                    )
                    xfa_flat = _flatten_counts(
                        algorithm="xfa_ci",
                        aux_name=aux_name,
                        policy=policy,
                        metrics=xfa_metrics,
                        audit=audit,
                        config=config,
                    )
                    by_base = {
                        int(row["collab_count"]): row
                        for row in base_flat
                    }
                    for row in xfa_flat:
                        base_row = by_base[int(row["collab_count"])]
                        row["delta_old_acc"] = float(row["old_acc"]) - float(base_row["old_acc"])
                        row["delta_seen_new_acc"] = float(row["seen_new_acc"]) - float(base_row["seen_new_acc"])
                        row["delta_unknown_reject_rate"] = (
                            float(row["unknown_reject_rate"]) - float(base_row["unknown_reject_rate"])
                        )
                        row["delta_unknown_FAR"] = float(row["unknown_FAR"]) - float(base_row["unknown_FAR"])
                        row["old_floor_pass"] = float(row["old_acc"]) >= float(args.old_floor)
                        row["old_not_drop_pass"] = float(row["delta_old_acc"]) >= -1e-12
                        row["verdict"] = (
                            "candidate"
                            if row["old_floor_pass"]
                            and row["old_not_drop_pass"]
                            and float(row["delta_unknown_reject_rate"]) > 0.0
                            and float(row["delta_unknown_FAR"]) <= 0.0
                            else "diagnostic_only"
                        )
                    all_rows.extend(base_flat)
                    all_rows.extend(xfa_flat)
                    best_rows.extend(xfa_flat)

    best_rows = sorted(
        best_rows,
        key=lambda row: (
            row.get("verdict") == "candidate",
            float(row.get("old_acc", 0.0)),
            float(row.get("unknown_reject_rate", 0.0)),
            -float(row.get("unknown_FAR", 1.0)),
        ),
        reverse=True,
    )
    write_csv_rows(args.output_dir / "xfa_ci_summary.csv", all_rows)
    write_csv_rows(args.output_dir / "xfa_ci_best_rows.csv", best_rows[:50])
    (args.output_dir / "xfa_ci_audit.json").write_text(
        json.dumps(
            {
                "algorithm": "XFA-CI",
                "description": __doc__,
                "base_evidence_csv": str(args.base_evidence_csv),
                "metadata_json": str(args.metadata_json),
                "audits": audits,
                "config": _jsonable_args(args),
                "candidate_count": sum(1 for row in best_rows if row.get("verdict") == "candidate"),
                "old_floor": float(args.old_floor),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
