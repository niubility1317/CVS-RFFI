#!/usr/bin/env python
"""Oracle unknown-holdout upper-bound diagnostic for Stage2-C features.

This is not a deployable protocol. It deliberately uses a small labeled slice
of target unknown rows as oracle negatives, then evaluates on held-out unknown
query rows. The purpose is to test whether the current ADV3B02 z_id feature
space contains a learnable known-vs-unknown direction at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    UNKNOWN_ROLE,
    _event_key,
    _scenario_of,
    _split_support_query_selected,
    _stable_score,
    load_feature_npz,
)
from phase2_scorpion_cvs_eval import _parse_ints, _parse_weighted_components, evaluate_scorpion  # noqa: E402
from phase2_support_ridge_adapter_eval import (  # noqa: E402
    _fit_ridge_head,
    _protocol_sets,
    _require_split,
    _score_ridge_head,
)
from phase2_virtual_negative_adapter_eval import _fit_binary_ridge, _score_binary_ridge  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "receiver_count",
        "event_id",
        "role",
        "true_label",
        "predicted_label",
        "accepted_label",
        "reject",
        "old_shield",
        "event_unknown_score",
        "event_unknown_risk",
        "high_risk_fraction",
        "vote_fraction",
        "local_accepts",
        "selected_receivers",
        "bytes_per_event",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("selected_receivers"), list):
                out["selected_receivers"] = "|".join(str(v) for v in out["selected_receivers"])
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def _support_known_threshold(
    support_features: np.ndarray,
    class_weights: np.ndarray,
    binary_weights: np.ndarray,
    *,
    class_temperature: float,
    boundary_temperature: float,
    quantile: float,
) -> float:
    class_probs = _score_ridge_head(support_features, class_weights, temperature=class_temperature)
    top = np.max(class_probs, axis=1)
    boundary = _score_binary_ridge(support_features, binary_weights, temperature=boundary_temperature)
    known_scores = np.sqrt(np.maximum(top, 0.0) * np.maximum(boundary, 0.0))
    if known_scores.size == 0:
        return 0.0
    return float(np.quantile(known_scores, max(0.0, min(1.0, float(quantile)))))


def build_oracle_unknown_evidence(
    payload: Mapping[str, Any],
    *,
    k_shot: int = 8,
    oracle_unknown_shot: int = 4,
    query_per_class: int = 20,
    seed: int = 4070404,
    ridge_lambda: float = 0.1,
    boundary_ridge_lambda: float = 0.1,
    class_temperature: float = 0.05,
    boundary_temperature: float = 0.25,
    support_threshold_quantile: float = 0.05,
    event_alignment_policy: str = "receiver_domain_ranked",
    support_selection_policy: str = "stable_first",
    evidence_packet_bytes: float = 128.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_labels, new_labels, unknown_labels, target_receivers, source_receivers = _protocol_sets(payload)
    alignment_policy = str(event_alignment_policy or "receiver_domain_ranked").strip().lower()
    if alignment_policy not in {"strict_event_key", "receiver_domain_ranked"}:
        raise ValueError("event_alignment_policy must be strict_event_key or receiver_domain_ranked")
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    features = np.asarray(payload["features"], dtype=np.float32)

    receiver_query: dict[str, dict[str, list[int]]] = {}
    receiver_models: dict[str, dict[str, Any]] = {}
    receiver_thresholds: dict[str, float] = {}
    receiver_support_counts: dict[str, int] = {}
    receiver_oracle_unknown_counts: dict[str, int] = {}

    start = time.perf_counter()
    for rx in target_receivers:
        support_indices: list[int] = []
        support_labels: list[str] = []
        oracle_unknown_indices: list[int] = []
        receiver_query[rx] = {"old": [], "seen_new": [], "unknown": []}
        for label in old_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_old",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            _require_split(rx, "target_old", label, support, query, k_shot=k_shot, query_per_class=query_per_class)
            support_indices.extend(support)
            support_labels.extend([label] * len(support))
            receiver_query[rx]["old"].extend(query)
        for label in new_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role="target_new",
                tx_id=label,
                rx_id=rx,
                k_shot=k_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            _require_split(rx, "target_new", label, support, query, k_shot=k_shot, query_per_class=query_per_class)
            support_indices.extend(support)
            support_labels.extend([label] * len(support))
            receiver_query[rx]["seen_new"].extend(query)
        for label in unknown_labels:
            support, query = _split_support_query_selected(
                payload,
                features=features,
                role=UNKNOWN_ROLE,
                tx_id=label,
                rx_id=rx,
                k_shot=oracle_unknown_shot,
                query_per_class=query_per_class,
                seed=seed,
                support_selection_policy=support_selection_policy,
            )
            _require_split(
                rx,
                UNKNOWN_ROLE,
                label,
                support,
                query,
                k_shot=oracle_unknown_shot,
                query_per_class=query_per_class,
            )
            oracle_unknown_indices.extend(support)
            receiver_query[rx]["unknown"].extend(query)

        support_arr = features[np.asarray(support_indices, dtype=int)]
        oracle_unknown_arr = features[np.asarray(oracle_unknown_indices, dtype=int)]
        class_labels, class_weights = _fit_ridge_head(support_arr, support_labels, ridge_lambda=ridge_lambda)
        binary_weights = _fit_binary_ridge(
            support_arr,
            oracle_unknown_arr,
            ridge_lambda=boundary_ridge_lambda,
        )
        threshold = _support_known_threshold(
            support_arr,
            class_weights,
            binary_weights,
            class_temperature=class_temperature,
            boundary_temperature=boundary_temperature,
            quantile=support_threshold_quantile,
        )
        receiver_models[rx] = {
            "class_labels": class_labels,
            "class_weights": class_weights,
            "binary_weights": binary_weights,
        }
        receiver_thresholds[rx] = threshold
        receiver_support_counts[rx] = len(support_indices)
        receiver_oracle_unknown_counts[rx] = len(oracle_unknown_indices)

    total_query_rows = sum(len(v[role]) for v in receiver_query.values() for role in ["old", "seen_new", "unknown"])
    per_row_ms = (time.perf_counter() - start) * 1000.0 / max(total_query_rows, 1)
    evidence: list[dict[str, Any]] = []
    for role_name, labels, raw_role in [
        ("old", old_labels, "target_old"),
        ("seen_new", new_labels, "target_new"),
        ("unknown", unknown_labels, UNKNOWN_ROLE),
    ]:
        for label in labels:
            by_rx_key: dict[str, dict[str, int]] = {}
            for rx in target_receivers:
                keyed: dict[str, int] = {}
                for idx in receiver_query[rx][role_name]:
                    if str(tx_ids[idx]) == str(label):
                        keyed[_event_key(payload, idx, role_name, label)] = int(idx)
                by_rx_key[rx] = keyed
            if alignment_policy == "strict_event_key":
                common = sorted(set.intersection(*(set(by_rx_key[rx]) for rx in target_receivers)))
                event_groups = [(event_id, {rx: by_rx_key[rx][event_id] for rx in target_receivers}) for event_id in common]
                row_alignment = "role_tx_day_sig_scenario"
            else:
                by_rx_scenario: dict[str, dict[str, list[int]]] = {}
                for rx in target_receivers:
                    by_rx_scenario[rx] = defaultdict(list)
                    for idx in receiver_query[rx][role_name]:
                        if str(tx_ids[idx]) == str(label):
                            by_rx_scenario[rx][_scenario_of(payload, idx)].append(int(idx))
                    for scenario in by_rx_scenario[rx]:
                        by_rx_scenario[rx][scenario] = sorted(
                            by_rx_scenario[rx][scenario],
                            key=lambda i: (
                                str(payload["day_ids"][i]),
                                str(payload["sig_ids"][i]),
                                _stable_score((rx, role_name, label, i), seed),
                            ),
                        )
                common_scenarios = sorted(set().union(*(set(by_rx_scenario[rx]) for rx in target_receivers)))
                event_groups = []
                for scenario in common_scenarios:
                    max_len = max((len(by_rx_scenario[rx].get(scenario, [])) for rx in target_receivers), default=0)
                    for event_i in range(max_len):
                        rx_to_idx = {
                            rx: by_rx_scenario[rx][scenario][event_i]
                            for rx in target_receivers
                            if event_i < len(by_rx_scenario[rx].get(scenario, []))
                        }
                        if rx_to_idx:
                            event_groups.append((f"{role_name}|{label}|{scenario}|rank{event_i:05d}", rx_to_idx))
                row_alignment = "receiver_domain_ranked_by_role_tx_scenario"
            for event_id, rx_to_idx in event_groups:
                for rx in sorted(rx_to_idx):
                    idx = rx_to_idx[rx]
                    model = receiver_models[rx]
                    class_probs = _score_ridge_head(features[[idx]], model["class_weights"], temperature=class_temperature)[0]
                    boundary_known = float(
                        _score_binary_ridge(features[[idx]], model["binary_weights"], temperature=boundary_temperature)[0]
                    )
                    order = np.argsort(-class_probs)
                    top_i = int(order[0])
                    second_i = int(order[1]) if len(order) > 1 else top_i
                    top_score = float(class_probs[top_i])
                    second_score = float(class_probs[second_i]) if second_i != top_i else 0.0
                    margin = float(top_score - second_score)
                    known_score = float(np.sqrt(max(top_score, 0.0) * max(boundary_known, 0.0)))
                    threshold = float(receiver_thresholds[rx])
                    threshold_risk = 0.0 if threshold <= 0.0 else max(0.0, min(1.0, (threshold - known_score) / threshold))
                    boundary_risk = max(0.0, min(1.0, 1.0 - boundary_known))
                    margin_risk = max(0.0, min(1.0, 1.0 - margin))
                    unknown_risk = float(max(threshold_risk, boundary_risk))
                    evidence.append(
                        {
                            "event_id": event_id,
                            "receiver_id": rx,
                            "role": role_name,
                            "true_label": "__unknown__" if role_name == "unknown" else str(tx_ids[idx]),
                            "predicted_label": str(model["class_labels"][top_i]),
                            "second_label": str(model["class_labels"][second_i]) if second_i != top_i else "",
                            "second_score": second_score,
                            "label_score_gap": margin,
                            "known_score": known_score,
                            "known_margin": margin,
                            "boundary_known_probability": boundary_known,
                            "effective_score_threshold": threshold,
                            "receiver_score_threshold": threshold,
                            "unknown_risk": unknown_risk,
                            "score_risk": threshold_risk,
                            "radius_risk": threshold_risk,
                            "margin_risk": margin_risk,
                            "mahalanobis_risk": 0.0,
                            "evt_risk": 0.0,
                            "oldness_risk": 0.0,
                            "virtual_unknown_risk": boundary_risk,
                            "class_negative_risk": boundary_risk,
                            "class_shell_risk": boundary_risk,
                            "reliability": float(max(0.0, min(1.0, 1.0 - boundary_risk))),
                            "receiver_deployment_prior": 1.0,
                            "receiver_class_reliability": 1.0,
                            "support_neighbor_count": int(receiver_support_counts[rx]),
                            "support_density": 1.0,
                            "oracle_unknown_support_count": int(receiver_oracle_unknown_counts[rx]),
                            "latency_ms": float(per_row_ms),
                            "bytes": float(evidence_packet_bytes),
                            "threshold_selection_label_scope": "oracle_unknown_holdout",
                            "calibration_role": "query",
                            "sat_scenario": _scenario_of(payload, idx),
                            "raw_role": raw_role,
                            "event_alignment": row_alignment,
                        }
                    )
    state_size_bytes = int(
        sum(model["class_weights"].nbytes + model["binary_weights"].nbytes for model in receiver_models.values())
    )
    metadata = {
        "source_receiver_ids": source_receivers,
        "target_receiver_ids": target_receivers,
        "old_tx_ids": old_labels,
        "seen_new_tx_ids": new_labels,
        "unknown_tx_ids": unknown_labels,
        "target_channel_view": ",".join(sorted({row["sat_scenario"] for row in evidence if row["sat_scenario"]})),
        "k_shot": int(k_shot),
        "oracle_unknown_shot": int(oracle_unknown_shot),
        "query_per_class": int(query_per_class),
        "adapter_type": "oracle_unknown_holdout_boundary",
        "adapter_update_scope": "target_old_seen_new_support_plus_oracle_unknown_support",
        "labeled_unknown_support_used_for_boundary_fit": True,
        "unknown_query_eval_only": True,
        "threshold_scope": "oracle_unknown_holdout",
        "non_deployment_diagnostic": True,
        "not_deployable_reason": "uses labeled target unknown support to fit known-vs-unknown boundary",
        "state_size_bytes": state_size_bytes,
        "evidence_bytes_per_receiver_event": float(evidence_packet_bytes),
        "event_alignment": row_alignment,
        "event_alignment_policy": alignment_policy,
        "receiver_support_counts": receiver_support_counts,
        "receiver_oracle_unknown_counts": receiver_oracle_unknown_counts,
        "receiver_score_thresholds": receiver_thresholds,
    }
    return evidence, metadata


def run_oracle_unknown_holdout(
    *,
    feature_npz: Path,
    collab_counts: Sequence[int] | None,
    k_shot: int,
    oracle_unknown_shot: int,
    query_per_class: int,
    seed: int,
    ridge_lambda: float,
    boundary_ridge_lambda: float,
    class_temperature: float,
    boundary_temperature: float,
    support_threshold_quantile: float,
    event_alignment_policy: str,
    support_selection_policy: str,
    evidence_packet_bytes: float,
    risk_components: Sequence[tuple[str, float]],
    unknown_gate: float,
    old_shield_gate: float,
) -> dict[str, Any]:
    evidence, metadata = build_oracle_unknown_evidence(
        load_feature_npz(feature_npz),
        k_shot=k_shot,
        oracle_unknown_shot=oracle_unknown_shot,
        query_per_class=query_per_class,
        seed=seed,
        ridge_lambda=ridge_lambda,
        boundary_ridge_lambda=boundary_ridge_lambda,
        class_temperature=class_temperature,
        boundary_temperature=boundary_temperature,
        support_threshold_quantile=support_threshold_quantile,
        event_alignment_policy=event_alignment_policy,
        support_selection_policy=support_selection_policy,
        evidence_packet_bytes=evidence_packet_bytes,
    )
    result = evaluate_scorpion(
        evidence,
        collab_counts=collab_counts,
        risk_components=risk_components,
        unknown_gate=unknown_gate,
        old_shield_gate=old_shield_gate,
        min_margin=0.0,
        min_pvalue=0.0,
        min_quality=0.0,
        evidence_packet_bytes=evidence_packet_bytes,
    )
    result["algorithm"] = "SCORPION-CVS-oracle-unknown-holdout-upper-bound"
    result["oracle_metadata"] = metadata
    result["non_deployment_diagnostic"] = True
    result["labeled_unknown_support_used_for_boundary_fit"] = True
    result["unknown_query_used_for_threshold_fit"] = False
    result["_evidence_rows"] = evidence
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_rows_csv", type=Path)
    parser.add_argument("--output_evidence_csv", type=Path)
    parser.add_argument("--collab_counts", default="all")
    parser.add_argument("--k_shot", type=_positive_int, default=8)
    parser.add_argument("--oracle_unknown_shot", type=_positive_int, default=4)
    parser.add_argument("--query_per_class", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=4070404)
    parser.add_argument("--ridge_lambda", type=float, default=0.1)
    parser.add_argument("--boundary_ridge_lambda", type=float, default=0.1)
    parser.add_argument("--class_temperature", type=float, default=0.05)
    parser.add_argument("--boundary_temperature", type=float, default=0.25)
    parser.add_argument("--support_threshold_quantile", type=float, default=0.05)
    parser.add_argument("--event_alignment_policy", choices=["strict_event_key", "receiver_domain_ranked"], default="receiver_domain_ranked")
    parser.add_argument("--support_selection_policy", choices=["stable_first", "centroid", "scenario_diverse"], default="stable_first")
    parser.add_argument("--evidence_packet_bytes", type=float, default=128.0)
    parser.add_argument(
        "--risk_components",
        default="virtual_unknown_risk:0.55,class_negative_risk:0.25,score_risk:0.10,margin_risk:0.10",
    )
    parser.add_argument("--unknown_gate", type=float, default=0.52)
    parser.add_argument("--old_shield_gate", type=float, default=0.68)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_oracle_unknown_holdout(
        feature_npz=args.feature_npz,
        collab_counts=_parse_ints(args.collab_counts),
        k_shot=int(args.k_shot),
        oracle_unknown_shot=int(args.oracle_unknown_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        boundary_ridge_lambda=float(args.boundary_ridge_lambda),
        class_temperature=float(args.class_temperature),
        boundary_temperature=float(args.boundary_temperature),
        support_threshold_quantile=float(args.support_threshold_quantile),
        event_alignment_policy=str(args.event_alignment_policy),
        support_selection_policy=str(args.support_selection_policy),
        evidence_packet_bytes=float(args.evidence_packet_bytes),
        risk_components=_parse_weighted_components(args.risk_components),
        unknown_gate=float(args.unknown_gate),
        old_shield_gate=float(args.old_shield_gate),
    )
    evidence_rows = result.pop("_evidence_rows")
    result["feature_npz"] = str(args.feature_npz)
    result["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    result["python_executable"] = sys.executable
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_rows_csv:
        _write_rows_csv(args.output_rows_csv, result["rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
