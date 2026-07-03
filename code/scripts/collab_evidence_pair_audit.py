from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.collaborative_open_set_qknn_eval import (  # noqa: E402
    UNKNOWN_LABEL,
    evaluate_collaborative_open_set_evidence,
)


EVAL_CONFIG_KEYS = {
    "unknown_risk_threshold",
    "accept_margin_threshold",
    "unknown_quantile",
    "fusion_policy",
    "consensus_gap_threshold",
    "consensus_score_threshold",
    "scorer_component_vote_threshold",
    "label_fusion_policy",
    "class_reliability_policy",
    "receiver_class_reliability_policy",
    "collaboration_policy",
    "latency_budget_ms",
    "max_event_bytes",
    "max_event_latency_ms",
    "adaptive_gain_min_risk",
    "adaptive_gain_latency_weight",
    "adaptive_gain_bytes_weight",
    "adaptive_gain_disagreement_weight",
    "rb_capr_utility_min_delta",
    "rb_capr_seen_new_balance_weight",
    "rb_capr_old_floor_weight",
    "rb_capr_unknown_confirm_weight",
    "rb_capr_max_avg_rx_target",
    "seen_new_rescue_enabled",
    "seen_new_rescue_risk_scale",
    "seen_new_rescue_min_score",
    "seen_new_rescue_min_margin",
    "seen_new_rescue_min_agreement",
    "conformal_rescue_enabled",
    "conformal_rescue_min_pvalue",
    "conformal_rescue_risk_scale",
    "conformal_rescue_min_agreement",
    "class_set_gate_enabled",
    "old_gate_min_receivers",
    "old_gate_max_effective_unknown_risk",
    "old_gate_max_component_agreement",
    "old_gate_min_support_density",
    "old_gate_max_radius_z",
    "seen_new_gate_min_receivers",
    "seen_new_gate_max_effective_unknown_risk",
    "seen_new_gate_max_component_agreement",
    "seen_new_gate_min_support_density",
    "seen_new_gate_max_radius_z",
    "candidate_set_min_receivers",
    "candidate_set_min_top1_receivers",
    "candidate_set_min_conformal_pvalue",
    "candidate_set_max_label_unknown_risk",
    "candidate_set_max_event_unknown_risk",
    "candidate_set_max_label_risk_component_agreement",
    "candidate_set_max_label_shell_risk",
    "candidate_set_shell_reject_risk",
    "candidate_set_event_high_unknown_risk_veto",
    "candidate_set_max_label_high_unknown_risk_fraction",
    "candidate_set_high_unknown_risk_threshold",
    "candidate_set_min_score_gap",
    "candidate_set_unknown_reject_risk",
    "threshold_selection_label_scope",
    "unknown_query_eval_only",
    "receiver_selection_policy",
}


PAIR_COLUMNS = [
    "receiver_pair",
    "total",
    "old_acc",
    "min_old_class_acc",
    "seen_new_acc",
    "min_seen_new_class_acc",
    "unknown_FAR",
    "unknown_reject_rate",
    "unknown_defer_rate",
    "known_coverage",
    "bytes_per_event",
    "latency_ms_p95",
    "mean_receiver_pair_label_disagreement",
    "mean_receiver_pair_unknown_risk_range",
    "mean_receiver_pair_score_range",
    "per_old_class_acc",
    "per_seen_new_class_acc",
    "open_set_confusion",
]


ERROR_COLUMNS = [
    "receiver_pair",
    "event_id",
    "role",
    "true_label",
    "decision",
    "output_label",
    "selected_receiver_ids",
    "selected_receiver_predictions",
    "receiver_pair_label_disagreement",
    "receiver_pair_unknown_risk_range",
    "receiver_pair_score_range",
    "unknown_risk",
    "label_unknown_risk",
    "label_shell_risk",
    "candidate_set_high_unknown_veto",
    "candidate_set_shell_veto",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_protocol_metadata(rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> dict[str, Any]:
    receivers = sorted({str(row.get("receiver_id", "")) for row in rows if row.get("receiver_id")})
    old_labels = sorted(
        {str(row.get("true_label", "")) for row in rows if str(row.get("role", "")) == "old" and row.get("true_label")}
    )
    seen_new_labels = sorted(
        {
            str(row.get("true_label", ""))
            for row in rows
            if str(row.get("role", "")) == "seen_new" and row.get("true_label")
        }
    )
    unknown_labels = sorted(
        {
            str(row.get("true_label", UNKNOWN_LABEL))
            for row in rows
            if str(row.get("role", "")) == "unknown"
        }
    ) or [UNKNOWN_LABEL]
    stage2_protocol = config.get("stage2_protocol", {}) if isinstance(config.get("stage2_protocol"), dict) else {}
    return {
        "source_receiver_ids": ["source_proxy"],
        "target_receiver_ids": receivers,
        "old_tx_ids": old_labels,
        "seen_new_tx_ids": seen_new_labels,
        "unknown_tx_ids": unknown_labels,
        "target_channel_view": stage2_protocol.get("target_channel_view", "leo_clear_weak"),
    }


def _eval_kwargs(rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = {key: config[key] for key in EVAL_CONFIG_KEYS if key in config}
    if "active_risk_components" in config:
        kwargs["scorer_risk_components"] = config["active_risk_components"]
    kwargs["collab_counts"] = "2"
    kwargs["collab_group_policy"] = "exact_k"
    kwargs["partial_collab_min_receivers"] = 2
    kwargs["include_event_results"] = True
    kwargs["protocol_metadata"] = _infer_protocol_metadata(rows, config)
    return kwargs


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_error_event(item: Mapping[str, Any]) -> bool:
    role = str(item.get("role", ""))
    decision = str(item.get("decision", ""))
    output = str(item.get("output_label", ""))
    truth = str(item.get("true_label", ""))
    if role in {"old", "seen_new"}:
        return not (decision == "accept" and output == truth)
    if role == "unknown":
        return decision != "unknown_reject"
    return True


def build_pair_audit(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    *,
    max_error_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receivers = sorted({str(row.get("receiver_id", "")) for row in rows if row.get("receiver_id")})
    eval_kwargs = _eval_kwargs(rows, config)
    pair_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for left, right in combinations(receivers, 2):
        pair = f"{left}+{right}"
        pair_input = [dict(row) for row in rows if row.get("receiver_id") in {left, right}]
        try:
            result = evaluate_collaborative_open_set_evidence(pair_input, **eval_kwargs)
            metrics = result["counts"]["2"]
        except ValueError as exc:
            pair_rows.append(
                {
                    "receiver_pair": pair,
                    "total": 0,
                    "old_acc": "",
                    "min_old_class_acc": "",
                    "seen_new_acc": "",
                    "min_seen_new_class_acc": "",
                    "unknown_FAR": "",
                    "unknown_reject_rate": "",
                    "unknown_defer_rate": "",
                    "known_coverage": "",
                    "bytes_per_event": "",
                    "latency_ms_p95": "",
                    "mean_receiver_pair_label_disagreement": "",
                    "mean_receiver_pair_unknown_risk_range": "",
                    "mean_receiver_pair_score_range": "",
                    "per_old_class_acc": "",
                    "per_seen_new_class_acc": "",
                    "open_set_confusion": f"error:{exc}",
                }
            )
            continue
        pair_rows.append(
            {
                "receiver_pair": pair,
                "total": metrics.get("total", 0),
                "old_acc": metrics.get("old_acc", 0.0),
                "min_old_class_acc": metrics.get("min_old_class_acc", 0.0),
                "seen_new_acc": metrics.get("seen_new_acc", 0.0),
                "min_seen_new_class_acc": metrics.get("min_seen_new_class_acc", 0.0),
                "unknown_FAR": metrics.get("unknown_FAR", 0.0),
                "unknown_reject_rate": metrics.get("unknown_reject_rate", 0.0),
                "unknown_defer_rate": metrics.get("unknown_defer_rate", 0.0),
                "known_coverage": metrics.get("known_coverage", 0.0),
                "bytes_per_event": metrics.get("bytes_per_event", 0.0),
                "latency_ms_p95": metrics.get("latency_ms_p95", 0.0),
                "mean_receiver_pair_label_disagreement": metrics.get(
                    "mean_receiver_pair_label_disagreement", 0.0
                ),
                "mean_receiver_pair_unknown_risk_range": metrics.get(
                    "mean_receiver_pair_unknown_risk_range", 0.0
                ),
                "mean_receiver_pair_score_range": metrics.get("mean_receiver_pair_score_range", 0.0),
                "per_old_class_acc": _json_cell(metrics.get("per_old_class_acc", {})),
                "per_seen_new_class_acc": _json_cell(metrics.get("per_seen_new_class_acc", {})),
                "open_set_confusion": _json_cell(metrics.get("open_set_confusion", {})),
            }
        )
        for item in metrics.get("event_results", []):
            if len(error_rows) >= int(max_error_rows):
                break
            if not _is_error_event(item):
                continue
            error_rows.append(
                {
                    "receiver_pair": pair,
                    "event_id": item.get("event_id", ""),
                    "role": item.get("role", ""),
                    "true_label": item.get("true_label", UNKNOWN_LABEL),
                    "decision": item.get("decision", ""),
                    "output_label": item.get("output_label", ""),
                    "selected_receiver_ids": item.get("selected_receiver_ids", ""),
                    "selected_receiver_predictions": item.get("selected_receiver_predictions", ""),
                    "receiver_pair_label_disagreement": item.get("receiver_pair_label_disagreement", 0.0),
                    "receiver_pair_unknown_risk_range": item.get("receiver_pair_unknown_risk_range", 0.0),
                    "receiver_pair_score_range": item.get("receiver_pair_score_range", 0.0),
                    "unknown_risk": item.get("unknown_risk", 0.0),
                    "label_unknown_risk": item.get("label_unknown_risk", 0.0),
                    "label_shell_risk": item.get("label_shell_risk", 0.0),
                    "candidate_set_high_unknown_veto": item.get("candidate_set_high_unknown_veto", False),
                    "candidate_set_shell_veto": item.get("candidate_set_shell_veto", False),
                }
            )
    pair_rows.sort(
        key=lambda row: (
            float(row["unknown_FAR"]) if row["unknown_FAR"] != "" else 1.0,
            -float(row["old_acc"]) if row["old_acc"] != "" else 0.0,
            -float(row["seen_new_acc"]) if row["seen_new_acc"] != "" else 0.0,
            str(row["receiver_pair"]),
        )
    )
    return pair_rows, error_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit receiver-pair collaborative open-set evidence.")
    parser.add_argument("--evidence_csv", required=True)
    parser.add_argument("--run_json")
    parser.add_argument("--output_pair_csv", required=True)
    parser.add_argument("--output_error_csv", required=True)
    parser.add_argument("--max_error_rows", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(Path(args.evidence_csv))
    config = _read_json(Path(args.run_json) if args.run_json else None)
    pair_rows, error_rows = build_pair_audit(rows, config, max_error_rows=int(args.max_error_rows))
    _write_csv(Path(args.output_pair_csv), pair_rows, PAIR_COLUMNS)
    _write_csv(Path(args.output_error_csv), error_rows, ERROR_COLUMNS)
    print(
        json.dumps(
            {
                "receiver_pair_rows": len(pair_rows),
                "error_rows": len(error_rows),
                "output_pair_csv": str(Path(args.output_pair_csv)),
                "output_error_csv": str(Path(args.output_error_csv)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
