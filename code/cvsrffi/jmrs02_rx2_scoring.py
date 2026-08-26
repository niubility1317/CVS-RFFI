"""Truth-last scoring for the focused JMRS02 RX2 repair screen."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
LEO_SCENARIOS = SCENARIOS[1:]


def score_rx2_records(
    predictions: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    truth_by_id = {str(row["sample_id"]): int(row["true_class"]) for row in truths}
    prediction_ids = [str(row["sample_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)) or len(truth_by_id) != len(truths):
        raise ValueError("duplicate sample IDs in RX2 streams")
    if set(prediction_ids) != set(truth_by_id):
        raise ValueError("RX2 prediction/truth closure mismatch")
    if any(bool(row.get("target_or_query_access", False)) for row in predictions):
        raise ValueError("target/query access marker found in RX2 prediction stream")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["row"]), str(row["scenario"]))].append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for row_name in ("B0", "RX0", "RX2"):
        if not all((row_name, scenario) in grouped for scenario in SCENARIOS):
            continue
        metrics[row_name] = {}
        for scenario in SCENARIOS:
            rows = grouped[(row_name, scenario)]
            truth = np.asarray([truth_by_id[str(row["sample_id"])] for row in rows])
            base = np.asarray([int(row["base_predicted_class"]) for row in rows])
            candidate = np.asarray([int(row["candidate_predicted_class"]) for row in rows])
            final = np.asarray([int(row["final_predicted_class"]) for row in rows])
            selected = np.asarray([bool(row["gate_selected"]) for row in rows])
            rescue = (base != truth) & (candidate == truth)
            harm = (base == truth) & (candidate != truth)
            by_receiver_final, by_receiver_base = {}, {}
            for receiver in sorted({int(item["receiver"]) for item in rows}):
                mask = np.asarray([int(item["receiver"]) == receiver for item in rows])
                by_receiver_final[str(receiver)] = float(np.mean(final[mask] == truth[mask]))
                by_receiver_base[str(receiver)] = float(np.mean(base[mask] == truth[mask]))
            selected_count = int(selected.sum())
            metrics[row_name][scenario] = {
                "count": len(rows),
                "base_accuracy": float(np.mean(base == truth)),
                "candidate_accuracy": float(np.mean(candidate == truth)),
                "final_accuracy": float(np.mean(final == truth)),
                "gain_pp": 100.0 * float(np.mean(final == truth) - np.mean(base == truth)),
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "selected_rescue_count": int((selected & rescue).sum()),
                "selected_harm_count": int((selected & harm).sum()),
                "gate_coverage": float(np.mean(selected)),
                "rescue_per_selected": float((selected & rescue).sum() / selected_count) if selected_count else None,
                "harm_per_1000_selected": float(1000.0 * (selected & harm).sum() / selected_count) if selected_count else None,
                "receiver_floor": min(by_receiver_final.values()),
                "base_receiver_floor": min(by_receiver_base.values()),
                "by_receiver_final": by_receiver_final,
                "by_receiver_base": by_receiver_base,
            }
    decision: dict[str, Any] = {}
    complete = all(row in metrics for row in ("B0", "RX0", "RX2"))
    if complete:
        leo_mean = {
            row: float(np.mean([metrics[row][scenario]["final_accuracy"] for scenario in LEO_SCENARIOS]))
            for row in ("B0", "RX0", "RX2")
        }
        gain_vs_b0 = 100.0 * (leo_mean["RX2"] - leo_mean["B0"])
        gain_vs_rx0 = 100.0 * (leo_mean["RX2"] - leo_mean["RX0"])
        clean_drop = 100.0 * (
            metrics["B0"]["clean"]["final_accuracy"] - metrics["RX2"]["clean"]["final_accuracy"]
        )
        floor_safe = all(
            100.0 * (
                metrics["RX2"][scenario]["base_receiver_floor"]
                - metrics["RX2"][scenario]["receiver_floor"]
            ) <= 0.30 + 1e-12
            for scenario in SCENARIOS
        )
        nonzero_gate = any(metrics["RX2"][scenario]["gate_coverage"] > 0.0 for scenario in LEO_SCENARIOS)
        decision["RX2"] = {
            "leo_mean_accuracy": leo_mean["RX2"],
            "gain_vs_b0_pp": gain_vs_b0,
            "gain_vs_rx0_pp": gain_vs_rx0,
            "clean_drop_pp": clean_drop,
            "receiver_floor_safe": floor_safe,
            "nonzero_gate": nonzero_gate,
            "passes_rx2": bool(
                gain_vs_b0 > 0.0 and gain_vs_rx0 > 0.0 and clean_drop <= 0.30 + 1e-12
                and floor_safe and nonzero_gate
            ),
        }
    return {
        "metrics": metrics,
        "decision": decision,
        "target_dg_claim_authorized": False,
        "role": "source_receiver_loro_architecture_repair",
    }


__all__ = ["score_rx2_records"]
