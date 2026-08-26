"""Truth-last scoring for the JMRS02 J1 role-correct source-LORO screen."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.jmrs02_j1 import J1_ROWS


LEO_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _accuracy(values: Sequence[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def score_j1_records(predictions: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    truth_by_id = {str(row["sample_id"]): int(row["true_class"]) for row in truths}
    prediction_ids = [str(row["sample_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)) or len(truth_by_id) != len(truths):
        raise ValueError("duplicate sample IDs in J1 streams")
    if set(prediction_ids) != set(truth_by_id):
        raise ValueError("J1 prediction/truth closure mismatch")
    if any(bool(row.get("target_or_query_access", False)) for row in predictions):
        raise ValueError("target/query access marker found in J1 prediction stream")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    nuisance_prediction, nuisance_target = [], []
    for row in predictions:
        grouped[(str(row["row"]), str(row["scenario"]))].append(row)
        if row.get("nuisance_prediction") is not None:
            nuisance_prediction.append(np.asarray(row["nuisance_prediction"], dtype=np.float64))
            nuisance_target.append(np.asarray(row["nuisance_target_proxy"], dtype=np.float64))
    metrics: dict[str, dict[str, Any]] = {}
    for row_name in sorted({key[0] for key in grouped}, key=lambda x: J1_ROWS.index(x)):
        metrics[row_name] = {}
        for scenario in sorted({key[1] for key in grouped if key[0] == row_name}):
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
                "final_gain_pp": 100.0 * float(np.mean(final == truth) - np.mean(base == truth)),
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "gate_selected_rescue_count": int((selected & rescue).sum()),
                "gate_selected_harm_count": int((selected & harm).sum()),
                "gate_coverage": float(np.mean(selected)),
                "rescue_precision": float((selected & rescue).sum() / selected_count) if selected_count else None,
                "rescue_recall": float((selected & rescue).sum() / rescue.sum()) if rescue.sum() else None,
                "harm_per_1000_selected": float(1000.0 * (selected & harm).sum() / selected_count) if selected_count else None,
                "receiver_floor": min(by_receiver_final.values()),
                "base_receiver_floor": min(by_receiver_base.values()),
                "by_receiver_final": by_receiver_final,
                "by_receiver_base": by_receiver_base,
            }
    nuisance: dict[str, Any] = {}
    if nuisance_prediction:
        pred = np.stack(nuisance_prediction)
        target = np.stack(nuisance_target)
        nuisance["P0"] = {
            "count": int(pred.shape[0]),
            "mae": float(np.mean(np.abs(pred - target))),
            "zero_baseline_mae": float(np.mean(np.abs(target))),
            "mae_improvement": float(np.mean(np.abs(target)) - np.mean(np.abs(pred - target))),
            "per_target_mae": np.mean(np.abs(pred - target), axis=0).tolist(),
            "tx_residual_claim": False,
            "target_scope": "received_waveform_clean_satellite_nuisance_proxy",
        }
    decisions: dict[str, Any] = {}
    complete_base = "B0" in metrics and all(s in metrics["B0"] for s in ("clean",) + LEO_SCENARIOS)
    base_leo_mean = float(np.mean([metrics["B0"][s]["final_accuracy"] for s in LEO_SCENARIOS])) if complete_base else float("nan")
    base_clean = metrics.get("B0", {}).get("clean", {})
    for row in ("RZ0", "RZ1", "RX1", "D1P"):
        if row not in metrics or not complete_base or not all(s in metrics[row] for s in ("clean",) + LEO_SCENARIOS):
            continue
        leo_mean = float(np.mean([metrics[row][s]["final_accuracy"] for s in LEO_SCENARIOS]))
        clean_drop_pp = 100.0 * (base_clean["final_accuracy"] - metrics[row]["clean"]["final_accuracy"])
        nonzero_gate = any(metrics[row][s]["gate_coverage"] > 0.0 for s in LEO_SCENARIOS)
        floor_safe = all(
            100.0 * (metrics[row][s]["base_receiver_floor"] - metrics[row][s]["receiver_floor"]) <= 0.30 + 1e-12
            for s in ("clean",) + LEO_SCENARIOS
        )
        gain_pp = 100.0 * (leo_mean - base_leo_mean)
        passed = bool(gain_pp > 0.0 and clean_drop_pp <= 0.30 + 1e-12 and nonzero_gate and floor_safe)
        decisions[row] = {
            "leo_mean_accuracy": leo_mean,
            "leo_mean_gain_pp": gain_pp,
            "clean_drop_pp": clean_drop_pp,
            "nonzero_gate": nonzero_gate,
            "receiver_floor_safe": floor_safe,
            "passes_j1": passed,
        }
    p0_pass = bool(nuisance.get("P0", {}).get("mae_improvement", 0.0) > 0.0)
    decisions["P0"] = {"passes_nuisance_proxy": p0_pass, "tx_residual_authorized": False}
    receiver_pass = [row for row in ("RZ1", "RX1") if decisions.get(row, {}).get("passes_j1")]
    spectral_pass = bool(decisions.get("D1P", {}).get("passes_j1"))
    return {
        "metrics": metrics,
        "nuisance": nuisance,
        "decision": {
            "status": "J1_SIGNAL" if receiver_pass or spectral_pass or p0_pass else "J1_NO_SIGNAL",
            "passing_receiver_rows": receiver_pass,
            "spectral_pass": spectral_pass,
            "phase_nuisance_proxy_pass": p0_pass,
            "eligible_next_combination": "BEST_RECEIVER+D1P" if receiver_pass and spectral_pass else None,
            "direct_old_branch_joint_authorized": False,
            "target_dg_claim_authorized": False,
            "row_decisions": decisions,
        },
    }


__all__ = ["score_j1_records"]
