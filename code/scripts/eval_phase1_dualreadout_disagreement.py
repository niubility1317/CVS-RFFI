#!/usr/bin/env python
"""Evaluate a frozen Phase1 dual-readout open-set rule.

The robust checkpoint supplies the registered-class prediction.  The angular
checkpoint supplies the source-calibrated confidence/margin/energy gate.  A
Jensen-Shannon disagreement gate is calibrated only on source samples that
both checkpoints classify correctly.  Held-TX rows never fit either gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval_phase1_logits_open_set_reject import (
    _class_tx_map,
    _load_npz,
    _parse_roles,
    _safe_rate,
    _source_thresholds,
    _write_score_table,
)

from cvsrffi.gate_metrics import binary_reject_metrics
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list


_META_KEYS = (
    "dataset_role",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "sat_scenarios",
    "channel_views",
)


def _verify_paired_payloads(angular: dict[str, Any], robust: dict[str, Any]) -> None:
    if tuple(angular["tx_logits"].shape) != tuple(robust["tx_logits"].shape):
        raise ValueError("angular and robust tx_logits shapes differ")
    for key in _META_KEYS:
        if list(angular[key]) != list(robust[key]):
            raise ValueError(f"angular and robust metadata differ for {key}")
    if not angular["sig_ids"] or any(not str(value).strip() for value in angular["sig_ids"]):
        raise ValueError("paired payloads require non-empty sig_ids for physical-row binding")


def _js_divergence(a_logits: torch.Tensor, b_logits: torch.Tensor) -> torch.Tensor:
    a = F.softmax(a_logits.float(), dim=1).clamp_min(1e-12)
    b = F.softmax(b_logits.float(), dim=1).clamp_min(1e-12)
    midpoint = (0.5 * (a + b)).clamp_min(1e-12)
    return 0.5 * (
        (a * (a.log() - midpoint.log())).sum(dim=1)
        + (b * (b.log() - midpoint.log())).sum(dim=1)
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    angular = _load_npz(args.angular_npz)
    robust = _load_npz(args.robust_npz)
    _verify_paired_payloads(angular, robust)

    a_logits = angular["tx_logits"]
    r_logits = robust["tx_logits"]
    roles = angular["dataset_role"]
    tx_ids = angular["tx_ids"]
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define Phase1 known class order")
    if a_logits.size(1) != len(source_tx_ids):
        raise ValueError("tx_logits class count does not match --source_tx_ids")

    calibration_roles = _parse_roles(args.calibration_roles)
    a_thresholds = _source_thresholds(
        logits=a_logits,
        roles=roles,
        tx_ids=tx_ids,
        source_tx_ids=source_tx_ids,
        calibration_roles=calibration_roles,
        conf_quantile=float(args.conf_quantile),
        margin_quantile=float(args.margin_quantile),
        energy_quantile=float(args.energy_quantile),
    )

    a_probs = F.softmax(a_logits, dim=1)
    a_conf, a_pred = a_probs.max(dim=1)
    a_top2 = torch.topk(a_logits, k=min(2, a_logits.size(1)), dim=1)
    a_margin = (
        a_top2.values[:, 0] - a_top2.values[:, 1]
        if a_top2.values.size(1) > 1
        else torch.full_like(a_conf, float("inf"))
    )
    a_energy = -torch.logsumexp(a_logits, dim=1)
    r_pred = r_logits.argmax(dim=1)
    js = _js_divergence(a_logits, r_logits)

    tx_to_class = {canonical_tx_id(tx): i for i, tx in enumerate(source_tx_ids)}
    js_calibration_mask = []
    for i, role in enumerate(roles):
        cls = tx_to_class.get(canonical_tx_id(tx_ids[i]))
        js_calibration_mask.append(
            bool(
                role in calibration_roles
                and cls is not None
                and int(a_pred[i].item()) == int(cls)
                and int(r_pred[i].item()) == int(cls)
            )
        )
    js_mask = torch.tensor(js_calibration_mask, dtype=torch.bool)
    if not bool(js_mask.any()):
        raise ValueError("no jointly correct source samples are available for disagreement calibration")
    js_max = float(np.quantile(js[js_mask].detach().cpu().numpy(), float(args.js_quantile)))

    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}
    class_to_tx = _class_tx_map(source_tx_ids)

    rows: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0

    for i in range(a_logits.size(0)):
        role = str(roles[i])
        tx = canonical_tx_id(tx_ids[i])
        pred_class = int(r_pred[i].item())
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        confidence_ok = float(a_conf[i].item()) >= a_thresholds["confidence_min"]
        margin_ok = float(a_margin[i].item()) >= a_thresholds["margin_min"]
        energy_ok = float(a_energy[i].item()) <= a_thresholds["energy_max"]
        agreement_ok = int(a_pred[i].item()) == pred_class
        disagreement_ok = float(js[i].item()) <= js_max
        accepted = bool(confidence_ok and margin_ok and energy_ok and agreement_ok and disagreement_ok)

        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and pred_tx == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            known_correct_accepted += int(accepted and closed_correct)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(float(js[i].item()))
            accepted_flags.append(accepted)
        rows.append(
            {
                "row": i,
                "role": role,
                "tx_id": tx,
                "rx_id": angular["rx_ids"][i],
                "day_id": angular["day_ids"][i],
                "channel_view": angular["channel_views"][i],
                "sat_scenario": angular["sat_scenarios"][i],
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "angular_pred_class": int(a_pred[i].item()),
                "robust_pred_class": pred_class,
                "robust_pred_tx_id": pred_tx,
                "accepted": int(accepted),
                "closed_correct_known": int(closed_correct),
                "accepted_correct_known": int(accepted and closed_correct),
                "angular_confidence": f"{float(a_conf[i].item()):.8f}",
                "angular_logit_margin": f"{float(a_margin[i].item()):.8f}",
                "angular_energy": f"{float(a_energy[i].item()):.8f}",
                "prediction_agreement": int(agreement_ok),
                "js_disagreement": f"{float(js[i].item()):.8f}",
            }
        )

    metrics: dict[str, Any] = {
        "phase": "phase1_dualreadout_disagreement_open_set_reject",
        "threshold_scope": "source_joint_correct_only_no_held_tx_tuning",
        "angular_npz": str(args.angular_npz),
        "robust_npz": str(args.robust_npz),
        "source_tx_ids": source_tx_ids,
        "known_query_roles": sorted(known_roles),
        "unknown_query_roles": sorted(unknown_roles),
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "angular_gate_calibration": a_thresholds,
        "disagreement_calibration": {
            "js_max": js_max,
            "js_quantile": float(args.js_quantile),
            "source_joint_correct_count": int(js_mask.sum().item()),
            "total_row_count": int(js_mask.numel()),
        },
        "gate_policy": {
            "registered_prediction_from": "robust_readout",
            "confidence_margin_energy_from": "angular_readout",
            "require_prediction_agreement": True,
            "require_js_within_source_quantile": True,
        },
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": _safe_rate(known_closed_correct, known_total),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy_after_reject": _safe_rate(known_correct_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "old_retention_vs_closed": None
        if known_closed_correct <= 0
        else float(known_correct_accepted) / float(known_closed_correct),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": _safe_rate(unknown_accepted, unknown_total),
        "unknown_reject_rate": None
        if unknown_total == 0
        else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "passes_unknown_far_target": None
        if unknown_total == 0
        else float(unknown_accepted) / float(unknown_total) <= float(args.unknown_far_target),
        "angular_manifest": angular.get("manifest", {}),
        "robust_manifest": robust.get("manifest", {}),
    }
    if y_unknown:
        metrics.update(
            binary_reject_metrics(
                torch.tensor(y_unknown, dtype=torch.bool),
                torch.tensor(reject_scores, dtype=torch.float32),
                torch.tensor(accepted_flags, dtype=torch.bool),
            )
        )
    if args.score_table_csv:
        _write_score_table(args.score_table_csv, rows)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angular_npz", required=True)
    parser.add_argument("--robust_npz", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--known_query_roles", default="source")
    parser.add_argument("--unknown_query_roles", default="proxy_unknown")
    parser.add_argument("--calibration_roles", default="source")
    parser.add_argument("--conf_quantile", type=float, default=0.05)
    parser.add_argument("--margin_quantile", type=float, default=0.05)
    parser.add_argument("--energy_quantile", type=float, default=0.95)
    parser.add_argument("--js_quantile", type=float, default=0.95)
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
