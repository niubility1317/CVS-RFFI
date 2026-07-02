#!/usr/bin/env python
"""Evaluate Phase1 classifier-logit open-set rejection.

This keeps the frozen Phase1 classifier top-1 label as the old-class output and
uses source-calibrated confidence/energy gates only for unknown rejection.
Unknown query samples are never used to set thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_str_array(value: np.ndarray, n: int) -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [str(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "tx_logits" not in data.files:
            raise ValueError(f"{path} does not contain tx_logits; re-export features with updated exporter")
        logits = torch.as_tensor(np.asarray(data["tx_logits"]), dtype=torch.float32)
        n = int(logits.shape[0])

        def pick(key: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else default

        manifest: dict[str, Any] = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "tx_logits": logits,
            "dataset_role": _as_str_array(pick("dataset_role", np.asarray([""] * n)), n),
            "tx_ids": _as_str_array(pick("tx_ids", np.asarray([""] * n)), n),
            "rx_ids": _as_str_array(pick("rx_ids", np.asarray([""] * n)), n),
            "day_ids": _as_str_array(pick("day_ids", np.asarray([""] * n)), n),
            "sat_scenarios": _as_str_array(pick("sat_scenarios", np.asarray([""] * n)), n),
            "channel_views": _as_str_array(pick("channel_views", np.asarray([""] * n)), n),
            "manifest": manifest,
        }


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _class_tx_map(source_tx_ids: Sequence[str]) -> dict[int, str]:
    items = [canonical_tx_id(x) for x in source_tx_ids]
    return {i: item for i, item in enumerate(items)}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _source_thresholds(
    *,
    logits: torch.Tensor,
    roles: Sequence[str],
    tx_ids: Sequence[str],
    source_tx_ids: Sequence[str],
    calibration_roles: set[str],
    conf_quantile: float,
    margin_quantile: float,
    energy_quantile: float,
) -> dict[str, Any]:
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1] if top2.values.size(1) > 1 else torch.full_like(conf, float("inf"))
    energy = -torch.logsumexp(logits, dim=1)
    tx_to_class = {canonical_tx_id(tx): i for i, tx in enumerate(source_tx_ids)}
    mask = []
    for i, role in enumerate(roles):
        cls = tx_to_class.get(canonical_tx_id(tx_ids[i]))
        mask.append(bool(role in calibration_roles and cls is not None and int(pred[i].item()) == int(cls)))
    mask_t = torch.tensor(mask, dtype=torch.bool)
    if not bool(mask_t.any()):
        raise ValueError("no correctly classified source samples are available for threshold calibration")
    conf_vals = conf[mask_t].detach().cpu().numpy()
    margin_vals = margin[mask_t].detach().cpu().numpy()
    energy_vals = energy[mask_t].detach().cpu().numpy()
    return {
        "confidence_min": float(np.quantile(conf_vals, float(conf_quantile))),
        "margin_min": float(np.quantile(margin_vals, float(margin_quantile))),
        "energy_max": float(np.quantile(energy_vals, float(energy_quantile))),
        "source_correct_count": int(mask_t.sum().item()),
        "source_threshold_count": int(mask_t.numel()),
        "conf_quantile": float(conf_quantile),
        "margin_quantile": float(margin_quantile),
        "energy_quantile": float(energy_quantile),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    logits = payload["tx_logits"]
    roles = payload["dataset_role"]
    tx_ids = payload["tx_ids"]
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define Phase1 known class order")
    class_to_tx = _class_tx_map(source_tx_ids)

    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1] if top2.values.size(1) > 1 else torch.full_like(conf, float("inf"))
    energy = -torch.logsumexp(logits, dim=1)
    th = _source_thresholds(
        logits=logits,
        roles=roles,
        tx_ids=tx_ids,
        source_tx_ids=source_tx_ids,
        calibration_roles=_parse_roles(args.calibration_roles),
        conf_quantile=float(args.conf_quantile),
        margin_quantile=float(args.margin_quantile),
        energy_quantile=float(args.energy_quantile),
    )
    use_conf_gate = not bool(getattr(args, "disable_conf_gate", False))
    use_margin_gate = not bool(getattr(args, "disable_margin_gate", False))
    use_energy_gate = not bool(getattr(args, "disable_energy_gate", False))

    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}

    rows: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    for i in range(logits.size(0)):
        role = str(roles[i])
        tx = canonical_tx_id(tx_ids[i])
        pred_class = int(pred[i].item())
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        conf_ok = float(conf[i].item()) >= th["confidence_min"]
        margin_ok = float(margin[i].item()) >= th["margin_min"]
        energy_ok = float(energy[i].item()) <= th["energy_max"]
        accepted = (
            (conf_ok or not use_conf_gate)
            and (margin_ok or not use_margin_gate)
            and (energy_ok or not use_energy_gate)
        )
        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and pred_tx == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            if accepted:
                known_correct_accepted += int(pred_tx == tx)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(float(-conf[i].item()))
            accepted_flags.append(bool(accepted))
        rows.append(
            {
                "row": i,
                "role": role,
                "tx_id": tx,
                "rx_id": payload["rx_ids"][i],
                "day_id": payload["day_ids"][i],
                "channel_view": payload["channel_views"][i],
                "sat_scenario": payload["sat_scenarios"][i],
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "pred_class": pred_class,
                "pred_tx_id": pred_tx,
                "accepted": int(accepted),
                "closed_correct_known": int(closed_correct),
                "accepted_correct_known": int(bool(accepted and closed_correct)),
                "confidence": f"{float(conf[i].item()):.8f}",
                "logit_margin": f"{float(margin[i].item()):.8f}",
                "energy": f"{float(energy[i].item()):.8f}",
            }
        )

    metrics = {
        "phase": "phase1_only_logits_open_set_reject",
        "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "source_tx_ids": source_tx_ids,
        "known_query_roles": sorted(known_roles),
        "unknown_query_roles": sorted(unknown_roles),
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "calibration": th,
        "gate_policy": {
            "use_confidence_gate": use_conf_gate,
            "use_margin_gate": use_margin_gate,
            "use_energy_gate": use_energy_gate,
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
        "unknown_reject_rate": None if unknown_total == 0 else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "passes_unknown_far_target": None
        if unknown_total == 0
        else (float(unknown_accepted) / float(unknown_total) <= float(args.unknown_far_target)),
        "manifest": payload.get("manifest", {}),
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
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _write_score_table(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument("--calibration_roles", default="source")
    parser.add_argument("--conf_quantile", type=float, default=0.05)
    parser.add_argument("--margin_quantile", type=float, default=0.05)
    parser.add_argument("--energy_quantile", type=float, default=0.95)
    parser.add_argument("--disable_conf_gate", action="store_true")
    parser.add_argument("--disable_margin_gate", action="store_true")
    parser.add_argument("--disable_energy_gate", action="store_true")
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
