#!/usr/bin/env python
"""Evaluate source-calibrated anchor-view rescue after Phase1 rejection.

The input rejection score table is produced by eval_phase1_multiview_reject.py.
This script does not tune on target-old labels or target-unknown query samples:
the rescue gate is calibrated only on correctly classified source-old anchor
views from the same exported feature NPZ.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
        return [canonical_tx_id(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
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
            "eq_ids": _as_str_array(pick("eq_ids", np.asarray([""] * n)), n),
            "sig_ids": _as_str_array(pick("sig_ids", np.asarray([str(i) for i in range(n)])), n),
            "channel_views": _as_str_array(pick("channel_views", np.asarray([""] * n)), n),
            "manifest": manifest,
        }


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _class_tx_map(source_tx_ids: Sequence[str]) -> dict[int, str]:
    return {i: canonical_tx_id(tx) for i, tx in enumerate(source_tx_ids)}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _anchor_groups(payload: Mapping[str, Any], *, anchor_view: str, source_tx_ids: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
    for i, role in enumerate(payload["dataset_role"]):
        key = (
            str(role),
            canonical_tx_id(payload["tx_ids"][i]),
            canonical_tx_id(payload["rx_ids"][i]),
            canonical_tx_id(payload["day_ids"][i]),
            canonical_tx_id(payload["eq_ids"][i]),
            canonical_tx_id(payload["sig_ids"][i]),
        )
        groups[key].append(int(i))

    class_to_tx = _class_tx_map(source_tx_ids)
    out: list[dict[str, Any]] = []
    logits = payload["tx_logits"]
    for key, idx in sorted(groups.items()):
        anchor_idx = [i for i in idx if str(payload["channel_views"][i]) == str(anchor_view)]
        if not anchor_idx:
            anchor_idx = [idx[0]]
        lo = logits[torch.tensor(anchor_idx, dtype=torch.long)].float().mean(dim=0)
        prob = F.softmax(lo, dim=0)
        conf, pred = prob.max(dim=0)
        top2 = torch.topk(lo, k=min(2, lo.numel()))
        margin = top2.values[0] - top2.values[1] if top2.values.numel() > 1 else torch.tensor(float("inf"))
        energy = -torch.logsumexp(lo, dim=0)
        pred_class = int(pred.item())
        out.append(
            {
                "role": key[0],
                "tx_id": key[1],
                "rx_id": key[2],
                "day_id": key[3],
                "eq_id": key[4],
                "sig_id": key[5],
                "anchor_pred_class": pred_class,
                "anchor_pred_tx_id": class_to_tx.get(pred_class, str(pred_class)),
                "anchor_confidence": float(conf.item()),
                "anchor_margin": float(margin.item()),
                "anchor_energy": float(energy.item()),
            }
        )
    return out


def _calibrate_anchor_gate(
    groups: Sequence[Mapping[str, Any]],
    *,
    source_tx_ids: Sequence[str],
    calibration_roles: set[str],
    confidence_quantile: float,
    margin_quantile: float,
    energy_quantile: float,
) -> dict[str, Any]:
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    good = [
        g
        for g in groups
        if str(g["role"]) in calibration_roles
        and canonical_tx_id(g["tx_id"]) in source_known
        and canonical_tx_id(g["anchor_pred_tx_id"]) == canonical_tx_id(g["tx_id"])
    ]
    if not good:
        raise ValueError("no correctly classified source anchor groups available for rescue calibration")
    conf = np.asarray([float(g["anchor_confidence"]) for g in good], dtype=np.float64)
    margin = np.asarray([float(g["anchor_margin"]) for g in good], dtype=np.float64)
    energy = np.asarray([float(g["anchor_energy"]) for g in good], dtype=np.float64)
    return {
        "confidence_min": float(np.quantile(conf, float(confidence_quantile))),
        "margin_min": float(np.quantile(margin, float(margin_quantile))),
        "energy_max": float(np.quantile(energy, float(energy_quantile))),
        "source_correct_anchor_count": int(len(good)),
        "confidence_quantile": float(confidence_quantile),
        "margin_quantile": float(margin_quantile),
        "energy_quantile": float(energy_quantile),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define Phase1 known class order")
    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    calibration_roles = _parse_roles(args.calibration_roles)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}

    payload = _load_npz(args.feature_npz)
    anchor = _anchor_groups(payload, anchor_view=str(args.anchor_view), source_tx_ids=source_tx_ids)
    score_rows = _read_rows(args.reject_score_table_csv)
    if len(anchor) != len(score_rows):
        raise ValueError(f"anchor group count {len(anchor)} does not match score rows {len(score_rows)}")
    th = _calibrate_anchor_gate(
        anchor,
        source_tx_ids=source_tx_ids,
        calibration_roles=calibration_roles,
        confidence_quantile=float(args.confidence_quantile),
        margin_quantile=float(args.margin_quantile),
        energy_quantile=float(args.energy_quantile),
    )

    rows: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    rescue_known = rescue_unknown = 0

    for i, (base, row) in enumerate(zip(anchor, score_rows)):
        tx = canonical_tx_id(base["tx_id"])
        role = str(base["role"])
        table_pred = canonical_tx_id(row.get("pred_tx_id", ""))
        anchor_pred = canonical_tx_id(base["anchor_pred_tx_id"])
        table_accepted = bool(int(row.get("accepted", "0") or 0))
        conf_ok = float(base["anchor_confidence"]) >= float(th["confidence_min"])
        margin_ok = float(base["anchor_margin"]) >= float(th["margin_min"]) or bool(args.disable_margin_gate)
        energy_ok = float(base["anchor_energy"]) <= float(th["energy_max"]) or bool(args.disable_energy_gate)
        match_ok = (anchor_pred == table_pred) or not bool(args.require_anchor_table_match)
        rescue = bool((not table_accepted) and conf_ok and margin_ok and energy_ok and match_ok)
        accepted = bool(table_accepted or rescue)
        final_pred = anchor_pred if rescue else table_pred

        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and table_pred == tx)
        final_correct = bool(is_known_query and final_pred == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            known_correct_accepted += int(accepted and final_correct)
            rescue_known += int(rescue)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
            rescue_unknown += int(rescue)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(float(row.get("unknown_score", 0.0) or 0.0))
            accepted_flags.append(bool(accepted))
        rows.append(
            {
                "group": i,
                "role": role,
                "tx_id": tx,
                "rx_id": base["rx_id"],
                "day_id": base["day_id"],
                "sig_id": base["sig_id"],
                "table_pred_tx_id": table_pred,
                "anchor_pred_tx_id": anchor_pred,
                "final_pred_tx_id": final_pred,
                "table_accepted": int(table_accepted),
                "anchor_rescue": int(rescue),
                "final_accepted": int(accepted),
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "closed_correct_known": int(closed_correct),
                "final_correct_known": int(final_correct and accepted),
                "unknown_score": row.get("unknown_score", ""),
                "anchor_confidence": f"{float(base['anchor_confidence']):.8f}",
                "anchor_margin": f"{float(base['anchor_margin']):.8f}",
                "anchor_energy": f"{float(base['anchor_energy']):.8f}",
            }
        )

    known_closed_accuracy = _safe_rate(known_closed_correct, known_total)
    known_full_accuracy = _safe_rate(known_correct_accepted, known_total)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    old_drop_pp = None
    if known_closed_accuracy is not None and known_full_accuracy is not None:
        old_drop_pp = 100.0 * (float(known_closed_accuracy) - float(known_full_accuracy))
    metrics = {
        "phase": "phase1_only_anchor_rescue_after_multiview_reject",
        "threshold_scope": "source_old_correct_anchor_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "reject_score_table_csv": str(args.reject_score_table_csv),
        "source_tx_ids": source_tx_ids,
        "anchor_view": str(args.anchor_view),
        "calibration": th,
        "rescue_policy": {
            "require_anchor_table_match": bool(args.require_anchor_table_match),
            "disable_margin_gate": bool(args.disable_margin_gate),
            "disable_energy_gate": bool(args.disable_energy_gate),
        },
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": known_closed_accuracy,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy_after_reject": known_full_accuracy,
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "old_drop_pp_vs_closed": old_drop_pp,
        "max_old_drop_pp": float(args.max_old_drop_pp),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": unknown_far,
        "unknown_reject_rate": None if unknown_total == 0 else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "rescued_known_query": int(rescue_known),
        "rescued_unknown_query": int(rescue_unknown),
        "passes_unknown_far_target": None if unknown_far is None else float(unknown_far) <= float(args.unknown_far_target),
        "passes_old_drop_target": None if old_drop_pp is None else float(old_drop_pp) <= float(args.max_old_drop_pp),
        "passes_dual_target": None
        if unknown_far is None or old_drop_pp is None
        else (float(unknown_far) <= float(args.unknown_far_target) and float(old_drop_pp) <= float(args.max_old_drop_pp)),
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
        _write_rows(args.score_table_csv, rows)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--reject_score_table_csv", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument("--calibration_roles", default="source")
    parser.add_argument("--anchor_view", default="anchor_base")
    parser.add_argument("--confidence_quantile", type=float, default=0.95)
    parser.add_argument("--margin_quantile", type=float, default=0.0)
    parser.add_argument("--energy_quantile", type=float, default=1.0)
    parser.add_argument("--require_anchor_table_match", action="store_true")
    parser.add_argument("--disable_margin_gate", action="store_true")
    parser.add_argument("--disable_energy_gate", action="store_true")
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=2.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
