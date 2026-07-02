#!/usr/bin/env python
"""Evaluate phase1-only open-set rejection from frozen z_id prototypes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics  # noqa: E402
from cvsrffi.hard_gate import GateThresholds, LocalComponentHardGate  # noqa: E402
from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_str_array(value: np.ndarray, n: int) -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [str(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_features(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        features = torch.as_tensor(np.asarray(data["features"]), dtype=torch.float32)
        n = int(features.shape[0])
        def pick(key: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else default

        payload: dict[str, Any] = {
            "features": features,
            "dataset_role": _as_str_array(pick("dataset_role", np.asarray([""] * n)), n),
            "tx_ids": _as_str_array(pick("tx_ids", np.asarray([""] * n)), n),
            "rx_ids": _as_str_array(pick("rx_ids", np.asarray([""] * n)), n),
            "day_ids": _as_str_array(pick("day_ids", np.asarray([""] * n)), n),
            "sat_scenarios": _as_str_array(pick("sat_scenarios", np.asarray([""] * n)), n),
            "channel_views": _as_str_array(pick("channel_views", np.asarray([""] * n)), n),
        }
        if "manifest_json" in data.files:
            manifest_raw = data["manifest_json"]
            try:
                payload["manifest"] = json.loads(str(np.asarray(manifest_raw).item()))
            except Exception:
                payload["manifest"] = {}
        else:
            payload["manifest"] = {}
    return payload


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _class_tx_map(source_tx_ids: Sequence[str]) -> dict[int, str]:
    items = [canonical_tx_id(x) for x in source_tx_ids]
    return {i: item for i, item in enumerate(items)}


def _nearest_scores(bank: VacuumGaussianPrototypeBank, z: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    scores = torch.empty((z.size(0), max(bank.classes.keys()) + 1), dtype=torch.float32)
    scores.fill_(-1e6)
    details: list[dict[str, Any]] = []
    for i in range(z.size(0)):
        per_class: list[tuple[int, float, int]] = []
        for class_id, comps in sorted(bank.classes.items()):
            best_d = math.inf
            best_comp = -1
            for comp in comps:
                d = float(bank.angular_distance_deg(z[i], comp.mu).item())
                if d < best_d:
                    best_d = d
                    best_comp = int(comp.component_id)
            if math.isfinite(best_d):
                scores[i, int(class_id)] = -best_d
                per_class.append((int(class_id), best_d, best_comp))
        per_class.sort(key=lambda row: row[1])
        top = per_class[0]
        second = per_class[1] if len(per_class) > 1 else (-1, math.inf, -1)
        details.append(
            {
                "pred_class": int(top[0]),
                "pred_component": int(top[2]),
                "d_own_deg": float(top[1]),
                "second_class": int(second[0]),
                "d_second_deg": float(second[1]),
                "geo_margin_deg": float(second[1] - top[1]) if math.isfinite(second[1]) else math.inf,
            }
        )
    return scores, details


def _calibrate_bank(
    bank: VacuumGaussianPrototypeBank,
    z: torch.Tensor,
    tx_ids: Sequence[str],
    roles: Sequence[str],
    *,
    source_tx_ids: Sequence[str],
    calibration_roles: set[str],
    core_quantile: float,
    max_core_radius_deg: float,
    min_samples_per_component: int,
) -> tuple[VacuumGaussianPrototypeBank, dict[str, Any]]:
    class_by_tx = {canonical_tx_id(tx): int(i) for i, tx in enumerate(source_tx_ids)}
    dist_by_component: dict[tuple[int, int], list[float]] = {}
    for i in range(z.size(0)):
        if roles[i] not in calibration_roles:
            continue
        class_id = class_by_tx.get(canonical_tx_id(tx_ids[i]))
        if class_id is None or class_id not in bank.classes:
            continue
        try:
            comp = bank.nearest_own_component(z[i], class_id)
        except KeyError:
            continue
        d = float(bank.angular_distance_deg(z[i], comp.mu).item())
        dist_by_component.setdefault((int(class_id), int(comp.component_id)), []).append(d)

    classes = {}
    changed = 0
    for class_id, comps in bank.classes.items():
        out = []
        for comp in comps:
            vals = dist_by_component.get((int(class_id), int(comp.component_id)), [])
            radius = min(float(comp.r_core_deg), float(max_core_radius_deg))
            if len(vals) >= int(min_samples_per_component):
                q = float(np.quantile(np.asarray(vals, dtype=np.float32), float(core_quantile)))
                radius = min(radius, q)
            if radius < float(comp.r_core_deg):
                changed += 1
            out.append(replace(comp, r_core_deg=float(radius), r_accept_deg=float(radius)))
        classes[int(class_id)] = out
    return VacuumGaussianPrototypeBank(classes, feature_key=bank.feature_key), {
        "calibration_roles": sorted(calibration_roles),
        "core_quantile": float(core_quantile),
        "max_core_radius_deg": float(max_core_radius_deg),
        "min_samples_per_component": int(min_samples_per_component),
        "components_with_shrunk_radius": int(changed),
    }


def _is_accepted(decision: str, accept_tail_review: bool) -> bool:
    if str(decision).startswith("ACCEPT"):
        return True
    return bool(accept_tail_review) and str(decision) == "REVIEW_KNOWN_TAIL"


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_features(args.feature_npz)
    z = payload["features"]
    roles = payload["dataset_role"]
    tx_ids = payload["tx_ids"]
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define phase1 known class order")
    class_to_tx = _class_tx_map(source_tx_ids)
    tx_to_class = {v: k for k, v in class_to_tx.items()}

    bank = VacuumGaussianPrototypeBank.from_phase2_package(args.prototype_package)
    bank, calibration = _calibrate_bank(
        bank,
        z,
        tx_ids,
        roles,
        source_tx_ids=source_tx_ids,
        calibration_roles=_parse_roles(args.calibration_roles),
        core_quantile=float(args.core_quantile),
        max_core_radius_deg=float(args.max_core_radius_deg),
        min_samples_per_component=int(args.min_samples_per_component),
    )
    scores, nearest = _nearest_scores(bank, z)
    gate = LocalComponentHardGate(
        bank,
        GateThresholds(
            logit_margin_core_min=float(args.min_geo_margin_deg),
            logit_margin_tail_min=float(args.min_geo_margin_deg),
            geo_margin_core_min_deg=float(args.min_geo_margin_deg),
            geo_margin_tail_min_deg=float(args.min_geo_margin_deg),
            allow_tail_auto_accept=False,
            use_density_gate=not bool(args.disable_density_gate),
            use_nll_gate=not bool(args.disable_nll_gate),
            use_energy_gate=False,
        ),
    )
    decisions = gate.batch_decide(z, scores)

    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}

    rows: list[dict[str, Any]] = []
    y_unknown = []
    reject_scores = []
    accepted_flags = []
    known_total = known_accepted = known_correct_full = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    for i, out in enumerate(decisions):
        decision = str(out.get("decision", ""))
        accepted = _is_accepted(decision, bool(args.accept_tail_review))
        role = str(roles[i])
        tx = canonical_tx_id(tx_ids[i])
        pred_class = int(out.get("class_id", nearest[i]["pred_class"]))
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        correct = bool(is_known_query and accepted and pred_tx == tx)
        if is_known_query:
            known_total += 1
            known_accepted += int(accepted)
            known_correct_full += int(correct)
            if accepted:
                known_correct_accepted += int(pred_tx == tx)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(float(nearest[i]["d_own_deg"] - nearest[i]["geo_margin_deg"]))
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
                "pred_component": nearest[i]["pred_component"],
                "decision": decision,
                "accepted": int(accepted),
                "correct_known": int(correct),
                "d_own_deg": f"{nearest[i]['d_own_deg']:.6f}",
                "d_second_deg": f"{nearest[i]['d_second_deg']:.6f}",
                "geo_margin_deg": f"{nearest[i]['geo_margin_deg']:.6f}",
            }
        )

    metrics = {
        "phase": "phase1_only_open_set_reject",
        "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "prototype_package": str(args.prototype_package),
        "source_tx_ids": source_tx_ids,
        "known_query_roles": sorted(known_roles),
        "unknown_query_roles": sorted(unknown_roles),
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "calibration": calibration,
        "known_query_count": int(known_total),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy": _safe_rate(known_correct_full, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
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
    parser.add_argument("--prototype_package", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument("--calibration_roles", default="source")
    parser.add_argument("--core_quantile", type=float, default=0.50)
    parser.add_argument("--max_core_radius_deg", type=float, default=8.0)
    parser.add_argument("--min_geo_margin_deg", type=float, default=6.0)
    parser.add_argument("--min_samples_per_component", type=int, default=3)
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--accept_tail_review", action="store_true")
    parser.add_argument("--disable_density_gate", action="store_true")
    parser.add_argument("--disable_nll_gate", action="store_true")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
