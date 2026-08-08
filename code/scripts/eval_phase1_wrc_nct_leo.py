#!/usr/bin/env python
"""Score three frozen source-only LEO floor views with one clean WRC-NCT readout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_wrc_nct import WRCNCTError, WRCNCTRuntime  # noqa: E402
from cvsrffi.wisig_fewshot_payload import parse_tx_id_list  # noqa: E402
from eval_phase1_wrc_nct import (  # noqa: E402
    _auc,
    _load_gi_bundle,
    _load_npz,
    _min_group_rate,
    _rate,
    _sha256,
)


EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
READOUT_SCHEMA = "cvs.phase1.wrc_nct_readout.v1"


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _as_binary(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise WRCNCTError(f"clean score field {field} must be binary")


def _load_readout(path: str | Path, source_tx_ids: Sequence[str], gi_bundle_sha256: str) -> tuple[dict[str, Any], str]:
    readout_path = Path(path)
    try:
        payload = json.loads(readout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WRCNCTError(f"cannot read immutable WRC readout: {error}") from error
    if str(payload.get("schema")) != READOUT_SCHEMA or payload.get("immutable") is not True:
        raise WRCNCTError("WRC readout must be immutable and use the frozen schema")
    if list(payload.get("source_tx_ids", [])) != list(source_tx_ids):
        raise WRCNCTError("WRC readout/source TX order mismatch")
    if str(payload.get("upstream_gi_bundle_sha256", "")) != str(gi_bundle_sha256):
        raise WRCNCTError("WRC readout upstream GI SHA256 mismatch")
    tau = payload.get("tau")
    if not isinstance(tau, (int, float)) or not math.isfinite(float(tau)):
        raise WRCNCTError("WRC readout has no finite frozen tau")
    if payload.get("outer_used_for_fit_or_calibration") is not False:
        raise WRCNCTError("WRC readout outer-use receipt is not closed")
    if payload.get("new_geometry") is not False or payload.get("learning_head") is not False:
        raise WRCNCTError("LEO scorer only accepts the frozen no-head/no-new-geometry readout")
    return payload, _sha256(readout_path)


def _load_clean_membership(path: str | Path) -> tuple[dict[str, dict[str, Any]], str]:
    clean_path = Path(path)
    try:
        with clean_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "physical_id",
                "source_reference",
                "source_calibration",
                "known_evaluation",
                "closed_correct_known",
                "full_correct_known",
            }
            if reader.fieldnames is None or required.difference(reader.fieldnames):
                missing = sorted(required.difference(reader.fieldnames or []))
                raise WRCNCTError(f"clean score CSV missing fields: {','.join(missing)}")
            source: dict[str, dict[str, Any]] = {}
            for row in reader:
                physical_id = str(row["physical_id"])
                flags = {
                    "reference": _as_binary(row["source_reference"], "source_reference"),
                    "calibration": _as_binary(row["source_calibration"], "source_calibration"),
                    "evaluation": _as_binary(row["known_evaluation"], "known_evaluation"),
                }
                active = [name for name, value in flags.items() if value]
                if len(active) == 0:
                    continue
                if len(active) != 1:
                    raise WRCNCTError("clean source row has non-exclusive R/C/E flags")
                if physical_id in source:
                    raise WRCNCTError("clean source physical ID appears more than once")
                source[physical_id] = {
                    "split": active[0],
                    "closed_correct": _as_binary(row["closed_correct_known"], "closed_correct_known"),
                    "full_correct": _as_binary(row["full_correct_known"], "full_correct_known"),
                }
    except OSError as error:
        raise WRCNCTError(f"cannot read clean score CSV: {error}") from error
    if not source:
        raise WRCNCTError("clean score CSV contains no frozen source R/C/E rows")
    if not {record["split"] for record in source.values()} == {"reference", "calibration", "evaluation"}:
        raise WRCNCTError("clean score CSV does not close all frozen R/C/E partitions")
    return source, _sha256(clean_path)


def _validate_clean_split_receipt(readout: Mapping[str, Any], clean_membership: Mapping[str, Mapping[str, Any]]) -> None:
    split = readout.get("split")
    if not isinstance(split, Mapping):
        raise WRCNCTError("immutable WRC readout has no split receipt")
    expected = {
        "reference": split.get("reference_rows"),
        "calibration": split.get("calibration_rows"),
        "evaluation": split.get("evaluation_rows"),
    }
    observed = {name: sum(record.get("split") == name for record in clean_membership.values()) for name in expected}
    if any(not isinstance(value, int) or int(value) != observed[name] for name, value in expected.items()):
        raise WRCNCTError("clean R/C/E membership does not match the frozen readout receipt")


def _drop_pp(closed: float | None, full: float | None) -> float | None:
    return None if closed is None or full is None else 100.0 * (closed - full)


def _summary(
    tx_ids: np.ndarray,
    rx_ids: np.ndarray,
    day_ids: np.ndarray,
    mask: np.ndarray,
    closed: np.ndarray,
    full: np.ndarray,
    accepted: np.ndarray | None,
) -> dict[str, Any]:
    groups = {
        "overall": (None, _rate(mask, closed), _rate(mask, full)),
        "min_class": (tx_ids, _min_group_rate(tx_ids, mask, closed), _min_group_rate(tx_ids, mask, full)),
        "min_rx": (rx_ids, _min_group_rate(rx_ids, mask, closed), _min_group_rate(rx_ids, mask, full)),
        "min_day": (day_ids, _min_group_rate(day_ids, mask, closed), _min_group_rate(day_ids, mask, full)),
    }
    result: dict[str, Any] = {"count": int(mask.sum())}
    for name, (_, closed_rate, full_rate) in groups.items():
        result[f"{name}_closed_accuracy"] = closed_rate
        result[f"{name}_full_accuracy"] = full_rate
        result[f"{name}_reject_additional_drop_pp"] = _drop_pp(closed_rate, full_rate)
    result["coverage"] = None if accepted is None else _rate(mask, accepted)
    return result


def _paired_clean_drop(clean: Mapping[str, Any], leo: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        name: None
        if clean.get(f"{name}_full_accuracy") is None or leo.get(f"{name}_full_accuracy") is None
        else 100.0 * (float(clean[f"{name}_full_accuracy"]) - float(leo[f"{name}_full_accuracy"]))
        for name in ("overall", "min_class", "min_rx", "min_day")
    }


def _max_by_metric(items: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in ("overall", "min_class", "min_rx", "min_day"):
        values = [float(item[name]) for item in items if item.get(name) is not None]
        out[name] = None if not values else float(max(values))
    return out


def _runtime_parity(gi_runtime: Any, runtime: WRCNCTRuntime, features: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        _, gi_d_class, gi_ratio = gi_runtime.eval()(features)
        d1, d2, ratio, _ = runtime.eval()(features)
        z = F.normalize(features.float(), dim=1)
        manual_d_class = (1.0 - z @ runtime.prototypes.T) / (runtime.scales.unsqueeze(0) + runtime.eps)
        manual_ordered = torch.sort(manual_d_class, dim=1).values
        manual_d1 = manual_ordered[:, 0]
        manual_d2 = manual_ordered[:, 1]
        manual_ratio = manual_d1 / (manual_d2 + runtime.eps)
    gi_ordered = torch.sort(gi_d_class, dim=1).values
    receipt = {
        "gi_d1_max_abs": float(torch.max(torch.abs(d1 - gi_ordered[:, 0])).item()),
        "gi_d2_max_abs": float(torch.max(torch.abs(d2 - gi_ordered[:, 1])).item()),
        "gi_ratio_max_abs": float(torch.max(torch.abs(ratio - gi_ratio)).item()),
        "manual_d1_max_abs": float(torch.max(torch.abs(d1 - manual_d1)).item()),
        "manual_d2_max_abs": float(torch.max(torch.abs(d2 - manual_d2)).item()),
        "manual_ratio_max_abs": float(torch.max(torch.abs(ratio - manual_ratio)).item()),
    }
    if max(receipt.values()) > 1.0e-5:
        raise WRCNCTError(f"frozen GI/WRC parity exceeded: {max(receipt.values())}")
    return receipt


def _write_scores(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise WRCNCTError("cannot write empty LEO source score CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_expected_scenarios(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if parsed != EXPECTED_SCENARIOS:
        raise WRCNCTError("expected scenarios are frozen to leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    return parsed


def _run(args: argparse.Namespace) -> dict[str, Any]:
    expected_scenarios = _parse_expected_scenarios(args.expected_scenarios)
    payload = _load_npz(args.feature_npz)
    gi_runtime, _, gi_bundle_sha256 = _load_gi_bundle(args.gi_bundle)
    source_tx_ids = tuple(parse_tx_id_list(args.source_tx_ids))
    readout, readout_sha256 = _load_readout(args.wrc_readout_json, source_tx_ids, gi_bundle_sha256)
    clean_membership, clean_scores_sha256 = _load_clean_membership(args.clean_scores_csv)
    _validate_clean_split_receipt(readout, clean_membership)
    if not np.all(payload["dataset_role"] == "source"):
        raise WRCNCTError("LEO evaluator accepts source-only rows")
    if set(payload["tx_ids"].tolist()) != set(source_tx_ids):
        raise WRCNCTError("LEO source TX set does not exactly match the frozen readout")
    physical_ids = payload["physical_ids"]
    leo_physical = set(physical_ids.tolist())
    clean_physical = set(clean_membership)
    if leo_physical != clean_physical:
        raise WRCNCTError("LEO and clean source physical-ID sets differ")
    if len(leo_physical) != int(physical_ids.size):
        raise WRCNCTError("LEO physical ID appears more than once")
    scenarios = np.asarray(payload["sat_scenarios"], dtype=object)
    if set(scenarios.tolist()) != set(expected_scenarios):
        raise WRCNCTError("LEO scenario set does not exactly match the frozen three scenarios")
    scenario_masks: dict[str, np.ndarray] = {name: scenarios == name for name in expected_scenarios}
    physical_by_scenario = {name: set(physical_ids[mask].tolist()) for name, mask in scenario_masks.items()}
    for left_index, left in enumerate(expected_scenarios):
        for right in expected_scenarios[left_index + 1 :]:
            if physical_by_scenario[left] & physical_by_scenario[right]:
                raise WRCNCTError("LEO scenario physical-ID sets must be disjoint")
    split = np.asarray([clean_membership[str(value)]["split"] for value in physical_ids], dtype=object)
    reference = split == "reference"
    calibration = split == "calibration"
    evaluation = split == "evaluation"
    if bool(np.any(reference & calibration)) or bool(np.any(reference & evaluation)) or bool(np.any(calibration & evaluation)):
        raise WRCNCTError("inherited clean R/C/E membership overlaps")
    if not np.all(reference | calibration | evaluation):
        raise WRCNCTError("LEO rows are not closed by inherited clean R/C/E membership")
    source_rx_ids = set(payload["rx_ids"].tolist())
    source_tx_set = set(source_tx_ids)
    for name, mask in scenario_masks.items():
        scenario_eval = mask & evaluation
        if set(payload["tx_ids"][scenario_eval].tolist()) != source_tx_set:
            raise WRCNCTError(f"scenario {name} evaluation lacks source TX coverage")
        if set(payload["rx_ids"][scenario_eval].tolist()) != source_rx_ids:
            raise WRCNCTError(f"scenario {name} evaluation lacks source RX coverage")
    runtime = WRCNCTRuntime(gi_runtime.prototypes, gi_runtime.scales, float(readout["tau"]), eps=gi_runtime.eps).eval()
    features = torch.as_tensor(payload["features"], dtype=torch.float32)
    parity = _runtime_parity(gi_runtime, runtime, features)
    with torch.no_grad():
        d1_t, d2_t, ratio_t, accepted_t = runtime(features)
    d1 = np.asarray(d1_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    d2 = np.asarray(d2_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    ratio = np.asarray(ratio_t.detach().cpu().tolist(), dtype=np.float32).reshape(-1)
    accepted = np.asarray(accepted_t.detach().cpu().tolist(), dtype=bool).reshape(-1)
    if payload["tx_logits"].shape[1] < len(source_tx_ids):
        raise WRCNCTError("frozen tx_logits do not cover all source TX classes")
    pred_index = np.asarray(payload["tx_logits"]).argmax(axis=1)
    p_local = np.asarray(
        [source_tx_ids[index] if 0 <= int(index) < len(source_tx_ids) else str(index) for index in pred_index],
        dtype=object,
    )
    closed = np.asarray(p_local == payload["tx_ids"], dtype=bool)
    full = closed & accepted
    clean_closed = np.asarray(
        [bool(clean_membership[str(value)]["closed_correct"]) for value in physical_ids], dtype=bool
    )
    clean_full = np.asarray(
        [bool(clean_membership[str(value)]["full_correct"]) for value in physical_ids], dtype=bool
    )
    scenario_metrics: dict[str, dict[str, Any]] = {}
    paired_drops: list[dict[str, float | None]] = []
    reject_drops: list[dict[str, float | None]] = []
    for name, mask in scenario_masks.items():
        scenario_eval = mask & evaluation
        leo_summary = _summary(payload["tx_ids"], payload["rx_ids"], payload["day_ids"], scenario_eval, closed, full, accepted)
        clean_summary = _summary(
            payload["tx_ids"], payload["rx_ids"], payload["day_ids"], scenario_eval, clean_closed, clean_full, None
        )
        paired = _paired_clean_drop(clean_summary, leo_summary)
        reject = {metric: leo_summary[f"{metric}_reject_additional_drop_pp"] for metric in paired}
        scenario_metrics[name] = {
            "scenario_total_count": int(mask.sum()),
            "known_evaluation": leo_summary,
            "paired_clean_baseline": clean_summary,
            "paired_clean_full_drop_pp": paired,
            "leo_reject_additional_drop_pp": reject,
        }
        paired_drops.append(paired)
        reject_drops.append(reject)
    metrics = {
        "schema": "cvs.phase1.wrc_nct_leo_floor.v1",
        "method": "WRC-NCT-frozen-LEO-floor",
        "evidence_boundary": "PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY",
        "expected_scenarios": list(expected_scenarios),
        "source_tx_ids": list(source_tx_ids),
        "tau": float(readout["tau"]),
        "threshold_policy": readout.get("threshold_policy"),
        "bindings": {
            "feature_npz_sha256": _sha256(args.feature_npz),
            "gi_bundle_sha256": gi_bundle_sha256,
            "wrc_readout_sha256": readout_sha256,
            "clean_scores_sha256": clean_scores_sha256,
        },
        "outer_used": False,
        "calibration_performed": False,
        "new_runtime_exported": False,
        "parity": parity,
        "scenario_metrics": scenario_metrics,
        "aggregate_max_paired_clean_full_drop_pp": _max_by_metric(paired_drops),
        "aggregate_max_leo_reject_additional_drop_pp": _max_by_metric(reject_drops),
    }
    _atomic_json(args.output_metrics_json, metrics)
    rows: list[dict[str, Any]] = []
    for index in range(payload["features"].shape[0]):
        rows.append(
            {
                "row": index,
                "physical_id": physical_ids[index],
                "tx_id": payload["tx_ids"][index],
                "rx_id": payload["rx_ids"][index],
                "day_id": payload["day_ids"][index],
                "channel_view": payload["channel_views"][index],
                "sat_scenario": scenarios[index],
                "source_reference": int(reference[index]),
                "source_calibration": int(calibration[index]),
                "known_evaluation": int(evaluation[index]),
                "p_local": p_local[index],
                "d1": f"{float(d1[index]):.9g}",
                "d2": f"{float(d2[index]):.9g}",
                "nct_ratio": f"{float(ratio[index]):.9g}",
                "accepted": int(accepted[index]),
                "closed_correct_known": int(bool(evaluation[index] and closed[index])),
                "full_correct_known": int(bool(evaluation[index] and full[index])),
                "wrc_readout_sha256": readout_sha256,
                "upstream_gi_bundle_sha256": gi_bundle_sha256,
                "clean_scores_sha256": clean_scores_sha256,
                "outer_used": 0,
                "calibration_performed": 0,
            }
        )
    _write_scores(args.output_scores_csv, rows)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", required=True)
    parser.add_argument("--gi-bundle", required=True)
    parser.add_argument("--wrc-readout-json", required=True)
    parser.add_argument("--clean-scores-csv", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--expected-scenarios", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--output-scores-csv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = _run(args)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
