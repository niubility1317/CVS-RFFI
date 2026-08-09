#!/usr/bin/env python
"""Final-only source-paired C/G closure for frozen P1-CP-SFCE checkpoints.

This CP-only thin evaluator reuses the already validated, pure-NumPy CB-SFCE
payload validators without changing that historical evaluator.  It owns its
own schema, immutable prior binding, and CP-specific non-compensating verdicts.
It never fits, calibrates, selects, loads checkpoint weights, or reads target
or proxy rows for pair-floor metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_phase1_cb_sfce_pair as _cb  # noqa: E402


EXPECTED_SCENARIOS = _cb.EXPECTED_SCENARIOS
EXPECTED_SOURCE_DAYS = _cb.EXPECTED_SOURCE_DAYS
EXPECTED_SOURCE_RXS = _cb.EXPECTED_SOURCE_RXS
EXPECTED_LEO_RUNTIME_VIEW = _cb.EXPECTED_LEO_RUNTIME_VIEW
FROZEN_FOLD_SOURCE_TX = _cb.FROZEN_FOLD_SOURCE_TX
METADATA_FIELDS = _cb.METADATA_FIELDS
CLASSIFICATION_METRICS = _cb.CLASSIFICATION_METRICS
FLOOR_DELTA_LIMIT_PP = _cb.FLOOR_DELTA_LIMIT_PP

EXPECTED_CLASSIFICATION_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_TRAINING_RUN_LEAF = "phase1_cp_sfce12_20260809_v2"

FROZEN_POSTFREEZE_CONTRACT = {
    "CPSFCE-POSTFREEZE-01": "final-only source diagnostics; no fit/calibration/selection",
    "CPSFCE-POSTFREEZE-02": (
        "C/G ordered clean/LEO metadata, physical keys, strict checkpoint SHA, "
        "classification-head, and immutable v2 candidate-path binding"
    ),
    "CPSFCE-POSTFREEZE-03": "single runtime LEO view plus satellite manifest profile and three disjoint scenarios",
    "CPSFCE-POSTFREEZE-04": "four source classifier floors and fixed proxy AUROC/FAR guardrail only",
    "CPSFCE-POSTFREEZE-05": "six-fold non-compensating clean, LEO, fold-equal, and 18-cell-equal gates",
}


class CPSFCEPostfreezePairError(RuntimeError):
    """Raised when a frozen CP-SFCE postfreeze evidence binding does not close."""


def _translate_cb_error(error: BaseException) -> CPSFCEPostfreezePairError:
    return CPSFCEPostfreezePairError(str(error))


def _canonical_training_root(value: str | Path) -> Path:
    """Return the only frozen CP-SFCE v2 training root.

    The postfreeze root is deliberately separate from this immutable checkpoint
    root.  Pair inputs may be under the postfreeze root, but their manifests
    and final checkpoint arguments must resolve back to this exact v2 tree.
    """

    root = Path(value).resolve()
    if root.name != EXPECTED_TRAINING_RUN_LEAF:
        raise CPSFCEPostfreezePairError(
            f"training run root leaf must be {EXPECTED_TRAINING_RUN_LEAF}: {root}"
        )
    if not root.is_dir():
        raise CPSFCEPostfreezePairError(f"training run root must already exist: {root}")
    return root


def _expected_final_checkpoint(training_root: Path, fold_index: int, arm: str) -> tuple[str, Path]:
    if arm not in {"C", "G"}:
        raise CPSFCEPostfreezePairError(f"unsupported frozen arm: {arm}")
    candidate = f"F{fold_index}{arm}_CP_SFCE12"
    return candidate, (training_root / candidate / "final_ssdg.pth").resolve()


def _require_exact_final_checkpoint(
    value: str | Path, expected: Path, *, label: str
) -> Path:
    observed = Path(value).resolve()
    if observed != expected:
        raise CPSFCEPostfreezePairError(
            f"{label} final checkpoint path does not match frozen candidate path: "
            f"expected={expected} observed={observed}"
        )
    return observed


def _validate_cp_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    """Bind a CP export/proxy manifest to its exact frozen training arm."""

    if str(manifest.get("classification_head_contract", "")) != EXPECTED_CLASSIFICATION_HEAD_CONTRACT:
        raise CPSFCEPostfreezePairError(
            f"{label} classification_head_contract must be {EXPECTED_CLASSIFICATION_HEAD_CONTRACT}"
        )
    checkpoint_value = str(manifest.get("checkpoint", "")).strip()
    if not checkpoint_value:
        raise CPSFCEPostfreezePairError(f"{label} lacks manifest checkpoint path")
    observed_checkpoint = Path(checkpoint_value).resolve()
    if observed_checkpoint != expected_checkpoint:
        raise CPSFCEPostfreezePairError(
            f"{label} manifest checkpoint path does not bind frozen {expected_candidate}: "
            f"expected={expected_checkpoint} observed={observed_checkpoint}"
        )


def _validate_cp_payload_identity(
    payload: Mapping[str, Any],
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CPSFCEPostfreezePairError(f"{label} lacks manifest")
    _validate_cp_manifest_identity(
        manifest,
        expected_checkpoint=expected_checkpoint,
        expected_candidate=expected_candidate,
        label=label,
    )


def _validate_cp_proxy_manifest_identity(
    path: str | Path,
    *,
    expected_checkpoint: Path,
    expected_candidate: str,
    label: str,
) -> None:
    """Verify the proxy's own embedded manifest before CB checks equality."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CPSFCEPostfreezePairError(f"{label} proxy diagnostic JSON is invalid") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("manifest"), Mapping):
        raise CPSFCEPostfreezePairError(f"{label} proxy diagnostic lacks manifest")
    _validate_cp_manifest_identity(
        raw["manifest"],
        expected_checkpoint=expected_checkpoint,
        expected_candidate=expected_candidate,
        label=f"{label} proxy manifest",
    )


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise CPSFCEPostfreezePairError(f"refusing to overwrite final-only pair output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise CPSFCEPostfreezePairError(f"refusing to overwrite temporary pair output: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _as_fold_index(record: Mapping[str, Any], *, label: str) -> int:
    try:
        fold_index = int(record["fold_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CPSFCEPostfreezePairError(f"{label} lacks a valid fold_index") from exc
    if fold_index not in range(1, 7):
        raise CPSFCEPostfreezePairError(f"{label} fold_index must be in [1,6]")
    return fold_index


def _load_prior_pair(
    path: str | Path,
    *,
    expected_scenarios: Sequence[str],
    source_sat_seed: int,
    matrix_id: str,
    output_root: Path,
    training_root: Path,
) -> dict[str, Any]:
    try:
        source = _cb._require_under_root(path, output_root, label="prior pair metrics JSON")
    except _cb.CBSFCEPostfreezePairError as exc:
        raise _translate_cb_error(exc) from exc
    if not source.is_file():
        raise CPSFCEPostfreezePairError(f"missing prior pair metrics JSON: {source}")
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CPSFCEPostfreezePairError(f"prior pair metrics JSON is invalid: {source}: {exc}") from exc
    if not isinstance(record, dict):
        raise CPSFCEPostfreezePairError(f"prior pair metrics JSON must encode an object: {source}")
    if record.get("schema") != "cvs.phase1.cp_sfce_postfreeze_pair.v1":
        raise CPSFCEPostfreezePairError(f"prior pair schema mismatch: {source}")
    if tuple(str(item) for item in record.get("expected_scenarios", [])) != tuple(expected_scenarios):
        raise CPSFCEPostfreezePairError(f"prior pair scenario contract mismatch: {source}")
    if int(record.get("source_sat_seed", -1)) != int(source_sat_seed):
        raise CPSFCEPostfreezePairError(f"prior pair satellite seed mismatch: {source}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise CPSFCEPostfreezePairError(f"prior pair matrix_id mismatch: {source}")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise CPSFCEPostfreezePairError(f"prior pair output root mismatch: {source}")
    if str(record.get("training_run_root", "")) != str(training_root):
        raise CPSFCEPostfreezePairError(f"prior pair training root mismatch: {source}")
    if record.get("matrix_aggregate") is not None:
        raise CPSFCEPostfreezePairError(f"prior pair must be a per-fold record, not an aggregate: {source}")
    record["_input_path"] = str(source)
    record["_input_sha256"] = _cb._sha256_file(source)
    return record


def _validate_pair_record_contract(
    record: Mapping[str, Any], *, output_root: Path, matrix_id: str, training_root: Path, label: str
) -> int:
    fold_index = _as_fold_index(record, label=label)
    if str(record.get("candidate_pair", "")) != f"F{fold_index}_C_vs_G":
        raise CPSFCEPostfreezePairError(f"{label} candidate_pair does not match frozen fold {fold_index}")
    if tuple(str(item) for item in record.get("source_tx_ids", [])) != FROZEN_FOLD_SOURCE_TX[fold_index]:
        raise CPSFCEPostfreezePairError(f"{label} source TX order does not match frozen fold {fold_index}")
    if str(record.get("postfreeze_matrix_id", "")) != str(matrix_id):
        raise CPSFCEPostfreezePairError(f"{label} matrix_id mismatch")
    if str(record.get("postfreeze_output_root", "")) != str(output_root):
        raise CPSFCEPostfreezePairError(f"{label} output root mismatch")
    if str(record.get("training_run_root", "")) != str(training_root):
        raise CPSFCEPostfreezePairError(f"{label} training root mismatch")
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CPSFCEPostfreezePairError(f"{label} lacks checkpoint bindings")
    if bindings.get("classification_head_contract") != EXPECTED_CLASSIFICATION_HEAD_CONTRACT:
        raise CPSFCEPostfreezePairError(f"{label} classification head contract mismatch")
    for arm in ("C", "G"):
        expected_candidate, expected_checkpoint = _expected_final_checkpoint(training_root, fold_index, arm)
        candidate_field = f"{arm.lower()}_candidate"
        checkpoint_field = f"{arm.lower()}_final_checkpoint_path"
        if bindings.get(candidate_field) != expected_candidate:
            raise CPSFCEPostfreezePairError(
                f"{label} {candidate_field} does not match frozen fold {fold_index}"
            )
        checkpoint_value = str(bindings.get(checkpoint_field, "")).strip()
        if not checkpoint_value or Path(checkpoint_value).resolve() != expected_checkpoint:
            raise CPSFCEPostfreezePairError(
                f"{label} {checkpoint_field} does not match frozen {expected_candidate}"
            )
    policy = record.get("policy")
    if not isinstance(policy, Mapping):
        raise CPSFCEPostfreezePairError(f"{label} lacks policy receipt")
    for field in (
        "fit_performed",
        "calibration_performed",
        "threshold_used_for_pair_metrics",
        "model_selection_performed",
        "checkpoint_weights_loaded",
    ):
        if policy.get(field) is not False:
            raise CPSFCEPostfreezePairError(f"{label} policy {field} is not strictly false")
    for field in ("proxy_rows_used_for_pair_metrics", "target_old_rows_used_for_pair_metrics"):
        if type(policy.get(field)) is not int or int(policy[field]) != 0:
            raise CPSFCEPostfreezePairError(f"{label} policy {field} is not strictly zero")
    gates = record.get("postfreeze_gates")
    if not isinstance(gates, Mapping) or not isinstance(gates.get("technical_binding"), Mapping):
        raise CPSFCEPostfreezePairError(f"{label} lacks technical binding receipt")
    if gates["technical_binding"].get("passed") is not True:
        raise CPSFCEPostfreezePairError(f"{label} technical binding is not strictly true")
    proxy = record.get("proxy_guardrail")
    if not isinstance(proxy, Mapping):
        raise CPSFCEPostfreezePairError(f"{label} lacks proxy guardrail receipt")
    try:
        c_auroc = float(proxy["C"]["AUROC_unknown"])
        g_auroc = float(proxy["G"]["AUROC_unknown"])
        c_far = float(proxy["C"]["unknown_FAR"])
        g_far = float(proxy["G"]["unknown_FAR"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CPSFCEPostfreezePairError(f"{label} proxy guardrail is malformed") from exc
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (c_auroc, g_auroc, c_far, g_far)
    ):
        raise CPSFCEPostfreezePairError(f"{label} proxy guardrail has non-finite or out-of-range value")
    auroc_ok = g_auroc >= c_auroc
    far_ok = g_far <= c_far
    if proxy.get("AUROC_unknown_non_decrease") is not auroc_ok:
        raise CPSFCEPostfreezePairError(f"{label} proxy AUROC guardrail is not strictly bound")
    if proxy.get("unknown_FAR_non_increase") is not far_ok:
        raise CPSFCEPostfreezePairError(f"{label} proxy FAR guardrail is not strictly bound")
    if proxy.get("passed") is not bool(auroc_ok and far_ok):
        raise CPSFCEPostfreezePairError(f"{label} proxy passed receipt is not strictly bound")
    return fold_index


def _record_deltas(
    record: Mapping[str, Any], scenarios: Sequence[str], *, label: str
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    try:
        clean = record["clean_source"]["G_minus_C_pp"]
        leo = record["leo_scenarios"]
    except (KeyError, TypeError) as exc:
        raise CPSFCEPostfreezePairError(f"{label} lacks classifier deltas") from exc
    clean_out: dict[str, float] = {}
    leo_out: dict[str, dict[str, float]] = {}
    for metric in CLASSIFICATION_METRICS:
        try:
            value = float(clean[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise CPSFCEPostfreezePairError(f"{label} clean delta lacks {metric}") from exc
        if not math.isfinite(value):
            raise CPSFCEPostfreezePairError(f"{label} clean delta is non-finite for {metric}")
        clean_out[metric] = value
    for scenario in scenarios:
        try:
            delta = leo[scenario]["G_minus_C_pp"]
        except (KeyError, TypeError) as exc:
            raise CPSFCEPostfreezePairError(f"{label} lacks LEO scenario {scenario}") from exc
        leo_out[scenario] = {}
        for metric in CLASSIFICATION_METRICS:
            try:
                value = float(delta[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise CPSFCEPostfreezePairError(
                    f"{label} LEO delta lacks {scenario}/{metric}"
                ) from exc
            if not math.isfinite(value):
                raise CPSFCEPostfreezePairError(f"{label} LEO delta is non-finite for {scenario}/{metric}")
            leo_out[scenario][metric] = value
    return clean_out, leo_out


def _fold_gates(
    clean_delta: Mapping[str, Any],
    leo_scenarios: Mapping[str, Mapping[str, Any]],
    proxy_guardrail: Mapping[str, Any],
    expected_scenarios: Sequence[str],
) -> dict[str, Any]:
    clean = _cb._floor_gate(clean_delta)
    scenario_floor = {
        scenario: _cb._floor_gate(leo_scenarios[scenario]["G_minus_C_pp"])
        for scenario in expected_scenarios
    }
    scenario_overall = [
        float(leo_scenarios[scenario]["G_minus_C_pp"]["overall_accuracy"])
        for scenario in expected_scenarios
    ]
    fold_equal_overall = float(np.mean(np.asarray(scenario_overall, dtype=np.float64)))
    leo_floor_passed = bool(all(gate["passed"] for gate in scenario_floor.values()))
    fold_overall_passed = fold_equal_overall >= 0.0
    passed = bool(clean["passed"] and leo_floor_passed and fold_overall_passed and proxy_guardrail["passed"])
    return {
        "technical_binding": {"passed": True},
        "clean_four_floors_ge_minus2pp": clean,
        "leo_scenario_four_floors_ge_minus2pp": {"by_scenario": scenario_floor, "passed": leo_floor_passed},
        "fold_three_scenario_equal_weight_overall_delta_pp": {
            "value": fold_equal_overall,
            "passed": fold_overall_passed,
        },
        "proxy_guardrail": dict(proxy_guardrail),
        "fold_verdict": "PENDING_GLOBAL_18_GRID" if passed else "REJECT_CP_SFCE_PERMANENT",
    }


def _matrix_aggregate(
    current: Mapping[str, Any],
    prior_paths: Sequence[str],
    *,
    expected_scenarios: Sequence[str],
    output_root: Path,
    matrix_id: str,
    training_root: Path,
) -> dict[str, Any]:
    fold_index = _validate_pair_record_contract(
        current,
        output_root=output_root,
        matrix_id=matrix_id,
        training_root=training_root,
        label="current pair",
    )
    if fold_index != 6:
        raise CPSFCEPostfreezePairError("matrix aggregate is frozen to the sixth and final pair")
    if len(prior_paths) != 5:
        raise CPSFCEPostfreezePairError("sixth pair requires exactly five prior per-fold metrics JSONs")
    records = [
        _load_prior_pair(
            path,
            expected_scenarios=expected_scenarios,
            source_sat_seed=int(current["source_sat_seed"]),
            matrix_id=matrix_id,
            output_root=output_root,
            training_root=training_root,
        )
        for path in prior_paths
    ]
    records.append(dict(current))
    fold_indices = [
        _validate_pair_record_contract(
            record,
            output_root=output_root,
            matrix_id=matrix_id,
            training_root=training_root,
            label="pair record",
        )
        for record in records
    ]
    if set(fold_indices) != set(range(1, 7)) or len(set(fold_indices)) != len(fold_indices):
        raise CPSFCEPostfreezePairError("matrix aggregate must contain exactly folds 1..6 once")
    records.sort(key=lambda record: _as_fold_index(record, label="pair record"))

    clean_passes: list[bool] = []
    leo_passes: list[bool] = []
    fold_equal_overall: dict[str, float] = {}
    technical_passes: list[bool] = []
    proxy_passes: list[bool] = []
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in CLASSIFICATION_METRICS}
    for record in records:
        clean_delta, leo_delta = _record_deltas(
            record, expected_scenarios, label=f"fold{record['fold_index']}"
        )
        clean_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in clean_delta.values()))
        values = [value for scenario in expected_scenarios for value in leo_delta[scenario].values()]
        leo_passes.append(all(value >= FLOOR_DELTA_LIMIT_PP for value in values))
        fold_value = float(np.mean([leo_delta[scenario]["overall_accuracy"] for scenario in expected_scenarios]))
        fold_equal_overall[f"F{record['fold_index']}"] = fold_value
        for scenario in expected_scenarios:
            for metric in CLASSIFICATION_METRICS:
                deltas_by_metric[metric].append(leo_delta[scenario][metric])
        technical_passes.append(record["postfreeze_gates"]["technical_binding"]["passed"] is True)
        proxy = record["proxy_guardrail"]
        proxy_passes.append(
            bool(
                float(proxy["G"]["AUROC_unknown"]) >= float(proxy["C"]["AUROC_unknown"])
                and float(proxy["G"]["unknown_FAR"]) <= float(proxy["C"]["unknown_FAR"])
            )
        )
    global_18 = {
        metric: float(np.mean(np.asarray(values, dtype=np.float64)))
        for metric, values in deltas_by_metric.items()
    }
    technical_passed = bool(all(technical_passes))
    clean_passed = bool(all(clean_passes))
    leo_passed = bool(all(leo_passes))
    fold_overall_passed = bool(all(value >= 0.0 for value in fold_equal_overall.values()))
    global_overall_passed = global_18["overall_accuracy"] >= 0.0
    proxy_passed = bool(all(proxy_passes))
    passed = bool(
        technical_passed
        and clean_passed
        and leo_passed
        and fold_overall_passed
        and global_overall_passed
        and proxy_passed
    )
    prior_bindings = [
        {"fold_index": int(record["fold_index"]), "metrics_json": record["_input_path"], "sha256": record["_input_sha256"]}
        for record in records
        if "_input_path" in record
    ]
    return {
        "fold_indices": [int(record["fold_index"]) for record in records],
        "prior_pair_metrics_bindings": prior_bindings,
        "global_18_cell_equal_weight_G_minus_C_pp": global_18,
        "gates": {
            "technical_binding": {"passed": technical_passed},
            "clean_6of6_four_floors_ge_minus2pp": {"passed": clean_passed, "by_fold": clean_passes},
            "leo_18of18_four_floors_ge_minus2pp": {"passed": leo_passed, "by_fold": leo_passes},
            "fold_three_scenario_equal_weight_overall_delta_pp": {
                "values": fold_equal_overall,
                "passed": fold_overall_passed,
            },
            "global_18_cell_equal_weight_overall_delta_pp": {
                "value": global_18["overall_accuracy"],
                "passed": global_overall_passed,
            },
            "proxy_AUROC_non_decrease_and_FAR_non_increase": {"passed": proxy_passed},
        },
        "verdict": (
            "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
            if passed
            else "REJECT_CP_SFCE_PERMANENT"
        ),
        "phase3_unknown_capability_claim": "NOT_EVALUATED",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one immutable CP C/G fold; F6 also seals the six-fold matrix."""

    try:
        source_tx_ids = _cb._parse_items(args.source_tx_ids, field="source_tx_ids")
        if len(source_tx_ids) != 4:
            raise CPSFCEPostfreezePairError("P1-CP-SFCE postfreeze is frozen to local4 source TX classes")
        fold_index = int(args.fold_index)
        if fold_index not in range(1, 7):
            raise CPSFCEPostfreezePairError("fold_index must be in [1,6]")
        if source_tx_ids != FROZEN_FOLD_SOURCE_TX[fold_index]:
            raise CPSFCEPostfreezePairError(f"source_tx_ids do not match frozen fold {fold_index}")
        if str(args.candidate_pair) != f"F{fold_index}_C_vs_G":
            raise CPSFCEPostfreezePairError(f"candidate_pair does not match frozen fold {fold_index}")
        matrix_id = str(args.postfreeze_matrix_id).strip()
        if not matrix_id:
            raise CPSFCEPostfreezePairError("postfreeze_matrix_id must be non-empty")
        output_root = _cb._canonical_root(args.postfreeze_output_root)
        training_root = _canonical_training_root(args.training_run_root)
        if training_root == output_root:
            raise CPSFCEPostfreezePairError("training run root must differ from postfreeze output root")
        c_candidate, expected_c_checkpoint = _expected_final_checkpoint(training_root, fold_index, "C")
        g_candidate, expected_g_checkpoint = _expected_final_checkpoint(training_root, fold_index, "G")
        c_final_checkpoint = _require_exact_final_checkpoint(
            args.c_final_checkpoint, expected_c_checkpoint, label="C"
        )
        g_final_checkpoint = _require_exact_final_checkpoint(
            args.g_final_checkpoint, expected_g_checkpoint, label="G"
        )
        for path, label in (
            (args.c_clean_npz, "C clean NPZ"),
            (args.g_clean_npz, "G clean NPZ"),
            (args.c_leo_npz, "C LEO NPZ"),
            (args.g_leo_npz, "G LEO NPZ"),
            (args.c_proxy_metrics_json, "C proxy metrics JSON"),
            (args.g_proxy_metrics_json, "G proxy metrics JSON"),
            (args.output_metrics_json, "pair output JSON"),
        ):
            _cb._require_under_root(path, output_root, label=label)
        expected_scenarios = _cb._parse_items(args.expected_scenarios, field="expected_scenarios")
        if expected_scenarios != EXPECTED_SCENARIOS:
            raise CPSFCEPostfreezePairError("expected_scenarios are frozen to the three leo_*_weak scenarios")
        expected_days = _cb._parse_items(args.expected_source_days, field="expected_source_days")
        expected_rxs = _cb._parse_items(args.expected_source_rxs, field="expected_source_rxs")
        if expected_days != EXPECTED_SOURCE_DAYS:
            raise CPSFCEPostfreezePairError("expected_source_days do not match the frozen WRC LEO v2 slice")
        if expected_rxs != EXPECTED_SOURCE_RXS:
            raise CPSFCEPostfreezePairError("expected_source_rxs do not match the frozen WRC LEO v2 slice")
        expected_source_count = int(args.expected_source_count)
        expected_target_old_count = int(args.expected_target_old_count)
        expected_proxy_count = int(args.expected_proxy_count)
        if min(expected_source_count, expected_target_old_count, expected_proxy_count) <= 0:
            raise CPSFCEPostfreezePairError("expected role counts must be positive")
        prior_paths = (
            _cb._parse_items(args.aggregate_prior_pair_metrics_json, field="aggregate_prior_pair_metrics_json")
            if args.aggregate_prior_pair_metrics_json
            else ()
        )
        if fold_index < 6 and prior_paths:
            raise CPSFCEPostfreezePairError("only the sixth pair may aggregate prior pair metrics")
        if fold_index == 6 and len(prior_paths) != 5:
            raise CPSFCEPostfreezePairError("sixth pair requires five prior pair metrics JSONs for the 18-cell gate")

        c_clean = _cb._load_npz(args.c_clean_npz)
        g_clean = _cb._load_npz(args.g_clean_npz)
        c_leo = _cb._load_npz(args.c_leo_npz)
        g_leo = _cb._load_npz(args.g_leo_npz)
        _validate_cp_payload_identity(
            c_clean,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C clean",
        )
        _validate_cp_payload_identity(
            c_leo,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C LEO",
        )
        _validate_cp_payload_identity(
            g_clean,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G clean",
        )
        _validate_cp_payload_identity(
            g_leo,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G LEO",
        )
        _cb._assert_pair_metadata(c_clean, g_clean, label="clean")
        _cb._assert_pair_metadata(c_leo, g_leo, label="LEO")
        if int(c_clean["features"].shape[1]) != int(c_leo["features"].shape[1]):
            raise CPSFCEPostfreezePairError("C clean/LEO z_id dimension mismatch")
        if int(g_clean["features"].shape[1]) != int(g_leo["features"].shape[1]):
            raise CPSFCEPostfreezePairError("G clean/LEO z_id dimension mismatch")
        c_clean_keys = _cb._validate_clean_payload(
            c_clean, source_tx_ids, expected_source_count, expected_target_old_count,
            expected_proxy_count, expected_days, expected_rxs, label="C clean"
        )
        g_clean_keys = _cb._validate_clean_payload(
            g_clean, source_tx_ids, expected_source_count, expected_target_old_count,
            expected_proxy_count, expected_days, expected_rxs, label="G clean"
        )
        c_leo_keys = _cb._validate_leo_payload(
            c_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days,
            expected_rxs, int(args.source_sat_seed), label="C LEO"
        )
        g_leo_keys = _cb._validate_leo_payload(
            g_leo, source_tx_ids, expected_source_count, expected_scenarios, expected_days,
            expected_rxs, int(args.source_sat_seed), label="G LEO"
        )
        if set(c_clean_keys.tolist()) != set(c_leo_keys.tolist()):
            raise CPSFCEPostfreezePairError("C clean/LEO source physical key sets differ")
        if set(g_clean_keys.tolist()) != set(g_leo_keys.tolist()):
            raise CPSFCEPostfreezePairError("G clean/LEO source physical key sets differ")
        if set(c_clean_keys.tolist()) != set(g_clean_keys.tolist()):
            raise CPSFCEPostfreezePairError("C/G clean source physical key sets differ")
        if set(c_leo_keys.tolist()) != set(g_leo_keys.tolist()):
            raise CPSFCEPostfreezePairError("C/G LEO source physical key sets differ")
        c_checkpoint_sha256 = _cb._checkpoint_sha256_from_manifest(c_clean, label="C clean")
        g_checkpoint_sha256 = _cb._checkpoint_sha256_from_manifest(g_clean, label="G clean")
        if c_checkpoint_sha256 != _cb._checkpoint_sha256_from_manifest(c_leo, label="C LEO"):
            raise CPSFCEPostfreezePairError("C clean/LEO source checkpoint SHA256 differs")
        if g_checkpoint_sha256 != _cb._checkpoint_sha256_from_manifest(g_leo, label="G LEO"):
            raise CPSFCEPostfreezePairError("G clean/LEO source checkpoint SHA256 differs")
        c_final_checkpoint_sha256 = _cb._bind_final_checkpoint(
            c_final_checkpoint, c_checkpoint_sha256, label="C"
        )
        g_final_checkpoint_sha256 = _cb._bind_final_checkpoint(
            g_final_checkpoint, g_checkpoint_sha256, label="G"
        )
        _validate_cp_proxy_manifest_identity(
            args.c_proxy_metrics_json,
            expected_checkpoint=expected_c_checkpoint,
            expected_candidate=c_candidate,
            label="C",
        )
        _validate_cp_proxy_manifest_identity(
            args.g_proxy_metrics_json,
            expected_checkpoint=expected_g_checkpoint,
            expected_candidate=g_candidate,
            label="G",
        )
        c_proxy = _cb._load_proxy_metrics(
            args.c_proxy_metrics_json, c_clean, source_tx_ids, expected_source_count,
            expected_proxy_count, label="C"
        )
        g_proxy = _cb._load_proxy_metrics(
            args.g_proxy_metrics_json, g_clean, source_tx_ids, expected_source_count,
            expected_proxy_count, label="G"
        )
        proxy_guardrail = _cb._proxy_guardrail(c_proxy, g_proxy)
        c_clean_summary = _cb._classification_summary(c_clean, _cb._source_mask(c_clean), source_tx_ids)
        g_clean_summary = _cb._classification_summary(g_clean, _cb._source_mask(g_clean), source_tx_ids)
        scenario_metrics: dict[str, Any] = {}
        for scenario in expected_scenarios:
            c_mask = np.asarray(c_leo["sat_scenarios"] == scenario, dtype=bool)
            g_mask = np.asarray(g_leo["sat_scenarios"] == scenario, dtype=bool)
            c_summary = _cb._classification_summary(c_leo, c_mask, source_tx_ids)
            g_summary = _cb._classification_summary(g_leo, g_mask, source_tx_ids)
            scenario_metrics[scenario] = {
                "C": c_summary,
                "G": g_summary,
                "G_minus_C_pp": _cb._delta_pp(c_summary, g_summary),
            }
        clean_delta = _cb._delta_pp(c_clean_summary, g_clean_summary)
        postfreeze_gates = _fold_gates(clean_delta, scenario_metrics, proxy_guardrail, expected_scenarios)
        metrics: dict[str, Any] = {
            "schema": "cvs.phase1.cp_sfce_postfreeze_pair.v1",
            "candidate_pair": str(args.candidate_pair),
            "fold_index": fold_index,
            "postfreeze_matrix_id": matrix_id,
            "postfreeze_output_root": str(output_root),
            "training_run_root": str(training_root),
            "evidence_boundary": "PHASE1_SOURCE_ONLY_FINAL_ONLY_DIAGNOSTIC",
            "frozen_contract": dict(FROZEN_POSTFREEZE_CONTRACT),
            "policy": {
                "fit_performed": False,
                "calibration_performed": False,
                "threshold_used_for_pair_metrics": False,
                "model_selection_performed": False,
                "checkpoint_weights_loaded": False,
                "proxy_rows_used_for_pair_metrics": 0,
                "target_old_rows_used_for_pair_metrics": 0,
                "proxy_guardrail_only": True,
                "proxy_guardrail_non_compensating": True,
            },
            "source_tx_ids": list(source_tx_ids),
            "expected_source_days": list(expected_days),
            "expected_source_rxs": list(expected_rxs),
            "expected_role_counts": {
                "source": expected_source_count,
                "target_old": expected_target_old_count,
                "proxy_unknown": expected_proxy_count,
            },
            "expected_scenarios": list(expected_scenarios),
            "source_sat_seed": int(args.source_sat_seed),
            "bindings": {
                "c_clean_npz_sha256": _cb._sha256_file(args.c_clean_npz),
                "g_clean_npz_sha256": _cb._sha256_file(args.g_clean_npz),
                "c_leo_npz_sha256": _cb._sha256_file(args.c_leo_npz),
                "g_leo_npz_sha256": _cb._sha256_file(args.g_leo_npz),
                "c_source_checkpoint_sha256": c_checkpoint_sha256,
                "g_source_checkpoint_sha256": g_checkpoint_sha256,
                "classification_head_contract": EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
                "c_candidate": c_candidate,
                "g_candidate": g_candidate,
                "c_final_checkpoint_path": str(c_final_checkpoint),
                "g_final_checkpoint_path": str(g_final_checkpoint),
                "c_final_checkpoint_sha256": c_final_checkpoint_sha256,
                "g_final_checkpoint_sha256": g_final_checkpoint_sha256,
                "c_proxy_metrics_json_sha256": c_proxy["sha256"],
                "g_proxy_metrics_json_sha256": g_proxy["sha256"],
                "checkpoint_weight_reading": "DISALLOWED",
            },
            "clean_source": {"C": c_clean_summary, "G": g_clean_summary, "G_minus_C_pp": clean_delta},
            "leo_scenarios": scenario_metrics,
            "proxy_guardrail": proxy_guardrail,
            "postfreeze_gates": postfreeze_gates,
            "matrix_aggregate": None,
        }
        if fold_index == 6:
            metrics["matrix_aggregate"] = _matrix_aggregate(
                metrics,
                prior_paths,
                expected_scenarios=expected_scenarios,
                output_root=output_root,
                matrix_id=matrix_id,
                training_root=training_root,
            )
        _atomic_write_json(args.output_metrics_json, metrics)
        return metrics
    except _cb.CBSFCEPostfreezePairError as exc:
        raise _translate_cb_error(exc) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-clean-npz", required=True)
    parser.add_argument("--g-clean-npz", required=True)
    parser.add_argument("--c-leo-npz", required=True)
    parser.add_argument("--g-leo-npz", required=True)
    parser.add_argument("--c-final-checkpoint", required=True)
    parser.add_argument("--g-final-checkpoint", required=True)
    parser.add_argument("--c-proxy-metrics-json", required=True)
    parser.add_argument("--g-proxy-metrics-json", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--candidate-pair", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--postfreeze-matrix-id", required=True)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--expected-scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--expected-source-days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--expected-source-rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-seed", type=int, default=7281718)
    parser.add_argument("--expected-source-count", type=int, default=1600)
    parser.add_argument("--expected-target-old-count", type=int, default=400)
    parser.add_argument("--expected-proxy-count", type=int, default=400)
    parser.add_argument("--aggregate-prior-pair-metrics-json", default="")
    parser.add_argument("--output-metrics-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    metrics = evaluate(build_parser().parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
