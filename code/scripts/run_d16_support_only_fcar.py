"""Run D16-FCAR on sealed strict-K10 enrollment packages only.

This development runner intentionally reuses the verified D14 pre-open loader,
strict-K10 payload validator, physical-batch-1 feature path, and immutable
output helpers.  It has no query, label-scorer, or formal-promotion input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (CODE, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_fcar import (  # noqa: E402
    AMPLITUDE_GRID,
    FcarHyperparameters,
    _score_numpy,
    evaluate_joint_leave_two_out,
    fit_before_after_locked,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)
from run_d14_support_only_pairwise_fisher_guard import (  # noqa: E402
    _build_feature_artifact,
    _canonical,
    _feature_provenance,
    _load_enrollment,
    _member,
    _normalise,
    _payload_rows,
    _sha256_file,
    _write_json_new,
    _write_jsonl_new,
    _write_text_new,
)


class D16RunnerError(ValueError):
    """Raised when the D16 support-only development contract fails closed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _candidates() -> tuple[FcarHyperparameters, ...]:
    common = {
        "rank": 8,
        "shrink": 0.5,
        "ridge": 0.01,
        "own_llr_quantile": 0.20,
        "rest_llr_quantile": 0.80,
        "min_llr_gap": 0.0,
        "activation_threshold": 0.5,
        "operator_id": "base",
    }
    return (
        FcarHyperparameters(
            candidate_id="d16_z0_true_zero_base",
            rank=0,
            shrink=1.0,
            ridge=0.01,
            margin_band=0.0,
            force_zero=True,
        ),
        FcarHyperparameters(
            candidate_id="d16_fcar_mb002",
            margin_band=0.02,
            **common,
        ),
        FcarHyperparameters(
            candidate_id="d16_fcar_mb004",
            margin_band=0.04,
            **common,
        ),
    )


def _candidate_lock(
    candidates: tuple[FcarHyperparameters, ...],
) -> dict[str, Any]:
    rows = [
        {
            "candidate_id": value.candidate_id,
            "rank": value.rank,
            "shrink": value.shrink,
            "ridge": value.ridge,
            "own_llr_quantile": value.own_llr_quantile,
            "rest_llr_quantile": value.rest_llr_quantile,
            "min_llr_gap": value.min_llr_gap,
            "activation_threshold": value.activation_threshold,
            "margin_band": value.margin_band,
            "amplitude_grid": list(AMPLITUDE_GRID),
            "operator_id": value.operator_id,
            "force_zero": value.force_zero,
        }
        for value in candidates
    ]
    return {
        "selection_scope": (
            "one_strict_k10_base_operator_candidate_shared_by_all_three_scenarios"
        ),
        "candidate_count": len(rows),
        "candidates": rows,
        "lock_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _old_reuse_lock(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
) -> None:
    old_classes = set(before["labels"].tolist())

    def keyed(rows: Mapping[str, np.ndarray]) -> dict[tuple[str, int], tuple[str, str]]:
        return {
            (str(rows["labels"][index]), int(rows["ranks"][index])): (
                str(rows["tokens"][index]),
                str(rows["hashes"][index]),
            )
            for index in range(len(rows["labels"]))
            if str(rows["labels"][index]) in old_classes
        }

    if keyed(before) != keyed(after):
        raise D16RunnerError("before/after old physical support exact-reuse drift")


def _cross_scenario_disjointness(
    by_scenario: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    token_sets = {
        scenario: set(rows["tokens"].tolist())
        for scenario, rows in by_scenario.items()
    }
    hash_sets = {
        scenario: set(rows["hashes"].tolist())
        for scenario, rows in by_scenario.items()
    }
    pair_rows = []
    scenarios = tuple(FORMAL_LEO_WEAK_SCENARIOS)
    for left_index, left in enumerate(scenarios):
        for right in scenarios[left_index + 1 :]:
            token_overlap = sorted(token_sets[left] & token_sets[right])
            hash_overlap = sorted(hash_sets[left] & hash_sets[right])
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "physical_sample_id_overlap_count": len(token_overlap),
                    "parent_received_iq_sha256_overlap_count": len(hash_overlap),
                    "pass": not token_overlap and not hash_overlap,
                }
            )
    if not all(row["pass"] for row in pair_rows):
        raise D16RunnerError(
            "physical support or received-IQ parent reused across LEO scenarios"
        )
    return {
        "policy": "pairwise_disjoint_after_enrollment_union",
        "pairs": pair_rows,
        "all_pairwise_disjoint": True,
    }


def _all_true(values: Mapping[str, bool]) -> bool:
    return all(bool(value) for value in values.values())


def _fold_floor_gate(fold: Mapping[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    old_classes = set(fold["after_old"]["per_class_accuracy"])
    new_classes = set(fold["after_new"]["per_class_accuracy"])
    floor_handles = sorted(set(fold.get("floor_handles", ())))
    rows = {}
    strict = False
    passed = True
    for handle in floor_handles:
        if handle in old_classes:
            candidate = float(
                fold["after_old"]["per_class_accuracy"][handle]
            )
            baseline = float(
                fold["base_after_old"]["per_class_accuracy"][handle]
            )
            before = float(
                fold["before_old"]["per_class_accuracy"][handle]
            )
            nondegraded = candidate + 1.0e-12 >= max(baseline, before)
            gain = candidate > baseline + 1.0e-12
            rows[handle] = {
                "role": "old",
                "before": before,
                "z0_after": baseline,
                "candidate_after": candidate,
                "nondegraded": nondegraded,
                "strict_gain_vs_z0": gain,
            }
        elif handle in new_classes:
            candidate = float(
                fold["after_new"]["per_class_accuracy"][handle]
            )
            baseline = float(
                fold["base_after_new"]["per_class_accuracy"][handle]
            )
            nondegraded = candidate + 1.0e-12 >= baseline
            gain = candidate > baseline + 1.0e-12
            rows[handle] = {
                "role": "new",
                "z0_after": baseline,
                "candidate_after": candidate,
                "nondegraded": nondegraded,
                "strict_gain_vs_z0": gain,
            }
        else:
            nondegraded = False
            gain = False
            rows[handle] = {
                "role": "unbound",
                "nondegraded": False,
                "strict_gain_vs_z0": False,
            }
        passed = passed and nondegraded
        strict = strict or gain
    return bool(floor_handles and passed), strict, {
        "floor_handles": floor_handles,
        "per_class": rows,
        "nondegraded": bool(floor_handles and passed),
        "strict_gain_vs_z0": strict,
    }


def _scenario_gate(result: Mapping[str, Any], *, force_zero: bool) -> dict[str, Any]:
    fold_rows = []
    for fold in result["folds"]:
        vs_z0 = fold["candidate_vs_z0_per_class_non_degraded"]
        old_after_vs_before = {
            handle: (
                float(fold["after_old"]["per_class_accuracy"][handle])
                + 1.0e-12
                >= float(fold["before_old"]["per_class_accuracy"][handle])
            )
            for handle in fold["before_old"]["per_class_accuracy"]
        }
        floor_non_degraded, strict_floor_gain, floor = _fold_floor_gate(fold)
        enabled_count = int(sum(bool(value) for value in fold["enabled"]))
        gate = bool(
            not force_zero
            and _all_true(vs_z0["before_old"])
            and _all_true(vs_z0["after_old"])
            and _all_true(vs_z0["after_new"])
            and _all_true(old_after_vs_before)
            and float(fold["old_forgetting"]) <= 1.0e-12
            and float(fold["joint"]["overall_accuracy"]) + 1.0e-12
            >= float(fold["base_joint"]["overall_accuracy"])
            and float(fold["H_old_new"]) + 1.0e-12
            >= float(fold["base_H_old_new"])
            and bool(fold["old_score_bitwise_locked"])
            and bool(fold["held_disjoint_from_selection"])
            and floor_non_degraded
            and strict_floor_gain
            and enabled_count > 0
        )
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "gate_pass": gate,
                "enabled_count": enabled_count,
                "before_old_per_class_non_degraded_vs_z0": _all_true(
                    vs_z0["before_old"]
                ),
                "after_old_per_class_non_degraded_vs_z0": _all_true(
                    vs_z0["after_old"]
                ),
                "after_new_per_class_non_degraded_vs_z0": _all_true(
                    vs_z0["after_new"]
                ),
                "after_old_per_class_non_degraded_vs_same_fold_before": (
                    _all_true(old_after_vs_before)
                ),
                "old_score_bitwise_locked": bool(
                    fold["old_score_bitwise_locked"]
                ),
                "held_disjoint_from_selection": bool(
                    fold["held_disjoint_from_selection"]
                ),
                "joint_non_degraded_vs_z0": (
                    float(fold["joint"]["overall_accuracy"]) + 1.0e-12
                    >= float(fold["base_joint"]["overall_accuracy"])
                ),
                "H_non_degraded_vs_z0": (
                    float(fold["H_old_new"]) + 1.0e-12
                    >= float(fold["base_H_old_new"])
                ),
                "floor": floor,
            }
        )
    return {
        "all_folds_gate_pass": all(row["gate_pass"] for row in fold_rows),
        "strict_floor_gain_in_every_fold": all(
            row["floor"]["strict_gain_vs_z0"] for row in fold_rows
        ),
        "folds": fold_rows,
    }


def _aggregate_floor(result: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for fold in result["folds"]:
        for handle in set(fold.get("floor_handles", ())):
            counts[str(handle)] = counts.get(str(handle), 0) + 1
    old = result["after_old"]["per_class_accuracy"]
    new = result["after_new"]["per_class_accuracy"]
    old_handles = sorted(handle for handle in counts if handle in old)
    new_handles = sorted(handle for handle in counts if handle in new)

    def block(handles: list[str], key: str, base_key: str) -> dict[str, Any]:
        current = result[key]["per_class_accuracy"]
        baseline = result[base_key]["per_class_accuracy"]
        return {
            "handles": handles,
            "selection_fold_count": {handle: counts[handle] for handle in handles},
            "candidate_per_class_accuracy": {
                handle: current[handle] for handle in handles
            },
            "z0_per_class_accuracy": {
                handle: baseline[handle] for handle in handles
            },
            "candidate_min_accuracy": (
                min(current[handle] for handle in handles)
                if handles
                else None
            ),
            "z0_min_accuracy": (
                min(baseline[handle] for handle in handles)
                if handles
                else None
            ),
        }

    return {
        "old_floor": block(old_handles, "after_old", "base_after_old"),
        "new_floor": block(new_handles, "after_new", "base_after_new"),
    }


def _candidate_summary(
    candidate: FcarHyperparameters,
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenario_gates = {
        scenario: _scenario_gate(
            evaluations[scenario], force_zero=candidate.force_zero
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    floors = {
        scenario: _aggregate_floor(evaluations[scenario])
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    old_floor_values = [
        block["old_floor"]["candidate_min_accuracy"]
        for block in floors.values()
        if block["old_floor"]["candidate_min_accuracy"] is not None
    ]
    new_floor_values = [
        block["new_floor"]["candidate_min_accuracy"]
        for block in floors.values()
        if block["new_floor"]["candidate_min_accuracy"] is not None
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "force_zero": candidate.force_zero,
        "margin_band": candidate.margin_band,
        "all_scenario_all_fold_gate_pass": (
            not candidate.force_zero
            and all(
                gate["all_folds_gate_pass"]
                for gate in scenario_gates.values()
            )
        ),
        "scenario_gates": scenario_gates,
        "floors": floors,
        "worst_old_floor": min(old_floor_values) if old_floor_values else None,
        "worst_new_floor": min(new_floor_values) if new_floor_values else None,
        "mean_H_old_new": float(
            np.mean(
                [
                    evaluations[scenario]["H_old_new"]
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                ]
            )
        ),
        "mean_joint_accuracy": float(
            np.mean(
                [
                    evaluations[scenario]["joint"]["overall_accuracy"]
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                ]
            )
        ),
    }


def _select_candidate(
    rows: list[Mapping[str, Any]],
    candidates: tuple[FcarHyperparameters, ...],
) -> tuple[str, bool]:
    passing = [
        row
        for row in rows
        if bool(row["all_scenario_all_fold_gate_pass"])
        and not bool(row["force_zero"])
    ]
    passing.sort(
        key=lambda row: (
            min(
                float(row["worst_old_floor"]),
                float(row["worst_new_floor"]),
            ),
            float(row["mean_H_old_new"]),
            float(row["mean_joint_accuracy"]),
            -float(row["margin_band"]),
        ),
        reverse=True,
    )
    selected_id = (
        str(passing[0]["candidate_id"])
        if passing
        else "d16_z0_true_zero_base"
    )
    selected = next(
        value for value in candidates if value.candidate_id == selected_id
    )
    return selected_id, bool(passing and not selected.force_zero)


def _enabled_amplitudes(state) -> dict[str, Any]:
    return {
        handle: {
            "enabled": bool(state.enabled[index]),
            "a_plus": float(state.a_plus[index]),
            "a_minus": float(state.a_minus[index]),
        }
        for index, handle in enumerate(state.classes)
    }


def _measure_pareto(
    state,
    probe_feature: np.ndarray,
    support_features: np.ndarray,
    *,
    repeats: int = 200,
) -> dict[str, Any]:
    support = _normalise(support_features)
    query = _normalise(probe_feature)
    for _ in range(10):
        _score_numpy(probe_feature, state)
        _ = query @ support.T
    start = time.perf_counter()
    for _ in range(repeats):
        _score_numpy(probe_feature, state)
    fcar_ms = (time.perf_counter() - start) * 1000.0 / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        _ = query @ support.T
    qknn_ms = (time.perf_counter() - start) * 1000.0 / repeats
    prototype_macs = int(len(state.classes) * state.feature_dim)
    residual_upper = int(
        state.resource["head_scalar_ops_per_sample_upper_bound"]
    )
    fcar_upper_macs = prototype_macs + residual_upper
    qknn_macs = int(len(support_features) * state.feature_dim)
    state_bytes = int(state.resource["persistent_array_state_bytes"])
    return {
        "benchmark_input": (
            "one_enrollment_support_row_resource_probe_no_query_open"
        ),
        "repeats": repeats,
        "fcar_head_latency_ms": fcar_ms,
        "identity_single_qknn_latency_ms": qknn_ms,
        "latency_delta_percent": 100.0 * (fcar_ms / qknn_ms - 1.0),
        "fcar_prototype_plus_residual_upper_bound_macs": fcar_upper_macs,
        "identity_single_qknn_exact_macs": qknn_macs,
        "mac_delta_percent": 100.0 * (fcar_upper_macs / qknn_macs - 1.0),
        "fcar_array_state_bytes": state_bytes,
        "identity_single_qknn_state_bytes": int(support_features.nbytes),
        "state_delta_percent": (
            100.0 * (state_bytes / support_features.nbytes - 1.0)
        ),
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "dense_query_graph": False,
        "backbone_forwards_per_physical_sample": 1,
        "fft_branches_per_physical_sample": 0,
    }


def _manifest_binding(
    before_manifest: Mapping[str, Any],
    after_manifest: Mapping[str, Any],
) -> None:
    if (
        before_manifest["receiver"] != after_manifest["receiver"]
        or int(before_manifest["seed"]) != int(after_manifest["seed"])
        or before_manifest["feature_runtime_sha256"]
        != after_manifest["feature_runtime_sha256"]
        or before_manifest["phase1_checkpoint_sha256"]
        != after_manifest["phase1_checkpoint_sha256"]
    ):
        raise D16RunnerError("before/after package binding drift")
    before_handles = {
        str(row["class_handle"])
        for row in before_manifest["registered_classes"]
    }
    after_handles = {
        str(row["class_handle"])
        for row in after_manifest["registered_classes"]
    }
    if not before_handles < after_handles:
        raise D16RunnerError("new-class registration set drift")


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = "development_select",
) -> dict[str, Any]:
    if mode != "development_select":
        raise D16RunnerError("D16 runner is development_select only")
    if (
        len(expected_before_seal_sha256) != 64
        or len(expected_after_seal_sha256) != 64
    ):
        raise D16RunnerError("external expected enrollment seal SHA required")
    if output.exists():
        raise D16RunnerError("output path already exists")

    before_payloads, before_manifest, before_preopen = _load_enrollment(
        before_root,
        before_seal,
        registration_state="before",
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_payloads, after_manifest, after_preopen = _load_enrollment(
        after_root,
        after_seal,
        registration_state="after",
        expected_seal_sha256=expected_after_seal_sha256,
    )
    _manifest_binding(before_manifest, after_manifest)
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = load_torchscript_backbone_same_fd(
        before_root,
        _member(before_manifest, "feature_runtime"),
        device=device,
    )
    output.mkdir(parents=True)

    module_path = CODE / "cvsrffi" / "stage2_fcar.py"
    runner_path = Path(__file__).resolve()
    artifact_provider_path = (
        CODE / "cvsrffi" / "stage2_joint_residual_logit_head.py"
    )
    registered_feature_path = (
        CODE / "cvsrffi" / "stage2_diag_cosine_exploration.py"
    )
    d14_runner_path = (
        CODE / "scripts" / "run_d14_support_only_pairwise_fisher_guard.py"
    )
    code_hashes = {
        "d16_module_sha256": _sha256_file(module_path),
        "d16_runner_sha256": _sha256_file(runner_path),
        "reused_d14_loader_runner_sha256": _sha256_file(d14_runner_path),
        "artifact_provider_sha256": _sha256_file(artifact_provider_path),
        "registered_feature_module_sha256": _sha256_file(
            registered_feature_path
        ),
    }
    feature_code_sha256 = hashlib.sha256(
        _canonical(
            {
                **code_hashes,
                "operator_id": "base",
                "feature_path": (
                    "reused_d14_physical_batch1_fixed_received_iq"
                ),
            }
        )
    ).hexdigest()
    runtime_sha256 = str(before_manifest["feature_runtime_sha256"])
    checkpoint_sha256 = str(before_manifest["phase1_checkpoint_sha256"])

    tracemalloc.start()
    run_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contexts: dict[str, dict[str, Any]] = {}
    after_rows_by_scenario = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = _payload_rows(
            before_payloads[scenario], before_manifest, scenario=scenario
        )
        after_rows = _payload_rows(
            after_payloads[scenario], after_manifest, scenario=scenario
        )
        _old_reuse_lock(before_rows, after_rows)
        after_rows_by_scenario[scenario] = after_rows
        feature_start = time.perf_counter()
        before_artifact = _build_feature_artifact(
            model,
            device,
            before_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
        )
        old_feature_by_token = {
            token: before_artifact.features[index]
            for index, token in enumerate(
                before_artifact.physical_sample_ids
            )
        }
        after_artifact = _build_feature_artifact(
            model,
            device,
            after_rows,
            runtime_sha256=runtime_sha256,
            feature_code_sha256=feature_code_sha256,
            checkpoint_sha256=checkpoint_sha256,
            reuse_by_token=old_feature_by_token,
        )
        contexts[scenario] = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "before_artifact": before_artifact,
            "after_artifact": after_artifact,
            "feature_seconds": time.perf_counter() - feature_start,
            "before_provenance": _feature_provenance(
                before_rows, before_artifact
            ),
            "after_provenance": _feature_provenance(
                after_rows, after_artifact
            ),
        }
    disjointness = _cross_scenario_disjointness(after_rows_by_scenario)

    candidates = _candidates()
    lock = _candidate_lock(candidates)
    trace: list[dict[str, Any]] = [
        {
            "phase": "candidate_lock",
            "hyperparameter_lock_sha256": lock["lock_sha256"],
            "candidate_count": len(candidates),
        }
    ]
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_rows = []
    for candidate in candidates:
        evaluations[candidate.candidate_id] = {}
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            context = contexts[scenario]
            evaluation, module_trace = evaluate_joint_leave_two_out(
                context["before_artifact"],
                context["before_rows"]["labels"],
                context["before_rows"]["ranks"],
                context["after_artifact"],
                context["after_rows"]["labels"],
                context["after_rows"]["ranks"],
                hyperparameters=candidate,
            )
            evaluations[candidate.candidate_id][scenario] = evaluation
            trace.extend(
                {
                    "candidate_id": candidate.candidate_id,
                    "scenario": scenario,
                    "hyperparameter_lock_sha256": lock["lock_sha256"],
                    **_jsonable(row),
                }
                for row in module_trace
            )
        summary = _candidate_summary(
            candidate, evaluations[candidate.candidate_id]
        )
        candidate_rows.append(summary)
        trace.append(
            {
                "phase": "candidate_three_scenario_summary",
                **_jsonable(summary),
            }
        )

    selected_id, support_candidate_pass = _select_candidate(
        candidate_rows, candidates
    )
    selected_hp = next(
        value for value in candidates if value.candidate_id == selected_id
    )
    selected_row = next(
        row for row in candidate_rows
        if row["candidate_id"] == selected_id
    )
    scenario_results = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        context = contexts[scenario]
        fit_start = time.perf_counter()
        fitted = fit_before_after_locked(
            context["before_artifact"],
            context["before_rows"]["labels"],
            context["before_rows"]["ranks"],
            context["after_artifact"],
            context["after_rows"]["labels"],
            context["after_rows"]["ranks"],
            k_shot=10,
            hyperparameters=selected_hp,
        )
        trace.extend(
            {
                "phase": "selected_full_support_fit",
                "candidate_id": selected_id,
                "scenario": scenario,
                **_jsonable(row),
            }
            for row in fitted.trace
        )
        evaluation = evaluations[selected_id][scenario]
        scenario_results[scenario] = {
            "joint_leave_two_out": evaluation,
            "all_fold_gate": selected_row["scenario_gates"][scenario],
            "floor": selected_row["floors"][scenario],
            "before_enabled_amplitudes": _enabled_amplitudes(
                fitted.before_state
            ),
            "after_enabled_amplitudes": _enabled_amplitudes(
                fitted.after_state
            ),
            "resource_before": dict(fitted.before_state.resource),
            "resource_after": dict(fitted.after_state.resource),
            "pareto_vs_identity_single_qknn": _measure_pareto(
                fitted.after_state,
                context["after_artifact"].features[:1],
                context["after_artifact"].features,
            ),
            "before_feature_provenance": context["before_provenance"],
            "after_feature_provenance": context["after_provenance"],
            "measured": {
                "feature_extraction_seconds": context["feature_seconds"],
                "selected_full_fit_seconds": (
                    time.perf_counter() - fit_start
                ),
            },
        }

    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = (
        "SUPPORT_ONLY_D16_DEVELOPMENT_SELECTED_NO_QUERY_OPEN"
        if support_candidate_pass
        else "SUPPORT_ONLY_D16_DEVELOPMENT_TRUE_Z0_NO_QUERY_OPEN"
    )
    training_log_sha256 = _write_jsonl_new(
        output / "training_log.jsonl", trace
    )
    before_seal_sha256 = _sha256_file(before_seal)
    after_seal_sha256 = _sha256_file(after_seal)
    audit = {
        "schema": "cvs.phase2.d16_support_only_audit.v1",
        "status": status,
        "claim_scope": "development_diagnostic_support_only_no_query_claim",
        "authority": "development_diagnostic_only",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "promotion_ready_for_query": False,
        "runner_mode": mode,
        "receiver": after_manifest["receiver"],
        "seed": int(after_manifest["seed"]),
        "k_shot": 10,
        "registration_states": ["before", "after"],
        "view_policy": (
            "one_fixed_received_iq_base_view_no_new_channel_state"
        ),
        "unified_hyperparameter_selection": {
            "selected_candidate_id": selected_id,
            "same_candidate_all_scenarios": True,
            "support_candidate_gate_pass": support_candidate_pass,
            "all_scenario_all_fold_gate_required": True,
            "true_zero_fallback_policy": (
                "if_any_scenario_or_fold_positive_gate_fails_select_rank0_force_zero"
            ),
            "true_zero_fallback_is_d16_improvement": False,
            "candidate_rows": candidate_rows,
        },
        "hyperparameter_lock": lock,
        "scenario_results": scenario_results,
        "cross_scenario_support_disjointness": disjointness,
        "query_package_opened": False,
        "query_truth_opened": False,
        "query_prediction_opened": False,
        "query_score_opened": False,
        "scorer_opened": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "additional_leo_channel_state_generation": False,
        "post_reception_views_count_as_additional_k": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_query_decision_policy": (
            "per_sample_all_registered_classes"
        ),
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": (
            "sealed_phase1_checkpoint_only"
        ),
        "runtime_authorization": {
            "feature_extraction_mode": (
                "reused_d14_internal_actual_iq_sha_physical_batch1"
            ),
            "sealed_runtime_sha256": runtime_sha256,
            "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
            "combined_feature_code_sha256": feature_code_sha256,
            **code_hashes,
        },
        "training_log_sha256": training_log_sha256,
        "measured_run_resource": {
            "wall_seconds": time.perf_counter() - run_start,
            "peak_python_tracemalloc_bytes": int(peak_python),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
            "device": str(device),
        },
        "before_package_root_sha256": (
            before_manifest["package_root_sha256"]
        ),
        "after_package_root_sha256": (
            after_manifest["package_root_sha256"]
        ),
        "before_seal_sha256": before_seal_sha256,
        "after_seal_sha256": after_seal_sha256,
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "preopen_audit": {
            "before": before_preopen,
            "after": after_preopen,
        },
    }
    audit = _jsonable(audit)
    audit_sha256 = _write_json_new(output / "support_audit.json", audit)

    lines = [
        "# D16-FCAR strict-K10 enrollment-only开发审计",
        "",
        f"状态：`{status}`。authority固定为`development_diagnostic_only`；只打开before/after enrollment-only support包，未开放query或scorer。",
        "",
        f"三场景统一选择`{selected_id}`；candidate lock SHA为`{lock['lock_sha256']}`。",
        "",
        "|场景|Before old/floor|After old/floor|Z0 new/floor|D16 new/floor|joint/H|forgetting|old lock|全fold门|enabled after|",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenario_results[scenario]
        joint = row["joint_leave_two_out"]
        old_floor = row["floor"]["old_floor"]["candidate_min_accuracy"]
        new_floor = row["floor"]["new_floor"]["candidate_min_accuracy"]
        z0_new_floor = row["floor"]["new_floor"]["z0_min_accuracy"]
        enabled = sum(
            int(value["enabled"])
            for value in row["after_enabled_amplitudes"].values()
        )
        lines.append(
            f"|`{scenario}`|"
            f"{joint['before_old']['overall_accuracy']:.4f}/"
            f"{joint['before_old']['min_class_accuracy']:.4f}|"
            f"{joint['after_old']['overall_accuracy']:.4f}/"
            f"{old_floor if old_floor is not None else joint['after_old']['min_class_accuracy']:.4f}|"
            f"{joint['base_after_new']['overall_accuracy']:.4f}/"
            f"{z0_new_floor if z0_new_floor is not None else joint['base_after_new']['min_class_accuracy']:.4f}|"
            f"{joint['after_new']['overall_accuracy']:.4f}/"
            f"{new_floor if new_floor is not None else joint['after_new']['min_class_accuracy']:.4f}|"
            f"{joint['joint']['overall_accuracy']:.4f}/"
            f"{joint['H_old_new']:.4f}|"
            f"{joint['old_forgetting']:.4f}|"
            f"{joint['old_score_bitwise_locked']}|"
            f"{row['all_fold_gate']['all_folds_gate_pass']}|"
            f"{enabled}|"
        )
    lines.extend(
        [
            "",
            "门基于全部scenario×全部outer held2 fold逐类判断，不允许aggregate平均抵消。任一fold出现旧类遗忘、任一旧/新类相对Z0下降、joint/H/floor下降、缺少严格floor收益或old-score锁失败，positive arm即失败；全部positive arm失败时保存true Z0。本runner不开放query。",
            "",
        ]
    )
    report_sha256 = _write_text_new(
        output / "report.md", "\n".join(lines)
    )
    receipt = {
        "schema": "cvs.phase2.d16_support_only_receipt.v1",
        "status": status,
        "authority": "development_diagnostic_only",
        "formal_launch_authority": False,
        "promotion_ready_for_query": False,
        "support_audit_sha256": audit_sha256,
        "training_log_sha256": training_log_sha256,
        "report_sha256": report_sha256,
        "query_opened": False,
        "selected_candidate_id": selected_id,
        "support_candidate_gate_pass": support_candidate_pass,
        "hyperparameter_lock_sha256": lock["lock_sha256"],
        "expected_before_seal_sha256": expected_before_seal_sha256,
        "expected_after_seal_sha256": expected_after_seal_sha256,
        "sealed_runtime_sha256": runtime_sha256,
        "sealed_phase1_checkpoint_sha256": checkpoint_sha256,
        "combined_feature_code_sha256": feature_code_sha256,
        **code_hashes,
    }
    receipt_sha256 = _write_json_new(
        output / "RECEIPT.json", receipt
    )
    return {
        "receipt_sha256": receipt_sha256,
        **receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mode", choices=("development_select",), required=True
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                before_root=args.before_root,
                before_seal=args.before_seal,
                expected_before_seal_sha256=args.before_seal_sha256,
                after_root=args.after_root,
                after_seal=args.after_seal,
                expected_after_seal_sha256=args.after_seal_sha256,
                output=args.output,
                device_name=args.device,
                mode=args.mode,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
