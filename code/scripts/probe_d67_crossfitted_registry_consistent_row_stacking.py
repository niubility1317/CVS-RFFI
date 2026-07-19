#!/usr/bin/env python3
"""D67 support-only cross-fitted registry-consistent affine row stacking."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D62_HELPER_PATH = SCRIPT_DIR / "probe_d62_crossfitted_fisher_row_splice.py"
D65_HELPER_PATH = SCRIPT_DIR / "probe_d65_frozen_stage2b_blocklda_append_only.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d67_row_stacking.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D67 could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d62 = _load_module("d67_d62_probe_helper", D62_HELPER_PATH)
d65 = _load_module("d67_d65_probe_helper", D65_HELPER_PATH)
core = _load_module("d67_row_stacking_core", CORE_PATH)
d43 = d62.d43


ARM = "crossfitted_registry_consistent_row_stacking"
STRUCTURE = "d62_d65_class_agnostic_closed_form_continuous_affine_row_stacking"
FORMULA = (
    "standardize each anonymous D62/D65 row by train-support one-vs-rest moments; "
    "alpha=clip(sum(w*d*(target-z62))/sum(w*d^2),0,1); map the convex score "
    "back to D62 row scale and compile one centered affine state"
)
INNER_FOLD_COUNT = 4
EXPECTED_REAL_FIT_COUNT = 60
D62_COMPONENT_RECORDS_PER_K8_FIT = 92
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D67ProbeError(RuntimeError):
    pass


def _helper_hashes_for_probe_root(probe_root: Path) -> dict[str, str]:
    root = probe_root.resolve()
    return {
        "d67_d62_helper_sha256": d43._sha256(
            root / "code" / "scripts" / D62_HELPER_PATH.name
        ),
        "d67_d65_helper_sha256": d43._sha256(
            root / "code" / "scripts" / D65_HELPER_PATH.name
        ),
        "d67_core_sha256": d43._sha256(
            root / "code" / "cvsrffi" / CORE_PATH.name
        ),
    }


def _d65_expert(
    d42: Any,
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = core.validate_symmetric_support(rows, labels, class_count, k_shot)
    old_count = int(old_class_count)
    if old_count < 2 or old_count > int(class_count):
        raise D67ProbeError("D67 D65 expert old-class count drift")
    old_mask = y < old_count
    old_x = x[old_mask]
    old_y = y[old_mask]
    if len(old_x) != old_count * int(k_shot):
        raise D67ProbeError("D67 D65 expert old support drift")
    covariance, _, fallback, residual_rank, residual_energy = d65._fit_stage2b_covariance(
        d42, old_x, old_y, old_count, int(k_shot)
    )
    means = d65._means(x, y, int(class_count))
    coefficient, intercept, equation_residual = d65._compile_rows(covariance, means)
    scores = x.astype(np.float32) @ coefficient.T + intercept[None, :]
    return coefficient, intercept, {
        "covariance_sha256": d65._covariance_sha256(covariance),
        "old_class_count": old_count,
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "unit_covariance_fallback": bool(fallback),
        "residual_rank": int(residual_rank),
        "residual_energy": float(residual_energy),
        "covariance_condition_number": float(np.linalg.cond(covariance)),
        "covariance_equation_residual_max": float(equation_residual),
        "support_accuracy": float(np.mean(np.argmax(scores, axis=1) == y)),
    }


def _standardization_audit(state: Any) -> dict[str, Any]:
    return {
        "positive_mean": np.asarray(state.positive_mean).tolist(),
        "negative_mean": np.asarray(state.negative_mean).tolist(),
        "within_scale": np.asarray(state.within_scale).tolist(),
        "gap_scale": np.asarray(state.gap_scale).tolist(),
        "scale": np.asarray(state.scale).tolist(),
    }


def build_d67_fit(
    d42: Any,
) -> tuple[Callable[..., Any], list[dict[str, Any]], dict[str, Any]]:
    d62_fit, d62_records = d62.build_d62_fit(d42)
    records: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {"pending_old_class_count": None, "completed_pairs": 0}

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x, y = core.validate_symmetric_support(rows, labels, class_count, k_shot)
        pending = lifecycle["pending_old_class_count"]
        if pending is None:
            old_count = int(class_count)
            phase = "stage2b_before"
            lifecycle["pending_old_class_count"] = old_count
        else:
            old_count = int(pending)
            if int(class_count) <= old_count:
                raise D67ProbeError("D67 Stage2-B/Stage2-C lifecycle order drift")
            phase = "stage2c_final"
            lifecycle["pending_old_class_count"] = None
            lifecycle["completed_pairs"] = int(lifecycle["completed_pairs"]) + 1

        d62_coefficient, d62_intercept, d62_audit = d62_fit(
            x, y, int(class_count), int(k_shot)
        )
        d65_coefficient, d65_intercept, d65_audit = _d65_expert(
            d42, x, y, int(class_count), int(k_shot), old_count
        )

        if int(k_shot) <= 4:
            final_coefficient = np.asarray(d62_coefficient, dtype=np.float32).copy()
            final_intercept = np.asarray(d62_intercept, dtype=np.float32).copy()
            alpha = np.zeros(int(class_count), dtype=np.float64)
            solver_audit = {
                "numerator": np.zeros(int(class_count)).tolist(),
                "denominator": np.zeros(int(class_count)).tolist(),
                "risk_d62": None,
                "risk_d65": None,
                "risk_stacked": None,
            }
            partition_audit: list[dict[str, Any]] = []
            full_d62_standardization = None
            full_d65_standardization = None
            compile_error = 0.0
            boundary_status = "k_le_4_exact_d62_fallback"
        else:
            partitions = core.four_rank_partitions(
                y, int(class_count), int(k_shot)
            )
            held_d62 = np.empty((len(x), int(class_count)), dtype=np.float64)
            held_d65 = np.empty_like(held_d62)
            held_once = np.zeros(len(x), dtype=np.int64)
            partition_audit = []
            for fold_index, (train, held) in enumerate(partitions):
                train_k = len(train) // int(class_count)
                fold_d62_coef, fold_d62_intercept, fold_d62_audit = d62_fit(
                    x[train], y[train], int(class_count), train_k
                )
                fold_d65_coef, fold_d65_intercept, fold_d65_audit = _d65_expert(
                    d42,
                    x[train],
                    y[train],
                    int(class_count),
                    train_k,
                    old_count,
                )
                fold_d62_state = core.standardize_affine_rows(
                    fold_d62_coef,
                    fold_d62_intercept,
                    x[train],
                    y[train],
                    int(class_count),
                )
                fold_d65_state = core.standardize_affine_rows(
                    fold_d65_coef,
                    fold_d65_intercept,
                    x[train],
                    y[train],
                    int(class_count),
                )
                held_d62[held] = core.standardized_scores(x[held], fold_d62_state)
                held_d65[held] = core.standardized_scores(x[held], fold_d65_state)
                held_once[held] += 1
                partition_audit.append(
                    {
                        "fold_index": fold_index,
                        "train_k_shot": train_k,
                        "train_row_count": int(len(train)),
                        "held_row_count": int(len(held)),
                        "train_index_sha256": hashlib.sha256(
                            np.ascontiguousarray(train, dtype=np.int64).tobytes()
                        ).hexdigest(),
                        "held_index_sha256": hashlib.sha256(
                            np.ascontiguousarray(held, dtype=np.int64).tobytes()
                        ).hexdigest(),
                        "held_train_intersection_count": int(
                            len(np.intersect1d(train, held))
                        ),
                        "d62_boundary_status": fold_d62_audit["d62_boundary_status"],
                        "d65_covariance_sha256": fold_d65_audit["covariance_sha256"],
                    }
                )
            if not np.array_equal(held_once, np.ones(len(x), dtype=np.int64)):
                raise D67ProbeError("D67 cross-fit held rows are not exact-once")
            alpha, raw_solver_audit = core.solve_class_balanced_convex_weights(
                held_d62, held_d65, y, int(class_count)
            )
            solver_audit = {
                name: np.asarray(value).tolist()
                for name, value in raw_solver_audit.items()
            }
            full_d62_state = core.standardize_affine_rows(
                d62_coefficient,
                d62_intercept,
                x,
                y,
                int(class_count),
            )
            full_d65_state = core.standardize_affine_rows(
                d65_coefficient,
                d65_intercept,
                x,
                y,
                int(class_count),
            )
            final_coefficient, final_intercept, compile_error = (
                core.compile_stacked_affine(full_d62_state, full_d65_state, alpha)
            )
            full_d62_standardization = _standardization_audit(full_d62_state)
            full_d65_standardization = _standardization_audit(full_d65_state)
            boundary_status = "crossfitted_continuous_row_stacking_active"

        compiled_scores = (
            x.astype(np.float32) @ final_coefficient.T
            + final_intercept[None, :]
        )
        audit = dict(d62_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d67_probe_arm": ARM,
                "d67_formula": FORMULA,
                "d67_phase": phase,
                "d67_boundary_status": boundary_status,
                "d67_actual_k": int(k_shot),
                "d67_class_count": int(class_count),
                "d67_old_class_count_for_lifecycle": old_count,
                "d67_crossfit_fold_count": len(partition_audit),
                "d67_crossfit_partition_audit": partition_audit,
                "d67_alpha_by_class": alpha.tolist(),
                "d67_alpha_min": float(np.min(alpha)),
                "d67_alpha_mean": float(np.mean(alpha)),
                "d67_alpha_max": float(np.max(alpha)),
                "d67_alpha_d62_boundary_count": int(np.sum(alpha == 0.0)),
                "d67_alpha_d65_boundary_count": int(np.sum(alpha == 1.0)),
                "d67_alpha_interior_count": int(np.sum((alpha > 0.0) & (alpha < 1.0))),
                "d67_solver_audit": solver_audit,
                "d67_full_d62_standardization": full_d62_standardization,
                "d67_full_d65_standardization": full_d65_standardization,
                "d67_d65_expert_audit": d65_audit,
                "d67_compile_float32_error_max": float(compile_error),
                "d67_compiled_support_accuracy": float(
                    np.mean(np.argmax(compiled_scores, axis=1) == y)
                ),
                "d67_class_id_specific_formula": False,
                "d67_old_new_role_specific_query_branch": False,
                "d67_scene_receiver_handle_specific_branch": False,
                "d67_uses_outer_held_or_query": False,
                "d67_hyperparameter_count": 0,
                "d67_query_joint_optimization": False,
                "d67_single_affine_state_only": True,
                "d67_actual_coefficient_fp32": final_coefficient.tolist(),
                "d67_actual_intercept_fp32": final_intercept.tolist(),
            }
        )
        records.append(
            {
                "phase": phase,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "alpha_sha256": hashlib.sha256(
                    np.ascontiguousarray(alpha, dtype=np.float64).tobytes()
                ).hexdigest(),
                "alpha_min": float(np.min(alpha)),
                "alpha_mean": float(np.mean(alpha)),
                "alpha_max": float(np.max(alpha)),
                "support_accuracy": float(
                    np.mean(np.argmax(compiled_scores, axis=1) == y)
                ),
            }
        )
        return final_coefficient, final_intercept, audit

    return fit, records, {"d62_records": d62_records, "lifecycle": lifecycle}


def _d62_stage_cost(d42: Any, k_shot: int, class_count: int) -> dict[str, int]:
    k, count = int(k_shot), int(class_count)
    if k <= 2:
        return {
            "fit_count": 0,
            "lda_macs": 0,
            "fisher_macs": 0,
            "reliability_macs": 0,
            "fusion_macs": 0,
            "gate_macs": 0,
        }
    main = int(d42._lda_fit_macs(k * count, count))
    inner = int(d42._lda_fit_macs((k - 1) * count, count))
    component_macs = 2 * main + 2 * k * inner
    extra_fit_count = 2 * (k + 1)
    dimension = int(d42.FEATURE_DIM)
    return {
        "fit_count": 2 * extra_fit_count,
        "lda_macs": 2 * component_macs,
        "fisher_macs": int(d62.d61._fisher_dense_macs(dimension, extra_fit_count)),
        "reliability_macs": int(2 * k * (k + 1) * dimension * count * count),
        "fusion_macs": int(2 * (dimension + 1) * count),
        "gate_macs": int(k * count * count * 8),
    }


def _d65_stage_cost(
    d42: Any, k_shot: int, class_count: int, old_class_count: int
) -> dict[str, int]:
    k, count, old = int(k_shot), int(class_count), int(old_class_count)
    dimension = int(d42.FEATURE_DIM)
    covariance = int(d42._lda_fit_macs(k * old, old))
    compile_macs = int(
        count * k * dimension
        + count * dimension * dimension
        + count * (2 * dimension + 1)
    )
    return {"fit_count": 1, "macs": covariance + compile_macs}


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        outer_k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        if outer_k <= 4:
            inner_d62_fit_count = inner_d62_macs = 0
            inner_d65_fit_count = inner_d65_macs = stacking_macs = 0
        else:
            held_per_class = outer_k // INNER_FOLD_COUNT
            train_k = outer_k - held_per_class
            inner_d62_fit_count = inner_d62_macs = 0
            inner_d65_fit_count = inner_d65_macs = 0
            stacking_macs = 0
            for count in (old_count, all_count):
                d62_cost = _d62_stage_cost(d42, train_k, count)
                inner_d62_fit_count += INNER_FOLD_COUNT * d62_cost["fit_count"]
                inner_d62_macs += INNER_FOLD_COUNT * sum(
                    d62_cost[name]
                    for name in (
                        "lda_macs",
                        "fisher_macs",
                        "reliability_macs",
                        "fusion_macs",
                        "gate_macs",
                    )
                )
                inner_d65 = _d65_stage_cost(d42, train_k, count, old_count)
                outer_d65 = _d65_stage_cost(d42, outer_k, count, old_count)
                inner_d65_fit_count += INNER_FOLD_COUNT * inner_d65["fit_count"] + outer_d65["fit_count"]
                inner_d65_macs += INNER_FOLD_COUNT * inner_d65["macs"] + outer_d65["macs"]
                stacking_macs += int(
                    10 * outer_k * count * count * (dimension + 1)
                    + 12 * outer_k * count * count
                    + 4 * count * dimension
                )
        added_macs = int(inner_d62_macs + inner_d65_macs + stacking_macs)
        resource.update(
            {
                "d67_crossfit_fold_count_per_stage": INNER_FOLD_COUNT if outer_k > 4 else 0,
                "d67_inner_d62_lda_fit_count": inner_d62_fit_count,
                "d67_inner_d62_total_adaptation_macs": inner_d62_macs,
                "d67_d65_expert_covariance_fit_count": inner_d65_fit_count,
                "d67_d65_expert_total_adaptation_macs": inner_d65_macs,
                "d67_standardization_stacking_scalar_macs": stacking_macs,
                "d67_total_added_adaptation_macs": added_macs,
                "d67_query_extra_macs": 0,
                "d67_persistent_state_extra_bytes": 0,
                "d67_optimizer_steps_extra": 0,
                "d67_resource_single_affine_state_only": True,
            }
        )
        resource["lda_closed_form_fit_count"] = int(
            resource["lda_closed_form_fit_count"]
            + inner_d62_fit_count
            + inner_d65_fit_count
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + added_macs
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D67ProbeError("D67 training row closure drift")
    alpha_values: list[float] = []
    fit_audits = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d67_crossfit_fold_count_per_stage", -1)) != 4
            or int(resource.get("d67_query_extra_macs", -1)) != 0
            or resource.get("d67_resource_single_affine_state_only") is not True
        ):
            raise D67ProbeError("D67 resource closure drift")
        for phase_name, expected_phase, expected_count in (
            ("before_covariance_audit", "stage2b_before", 6),
            ("final_covariance_audit", "stage2c_final", 11),
        ):
            audit = row["geometry_summary"][phase_name]
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d67_probe_arm": ARM,
                "d67_formula": FORMULA,
                "d67_phase": expected_phase,
                "d67_boundary_status": "crossfitted_continuous_row_stacking_active",
                "d67_actual_k": 8,
                "d67_class_count": expected_count,
                "d67_old_class_count_for_lifecycle": 6,
                "d67_crossfit_fold_count": 4,
                "d67_class_id_specific_formula": False,
                "d67_old_new_role_specific_query_branch": False,
                "d67_scene_receiver_handle_specific_branch": False,
                "d67_uses_outer_held_or_query": False,
                "d67_hyperparameter_count": 0,
                "d67_query_joint_optimization": False,
                "d67_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D67ProbeError("D67 exact audit drift")
            alpha = np.asarray(audit.get("d67_alpha_by_class"), dtype=np.float64)
            if (
                alpha.shape != (expected_count,)
                or not np.isfinite(alpha).all()
                or np.any(alpha < 0.0)
                or np.any(alpha > 1.0)
            ):
                raise D67ProbeError("D67 alpha closure drift")
            partitions = audit.get("d67_crossfit_partition_audit")
            if (
                not isinstance(partitions, list)
                or len(partitions) != 4
                or any(int(item.get("held_train_intersection_count", -1)) != 0 for item in partitions)
            ):
                raise D67ProbeError("D67 partition audit drift")
            alpha_values.extend(float(value) for value in alpha)
            fit_audits += 1
    return {
        "verified_d67_target_row_count": len(target),
        "verified_d67_fit_audit_count": fit_audits,
        "verified_d67_crossfit_partition_count": 4 * fit_audits,
        "verified_d67_alpha_min": float(np.min(alpha_values)),
        "verified_d67_alpha_mean": float(np.mean(alpha_values)),
        "verified_d67_alpha_max": float(np.max(alpha_values)),
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D67ProbeError("D67 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d67-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--executed-probe-script", type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if known.verify_existing:
        if known.executed_probe_script is None:
            raise D67ProbeError("D67 existing-output verification requires executed script")
        executed_script = known.executed_probe_script.resolve()
        if not output.is_dir():
            raise D67ProbeError("D67 existing output is missing")
        metadata_path = output / "D67_PROBE_METADATA.json"
        if metadata_path.exists():
            raise D67ProbeError("D67 metadata already exists; refusing overwrite")
        executed_script_sha = d43._sha256(executed_script)
        executed_probe_root = executed_script.parents[2]
        helper_hashes = _helper_hashes_for_probe_root(executed_probe_root)
        evidence = _verify_output(output, executed_script_sha, helper_hashes)
        fit_count = int(evidence["verified_d67_fit_audit_count"])
        if fit_count != EXPECTED_REAL_FIT_COUNT:
            raise D67ProbeError(
                f"D67 existing fit-audit count drift: {fit_count} != {EXPECTED_REAL_FIT_COUNT}"
            )
        metadata = {
            "schema": "cvs.phase2.d67.crossfitted_registry_consistent_row_stacking_probe.v1",
            "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE_POST_RUN_CLOSURE",
            "arm": known.d67_arm,
            "formal_candidate": False,
            "probe_forced_nonpromotable": True,
            "selected_only_full_k10_refit_allowed": False,
            "query_opened": False,
            "post_run_verifier_only": True,
            "post_run_verifier_reason": "completed 105-row run ended after artifact write on stale 30-vs-60 fit-count assertion",
            "executed_probe_script_path": str(executed_script),
            "executed_probe_script_sha256": executed_script_sha,
            "verifier_script_path": str(Path(__file__).resolve()),
            "verifier_script_sha256": d43._sha256(Path(__file__).resolve()),
            "formula": FORMULA,
            "fit_audit_count": fit_count,
            "nested_d62_component_fit_execution_count": int(
                fit_count * D62_COMPONENT_RECORDS_PER_K8_FIT
            ),
            "runtime_root": str(known.runtime_root.resolve()),
            "probe_root": str(known.probe_root.resolve()),
            **evidence,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if known.executed_probe_script is not None:
        raise D67ProbeError("D67 executed script is only valid with --verify-existing")
    if output.exists():
        raise D67ProbeError(f"D67 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = _helper_hashes_for_probe_root(known.probe_root)
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d67_locked_d42_runner", 1
    state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, records, state = build_d67_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D67ProbeError("D67 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d67_arm,
            probe_script_sha256=script_sha,
            extra_source_closure=helper_hashes,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_macs is not None:
            d42._lda_fit_macs = original_macs
        if d42 is not None and original_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    if len(records) != EXPECTED_REAL_FIT_COUNT:
        raise D67ProbeError(
            f"D67 fit record count drift: {len(records)} != {EXPECTED_REAL_FIT_COUNT}"
        )
    d62_records = state.get("d62_records", [])
    expected_d62_records = EXPECTED_REAL_FIT_COUNT * D62_COMPONENT_RECORDS_PER_K8_FIT
    if len(d62_records) != expected_d62_records:
        raise D67ProbeError(
            f"D67 nested D62 component-fit count drift: {len(d62_records)} != {expected_d62_records}"
        )
    if state.get("lifecycle", {}).get("pending_old_class_count") is not None:
        raise D67ProbeError("D67 lifecycle ended with pending Stage2-B state")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d67.crossfitted_registry_consistent_row_stacking_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d67_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "fit_record_count": len(records),
        "fit_record_sha256": record_sha,
        "nested_d62_component_fit_execution_count": len(d62_records),
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D67_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
