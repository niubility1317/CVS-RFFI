#!/usr/bin/env python3
"""D68 support-only signed calibration of the frozen D65 registry head."""

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
D67_HELPER_PATH = (
    SCRIPT_DIR / "probe_d67_crossfitted_registry_consistent_row_stacking.py"
)
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d68_signed_calibration.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D68 could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d67 = _load_module("d68_d67_probe_helper", D67_HELPER_PATH)
core = _load_module("d68_signed_calibration_core", CORE_PATH)
d43, d62, d65 = d67.d43, d67.d62, d67.d65


ARM = "crossfitted_signed_frozen_registry_calibration"
STRUCTURE = "d65_frozen_covariance_crossfitted_signed_row_standardization"
FORMULA = (
    "leave-one-rank-out D65 held scores lock each anonymous row orientation by "
    "sign(mean_positive-mean_negative); rows are centered and scaled by "
    "max(within,abs(gap)/2,eps); Stage2-B signed old rows and the common affine "
    "term freeze bitwise while Stage2-C appends only identically signed new rows"
)
EXPECTED_REAL_FIT_COUNT = 60
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D68ProbeError(RuntimeError):
    pass


def _helper_hashes_for_probe_root(probe_root: Path) -> dict[str, str]:
    root = probe_root.resolve()
    return {
        "d68_d67_helper_sha256": d43._sha256(
            root / "code" / "scripts" / D67_HELPER_PATH.name
        ),
        "d68_d62_helper_sha256": d43._sha256(
            root / "code" / "scripts" / d67.D62_HELPER_PATH.name
        ),
        "d68_d65_helper_sha256": d43._sha256(
            root / "code" / "scripts" / d67.D65_HELPER_PATH.name
        ),
        "d68_d67_core_sha256": d43._sha256(
            root / "code" / "cvsrffi" / d67.CORE_PATH.name
        ),
        "d68_core_sha256": d43._sha256(
            root / "code" / "cvsrffi" / CORE_PATH.name
        ),
    }


def _standardization_audit(state: Any) -> dict[str, Any]:
    return {
        "positive_mean": np.asarray(state.positive_mean).tolist(),
        "negative_mean": np.asarray(state.negative_mean).tolist(),
        "within_scale": np.asarray(state.within_scale).tolist(),
        "gap_scale": np.asarray(state.gap_scale).tolist(),
        "scale": np.asarray(state.scale).tolist(),
    }


def build_d68_fit(
    d42: Any,
) -> tuple[Callable[..., Any], list[dict[str, Any]], dict[str, Any]]:
    d65_fit, d65_records, d65_lifecycle = d65.build_d65_fit(d42)
    d62_fit, d62_records = d62.build_d62_fit(d42)
    records: list[dict[str, Any]] = []
    calibration_lifecycle: dict[str, Any] = {"pending": None, "completed_pairs": 0}

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x, y = core.validate_symmetric_support(rows, labels, class_count, k_shot)
        d65_coefficient, d65_intercept, d65_audit = d65_fit(
            x, y, int(class_count), int(k_shot)
        )
        old_count = int(d65_audit["d65_old_class_count"])
        d65_phase = d65_audit["d65_phase"]
        phase = (
            "stage2b_before"
            if d65_phase == "stage2b_fit_and_freeze"
            else "stage2c_final"
        )

        if int(k_shot) == 1:
            final_coefficient, final_intercept, audit = d62_fit(
                x, y, int(class_count), int(k_shot)
            )
            orientation = np.ones(int(class_count), dtype=np.float64)
            delta = np.zeros(int(class_count), dtype=np.float64)
            risk_raw = risk_signed = None
            partition_audit: list[dict[str, Any]] = []
            fold_agreement = np.zeros(int(class_count), dtype=np.int64)
            full_standardization = None
            compile_error = 0.0
            support_accuracy = float(
                np.mean(
                    np.argmax(
                        x.astype(np.float32) @ final_coefficient.T
                        + final_intercept[None, :],
                        axis=1,
                    )
                    == y
                )
            )
            boundary_status = "k1_exact_d62_fallback"
            old_rows_bitwise_unchanged = None
            common_affine_sha256 = None
            if phase == "stage2b_before":
                calibration_lifecycle["pending"] = {"k1_fallback": True}
            else:
                if calibration_lifecycle["pending"] != {"k1_fallback": True}:
                    raise D68ProbeError("D68 K1 lifecycle order drift")
                calibration_lifecycle["pending"] = None
                calibration_lifecycle["completed_pairs"] += 1
        else:
            partitions = core.leave_one_rank_partitions(
                y, int(class_count), int(k_shot)
            )
            held_scores = np.empty((len(x), int(class_count)), dtype=np.float64)
            held_once = np.zeros(len(x), dtype=np.int64)
            fold_deltas: list[np.ndarray] = []
            partition_audit = []
            for fold_index, (train, held) in enumerate(partitions):
                train_k = int(k_shot) - 1
                fold_coefficient, fold_intercept, fold_d65_audit = d67._d65_expert(
                    d42,
                    x[train],
                    y[train],
                    int(class_count),
                    train_k,
                    old_count,
                )
                fold_state = core.standardize_affine_rows(
                    fold_coefficient,
                    fold_intercept,
                    x[train],
                    y[train],
                    int(class_count),
                )
                fold_scores = core.standardized_scores(x[held], fold_state)
                held_scores[held] = fold_scores
                held_once[held] += 1
                held_labels = y[held]
                fold_delta = np.asarray(
                    [
                        float(np.mean(fold_scores[held_labels == index, index]))
                        - float(np.mean(fold_scores[held_labels != index, index]))
                        for index in range(int(class_count))
                    ],
                    dtype=np.float64,
                )
                fold_deltas.append(fold_delta)
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
                        "held_delta": fold_delta.tolist(),
                        "d65_covariance_sha256": fold_d65_audit[
                            "covariance_sha256"
                        ],
                    }
                )
            if not np.array_equal(held_once, np.ones(len(x), dtype=np.int64)):
                raise D68ProbeError("D68 cross-fit held rows are not exact-once")
            recomputed_orientation, solver_audit = core.solve_orientations(
                held_scores, y, int(class_count)
            )
            delta = np.asarray(solver_audit["crossfit_delta"], dtype=np.float64)
            risk_raw = np.asarray(solver_audit["risk_raw"], dtype=np.float64)
            risk_signed = np.asarray(solver_audit["risk_signed"], dtype=np.float64)
            fold_signs = np.where(np.asarray(fold_deltas) >= 0.0, 1.0, -1.0)
            fold_agreement = np.sum(
                fold_signs == recomputed_orientation[None, :], axis=0
            ).astype(np.int64)
            full_state = core.standardize_affine_rows(
                d65_coefficient,
                d65_intercept,
                x,
                y,
                int(class_count),
            )
            raw_coefficient = (
                recomputed_orientation[:, None] * full_state.coefficient
            )
            raw_intercept = recomputed_orientation * full_state.intercept
            signed_scores = (
                core.standardized_scores(x, full_state)
                * recomputed_orientation[None, :]
            )
            if phase == "stage2b_before":
                common_coefficient = np.mean(raw_coefficient, axis=0)
                common_intercept = float(np.mean(raw_intercept))
                final_coefficient = (
                    raw_coefficient - common_coefficient[None, :]
                ).astype(np.float32)
                final_intercept = (raw_intercept - common_intercept).astype(
                    np.float32
                )
                orientation = recomputed_orientation.copy()
                calibration_lifecycle["pending"] = {
                    "old_class_count": old_count,
                    "coefficient": final_coefficient.copy(),
                    "intercept": final_intercept.copy(),
                    "orientation": orientation.copy(),
                    "common_coefficient": common_coefficient.copy(),
                    "common_intercept": common_intercept,
                }
                old_rows_bitwise_unchanged = True
            else:
                pending = calibration_lifecycle.get("pending")
                if (
                    not isinstance(pending, dict)
                    or int(pending.get("old_class_count", -1)) != old_count
                    or int(class_count) <= old_count
                ):
                    raise D68ProbeError("D68 calibrated lifecycle order drift")
                common_coefficient = np.asarray(
                    pending["common_coefficient"], dtype=np.float64
                )
                common_intercept = float(pending["common_intercept"])
                final_coefficient = (
                    raw_coefficient - common_coefficient[None, :]
                ).astype(np.float32)
                final_intercept = (raw_intercept - common_intercept).astype(
                    np.float32
                )
                final_coefficient[:old_count] = np.asarray(
                    pending["coefficient"], dtype=np.float32
                )
                final_intercept[:old_count] = np.asarray(
                    pending["intercept"], dtype=np.float32
                )
                orientation = recomputed_orientation.copy()
                orientation[:old_count] = np.asarray(
                    pending["orientation"], dtype=np.float64
                )
                old_rows_bitwise_unchanged = bool(
                    np.array_equal(
                        final_coefficient[:old_count], pending["coefficient"]
                    )
                    and np.array_equal(
                        final_intercept[:old_count], pending["intercept"]
                    )
                )
                if not old_rows_bitwise_unchanged:
                    raise D68ProbeError("D68 calibrated old rows changed")
                calibration_lifecycle["pending"] = None
                calibration_lifecycle["completed_pairs"] += 1
            common_affine_sha256 = hashlib.sha256(
                np.ascontiguousarray(
                    np.concatenate(
                        [common_coefficient, np.asarray([common_intercept])]
                    ),
                    dtype=np.float64,
                ).tobytes()
            ).hexdigest()
            compiled_scores = (
                x.astype(np.float32) @ final_coefficient.T
                + final_intercept[None, :]
            )
            common_scores = (
                x.astype(np.float64) @ common_coefficient + common_intercept
            )
            checked_slice = slice(None) if phase == "stage2b_before" else slice(old_count, None)
            expected = signed_scores[:, checked_slice] - common_scores[:, None]
            compile_error = float(
                np.max(np.abs(expected - compiled_scores[:, checked_slice]))
            )
            risk_signed = core.class_balanced_squared_risk(
                held_scores * orientation[None, :], y, int(class_count)
            )
            support_accuracy = float(
                np.mean(np.argmax(compiled_scores, axis=1) == y)
            )
            full_standardization = _standardization_audit(full_state)
            audit = dict(d65_audit)
            boundary_status = "crossfitted_signed_frozen_calibration_active"

        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d68_probe_arm": ARM,
                "d68_formula": FORMULA,
                "d68_phase": phase,
                "d68_boundary_status": boundary_status,
                "d68_actual_k": int(k_shot),
                "d68_class_count": int(class_count),
                "d68_old_class_count_for_lifecycle": old_count,
                "d68_crossfit_fold_count": len(partition_audit),
                "d68_crossfit_partition_audit": partition_audit,
                "d68_orientation_by_class": orientation.tolist(),
                "d68_recomputed_orientation_by_class": recomputed_orientation.tolist()
                if int(k_shot) > 1
                else orientation.tolist(),
                "d68_orientation_negative_count": int(np.sum(orientation < 0.0)),
                "d68_crossfit_delta": delta.tolist(),
                "d68_orientation_fold_agreement_count": fold_agreement.tolist(),
                "d68_crossfit_risk_raw": None
                if risk_raw is None
                else risk_raw.tolist(),
                "d68_crossfit_risk_signed": None
                if risk_signed is None
                else risk_signed.tolist(),
                "d68_full_standardization": full_standardization,
                "d68_compile_float32_error_max": float(compile_error),
                "d68_compiled_support_accuracy": support_accuracy,
                "d68_stage2b_common_affine_sha256": common_affine_sha256,
                "d68_old_row_fp32_bitwise_unchanged": old_rows_bitwise_unchanged,
                "d68_calibrated_lifecycle_policy": "freeze_stage2b_signed_old_rows_append_only_signed_new_rows",
                "d68_ground_component_input_count": 0,
                "d68_class_id_specific_formula": False,
                "d68_old_new_role_specific_query_branch": False,
                "d68_scene_receiver_handle_specific_branch": False,
                "d68_uses_outer_held_or_query": False,
                "d68_hyperparameter_count": 0,
                "d68_query_joint_optimization": False,
                "d68_single_affine_state_only": True,
                "d68_actual_coefficient_fp32": np.asarray(
                    final_coefficient, dtype=np.float32
                ).tolist(),
                "d68_actual_intercept_fp32": np.asarray(
                    final_intercept, dtype=np.float32
                ).tolist(),
            }
        )
        records.append(
            {
                "phase": phase,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "orientation_sha256": hashlib.sha256(
                    np.ascontiguousarray(orientation, dtype=np.float64).tobytes()
                ).hexdigest(),
                "negative_orientation_count": int(np.sum(orientation < 0.0)),
                "support_accuracy": support_accuracy,
            }
        )
        return (
            np.asarray(final_coefficient, dtype=np.float32),
            np.asarray(final_intercept, dtype=np.float32),
            audit,
        )

    return fit, records, {
        "d65_records": d65_records,
        "d65_lifecycle": d65_lifecycle,
        "calibration_lifecycle": calibration_lifecycle,
        "d62_records": d62_records,
    }


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        outer_k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = int(d42.FEATURE_DIM)
        base_lda_macs = int(resource["estimated_lda_fit_macs"])
        if outer_k == 1:
            full_fit_count = int(resource["lda_closed_form_fit_count"])
            inner_fit_count = inner_macs = calibration_macs = 0
            actual_lda_macs = base_lda_macs
            fold_count = 0
        else:
            train_k = outer_k - 1
            full_covariance_macs = int(original_macs(outer_k * old_count, old_count))
            append_count = all_count - old_count
            append_macs = int(
                append_count * outer_k * dimension
                + append_count * dimension * dimension
                + append_count * (2 * dimension + 1)
            )
            full_fit_count = 1
            inner_fit_count = 2 * outer_k
            inner_macs = 0
            calibration_macs = 0
            for count in (old_count, all_count):
                inner = d67._d65_stage_cost(d42, train_k, count, old_count)
                inner_macs += outer_k * int(inner["macs"])
                calibration_macs += int(
                    outer_k * count * count * (dimension + 8)
                    + 8 * outer_k * count
                    + 4 * count * dimension
                )
            actual_lda_macs = full_covariance_macs + append_macs + inner_macs
            fold_count = outer_k
        added_macs = int(actual_lda_macs - base_lda_macs + calibration_macs)
        resource.update(
            {
                "lda_closed_form_fit_count": full_fit_count + inner_fit_count,
                "estimated_lda_fit_macs": actual_lda_macs,
                "d68_crossfit_fold_count_per_stage": fold_count,
                "d68_full_d65_covariance_fit_count": full_fit_count,
                "d68_inner_d65_covariance_fit_count": inner_fit_count,
                "d68_inner_d65_total_adaptation_macs": inner_macs,
                "d68_calibration_scalar_macs": calibration_macs,
                "d68_total_added_adaptation_macs": added_macs,
                "d68_ground_component_input_count": 0,
                "d68_query_extra_macs": 0,
                "d68_persistent_state_extra_bytes": 0,
                "d68_optimizer_steps_extra": 0,
                "d68_resource_single_affine_state_only": True,
            }
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
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D68ProbeError("D68 training row closure drift")
    orientation_values: list[float] = []
    fit_audits = partition_count = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d68_crossfit_fold_count_per_stage", -1)) != 8
            or int(resource.get("d68_inner_d65_covariance_fit_count", -1)) != 16
            or int(resource.get("d68_ground_component_input_count", -1)) != 0
            or int(resource.get("d68_query_extra_macs", -1)) != 0
            or resource.get("d68_resource_single_affine_state_only") is not True
        ):
            raise D68ProbeError("D68 resource closure drift")
        for phase_name, expected_phase, expected_count in (
            ("before_covariance_audit", "stage2b_before", 6),
            ("final_covariance_audit", "stage2c_final", 11),
        ):
            audit = row["geometry_summary"][phase_name]
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d68_probe_arm": ARM,
                "d68_formula": FORMULA,
                "d68_phase": expected_phase,
                "d68_boundary_status": "crossfitted_signed_frozen_calibration_active",
                "d68_actual_k": 8,
                "d68_class_count": expected_count,
                "d68_old_class_count_for_lifecycle": 6,
                "d68_crossfit_fold_count": 8,
                "d68_ground_component_input_count": 0,
                "d68_class_id_specific_formula": False,
                "d68_old_new_role_specific_query_branch": False,
                "d68_scene_receiver_handle_specific_branch": False,
                "d68_uses_outer_held_or_query": False,
                "d68_hyperparameter_count": 0,
                "d68_query_joint_optimization": False,
                "d68_single_affine_state_only": True,
                "d68_old_row_fp32_bitwise_unchanged": True,
                "d68_calibrated_lifecycle_policy": "freeze_stage2b_signed_old_rows_append_only_signed_new_rows",
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D68ProbeError("D68 exact audit drift")
            orientation = np.asarray(
                audit.get("d68_orientation_by_class"), dtype=np.float64
            )
            if (
                orientation.shape != (expected_count,)
                or not np.all(np.isin(orientation, (-1.0, 1.0)))
            ):
                raise D68ProbeError("D68 orientation closure drift")
            partitions = audit.get("d68_crossfit_partition_audit")
            if (
                not isinstance(partitions, list)
                or len(partitions) != 8
                or any(
                    int(item.get("held_train_intersection_count", -1)) != 0
                    for item in partitions
                )
            ):
                raise D68ProbeError("D68 partition audit drift")
            orientation_values.extend(float(value) for value in orientation)
            fit_audits += 1
            partition_count += len(partitions)
    return {
        "verified_d68_target_row_count": len(target),
        "verified_d68_fit_audit_count": fit_audits,
        "verified_d68_crossfit_partition_count": partition_count,
        "verified_d68_negative_orientation_count": int(
            np.sum(np.asarray(orientation_values) < 0.0)
        ),
        "verified_d68_orientation_min": float(np.min(orientation_values)),
        "verified_d68_orientation_max": float(np.max(orientation_values)),
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D68ProbeError("D68 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d68-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D68ProbeError(f"D68 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = _helper_hashes_for_probe_root(known.probe_root)
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d68_locked_d42_runner", 1
    state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, records, state = build_d68_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D68ProbeError("D68 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d68_arm,
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
        raise D68ProbeError(
            f"D68 fit record count drift: {len(records)} != {EXPECTED_REAL_FIT_COUNT}"
        )
    lifecycle = state.get("d65_lifecycle", {})
    if lifecycle.get("pending") is not None:
        raise D68ProbeError("D68 lifecycle ended with pending Stage2-B state")
    calibration_lifecycle = state.get("calibration_lifecycle", {})
    if (
        calibration_lifecycle.get("pending") is not None
        or int(calibration_lifecycle.get("completed_pairs", -1)) != 30
    ):
        raise D68ProbeError("D68 calibrated lifecycle did not close")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d68.crossfitted_signed_frozen_registry_calibration_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d68_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "ground_component_used": False,
        "ground_component_exclusion_reason": "D22 component is formal_phase2_eligible=false and UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "fit_record_count": len(records),
        "fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D68_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
