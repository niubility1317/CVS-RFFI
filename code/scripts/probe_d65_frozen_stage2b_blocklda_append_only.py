#!/usr/bin/env python3
"""D65 support-only frozen Stage2-B block-LDA append-only probe."""

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
D43_HELPER_PATH = SCRIPT_DIR / "probe_d43_structured_covariance.py"
SPEC = importlib.util.spec_from_file_location("d65_d43_probe_helper", D43_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D65 could not load D43 helper")
d43 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d43)


ARM = "frozen_stage2b_blocklda_append_only"
STRUCTURE = "stage2b_frozen_three_block_auto_shrinkage_append_only_rows"
STATE_COVARIANCE_POLICY = "sklearn_lsqr_auto_shrinkage_equal_prior"
FORMULA = (
    "fit one three-block covariance on Stage2-B old support; freeze it; build every "
    "class row by the same Sigma_B^-1 mean formula and append new rows bitwise"
)
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D65ProbeError(RuntimeError):
    pass


def _validate_symmetric_support(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(int(np.sum(y == index)) != int(k_shot) for index in range(int(class_count)))
    ):
        raise D65ProbeError("D65 requires finite exact symmetric support")
    return x, y


def _means(rows: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    return np.stack(
        [rows[labels == index].mean(axis=0) for index in range(int(class_count))]
    )


def _covariance_sha256(covariance: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(covariance, dtype=np.float64).tobytes()
    ).hexdigest()


def _compile_rows(
    covariance: np.ndarray, means: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    coefficient = np.linalg.lstsq(covariance, means.T, rcond=None)[0].T
    intercept = -0.5 * np.diag(means @ coefficient.T)
    residual = float(
        np.max(np.abs(covariance @ coefficient.T - means.T))
    )
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise D65ProbeError("D65 affine rows became non-finite")
    return coefficient.astype(np.float32), intercept.astype(np.float32), residual


def _fit_stage2b_covariance(
    d42: Any,
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    means = _means(rows, labels, class_count)
    residuals = rows - means[labels]
    residual_rank = int(np.linalg.matrix_rank(residuals))
    residual_energy = float(np.sum(residuals**2))
    fallback = bool(
        int(k_shot) == 1
        or residual_rank == 0
        or not np.isfinite(residual_energy)
        or residual_energy <= float(d42.ENERGY_EPSILON)
    )
    if fallback:
        covariance = np.eye(rows.shape[1], dtype=np.float64)
    else:
        priors = np.full(int(class_count), 1.0 / int(class_count), dtype=np.float64)
        estimator = d43.LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
        )
        estimator.fit(rows, labels)
        if not np.array_equal(
            np.asarray(estimator.classes_), np.arange(int(class_count), dtype=np.int64)
        ):
            raise D65ProbeError("D65 sklearn class order drift")
        covariance = d43._structured_covariance(
            np.asarray(estimator.covariance_, dtype=np.float64),
            "block3_centered",
            tuple(d42.BLOCK_SLICES),
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
        raise D65ProbeError("D65 Stage2-B covariance is not positive definite")
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    return covariance, means, fallback, residual_rank, residual_energy


def build_d65_fit(
    d42: Any,
) -> tuple[Callable[..., Any], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {"pending": None, "completed_pairs": 0}

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x, y = _validate_symmetric_support(rows, labels, class_count, k_shot)
        pending = lifecycle["pending"]
        if pending is None:
            covariance, class_means, fallback, residual_rank, residual_energy = (
                _fit_stage2b_covariance(d42, x, y, class_count, k_shot)
            )
            coefficient, intercept, equation_residual = _compile_rows(
                covariance, class_means
            )
            phase = "stage2b_fit_and_freeze"
            old_count, new_count = int(class_count), 0
            lifecycle["pending"] = {
                "covariance": covariance,
                "covariance_sha256": _covariance_sha256(covariance),
                "coefficient": coefficient.copy(),
                "intercept": intercept.copy(),
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "fallback": fallback,
                "residual_rank": residual_rank,
                "residual_energy": residual_energy,
                "condition": float(np.linalg.cond(covariance)),
            }
            old_rows_bitwise = True
        else:
            old_count = int(pending["class_count"])
            new_count = int(class_count) - old_count
            old_labels = y[: old_count * int(k_shot)]
            appended_labels = y[old_count * int(k_shot) :]
            if (
                int(k_shot) != int(pending["k_shot"])
                or new_count <= 0
                or not np.array_equal(np.unique(old_labels), np.arange(old_count))
                or not np.array_equal(
                    np.unique(appended_labels), np.arange(old_count, int(class_count))
                )
                or any(
                    int(np.sum(old_labels == index)) != int(k_shot)
                    for index in range(old_count)
                )
                or any(
                    int(np.sum(appended_labels == index)) != int(k_shot)
                    for index in range(old_count, int(class_count))
                )
            ):
                raise D65ProbeError("D65 Stage2-B/Stage2-C lifecycle order drift")
            covariance = np.asarray(pending["covariance"], dtype=np.float64)
            new_rows = x[old_count * int(k_shot) :]
            new_labels = y[old_count * int(k_shot) :] - old_count
            new_means = _means(new_rows, new_labels, new_count)
            new_coefficient, new_intercept, equation_residual = _compile_rows(
                covariance, new_means
            )
            coefficient = np.concatenate(
                [np.asarray(pending["coefficient"], dtype=np.float32), new_coefficient],
                axis=0,
            )
            intercept = np.concatenate(
                [np.asarray(pending["intercept"], dtype=np.float32), new_intercept],
                axis=0,
            )
            old_rows_bitwise = bool(
                np.array_equal(coefficient[:old_count], pending["coefficient"])
                and np.array_equal(intercept[:old_count], pending["intercept"])
            )
            if not old_rows_bitwise:
                raise D65ProbeError("D65 FP32 old rows changed during registration")
            fallback = bool(pending["fallback"])
            residual_rank = int(pending["residual_rank"])
            residual_energy = float(pending["residual_energy"])
            phase = "stage2c_append_only"
            lifecycle["pending"] = None
            lifecycle["completed_pairs"] = int(lifecycle["completed_pairs"]) + 1

        scores = x.astype(np.float32) @ coefficient.T + intercept[None, :]
        support_accuracy = float(np.mean(np.argmax(scores, axis=1) == y))
        covariance_sha = _covariance_sha256(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        audit = {
            "solver": "lsqr",
            "shrinkage": "auto_stage2b_frozen",
            "prior_policy": "equal_common_term_omitted",
            "covariance_policy": STATE_COVARIANCE_POLICY,
            "unit_covariance_fallback": fallback,
            "support_rows": int(len(x)),
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "coefficient_source": "frozen_stage2b_block_sigma_inverse_class_mean",
            "covariance_equation_residual_max": equation_residual,
            "d43_probe_arm": ARM,
            "d43_covariance_structure": STRUCTURE,
            "d43_class_common_affine_omitted": True,
            "d65_probe_arm": ARM,
            "d65_formula": FORMULA,
            "d65_phase": phase,
            "d65_stage2b_covariance_sha256": covariance_sha,
            "d65_stage2b_covariance_eigenvalue_min": float(np.min(eigenvalues)),
            "d65_stage2b_covariance_eigenvalue_max": float(np.max(eigenvalues)),
            "d65_stage2b_covariance_condition_number": float(
                np.max(eigenvalues) / np.min(eigenvalues)
            ),
            "d65_stage2b_residual_rank": residual_rank,
            "d65_stage2b_residual_energy": residual_energy,
            "d65_old_class_count": old_count,
            "d65_appended_class_count": new_count,
            "d65_old_row_fp32_bitwise_unchanged": old_rows_bitwise,
            "d65_compiled_support_accuracy": support_accuracy,
            "d65_actual_k": int(k_shot),
            "d65_class_id_specific_formula": False,
            "d65_old_new_role_specific_query_branch": False,
            "d65_scene_receiver_handle_specific_branch": False,
            "d65_uses_outer_held_or_query": False,
            "d65_hyperparameter_count": 0,
            "d65_query_joint_optimization": False,
            "d65_single_affine_state_only": True,
            "d65_actual_coefficient_fp32": coefficient.tolist(),
            "d65_actual_intercept_fp32": intercept.tolist(),
        }
        records.append(
            {
                "phase": phase,
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "covariance_sha256": covariance_sha,
                "old_row_fp32_bitwise_unchanged": old_rows_bitwise,
                "support_accuracy": support_accuracy,
            }
        )
        return coefficient, intercept, audit

    return fit, records, lifecycle


def _state_old_rows_equal(before: Any, final: Any) -> bool:
    count = len(before.classes)
    if tuple(final.classes[:count]) != tuple(before.classes):
        return False
    names = (
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "coef_fp32",
        "intercept_fp32",
    )
    for name in names:
        first = np.asarray(getattr(before, name))
        second = np.asarray(getattr(final, name))
        if first.shape[0] == 0:
            if second.shape[0] != 0:
                return False
        elif not np.array_equal(first, second[:count]):
            return False
    return np.array_equal(before.log_diag_fp32, final.log_diag_fp32)


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        int8_equal = _state_old_rows_equal(result.before_state, result.state)
        fp32_equal = _state_old_rows_equal(
            result.matched_fp32_before_state, result.matched_fp32_state
        )
        if not int8_equal or not fp32_equal:
            raise D65ProbeError("D65 compiled old-state rows changed during append")
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        old_k, new_k = int(resource["old_k_shot"]), int(resource["new_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        new_count = all_count - old_count
        covariance_fit_macs = int(original_macs(old_count * old_k, old_count))
        append_mean_macs = int(new_count * new_k * dimension)
        append_row_solve_macs = int(new_count * dimension * dimension)
        append_intercept_macs = int(new_count * (2 * dimension + 1))
        append_macs = append_mean_macs + append_row_solve_macs + append_intercept_macs
        resource.update(
            {
                "lda_closed_form_fit_count": 1,
                "estimated_lda_fit_macs": covariance_fit_macs + append_macs,
                "d65_stage2b_covariance_fit_count": 1,
                "d65_append_row_count": new_count,
                "d65_append_row_macs": append_macs,
                "d65_int8_old_rows_bitwise_unchanged": int8_equal,
                "d65_fp32_old_rows_bitwise_unchanged": fp32_equal,
                "d65_query_extra_macs": 0,
                "d65_persistent_state_extra_bytes": 0,
                "d65_optimizer_steps_extra": 0,
                "d65_resource_single_affine_state_only": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"]
            + covariance_fit_macs
            + append_macs
        )
        geometry = dict(result.geometry_audit)
        geometry.update(
            {
                "d65_int8_old_rows_bitwise_unchanged": int8_equal,
                "d65_fp32_old_rows_bitwise_unchanged": fp32_equal,
                "d65_append_only_registry_closure": True,
            }
        )
        return replace(result, resource_audit=resource, geometry_audit=geometry)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D65ProbeError("D65 training row closure drift")
    fit_audits = covariance_pairs = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d65_stage2b_covariance_fit_count", -1)) != 1
            or int(resource.get("d65_append_row_count", -1)) != 5
            or resource.get("d65_int8_old_rows_bitwise_unchanged") is not True
            or resource.get("d65_fp32_old_rows_bitwise_unchanged") is not True
            or int(resource.get("d65_query_extra_macs", -1)) != 0
            or resource.get("d65_resource_single_affine_state_only") is not True
        ):
            raise D65ProbeError("D65 resource/state closure drift")
        before = row["geometry_summary"]["before_covariance_audit"]
        final = row["geometry_summary"]["final_covariance_audit"]
        for audit, phase, class_count, appended in (
            (before, "stage2b_fit_and_freeze", 6, 0),
            (final, "stage2c_append_only", 11, 5),
        ):
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d65_probe_arm": ARM,
                "d65_formula": FORMULA,
                "d65_phase": phase,
                "d65_actual_k": 8,
                "d65_old_class_count": 6,
                "d65_appended_class_count": appended,
                "d65_old_row_fp32_bitwise_unchanged": True,
                "d65_class_id_specific_formula": False,
                "d65_old_new_role_specific_query_branch": False,
                "d65_scene_receiver_handle_specific_branch": False,
                "d65_uses_outer_held_or_query": False,
                "d65_hyperparameter_count": 0,
                "d65_query_joint_optimization": False,
                "d65_single_affine_state_only": True,
                "class_count": class_count,
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D65ProbeError("D65 exact audit drift")
            fit_audits += 1
        if before["d65_stage2b_covariance_sha256"] != final["d65_stage2b_covariance_sha256"]:
            raise D65ProbeError("D65 frozen covariance hash changed")
        covariance_pairs += 1
    return {
        "verified_d65_target_row_count": len(target),
        "verified_d65_fit_audit_count": fit_audits,
        "verified_d65_frozen_covariance_pair_count": covariance_pairs,
        "verified_d65_int8_old_rows_bitwise_unchanged": True,
        "verified_d65_fp32_old_rows_bitwise_unchanged": True,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D65ProbeError("D65 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d65-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D65ProbeError(f"D65 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {"d65_d43_helper_sha256": d43._sha256(D43_HELPER_PATH)}
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d65_locked_d42_runner", 1
    records: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {}
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, records, lifecycle = build_d65_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D65ProbeError("D65 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d65_arm,
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
    if len(records) != 60 or lifecycle.get("pending") is not None or lifecycle.get("completed_pairs") != 30:
        raise D65ProbeError("D65 lifecycle call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d65.frozen_stage2b_blocklda_append_only_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d65_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "fit_execution_count": len(records),
        "fit_record_sha256": record_sha,
        "completed_before_final_pairs": int(lifecycle["completed_pairs"]),
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D65_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
