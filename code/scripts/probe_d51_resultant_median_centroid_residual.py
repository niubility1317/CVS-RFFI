#!/usr/bin/env python3
"""D51 probe: resultant-scaled coordinate-median centroid residual on D45."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D45_HELPER_PATH = SCRIPT_DIR / "probe_d45_inner_loo_reliability_fusion.py"
SPEC = importlib.util.spec_from_file_location("d51_d45_probe_helper", D45_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D51 could not load D45 helper")
d45 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d45)
d44 = d45.d44
d43 = d45.d43


ARM = "resultant_median_centroid_residual"
STRUCTURE = "d45_plus_resultant_scaled_coordinate_median_centroid_direction_residual"
FORMULA = "deltaW_c=(1-rho_c)*(q_c-p_c)/class_centered_support_rms(q-p)"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D51ProbeError(RuntimeError):
    pass


def _geometry(
    transformed: Any,
    targets: Any,
    class_count: int,
    k_shot: int,
) -> dict[str, Any]:
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    c = int(class_count)
    k = int(k_shot)
    if rows.ndim != 2 or labels.shape != (len(rows),) or c < 2 or k < 1:
        raise D51ProbeError("D51 support shape drift")
    if rows.shape[0] != c * k or not np.isfinite(rows).all():
        raise D51ProbeError("D51 support cardinality/nonfinite drift")
    if sorted(np.unique(labels).tolist()) != list(range(c)):
        raise D51ProbeError("D51 target registry drift")
    indices = [np.flatnonzero(labels == index) for index in range(c)]
    if any(len(item) != k for item in indices):
        raise D51ProbeError("D51 unequal K drift")
    row_norm_error = float(np.max(np.abs(np.linalg.norm(rows, axis=1) - 1.0)))
    if row_norm_error > UNIT_TOLERANCE:
        raise D51ProbeError("D51 requires D42 global unit-sphere support")

    means = np.stack([rows[item].mean(axis=0) for item in indices], axis=0)
    medians = np.stack([np.median(rows[item], axis=0) for item in indices], axis=0)
    resultant = np.linalg.norm(means, axis=1)
    median_norm = np.linalg.norm(medians, axis=1)
    if (
        np.any(resultant <= EPSILON)
        or np.any(resultant > 1.0 + UNIT_TOLERANCE)
        or np.any(median_norm <= EPSILON)
        or not np.isfinite(resultant).all()
        or not np.isfinite(median_norm).all()
    ):
        raise D51ProbeError("D51 centroid norm drift")
    mean_prototype = means / resultant[:, None]
    median_prototype = medians / median_norm[:, None]
    residual_direction = median_prototype - mean_prototype
    gamma = 1.0 - resultant
    if np.any(gamma < -UNIT_TOLERANCE) or np.any(gamma >= 1.0):
        raise D51ProbeError("D51 resultant scale drift")

    exact_fallback = k <= 2 or float(np.max(np.abs(residual_direction))) <= EPSILON
    if exact_fallback:
        scale = None
        correction = np.zeros_like(residual_direction)
        status = "k1_k2_or_zero_residual_exact_d45_fallback"
    else:
        scale = float(
            d44._class_centered_logit_rms(
                rows,
                residual_direction,
                np.zeros(c, dtype=np.float64),
            )
        )
        if not np.isfinite(scale) or scale <= EPSILON:
            raise D51ProbeError("D51 residual RMS degeneracy")
        correction = gamma[:, None] * residual_direction / scale
        if not np.isfinite(correction).all():
            raise D51ProbeError("D51 correction drift")
        status = "resultant_scaled_coordinate_median_residual_active"
    return {
        "rows": rows,
        "labels": labels,
        "mean": means,
        "median": medians,
        "resultant": resultant,
        "median_norm": median_norm,
        "mean_prototype": mean_prototype,
        "median_prototype": median_prototype,
        "residual_direction": residual_direction,
        "gamma": gamma,
        "scale": scale,
        "correction": correction,
        "status": status,
        "row_norm_error": row_norm_error,
        "exact_fallback": exact_fallback,
    }


def build_d51_fit(d42: Any) -> Any:
    base_fit = d45.build_inner_loo_reliability_fit(d42)

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        base_coef, base_intercept, base_audit = base_fit(
            transformed, targets, class_count, k_shot
        )
        geometry = _geometry(transformed, targets, class_count, k_shot)
        if geometry["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            combined_coef = np.asarray(base_coef, dtype=np.float64) + geometry["correction"]
            final_coef64, final_intercept64 = d43._center_affine_scores(
                combined_coef, np.asarray(base_intercept, dtype=np.float64)
            )
            final_coef = np.asarray(final_coef64, dtype=np.float32)
            final_intercept = np.asarray(final_intercept64, dtype=np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d51_probe_arm": ARM,
                "d51_formula": FORMULA,
                "d51_boundary_status": geometry["status"],
                "d51_actual_k": int(k_shot),
                "d51_class_count": int(class_count),
                "d51_even_k_coordinate_median_policy": "mean_of_two_middle_order_statistics",
                "d51_coordinate_rotation_invariance_claim_allowed": False,
                "d51_class_id_specific_formula": False,
                "d51_old_new_role_specific_branch": False,
                "d51_scene_receiver_handle_specific_branch": False,
                "d51_reliability_uses_outer_held_or_query": False,
                "d51_residual_coefficient_scan_count": 0,
                "d51_support_transformed_fp32": np.asarray(
                    transformed, dtype=np.float32
                ).tolist(),
                "d51_support_targets": np.asarray(targets, dtype=np.int64).tolist(),
                "d51_support_row_norm_max_abs_error": geometry["row_norm_error"],
                "d51_mean_centroid_fp64": geometry["mean"].tolist(),
                "d51_coordinate_median_centroid_fp64": geometry["median"].tolist(),
                "d51_resultant_norm_by_class": geometry["resultant"].tolist(),
                "d51_median_norm_by_class": geometry["median_norm"].tolist(),
                "d51_mean_unit_prototype_fp64": geometry["mean_prototype"].tolist(),
                "d51_median_unit_prototype_fp64": geometry["median_prototype"].tolist(),
                "d51_residual_direction_fp64": geometry["residual_direction"].tolist(),
                "d51_gamma_by_class": geometry["gamma"].tolist(),
                "d51_residual_support_logit_rms": geometry["scale"],
                "d51_coefficient_correction_fp64": geometry["correction"].tolist(),
                "d51_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d51_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d51_actual_coefficient_fp32": final_coef.tolist(),
                "d51_actual_intercept_fp32": final_intercept.tolist(),
                "d51_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit


def _extra_resource(k: int, old_count: int, all_count: int, dimension: int) -> tuple[int, int]:
    if k < 1 or old_count < 2 or all_count < old_count or dimension < 1:
        raise D51ProbeError("D51 resource input drift")
    class_sum = old_count + all_count
    class_square_sum = old_count * old_count + all_count * all_count
    numeric = (
        k * class_sum * dimension
        + 13 * class_sum * dimension
        + 2 * k * class_square_sum * dimension
        + 4 * k * class_square_sum
    )
    comparisons = int(
        class_sum * dimension * k * max(1, math.ceil(math.log2(max(2, k))))
    )
    return int(numeric), comparisons


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d45._install_d45_core_resource_accounting(d42)
    d45_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d45_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = int(resource["coefficient_dimension"])
        numeric, comparisons = _extra_resource(k, old_count, all_count, dimension)
        resource.update(
            {
                "d51_additional_lda_fit_count": 0,
                "d51_additional_optimizer_steps": 0,
                "d51_additional_query_state_count": 0,
                "d51_query_sidecar_bytes": 0,
                "d51_extra_adaptation_mac_equivalents": numeric,
                "d51_coordinate_median_comparison_upper_bound": comparisons,
                "d51_resource_reuses_d45_exact_inventory": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + numeric
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _allclose(actual: Any, expected: Any, atol: float = 2.0e-7) -> bool:
    left = np.asarray(actual)
    right = np.asarray(expected)
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=atol)


def _verify_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    target_rows = [
        row
        for row in training_rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(target_rows) != 30:
        raise D51ProbeError("D51 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            rows = np.asarray(audit["d51_support_transformed_fp32"], dtype=np.float64)
            targets = np.asarray(audit["d51_support_targets"], dtype=np.int64)
            c = int(audit["d51_class_count"])
            k = int(audit["d51_actual_k"])
            class_counts.append(c)
            geometry = _geometry(rows, targets, c, k)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d51_probe_arm": ARM,
                "d51_formula": FORMULA,
                "d51_boundary_status": geometry["status"],
                "d51_even_k_coordinate_median_policy": "mean_of_two_middle_order_statistics",
                "d51_coordinate_rotation_invariance_claim_allowed": False,
                "d51_class_id_specific_formula": False,
                "d51_old_new_role_specific_branch": False,
                "d51_scene_receiver_handle_specific_branch": False,
                "d51_reliability_uses_outer_held_or_query": False,
                "d51_residual_coefficient_scan_count": 0,
                "d51_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D51ProbeError("D51 exact audit drift")
            pairs = (
                ("d51_mean_centroid_fp64", geometry["mean"]),
                ("d51_coordinate_median_centroid_fp64", geometry["median"]),
                ("d51_resultant_norm_by_class", geometry["resultant"]),
                ("d51_median_norm_by_class", geometry["median_norm"]),
                ("d51_mean_unit_prototype_fp64", geometry["mean_prototype"]),
                ("d51_median_unit_prototype_fp64", geometry["median_prototype"]),
                ("d51_residual_direction_fp64", geometry["residual_direction"]),
                ("d51_gamma_by_class", geometry["gamma"]),
                ("d51_coefficient_correction_fp64", geometry["correction"]),
            )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D51ProbeError("D51 geometry closure drift")
            if geometry["scale"] is None:
                if audit.get("d51_residual_support_logit_rms") is not None:
                    raise D51ProbeError("D51 fallback RMS drift")
            elif not math.isclose(
                float(audit["d51_residual_support_logit_rms"]),
                geometry["scale"],
                rel_tol=0.0,
                abs_tol=3.0e-7,
            ):
                raise D51ProbeError("D51 residual RMS closure drift")
            base_coef = np.asarray(audit["d51_base_coefficient_fp32"], dtype=np.float64)
            base_intercept = np.asarray(audit["d51_base_intercept_fp32"], dtype=np.float64)
            if geometry["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    base_coef + geometry["correction"], base_intercept
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d51_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D51ProbeError("D51 actual coefficient drift")
            if not _allclose(audit["d51_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D51ProbeError("D51 actual intercept drift")
        numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]), class_counts[0], class_counts[1], 288
        )
        if (
            resource.get("d51_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d51_coordinate_median_comparison_upper_bound") != comparisons
            or resource.get("d51_resource_reuses_d45_exact_inventory") is not True
        ):
            raise D51ProbeError("D51 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d45.ARM
            audit["d43_covariance_structure"] = d45.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d51_extra_adaptation_mac_equivalents"]
        )
    d45._verify_d45_fit_audits(sanitized)
    return len(target_rows)


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D51ProbeError("D51 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d51_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d51-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D51ProbeError(f"D51 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d51_d45_helper_sha256": d43._sha256(D45_HELPER_PATH),
        "d51_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d51_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d51_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d51_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D51ProbeError("D51 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d51_arm,
            probe_script_sha256=script_sha,
            extra_source_closure=helper_hashes,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
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
    evidence = _verify_output(output, script_sha, helper_hashes)
    metadata = {
        "schema": "cvs.phase2.d51.resultant_median_centroid_residual_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d51_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D51_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
