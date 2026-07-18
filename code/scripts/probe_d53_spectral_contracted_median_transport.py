#!/usr/bin/env python3
"""D53 probe: spectrally contracted median transport on D45."""

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
SPEC = importlib.util.spec_from_file_location("d53_d45_probe_helper", D45_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D53 could not load D45 helper")
d45 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d45)
d44 = d45.d44
d43 = d45.d43


ARM = "spectral_contracted_median_transport"
STRUCTURE = "d45_plus_spectral_contracted_coordinate_median_transport"
FORMULA = "deltaW=diag(1-rho)*(U*M0.T/||M0||2^2)*W0"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D53ProbeError(RuntimeError):
    pass


def _geometry(
    transformed: Any,
    targets: Any,
    class_count: int,
    k_shot: int,
    base_coefficient: Any,
) -> dict[str, Any]:
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    c = int(class_count)
    k = int(k_shot)
    base = np.asarray(base_coefficient, dtype=np.float64)
    if rows.ndim != 2 or labels.shape != (len(rows),) or c < 2 or k < 1:
        raise D53ProbeError("D53 support shape drift")
    if rows.shape[0] != c * k or not np.isfinite(rows).all():
        raise D53ProbeError("D53 support cardinality/nonfinite drift")
    if base.shape != (c, rows.shape[1]) or not np.isfinite(base).all():
        raise D53ProbeError("D53 base coefficient shape/nonfinite drift")
    if sorted(np.unique(labels).tolist()) != list(range(c)):
        raise D53ProbeError("D53 target registry drift")
    indices = [np.flatnonzero(labels == index) for index in range(c)]
    if any(len(item) != k for item in indices):
        raise D53ProbeError("D53 unequal K drift")
    row_norm_error = float(np.max(np.abs(np.linalg.norm(rows, axis=1) - 1.0)))
    if row_norm_error > UNIT_TOLERANCE:
        raise D53ProbeError("D53 requires D42 global unit-sphere support")

    means = np.stack([rows[item].mean(axis=0) for item in indices], axis=0)
    medians = np.stack([np.median(rows[item], axis=0) for item in indices], axis=0)
    resultant = np.linalg.norm(means, axis=1)
    if (
        np.any(resultant <= EPSILON)
        or np.any(resultant > 1.0 + UNIT_TOLERANCE)
        or not np.isfinite(resultant).all()
    ):
        raise D53ProbeError("D53 centroid norm drift")
    raw_direction = medians - means
    direction_norm = np.linalg.norm(raw_direction, axis=1)
    centered_mean = means - means.mean(axis=0, keepdims=True)
    centered_base = base - base.mean(axis=0, keepdims=True)
    base_discriminant_norm = np.linalg.norm(centered_base, axis=1)
    if np.any(base_discriminant_norm <= EPSILON) or not np.isfinite(
        base_discriminant_norm
    ).all():
        raise D53ProbeError("D53 base discriminant norm drift")
    gamma = 1.0 - resultant
    if np.any(gamma < -UNIT_TOLERANCE) or np.any(gamma >= 1.0):
        raise D53ProbeError("D53 resultant scale drift")

    mean_spectral_norm = float(np.linalg.norm(centered_mean, ord=2))
    direction_spectral_norm = float(np.linalg.norm(raw_direction, ord=2))
    tau = mean_spectral_norm * mean_spectral_norm
    exact_fallback = k <= 2 or float(np.max(direction_norm)) <= EPSILON
    if exact_fallback:
        transport = np.zeros((c, c), dtype=np.float64)
        transport_spectral_norm = 0.0
        transport_bound = 0.0
        transported_direction = np.zeros_like(raw_direction)
        correction = np.zeros_like(raw_direction)
        status = "k1_k2_or_zero_direction_exact_d45_fallback"
    else:
        if not np.isfinite(tau) or tau <= EPSILON:
            raise D53ProbeError("D53 centered mean spectral degeneracy")
        transport = raw_direction @ centered_mean.T / tau
        transport_spectral_norm = float(np.linalg.norm(transport, ord=2))
        transport_bound = direction_spectral_norm / mean_spectral_norm
        if (
            not np.isfinite(transport).all()
            or not np.isfinite(transport_spectral_norm)
            or transport_spectral_norm > transport_bound + 3.0e-10
        ):
            raise D53ProbeError("D53 spectral transport bound drift")
        transported_direction = transport @ centered_base
        correction = gamma[:, None] * transported_direction
        if not np.isfinite(correction).all():
            raise D53ProbeError("D53 correction drift")
        status = "spectral_contracted_median_transport_active"
    return {
        "rows": rows,
        "labels": labels,
        "mean": means,
        "median": medians,
        "resultant": resultant,
        "raw_direction": raw_direction,
        "direction_norm": direction_norm,
        "centered_mean": centered_mean,
        "centered_base": centered_base,
        "base_discriminant_norm": base_discriminant_norm,
        "gamma": gamma,
        "mean_spectral_norm": mean_spectral_norm,
        "direction_spectral_norm": direction_spectral_norm,
        "tau": tau,
        "transport": transport,
        "transport_spectral_norm": transport_spectral_norm,
        "transport_bound": transport_bound,
        "transported_direction": transported_direction,
        "correction": correction,
        "status": status,
        "row_norm_error": row_norm_error,
        "exact_fallback": exact_fallback,
    }


def build_d53_fit(d42: Any) -> Any:
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
        geometry = _geometry(
            transformed, targets, class_count, k_shot, base_coef
        )
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
                "d53_probe_arm": ARM,
                "d53_formula": FORMULA,
                "d53_boundary_status": geometry["status"],
                "d53_actual_k": int(k_shot),
                "d53_class_count": int(class_count),
                "d53_even_k_coordinate_median_policy": "mean_of_two_middle_order_statistics",
                "d53_coordinate_rotation_invariance_claim_allowed": False,
                "d53_class_id_specific_formula": False,
                "d53_old_new_role_specific_branch": False,
                "d53_scene_receiver_handle_specific_branch": False,
                "d53_reliability_uses_outer_held_or_query": False,
                "d53_residual_coefficient_scan_count": 0,
                "d53_scale_hyperparameter_count": 0,
                "d53_query_selected_scale": False,
                "d53_support_transformed_fp32": np.asarray(
                    transformed, dtype=np.float32
                ).tolist(),
                "d53_support_targets": np.asarray(targets, dtype=np.int64).tolist(),
                "d53_support_row_norm_max_abs_error": geometry["row_norm_error"],
                "d53_mean_centroid_fp64": geometry["mean"].tolist(),
                "d53_coordinate_median_centroid_fp64": geometry["median"].tolist(),
                "d53_resultant_norm_by_class": geometry["resultant"].tolist(),
                "d53_raw_median_mean_direction_fp64": geometry["raw_direction"].tolist(),
                "d53_direction_norm_by_class": geometry["direction_norm"].tolist(),
                "d53_centered_mean_centroid_fp64": geometry["centered_mean"].tolist(),
                "d53_base_discriminant_norm_by_class": geometry[
                    "base_discriminant_norm"
                ].tolist(),
                "d53_gamma_by_class": geometry["gamma"].tolist(),
                "d53_mean_spectral_norm": geometry["mean_spectral_norm"],
                "d53_direction_spectral_norm": geometry["direction_spectral_norm"],
                "d53_spectral_tau": geometry["tau"],
                "d53_transport_matrix_fp64": geometry["transport"].tolist(),
                "d53_transport_spectral_norm": geometry["transport_spectral_norm"],
                "d53_transport_spectral_bound": geometry["transport_bound"],
                "d53_transported_direction_fp64": geometry[
                    "transported_direction"
                ].tolist(),
                "d53_coefficient_correction_fp64": geometry["correction"].tolist(),
                "d53_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d53_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d53_actual_coefficient_fp32": final_coef.tolist(),
                "d53_actual_intercept_fp32": final_intercept.tolist(),
                "d53_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit


def _extra_resource(k: int, old_count: int, all_count: int, dimension: int) -> tuple[int, int]:
    if k < 1 or old_count < 2 or all_count < old_count or dimension < 1:
        raise D53ProbeError("D53 resource input drift")
    class_sum = old_count + all_count
    class_square_sum = old_count * old_count + all_count * all_count
    numeric = k * class_sum * dimension + 6 * class_sum * dimension + 8 * class_square_sum * dimension
    comparisons = int(
        class_sum * dimension * k * max(1, math.ceil(math.log2(max(2, k))))
    )
    return int(numeric), comparisons


def _state_dimension(state: Any) -> int:
    log_diag = np.asarray(getattr(state, "log_diag_fp32", None))
    if log_diag.ndim != 1 or len(log_diag) < 1:
        raise D53ProbeError("D53 state feature dimension drift")
    return int(len(log_diag))


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d45._install_d45_core_resource_accounting(d42)
    d45_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d45_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = _state_dimension(result.state)
        numeric, comparisons = _extra_resource(k, old_count, all_count, dimension)
        resource.update(
            {
                "d53_additional_lda_fit_count": 0,
                "d53_additional_optimizer_steps": 0,
                "d53_additional_query_state_count": 0,
                "d53_query_sidecar_bytes": 0,
                "d53_extra_adaptation_mac_equivalents": numeric,
                "d53_coordinate_median_comparison_upper_bound": comparisons,
                "d53_resource_reuses_d45_exact_inventory": True,
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
        raise D53ProbeError("D53 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            rows = np.asarray(audit["d53_support_transformed_fp32"], dtype=np.float64)
            targets = np.asarray(audit["d53_support_targets"], dtype=np.int64)
            c = int(audit["d53_class_count"])
            k = int(audit["d53_actual_k"])
            class_counts.append(c)
            base_coef = np.asarray(
                audit["d53_base_coefficient_fp32"], dtype=np.float64
            )
            base_intercept = np.asarray(
                audit["d53_base_intercept_fp32"], dtype=np.float64
            )
            geometry = _geometry(rows, targets, c, k, base_coef)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d53_probe_arm": ARM,
                "d53_formula": FORMULA,
                "d53_boundary_status": geometry["status"],
                "d53_even_k_coordinate_median_policy": "mean_of_two_middle_order_statistics",
                "d53_coordinate_rotation_invariance_claim_allowed": False,
                "d53_class_id_specific_formula": False,
                "d53_old_new_role_specific_branch": False,
                "d53_scene_receiver_handle_specific_branch": False,
                "d53_reliability_uses_outer_held_or_query": False,
                "d53_residual_coefficient_scan_count": 0,
                "d53_scale_hyperparameter_count": 0,
                "d53_query_selected_scale": False,
                "d53_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D53ProbeError("D53 exact audit drift")
            pairs = (
                ("d53_mean_centroid_fp64", geometry["mean"]),
                ("d53_coordinate_median_centroid_fp64", geometry["median"]),
                ("d53_resultant_norm_by_class", geometry["resultant"]),
                ("d53_raw_median_mean_direction_fp64", geometry["raw_direction"]),
                ("d53_direction_norm_by_class", geometry["direction_norm"]),
                ("d53_centered_mean_centroid_fp64", geometry["centered_mean"]),
                (
                    "d53_base_discriminant_norm_by_class",
                    geometry["base_discriminant_norm"],
                ),
                ("d53_gamma_by_class", geometry["gamma"]),
                ("d53_transport_matrix_fp64", geometry["transport"]),
                (
                    "d53_transported_direction_fp64",
                    geometry["transported_direction"],
                ),
                ("d53_coefficient_correction_fp64", geometry["correction"]),
            )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D53ProbeError("D53 geometry closure drift")
            scalars = (
                ("d53_mean_spectral_norm", geometry["mean_spectral_norm"]),
                ("d53_direction_spectral_norm", geometry["direction_spectral_norm"]),
                ("d53_spectral_tau", geometry["tau"]),
                ("d53_transport_spectral_norm", geometry["transport_spectral_norm"]),
                ("d53_transport_spectral_bound", geometry["transport_bound"]),
            )
            if any(
                not math.isclose(
                    float(audit.get(name)), value, rel_tol=0.0, abs_tol=3.0e-7
                )
                for name, value in scalars
            ):
                raise D53ProbeError("D53 spectral scalar closure drift")
            if geometry["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    base_coef + geometry["correction"], base_intercept
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d53_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D53ProbeError("D53 actual coefficient drift")
            if not _allclose(audit["d53_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D53ProbeError("D53 actual intercept drift")
        numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]), class_counts[0], class_counts[1], 288
        )
        if (
            resource.get("d53_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d53_coordinate_median_comparison_upper_bound") != comparisons
            or resource.get("d53_resource_reuses_d45_exact_inventory") is not True
        ):
            raise D53ProbeError("D53 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d45.ARM
            audit["d43_covariance_structure"] = d45.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d53_extra_adaptation_mac_equivalents"]
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
        raise D53ProbeError("D53 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d53_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d53-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D53ProbeError(f"D53 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d53_d45_helper_sha256": d43._sha256(D45_HELPER_PATH),
        "d53_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d53_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d53_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d53_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D53ProbeError("D53 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d53_arm,
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
        "schema": "cvs.phase2.d53.spectral_contracted_median_transport_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d53_arm,
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
    (output / "D53_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
