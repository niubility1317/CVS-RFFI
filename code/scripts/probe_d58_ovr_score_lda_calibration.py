#!/usr/bin/env python3
"""D58 probe: support inner-held one-vs-rest score-space LDA calibration."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D56_HELPER_PATH = SCRIPT_DIR / "probe_d56_loo_confusion_flow_intercept.py"
SPEC = importlib.util.spec_from_file_location("d58_d56_probe_helper", D56_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D58 could not load D56 helper")
d56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d56)
d46 = d56.d46
d45 = d46.d45
d44 = d45.d44
d43 = d45.d43


ARM = "ovr_score_lda_calibration"
STRUCTURE = "d46_plus_support_inner_held_ovr_score_lda_calibration"
FORMULA = "a=(mu_pos-mu_neg)/pooled_var;d=-0.5*a*(mu_pos+mu_neg);normalize_mean_a"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D58ProbeError(RuntimeError):
    pass


def _ovr_score_lda_calibration(
    audit: dict[str, Any], k_shot: int, class_count: int
) -> dict[str, Any]:
    k = int(k_shot)
    c = int(class_count)
    if c < 2 or k < 1:
        raise D58ProbeError("D58 calibration input drift")
    if k <= 2:
        zeros = np.zeros(c, dtype=np.float64)
        ones = np.ones(c, dtype=np.float64)
        return {
            "positive_mean": None,
            "negative_mean": None,
            "positive_variance": None,
            "negative_variance": None,
            "pooled_variance": None,
            "raw_slope": ones,
            "mean_raw_slope": 1.0,
            "normalized_slope": ones,
            "normalized_intercept": zeros,
            "base_prediction_by_fold": None,
            "calibrated_prediction_by_fold": None,
            "base_correct_count": 0,
            "calibrated_correct_count": 0,
            "status": "k1_k2_exact_d46_fallback",
            "exact_fallback": True,
        }
    full = np.asarray(audit.get("d56_full_inner_held_score_fp64"), dtype=np.float64)
    block = np.asarray(audit.get("d56_block_inner_held_score_fp64"), dtype=np.float64)
    true_classes = audit.get("d56_held_true_class_by_fold")
    stored_predictions = audit.get("d56_held_prediction_by_fold")
    full_weight = np.asarray(audit.get("d46_full_weight_by_class"), dtype=np.float64)
    block_weight = np.asarray(audit.get("d46_block_weight_by_class"), dtype=np.float64)
    full_scale = float(audit.get("d46_full_support_logit_rms", float("nan")))
    block_scale = float(audit.get("d46_block_support_logit_rms", float("nan")))
    if (
        full.shape != (k, c, c)
        or block.shape != (k, c, c)
        or not np.isfinite(full).all()
        or not np.isfinite(block).all()
        or not isinstance(true_classes, list)
        or len(true_classes) != k
        or not isinstance(stored_predictions, list)
        or len(stored_predictions) != k
        or full_weight.shape != (c,)
        or block_weight.shape != (c,)
        or not np.isfinite(full_weight).all()
        or not np.isfinite(block_weight).all()
        or not np.allclose(full_weight + block_weight, 1.0, rtol=0.0, atol=3.0e-7)
        or not np.isfinite(full_scale)
        or not np.isfinite(block_scale)
        or full_scale <= 0.0
        or block_scale <= 0.0
    ):
        raise D58ProbeError("D58 D56 held-score evidence drift")
    fused = (
        full * full_weight[None, None, :] / full_scale
        + block * block_weight[None, None, :] / block_scale
    )
    normalized_true = [[int(value) for value in fold] for fold in true_classes]
    if any(len(fold) != c or sorted(fold) != list(range(c)) for fold in normalized_true):
        raise D58ProbeError("D58 true-class closure drift")
    if np.argmax(fused, axis=2).astype(np.int64).tolist() != stored_predictions:
        raise D58ProbeError("D58 D56 held-prediction closure drift")
    flat_scores = fused.reshape(k * c, c)
    flat_true = np.asarray(normalized_true, dtype=np.int64).reshape(k * c)
    positive_mean = np.zeros(c, dtype=np.float64)
    negative_mean = np.zeros(c, dtype=np.float64)
    positive_variance = np.zeros(c, dtype=np.float64)
    negative_variance = np.zeros(c, dtype=np.float64)
    for candidate_class in range(c):
        positive = flat_scores[flat_true == candidate_class, candidate_class]
        negative = flat_scores[flat_true != candidate_class, candidate_class]
        if positive.shape != (k,) or negative.shape != (k * (c - 1),):
            raise D58ProbeError("D58 one-vs-rest cardinality drift")
        positive_mean[candidate_class] = float(np.mean(positive))
        negative_mean[candidate_class] = float(np.mean(negative))
        positive_variance[candidate_class] = float(
            np.mean((positive - positive_mean[candidate_class]) ** 2)
        )
        negative_variance[candidate_class] = float(
            np.mean((negative - negative_mean[candidate_class]) ** 2)
        )
    pooled_variance = 0.5 * (positive_variance + negative_variance)
    separation = positive_mean - negative_mean
    valid = bool(
        np.isfinite(positive_mean).all()
        and np.isfinite(negative_mean).all()
        and np.isfinite(pooled_variance).all()
        and np.all(pooled_variance > EPSILON)
        and np.all(separation > EPSILON)
    )
    if valid:
        raw_slope = separation / pooled_variance
        mean_raw_slope = float(np.mean(raw_slope))
        if not np.isfinite(mean_raw_slope) or mean_raw_slope <= EPSILON:
            raise D58ProbeError("D58 mean slope drift")
        normalized_slope = raw_slope / mean_raw_slope
        normalized_intercept = (
            -0.5 * raw_slope * (positive_mean + negative_mean) / mean_raw_slope
        )
        calibrated = (
            fused * normalized_slope[None, None, :]
            + normalized_intercept[None, None, :]
        )
        status = "support_inner_held_ovr_score_lda_calibration_active"
        exact_fallback = False
    else:
        raw_slope = np.ones(c, dtype=np.float64)
        mean_raw_slope = 1.0
        normalized_slope = np.ones(c, dtype=np.float64)
        normalized_intercept = np.zeros(c, dtype=np.float64)
        calibrated = fused.copy()
        status = "nonpositive_or_degenerate_exact_d46_fallback"
        exact_fallback = True
    base_prediction = np.argmax(fused, axis=2).astype(np.int64)
    calibrated_prediction = np.argmax(calibrated, axis=2).astype(np.int64)
    truth = np.asarray(normalized_true, dtype=np.int64)
    return {
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_variance": positive_variance,
        "negative_variance": negative_variance,
        "pooled_variance": pooled_variance,
        "raw_slope": raw_slope,
        "mean_raw_slope": mean_raw_slope,
        "normalized_slope": normalized_slope,
        "normalized_intercept": normalized_intercept,
        "base_prediction_by_fold": base_prediction.tolist(),
        "calibrated_prediction_by_fold": calibrated_prediction.tolist(),
        "base_correct_count": int(np.sum(base_prediction == truth)),
        "calibrated_correct_count": int(np.sum(calibrated_prediction == truth)),
        "status": status,
        "exact_fallback": exact_fallback,
    }


def build_d58_fit(d42: Any) -> Any:
    base_fit = d56.build_d56_fit(d42)

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        _, _, base_audit = base_fit(
            transformed, targets, class_count, k_shot
        )
        base_coef = np.asarray(base_audit["d56_base_coefficient_fp32"], dtype=np.float32)
        base_intercept = np.asarray(base_audit["d56_base_intercept_fp32"], dtype=np.float32)
        calibration = _ovr_score_lda_calibration(base_audit, k_shot, class_count)
        if calibration["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            final_coef64, final_intercept64 = d43._center_affine_scores(
                calibration["normalized_slope"][:, None]
                * np.asarray(base_coef, dtype=np.float64),
                calibration["normalized_slope"]
                * np.asarray(base_intercept, dtype=np.float64)
                + calibration["normalized_intercept"],
            )
            final_coef = np.asarray(final_coef64, dtype=np.float32)
            final_intercept = np.asarray(final_intercept64, dtype=np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d58_probe_arm": ARM,
                "d58_formula": FORMULA,
                "d58_boundary_status": calibration["status"],
                "d58_actual_k": int(k_shot),
                "d58_class_count": int(class_count),
                "d58_class_id_specific_formula": False,
                "d58_old_new_role_specific_branch": False,
                "d58_scene_receiver_handle_specific_branch": False,
                "d58_reliability_uses_outer_held_or_query": False,
                "d58_hyperparameter_count": 0,
                "d58_query_selected_calibration": False,
                "d58_balanced_positive_negative_weight": True,
                "d58_positive_mean_by_class": None if calibration["positive_mean"] is None else calibration["positive_mean"].tolist(),
                "d58_negative_mean_by_class": None if calibration["negative_mean"] is None else calibration["negative_mean"].tolist(),
                "d58_positive_variance_by_class": None if calibration["positive_variance"] is None else calibration["positive_variance"].tolist(),
                "d58_negative_variance_by_class": None if calibration["negative_variance"] is None else calibration["negative_variance"].tolist(),
                "d58_pooled_variance_by_class": None if calibration["pooled_variance"] is None else calibration["pooled_variance"].tolist(),
                "d58_raw_slope_by_class": calibration["raw_slope"].tolist(),
                "d58_mean_raw_slope": calibration["mean_raw_slope"],
                "d58_normalized_slope_by_class": calibration["normalized_slope"].tolist(),
                "d58_normalized_intercept_by_class": calibration["normalized_intercept"].tolist(),
                "d58_base_prediction_by_fold": calibration["base_prediction_by_fold"],
                "d58_calibrated_prediction_by_fold": calibration["calibrated_prediction_by_fold"],
                "d58_base_correct_count": calibration["base_correct_count"],
                "d58_calibrated_correct_count": calibration["calibrated_correct_count"],
                "d58_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d58_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d58_actual_coefficient_fp32": final_coef.tolist(),
                "d58_actual_intercept_fp32": final_intercept.tolist(),
                "d58_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit


def _extra_resource(
    k: int,
    old_count: int,
    all_count: int,
    dimension: int,
) -> tuple[int, int]:
    if k < 1 or old_count < 2 or all_count < old_count or dimension < 1:
        raise D58ProbeError("D58 resource input drift")
    if k <= 2:
        return 0, 0
    # D56 already materializes every inner-held score. D58 only computes balanced
    # one-vs-rest moments and applies one affine row scale/intercept per class.
    numeric = sum(
        8 * k * c * c + 24 * k * c + 12 * c + 2 * (dimension + 1) * c
        for c in (old_count, all_count)
    )
    comparisons = sum(2 * k * c + 3 * c for c in (old_count, all_count))
    return int(numeric), int(comparisons)


def _state_dimension(state: Any) -> int:
    log_diag = np.asarray(getattr(state, "log_diag_fp32", None))
    if log_diag.ndim != 1 or len(log_diag) < 1:
        raise D58ProbeError("D58 state feature dimension drift")
    return int(len(log_diag))


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d56._install_resource_accounting(d42)
    d56_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d56_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = _state_dimension(result.state)
        numeric, comparisons = _extra_resource(k, old_count, all_count, dimension)
        resource.update(
            {
                "d58_additional_lda_fit_count": 0,
                "d58_additional_lda_fit_macs": 0,
                "d58_additional_optimizer_steps": 0,
                "d58_additional_query_state_count": 0,
                "d58_query_sidecar_bytes": 0,
                "d58_extra_adaptation_mac_equivalents": numeric,
                "d58_additional_comparison_count": comparisons,
                "d58_resource_reuses_d56_exact_fit_inventory": True,
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
        raise D58ProbeError("D58 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            c = int(audit["d58_class_count"])
            k = int(audit["d58_actual_k"])
            class_counts.append(c)
            calibration = _ovr_score_lda_calibration(audit, k, c)
            base_coef = np.asarray(audit["d58_base_coefficient_fp32"], dtype=np.float64)
            base_intercept = np.asarray(audit["d58_base_intercept_fp32"], dtype=np.float64)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d58_probe_arm": ARM,
                "d58_formula": FORMULA,
                "d58_boundary_status": calibration["status"],
                "d58_class_id_specific_formula": False,
                "d58_old_new_role_specific_branch": False,
                "d58_scene_receiver_handle_specific_branch": False,
                "d58_reliability_uses_outer_held_or_query": False,
                "d58_hyperparameter_count": 0,
                "d58_query_selected_calibration": False,
                "d58_balanced_positive_negative_weight": True,
                "d58_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D58ProbeError("D58 exact audit drift")
            pairs = (
                ("d58_raw_slope_by_class", calibration["raw_slope"]),
                ("d58_normalized_slope_by_class", calibration["normalized_slope"]),
                ("d58_normalized_intercept_by_class", calibration["normalized_intercept"]),
            )
            if k > 2:
                pairs += (
                    ("d58_positive_mean_by_class", calibration["positive_mean"]),
                    ("d58_negative_mean_by_class", calibration["negative_mean"]),
                    ("d58_positive_variance_by_class", calibration["positive_variance"]),
                    ("d58_negative_variance_by_class", calibration["negative_variance"]),
                    ("d58_pooled_variance_by_class", calibration["pooled_variance"]),
                )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D58ProbeError("D58 calibration closure drift")
            if not np.isclose(
                float(audit["d58_mean_raw_slope"]),
                float(calibration["mean_raw_slope"]),
                rtol=0.0,
                atol=3.0e-7,
            ):
                raise D58ProbeError("D58 mean slope drift")
            if (
                audit.get("d58_base_prediction_by_fold")
                != calibration["base_prediction_by_fold"]
                or audit.get("d58_calibrated_prediction_by_fold")
                != calibration["calibrated_prediction_by_fold"]
                or audit.get("d58_base_correct_count")
                != calibration["base_correct_count"]
                or audit.get("d58_calibrated_correct_count")
                != calibration["calibrated_correct_count"]
            ):
                raise D58ProbeError("D58 held prediction closure drift")
            if calibration["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    calibration["normalized_slope"][:, None] * base_coef,
                    calibration["normalized_slope"] * base_intercept
                    + calibration["normalized_intercept"],
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d58_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D58ProbeError("D58 actual coefficient drift")
            if not _allclose(audit["d58_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D58ProbeError("D58 actual intercept drift")
        numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]), class_counts[0], class_counts[1], 288
        )
        if (
            resource.get("d58_additional_lda_fit_count") != 0
            or resource.get("d58_additional_lda_fit_macs") != 0
            or resource.get("d58_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d58_additional_comparison_count") != comparisons
            or resource.get("d58_resource_reuses_d56_exact_fit_inventory") is not True
        ):
            raise D58ProbeError("D58 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d56.ARM
            audit["d43_covariance_structure"] = d56.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d58_extra_adaptation_mac_equivalents"]
        )
    d56._verify_fit_audits(sanitized)
    return len(target_rows)


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D58ProbeError("D58 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d58_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d58-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D58ProbeError(f"D58 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d58_d56_helper_sha256": d43._sha256(D56_HELPER_PATH),
        "d58_d46_helper_sha256": d43._sha256(d56.D46_HELPER_PATH),
        "d58_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d58_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d58_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d58_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d58_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D58ProbeError("D58 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d58_arm,
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
        "schema": "cvs.phase2.d58.ovr_score_lda_calibration_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d58_arm,
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
    (output / "D58_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
