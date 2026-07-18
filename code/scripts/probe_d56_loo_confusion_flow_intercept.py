#!/usr/bin/env python3
"""D56 probe: LOO confusion-flow intercept balancing on D46."""

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
D46_HELPER_PATH = SCRIPT_DIR / "probe_d46_classwise_loo_reliability_fusion.py"
SPEC = importlib.util.spec_from_file_location("d56_d46_probe_helper", D46_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D56 could not load D46 helper")
d46 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d46)
d45 = d46.d45
d44 = d45.d44
d43 = d45.d43


ARM = "loo_confusion_flow_intercept"
STRUCTURE = "d46_plus_loo_confusion_flow_intercept"
FORMULA = "delta_b_c=(out_degree_c-in_degree_c)/(k_shot*class_count)"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D56ProbeError(RuntimeError):
    pass


def _confusion_flow(
    full_scores: Any,
    block_scores: Any,
    held_indices_by_fold: Any,
    targets: np.ndarray,
    audit: dict[str, Any],
    k_shot: int,
    class_count: int,
) -> dict[str, Any]:
    k = int(k_shot)
    c = int(class_count)
    if c < 2 or k < 1:
        raise D56ProbeError("D56 confusion-flow input drift")
    if k <= 2:
        zeros = np.zeros(c, dtype=np.float64)
        return {
            "full_scores": None,
            "block_scores": None,
            "true_classes": None,
            "predicted_classes": None,
            "out_degree": zeros.astype(np.int64),
            "in_degree": zeros.astype(np.int64),
            "delta_intercept": zeros,
            "status": "k1_k2_exact_d46_fallback",
            "exact_fallback": True,
        }
    full = np.asarray(full_scores, dtype=np.float64)
    block = np.asarray(block_scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    full_weight = np.asarray(audit.get("d46_full_weight_by_class"), dtype=np.float64)
    block_weight = np.asarray(audit.get("d46_block_weight_by_class"), dtype=np.float64)
    full_scale = float(audit.get("d46_full_support_logit_rms", float("nan")))
    block_scale = float(audit.get("d46_block_support_logit_rms", float("nan")))
    held = held_indices_by_fold
    if (
        full.shape != (k, c, c)
        or block.shape != (k, c, c)
        or not np.isfinite(full).all()
        or not np.isfinite(block).all()
        or labels.shape != (k * c,)
        or full_weight.shape != (c,)
        or block_weight.shape != (c,)
        or not np.isfinite(full_weight).all()
        or not np.isfinite(block_weight).all()
        or not np.allclose(full_weight + block_weight, 1.0, rtol=0.0, atol=3.0e-7)
        or not np.isfinite(full_scale)
        or not np.isfinite(block_scale)
        or full_scale <= 0.0
        or block_scale <= 0.0
        or not isinstance(held, list)
        or len(held) != k
    ):
        raise D56ProbeError("D56 D46 held-score evidence drift")
    fused = (
        full * full_weight[None, None, :] / full_scale
        + block * block_weight[None, None, :] / block_scale
    )
    out_degree = np.zeros(c, dtype=np.int64)
    in_degree = np.zeros(c, dtype=np.int64)
    true_classes: list[list[int]] = []
    predicted_classes: list[list[int]] = []
    for fold_index, held_raw in enumerate(held):
        indices = [int(value) for value in held_raw]
        if len(indices) != c or len(set(indices)) != c:
            raise D56ProbeError("D56 held partition cardinality drift")
        fold_true = [int(labels[index]) for index in indices]
        if sorted(fold_true) != list(range(c)):
            raise D56ProbeError("D56 held partition class coverage drift")
        fold_pred = np.argmax(fused[fold_index], axis=1).astype(np.int64).tolist()
        for true_class, predicted_class in zip(fold_true, fold_pred):
            if true_class != predicted_class:
                out_degree[true_class] += 1
                in_degree[predicted_class] += 1
        true_classes.append(fold_true)
        predicted_classes.append(fold_pred)
    denominator = int(k * c)
    delta_intercept = (out_degree - in_degree).astype(np.float64) / denominator
    if (
        int(np.sum(out_degree)) != int(np.sum(in_degree))
        or not np.isfinite(delta_intercept).all()
        or abs(float(np.sum(delta_intercept))) > 3.0e-15
    ):
        raise D56ProbeError("D56 confusion-flow conservation drift")
    return {
        "full_scores": full,
        "block_scores": block,
        "true_classes": true_classes,
        "predicted_classes": predicted_classes,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "delta_intercept": delta_intercept,
        "status": "loo_confusion_flow_intercept_active",
        "exact_fallback": False,
    }


def build_d56_fit(d42: Any) -> Any:
    base_fit = d46.build_classwise_loo_reliability_fit(d42)
    full_base_fit = d45._build_locked_d42_full_component_fit(d42)
    block_base_fit = d43.build_structured_fit(d42, "block3_centered")

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        base_coef, base_intercept, base_audit = base_fit(
            transformed, targets, class_count, k_shot
        )
        if int(k_shot) <= 2:
            flow = _confusion_flow(
                None, None, None, targets, base_audit, k_shot, class_count
            )
        else:
            full_collector: list[np.ndarray] = []
            block_collector: list[np.ndarray] = []
            full_fit = d46._canonical_component_fit(full_base_fit, [])
            block_fit = d46._canonical_component_fit(block_base_fit, [])
            _, _, full_partition = d45._inner_loo_component_ce(
                full_fit,
                transformed,
                targets,
                class_count,
                k_shot,
                full_collector,
            )
            _, _, block_partition = d45._inner_loo_component_ce(
                block_fit,
                transformed,
                targets,
                class_count,
                k_shot,
                block_collector,
            )
            if (
                len(full_collector) != 1
                or len(block_collector) != 1
                or full_partition["held_support_row_indices_by_fold"]
                != block_partition["held_support_row_indices_by_fold"]
                or full_partition["held_support_row_indices_by_fold"]
                != base_audit["d46_full_inner_partition_audit"][
                    "held_support_row_indices_by_fold"
                ]
            ):
                raise D56ProbeError("D56 extra inner partition drift")
            flow = _confusion_flow(
                full_collector[0],
                block_collector[0],
                full_partition["held_support_row_indices_by_fold"],
                targets,
                base_audit,
                k_shot,
                class_count,
            )
        if flow["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            final_coef64, final_intercept64 = d43._center_affine_scores(
                np.asarray(base_coef, dtype=np.float64),
                np.asarray(base_intercept, dtype=np.float64)
                + flow["delta_intercept"],
            )
            final_coef = np.asarray(final_coef64, dtype=np.float32)
            final_intercept = np.asarray(final_intercept64, dtype=np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d56_probe_arm": ARM,
                "d56_formula": FORMULA,
                "d56_boundary_status": flow["status"],
                "d56_actual_k": int(k_shot),
                "d56_class_count": int(class_count),
                "d56_class_id_specific_formula": False,
                "d56_old_new_role_specific_branch": False,
                "d56_scene_receiver_handle_specific_branch": False,
                "d56_reliability_uses_outer_held_or_query": False,
                "d56_hyperparameter_count": 0,
                "d56_query_selected_compensation": False,
                "d56_one_shot_no_flow_recompute": True,
                "d56_normalization_denominator": int(k_shot * class_count),
                "d56_full_inner_held_score_fp64": None if flow["full_scores"] is None else flow["full_scores"].tolist(),
                "d56_block_inner_held_score_fp64": None if flow["block_scores"] is None else flow["block_scores"].tolist(),
                "d56_held_true_class_by_fold": flow["true_classes"],
                "d56_held_prediction_by_fold": flow["predicted_classes"],
                "d56_out_degree_by_class": flow["out_degree"].tolist(),
                "d56_in_degree_by_class": flow["in_degree"].tolist(),
                "d56_centered_intercept_compensation_fp64": flow["delta_intercept"].tolist(),
                "d56_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d56_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d56_actual_coefficient_fp32": final_coef.tolist(),
                "d56_actual_intercept_fp32": final_intercept.tolist(),
                "d56_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit


def _extra_resource(
    k: int,
    old_count: int,
    all_count: int,
    dimension: int,
    original_lda_fit_macs: Any,
) -> tuple[int, int, int, int]:
    if k < 1 or old_count < 2 or all_count < old_count or dimension < 1:
        raise D56ProbeError("D56 resource input drift")
    if k <= 2:
        return 0, 0, 0, 0
    extra_fit_count = 4 * k
    extra_lda_macs = 2 * k * int(
        original_lda_fit_macs(old_count * (k - 1), old_count)
    ) + 2 * k * int(
        original_lda_fit_macs(all_count * (k - 1), all_count)
    )
    numeric = sum(6 * k * c * c + 4 * k * c for c in (old_count, all_count))
    comparisons = sum(k * c * (c - 1) for c in (old_count, all_count))
    return int(extra_fit_count), int(extra_lda_macs), int(numeric), int(comparisons)


def _state_dimension(state: Any) -> int:
    log_diag = np.asarray(getattr(state, "log_diag_fp32", None))
    if log_diag.ndim != 1 or len(log_diag) < 1:
        raise D56ProbeError("D56 state feature dimension drift")
    return int(len(log_diag))


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d46._install_d46_resource_accounting(d42)
    d45_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d45_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        k = int(resource["old_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        dimension = _state_dimension(result.state)
        extra_fits, extra_lda_macs, numeric, comparisons = _extra_resource(
            k, old_count, all_count, dimension, original_macs
        )
        resource.update(
            {
                "d56_additional_lda_fit_count": extra_fits,
                "d56_additional_lda_fit_macs": extra_lda_macs,
                "d56_additional_optimizer_steps": 0,
                "d56_additional_query_state_count": 0,
                "d56_query_sidecar_bytes": 0,
                "d56_extra_adaptation_mac_equivalents": numeric,
                "d56_additional_comparison_count": comparisons,
                "d56_resource_reuses_d46_query_state": True,
            }
        )
        resource["lda_closed_form_fit_count"] = int(
            resource["lda_closed_form_fit_count"] + extra_fits
        )
        resource["estimated_lda_fit_macs"] = int(
            resource["estimated_lda_fit_macs"] + extra_lda_macs
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + extra_lda_macs + numeric
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
        raise D56ProbeError("D56 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            c = int(audit["d56_class_count"])
            k = int(audit["d56_actual_k"])
            class_counts.append(c)
            if k <= 2:
                flow = _confusion_flow(None, None, None, np.zeros(k * c), audit, k, c)
            else:
                held = audit["d46_full_inner_partition_audit"][
                    "held_support_row_indices_by_fold"
                ]
                true_by_fold = audit.get("d56_held_true_class_by_fold")
                if not isinstance(true_by_fold, list) or len(true_by_fold) != k:
                    raise D56ProbeError("D56 held true-class audit drift")
                labels = np.full(k * c, -1, dtype=np.int64)
                for indices, true_classes in zip(held, true_by_fold):
                    if len(indices) != c or len(true_classes) != c:
                        raise D56ProbeError("D56 held true-class cardinality drift")
                    for index, true_class in zip(indices, true_classes):
                        labels[int(index)] = int(true_class)
                if np.any(labels < 0):
                    raise D56ProbeError("D56 held true-class coverage drift")
                flow = _confusion_flow(
                    audit.get("d56_full_inner_held_score_fp64"),
                    audit.get("d56_block_inner_held_score_fp64"),
                    held,
                    labels,
                    audit,
                    k,
                    c,
                )
            base_coef = np.asarray(audit["d56_base_coefficient_fp32"], dtype=np.float64)
            base_intercept = np.asarray(audit["d56_base_intercept_fp32"], dtype=np.float64)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d56_probe_arm": ARM,
                "d56_formula": FORMULA,
                "d56_boundary_status": flow["status"],
                "d56_class_id_specific_formula": False,
                "d56_old_new_role_specific_branch": False,
                "d56_scene_receiver_handle_specific_branch": False,
                "d56_reliability_uses_outer_held_or_query": False,
                "d56_hyperparameter_count": 0,
                "d56_query_selected_compensation": False,
                "d56_one_shot_no_flow_recompute": True,
                "d56_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D56ProbeError("D56 exact audit drift")
            if audit.get("d56_normalization_denominator") != k * c:
                raise D56ProbeError("D56 normalization denominator drift")
            pairs = (
                ("d56_out_degree_by_class", flow["out_degree"]),
                ("d56_in_degree_by_class", flow["in_degree"]),
                ("d56_centered_intercept_compensation_fp64", flow["delta_intercept"]),
            )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D56ProbeError("D56 confusion-flow closure drift")
            if audit.get("d56_held_prediction_by_fold") != flow["predicted_classes"]:
                raise D56ProbeError("D56 held prediction drift")
            if flow["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    base_coef, base_intercept + flow["delta_intercept"]
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d56_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D56ProbeError("D56 actual coefficient drift")
            if not _allclose(audit["d56_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D56ProbeError("D56 actual intercept drift")
        extra_fits, extra_lda_macs, numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]),
            class_counts[0],
            class_counts[1],
            288,
            lambda rows, classes: d45._expected_lda_fit_macs(rows, classes, 288),
        )
        if (
            resource.get("d56_additional_lda_fit_count") != extra_fits
            or resource.get("d56_additional_lda_fit_macs") != extra_lda_macs
            or resource.get("d56_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d56_additional_comparison_count") != comparisons
            or resource.get("d56_resource_reuses_d46_query_state") is not True
        ):
            raise D56ProbeError("D56 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d46.ARM
            audit["d43_covariance_structure"] = d46.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d56_additional_lda_fit_macs"]
            - row["resource"]["d56_extra_adaptation_mac_equivalents"]
        )
        row["resource"]["estimated_lda_fit_macs"] = int(
            row["resource"]["estimated_lda_fit_macs"]
            - row["resource"]["d56_additional_lda_fit_macs"]
        )
        row["resource"]["lda_closed_form_fit_count"] = int(
            row["resource"]["lda_closed_form_fit_count"]
            - row["resource"]["d56_additional_lda_fit_count"]
        )
    d46._verify_d46_fit_audits(sanitized)
    return len(target_rows)


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D56ProbeError("D56 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d56_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d56-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D56ProbeError(f"D56 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d56_d46_helper_sha256": d43._sha256(D46_HELPER_PATH),
        "d56_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d56_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d56_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d56_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d56_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D56ProbeError("D56 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d56_arm,
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
        "schema": "cvs.phase2.d56.loo_confusion_flow_intercept_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d56_arm,
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
    (output / "D56_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
