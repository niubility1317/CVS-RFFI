#!/usr/bin/env python3
"""D57 probe: cross-fitted bidirectional confusion-flow gating on D46."""

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
SPEC = importlib.util.spec_from_file_location("d57_d56_probe_helper", D56_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D57 could not load D56 helper")
d56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d56)
d46 = d56.d46
d45 = d46.d45
d44 = d45.d44
d43 = d45.d43


ARM = "crossfitted_bidirectional_flow_gate"
STRUCTURE = "d46_plus_crossfitted_bidirectional_flow_gate"
FORMULA = "accept_c=(positive_adjusted>=positive_base)&(fp_adjusted<=fp_base)&strict;delta=accept*flow"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D57ProbeError(RuntimeError):
    pass


def _bidirectional_gate(audit: dict[str, Any], k_shot: int, class_count: int) -> dict[str, Any]:
    k = int(k_shot)
    c = int(class_count)
    if c < 2 or k < 1:
        raise D57ProbeError("D57 gate input drift")
    if k <= 2:
        zeros = np.zeros(c, dtype=np.float64)
        return {
            "base_positive_correct": zeros.astype(np.int64),
            "coordinate_positive_correct": zeros.astype(np.int64),
            "joint_positive_correct": zeros.astype(np.int64),
            "base_false_positive": zeros.astype(np.int64),
            "coordinate_false_positive": zeros.astype(np.int64),
            "joint_false_positive": zeros.astype(np.int64),
            "initial_accept_mask": zeros.astype(bool),
            "final_accept_mask": zeros.astype(bool),
            "crossfit_delta_by_fold": None,
            "delta_intercept": zeros,
            "status": "k1_k2_exact_d46_fallback",
            "atomic_fallback": True,
            "exact_fallback": True,
        }
    full = np.asarray(audit.get("d56_full_inner_held_score_fp64"), dtype=np.float64)
    block = np.asarray(audit.get("d56_block_inner_held_score_fp64"), dtype=np.float64)
    true_classes = audit.get("d56_held_true_class_by_fold")
    base_predictions = audit.get("d56_held_prediction_by_fold")
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
        or not isinstance(base_predictions, list)
        or len(base_predictions) != k
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
        raise D57ProbeError("D57 D56 held-score evidence drift")
    fused = (
        full * full_weight[None, None, :] / full_scale
        + block * block_weight[None, None, :] / block_scale
    )
    recomputed_predictions = np.argmax(fused, axis=2).astype(np.int64).tolist()
    normalized_true = [[int(value) for value in fold] for fold in true_classes]
    if (
        any(len(fold) != c or sorted(fold) != list(range(c)) for fold in normalized_true)
        or recomputed_predictions != base_predictions
    ):
        raise D57ProbeError("D57 D56 held prediction closure drift")

    base_positive = np.zeros(c, dtype=np.int64)
    base_false_positive = np.zeros(c, dtype=np.int64)
    coordinate_positive = np.zeros(c, dtype=np.int64)
    coordinate_false_positive = np.zeros(c, dtype=np.int64)
    crossfit_delta: list[np.ndarray] = []
    for held_fold in range(k):
        out_degree = np.zeros(c, dtype=np.int64)
        in_degree = np.zeros(c, dtype=np.int64)
        for evidence_fold in range(k):
            if evidence_fold == held_fold:
                continue
            for true_class, predicted_class in zip(
                normalized_true[evidence_fold], recomputed_predictions[evidence_fold]
            ):
                if true_class != predicted_class:
                    out_degree[true_class] += 1
                    in_degree[predicted_class] += 1
        fold_delta = (out_degree - in_degree).astype(np.float64) / ((k - 1) * c)
        if abs(float(np.sum(fold_delta))) > 3.0e-15:
            raise D57ProbeError("D57 crossfit flow conservation drift")
        crossfit_delta.append(fold_delta)
        fold_true = np.asarray(normalized_true[held_fold], dtype=np.int64)
        fold_base = np.asarray(recomputed_predictions[held_fold], dtype=np.int64)
        for candidate_class in range(c):
            positive = fold_true == candidate_class
            negative = ~positive
            base_positive[candidate_class] += int(
                np.sum(fold_base[positive] == candidate_class)
            )
            base_false_positive[candidate_class] += int(
                np.sum(fold_base[negative] == candidate_class)
            )
            coordinate_scores = fused[held_fold].copy()
            coordinate_scores[:, candidate_class] += fold_delta[candidate_class]
            coordinate_pred = np.argmax(coordinate_scores, axis=1)
            coordinate_positive[candidate_class] += int(
                np.sum(coordinate_pred[positive] == candidate_class)
            )
            coordinate_false_positive[candidate_class] += int(
                np.sum(coordinate_pred[negative] == candidate_class)
            )
    initial_accept = (
        (coordinate_positive >= base_positive)
        & (coordinate_false_positive <= base_false_positive)
        & (
            (coordinate_positive > base_positive)
            | (coordinate_false_positive < base_false_positive)
        )
    )
    joint_positive = np.zeros(c, dtype=np.int64)
    joint_false_positive = np.zeros(c, dtype=np.int64)
    for held_fold, fold_delta in enumerate(crossfit_delta):
        joint_scores = fused[held_fold] + (fold_delta * initial_accept)[None, :]
        joint_pred = np.argmax(joint_scores, axis=1)
        fold_true = np.asarray(normalized_true[held_fold], dtype=np.int64)
        for candidate_class in range(c):
            positive = fold_true == candidate_class
            negative = ~positive
            joint_positive[candidate_class] += int(
                np.sum(joint_pred[positive] == candidate_class)
            )
            joint_false_positive[candidate_class] += int(
                np.sum(joint_pred[negative] == candidate_class)
            )
    atomic_safe = bool(
        np.all(joint_positive >= base_positive)
        and np.all(joint_false_positive <= base_false_positive)
    )
    final_accept = initial_accept if atomic_safe else np.zeros(c, dtype=bool)
    full_flow = np.asarray(
        audit.get("d56_centered_intercept_compensation_fp64"), dtype=np.float64
    )
    if full_flow.shape != (c,) or not np.isfinite(full_flow).all():
        raise D57ProbeError("D57 full-flow evidence drift")
    delta_intercept = full_flow * final_accept
    delta_intercept -= float(np.mean(delta_intercept))
    exact_fallback = not bool(np.any(final_accept))
    status = (
        "crossfitted_bidirectional_flow_gate_active"
        if not exact_fallback
        else (
            "joint_gate_atomic_d46_fallback"
            if bool(np.any(initial_accept)) and not atomic_safe
            else "no_coordinate_accepted_exact_d46_fallback"
        )
    )
    return {
        "base_positive_correct": base_positive,
        "coordinate_positive_correct": coordinate_positive,
        "joint_positive_correct": joint_positive,
        "base_false_positive": base_false_positive,
        "coordinate_false_positive": coordinate_false_positive,
        "joint_false_positive": joint_false_positive,
        "initial_accept_mask": initial_accept,
        "final_accept_mask": final_accept,
        "crossfit_delta_by_fold": np.stack(crossfit_delta, axis=0),
        "delta_intercept": delta_intercept,
        "status": status,
        "atomic_fallback": not atomic_safe,
        "exact_fallback": exact_fallback,
    }


def build_d57_fit(d42: Any) -> Any:
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
        gate = _bidirectional_gate(base_audit, k_shot, class_count)
        if gate["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            final_coef64, final_intercept64 = d43._center_affine_scores(
                np.asarray(base_coef, dtype=np.float64),
                np.asarray(base_intercept, dtype=np.float64)
                + gate["delta_intercept"],
            )
            final_coef = np.asarray(final_coef64, dtype=np.float32)
            final_intercept = np.asarray(final_intercept64, dtype=np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d57_probe_arm": ARM,
                "d57_formula": FORMULA,
                "d57_boundary_status": gate["status"],
                "d57_actual_k": int(k_shot),
                "d57_class_count": int(class_count),
                "d57_class_id_specific_formula": False,
                "d57_old_new_role_specific_branch": False,
                "d57_scene_receiver_handle_specific_branch": False,
                "d57_reliability_uses_outer_held_or_query": False,
                "d57_hyperparameter_count": 0,
                "d57_query_selected_compensation": False,
                "d57_crossfit_held_fold_excluded": True,
                "d57_coordinate_order_used": False,
                "d57_base_positive_correct_by_class": gate["base_positive_correct"].tolist(),
                "d57_coordinate_positive_correct_by_class": gate["coordinate_positive_correct"].tolist(),
                "d57_joint_positive_correct_by_class": gate["joint_positive_correct"].tolist(),
                "d57_base_false_positive_by_class": gate["base_false_positive"].tolist(),
                "d57_coordinate_false_positive_by_class": gate["coordinate_false_positive"].tolist(),
                "d57_joint_false_positive_by_class": gate["joint_false_positive"].tolist(),
                "d57_initial_accept_mask": gate["initial_accept_mask"].tolist(),
                "d57_final_accept_mask": gate["final_accept_mask"].tolist(),
                "d57_crossfit_delta_by_fold": None if gate["crossfit_delta_by_fold"] is None else gate["crossfit_delta_by_fold"].tolist(),
                "d57_joint_atomic_fallback": gate["atomic_fallback"],
                "d57_centered_intercept_compensation_fp64": gate["delta_intercept"].tolist(),
                "d57_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d57_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d57_actual_coefficient_fp32": final_coef.tolist(),
                "d57_actual_intercept_fp32": final_intercept.tolist(),
                "d57_single_affine_state_only": True,
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
        raise D57ProbeError("D57 resource input drift")
    if k <= 2:
        return 0, 0
    # D56 already materializes every leave-one-fold-out score used here. D57 only
    # accumulates cross-fitted confusion counts, evaluates one coordinate at a
    # time, and performs the atomic joint check; it performs no additional fit.
    numeric = sum(8 * k * c * c + 8 * k * c for c in (old_count, all_count))
    comparisons = sum(4 * k * c * c for c in (old_count, all_count))
    return int(numeric), int(comparisons)


def _state_dimension(state: Any) -> int:
    log_diag = np.asarray(getattr(state, "log_diag_fp32", None))
    if log_diag.ndim != 1 or len(log_diag) < 1:
        raise D57ProbeError("D57 state feature dimension drift")
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
                "d57_additional_lda_fit_count": 0,
                "d57_additional_lda_fit_macs": 0,
                "d57_additional_optimizer_steps": 0,
                "d57_additional_query_state_count": 0,
                "d57_query_sidecar_bytes": 0,
                "d57_extra_adaptation_mac_equivalents": numeric,
                "d57_additional_comparison_count": comparisons,
                "d57_resource_reuses_d56_exact_fit_inventory": True,
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
        raise D57ProbeError("D57 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            c = int(audit["d57_class_count"])
            k = int(audit["d57_actual_k"])
            class_counts.append(c)
            gate = _bidirectional_gate(audit, k, c)
            base_coef = np.asarray(audit["d57_base_coefficient_fp32"], dtype=np.float64)
            base_intercept = np.asarray(audit["d57_base_intercept_fp32"], dtype=np.float64)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d57_probe_arm": ARM,
                "d57_formula": FORMULA,
                "d57_boundary_status": gate["status"],
                "d57_class_id_specific_formula": False,
                "d57_old_new_role_specific_branch": False,
                "d57_scene_receiver_handle_specific_branch": False,
                "d57_reliability_uses_outer_held_or_query": False,
                "d57_hyperparameter_count": 0,
                "d57_query_selected_compensation": False,
                "d57_crossfit_held_fold_excluded": True,
                "d57_coordinate_order_used": False,
                "d57_joint_atomic_fallback": gate["atomic_fallback"],
                "d57_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D57ProbeError("D57 exact audit drift")
            pairs = (
                ("d57_base_positive_correct_by_class", gate["base_positive_correct"]),
                ("d57_coordinate_positive_correct_by_class", gate["coordinate_positive_correct"]),
                ("d57_joint_positive_correct_by_class", gate["joint_positive_correct"]),
                ("d57_base_false_positive_by_class", gate["base_false_positive"]),
                ("d57_coordinate_false_positive_by_class", gate["coordinate_false_positive"]),
                ("d57_joint_false_positive_by_class", gate["joint_false_positive"]),
                ("d57_initial_accept_mask", gate["initial_accept_mask"]),
                ("d57_final_accept_mask", gate["final_accept_mask"]),
                ("d57_centered_intercept_compensation_fp64", gate["delta_intercept"]),
            )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D57ProbeError("D57 gate closure drift")
            if gate["crossfit_delta_by_fold"] is None:
                if audit.get("d57_crossfit_delta_by_fold") is not None:
                    raise D57ProbeError("D57 crossfit fallback audit drift")
            elif not _allclose(
                audit.get("d57_crossfit_delta_by_fold"),
                gate["crossfit_delta_by_fold"],
                3.0e-7,
            ):
                raise D57ProbeError("D57 crossfit delta drift")
            if gate["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    base_coef, base_intercept + gate["delta_intercept"]
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d57_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D57ProbeError("D57 actual coefficient drift")
            if not _allclose(audit["d57_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D57ProbeError("D57 actual intercept drift")
        numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]), class_counts[0], class_counts[1], 288
        )
        if (
            resource.get("d57_additional_lda_fit_count") != 0
            or resource.get("d57_additional_lda_fit_macs") != 0
            or resource.get("d57_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d57_additional_comparison_count") != comparisons
            or resource.get("d57_resource_reuses_d56_exact_fit_inventory") is not True
        ):
            raise D57ProbeError("D57 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d56.ARM
            audit["d43_covariance_structure"] = d56.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d57_extra_adaptation_mac_equivalents"]
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
        raise D57ProbeError("D57 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d57_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d57-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D57ProbeError(f"D57 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d57_d56_helper_sha256": d43._sha256(D56_HELPER_PATH),
        "d57_d46_helper_sha256": d43._sha256(d56.D46_HELPER_PATH),
        "d57_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d57_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d57_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d57_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d57_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D57ProbeError("D57 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d57_arm,
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
        "schema": "cvs.phase2.d57.crossfitted_bidirectional_flow_gate_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d57_arm,
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
    (output / "D57_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
