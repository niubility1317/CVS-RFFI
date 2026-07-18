#!/usr/bin/env python3
"""D55 probe: centered LOO class-difficulty intercept compensation on D46."""

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
D46_HELPER_PATH = SCRIPT_DIR / "probe_d46_classwise_loo_reliability_fusion.py"
SPEC = importlib.util.spec_from_file_location("d55_d46_probe_helper", D46_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D55 could not load D46 helper")
d46 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d46)
d45 = d46.d45
d44 = d45.d44
d43 = d45.d43


ARM = "centered_loo_class_difficulty_intercept"
STRUCTURE = "d46_plus_centered_loo_class_difficulty_intercept"
FORMULA = "delta_b_c=sum_g(w_g_c*CE_g_c)-mean_j(sum_g(w_g_j*CE_g_j))"
EPSILON = 1.0e-10
UNIT_TOLERANCE = 2.0e-6
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D55ProbeError(RuntimeError):
    pass


def _difficulty(audit: dict[str, Any], k_shot: int, class_count: int) -> dict[str, Any]:
    k = int(k_shot)
    c = int(class_count)
    if c < 2 or k < 1:
        raise D55ProbeError("D55 difficulty input drift")
    if k <= 2:
        zeros = np.zeros(c, dtype=np.float64)
        return {
            "full_ce": None,
            "block_ce": None,
            "full_weight": None,
            "block_weight": None,
            "difficulty": zeros,
            "delta_intercept": zeros,
            "status": "k1_k2_exact_d46_fallback",
            "exact_fallback": True,
        }
    full_ce = np.asarray(audit.get("d46_full_inner_loo_ce_by_class"), dtype=np.float64)
    block_ce = np.asarray(audit.get("d46_block_inner_loo_ce_by_class"), dtype=np.float64)
    full_weight = np.asarray(audit.get("d46_full_weight_by_class"), dtype=np.float64)
    block_weight = np.asarray(audit.get("d46_block_weight_by_class"), dtype=np.float64)
    arrays = (full_ce, block_ce, full_weight, block_weight)
    if any(value.shape != (c,) or not np.isfinite(value).all() for value in arrays):
        raise D55ProbeError("D55 D46 classwise evidence drift")
    if not np.allclose(full_weight + block_weight, 1.0, rtol=0.0, atol=3.0e-7):
        raise D55ProbeError("D55 D46 weight simplex drift")
    difficulty = full_weight * full_ce + block_weight * block_ce
    delta_intercept = difficulty - float(np.mean(difficulty))
    if (
        not np.isfinite(difficulty).all()
        or not np.isfinite(delta_intercept).all()
        or abs(float(np.sum(delta_intercept))) > 3.0e-10
    ):
        raise D55ProbeError("D55 centered difficulty drift")
    return {
        "full_ce": full_ce,
        "block_ce": block_ce,
        "full_weight": full_weight,
        "block_weight": block_weight,
        "difficulty": difficulty,
        "delta_intercept": delta_intercept,
        "status": "centered_loo_class_difficulty_intercept_active",
        "exact_fallback": False,
    }


def build_d55_fit(d42: Any) -> Any:
    base_fit = d46.build_classwise_loo_reliability_fit(d42)

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        base_coef, base_intercept, base_audit = base_fit(
            transformed, targets, class_count, k_shot
        )
        difficulty = _difficulty(base_audit, k_shot, class_count)
        if difficulty["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            final_coef64, final_intercept64 = d43._center_affine_scores(
                np.asarray(base_coef, dtype=np.float64),
                np.asarray(base_intercept, dtype=np.float64)
                + difficulty["delta_intercept"],
            )
            final_coef = np.asarray(final_coef64, dtype=np.float32)
            final_intercept = np.asarray(final_intercept64, dtype=np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d55_probe_arm": ARM,
                "d55_formula": FORMULA,
                "d55_boundary_status": difficulty["status"],
                "d55_actual_k": int(k_shot),
                "d55_class_count": int(class_count),
                "d55_class_id_specific_formula": False,
                "d55_old_new_role_specific_branch": False,
                "d55_scene_receiver_handle_specific_branch": False,
                "d55_reliability_uses_outer_held_or_query": False,
                "d55_hyperparameter_count": 0,
                "d55_query_selected_compensation": False,
                "d55_full_ce_by_class": None if difficulty["full_ce"] is None else difficulty["full_ce"].tolist(),
                "d55_block_ce_by_class": None if difficulty["block_ce"] is None else difficulty["block_ce"].tolist(),
                "d55_full_weight_by_class": None if difficulty["full_weight"] is None else difficulty["full_weight"].tolist(),
                "d55_block_weight_by_class": None if difficulty["block_weight"] is None else difficulty["block_weight"].tolist(),
                "d55_weighted_difficulty_by_class": difficulty["difficulty"].tolist(),
                "d55_centered_intercept_compensation_fp64": difficulty["delta_intercept"].tolist(),
                "d55_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d55_base_intercept_fp32": np.asarray(
                    base_intercept, dtype=np.float32
                ).tolist(),
                "d55_actual_coefficient_fp32": final_coef.tolist(),
                "d55_actual_intercept_fp32": final_intercept.tolist(),
                "d55_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit


def _extra_resource(k: int, old_count: int, all_count: int, dimension: int) -> tuple[int, int]:
    if k < 1 or old_count < 2 or all_count < old_count or dimension < 1:
        raise D55ProbeError("D55 resource input drift")
    class_sum = old_count + all_count
    numeric = 8 * class_sum
    return int(numeric), 0


def _state_dimension(state: Any) -> int:
    log_diag = np.asarray(getattr(state, "log_diag_fp32", None))
    if log_diag.ndim != 1 or len(log_diag) < 1:
        raise D55ProbeError("D55 state feature dimension drift")
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
        numeric, comparisons = _extra_resource(k, old_count, all_count, dimension)
        resource.update(
            {
                "d55_additional_lda_fit_count": 0,
                "d55_additional_optimizer_steps": 0,
                "d55_additional_query_state_count": 0,
                "d55_query_sidecar_bytes": 0,
                "d55_extra_adaptation_mac_equivalents": numeric,
                "d55_additional_comparison_count": comparisons,
                "d55_resource_reuses_d46_exact_inventory": True,
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
        raise D55ProbeError("D55 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in target_rows:
        resource = row["resource"]
        class_counts = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            c = int(audit["d55_class_count"])
            k = int(audit["d55_actual_k"])
            class_counts.append(c)
            difficulty = _difficulty(audit, k, c)
            base_coef = np.asarray(audit["d55_base_coefficient_fp32"], dtype=np.float64)
            base_intercept = np.asarray(audit["d55_base_intercept_fp32"], dtype=np.float64)
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d55_probe_arm": ARM,
                "d55_formula": FORMULA,
                "d55_boundary_status": difficulty["status"],
                "d55_class_id_specific_formula": False,
                "d55_old_new_role_specific_branch": False,
                "d55_scene_receiver_handle_specific_branch": False,
                "d55_reliability_uses_outer_held_or_query": False,
                "d55_hyperparameter_count": 0,
                "d55_query_selected_compensation": False,
                "d55_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D55ProbeError("D55 exact audit drift")
            pairs = (
                ("d55_weighted_difficulty_by_class", difficulty["difficulty"]),
                (
                    "d55_centered_intercept_compensation_fp64",
                    difficulty["delta_intercept"],
                ),
            )
            if any(not _allclose(audit.get(name), value, 3.0e-7) for name, value in pairs):
                raise D55ProbeError("D55 difficulty closure drift")
            if difficulty["exact_fallback"]:
                expected_coef = base_coef.astype(np.float32)
                expected_intercept = base_intercept.astype(np.float32)
            else:
                expected_coef64, expected_intercept64 = d43._center_affine_scores(
                    base_coef, base_intercept + difficulty["delta_intercept"]
                )
                expected_coef = expected_coef64.astype(np.float32)
                expected_intercept = expected_intercept64.astype(np.float32)
            if not _allclose(audit["d55_actual_coefficient_fp32"], expected_coef, 3.0e-7):
                raise D55ProbeError("D55 actual coefficient drift")
            if not _allclose(audit["d55_actual_intercept_fp32"], expected_intercept, 3.0e-7):
                raise D55ProbeError("D55 actual intercept drift")
        numeric, comparisons = _extra_resource(
            int(resource["old_k_shot"]), class_counts[0], class_counts[1], 288
        )
        if (
            resource.get("d55_extra_adaptation_mac_equivalents") != numeric
            or resource.get("d55_additional_comparison_count") != comparisons
            or resource.get("d55_resource_reuses_d46_exact_inventory") is not True
        ):
            raise D55ProbeError("D55 resource closure drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d46.ARM
            audit["d43_covariance_structure"] = d46.STRUCTURE
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d55_extra_adaptation_mac_equivalents"]
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
        raise D55ProbeError("D55 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, "verified_d55_training_row_count": _verify_fit_audits(rows), **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d55-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D55ProbeError(f"D55 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d55_d46_helper_sha256": d43._sha256(D46_HELPER_PATH),
        "d55_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d55_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d55_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name = "d55_locked_d42_runner"
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_d55_fit(d42)
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D55ProbeError("D55 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d55_arm,
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
        "schema": "cvs.phase2.d55.centered_loo_class_difficulty_intercept_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d55_arm,
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
    (output / "D55_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
