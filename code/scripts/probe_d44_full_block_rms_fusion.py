#!/usr/bin/env python3
"""D44 diagnostic probe: fixed equal fusion of full and 3-block LDA logits."""

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
D43_SPEC = importlib.util.spec_from_file_location("d44_d43_probe_helper", D43_HELPER_PATH)
if D43_SPEC is None or D43_SPEC.loader is None:
    raise RuntimeError("D44 could not load the D43 probe helper")
d43 = importlib.util.module_from_spec(D43_SPEC)
D43_SPEC.loader.exec_module(d43)


ARM = "full_block_rms_equal"
STRUCTURE = "full_centered_plus_block3_centered_support_rms_equal_fusion"
FUSION_WEIGHT = 0.5
SCALE_EPSILON = 1.0e-12
SCALE_FORMULA = (
    "sqrt(mean_over_support_and_classes((score-row_class_mean)^2))"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D44ProbeError(RuntimeError):
    pass


def _class_centered_logit_rms(
    transformed: np.ndarray, coefficients: np.ndarray, intercept: np.ndarray
) -> float:
    rows = np.asarray(transformed, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    scores = rows @ coef.T + bias[None, :]
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(scale) or scale <= SCALE_EPSILON:
        raise D44ProbeError("D44 support logit RMS is degenerate")
    return scale


def build_full_block_rms_fit(
    d42: Any,
) -> Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    full_fit = d43.build_structured_fit(d42, "full_centered_control")
    block_fit = d43.build_structured_fit(d42, "block3_centered")

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        full_coef, full_intercept, full_audit = full_fit(
            transformed, targets, class_count, k_shot
        )
        block_coef, block_intercept, block_audit = block_fit(
            transformed, targets, class_count, k_shot
        )
        full_scale = _class_centered_logit_rms(
            transformed, full_coef, full_intercept
        )
        block_scale = _class_centered_logit_rms(
            transformed, block_coef, block_intercept
        )
        fused_coef64 = FUSION_WEIGHT * (
            np.asarray(full_coef, dtype=np.float64) / full_scale
            + np.asarray(block_coef, dtype=np.float64) / block_scale
        )
        fused_intercept64 = FUSION_WEIGHT * (
            np.asarray(full_intercept, dtype=np.float64) / full_scale
            + np.asarray(block_intercept, dtype=np.float64) / block_scale
        )
        centered_coef, centered_intercept = d43._center_affine_scores(
            fused_coef64, fused_intercept64
        )
        coef32 = centered_coef.astype(np.float32)
        intercept32 = centered_intercept.astype(np.float32)
        support_scores = (
            np.asarray(transformed, dtype=np.float32) @ coef32.T
            + intercept32[None, :]
        )
        if (
            coef32.shape != (class_count, d42.FEATURE_DIM)
            or intercept32.shape != (class_count,)
            or not np.isfinite(coef32).all()
            or not np.isfinite(intercept32).all()
            or not np.isfinite(support_scores).all()
        ):
            raise D44ProbeError("D44 fused affine state drift")
        component_condition_numbers = [
            float(value)
            for value in (
                full_audit.get("d43_covariance_condition_number"),
                block_audit.get("d43_covariance_condition_number"),
            )
            if value is not None
        ]
        audit = dict(full_audit)
        audit.update(
            {
                "coefficient_source": (
                    "d44_support_class_centered_logit_rms_normalized_"
                    "full_block_equal_affine_fusion"
                ),
                "covariance_equation_residual_max": float(
                    max(
                        float(full_audit["covariance_equation_residual_max"]),
                        float(block_audit["covariance_equation_residual_max"]),
                    )
                ),
                "sklearn_prediction_equivalent": None,
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d44_probe_arm": ARM,
                "d44_component_arms": [
                    "full_centered_control",
                    "block3_centered",
                ],
                "d44_scale_formula": SCALE_FORMULA,
                "d44_scale_uses_labels_or_roles": False,
                "d44_full_support_logit_rms": full_scale,
                "d44_block_support_logit_rms": block_scale,
                "d44_full_weight": FUSION_WEIGHT,
                "d44_block_weight": FUSION_WEIGHT,
                "d44_weight_scan_count": 0,
                "d44_class_or_scenario_specific_branch": False,
                "d44_fused_coefficient_mean_max_abs": float(
                    np.max(np.abs(coef32.astype(np.float64).mean(axis=0)))
                ),
                "d44_fused_intercept_mean_abs": float(
                    abs(intercept32.astype(np.float64).mean())
                ),
                "d43_covariance_condition_number": (
                    float(max(component_condition_numbers))
                    if component_condition_numbers
                    else None
                ),
            }
        )
        return coef32, intercept32, audit

    return fit


def _install_d44_core_resource_accounting(d42: Any) -> tuple[Any, Any]:
    """Account for two covariance fits at both before and final stages."""

    original_lda_fit_macs = d42._lda_fit_macs
    original_top_level_fit = d42.fit_d42_unified_shrinkage_lda

    def doubled_lda_fit_macs(row_count: int, class_count: int) -> int:
        return 2 * int(original_lda_fit_macs(row_count, class_count))

    def fit_with_d44_resource_audit(*args: Any, **kwargs: Any) -> Any:
        result = original_top_level_fit(*args, **kwargs)
        resource = dict(result.resource_audit)
        resource.update(
            {
                "lda_closed_form_fit_count": 4,
                "d44_component_lda_fit_count_per_stage": 2,
                "d44_component_geometry_count": 2,
                "d44_fused_query_state_count": 1,
            }
        )
        return replace(result, resource_audit=resource)

    d42._lda_fit_macs = doubled_lda_fit_macs
    d42.fit_d42_unified_shrinkage_lda = fit_with_d44_resource_audit
    return original_lda_fit_macs, original_top_level_fit


def _verify_d44_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d44_rows = [
        row
        for row in training_rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d44_rows) != 30:
        raise D44ProbeError("D44 training-row closure drift")
    required_audit = {
        "d44_probe_arm": ARM,
        "d44_component_arms": [
            "full_centered_control",
            "block3_centered",
        ],
        "d44_scale_formula": SCALE_FORMULA,
        "d44_scale_uses_labels_or_roles": False,
        "d44_full_weight": FUSION_WEIGHT,
        "d44_block_weight": FUSION_WEIGHT,
        "d44_weight_scan_count": 0,
        "d44_class_or_scenario_specific_branch": False,
    }
    required_resource = {
        "lda_closed_form_fit_count": 4,
        "d44_component_lda_fit_count_per_stage": 2,
        "d44_component_geometry_count": 2,
        "d44_fused_query_state_count": 1,
    }
    for row in d44_rows:
        geometry = row.get("geometry_summary")
        if not isinstance(geometry, dict):
            raise D44ProbeError("D44 geometry summary missing")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = geometry.get(field)
            if not isinstance(audit, dict):
                raise D44ProbeError(f"D44 fit audit missing from {field}")
            for name, expected in required_audit.items():
                if audit.get(name) != expected:
                    raise D44ProbeError(
                        f"D44 fit audit drift for {field}.{name}"
                    )
            for name in (
                "d44_full_support_logit_rms",
                "d44_block_support_logit_rms",
            ):
                value = audit.get(name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not np.isfinite(float(value))
                    or float(value) <= SCALE_EPSILON
                ):
                    raise D44ProbeError(
                        f"D44 fit audit has invalid positive scale: {field}.{name}"
                    )
        resource = row.get("resource")
        if not isinstance(resource, dict):
            raise D44ProbeError("D44 row resource audit missing")
        for name, expected in required_resource.items():
            if resource.get(name) != expected:
                raise D44ProbeError(f"D44 resource audit drift for {name}")
        metric_macs = resource.get("estimated_metric_adaptation_macs")
        lda_macs = resource.get("estimated_lda_fit_macs")
        total_macs = resource.get("estimated_adaptation_macs")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (metric_macs, lda_macs, total_macs)
            )
            or total_macs != metric_macs + lda_macs
        ):
            raise D44ProbeError("D44 adaptation MAC closure drift")
    return len(d44_rows)


def _verify_d44_output(
    output: Path,
    probe_script_sha256: str,
    d43_helper_sha256: str,
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    if (
        support.get("candidate_lock", {})
        .get("source_closure", {})
        .get("d44_d43_helper_sha256")
        != d43_helper_sha256
    ):
        raise D44ProbeError("D44 helper source closure drift")
    training_rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    verified_d44_fit_row_count = _verify_d44_fit_audits(training_rows)
    return {
        **evidence,
        "verified_d44_fit_row_count": verified_d44_fit_row_count,
        "verified_d44_helper_sha256": d43_helper_sha256,
        "verified_d44_resource_fit_count": 4,
        "verified_d44_fused_query_state_count": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d44-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D44ProbeError(f"D44 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_lda_fit_macs = None
    original_top_level_fit = None
    runner_module_name = "d44_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d43_helper_sha256 = d43._sha256(D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_full_block_rms_fit(d42)
        original_lda_fit_macs, original_top_level_fit = (
            _install_d44_core_resource_accounting(d42)
        )
        runner = (
            known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        )
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D44ProbeError("D44 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d44_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d44_d43_helper_sha256": d43_helper_sha256,
            },
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_lda_fit_macs is not None:
            d42._lda_fit_macs = original_lda_fit_macs
        if d42 is not None and original_top_level_fit is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top_level_fit
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_d44_output(
        output,
        probe_script_sha256,
        d43_helper_sha256,
    )
    metadata = {
        "schema": "cvs.phase2.d44.full_block_rms_fusion_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d44_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "fusion_weight": FUSION_WEIGHT,
        "scale_formula": SCALE_FORMULA,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        "runtime_legacy_sha256": d43.RUNTIME_LEGACY_SHA256,
        "runtime_module_sha256": dict(d43.RUNTIME_MODULE_SHA256),
        **evidence,
    }
    (output / "D44_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
