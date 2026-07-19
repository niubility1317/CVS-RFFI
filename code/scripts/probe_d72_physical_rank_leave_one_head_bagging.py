#!/usr/bin/env python3
"""D72 support-only physical-rank leave-one affine-head bagging probe."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D62_HELPER_PATH = SCRIPT_DIR / "probe_d62_crossfitted_fisher_row_splice.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d72_leave_one_head_bagging.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D72 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d72_d62_probe_helper", D62_HELPER_PATH)
core = _load("d72_leave_one_bagging_core", CORE_PATH)
d43 = d62.d43

ARM = "physical_rank_leave_one_head_bagging"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "freeze D62 metric; fit one complete anonymous D62 affine head per "
    "physical-rank leave-one support subset; arithmetic-mean and compile "
    "to one all-registered int8 affine state"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D72ProbeError(RuntimeError):
    pass


def build_d72_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
    base_fit, component_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, audit = base_fit(
            rows, labels, class_count, k_shot
        )
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d72_probe_arm": ARM,
                "d72_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, component_records


def _stage_added_resource(d42: Any, k_shot: int, class_count: int) -> dict[str, int]:
    k = int(k_shot)
    classes = int(class_count)
    if k <= 2:
        return {
            "d62_fit_count": 0,
            "component_fit_count": 0,
            "lda_fit_macs": 0,
            "fisher_dense_macs": 0,
            "gate_scalar_macs": 0,
            "mean_scalar_macs": 0,
        }
    inner_k = k - 1
    per_head_component_fits = 4 * (inner_k + 1)
    per_head_lda = 4 * int(d42._lda_fit_macs(inner_k * classes, classes))
    per_head_lda += 4 * inner_k * int(
        d42._lda_fit_macs((inner_k - 1) * classes, classes)
    )
    fisher_fit_count = k * 2 * (inner_k + 1)
    fisher = int(d62.d61._fisher_dense_macs(int(d42.FEATURE_DIM), fisher_fit_count))
    gate = int(k * inner_k * classes * classes * 8)
    mean = int(k * classes * (int(d42.FEATURE_DIM) + 1))
    return {
        "d62_fit_count": k,
        "component_fit_count": k * per_head_component_fits,
        "lda_fit_macs": k * per_head_lda,
        "fisher_dense_macs": fisher,
        "gate_scalar_macs": gate,
        "mean_scalar_macs": mean,
    }


def _compile_pair(
    d42: Any,
    template: Any,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> tuple[Any, Any, dict[str, float]]:
    int8, quant = d42._compile_state(
        tuple(template.classes),
        int(template.old_class_count),
        template.log_diag_fp32,
        coefficient,
        intercept,
        str(template.covariance_policy),
        precision="int8",
    )
    fp32, _ = d42._compile_state(
        tuple(template.classes),
        int(template.old_class_count),
        template.log_diag_fp32,
        coefficient,
        intercept,
        str(template.covariance_policy),
        precision="fp32",
    )
    return int8, fp32, quant


class BaggingRegistry:
    def __init__(self, d42: Any, base_fit: Any) -> None:
        self.d42 = d42
        self.base_fit = base_fit
        self.top_fit_count = 0
        self.inner_base_fit_count = 0
        self.records: list[dict[str, Any]] = []

    def wrap_top(self, base_top: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = base_top(*args, **kwargs)
            old_features = np.asarray(args[0], dtype=np.float32)
            old_labels = tuple(str(value) for value in args[1])
            old_classes = tuple(str(value) for value in args[2])
            new_features = np.asarray(args[3], dtype=np.float32)
            new_labels = tuple(str(value) for value in args[4])
            new_classes = tuple(str(value) for value in args[5])
            old_targets = np.asarray(
                [old_classes.index(value) for value in old_labels], dtype=np.int64
            )
            all_classes = old_classes + new_classes
            all_features = np.concatenate([old_features, new_features], axis=0)
            all_labels = old_labels + new_labels
            all_targets = np.asarray(
                [all_classes.index(value) for value in all_labels], dtype=np.int64
            )
            old_k = int(len(old_features) // len(old_classes))
            all_k = int(len(all_features) // len(all_classes))
            old_transformed = self.d42._transform(
                old_features, result.before_state.log_diag_fp32
            )
            all_transformed = self.d42._transform(
                all_features, result.state.log_diag_fp32
            )
            before_full_w = self.d42.decode_d42_coefficients(
                result.matched_fp32_before_state
            )
            before_full_b = result.matched_fp32_before_state.intercept_fp32
            final_full_w = self.d42.decode_d42_coefficients(
                result.matched_fp32_state
            )
            final_full_b = result.matched_fp32_state.intercept_fp32
            before_w, before_b, before_audit = (
                core.fit_leave_one_bagged_affine(
                    old_transformed,
                    old_targets,
                    len(old_classes),
                    old_k,
                    self.base_fit,
                    before_full_w,
                    before_full_b,
                )
            )
            final_w, final_b, final_audit = core.fit_leave_one_bagged_affine(
                all_transformed,
                all_targets,
                len(all_classes),
                all_k,
                self.base_fit,
                final_full_w,
                final_full_b,
            )
            before_int8, before_fp32, before_quant = _compile_pair(
                self.d42, result.before_state, before_w, before_b
            )
            final_int8, final_fp32, final_quant = _compile_pair(
                self.d42, result.state, final_w, final_b
            )
            before_int8_scores = self.d42.score_d42_unified_shrinkage_lda(
                before_int8, old_features
            )
            before_fp32_scores = self.d42.score_d42_unified_shrinkage_lda(
                before_fp32, old_features
            )
            final_int8_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_int8, all_features
            )
            final_fp32_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_fp32, all_features
            )
            before_changes = int(
                np.sum(
                    np.argmax(before_int8_scores, axis=1)
                    != np.argmax(before_fp32_scores, axis=1)
                )
            )
            final_changes = int(
                np.sum(
                    np.argmax(final_int8_scores, axis=1)
                    != np.argmax(final_fp32_scores, axis=1)
                )
            )
            before_error = float(
                np.max(np.abs(before_int8_scores - before_fp32_scores))
            )
            final_error = float(
                np.max(np.abs(final_int8_scores - final_fp32_scores))
            )
            geometry = dict(result.geometry_audit)
            geometry.update(
                {
                    "d72_probe_arm": ARM,
                    "d72_formula": FORMULA,
                    "d72_before_bagging_audit": before_audit,
                    "d72_final_bagging_audit": final_audit,
                    "d72_class_id_specific_formula": False,
                    "d72_old_new_role_specific_branch": False,
                    "d72_scene_receiver_handle_specific_branch": False,
                    "d72_uses_outer_held_or_query_for_fit": False,
                    "d72_query_joint_optimization": False,
                    "d72_ground_component_input_count": 0,
                    "d72_single_affine_state_only": True,
                    "d72_dense_query_graph_bytes": 0,
                    "before_coefficient_quantization_error_mean": before_quant[
                        "coefficient_quantization_error_mean"
                    ],
                    "before_coefficient_quantization_error_max": before_quant[
                        "coefficient_quantization_error_max"
                    ],
                    "before_intercept_quantization_error_mean": before_quant[
                        "intercept_quantization_error_mean"
                    ],
                    "before_intercept_quantization_error_max": before_quant[
                        "intercept_quantization_error_max"
                    ],
                    "final_coefficient_quantization_error_mean": final_quant[
                        "coefficient_quantization_error_mean"
                    ],
                    "final_coefficient_quantization_error_max": final_quant[
                        "coefficient_quantization_error_max"
                    ],
                    "final_intercept_quantization_error_mean": final_quant[
                        "intercept_quantization_error_mean"
                    ],
                    "final_intercept_quantization_error_max": final_quant[
                        "intercept_quantization_error_max"
                    ],
                    "before_support_score_max_abs_error": before_error,
                    "final_support_score_max_abs_error": final_error,
                    "int8_vs_fp32_before_support_argmax_change_count": before_changes,
                    "int8_vs_fp32_final_support_argmax_change_count": final_changes,
                }
            )
            before_resource = _stage_added_resource(
                self.d42, old_k, len(old_classes)
            )
            final_resource = _stage_added_resource(
                self.d42, all_k, len(all_classes)
            )
            added_fit_count = (
                before_resource["d62_fit_count"]
                + final_resource["d62_fit_count"]
            )
            added_component_fits = (
                before_resource["component_fit_count"]
                + final_resource["component_fit_count"]
            )
            added_lda = (
                before_resource["lda_fit_macs"]
                + final_resource["lda_fit_macs"]
            )
            added_fisher = (
                before_resource["fisher_dense_macs"]
                + final_resource["fisher_dense_macs"]
            )
            added_gate = (
                before_resource["gate_scalar_macs"]
                + final_resource["gate_scalar_macs"]
            )
            added_mean = (
                before_resource["mean_scalar_macs"]
                + final_resource["mean_scalar_macs"]
            )
            added_total = added_lda + added_fisher + added_gate + added_mean
            resource = dict(result.resource_audit)
            resource.update(
                {
                    "d72_inner_d62_fit_count": int(added_fit_count),
                    "d72_inner_component_fit_count": int(added_component_fits),
                    "d72_inner_lda_fit_macs": int(added_lda),
                    "d72_inner_fisher_dense_mac_upper_bound": int(added_fisher),
                    "d72_gate_scalar_mac_equivalents": int(added_gate),
                    "d72_mean_scalar_mac_equivalents": int(added_mean),
                    "d72_total_added_adaptation_macs": int(added_total),
                    "d72_query_extra_mac_equivalents": 0,
                    "d72_persistent_state_extra_bytes": 0,
                    "d72_ground_component_input_count": 0,
                    "d72_dense_query_graph_bytes": 0,
                    "d72_single_affine_state_only": True,
                    "persistent_state_bytes": int(final_int8.persistent_state_bytes),
                    "registry_state_bytes": int(final_int8.registry_state_bytes),
                    "int8_vs_fp32_before_support_argmax_change_count": before_changes,
                    "int8_vs_fp32_final_support_argmax_change_count": final_changes,
                }
            )
            resource["lda_closed_form_fit_count"] = int(
                resource["lda_closed_form_fit_count"] + added_component_fits
            )
            resource["estimated_lda_fit_macs"] = int(
                resource["estimated_lda_fit_macs"] + added_lda
            )
            resource["estimated_adaptation_macs"] = int(
                resource["estimated_adaptation_macs"] + added_total
            )
            self.top_fit_count += 1
            self.inner_base_fit_count += int(added_fit_count)
            self.records.append(
                {
                    "before_status": before_audit["status"],
                    "final_status": final_audit["status"],
                    "before_fit_count": before_audit["leave_one_fit_count"],
                    "final_fit_count": final_audit["leave_one_fit_count"],
                    "before_support_prediction_change_count": before_audit[
                        "support_prediction_change_count"
                    ],
                    "final_support_prediction_change_count": final_audit[
                        "support_prediction_change_count"
                    ],
                    "before_coefficient_dispersion_rms": before_audit[
                        "coefficient_dispersion_rms"
                    ],
                    "final_coefficient_dispersion_rms": final_audit[
                        "coefficient_dispersion_rms"
                    ],
                }
            )
            return replace(
                result,
                before_state=before_int8,
                state=final_int8,
                matched_fp32_before_state=before_fp32,
                matched_fp32_state=final_fp32,
                geometry_audit=geometry,
                resource_audit=resource,
            )

        return wrapped


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D72ProbeError("D72 training row closure drift")
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        resource = row["resource"]
        resource["lda_closed_form_fit_count"] -= resource[
            "d72_inner_component_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d72_inner_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d72_total_added_adaptation_macs"
        ]
        for field in ("before_covariance_audit", "final_covariance_audit"):
            row["geometry_summary"][field]["d43_probe_arm"] = d62.ARM
    d62_evidence = d62._verify_rows(sanitized)
    before_fit_count = final_fit_count = 0
    support_changes = 0
    for row in target:
        geometry = row["geometry_summary"]
        resource = row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d72_probe_arm": ARM,
                "d72_formula": FORMULA,
                "d72_class_id_specific_formula": False,
                "d72_old_new_role_specific_branch": False,
                "d72_scene_receiver_handle_specific_branch": False,
                "d72_uses_outer_held_or_query_for_fit": False,
                "d72_query_joint_optimization": False,
                "d72_ground_component_input_count": 0,
                "d72_single_affine_state_only": True,
                "d72_dense_query_graph_bytes": 0,
            }.items()
        ):
            raise D72ProbeError("D72 geometry closure drift")
        for audit, class_count in (
            (geometry["d72_before_bagging_audit"], 6),
            (geometry["d72_final_bagging_audit"], 11),
        ):
            if (
                audit["status"]
                != "physical_rank_leave_one_head_bagging_active"
                or audit["leave_one_fit_count"] != 8
                or audit["inner_k_shot"] != 7
                or audit["partition_exact_once"] is not True
                or len(audit["partition_audit"]) != 8
                or any(
                    part["train_held_overlap_count"] != 0
                    or part["held_class_histogram"] != [1] * class_count
                    for part in audit["partition_audit"]
                )
            ):
                raise D72ProbeError("D72 partition/bagging closure drift")
        before_fit_count += int(
            geometry["d72_before_bagging_audit"]["leave_one_fit_count"]
        )
        final_fit_count += int(
            geometry["d72_final_bagging_audit"]["leave_one_fit_count"]
        )
        support_changes += int(
            geometry["d72_before_bagging_audit"][
                "support_prediction_change_count"
            ]
            + geometry["d72_final_bagging_audit"][
                "support_prediction_change_count"
            ]
        )
        if any(
            resource.get(name) != value
            for name, value in {
                "d72_inner_d62_fit_count": 16,
                "d72_inner_component_fit_count": 512,
                "d72_query_extra_mac_equivalents": 0,
                "d72_persistent_state_extra_bytes": 0,
                "d72_ground_component_input_count": 0,
                "d72_dense_query_graph_bytes": 0,
                "d72_single_affine_state_only": True,
            }.items()
        ):
            raise D72ProbeError("D72 resource closure drift")
        if int(resource["persistent_state_bytes"]) > 256 * 1024:
            raise D72ProbeError("D72 state cap drift")
    return {
        **d62_evidence,
        "verified_d72_target_row_count": len(target),
        "verified_d72_fit_audit_count": 2 * len(target),
        "verified_d72_before_leave_one_fit_count": before_fit_count,
        "verified_d72_final_leave_one_fit_count": final_fit_count,
        "verified_d72_support_prediction_change_count": support_changes,
        "verified_d72_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D72ProbeError("D72 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d72-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D72ProbeError(f"D72 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d72_core_sha256": d43._sha256(CORE_PATH),
        "d72_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d72_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d72_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d72_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d72_locked_d42_runner", 1
    registry = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, component_records = build_d72_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = BaggingRegistry(d42, fit)
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D72ProbeError("D72 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d72_arm,
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
    if (
        registry is None
        or registry.top_fit_count != 30
        or registry.inner_base_fit_count != 480
        or len(registry.records) != 30
        or len(component_records) != 8760
    ):
        raise D72ProbeError("D72 fit/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(
            registry.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d72.physical_rank_leave_one_head_bagging_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d72_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "top_fit_count": registry.top_fit_count,
        "inner_d62_fit_count": registry.inner_base_fit_count,
        "component_fit_execution_count": len(component_records),
        "fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D72_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
