#!/usr/bin/env python3
"""D77 support-only ground-preconditioned all-class common-descent probe."""

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
D66_HELPER_PATH = SCRIPT_DIR / "probe_d66_ground_domain_reliability_residual.py"
CORE_PATH = (
    SCRIPT_DIR.parent
    / "cvsrffi"
    / "stage2_d77_ground_preconditioned_common_descent.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D77 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d66 = _load("d77_d66_probe_helper", D66_HELPER_PATH)
core = _load("d77_ground_preconditioned_common_descent_core", CORE_PATH)
d62, d43 = d66.d62, d66.d43

ARM = "ground_preconditioned_allclass_common_descent"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "derive a positive determinant-neutral coordinate metric from immutable int8 "
    "ground domain reliability; obtain one CE gradient for every registered class "
    "from leave-one-physical-rank equal-prior LDA; solve a fixed-20-step class-simplex "
    "minimum-M-norm combination; apply its analytic common-descent residual directly "
    "to frozen D62 final rows and compile one int8 affine state without refit"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D77ProbeError(RuntimeError):
    pass


def build_d77_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
    base_fit, component_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, audit = base_fit(rows, labels, class_count, k_shot)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d77_probe_arm": ARM,
                "d77_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, component_records


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


def _resource_upper_bounds(
    *,
    k_shot: int,
    class_count: int,
    dimension: int,
    lda_macs: int,
    ground_statistics_macs: int,
) -> dict[str, int]:
    k, classes, d = int(k_shot), int(class_count), int(dimension)
    gradient = int(k * classes * (4 * classes * d + 4 * d + 2 * classes))
    frank_wolfe = int(
        core.FW_ITERATIONS * (4 * classes * classes * d + 8 * classes * d + 6 * classes)
    )
    ce_audit = int(2 * k * classes * (2 * classes * d + 3 * classes + d))
    precondition = int(classes * d)
    compile_macs = int(2 * classes * d)
    non_lda = int(
        ground_statistics_macs
        + gradient
        + frank_wolfe
        + ce_audit
        + precondition
        + compile_macs
    )
    return {
        "crossfit_lda_fit_macs": int(lda_macs),
        "oof_gradient_mac_upper_bound": gradient,
        "frank_wolfe_mac_upper_bound": frank_wolfe,
        "oof_ce_audit_mac_upper_bound": ce_audit,
        "preconditioner_application_macs": precondition,
        "affine_compile_mac_equivalents": compile_macs,
        "ground_statistics_macs": int(ground_statistics_macs),
        "non_lda_total": non_lda,
        "total_added": int(lda_macs + non_lda),
    }


class GroundCommonDescentRegistry:
    def __init__(
        self,
        d42: Any,
        native_lda_fit: Any,
        native_lda_macs: Any,
        preconditioner: np.ndarray,
        preconditioner_audit: dict[str, Any],
        ground_audit: dict[str, Any],
    ) -> None:
        self.d42 = d42
        self.native_lda_fit = native_lda_fit
        self.native_lda_macs = native_lda_macs
        self.preconditioner = np.asarray(preconditioner, dtype=np.float64)
        self.preconditioner_audit = dict(preconditioner_audit)
        self.ground_audit = dict(ground_audit)
        self.top_fit_count = 0
        self.active_count = 0
        self.fallback_count = 0
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
            all_classes = old_classes + new_classes
            all_features = np.concatenate([old_features, new_features], axis=0)
            all_labels = old_labels + new_labels
            all_targets = np.asarray(
                [all_classes.index(value) for value in all_labels], dtype=np.int64
            )
            all_k = int(len(all_features) // len(all_classes))
            transformed = self.d42._transform(
                all_features, result.state.log_diag_fp32
            )
            fitted_w = self.d42.decode_d42_coefficients(
                result.matched_fp32_state
            )
            fitted_b = np.asarray(
                result.matched_fp32_state.intercept_fp32, dtype=np.float32
            )
            residual, descent_audit = core.fit_ground_preconditioned_common_descent(
                transformed,
                all_targets,
                len(all_classes),
                all_k,
                base_coefficient=fitted_w,
                preconditioner=self.preconditioner,
                lda_fit=self.native_lda_fit,
            )
            active = bool(descent_audit["residual_active"])
            compiled_w = np.asarray(fitted_w, dtype=np.float64) + np.asarray(
                residual, dtype=np.float64
            )
            compiled_w = np.asarray(compiled_w, dtype=np.float32)
            compiled_b = np.asarray(fitted_b, dtype=np.float32)
            final_int8, final_fp32, quant = _compile_pair(
                self.d42, result.state, compiled_w, compiled_b
            )
            base_scores = self.d42.score_d42_unified_shrinkage_lda(
                result.matched_fp32_state, all_features
            )
            final_int8_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_int8, all_features
            )
            final_fp32_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_fp32, all_features
            )
            base_prediction = np.argmax(base_scores, axis=1)
            final_prediction = np.argmax(final_fp32_scores, axis=1)
            quant_changes = int(
                np.sum(np.argmax(final_int8_scores, axis=1) != final_prediction)
            )
            quant_error = float(
                np.max(np.abs(final_int8_scores - final_fp32_scores))
            )
            descent_audit = dict(descent_audit)
            descent_audit.update(
                {
                    "support_prediction_change_count": int(
                        np.sum(base_prediction != final_prediction)
                    ),
                    "support_accuracy_base": float(
                        np.mean(base_prediction == all_targets)
                    ),
                    "support_accuracy_updated": float(
                        np.mean(final_prediction == all_targets)
                    ),
                    "ground_preconditioner_sha256": self.preconditioner_audit[
                        "preconditioner_sha256"
                    ],
                    "ground_component_input_count": int(
                        self.ground_audit["ground_active_domain_class_cells"]
                    ),
                    "ground_component_formal_phase2_eligible": bool(
                        self.ground_audit["component_formal_phase2_eligible"]
                    ),
                    "ground_component_provenance_status": self.ground_audit[
                        "component_provenance_status"
                    ],
                }
            )
            geometry = dict(result.geometry_audit)
            geometry.update(
                {
                    "d77_probe_arm": ARM,
                    "d77_formula": FORMULA,
                    "d77_ground_preconditioner_audit": self.preconditioner_audit,
                    "d77_common_descent_audit": descent_audit,
                    "d77_before_state_exact_d62_unchanged": True,
                    "d77_class_id_specific_formula": False,
                    "d77_old_new_role_specific_branch": False,
                    "d77_query_role_specific_branch": False,
                    "d77_scene_receiver_handle_specific_branch": False,
                    "d77_uses_outer_held_or_query_for_fit": False,
                    "d77_query_joint_optimization": False,
                    "d77_ground_component_input_count": int(
                        self.ground_audit["ground_active_domain_class_cells"]
                    ),
                    "d77_ground_class_score_access": False,
                    "d77_ground_component_update_access": False,
                    "d77_residual_persisted_separately": False,
                    "d77_residual_compiled_into_affine": active,
                    "d77_single_affine_state_only": True,
                    "d77_dense_query_graph_bytes": 0,
                    "metric_frozen_during_stage2c": True,
                    "stage2c_log_diag_frozen": True,
                    "stage2c_classifier": (
                        "d77_ground_preconditioned_common_descent_compiled_affine"
                        if active
                        else "d77_degenerate_exact_d62_fallback"
                    ),
                    "final_coefficient_quantization_error_mean": quant[
                        "coefficient_quantization_error_mean"
                    ],
                    "final_coefficient_quantization_error_max": quant[
                        "coefficient_quantization_error_max"
                    ],
                    "final_intercept_quantization_error_mean": quant[
                        "intercept_quantization_error_mean"
                    ],
                    "final_intercept_quantization_error_max": quant[
                        "intercept_quantization_error_max"
                    ],
                    "final_support_score_max_abs_error": quant_error,
                    "int8_vs_fp32_final_support_argmax_change_count": quant_changes,
                }
            )
            k = int(descent_audit["crossfit_fold_count"])
            classes = int(descent_audit["class_count"])
            dimension = int(descent_audit["dimension"])
            lda_macs = int(
                k * self.native_lda_macs((k - 1) * classes, classes)
            )
            bounds = _resource_upper_bounds(
                k_shot=k,
                class_count=classes,
                dimension=dimension,
                lda_macs=lda_macs,
                ground_statistics_macs=int(
                    self.ground_audit[
                        "ground_reliability_statistics_scalar_mac_equivalents"
                    ]
                ),
            )
            component_bytes = int(
                self.ground_audit["ground_int8_component_logical_state_bytes"]
            )
            head_bytes = int(final_int8.persistent_state_bytes)
            base_steps = int(result.resource_audit["optimizer_steps"])
            trace = [dict(item) for item in result.training_trace]
            for item in descent_audit["optimizer_objective_trace"]:
                trace.append(
                    {
                        "phase": "stage2c_ground_preconditioned_common_descent_frank_wolfe",
                        "epoch": int(result.resource_audit["adaptation_epochs"]),
                        "optimizer_step": base_steps + int(item["iteration"]),
                        "fw_iteration": int(item["iteration"]),
                        "loss": float(item["objective_after"]),
                        "ce_loss": float(descent_audit["oof_ce_after_mean"]),
                        "metric_gradient_norm": float(
                            np.sqrt(max(0.0, item["objective_after"]))
                        ),
                        "line_search_gamma": float(item["line_search_gamma"]),
                        "support_accuracy": float(
                            descent_audit["oof_updated_correct_count"]
                            / descent_audit["crossfit_held_row_count"]
                        ),
                        "prototype_anchor_loss": 0.0,
                        "ground_component_input_count": int(
                            self.ground_audit["ground_active_domain_class_cells"]
                        ),
                        "query_rows_used": 0,
                    }
                )
            resource = dict(result.resource_audit)
            resource.update(
                {
                    "d77_crossfit_fold_count": k,
                    "d77_crossfit_held_row_count": int(
                        descent_audit["crossfit_held_row_count"]
                    ),
                    "d77_crossfit_lda_fit_count": k,
                    "d77_crossfit_lda_fit_macs": bounds[
                        "crossfit_lda_fit_macs"
                    ],
                    "d77_oof_gradient_mac_upper_bound": bounds[
                        "oof_gradient_mac_upper_bound"
                    ],
                    "d77_frank_wolfe_mac_upper_bound": bounds[
                        "frank_wolfe_mac_upper_bound"
                    ],
                    "d77_oof_ce_audit_mac_upper_bound": bounds[
                        "oof_ce_audit_mac_upper_bound"
                    ],
                    "d77_preconditioner_application_macs": bounds[
                        "preconditioner_application_macs"
                    ],
                    "d77_affine_compile_mac_equivalents": bounds[
                        "affine_compile_mac_equivalents"
                    ],
                    "d77_ground_statistics_macs": bounds["ground_statistics_macs"],
                    "d77_non_lda_added_adaptation_macs": bounds["non_lda_total"],
                    "d77_total_added_adaptation_macs": bounds["total_added"],
                    "d77_frank_wolfe_optimizer_steps": core.FW_ITERATIONS,
                    "d77_transient_simplex_parameter_count": classes,
                    "d77_residual_active": active,
                    "d77_query_extra_mac_equivalents": 0,
                    "d77_query_extra_state_bytes": 0,
                    "d77_ground_component_input_count": int(
                        self.ground_audit["ground_active_domain_class_cells"]
                    ),
                    "d77_ground_component_update_access": False,
                    "d77_ground_class_score_access": False,
                    "d77_ground_component_logical_state_bytes": component_bytes,
                    "d77_transient_dequantized_ground_bytes": int(
                        self.ground_audit["transient_dequantized_ground_bytes"]
                    ),
                    "d77_compiled_affine_state_bytes": head_bytes,
                    "d77_component_inclusive_persistent_state_bytes": (
                        head_bytes + component_bytes
                    ),
                    "d77_dense_query_graph_bytes": 0,
                    "d77_single_affine_state_only": True,
                    "persistent_state_bytes": head_bytes + component_bytes,
                    "registry_state_bytes": int(final_int8.registry_state_bytes),
                    "ground_int8_component_input_count": int(
                        self.ground_audit["ground_active_domain_class_cells"]
                    ),
                    "ground_int8_update_access": False,
                    "int8_vs_fp32_final_support_argmax_change_count": quant_changes,
                    "complete_loss_trace": trace,
                }
            )
            resource["lda_closed_form_fit_count"] = int(
                resource["lda_closed_form_fit_count"] + k
            )
            resource["estimated_lda_fit_macs"] = int(
                resource["estimated_lda_fit_macs"] + lda_macs
            )
            resource["estimated_adaptation_macs"] = int(
                resource["estimated_adaptation_macs"] + bounds["total_added"]
            )
            resource["estimated_metric_adaptation_macs"] = int(
                resource["estimated_metric_adaptation_macs"]
                + bounds["non_lda_total"]
            )
            resource["optimizer_steps"] = int(
                resource["optimizer_steps"] + core.FW_ITERATIONS
            )
            resource["stage2c_optimizer_steps"] = int(
                resource["stage2c_optimizer_steps"] + core.FW_ITERATIONS
            )
            resource["trainable_parameters"] = int(
                resource["trainable_parameters"] + classes
            )
            resource["optimizer_step_cap_pass"] = bool(
                resource["optimizer_steps"] <= resource["optimizer_step_cap"]
            )
            resource["trainable_parameter_cap_pass"] = bool(
                resource["trainable_parameters"]
                <= resource["trainable_parameter_cap"]
            )
            resource["persistent_state_cap_pass"] = bool(
                resource["persistent_state_bytes"]
                <= resource["persistent_state_cap_bytes"]
            )
            self.top_fit_count += 1
            self.active_count += int(active)
            self.fallback_count += int(not active)
            self.records.append(
                {
                    "status": descent_audit["status"],
                    "residual_sha256": descent_audit["residual_sha256"],
                    "oof_ce_delta_mean": descent_audit["oof_ce_delta_mean"],
                    "oof_ce_delta_max_class": descent_audit[
                        "oof_ce_delta_max_class"
                    ],
                    "support_prediction_change_count": descent_audit[
                        "support_prediction_change_count"
                    ],
                }
            )
            return replace(
                result,
                state=final_int8,
                matched_fp32_state=final_fp32,
                training_trace=tuple(trace),
                geometry_audit=geometry,
                resource_audit=resource,
            )

        return wrapped


def _install_runner_resource_accounting(runner: Any) -> None:
    original_evaluate = runner._evaluate_d42_fold

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_evaluate(*args, **kwargs)
        resource = dict(row["resource"])
        if "d77_ground_component_logical_state_bytes" not in resource:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d77_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d77_compiled_affine_state_bytes": head_bytes,
                "d77_component_inclusive_persistent_state_bytes": total_bytes,
                "persistent_state_bytes": total_bytes,
                "persistent_state_cap_pass": total_bytes
                <= int(resource["persistent_state_cap_bytes"]),
            }
        )
        return {**row, "resource": resource}

    runner._evaluate_d42_fold = evaluate


def _sanitize_for_d62(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        resource = row["resource"]
        resource["lda_closed_form_fit_count"] -= resource[
            "d77_crossfit_lda_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d77_crossfit_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d77_total_added_adaptation_macs"
        ]
        resource["estimated_metric_adaptation_macs"] -= resource[
            "d77_non_lda_added_adaptation_macs"
        ]
        steps = int(resource["d77_frank_wolfe_optimizer_steps"])
        resource["optimizer_steps"] -= steps
        resource["total_optimizer_steps"] -= steps
        resource["stage2c_optimizer_steps"] -= steps
        classes = int(resource["d77_transient_simplex_parameter_count"])
        resource["trainable_parameters"] -= classes
        resource["peak_trainable_parameters"] -= classes
        resource["complete_loss_trace"] = resource["complete_loss_trace"][:-steps]
        resource["persistent_state_bytes"] = resource[
            "d77_compiled_affine_state_bytes"
        ]
        resource["ground_int8_component_input_count"] = 0
        resource["optimizer_step_cap_pass"] = True
        resource["trainable_parameter_cap_pass"] = True
        resource["persistent_state_cap_pass"] = True
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            audit["d43_probe_arm"] = d62.ARM
            audit["d43_covariance_structure"] = d62.STRUCTURE
    return sanitized


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D77ProbeError("D77 training row closure drift")
    d62_evidence = d62._verify_rows(_sanitize_for_d62(rows))
    active = fallback = 0
    residual_hashes: set[str] = set()
    ce_deltas: list[float] = []
    for row in target:
        geometry, resource = row["geometry_summary"], row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d77_probe_arm": ARM,
                "d77_formula": FORMULA,
                "d77_before_state_exact_d62_unchanged": True,
                "d77_class_id_specific_formula": False,
                "d77_old_new_role_specific_branch": False,
                "d77_query_role_specific_branch": False,
                "d77_scene_receiver_handle_specific_branch": False,
                "d77_uses_outer_held_or_query_for_fit": False,
                "d77_query_joint_optimization": False,
                "d77_ground_component_input_count": 84,
                "d77_ground_class_score_access": False,
                "d77_ground_component_update_access": False,
                "d77_residual_persisted_separately": False,
                "d77_single_affine_state_only": True,
                "d77_dense_query_graph_bytes": 0,
            }.items()
        ):
            raise D77ProbeError("D77 geometry closure drift")
        audit = geometry["d77_common_descent_audit"]
        preconditioner = geometry["d77_ground_preconditioner_audit"]
        is_active = bool(audit["residual_active"])
        active += int(is_active)
        fallback += int(not is_active)
        if (
            int(audit["class_count"]) != 11
            or int(audit["k_shot"]) != 8
            or int(audit["dimension"]) != 288
            or int(audit["crossfit_fold_count"]) != 8
            or int(audit["crossfit_lda_fit_count"]) != 8
            or int(audit["crossfit_held_row_count"]) != 88
            or int(audit["class_gradient_count"]) != 11
            or int(audit["frank_wolfe_iterations"]) != core.FW_ITERATIONS
            or len(audit["optimizer_objective_trace"]) != core.FW_ITERATIONS
            or len(audit["oof_per_class_ce_delta"]) != 11
            or int(audit["query_rows_used"]) != 0
            or audit["ground_class_score_access"] is not False
            or audit["ground_component_formal_phase2_eligible"] is not False
            or audit["ground_component_input_count"] != 84
        ):
            raise D77ProbeError("D77 common-descent audit closure drift")
        if (
            int(preconditioner["z_dimension"]) != 160
            or int(preconditioner["feature_dimension"]) != 288
            or abs(float(preconditioner["preconditioner_z_geometric_mean"]) - 1.0)
            > 1e-12
            or preconditioner["ground_class_score_access"] is not False
        ):
            raise D77ProbeError("D77 preconditioner closure drift")
        if is_active:
            if (
                audit["status"]
                != "ground_preconditioned_allclass_common_descent_active"
                or float(audit["minimum_common_descent_inner_product"]) <= 0.0
                or float(audit["oof_ce_delta_max_class"])
                > float(audit["ce_numeric_tolerance"])
                or float(audit["oof_ce_delta_min_class"])
                >= -float(audit["ce_numeric_tolerance"])
            ):
                raise D77ProbeError("D77 active descent drift")
        elif audit["status"] != "degenerate_minimum_norm_exact_d62_fallback":
            raise D77ProbeError("D77 fallback drift")
        if any(
            resource.get(name) != value
            for name, value in {
                "d77_crossfit_fold_count": 8,
                "d77_crossfit_held_row_count": 88,
                "d77_crossfit_lda_fit_count": 8,
                "d77_frank_wolfe_optimizer_steps": 20,
                "d77_transient_simplex_parameter_count": 11,
                "d77_query_extra_mac_equivalents": 0,
                "d77_query_extra_state_bytes": 0,
                "d77_ground_component_input_count": 84,
                "d77_ground_component_update_access": False,
                "d77_ground_class_score_access": False,
                "d77_ground_component_logical_state_bytes": 25428,
                "d77_dense_query_graph_bytes": 0,
                "d77_single_affine_state_only": True,
                "ground_int8_component_input_count": 84,
                "ground_int8_update_access": False,
            }.items()
        ):
            raise D77ProbeError("D77 resource closure drift")
        if (
            int(resource["optimizer_steps"]) != 40
            or int(resource["total_optimizer_steps"]) != 40
            or int(resource["stage2c_optimizer_steps"]) != 20
            or int(resource["adaptation_epochs"]) != 20
            or len(resource["complete_loss_trace"]) != 40
            or int(resource["persistent_state_bytes"])
            != int(resource["d77_compiled_affine_state_bytes"])
            + int(resource["d77_ground_component_logical_state_bytes"])
            or int(resource["persistent_state_bytes"]) > 256 * 1024
            or int(resource["peak_trainable_parameters"]) > 80000
        ):
            raise D77ProbeError("D77 cap/resource drift")
        residual_hashes.add(str(audit["residual_sha256"]))
        ce_deltas.append(float(audit["oof_ce_delta_mean"]))
    return {
        **d62_evidence,
        "verified_d77_target_row_count": len(target),
        "verified_d77_common_descent_audit_count": len(target),
        "verified_d77_active_count": active,
        "verified_d77_fallback_count": fallback,
        "verified_d77_unique_residual_count": len(residual_hashes),
        "verified_d77_oof_ce_delta_min": min(ce_deltas),
        "verified_d77_oof_ce_delta_max": max(ce_deltas),
        "verified_d77_ground_component_input_count": 84,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D77ProbeError("D77 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d77-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-component-dir", required=True, type=Path)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D77ProbeError(f"D77 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d77_core_sha256": d43._sha256(CORE_PATH),
        "d77_d66_helper_sha256": d43._sha256(D66_HELPER_PATH),
        "d77_d62_helper_sha256": d43._sha256(d66.D62_HELPER_PATH),
        "d77_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d77_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d77_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    component_dir = known.ground_component_dir.resolve()
    component_npz = component_dir / d66.NPZ_NAME
    component_manifest = component_dir / d66.MANIFEST_NAME
    entry_npz_sha = d43._sha256(component_npz)
    entry_manifest_sha = d43._sha256(component_manifest)
    shared_scale, ground_audit = d66.load_ground_domain_reliability(
        component_dir,
        known.ground_manifest_sha256,
        288,
    )
    if (
        ground_audit["ground_active_domain_class_cells"] != 84
        or ground_audit["component_formal_phase2_eligible"] is not False
        or ground_audit["component_provenance_status"]
        != "UNVERIFIED_UNDER_CURRENT_PROTOCOL"
    ):
        raise D77ProbeError("D77 locked diagnostic ground component drift")
    preconditioner, preconditioner_audit = core.ground_reliability_preconditioner(
        shared_scale
    )
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = registry = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d77_locked_d42_runner", 1
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit, original_macs = d42._fit_equal_prior_lda, d42._lda_fit_macs
        fit, component_records = build_d77_fit(d42)
        d42._fit_equal_prior_lda = fit
        _, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = GroundCommonDescentRegistry(
            d42,
            original_fit,
            original_macs,
            preconditioner,
            preconditioner_audit,
            ground_audit,
        )
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D77ProbeError("D77 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        _install_runner_resource_accounting(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d77_arm,
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
        or registry.active_count + registry.fallback_count != 30
        or len(registry.records) != 30
        or len(component_records) != 1080
        or d43._sha256(component_npz) != entry_npz_sha
        or d43._sha256(component_manifest) != entry_manifest_sha
    ):
        raise D77ProbeError("D77 fit/component/read-only closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(registry.records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d77.ground_preconditioned_common_descent_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d77_arm,
        "formal_candidate": False,
        "component_formal_phase2_eligible": False,
        "component_provenance_status": "UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "top_fit_count": registry.top_fit_count,
        "active_count": registry.active_count,
        "fallback_count": registry.fallback_count,
        "component_fit_execution_count": len(component_records),
        "fit_record_sha256": record_sha,
        "ground_component_entry_npz_sha256": entry_npz_sha,
        "ground_component_exit_npz_sha256": d43._sha256(component_npz),
        "ground_component_entry_manifest_sha256": entry_manifest_sha,
        "ground_component_exit_manifest_sha256": d43._sha256(component_manifest),
        "ground_preconditioner_audit": preconditioner_audit,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D77_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
