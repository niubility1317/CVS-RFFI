#!/usr/bin/env python3
"""D73 support-only one-step conflict-projected joint metric probe."""

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
CORE_PATH = (
    SCRIPT_DIR.parent / "cvsrffi" / "stage2_d73_conflict_projected_joint_metric.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D73 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d73_d62_probe_helper", D62_HELPER_PATH)
core = _load("d73_conflict_projected_metric_core", CORE_PATH)
d43 = d62.d43

ARM = "conflict_projected_joint_metric"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "freeze D62 before state; form equal-priority old/new all-registered "
    "leave-one prototype gradients; symmetric project conflicts; take one "
    "sqrt(K/(K+D)) log-diagonal step; refit and compile one D62 int8 head"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D73ProbeError(RuntimeError):
    pass


def build_d73_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
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
                "d73_probe_arm": ARM,
                "d73_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, component_records


def _added_refit_resource(d42: Any, k_shot: int, class_count: int) -> dict[str, int]:
    k = int(k_shot)
    classes = int(class_count)
    if k <= 2:
        return {
            "component_fit_count": 0,
            "lda_fit_macs": 0,
            "fisher_dense_macs": 0,
            "gate_scalar_macs": 0,
        }
    component_fits = 4 * (k + 1)
    lda = 4 * int(d42._lda_fit_macs(k * classes, classes))
    lda += 4 * k * int(
        d42._lda_fit_macs((k - 1) * classes, classes)
    )
    fisher = int(
        d62.d61._fisher_dense_macs(int(d42.FEATURE_DIM), 2 * (k + 1))
    )
    gate = int(k * classes * classes * 8)
    return {
        "component_fit_count": int(component_fits),
        "lda_fit_macs": int(lda),
        "fisher_dense_macs": int(fisher),
        "gate_scalar_macs": int(gate),
    }


def _gradient_mac_upper_bound(
    dimension: int, class_count: int, k_shot: int
) -> int:
    rows = int(class_count) * int(k_shot)
    return int(8 * rows * int(class_count) * int(dimension) + 6 * rows * int(class_count) + 16 * int(dimension))


def _float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32).tobytes()
    ).hexdigest()


def _compile_pair(
    d42: Any,
    template: Any,
    log_diag: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> tuple[Any, Any, dict[str, float]]:
    int8, quant = d42._compile_state(
        tuple(template.classes),
        int(template.old_class_count),
        log_diag,
        coefficient,
        intercept,
        str(template.covariance_policy),
        precision="int8",
    )
    fp32, _ = d42._compile_state(
        tuple(template.classes),
        int(template.old_class_count),
        log_diag,
        coefficient,
        intercept,
        str(template.covariance_policy),
        precision="fp32",
    )
    return int8, fp32, quant


class MetricRegistry:
    def __init__(self, d42: Any, base_fit: Any) -> None:
        self.d42 = d42
        self.base_fit = base_fit
        self.top_fit_count = 0
        self.extra_d62_fit_count = 0
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
            updated_log_diag, metric_audit = (
                core.fit_conflict_projected_log_diag(
                    all_features,
                    all_targets,
                    len(old_classes),
                    len(all_classes),
                    all_k,
                    result.state.log_diag_fp32,
                )
            )
            base_log_diag_sha256 = _float32_sha256(result.state.log_diag_fp32)
            updated_log_diag_sha256 = _float32_sha256(updated_log_diag)
            metric_state_changed = base_log_diag_sha256 != updated_log_diag_sha256
            if int(metric_audit["stage2c_step_count"]) == 1 and not metric_state_changed:
                raise D73ProbeError("D73 active metric state did not change")
            transformed = self.d42._transform(all_features, updated_log_diag)
            if int(metric_audit["stage2c_step_count"]) == 0:
                final_w = self.d42.decode_d42_coefficients(
                    result.matched_fp32_state
                )
                final_b = np.asarray(
                    result.matched_fp32_state.intercept_fp32, dtype=np.float32
                )
                fit_audit = dict(result.geometry_audit["final_covariance_audit"])
                added = {
                    "component_fit_count": 0,
                    "lda_fit_macs": 0,
                    "fisher_dense_macs": 0,
                    "gate_scalar_macs": 0,
                }
            else:
                final_w, final_b, fit_audit = self.base_fit(
                    transformed,
                    all_targets,
                    len(all_classes),
                    all_k,
                )
                self.extra_d62_fit_count += 1
                added = _added_refit_resource(
                    self.d42, all_k, len(all_classes)
                )
            final_int8, final_fp32, quant = _compile_pair(
                self.d42,
                result.state,
                updated_log_diag,
                final_w,
                final_b,
            )
            int8_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_int8, all_features
            )
            fp32_scores = self.d42.score_d42_unified_shrinkage_lda(
                final_fp32, all_features
            )
            quant_changes = int(
                np.sum(
                    np.argmax(int8_scores, axis=1)
                    != np.argmax(fp32_scores, axis=1)
                )
            )
            quant_error = float(np.max(np.abs(int8_scores - fp32_scores)))
            geometry = dict(result.geometry_audit)
            geometry.update(
                {
                    "final_covariance_audit": fit_audit,
                    "d73_probe_arm": ARM,
                    "d73_formula": FORMULA,
                    "d73_metric_audit": metric_audit,
                    "d73_base_final_log_diag_sha256": base_log_diag_sha256,
                    "d73_updated_final_log_diag_sha256": updated_log_diag_sha256,
                    "d73_metric_state_changed": metric_state_changed,
                    "d73_before_state_exact_d62_unchanged": True,
                    "d73_class_id_specific_formula": False,
                    "d73_support_role_tasks_equal_priority": True,
                    "d73_query_role_specific_branch": False,
                    "d73_scene_receiver_handle_specific_branch": False,
                    "d73_uses_outer_held_or_query_for_fit": False,
                    "d73_query_joint_optimization": False,
                    "d73_ground_component_input_count": 0,
                    "d73_single_affine_state_only": True,
                    "d73_dense_query_graph_bytes": 0,
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
                    "old_log_diag_final_sha256": updated_log_diag_sha256,
                    "old_log_diag_bitwise_unchanged": False,
                    "metric_frozen_during_stage2c": False,
                    "stage2c_log_diag_frozen": False,
                    "metric_source": "d42_stage2b_plus_d73_one_step_support_only",
                    "stage2c_classifier": "d73_conflict_projected_metric_plus_d62_joint_lda",
                }
            )
            dimension = int(self.d42.FEATURE_DIM)
            gradient_macs = _gradient_mac_upper_bound(
                dimension, len(all_classes), all_k
            )
            added_total = int(
                added["lda_fit_macs"]
                + added["fisher_dense_macs"]
                + added["gate_scalar_macs"]
                + gradient_macs
            )
            resource = dict(result.resource_audit)
            base_metric_macs = int(resource["estimated_metric_adaptation_macs"])
            trace = [dict(item) for item in result.training_trace]
            if int(metric_audit["stage2c_step_count"]) == 1:
                trace.append(
                    {
                        "epoch": int(resource["adaptation_epochs"]) + 1,
                        "optimizer_step": int(resource["optimizer_steps"]) + 1,
                        "phase": "stage2c_equal_priority_conflict_projected_metric",
                        "old_leave_one_ce_loss_before": metric_audit[
                            "old_loss_before"
                        ],
                        "new_leave_one_ce_loss_before": metric_audit[
                            "new_loss_before"
                        ],
                        "old_leave_one_ce_loss_after": metric_audit[
                            "old_loss_after"
                        ],
                        "new_leave_one_ce_loss_after": metric_audit[
                            "new_loss_after"
                        ],
                        "old_gradient_norm": metric_audit["old_gradient_l2"],
                        "new_gradient_norm": metric_audit["new_gradient_l2"],
                        "task_gradient_cosine": metric_audit[
                            "task_gradient_cosine"
                        ],
                        "conflict_projection_active": metric_audit[
                            "conflict_projection_active"
                        ],
                        "delta_l2": metric_audit["delta_l2"],
                        "query_rows_used": 0,
                    }
                )
            step_count = int(metric_audit["stage2c_step_count"])
            resource.update(
                {
                    "complete_loss_trace": trace,
                    "d73_stage2c_step_count": step_count,
                    "d73_metric_parameter_count": dimension,
                    "d73_additional_component_fit_count": int(
                        added["component_fit_count"]
                    ),
                    "d73_additional_lda_fit_macs": int(added["lda_fit_macs"]),
                    "d73_fisher_dense_mac_upper_bound": int(
                        added["fisher_dense_macs"]
                    ),
                    "d73_gate_scalar_mac_equivalents": int(
                        added["gate_scalar_macs"]
                    ),
                    "d73_gradient_mac_equivalent_upper_bound": gradient_macs,
                    "d73_base_metric_adaptation_macs": base_metric_macs,
                    "d73_total_added_adaptation_macs": added_total,
                    "d73_query_extra_mac_equivalents": 0,
                    "d73_persistent_state_extra_bytes": 0,
                    "d73_ground_component_input_count": 0,
                    "d73_dense_query_graph_bytes": 0,
                    "d73_single_affine_state_only": True,
                    "persistent_state_bytes": int(final_int8.persistent_state_bytes),
                    "registry_state_bytes": int(final_int8.registry_state_bytes),
                    "int8_vs_fp32_final_support_argmax_change_count": quant_changes,
                }
            )
            resource["lda_closed_form_fit_count"] = int(
                resource["lda_closed_form_fit_count"]
                + added["component_fit_count"]
            )
            resource["estimated_lda_fit_macs"] = int(
                resource["estimated_lda_fit_macs"] + added["lda_fit_macs"]
            )
            resource["estimated_adaptation_macs"] = int(
                resource["estimated_adaptation_macs"] + added_total
            )
            resource["estimated_metric_adaptation_macs"] = int(
                resource["estimated_metric_adaptation_macs"] + gradient_macs
            )
            for field in (
                "adaptation_epochs",
                "metric_epochs",
                "optimizer_steps",
                "metric_optimizer_steps",
                "stage2c_optimizer_steps",
            ):
                if field not in resource:
                    raise D73ProbeError(f"D73 base resource missing {field}")
                resource[field] = int(resource[field]) + step_count
            resource["adaptation_epoch_cap_pass"] = bool(
                resource["adaptation_epochs"] <= resource["adaptation_epoch_cap"]
            )
            resource["optimizer_step_cap_pass"] = bool(
                resource["optimizer_steps"] <= resource["optimizer_step_cap"]
            )
            self.top_fit_count += 1
            self.records.append(
                {
                    "status": metric_audit["status"],
                    "task_gradient_cosine": metric_audit[
                        "task_gradient_cosine"
                    ],
                    "conflict_projection_active": metric_audit[
                        "conflict_projection_active"
                    ],
                    "old_loss_before": metric_audit["old_loss_before"],
                    "old_loss_after": metric_audit["old_loss_after"],
                    "new_loss_before": metric_audit["new_loss_before"],
                    "new_loss_after": metric_audit["new_loss_after"],
                    "delta_l2": metric_audit["delta_l2"],
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
            "d73_additional_component_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d73_additional_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d73_total_added_adaptation_macs"
        ]
        resource["estimated_metric_adaptation_macs"] -= resource[
            "d73_gradient_mac_equivalent_upper_bound"
        ]
        step_count = int(resource["d73_stage2c_step_count"])
        if step_count:
            resource["complete_loss_trace"] = resource["complete_loss_trace"][:-1]
        for field in (
            "adaptation_epochs",
            "metric_epochs",
            "optimizer_steps",
            "total_optimizer_steps",
            "metric_optimizer_steps",
            "stage2c_optimizer_steps",
        ):
            resource[field] -= step_count
        for field in ("before_covariance_audit", "final_covariance_audit"):
            row["geometry_summary"][field]["d43_probe_arm"] = d62.ARM
    return sanitized


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D73ProbeError("D73 training row closure drift")
    d62_evidence = d62._verify_rows(_sanitize_for_d62(rows))
    projection_count = 0
    cosines: list[float] = []
    old_loss_changes: list[float] = []
    new_loss_changes: list[float] = []
    expected_radius = float(np.sqrt(8.0 / (8.0 + 288.0)))
    for row in target:
        geometry = row["geometry_summary"]
        resource = row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d73_probe_arm": ARM,
                "d73_formula": FORMULA,
                "d73_before_state_exact_d62_unchanged": True,
                "d73_metric_state_changed": True,
                "d73_class_id_specific_formula": False,
                "d73_support_role_tasks_equal_priority": True,
                "d73_query_role_specific_branch": False,
                "d73_scene_receiver_handle_specific_branch": False,
                "d73_uses_outer_held_or_query_for_fit": False,
                "d73_query_joint_optimization": False,
                "d73_ground_component_input_count": 0,
                "d73_single_affine_state_only": True,
                "d73_dense_query_graph_bytes": 0,
                "old_log_diag_bitwise_unchanged": False,
                "metric_frozen_during_stage2c": False,
                "stage2c_log_diag_frozen": False,
                "metric_source": "d42_stage2b_plus_d73_one_step_support_only",
                "stage2c_classifier": "d73_conflict_projected_metric_plus_d62_joint_lda",
            }.items()
        ):
            raise D73ProbeError("D73 geometry closure drift")
        if (
            geometry["d73_base_final_log_diag_sha256"]
            == geometry["d73_updated_final_log_diag_sha256"]
            or geometry["old_log_diag_final_sha256"]
            != geometry["d73_updated_final_log_diag_sha256"]
        ):
            raise D73ProbeError("D73 metric lifecycle hash drift")
        audit = geometry["d73_metric_audit"]
        if (
            audit["status"]
            != "one_step_conflict_projected_joint_metric_active"
            or int(audit["k_shot"]) != 8
            or int(audit["dimension"]) != 288
            or int(audit["stage2c_step_count"]) != 1
            or audit["first_order_both_nonincreasing"] is not True
            or abs(float(audit["delta_l2"]) - expected_radius) > 1e-9
            or abs(float(audit["delta_mean"])) > 1e-12
        ):
            raise D73ProbeError("D73 metric audit closure drift")
        if any(
            resource.get(name) != value
            for name, value in {
                "d73_stage2c_step_count": 1,
                "d73_metric_parameter_count": 288,
                "d73_additional_component_fit_count": 36,
                "d73_query_extra_mac_equivalents": 0,
                "d73_persistent_state_extra_bytes": 0,
                "d73_ground_component_input_count": 0,
                "d73_dense_query_graph_bytes": 0,
                "d73_single_affine_state_only": True,
            }.items()
        ):
            raise D73ProbeError("D73 resource closure drift")
        if (
            int(resource["optimizer_steps"]) != 21
            or int(resource["adaptation_epochs"]) != 21
            or int(resource["estimated_metric_adaptation_macs"])
            != int(resource["d73_base_metric_adaptation_macs"])
            + int(resource["d73_gradient_mac_equivalent_upper_bound"])
            or int(resource["persistent_state_bytes"]) > 256 * 1024
            or len(resource["complete_loss_trace"]) != 21
        ):
            raise D73ProbeError("D73 cap/training trace drift")
        projection_count += int(audit["conflict_projection_active"])
        cosines.append(float(audit["task_gradient_cosine"]))
        old_loss_changes.append(
            float(audit["old_loss_after"] - audit["old_loss_before"])
        )
        new_loss_changes.append(
            float(audit["new_loss_after"] - audit["new_loss_before"])
        )
    return {
        **d62_evidence,
        "verified_d73_target_row_count": len(target),
        "verified_d73_metric_audit_count": len(target),
        "verified_d73_conflict_projection_count": projection_count,
        "verified_d73_task_gradient_cosine_min": min(cosines),
        "verified_d73_task_gradient_cosine_max": max(cosines),
        "verified_d73_old_loss_change_max": max(old_loss_changes),
        "verified_d73_new_loss_change_max": max(new_loss_changes),
        "verified_d73_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D73ProbeError("D73 helper source closure drift")
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
    parser.add_argument("--d73-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D73ProbeError(f"D73 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d73_core_sha256": d43._sha256(CORE_PATH),
        "d73_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d73_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d73_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d73_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d73_locked_d42_runner", 1
    registry = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, component_records = build_d73_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = MetricRegistry(d42, fit)
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D73ProbeError("D73 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d73_arm,
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
        or registry.extra_d62_fit_count != 30
        or len(registry.records) != 30
        or len(component_records) != 1620
    ):
        raise D73ProbeError("D73 fit/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(
            registry.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d73.conflict_projected_joint_metric_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d73_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "top_fit_count": registry.top_fit_count,
        "extra_d62_fit_count": registry.extra_d62_fit_count,
        "component_fit_execution_count": len(component_records),
        "fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D73_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
