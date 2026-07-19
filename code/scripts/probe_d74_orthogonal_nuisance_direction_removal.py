#!/usr/bin/env python3
"""D74 support-only class-centroid-orthogonal nuisance-removal probe."""

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
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d74_orthogonal_nuisance_removal.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D74 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d74_d62_probe_helper", D62_HELPER_PATH)
core = _load("d74_orthogonal_nuisance_core", CORE_PATH)
d43 = d62.d43

ARM = "orthogonal_nuisance_direction_removal"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "freeze D62 before state; in all-registered D42 support remove the single "
    "largest within-class residual direction orthogonal to the centered class-mean "
    "span; refit D62 and compile W(I-uuT) into one int8 affine head"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D74ProbeError(RuntimeError):
    pass


def build_d74_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
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
                "d74_probe_arm": ARM,
                "d74_formula": FORMULA,
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


def _projection_mac_upper_bound(
    dimension: int, class_count: int, k_shot: int
) -> int:
    rows = int(class_count) * int(k_shot)
    d = int(dimension)
    classes = int(class_count)
    return int(8 * rows * rows * d + 4 * classes * classes * d + 8 * rows * d)


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


class ProjectionRegistry:
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
            transformed = self.d42._transform(
                all_features, result.state.log_diag_fp32
            )
            direction, projected, projection_audit = (
                core.fit_orthogonal_nuisance_direction(
                    transformed,
                    all_targets,
                    len(all_classes),
                    all_k,
                )
            )
            if projection_audit["projection_active"]:
                fitted_w, fitted_b, fit_audit = self.base_fit(
                    projected,
                    all_targets,
                    len(all_classes),
                    all_k,
                )
                direction64 = np.asarray(direction, dtype=np.float64)
                fitted64 = np.asarray(fitted_w, dtype=np.float64)
                compiled_w = fitted64 - np.outer(
                    fitted64 @ direction64, direction64
                )
                compiled_w = np.asarray(compiled_w, dtype=np.float32)
                compiled_b = np.asarray(fitted_b, dtype=np.float32)
                self.extra_d62_fit_count += 1
                added = _added_refit_resource(
                    self.d42, all_k, len(all_classes)
                )
            else:
                compiled_w = self.d42.decode_d42_coefficients(
                    result.matched_fp32_state
                )
                compiled_b = np.asarray(
                    result.matched_fp32_state.intercept_fp32, dtype=np.float32
                )
                fit_audit = dict(result.geometry_audit["final_covariance_audit"])
                added = {
                    "component_fit_count": 0,
                    "lda_fit_macs": 0,
                    "fisher_dense_macs": 0,
                    "gate_scalar_macs": 0,
                }
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
                np.sum(
                    np.argmax(final_int8_scores, axis=1)
                    != final_prediction
                )
            )
            quant_error = float(
                np.max(np.abs(final_int8_scores - final_fp32_scores))
            )
            projection_audit = dict(projection_audit)
            projection_audit.update(
                {
                    "support_prediction_change_count": int(
                        np.sum(base_prediction != final_prediction)
                    ),
                    "support_accuracy_base": float(
                        np.mean(base_prediction == all_targets)
                    ),
                    "support_accuracy_projected": float(
                        np.mean(final_prediction == all_targets)
                    ),
                    "compiled_direction_coefficient_max_abs": float(
                        np.max(
                            np.abs(
                                np.asarray(compiled_w, dtype=np.float64)
                                @ np.asarray(direction, dtype=np.float64)
                            )
                        )
                    ),
                }
            )
            geometry = dict(result.geometry_audit)
            geometry.update(
                {
                    "final_covariance_audit": fit_audit,
                    "d74_probe_arm": ARM,
                    "d74_formula": FORMULA,
                    "d74_projection_audit": projection_audit,
                    "d74_before_state_exact_d62_unchanged": True,
                    "d74_class_id_specific_formula": False,
                    "d74_old_new_role_specific_branch": False,
                    "d74_query_role_specific_branch": False,
                    "d74_scene_receiver_handle_specific_branch": False,
                    "d74_uses_outer_held_or_query_for_fit": False,
                    "d74_query_joint_optimization": False,
                    "d74_ground_component_input_count": 0,
                    "d74_projection_direction_persisted": False,
                    "d74_projection_compiled_into_affine": True,
                    "d74_single_affine_state_only": True,
                    "d74_dense_query_graph_bytes": 0,
                    "metric_frozen_during_stage2c": True,
                    "stage2c_log_diag_frozen": True,
                    "stage2c_classifier": "d74_projected_d62_joint_lda_compiled_affine",
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
            dimension = int(self.d42.FEATURE_DIM)
            projection_macs = _projection_mac_upper_bound(
                dimension, len(all_classes), all_k
            )
            compile_macs = int(2 * len(all_classes) * dimension)
            added_total = int(
                added["lda_fit_macs"]
                + added["fisher_dense_macs"]
                + added["gate_scalar_macs"]
                + projection_macs
                + compile_macs
            )
            resource = dict(result.resource_audit)
            base_metric_macs = int(resource["estimated_metric_adaptation_macs"])
            resource.update(
                {
                    "d74_projection_removed_rank": int(
                        projection_audit["projection_removed_rank"]
                    ),
                    "d74_additional_component_fit_count": int(
                        added["component_fit_count"]
                    ),
                    "d74_additional_lda_fit_macs": int(added["lda_fit_macs"]),
                    "d74_fisher_dense_mac_upper_bound": int(
                        added["fisher_dense_macs"]
                    ),
                    "d74_gate_scalar_mac_equivalents": int(
                        added["gate_scalar_macs"]
                    ),
                    "d74_projection_mac_equivalent_upper_bound": projection_macs,
                    "d74_affine_compile_mac_equivalents": compile_macs,
                    "d74_base_metric_adaptation_macs": base_metric_macs,
                    "d74_total_added_adaptation_macs": added_total,
                    "d74_query_extra_mac_equivalents": 0,
                    "d74_persistent_state_extra_bytes": 0,
                    "d74_ground_component_input_count": 0,
                    "d74_dense_query_graph_bytes": 0,
                    "d74_single_affine_state_only": True,
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
                resource["estimated_metric_adaptation_macs"]
                + projection_macs
                + compile_macs
            )
            self.top_fit_count += 1
            self.records.append(
                {
                    "status": projection_audit["status"],
                    "centroid_span_rank": projection_audit[
                        "centroid_span_rank"
                    ],
                    "orthogonal_residual_rank": projection_audit[
                        "orthogonal_residual_rank"
                    ],
                    "removed_residual_energy_fraction": projection_audit[
                        "removed_residual_energy_fraction"
                    ],
                    "support_prediction_change_count": projection_audit[
                        "support_prediction_change_count"
                    ],
                    "direction_sha256": projection_audit["direction_sha256"],
                }
            )
            return replace(
                result,
                state=final_int8,
                matched_fp32_state=final_fp32,
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
            "d74_additional_component_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d74_additional_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d74_total_added_adaptation_macs"
        ]
        resource["estimated_metric_adaptation_macs"] -= (
            resource["d74_projection_mac_equivalent_upper_bound"]
            + resource["d74_affine_compile_mac_equivalents"]
        )
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
        raise D74ProbeError("D74 training row closure drift")
    d62_evidence = d62._verify_rows(_sanitize_for_d62(rows))
    unique_directions: set[str] = set()
    support_changes = 0
    removed_fractions: list[float] = []
    for row in target:
        geometry = row["geometry_summary"]
        resource = row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d74_probe_arm": ARM,
                "d74_formula": FORMULA,
                "d74_before_state_exact_d62_unchanged": True,
                "d74_class_id_specific_formula": False,
                "d74_old_new_role_specific_branch": False,
                "d74_query_role_specific_branch": False,
                "d74_scene_receiver_handle_specific_branch": False,
                "d74_uses_outer_held_or_query_for_fit": False,
                "d74_query_joint_optimization": False,
                "d74_ground_component_input_count": 0,
                "d74_projection_direction_persisted": False,
                "d74_projection_compiled_into_affine": True,
                "d74_single_affine_state_only": True,
                "d74_dense_query_graph_bytes": 0,
                "metric_frozen_during_stage2c": True,
                "stage2c_log_diag_frozen": True,
            }.items()
        ):
            raise D74ProbeError("D74 geometry closure drift")
        audit = geometry["d74_projection_audit"]
        if (
            audit["status"] != "rank1_orthogonal_nuisance_removal_active"
            or audit["projection_active"] is not True
            or int(audit["k_shot"]) != 8
            or int(audit["dimension"]) != 288
            or int(audit["projection_removed_rank"]) != 1
            or int(audit["projection_rank"]) != 287
            or float(audit["centroid_direction_max_abs"]) > 1e-9
            or float(audit["projector_idempotence_max_abs_error"]) > 1e-10
            or float(audit["compiled_direction_coefficient_max_abs"]) > 1e-5
        ):
            raise D74ProbeError("D74 projection audit closure drift")
        if any(
            resource.get(name) != value
            for name, value in {
                "d74_projection_removed_rank": 1,
                "d74_additional_component_fit_count": 36,
                "d74_query_extra_mac_equivalents": 0,
                "d74_persistent_state_extra_bytes": 0,
                "d74_ground_component_input_count": 0,
                "d74_dense_query_graph_bytes": 0,
                "d74_single_affine_state_only": True,
            }.items()
        ):
            raise D74ProbeError("D74 resource closure drift")
        if (
            int(resource["optimizer_steps"]) != 20
            or int(resource["adaptation_epochs"]) != 20
            or len(resource["complete_loss_trace"]) != 20
            or int(resource["estimated_metric_adaptation_macs"])
            != int(resource["d74_base_metric_adaptation_macs"])
            + int(resource["d74_projection_mac_equivalent_upper_bound"])
            + int(resource["d74_affine_compile_mac_equivalents"])
            or int(resource["persistent_state_bytes"]) > 256 * 1024
        ):
            raise D74ProbeError("D74 cap/resource drift")
        unique_directions.add(str(audit["direction_sha256"]))
        support_changes += int(audit["support_prediction_change_count"])
        removed_fractions.append(float(audit["removed_residual_energy_fraction"]))
    return {
        **d62_evidence,
        "verified_d74_target_row_count": len(target),
        "verified_d74_projection_audit_count": len(target),
        "verified_d74_unique_direction_count": len(unique_directions),
        "verified_d74_support_prediction_change_count": support_changes,
        "verified_d74_removed_residual_energy_fraction_min": min(
            removed_fractions
        ),
        "verified_d74_removed_residual_energy_fraction_max": max(
            removed_fractions
        ),
        "verified_d74_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D74ProbeError("D74 helper source closure drift")
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
    parser.add_argument("--d74-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D74ProbeError(f"D74 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d74_core_sha256": d43._sha256(CORE_PATH),
        "d74_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d74_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d74_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d74_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d74_locked_d42_runner", 1
    registry = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, component_records = build_d74_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = ProjectionRegistry(d42, fit)
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D74ProbeError("D74 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d74_arm,
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
        raise D74ProbeError("D74 fit/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(
            registry.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d74.orthogonal_nuisance_removal_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d74_arm,
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
    (output / "D74_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
