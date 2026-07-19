#!/usr/bin/env python3
"""D78 support-only ground-tangent smooth-worst top-2 margin probe."""

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
D77_HELPER_PATH = SCRIPT_DIR / "probe_d77_ground_preconditioned_allclass_common_descent.py"
CORE_PATH = (
    SCRIPT_DIR.parent
    / "cvsrffi"
    / "stage2_d78_ground_tangent_worstclass_margin.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D78 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d77 = _load("d78_d77_integration_scaffold", D77_HELPER_PATH)
core = _load("d78_ground_tangent_core", CORE_PATH)
d66, d62, d43 = d77.d66, d77.d62, d77.d43

ARM = "ground_tangent_worstclass_top2_margin"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "class-center immutable int8 ground domain-class prototypes; take the first "
    "min(domain_count-1,numerical_rank) right-singular directions as a shared "
    "domain tangent basis; obtain fixed top rivals from physical-rank crossfit "
    "target-support LDA; minimize a fixed-temperature smooth maximum of equal-class "
    "top-2 logistic losses for 20 accepted steps in the tangent subspace; center "
    "class rows, apply one class-agnostic Frobenius trust ball, and compile the "
    "residual directly into frozen D62 final rows"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D78ProbeError(RuntimeError):
    """Raised when the D78 probe violates its locked design."""


def load_ground_tangent(
    component_dir: Path,
    manifest_sha256: str,
    feature_dim: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Reuse D66 validation, then derive the D78 basis from the same sealed bytes."""

    _, ground_audit = d66.load_ground_domain_reliability(
        component_dir, manifest_sha256, feature_dim
    )
    npz_path = component_dir / d66.NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as payload:
        q = np.asarray(payload["domain_class_q"], dtype=np.int8)
        scales = np.asarray(payload["domain_class_scale"], dtype=np.float16)
        mask = np.asarray(payload["domain_class_mask"], dtype=np.uint8)
        domains = np.asarray(payload["domain_registry"], dtype=np.int16)
        classes = np.asarray(payload["class_registry"]).astype(str)
    domain_order = np.argsort(domains, kind="stable")
    class_order = np.argsort(classes, kind="stable")
    q = q[domain_order][:, class_order]
    scales = scales[domain_order][:, class_order]
    mask = mask[domain_order][:, class_order]
    prototypes = q.astype(np.float64) * scales.astype(np.float64)[..., None]
    basis, tangent_audit = core.ground_domain_tangent_basis(
        prototypes, mask, feature_dim=int(feature_dim)
    )
    domains_n, classes_n, z_dim = prototypes.shape
    tangent_rank = int(basis.shape[1])
    statistics_macs = int(
        2 * domains_n * classes_n * z_dim
        + 4 * domains_n * classes_n * z_dim * z_dim
        + 20 * z_dim * z_dim * z_dim
    )
    ground = dict(ground_audit)
    ground["ground_reliability_statistics_scalar_mac_equivalents"] = statistics_macs
    ground["ground_tangent_statistics_mac_upper_bound"] = statistics_macs
    ground["ground_tangent_rank"] = tangent_rank
    return basis, tangent_audit, ground


def _resource_upper_bounds(
    *,
    k_shot: int,
    class_count: int,
    dimension: int,
    lda_macs: int,
    ground_statistics_macs: int,
) -> dict[str, int]:
    held = int(k_shot) * int(class_count)
    rank = 13
    projection = int(2 * held * dimension * rank)
    top2_setup = int(held * class_count + 4 * held)
    optimize = int(
        core.OPTIMIZER_STEPS
        * (12 * held * class_count * rank + 8 * class_count * rank)
    )
    ce_audit = int(4 * held * class_count)
    compile_macs = int(2 * class_count * rank * dimension)
    non_lda = int(
        ground_statistics_macs
        + projection
        + top2_setup
        + optimize
        + ce_audit
        + compile_macs
    )
    return {
        # Compatibility names consumed by the inherited integration scaffold.
        "crossfit_lda_fit_macs": int(lda_macs),
        "oof_gradient_mac_upper_bound": top2_setup,
        "frank_wolfe_mac_upper_bound": optimize,
        "oof_ce_audit_mac_upper_bound": ce_audit,
        "preconditioner_application_macs": projection,
        "affine_compile_mac_equivalents": compile_macs,
        "ground_statistics_macs": int(ground_statistics_macs),
        "non_lda_total": non_lda,
        "total_added": int(lda_macs + non_lda),
    }


# The inherited registry reads these globals from the D77 module at call time.
d77.core = core
d77.ARM = ARM
d77.FORMULA = FORMULA
d77._resource_upper_bounds = _resource_upper_bounds


class GroundTangentRegistry(d77.GroundCommonDescentRegistry):
    """Reuse D77's tested D62/quantization scaffold and relabel D78 evidence."""

    def wrap_top(self, base_top: Any) -> Any:
        inherited = super().wrap_top(base_top)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = inherited(*args, **kwargs)
            geometry = dict(result.geometry_audit)
            for key in list(geometry):
                if key.startswith("d77_"):
                    geometry["d78_" + key[4:]] = geometry.pop(key)
            geometry["d78_ground_tangent_audit"] = geometry.pop(
                "d78_ground_preconditioner_audit"
            )
            margin_audit = dict(geometry.pop("d78_common_descent_audit"))
            margin_audit["ground_tangent_projector_sha256"] = margin_audit.pop(
                "ground_preconditioner_sha256"
            )
            geometry["d78_worstclass_margin_audit"] = margin_audit
            geometry["stage2c_classifier"] = (
                "d78_ground_tangent_worstclass_margin_compiled_affine"
                if margin_audit["residual_active"]
                else "d78_zero_projected_gradient_exact_d62_fallback"
            )

            trace = [dict(item) for item in result.training_trace]
            for item in trace:
                if item.get("phase") == (
                    "stage2c_ground_preconditioned_common_descent_frank_wolfe"
                ):
                    item["phase"] = "stage2c_ground_tangent_worstclass_top2_margin"
                    item["tangent_iteration"] = item.pop("fw_iteration")

            resource = dict(result.resource_audit)
            for key in list(resource):
                if key.startswith("d77_"):
                    resource["d78_" + key[4:]] = resource.pop(key)
            resource["d78_tangent_projection_macs"] = resource.pop(
                "d78_preconditioner_application_macs"
            )
            resource["d78_worstclass_optimizer_mac_upper_bound"] = resource.pop(
                "d78_frank_wolfe_mac_upper_bound"
            )
            resource["d78_top2_setup_mac_upper_bound"] = resource.pop(
                "d78_oof_gradient_mac_upper_bound"
            )
            resource["d78_top2_ce_audit_mac_upper_bound"] = resource.pop(
                "d78_oof_ce_audit_mac_upper_bound"
            )
            resource["d78_optimizer_steps"] = resource.pop(
                "d78_frank_wolfe_optimizer_steps"
            )
            resource.pop("d78_transient_simplex_parameter_count")
            classes = int(margin_audit["class_count"])
            rank = int(margin_audit["tangent_rank"])
            tangent_parameters = classes * rank
            resource["d78_transient_tangent_parameter_count"] = tangent_parameters
            resource["trainable_parameters"] = int(
                resource["trainable_parameters"] + tangent_parameters - classes
            )
            resource["trainable_parameter_cap_pass"] = bool(
                resource["trainable_parameters"]
                <= resource["trainable_parameter_cap"]
            )
            resource["complete_loss_trace"] = trace
            return replace(
                result,
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
        if "d78_ground_component_logical_state_bytes" not in resource:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d78_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d78_compiled_affine_state_bytes": head_bytes,
                "d78_component_inclusive_persistent_state_bytes": total_bytes,
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
            "d78_crossfit_lda_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d78_crossfit_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d78_total_added_adaptation_macs"
        ]
        resource["estimated_metric_adaptation_macs"] -= resource[
            "d78_non_lda_added_adaptation_macs"
        ]
        steps = int(resource["d78_optimizer_steps"])
        resource["optimizer_steps"] -= steps
        resource["total_optimizer_steps"] -= steps
        resource["stage2c_optimizer_steps"] -= steps
        parameters = int(resource["d78_transient_tangent_parameter_count"])
        resource["trainable_parameters"] -= parameters
        resource["peak_trainable_parameters"] -= parameters
        resource["complete_loss_trace"] = resource["complete_loss_trace"][:-steps]
        resource["persistent_state_bytes"] = resource[
            "d78_compiled_affine_state_bytes"
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
        raise D78ProbeError("D78 training row closure drift")
    d62_evidence = d62._verify_rows(_sanitize_for_d62(rows))
    active = fallback = changed = 0
    residual_hashes: set[str] = set()
    objective_deltas: list[float] = []
    margin_count_deltas: list[int] = []
    for row in target:
        geometry, resource = row["geometry_summary"], row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d78_probe_arm": ARM,
                "d78_formula": FORMULA,
                "d78_before_state_exact_d62_unchanged": True,
                "d78_class_id_specific_formula": False,
                "d78_old_new_role_specific_branch": False,
                "d78_query_role_specific_branch": False,
                "d78_scene_receiver_handle_specific_branch": False,
                "d78_uses_outer_held_or_query_for_fit": False,
                "d78_query_joint_optimization": False,
                "d78_ground_component_input_count": 84,
                "d78_ground_class_score_access": False,
                "d78_ground_component_update_access": False,
                "d78_residual_persisted_separately": False,
                "d78_single_affine_state_only": True,
                "d78_dense_query_graph_bytes": 0,
            }.items()
        ):
            raise D78ProbeError("D78 geometry closure drift")
        audit = geometry["d78_worstclass_margin_audit"]
        tangent = geometry["d78_ground_tangent_audit"]
        is_active = bool(audit["residual_active"])
        active += int(is_active)
        fallback += int(not is_active)
        changed += int(audit["support_prediction_change_count"] > 0)
        if (
            int(audit["class_count"]) != 11
            or int(audit["k_shot"]) != 8
            or int(audit["dimension"]) != 288
            or int(audit["tangent_rank"]) != 13
            or int(audit["crossfit_fold_count"]) != 8
            or int(audit["crossfit_lda_fit_count"]) != 8
            or int(audit["crossfit_held_row_count"]) != 88
            or int(audit["optimizer_iterations"]) != core.OPTIMIZER_STEPS
            or len(audit["optimizer_objective_trace"]) != core.OPTIMIZER_STEPS
            or float(audit["objective_delta"]) > 1e-10
            or int(audit["query_rows_used"]) != 0
            or audit["ground_class_score_access"] is not False
            or audit["ground_component_formal_phase2_eligible"] is not False
            or audit["ground_component_input_count"] != 84
        ):
            raise D78ProbeError("D78 margin audit closure drift")
        if any(
            float(item["objective_after"]) > float(item["objective_before"]) + 1e-10
            for item in audit["optimizer_objective_trace"]
        ):
            raise D78ProbeError("D78 optimizer monotonicity drift")
        if (
            int(tangent["z_dimension"]) != 160
            or int(tangent["feature_dimension"]) != 288
            or int(tangent["ground_component_input_count"]) != 84
            or int(tangent["tangent_rank"]) != 13
            or tangent["rank_rule"]
            != "min(ground_domain_count_minus_one,numerical_rank)"
            or tangent["ground_class_score_access"] is not False
        ):
            raise D78ProbeError("D78 tangent closure drift")
        if is_active:
            if audit["status"] != "ground_tangent_worstclass_top2_margin_active":
                raise D78ProbeError("D78 active status drift")
        elif audit["status"] != "zero_projected_gradient_exact_d62_fallback":
            raise D78ProbeError("D78 fallback status drift")
        if any(
            resource.get(name) != value
            for name, value in {
                "d78_crossfit_fold_count": 8,
                "d78_crossfit_held_row_count": 88,
                "d78_crossfit_lda_fit_count": 8,
                "d78_optimizer_steps": 20,
                "d78_transient_tangent_parameter_count": 143,
                "d78_query_extra_mac_equivalents": 0,
                "d78_query_extra_state_bytes": 0,
                "d78_ground_component_input_count": 84,
                "d78_ground_component_update_access": False,
                "d78_ground_class_score_access": False,
                "d78_ground_component_logical_state_bytes": 25428,
                "d78_dense_query_graph_bytes": 0,
                "d78_single_affine_state_only": True,
                "ground_int8_component_input_count": 84,
                "ground_int8_update_access": False,
            }.items()
        ):
            raise D78ProbeError("D78 resource closure drift")
        if (
            int(resource["optimizer_steps"]) != 40
            or int(resource["total_optimizer_steps"]) != 40
            or int(resource["stage2c_optimizer_steps"]) != 20
            or int(resource["adaptation_epochs"]) != 20
            or len(resource["complete_loss_trace"]) != 40
            or int(resource["persistent_state_bytes"])
            != int(resource["d78_compiled_affine_state_bytes"])
            + int(resource["d78_ground_component_logical_state_bytes"])
            or int(resource["persistent_state_bytes"]) > 256 * 1024
            or int(resource["peak_trainable_parameters"]) > 80000
        ):
            raise D78ProbeError("D78 cap/resource drift")
        residual_hashes.add(str(audit["residual_sha256"]))
        objective_deltas.append(float(audit["objective_delta"]))
        margin_count_deltas.append(
            int(audit["nonpositive_margin_count_after"])
            - int(audit["nonpositive_margin_count_before"])
        )
    return {
        **d62_evidence,
        "verified_d78_target_row_count": len(target),
        "verified_d78_margin_audit_count": len(target),
        "verified_d78_active_count": active,
        "verified_d78_fallback_count": fallback,
        "verified_d78_support_prediction_changed_row_count": changed,
        "verified_d78_unique_residual_count": len(residual_hashes),
        "verified_d78_objective_delta_min": min(objective_deltas),
        "verified_d78_objective_delta_max": max(objective_deltas),
        "verified_d78_nonpositive_margin_count_delta_min": min(
            margin_count_deltas
        ),
        "verified_d78_nonpositive_margin_count_delta_max": max(
            margin_count_deltas
        ),
        "verified_d78_ground_component_input_count": 84,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D78ProbeError("D78 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d78-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-component-dir", required=True, type=Path)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D78ProbeError(f"D78 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d78_core_sha256": d43._sha256(CORE_PATH),
        "d78_d77_scaffold_sha256": d43._sha256(D77_HELPER_PATH),
        "d78_d66_helper_sha256": d43._sha256(d77.D66_HELPER_PATH),
        "d78_d62_helper_sha256": d43._sha256(d66.D62_HELPER_PATH),
        "d78_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d78_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d78_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    component_dir = known.ground_component_dir.resolve()
    component_npz = component_dir / d66.NPZ_NAME
    component_manifest = component_dir / d66.MANIFEST_NAME
    entry_npz_sha = d43._sha256(component_npz)
    entry_manifest_sha = d43._sha256(component_manifest)
    tangent_basis, tangent_audit, ground_audit = load_ground_tangent(
        component_dir, known.ground_manifest_sha256, 288
    )
    if (
        ground_audit["ground_active_domain_class_cells"] != 84
        or ground_audit["ground_tangent_rank"] != 13
        or ground_audit["component_formal_phase2_eligible"] is not False
        or ground_audit["component_provenance_status"]
        != "UNVERIFIED_UNDER_CURRENT_PROTOCOL"
    ):
        raise D78ProbeError("D78 locked diagnostic ground component drift")
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = registry = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d78_locked_d42_runner", 1
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit, original_macs = d42._fit_equal_prior_lda, d42._lda_fit_macs
        fit, component_records = d77.build_d77_fit(d42)
        d42._fit_equal_prior_lda = fit
        _, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = GroundTangentRegistry(
            d42,
            original_fit,
            original_macs,
            tangent_basis,
            tangent_audit,
            ground_audit,
        )
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D78ProbeError("D78 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        _install_runner_resource_accounting(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d78_arm,
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
        raise D78ProbeError("D78 fit/component/read-only closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(registry.records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d78.ground_tangent_worstclass_margin_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d78_arm,
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
        "ground_tangent_audit": tangent_audit,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D78_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
