#!/usr/bin/env python3
"""D70 support-only cross-fitted atomic lifecycle row-replacement probe."""

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
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d70_atomic_lifecycle.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D70 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d70_d62_probe_helper", D62_HELPER_PATH)
core = _load("d70_atomic_lifecycle_core", CORE_PATH)
d43 = d62.d43

ARM = "crossfitted_atomic_lifecycle_row_replacement"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "use D62 final joint as base; accept Stage2-B old rows only through a "
    "twofold support-held coordinate TP/FP gate plus all-class atomic gate"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D70ProbeError(RuntimeError):
    pass


def build_d70_fit(d42: Any) -> tuple[Any, Any, list[dict[str, Any]]]:
    d62_fit, component_records = d62.build_d62_fit(d42)
    lifecycle = core.AtomicLifecycleRowReplacement(d62_fit)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, audit = lifecycle(
            rows, labels, class_count, k_shot
        )
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d70_probe_arm": ARM,
                "d70_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, lifecycle, component_records


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        inner_specs = ((4, 6), (4, 11), (4, 6), (4, 11))
        extra_component_fits = 0
        extra_lda = 0
        for k_shot, class_count in inner_specs:
            fits = 2 * (k_shot + 1)
            extra_component_fits += fits
            extra_lda += 2 * int(
                d42._lda_fit_macs(k_shot * class_count, class_count)
            )
            extra_lda += 2 * k_shot * int(
                d42._lda_fit_macs((k_shot - 1) * class_count, class_count)
            )
        fisher = d62.d61._fisher_dense_macs(dimension, extra_component_fits)
        held_score_macs = int(2 * (4 * 11) * (11 + 6) * dimension)
        gate_scalar = int(2 * (4 * 11) * 11 * 6)
        added = extra_lda + fisher + held_score_macs + gate_scalar
        resource.update(
            {
                "d70_inner_d62_fit_count": 4,
                "d70_inner_component_fit_count": extra_component_fits,
                "d70_inner_lda_fit_macs": extra_lda,
                "d70_inner_fisher_dense_mac_upper_bound": fisher,
                "d70_held_score_macs": held_score_macs,
                "d70_gate_scalar_mac_equivalents": gate_scalar,
                "d70_total_added_adaptation_macs": added,
                "d70_query_extra_macs": 0,
                "d70_persistent_state_extra_bytes": 0,
                "d70_optimizer_steps_extra": 0,
                "d70_ground_component_input_count": 0,
                "d70_resource_single_affine_state_only": True,
            }
        )
        resource["lda_closed_form_fit_count"] = int(
            resource["lda_closed_form_fit_count"] + extra_component_fits
        )
        resource["estimated_lda_fit_macs"] = int(
            resource["estimated_lda_fit_macs"] + extra_lda
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + added
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D70ProbeError("D70 training row closure drift")
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        resource = row["resource"]
        resource["lda_closed_form_fit_count"] -= resource[
            "d70_inner_component_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource["d70_inner_lda_fit_macs"]
        resource["estimated_adaptation_macs"] -= resource[
            "d70_total_added_adaptation_macs"
        ]
        for field in ("before_covariance_audit", "final_covariance_audit"):
            row["geometry_summary"][field]["d43_probe_arm"] = d62.ARM
    d62_evidence = d62._verify_rows(sanitized)

    active = accepted = atomic_fallback = 0
    for row in target:
        resource = row["resource"]
        expected_resource = {
            "d70_inner_d62_fit_count": 4,
            "d70_inner_component_fit_count": 40,
            "d70_query_extra_macs": 0,
            "d70_persistent_state_extra_bytes": 0,
            "d70_optimizer_steps_extra": 0,
            "d70_ground_component_input_count": 0,
            "d70_resource_single_affine_state_only": True,
        }
        if any(
            resource.get(name) != value for name, value in expected_resource.items()
        ):
            raise D70ProbeError("D70 resource closure drift")
        before = row["geometry_summary"]["before_covariance_audit"]
        final = row["geometry_summary"]["final_covariance_audit"]
        common = {
            "d43_probe_arm": ARM,
            "d43_covariance_structure": STRUCTURE,
            "d70_probe_arm": ARM,
            "d70_formula": FORMULA,
            "d70_actual_k": 8,
            "d70_class_id_specific_formula": False,
            "d70_old_new_role_specific_query_branch": False,
            "d70_scene_receiver_handle_specific_branch": False,
            "d70_uses_outer_held_or_query": False,
            "d70_query_joint_optimization": False,
            "d70_hyperparameter_count": 0,
            "d70_ground_component_input_count": 0,
            "d70_single_affine_state_only": True,
            "d70_new_rows_match_joint_d62": True,
        }
        if any(before.get(name) != value for name, value in common.items()) or any(
            final.get(name) != value for name, value in common.items()
        ):
            raise D70ProbeError("D70 common audit drift")
        if (
            before["d70_phase"] != "stage2b_exact_d62_and_freeze_candidate"
            or before["d70_class_count"] != 6
            or before["d70_appended_class_count"] != 0
            or before["d70_partition_audit"] != []
            or before["d70_final_accept_mask"] != [False] * 6
            or before["d70_exact_d62_fallback"] is not True
        ):
            raise D70ProbeError("D70 before audit drift")
        partitions = final["d70_partition_audit"]
        held = [index for partition in partitions for index in partition["held_indices"]]
        mask = np.asarray(final["d70_final_accept_mask"], dtype=bool)
        if (
            final["d70_phase"] != "stage2c_atomic_old_row_replacement"
            or final["d70_class_count"] != 11
            or final["d70_old_class_count"] != 6
            or final["d70_appended_class_count"] != 5
            or len(partitions) != 2
            or sorted(held) != list(range(88))
            or any(partition["train_held_overlap_count"] != 0 for partition in partitions)
            or mask.shape != (6,)
        ):
            raise D70ProbeError("D70 final partition/mask drift")
        base_positive = np.asarray(final["d70_base_positive_by_class"])
        base_fp = np.asarray(final["d70_base_false_positive_by_class"])
        joint_positive = np.asarray(final["d70_joint_positive_by_class"])
        joint_fp = np.asarray(final["d70_joint_false_positive_by_class"])
        if mask.any():
            active += 1
            accepted += int(mask.sum())
            if not np.all(joint_positive >= base_positive) or not np.all(
                joint_fp <= base_fp
            ):
                raise D70ProbeError("D70 active mask is not atomic safe")
        if final["d70_gate_status"] == "joint_atomic_failure_exact_d62_fallback":
            atomic_fallback += 1
        actual_coef = np.asarray(final["d70_actual_coefficient_fp32"], dtype=np.float32)
        actual_bias = np.asarray(final["d70_actual_intercept_fp32"], dtype=np.float32)
        base_coef = np.asarray(final["d70_base_joint_coefficient_fp32"], dtype=np.float32)
        base_bias = np.asarray(final["d70_base_joint_intercept_fp32"], dtype=np.float32)
        if not mask.any() and (
            not np.array_equal(actual_coef, base_coef)
            or not np.array_equal(actual_bias, base_bias)
        ):
            raise D70ProbeError("D70 empty mask is not exact D62")
        if not np.array_equal(actual_coef[6:], base_coef[6:]) or not np.array_equal(
            actual_bias[6:], base_bias[6:]
        ):
            raise D70ProbeError("D70 new rows differ from D62 joint")
    return {
        **d62_evidence,
        "verified_d70_target_row_count": len(target),
        "verified_d70_fit_audit_count": 2 * len(target),
        "verified_d70_active_fit_count": active,
        "verified_d70_accepted_old_row_count": accepted,
        "verified_d70_atomic_fallback_count": atomic_fallback,
        "verified_d70_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D70ProbeError("D70 helper source closure drift")
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
    parser.add_argument("--d70-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D70ProbeError(f"D70 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d70_core_sha256": d43._sha256(CORE_PATH),
        "d70_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d70_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d70_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d70_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d70_locked_d42_runner", 1
    lifecycle = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, lifecycle, component_records = build_d70_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D70ProbeError("D70 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d70_arm,
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
        lifecycle is None
        or lifecycle.pending
        or lifecycle.completed_pairs != 30
        or len(lifecycle.records) != 60
        or lifecycle.inner_fit_count != 120
        or len(component_records) != 2280
    ):
        raise D70ProbeError("D70 lifecycle/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(
            lifecycle.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d70.crossfitted_atomic_lifecycle_row_replacement_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d70_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "lifecycle_fit_record_count": len(lifecycle.records),
        "lifecycle_completed_pair_count": lifecycle.completed_pairs,
        "inner_d62_fit_count": lifecycle.inner_fit_count,
        "component_fit_execution_count": len(component_records),
        "lifecycle_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D70_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
