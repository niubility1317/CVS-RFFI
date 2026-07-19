#!/usr/bin/env python3
"""D69 support-only frozen D62 old rows plus joint-D62 new-row probe."""

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
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d69_frozen_d62_append.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D69 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d69_d62_probe_helper", D62_HELPER_PATH)
core = _load("d69_frozen_append_core", CORE_PATH)
d43 = d62.d43

ARM = "frozen_d62_old_append_d62_new"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "fit D62 on Stage2-B and freeze all old affine rows; fit the same D62 on "
    "Stage2-C joint support and append only its new affine rows without calibration"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D69ProbeError(RuntimeError):
    pass


def build_d69_fit(d42: Any) -> tuple[Any, Any, list[dict[str, Any]]]:
    d62_fit, component_records = d62.build_d62_fit(d42)
    lifecycle = core.FrozenD62AppendLifecycle(d62_fit)

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
                "d69_probe_arm": ARM,
                "d69_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, lifecycle, component_records


def _state_old_rows_equal(before: Any, final: Any) -> bool:
    count = len(before.classes)
    if tuple(final.classes[:count]) != tuple(before.classes):
        return False
    for name in (
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "coef_fp32",
        "intercept_fp32",
    ):
        left = np.asarray(getattr(before, name))
        right = np.asarray(getattr(final, name))
        if not np.array_equal(left, right[:count]):
            return False
    return True


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        int8_equal = _state_old_rows_equal(result.before_state, result.state)
        fp32_equal = _state_old_rows_equal(
            result.matched_fp32_before_state, result.matched_fp32_state
        )
        if not int8_equal or not fp32_equal:
            raise D69ProbeError("D69 compiled old-state rows changed during append")
        resource = dict(result.resource_audit)
        resource.update(
            {
                "d69_int8_old_rows_bitwise_unchanged": int8_equal,
                "d69_fp32_old_rows_bitwise_unchanged": fp32_equal,
                "d69_append_row_count": len(result.state.classes)
                - len(result.before_state.classes),
                "d69_query_extra_macs": 0,
                "d69_persistent_state_extra_bytes": 0,
                "d69_optimizer_steps_extra": 0,
                "d69_ground_component_input_count": 0,
                "d69_resource_single_affine_state_only": True,
            }
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
        raise D69ProbeError("D69 training row closure drift")

    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        for field in ("before_covariance_audit", "final_covariance_audit"):
            row["geometry_summary"][field]["d43_probe_arm"] = d62.ARM
    d62_evidence = d62._verify_rows(sanitized)

    phases = {
        "before_covariance_audit": ("stage2b_d62_fit_and_freeze", 6, 0),
        "final_covariance_audit": ("stage2c_append_d62_joint_new_rows", 6, 5),
    }
    for row in target:
        resource = row["resource"]
        if any(
            resource.get(name) != expected
            for name, expected in {
                "d69_int8_old_rows_bitwise_unchanged": True,
                "d69_fp32_old_rows_bitwise_unchanged": True,
                "d69_append_row_count": 5,
                "d69_query_extra_macs": 0,
                "d69_persistent_state_extra_bytes": 0,
                "d69_optimizer_steps_extra": 0,
                "d69_ground_component_input_count": 0,
                "d69_resource_single_affine_state_only": True,
            }.items()
        ):
            raise D69ProbeError("D69 resource/state closure drift")
        before = row["geometry_summary"]["before_covariance_audit"]
        final = row["geometry_summary"]["final_covariance_audit"]
        for field, (phase, old_count, appended_count) in phases.items():
            audit = row["geometry_summary"][field]
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d69_probe_arm": ARM,
                "d69_formula": FORMULA,
                "d69_phase": phase,
                "d69_actual_k": 8,
                "d69_old_class_count": old_count,
                "d69_appended_class_count": appended_count,
                "d69_old_row_fp32_bitwise_unchanged": True,
                "d69_new_row_fp32_matches_joint_d62": True,
                "d69_class_id_specific_formula": False,
                "d69_old_new_role_specific_query_branch": False,
                "d69_scene_receiver_handle_specific_branch": False,
                "d69_uses_outer_held_or_query": False,
                "d69_query_joint_optimization": False,
                "d69_hyperparameter_count": 0,
                "d69_ground_component_input_count": 0,
                "d69_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D69ProbeError(f"D69 exact audit drift in {field}")
        if before["d69_actual_row_sha256"] != before["d69_joint_d62_row_sha256"]:
            raise D69ProbeError("D69 before state is not exact D62")
        if final["d69_before_old_row_sha256"] != before["d69_actual_row_sha256"]:
            raise D69ProbeError("D69 frozen old-row hash drift")
        old_count = int(final["d69_old_class_count"])
        actual_coef = np.asarray(final["d69_actual_coefficient_fp32"], dtype=np.float32)
        actual_bias = np.asarray(final["d69_actual_intercept_fp32"], dtype=np.float32)
        joint_coef = np.asarray(
            final["d69_joint_d62_coefficient_fp32"], dtype=np.float32
        )
        joint_bias = np.asarray(final["d69_joint_d62_intercept_fp32"], dtype=np.float32)
        if not np.array_equal(actual_coef[old_count:], joint_coef[old_count:]) or not np.array_equal(
            actual_bias[old_count:], joint_bias[old_count:]
        ):
            raise D69ProbeError("D69 new rows differ from joint D62")
    return {
        **d62_evidence,
        "verified_d69_target_row_count": len(target),
        "verified_d69_fit_audit_count": 2 * len(target),
        "verified_d69_int8_old_rows_bitwise_unchanged": True,
        "verified_d69_fp32_old_rows_bitwise_unchanged": True,
        "verified_d69_new_rows_match_joint_d62": True,
        "verified_d69_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D69ProbeError("D69 helper source closure drift")
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
    parser.add_argument("--d69-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D69ProbeError(f"D69 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d69_core_sha256": d43._sha256(CORE_PATH),
        "d69_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d69_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d69_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d69_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d69_locked_d42_runner", 1
    lifecycle = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, lifecycle, component_records = build_d69_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D69ProbeError("D69 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d69_arm,
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
    if lifecycle is None or lifecycle.pending or lifecycle.completed_pairs != 30:
        raise D69ProbeError("D69 lifecycle pair closure drift")
    if len(lifecycle.records) != 60 or len(component_records) != 30 * 36:
        raise D69ProbeError("D69 fit execution count drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    lifecycle_sha = hashlib.sha256(
        json.dumps(
            lifecycle.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d69.frozen_d62_old_append_d62_new_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d69_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "lifecycle_fit_record_count": len(lifecycle.records),
        "lifecycle_completed_pair_count": lifecycle.completed_pairs,
        "lifecycle_record_sha256": lifecycle_sha,
        "component_fit_execution_count": len(component_records),
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D69_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
