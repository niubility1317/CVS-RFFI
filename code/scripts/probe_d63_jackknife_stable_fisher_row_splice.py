#!/usr/bin/env python3
"""D63 support-only jackknife-stable Fisher row-splice probe."""

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
D62_HELPER_PATH = SCRIPT_DIR / "probe_d62_crossfitted_fisher_row_splice.py"
SPEC = importlib.util.spec_from_file_location("d63_d62_probe_helper", D62_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D63 could not load D62 helper")
d62 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d62)
d43 = d62.d43


ARM = "jackknife_stable_fisher_row_splice"
STRUCTURE = "d46_base_plus_jackknife_stable_crossfitted_d61_affine_rows"
FORMULA = (
    "aggregate coordinate Pareto strict and every leave-one-fold coordinate Pareto "
    "nondegrading; aggregate plus every leave-one-fold atomic joint; row_c=D61 else D46"
)
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))

# D62 is an immutable sealed helper. Its builder deliberately resolves these
# names dynamically, so this D63-only module namespace can reuse the exact
# component construction while emitting the preregistered D63 arm identity.
d62.ARM = ARM
d62.STRUCTURE = STRUCTURE
d62.FORMULA = FORMULA


class D63ProbeError(RuntimeError):
    pass


def _all_class_pareto(
    candidate_scores: np.ndarray,
    base_positive: np.ndarray,
    base_false_positive: np.ndarray,
    truth: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    positive, false_positive = d62._counts(candidate_scores, truth)
    safe = bool(
        np.all(positive >= base_positive)
        and np.all(false_positive <= base_false_positive)
    )
    return safe, positive, false_positive


def _jackknife_pareto_gate(
    base_scores: np.ndarray, residual_scores: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    base = np.asarray(base_scores, dtype=np.float64)
    residual = np.asarray(residual_scores, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.int64)
    if (
        base.shape != residual.shape
        or base.ndim != 3
        or targets.shape != base.shape[:2]
        or base.shape[0] < 3
        or not np.isfinite(base).all()
        or not np.isfinite(residual).all()
    ):
        raise D63ProbeError("D63 gate evidence drift")
    fold_count, _, class_count = base.shape
    base_positive, base_fp = d62._counts(base, targets)
    coordinate_positive = np.zeros(class_count, dtype=np.int64)
    coordinate_fp = np.zeros(class_count, dtype=np.int64)
    coordinate_hybrids: list[np.ndarray] = []
    for class_index in range(class_count):
        hybrid = base.copy()
        hybrid[:, :, class_index] = residual[:, :, class_index]
        positive, false_positive = d62._counts(hybrid, targets)
        coordinate_positive[class_index] = positive[class_index]
        coordinate_fp[class_index] = false_positive[class_index]
        coordinate_hybrids.append(hybrid)
    aggregate_initial = (
        (coordinate_positive >= base_positive)
        & (coordinate_fp <= base_fp)
        & ((coordinate_positive > base_positive) | (coordinate_fp < base_fp))
    )

    jackknife_coordinate_safe = np.ones((fold_count, class_count), dtype=bool)
    for leave_index in range(fold_count):
        keep = np.arange(fold_count) != leave_index
        subset_base_positive, subset_base_fp = d62._counts(base[keep], targets[keep])
        for class_index, hybrid in enumerate(coordinate_hybrids):
            safe, _, _ = _all_class_pareto(
                hybrid[keep], subset_base_positive, subset_base_fp, targets[keep]
            )
            jackknife_coordinate_safe[leave_index, class_index] = safe
    stable_initial = aggregate_initial & np.all(jackknife_coordinate_safe, axis=0)

    joint = base.copy()
    joint[:, :, stable_initial] = residual[:, :, stable_initial]
    joint_aggregate_safe, joint_positive, joint_fp = _all_class_pareto(
        joint, base_positive, base_fp, targets
    )
    joint_jackknife_safe = np.ones(fold_count, dtype=bool)
    for leave_index in range(fold_count):
        keep = np.arange(fold_count) != leave_index
        subset_base_positive, subset_base_fp = d62._counts(base[keep], targets[keep])
        joint_jackknife_safe[leave_index], _, _ = _all_class_pareto(
            joint[keep], subset_base_positive, subset_base_fp, targets[keep]
        )
    atomic_safe = bool(joint_aggregate_safe and np.all(joint_jackknife_safe))
    final = stable_initial if atomic_safe else np.zeros(class_count, dtype=bool)
    if np.any(final):
        status = "jackknife_stable_fisher_row_splice_active"
    elif np.any(stable_initial):
        status = "jackknife_joint_atomic_exact_d46_fallback"
    elif np.any(aggregate_initial):
        status = "jackknife_coordinate_exact_d46_fallback"
    else:
        status = "aggregate_no_row_exact_d46_fallback"
    return {
        "base_positive": base_positive,
        "base_false_positive": base_fp,
        "coordinate_positive": coordinate_positive,
        "coordinate_false_positive": coordinate_fp,
        "joint_positive": joint_positive,
        "joint_false_positive": joint_fp,
        "aggregate_initial_accept": aggregate_initial,
        "jackknife_coordinate_safe": jackknife_coordinate_safe,
        "initial_accept": stable_initial,
        "final_accept": final,
        "joint_aggregate_safe": joint_aggregate_safe,
        "joint_jackknife_safe": joint_jackknife_safe,
        "atomic_safe": atomic_safe,
        "status": status,
        "exact_fallback": not bool(np.any(final)),
    }


def build_d63_fit(d42: Any) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    def capturing_gate(
        base_scores: np.ndarray, residual_scores: np.ndarray, truth: np.ndarray
    ) -> dict[str, Any]:
        gate = _jackknife_pareto_gate(base_scores, residual_scores, truth)
        captured.append(gate)
        return gate

    d62._pareto_gate = capturing_gate
    helper_fit, call_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        capture_before = len(captured)
        coefficient, intercept, helper_audit = helper_fit(
            rows, labels, class_count, k_shot
        )
        audit = dict(helper_audit)
        if int(k_shot) > 2:
            if len(captured) != capture_before + 1:
                raise D63ProbeError("D63 gate capture drift")
            gate = captured[-1]
            aggregate = gate["aggregate_initial_accept"]
            stable = gate["initial_accept"]
            coordinate_safe = gate["jackknife_coordinate_safe"]
            joint_jackknife = gate["joint_jackknife_safe"]
            joint_aggregate = gate["joint_aggregate_safe"]
        else:
            aggregate = stable = np.zeros(class_count, dtype=bool)
            coordinate_safe = np.zeros((0, class_count), dtype=bool)
            joint_jackknife = np.zeros(0, dtype=bool)
            joint_aggregate = True
        audit.update(
            {
                "d63_probe_arm": ARM,
                "d63_formula": FORMULA,
                "d63_boundary_status": audit["d62_boundary_status"],
                "d63_actual_k": int(k_shot),
                "d63_class_count": int(class_count),
                "d63_class_id_specific_formula": False,
                "d63_old_new_role_specific_branch": False,
                "d63_scene_receiver_handle_specific_branch": False,
                "d63_uses_outer_held_or_query": False,
                "d63_hyperparameter_count": 0,
                "d63_aggregate_initial_accept_mask": aggregate.tolist(),
                "d63_jackknife_coordinate_safe_by_fold": coordinate_safe.tolist(),
                "d63_jackknife_stable_accept_mask": stable.tolist(),
                "d63_final_accept_mask": audit["d62_final_accept_mask"],
                "d63_joint_aggregate_safe": bool(joint_aggregate),
                "d63_joint_jackknife_safe_by_fold": joint_jackknife.tolist(),
                "d63_joint_atomic_safe": audit["d62_joint_atomic_safe"],
                "d63_single_affine_state_only": True,
            }
        )
        return coefficient, intercept, audit

    return fit, call_records


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    helper_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = helper_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_k, new_k = int(resource["old_k_shot"]), int(resource["new_k_shot"])
        old_count, all_count = len(result.before_state.classes), len(result.state.classes)
        jackknife_scalar = int(
            sum(k * k * c * c * 8 for k, c in ((old_k, old_count), (new_k, all_count)))
        )
        resource.update(
            {
                "d63_jackknife_gate_scalar_mac_equivalents": jackknife_scalar,
                "d63_query_extra_macs": 0,
                "d63_persistent_state_extra_bytes": 0,
                "d63_optimizer_steps_extra": 0,
                "d63_resource_single_affine_state_only": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + jackknife_scalar
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(rows))
    for row in sanitized:
        resource = row.get("resource")
        if isinstance(resource, dict) and "d63_jackknife_gate_scalar_mac_equivalents" in resource:
            resource["estimated_adaptation_macs"] -= resource[
                "d63_jackknife_gate_scalar_mac_equivalents"
            ]
    helper_evidence = d62._verify_rows(sanitized)
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    active = aggregate_rows = stable_rows = pruned_rows = atomic = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d63_query_extra_macs", -1)) != 0
            or resource.get("d63_resource_single_affine_state_only") is not True
        ):
            raise D63ProbeError("D63 resource closure drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d63_probe_arm": ARM,
                "d63_formula": FORMULA,
                "d63_actual_k": 8,
                "d63_class_id_specific_formula": False,
                "d63_old_new_role_specific_branch": False,
                "d63_scene_receiver_handle_specific_branch": False,
                "d63_uses_outer_held_or_query": False,
                "d63_hyperparameter_count": 0,
                "d63_single_affine_state_only": True,
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D63ProbeError("D63 exact audit drift")
            aggregate = np.asarray(audit["d63_aggregate_initial_accept_mask"], dtype=bool)
            stable = np.asarray(audit["d63_jackknife_stable_accept_mask"], dtype=bool)
            final = np.asarray(audit["d63_final_accept_mask"], dtype=bool)
            if aggregate.shape != stable.shape or stable.shape != final.shape:
                raise D63ProbeError("D63 accept mask drift")
            if np.any(stable & ~aggregate) or np.any(final & ~stable):
                raise D63ProbeError("D63 monotone gate drift")
            aggregate_rows += int(np.sum(aggregate))
            stable_rows += int(np.sum(stable))
            pruned_rows += int(np.sum(aggregate & ~stable))
            active += int(np.any(final))
            atomic += int(not bool(audit["d63_joint_atomic_safe"]))
    return {
        **helper_evidence,
        "verified_d63_target_row_count": len(target),
        "verified_d63_fit_audit_count": 2 * len(target),
        "verified_d63_active_fit_count": active,
        "verified_d63_aggregate_candidate_row_count": aggregate_rows,
        "verified_d63_jackknife_stable_row_count": stable_rows,
        "verified_d63_jackknife_pruned_row_count": pruned_rows,
        "verified_d63_atomic_fallback_count": atomic,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    try:
        evidence = d43._verify_probe_output(output, ARM, script_sha)
    except d43.D43ProbeError as error:
        if "D43 fit audit missing from" not in str(error):
            raise
        receipt = d43._read_json(output / "RECEIPT.json")
        evidence = {
            "base_runner_receipt_sha256": d43._sha256(output / "RECEIPT.json"),
            "verified_training_row_count": int(receipt["training_log_row_count"]),
            "verified_query_opened": False,
            "verified_forced_nonpromotable": True,
            "d63_generic_probe_guard_verified_through_fit_audit_boundary": True,
        }
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D63ProbeError("D63 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d63-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D63ProbeError(f"D63 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d63_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d63_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d63_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d63_d43_helper_sha256": d43._sha256(d62.d61.d46.d44.D43_HELPER_PATH),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d63_locked_d42_runner", 1
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d63_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D63ProbeError("D63 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d63_arm,
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
    expected_calls = 30 * 36
    if len(call_records) != expected_calls:
        raise D63ProbeError(
            f"D63 component-fit count drift: {len(call_records)} != {expected_calls}"
        )
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d63.jackknife_stable_fisher_row_splice_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d63_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D63_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
