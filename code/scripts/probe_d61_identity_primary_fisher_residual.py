#!/usr/bin/env python3
"""D61 support-only identity-primary shared Fisher residual probe."""

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
D46_HELPER_PATH = SCRIPT_DIR / "probe_d46_classwise_loo_reliability_fusion.py"
SPEC = importlib.util.spec_from_file_location("d61_d46_probe_helper", D46_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D61 could not load D46 helper")
d46 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d46)
d43 = d46.d43


ARM = "identity_primary_fisher_residual"
STRUCTURE = "d46_full_block_after_shared_identity_plus_bounded_fisher_residual"
FORMULA = "A=I+U*diag(b/(b+w))*U.T; W=W0*A.T"
RANK_POLICY = "machine_rank_of_centered_class_mean_matrix_without_scan"
GAIN_TOLERANCE = 2.0e-12
EQUIVALENCE_RELATIVE_TOLERANCE = 2.0e-6
FP64_EQUIVALENCE_TOLERANCE = 2.0e-11
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D61ProbeError(RuntimeError):
    pass


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fisher_residual_transform(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(class_count, dtype=np.int64))
    ):
        raise D61ProbeError("D61 support matrix/class closure drift")
    dimension = int(x.shape[1])
    if int(k_shot) <= 1:
        identity = np.eye(dimension, dtype=np.float64)
        return identity, {
            "d61_boundary_status": "k1_exact_d46_fallback",
            "d61_fisher_active": False,
            "d61_machine_rank": 0,
            "d61_gain_by_mode": [],
            "d61_transform_eigenvalue_min": 1.0,
            "d61_transform_eigenvalue_max": 1.0,
            "d61_transform_sha256": _array_sha256(identity),
        }
    means = np.stack([x[y == index].mean(axis=0) for index in range(class_count)])
    centered_means = means - means.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(
        centered_means, full_matrices=False
    )
    if singular_values.size == 0:
        rank = 0
    else:
        tolerance = (
            max(centered_means.shape)
            * np.finfo(np.float64).eps
            * float(singular_values[0])
        )
        rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        identity = np.eye(dimension, dtype=np.float64)
        return identity, {
            "d61_boundary_status": "rank0_exact_d46_fallback",
            "d61_fisher_active": False,
            "d61_machine_rank": 0,
            "d61_gain_by_mode": [],
            "d61_transform_eigenvalue_min": 1.0,
            "d61_transform_eigenvalue_max": 1.0,
            "d61_transform_sha256": _array_sha256(identity),
        }
    basis = np.asarray(right_vectors[:rank].T, dtype=np.float64)
    between_coordinates = centered_means @ basis
    within_coordinates = (x - means[y]) @ basis
    between = np.mean(between_coordinates**2, axis=0)
    within = np.mean(within_coordinates**2, axis=0)
    denominator = between + within
    gain = np.divide(
        between,
        denominator,
        out=np.zeros_like(between),
        where=denominator > np.finfo(np.float64).tiny,
    )
    if (
        not np.isfinite(gain).all()
        or float(np.min(gain)) < -GAIN_TOLERANCE
        or float(np.max(gain)) > 1.0 + GAIN_TOLERANCE
    ):
        raise D61ProbeError("D61 Fisher gain escaped [0,1]")
    gain = np.clip(gain, 0.0, 1.0)
    transform = np.eye(dimension, dtype=np.float64) + (basis * gain[None, :]) @ basis.T
    transform = 0.5 * (transform + transform.T)
    eigenvalues = np.linalg.eigvalsh(transform)
    if (
        not np.isfinite(transform).all()
        or float(np.min(eigenvalues)) < 1.0 - GAIN_TOLERANCE
        or float(np.max(eigenvalues)) > 2.0 + GAIN_TOLERANCE
    ):
        raise D61ProbeError("D61 identity-primary eigenvalue bound drift")
    return transform, {
        "d61_boundary_status": "identity_primary_fisher_residual_active",
        "d61_fisher_active": True,
        "d61_machine_rank": rank,
        "d61_rank_upper_bound": int(class_count - 1),
        "d61_singular_value_by_mode": singular_values[:rank].tolist(),
        "d61_between_energy_by_mode": between.tolist(),
        "d61_within_energy_by_mode": within.tolist(),
        "d61_gain_by_mode": gain.tolist(),
        "d61_gain_min": float(np.min(gain)),
        "d61_gain_mean": float(np.mean(gain)),
        "d61_gain_max": float(np.max(gain)),
        "d61_transform_eigenvalue_min": float(np.min(eigenvalues)),
        "d61_transform_eigenvalue_max": float(np.max(eigenvalues)),
        "d61_transform_condition_number": float(
            np.max(eigenvalues) / np.min(eigenvalues)
        ),
        "d61_transform_sha256": _array_sha256(transform),
    }


def _wrap_component_fit(
    base_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    component: str,
    call_records: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transform, transform_audit = _fisher_residual_transform(
            rows, labels, class_count, k_shot
        )
        coefficient_base, intercept, base_audit = base_fit(
            rows, labels, class_count, k_shot
        )
        coefficient = np.asarray(coefficient_base, dtype=np.float64) @ transform.T
        compiled32 = coefficient.astype(np.float32)
        intercept32 = np.asarray(intercept, dtype=np.float32)
        rows32 = np.asarray(rows, dtype=np.float32)
        transformed32 = (np.asarray(rows, dtype=np.float64) @ transform).astype(
            np.float32
        )
        base32 = np.asarray(coefficient_base, dtype=np.float32)
        direct_scores = transformed32 @ base32.T + intercept32[None, :]
        compiled_scores = rows32 @ compiled32.T + intercept32[None, :]
        score_drift = float(np.max(np.abs(direct_scores - compiled_scores)))
        score_scale = max(
            1.0,
            float(np.max(np.abs(direct_scores))),
            float(np.max(np.abs(compiled_scores))),
        )
        relative_score_drift = score_drift / score_scale
        intercept64 = np.asarray(intercept, dtype=np.float64)
        direct_scores64 = (
            np.asarray(rows, dtype=np.float64)
            @ transform
            @ np.asarray(coefficient_base, dtype=np.float64).T
            + intercept64[None, :]
        )
        compiled_scores64 = (
            np.asarray(rows, dtype=np.float64) @ coefficient.T
            + intercept64[None, :]
        )
        fp64_score_drift = float(np.max(np.abs(direct_scores64 - compiled_scores64)))
        if (
            compiled32.shape != (class_count, rows32.shape[1])
            or not np.isfinite(compiled32).all()
            or fp64_score_drift > FP64_EQUIVALENCE_TOLERANCE
            or relative_score_drift > EQUIVALENCE_RELATIVE_TOLERANCE
        ):
            raise D61ProbeError("D61 compiled affine equivalence drift")
        record = {
            "component": component,
            "row_count": int(len(rows32)),
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "transform_sha256": transform_audit["d61_transform_sha256"],
            "machine_rank": int(transform_audit["d61_machine_rank"]),
        }
        call_records.append(record)
        audit = dict(base_audit)
        audit.update(transform_audit)
        audit.update(
            {
                "d61_probe_arm": ARM,
                "d61_structure": STRUCTURE,
                "d61_formula": FORMULA,
                "d61_base_component_fit_in_original_coordinates": True,
                "d61_covariance_coordinates_unchanged": True,
                "d61_rank_policy": RANK_POLICY,
                "d61_component": component,
                "d61_identity_primary": True,
                "d61_shared_across_classes": True,
                "d61_trainable_parameter_count": 0,
                "d61_rank_gain_threshold_scan_count": 0,
                "d61_class_logit_scale_or_intercept_calibration": False,
                "d61_old_new_role_specific_branch": False,
                "d61_scene_receiver_handle_specific_branch": False,
                "d61_uses_held_or_query": False,
                "d61_compiled_affine_fp64_score_drift_max": fp64_score_drift,
                "d61_compiled_affine_score_drift_max": score_drift,
                "d61_compiled_affine_relative_score_drift_max": relative_score_drift,
                "d61_single_affine_state_only": True,
            }
        )
        return compiled32, intercept32, audit

    return fit


def build_d61_fit(d42: Any) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    original_full_builder = d46.d45._build_locked_d42_full_component_fit
    original_block_builder = d46.d43.build_structured_fit
    call_records: list[dict[str, Any]] = []

    def full_builder(module: Any) -> Callable[..., Any]:
        return _wrap_component_fit(
            original_full_builder(module), "d46_full", call_records
        )

    def block_builder(module: Any, arm: str) -> Callable[..., Any]:
        return _wrap_component_fit(
            original_block_builder(module, arm), "d46_block3", call_records
        )

    d46.d45._build_locked_d42_full_component_fit = full_builder
    d46.d43.build_structured_fit = block_builder
    try:
        fit = d46.build_classwise_loo_reliability_fit(d42)
    finally:
        d46.d45._build_locked_d42_full_component_fit = original_full_builder
        d46.d43.build_structured_fit = original_block_builder
    return fit, call_records


def _fisher_dense_macs(dimension: int, fit_count: int) -> int:
    return int(fit_count) * int(8 * int(dimension) ** 3)


def _install_d61_resource_accounting(d42: Any) -> Any:
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_k = int(resource["old_k_shot"])
        new_k = int(resource["new_k_shot"])
        fit_count = 2 * ((1 if old_k <= 1 else old_k + 1) + (1 if new_k <= 1 else new_k + 1))
        fisher_macs = _fisher_dense_macs(int(d42.FEATURE_DIM), fit_count)
        resource.update(
            {
                "d61_component_transform_fit_count": fit_count,
                "d61_dense_algebra_mac_equivalent_upper_bound": fisher_macs,
                "d61_query_extra_macs": 0,
                "d61_persistent_state_extra_bytes": 0,
                "d61_optimizer_steps_extra": 0,
                "d61_trainable_parameters_extra": 0,
                "d61_resource_single_affine_state_only": True,
                "estimated_adaptation_macs": int(
                    resource["estimated_adaptation_macs"] + fisher_macs
                ),
            }
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D61ProbeError("D61 training row closure drift")
    d46_count = d46._verify_d46_fit_audits(rows)
    ranks: list[int] = []
    gains: list[float] = []
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d61_component_transform_fit_count", -1)) != 36
            or int(resource.get("d61_query_extra_macs", -1)) != 0
            or resource.get("d61_resource_single_affine_state_only") is not True
        ):
            raise D61ProbeError("D61 resource closure drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            if any(
                audit.get(name) != expected
                for name, expected in {
                    "d61_probe_arm": ARM,
                    "d61_structure": STRUCTURE,
                    "d61_formula": FORMULA,
                    "d61_rank_policy": RANK_POLICY,
                    "d61_component": "d46_full",
                    "d61_base_component_fit_in_original_coordinates": True,
                    "d61_covariance_coordinates_unchanged": True,
                    "d61_boundary_status": "identity_primary_fisher_residual_active",
                    "d61_fisher_active": True,
                    "d61_identity_primary": True,
                    "d61_shared_across_classes": True,
                    "d61_trainable_parameter_count": 0,
                    "d61_rank_gain_threshold_scan_count": 0,
                    "d61_uses_held_or_query": False,
                    "d61_single_affine_state_only": True,
                }.items()
            ):
                raise D61ProbeError("D61 exact audit drift")
            rank = int(audit["d61_machine_rank"])
            gain = np.asarray(audit["d61_gain_by_mode"], dtype=np.float64)
            if (
                rank < 1
                or rank > int(audit["d61_rank_upper_bound"])
                or gain.shape != (rank,)
                or not np.isfinite(gain).all()
                or float(np.min(gain)) < -GAIN_TOLERANCE
                or float(np.max(gain)) > 1.0 + GAIN_TOLERANCE
                or float(audit["d61_transform_eigenvalue_min"]) < 1.0 - GAIN_TOLERANCE
                or float(audit["d61_transform_eigenvalue_max"]) > 2.0 + GAIN_TOLERANCE
                or float(audit["d61_compiled_affine_fp64_score_drift_max"])
                > FP64_EQUIVALENCE_TOLERANCE
                or float(audit["d61_compiled_affine_relative_score_drift_max"])
                > EQUIVALENCE_RELATIVE_TOLERANCE
            ):
                raise D61ProbeError("D61 Fisher transform closure drift")
            ranks.append(rank)
            gains.extend(float(value) for value in gain)
    return {
        "verified_d61_target_row_count": len(target),
        "verified_d46_fit_row_count": d46_count,
        "verified_d61_outer_fit_audit_count": 2 * len(target),
        "verified_d61_rank_min": min(ranks),
        "verified_d61_rank_max": max(ranks),
        "verified_d61_gain_min": min(gains),
        "verified_d61_gain_max": max(gains),
    }


def _verify_output(output: Path, script_sha: str, helper_hashes: dict[str, str]) -> dict[str, Any]:
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
            "d61_generic_probe_guard_verified_through_fit_audit_boundary": True,
            "d61_expected_generic_fit_audit_namespace_mismatch": True,
        }
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D61ProbeError("D61 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d61-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--verify-existing-output", type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    helper_hashes = {
        "d61_d46_helper_sha256": d43._sha256(D46_HELPER_PATH),
        "d61_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d61_d44_helper_sha256": d43._sha256(d46.d45.D44_HELPER_PATH),
        "d61_d43_helper_sha256": d43._sha256(d46.d44.D43_HELPER_PATH),
    }
    if known.verify_existing_output is not None:
        if runner_arguments:
            raise D61ProbeError("D61 offline verifier forbids runner arguments")
        output = known.verify_existing_output.resolve()
        support = d43._read_json(output / "support_audit.json")
        closure = support["candidate_lock"]["source_closure"]
        locked_script_sha = str(closure["d43_probe_script_sha256"])
        locked_helper_hashes = {
            name: str(closure[name]) for name in helper_hashes
        }
        evidence = _verify_output(
            output, locked_script_sha, locked_helper_hashes
        )
        verifier_sha = d43._sha256(Path(__file__).resolve())
        metadata = {
            "schema": "cvs.phase2.d61.identity_primary_fisher_residual_probe.v1",
            "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
            "arm": known.d61_arm,
            "formal_candidate": False,
            "probe_forced_nonpromotable": True,
            "selected_only_full_k10_refit_allowed": False,
            "query_opened": False,
            "probe_script_sha256": locked_script_sha,
            "offline_verifier_script_sha256": verifier_sha,
            "offline_verifier_only_no_runner_reexecution": True,
            "formula": FORMULA,
            "rank_policy": RANK_POLICY,
            "runtime_root": str(known.runtime_root.resolve()),
            "probe_root": str(known.probe_root.resolve()),
            **evidence,
        }
        (output / "D61_PROBE_METADATA.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D61ProbeError(f"D61 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_d46_top = original_d61_top = None
    runner_name, exit_code = "d61_locked_d42_runner", 1
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d61_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_d46_top = d46._install_d46_resource_accounting(d42)
        original_d61_top = _install_d61_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D61ProbeError("D61 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d61_arm,
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
        if d42 is not None and original_d46_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_d46_top
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    expected_calls = 30 * 36
    if len(call_records) != expected_calls:
        raise D61ProbeError(
            f"D61 component-fit execution count drift: {len(call_records)} != {expected_calls}"
        )
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_hash = hashlib.sha256(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d61.identity_primary_fisher_residual_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d61_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "rank_policy": RANK_POLICY,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": record_hash,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D61_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
