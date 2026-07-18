#!/usr/bin/env python3
"""D60 support-only cross-block spectral stability shrinkage probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


SCRIPT_DIR = Path(__file__).resolve().parent
D59_HELPER_PATH = SCRIPT_DIR / "probe_d59_full_block_spd_geodesic_midpoint.py"
SPEC = importlib.util.spec_from_file_location("d60_d59_probe_helper", D59_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D60 could not load D59 helper")
d59 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d59)
d43 = d59.d43


ARM = "crossblock_spectral_stability_shrinkage"
STRUCTURE = "block3_plus_rankwise_stability_contracted_crossblock_spectrum"
FORMULA = "s_j=mean(q_rj)^2/mean(q_rj^2); A*=I+V*diag(s*lambda)*V.T"
PARTITION = "per_class_support_row_rank_leave_one_out_exact_once"
STABILITY_TOLERANCE = 3.0e-12
SPD_TOLERANCE = 1.0e-12
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D60ProbeError(RuntimeError):
    pass


def _auto_covariance(
    rows: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    priors = np.full(int(class_count), 1.0 / int(class_count), dtype=np.float64)
    estimator = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
    )
    estimator.fit(np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64))
    if not np.array_equal(estimator.classes_, np.arange(class_count, dtype=np.int64)):
        raise D60ProbeError("D60 inner covariance class order drift")
    covariance = d59._symmetric(np.asarray(estimator.covariance_, dtype=np.float64))
    d59._eigh_spd(covariance, "d60_inner_full")
    return covariance


def _rankwise_partitions(
    labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[list[np.ndarray], dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int64)
    indices = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(item) != int(k_shot) for item in indices):
        raise D60ProbeError("D60 requires exact symmetric K support")
    held_by_fold: list[np.ndarray] = []
    train_by_fold: list[list[int]] = []
    all_indices = set(range(len(y)))
    for rank in range(int(k_shot)):
        held = np.asarray([item[rank] for item in indices], dtype=np.int64)
        held_by_fold.append(held)
        train_by_fold.append(sorted(all_indices - set(int(value) for value in held)))
    flat = [int(value) for held in held_by_fold for value in held]
    if sorted(flat) != list(range(len(y))) or len(set(flat)) != len(y):
        raise D60ProbeError("D60 inner held exact-once closure drift")
    return held_by_fold, {
        "partition_unit": PARTITION,
        "held_support_row_indices_by_fold": [held.tolist() for held in held_by_fold],
        "train_support_row_indices_by_fold": train_by_fold,
        "held_support_row_exact_once_coverage": True,
        "train_held_overlap_count": 0,
        "held_rows_per_fold": int(class_count),
        "train_rows_per_fold": int(len(y) - class_count),
        "fold_count": int(k_shot),
    }


def _stability_contracted_covariance(
    full_covariance: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    block_slices: tuple[slice, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    full = d59._symmetric(full_covariance)
    block = d59._three_block_covariance(full, block_slices)
    if int(k_shot) <= 2:
        midpoint, midpoint_audit = d59._spd_geometric_midpoint(block, full)
        return midpoint, {
            "d60_boundary_status": "k2_exact_d59_midpoint_fallback",
            "d60_stability_active": False,
            "d60_inner_partition": None,
            "d60_stability_by_mode": None,
            "d60_full_crossblock_eigenvalue_by_mode": None,
            "d60_fold_rayleigh_sha256": None,
            "d60_contracted_normalized_eigenvalue_min": None,
            "d60_contracted_normalized_eigenvalue_max": None,
            "d60_midpoint_fallback_audit": midpoint_audit,
        }
    block_values, block_vectors = d59._eigh_spd(block, "d60_block")
    block_half = d59._matrix_power_from_eigh(block_values, block_vectors, 0.5)
    block_inverse_half = d59._matrix_power_from_eigh(
        block_values, block_vectors, -0.5
    )
    normalized_full = d59._symmetric(block_inverse_half @ full @ block_inverse_half)
    full_normalized_values, _ = d59._eigh_spd(normalized_full, "d60_normalized_full")
    crossblock = d59._symmetric(normalized_full - np.eye(full.shape[0]))
    cross_values, cross_vectors = np.linalg.eigh(crossblock)
    if not np.isfinite(cross_values).all() or not np.isfinite(cross_vectors).all():
        raise D60ProbeError("D60 cross-block eigensystem drift")
    held_by_fold, partition = _rankwise_partitions(labels, class_count, k_shot)
    fold_rayleigh: list[np.ndarray] = []
    all_mask = np.ones(len(rows), dtype=bool)
    for held in held_by_fold:
        mask = all_mask.copy()
        mask[held] = False
        fold_full = _auto_covariance(rows[mask], labels[mask], class_count)
        fold_block = d59._three_block_covariance(fold_full, block_slices)
        fold_cross = d59._symmetric(
            block_inverse_half @ (fold_full - fold_block) @ block_inverse_half
        )
        rayleigh = np.einsum(
            "ij,ij->j", cross_vectors, fold_cross @ cross_vectors, optimize=True
        )
        fold_rayleigh.append(np.asarray(rayleigh, dtype=np.float64))
    q = np.stack(fold_rayleigh, axis=0)
    q_mean = np.mean(q, axis=0)
    q_mean_square = np.mean(q**2, axis=0)
    stability = np.zeros_like(q_mean)
    nonzero = q_mean_square > d59.SPD_EPSILON
    stability[nonzero] = q_mean[nonzero] ** 2 / q_mean_square[nonzero]
    if (
        not np.isfinite(stability).all()
        or float(np.min(stability)) < -STABILITY_TOLERANCE
        or float(np.max(stability)) > 1.0 + STABILITY_TOLERANCE
    ):
        raise D60ProbeError("D60 spectral stability escaped [0,1]")
    stability = np.clip(stability, 0.0, 1.0)
    contracted_values = 1.0 + stability * cross_values
    if float(np.min(contracted_values)) <= SPD_TOLERANCE:
        raise D60ProbeError("D60 contracted normalized covariance is not SPD")
    contracted_normalized = d59._symmetric(
        (cross_vectors * contracted_values[None, :]) @ cross_vectors.T
    )
    covariance = d59._symmetric(block_half @ contracted_normalized @ block_half)
    covariance_values, _ = d59._eigh_spd(covariance, "d60_contracted")
    q_hash = hashlib.sha256(
        np.ascontiguousarray(q, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    return covariance, {
        "d60_boundary_status": "crossblock_spectral_stability_shrinkage_active",
        "d60_stability_active": True,
        "d60_inner_partition": partition,
        "d60_stability_by_mode": stability.tolist(),
        "d60_full_crossblock_eigenvalue_by_mode": cross_values.tolist(),
        "d60_fold_rayleigh_sha256": q_hash,
        "d60_fold_rayleigh_shape": list(q.shape),
        "d60_fold_rayleigh_mean_abs": float(np.mean(np.abs(q))),
        "d60_fold_rayleigh_mean_square": float(np.mean(q**2)),
        "d60_stability_min": float(np.min(stability)),
        "d60_stability_mean": float(np.mean(stability)),
        "d60_stability_max": float(np.max(stability)),
        "d60_stability_zero_count": int(np.sum(stability <= 1.0e-15)),
        "d60_stability_near_one_count": int(np.sum(stability >= 1.0 - 1.0e-12)),
        "d60_full_normalized_eigenvalue_min": float(np.min(full_normalized_values)),
        "d60_full_normalized_eigenvalue_max": float(np.max(full_normalized_values)),
        "d60_contracted_normalized_eigenvalue_min": float(np.min(contracted_values)),
        "d60_contracted_normalized_eigenvalue_max": float(np.max(contracted_values)),
        "d60_contracted_covariance_eigenvalue_min": float(np.min(covariance_values)),
        "d60_contracted_covariance_eigenvalue_max": float(np.max(covariance_values)),
        "d60_contracted_covariance_condition_number": float(
            np.max(covariance_values) / np.min(covariance_values)
        ),
        "d60_full_covariance_sha256": d59._array_sha256(full),
        "d60_block_covariance_sha256": d59._array_sha256(block),
        "d60_contracted_covariance_sha256": d59._array_sha256(covariance),
    }


def build_d60_fit(d42: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    original_structured = d43._structured_covariance
    context: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    captured: list[dict[str, Any]] = []

    def structure(
        covariance: np.ndarray, arm: str, block_slices: tuple[slice, ...]
    ) -> np.ndarray:
        if arm != ARM:
            return original_structured(covariance, arm, block_slices)
        if len(context) != 1:
            raise D60ProbeError("D60 fit context cardinality drift")
        rows, labels, class_count, k_shot = context[0]
        result, evidence = _stability_contracted_covariance(
            covariance, rows, labels, class_count, k_shot, block_slices
        )
        captured.append(evidence)
        return result

    d43._structured_covariance = structure
    base_fit = d43.build_structured_fit(d42, ARM)

    def fit(
        transformed: np.ndarray, targets: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if context:
            raise D60ProbeError("D60 nested fit context is forbidden")
        before = len(captured)
        context.append(
            (
                np.asarray(transformed, dtype=np.float64),
                np.asarray(targets, dtype=np.int64),
                int(class_count),
                int(k_shot),
            )
        )
        try:
            coefficient, intercept, base_audit = base_fit(
                transformed, targets, class_count, k_shot
            )
        finally:
            context.clear()
        fallback = bool(base_audit.get("unit_covariance_fallback"))
        if fallback:
            if len(captured) != before:
                raise D60ProbeError("D60 K1 fallback unexpectedly built covariance")
            evidence = {
                "d60_boundary_status": "k1_rank0_or_zero_energy_exact_d42_fallback",
                "d60_stability_active": False,
                "d60_inner_partition": None,
            }
        else:
            if len(captured) != before + 1:
                raise D60ProbeError("D60 captured evidence cardinality drift")
            evidence = dict(captured[-1])
        audit = dict(base_audit)
        audit.update(evidence)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d60_probe_arm": ARM,
                "d60_formula": FORMULA,
                "d60_partition_policy": PARTITION,
                "d60_class_shared_covariance": True,
                "d60_class_logit_scale_or_intercept_calibration": False,
                "d60_class_id_specific_formula": False,
                "d60_old_new_role_specific_branch": False,
                "d60_scene_receiver_handle_specific_branch": False,
                "d60_uses_outer_held_or_query": False,
                "d60_threshold_or_rank_scan_count": 0,
                "d60_single_affine_state_only": True,
            }
        )
        return coefficient, intercept, audit

    return fit, original_structured


def _spectral_macs(dimension: int) -> int:
    return int(28 * int(dimension) ** 3)


def _install_resource_accounting(d42: Any) -> Any:
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_k = int(resource["old_k_shot"])
        new_k = int(resource["new_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        inner_fit_count = (old_k if old_k >= 3 else 0) + (new_k if new_k >= 3 else 0)
        inner_macs = 0
        if old_k >= 3:
            inner_macs += old_k * int(d42._lda_fit_macs((old_k - 1) * old_count, old_count))
        if new_k >= 3:
            inner_macs += new_k * int(d42._lda_fit_macs((new_k - 1) * all_count, all_count))
        active = int(old_k >= 3) + int(new_k >= 3)
        spectral = active * _spectral_macs(int(d42.FEATURE_DIM))
        resource.update(
            {
                "d60_inner_covariance_fit_count": inner_fit_count,
                "d60_inner_covariance_fit_macs": inner_macs,
                "d60_spectral_dense_algebra_mac_equivalent_upper_bound": spectral,
                "d60_query_extra_macs": 0,
                "d60_persistent_state_extra_bytes": 0,
                "d60_optimizer_steps_extra": 0,
                "d60_resource_single_affine_state_only": True,
                "lda_closed_form_fit_count": int(resource["lda_closed_form_fit_count"] + inner_fit_count),
                "estimated_lda_fit_macs": int(resource["estimated_lda_fit_macs"] + inner_macs),
                "estimated_adaptation_macs": int(
                    resource["estimated_adaptation_macs"] + inner_macs + spectral
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
        raise D60ProbeError("D60 training row closure drift")
    audits = 0
    stability_min, stability_max = 1.0, 0.0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("lda_closed_form_fit_count", -1)) != 18
            or int(resource.get("d60_inner_covariance_fit_count", -1)) != 16
            or int(resource.get("d60_query_extra_macs", -1)) != 0
            or resource.get("d60_resource_single_affine_state_only") is not True
        ):
            raise D60ProbeError("D60 resource closure drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            if any(
                audit.get(name) != expected
                for name, expected in {
                    "d43_probe_arm": ARM,
                    "d43_covariance_structure": STRUCTURE,
                    "d60_probe_arm": ARM,
                    "d60_formula": FORMULA,
                    "d60_partition_policy": PARTITION,
                    "d60_boundary_status": "crossblock_spectral_stability_shrinkage_active",
                    "d60_stability_active": True,
                    "d60_class_shared_covariance": True,
                    "d60_class_logit_scale_or_intercept_calibration": False,
                    "d60_uses_outer_held_or_query": False,
                    "d60_threshold_or_rank_scan_count": 0,
                    "d60_single_affine_state_only": True,
                }.items()
            ):
                raise D60ProbeError("D60 exact audit drift")
            stability = np.asarray(audit["d60_stability_by_mode"], dtype=np.float64)
            if (
                stability.shape != (288,)
                or not np.isfinite(stability).all()
                or float(np.min(stability)) < -STABILITY_TOLERANCE
                or float(np.max(stability)) > 1.0 + STABILITY_TOLERANCE
                or audit["d60_inner_partition"]["held_support_row_exact_once_coverage"] is not True
                or audit["d60_fold_rayleigh_shape"] != [8, 288]
                or float(audit["d60_contracted_normalized_eigenvalue_min"]) <= 0.0
            ):
                raise D60ProbeError("D60 spectral/partition closure drift")
            stability_min = min(stability_min, float(np.min(stability)))
            stability_max = max(stability_max, float(np.max(stability)))
            audits += 1
    return {
        "verified_d60_target_row_count": len(target),
        "verified_d60_active_fit_audit_count": audits,
        "verified_d60_stability_min": stability_min,
        "verified_d60_stability_max": stability_max,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D60ProbeError("D60 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d60-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D60ProbeError(f"D60 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d60_d59_helper_sha256": d43._sha256(D59_HELPER_PATH),
        "d60_d43_helper_sha256": d43._sha256(d59.D43_HELPER_PATH),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_top = original_structured = None
    runner_name, exit_code = "d60_locked_d42_runner", 1
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, original_structured = build_d60_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D60ProbeError("D60 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d60_arm,
            probe_script_sha256=script_sha,
            extra_source_closure=helper_hashes,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top
        if original_structured is not None:
            d43._structured_covariance = original_structured
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_output(output, script_sha, helper_hashes)
    metadata = {
        "schema": "cvs.phase2.d60.crossblock_spectral_stability_shrinkage_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d60_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D60_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
