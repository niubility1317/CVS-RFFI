#!/usr/bin/env python3
"""D59 support-only full/block SPD geodesic-midpoint covariance probe."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
D43_HELPER_PATH = SCRIPT_DIR / "probe_d43_structured_covariance.py"
SPEC = importlib.util.spec_from_file_location("d59_d43_probe_helper", D43_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D59 could not load D43 helper")
d43 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d43)


ARM = "full_block_spd_geodesic_midpoint"
STRUCTURE = "affine_invariant_spd_midpoint_full_auto_and_block3"
FORMULA = "G=B^(1/2)*(B^(-1/2)*F*B^(-1/2))^(1/2)*B^(1/2)"
RICCATI_FORMULA = "G*B^(-1)*G=F"
SPD_EPSILON = 1.0e-12
SYMMETRY_TOLERANCE = 3.0e-10
RICCATI_RELATIVE_TOLERANCE = 3.0e-9

if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D59ProbeError(RuntimeError):
    pass


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise D59ProbeError("D59 covariance must be square")
    if not np.isfinite(value).all():
        raise D59ProbeError("D59 covariance became non-finite")
    return 0.5 * (value + value.T)


def _eigh_spd(matrix: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    symmetric = _symmetric(matrix)
    asymmetry = float(np.max(np.abs(np.asarray(matrix, dtype=np.float64) - symmetric)))
    scale = max(1.0, float(np.max(np.abs(symmetric))))
    if asymmetry > SYMMETRY_TOLERANCE * scale:
        raise D59ProbeError(f"D59 {name} covariance symmetry drift")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if (
        not np.isfinite(eigenvalues).all()
        or not np.isfinite(eigenvectors).all()
        or float(np.min(eigenvalues)) <= SPD_EPSILON
    ):
        raise D59ProbeError(f"D59 {name} covariance is not strictly SPD")
    return eigenvalues, eigenvectors


def _matrix_power_from_eigh(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, exponent: float
) -> np.ndarray:
    powered = np.power(np.asarray(eigenvalues, dtype=np.float64), float(exponent))
    return _symmetric((eigenvectors * powered[None, :]) @ eigenvectors.T)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _affine_invariant_distance(base: np.ndarray, target: np.ndarray) -> float:
    base_values, base_vectors = _eigh_spd(base, "distance_base")
    base_inverse_half = _matrix_power_from_eigh(base_values, base_vectors, -0.5)
    whitened = _symmetric(base_inverse_half @ _symmetric(target) @ base_inverse_half)
    whitened_values, _ = _eigh_spd(whitened, "distance_whitened")
    return float(np.linalg.norm(np.log(whitened_values)))


def _spd_geometric_midpoint(
    block_covariance: np.ndarray, full_covariance: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the affine-invariant SPD midpoint B#F and numerical audit."""

    block = _symmetric(block_covariance)
    full = _symmetric(full_covariance)
    if block.shape != full.shape:
        raise D59ProbeError("D59 full/block covariance shape drift")
    block_values, block_vectors = _eigh_spd(block, "block")
    full_values, _ = _eigh_spd(full, "full")
    block_half = _matrix_power_from_eigh(block_values, block_vectors, 0.5)
    block_inverse_half = _matrix_power_from_eigh(block_values, block_vectors, -0.5)
    normalized = _symmetric(block_inverse_half @ full @ block_inverse_half)
    normalized_values, normalized_vectors = _eigh_spd(normalized, "normalized")
    normalized_half = _matrix_power_from_eigh(
        normalized_values, normalized_vectors, 0.5
    )
    midpoint = _symmetric(block_half @ normalized_half @ block_half)
    midpoint_values, midpoint_vectors = _eigh_spd(midpoint, "midpoint")
    block_inverse = _matrix_power_from_eigh(block_values, block_vectors, -1.0)
    riccati = _symmetric(midpoint @ block_inverse @ midpoint)
    riccati_relative = float(
        np.linalg.norm(riccati - full, ord="fro")
        / max(float(np.linalg.norm(full, ord="fro")), SPD_EPSILON)
    )
    if riccati_relative > RICCATI_RELATIVE_TOLERANCE:
        raise D59ProbeError("D59 midpoint Riccati closure drift")
    block_to_full = float(np.linalg.norm(np.log(normalized_values)))
    block_to_midpoint = _affine_invariant_distance(block, midpoint)
    midpoint_to_full = _affine_invariant_distance(midpoint, full)
    half_distance_error = float(
        max(
            abs(block_to_midpoint - 0.5 * block_to_full),
            abs(midpoint_to_full - 0.5 * block_to_full),
        )
    )
    if half_distance_error > 2.0e-7 * max(1.0, block_to_full):
        raise D59ProbeError("D59 midpoint affine-distance closure drift")
    return midpoint, {
        "d59_formula": FORMULA,
        "d59_riccati_formula": RICCATI_FORMULA,
        "d59_dimension": int(midpoint.shape[0]),
        "d59_full_covariance_sha256": _array_sha256(full),
        "d59_block_covariance_sha256": _array_sha256(block),
        "d59_midpoint_covariance_sha256": _array_sha256(midpoint),
        "d59_full_eigenvalue_min": float(np.min(full_values)),
        "d59_full_eigenvalue_max": float(np.max(full_values)),
        "d59_block_eigenvalue_min": float(np.min(block_values)),
        "d59_block_eigenvalue_max": float(np.max(block_values)),
        "d59_midpoint_eigenvalue_min": float(np.min(midpoint_values)),
        "d59_midpoint_eigenvalue_max": float(np.max(midpoint_values)),
        "d59_full_condition_number": float(np.max(full_values) / np.min(full_values)),
        "d59_block_condition_number": float(
            np.max(block_values) / np.min(block_values)
        ),
        "d59_midpoint_condition_number": float(
            np.max(midpoint_values) / np.min(midpoint_values)
        ),
        "d59_normalized_eigenvalue_min": float(np.min(normalized_values)),
        "d59_normalized_eigenvalue_max": float(np.max(normalized_values)),
        "d59_riccati_relative_frobenius_residual": riccati_relative,
        "d59_affine_distance_block_to_full": block_to_full,
        "d59_affine_distance_block_to_midpoint": block_to_midpoint,
        "d59_affine_distance_midpoint_to_full": midpoint_to_full,
        "d59_affine_half_distance_max_abs_error": half_distance_error,
        "d59_midpoint_eigenvector_orthogonality_error": float(
            np.max(
                np.abs(
                    midpoint_vectors.T @ midpoint_vectors
                    - np.eye(midpoint.shape[0], dtype=np.float64)
                )
            )
        ),
    }


def _three_block_covariance(
    full_covariance: np.ndarray, block_slices: tuple[slice, ...]
) -> np.ndarray:
    full = _symmetric(full_covariance)
    block = np.zeros_like(full)
    covered = np.zeros(full.shape[0], dtype=np.int64)
    for block_slice in block_slices:
        indices = np.arange(full.shape[0])[block_slice]
        if len(indices) == 0:
            raise D59ProbeError("D59 empty covariance block")
        block[np.ix_(indices, indices)] = full[np.ix_(indices, indices)]
        covered[indices] += 1
    if not np.array_equal(covered, np.ones_like(covered)):
        raise D59ProbeError("D59 covariance blocks must partition the feature axis")
    _eigh_spd(block, "block3")
    return block


def build_d59_fit(
    d42: Any,
) -> tuple[
    Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    Callable[..., np.ndarray],
]:
    """Build D43-compatible fit while capturing D59 midpoint evidence."""

    original_structured = d43._structured_covariance
    captured: list[dict[str, Any]] = []

    def midpoint_structure(
        covariance: np.ndarray, arm: str, block_slices: tuple[slice, ...]
    ) -> np.ndarray:
        if arm != ARM:
            return original_structured(covariance, arm, block_slices)
        full = _symmetric(covariance)
        block = _three_block_covariance(full, block_slices)
        midpoint, evidence = _spd_geometric_midpoint(block, full)
        off_block = full - block
        evidence.update(
            {
                "d59_full_frobenius_norm": float(np.linalg.norm(full, ord="fro")),
                "d59_block_frobenius_norm": float(np.linalg.norm(block, ord="fro")),
                "d59_off_block_frobenius_norm": float(
                    np.linalg.norm(off_block, ord="fro")
                ),
                "d59_off_block_energy_fraction": float(
                    np.sum(off_block**2) / max(np.sum(full**2), SPD_EPSILON)
                ),
            }
        )
        captured.append(evidence)
        return midpoint

    d43._structured_covariance = midpoint_structure
    base_fit = d43.build_structured_fit(d42, ARM)

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        capture_count = len(captured)
        coefficient, intercept, base_audit = base_fit(
            transformed, targets, class_count, k_shot
        )
        fallback = bool(base_audit.get("unit_covariance_fallback"))
        if fallback:
            if len(captured) != capture_count:
                raise D59ProbeError("D59 fallback unexpectedly built midpoint")
            evidence = {
                "d59_formula": FORMULA,
                "d59_riccati_formula": RICCATI_FORMULA,
                "d59_boundary_status": "k1_rank0_or_zero_energy_exact_d42_fallback",
                "d59_midpoint_active": False,
            }
        else:
            if len(captured) != capture_count + 1:
                raise D59ProbeError("D59 midpoint evidence cardinality drift")
            evidence = dict(captured[-1])
            evidence.update(
                {
                    "d59_boundary_status": "full_block_spd_geodesic_midpoint_active",
                    "d59_midpoint_active": True,
                }
            )
        audit = dict(base_audit)
        audit.update(evidence)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d59_probe_arm": ARM,
                "d59_class_shared_covariance": True,
                "d59_class_logit_scale_or_intercept_calibration": False,
                "d59_class_id_specific_formula": False,
                "d59_old_new_role_specific_branch": False,
                "d59_scene_receiver_handle_specific_branch": False,
                "d59_uses_outer_held_or_query": False,
                "d59_geodesic_position": "exact_midpoint_no_tunable_weight",
                "d59_hyperparameter_scan_count": 0,
                "d59_single_affine_state_only": True,
            }
        )
        return coefficient, intercept, audit

    return fit, original_structured


def _extra_resource_per_active_fit(dimension: int) -> int:
    if dimension < 1:
        raise D59ProbeError("D59 resource dimension drift")
    # Conservative dense-algebra MAC-equivalent upper bound: eigensystems,
    # SPD powers, whitening, Riccati closure and affine-distance checks.
    return int(40 * dimension**3)


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        active_fit_count = int(int(resource["old_k_shot"]) > 1) + int(
            int(resource["new_k_shot"]) > 1
        )
        extra = active_fit_count * _extra_resource_per_active_fit(dimension)
        resource.update(
            {
                "d59_spd_midpoint_active_fit_count": active_fit_count,
                "d59_spd_midpoint_dense_algebra_mac_equivalent_upper_bound": extra,
                "d59_query_extra_macs": 0,
                "d59_persistent_state_extra_bytes": 0,
                "d59_optimizer_steps_extra": 0,
                "d59_resource_single_affine_state_only": True,
                "estimated_adaptation_macs": int(
                    resource["estimated_adaptation_macs"] + extra
                ),
            }
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_top, wrapped


def _verify_d59_rows(training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_rows = [
        row
        for row in training_rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(training_rows) != 105 or len(target_rows) != 30:
        raise D59ProbeError("D59 training-row closure drift")
    active_audits = 0
    riccati_max = 0.0
    half_distance_max = 0.0
    for row in target_rows:
        resource = row.get("resource", {})
        if (
            int(resource.get("d59_spd_midpoint_active_fit_count", -1)) != 2
            or int(resource.get("d59_query_extra_macs", -1)) != 0
            or int(resource.get("d59_persistent_state_extra_bytes", -1)) != 0
            or int(resource.get("d59_optimizer_steps_extra", -1)) != 0
            or resource.get("d59_resource_single_affine_state_only") is not True
        ):
            raise D59ProbeError("D59 resource closure drift")
        expected_extra = 2 * _extra_resource_per_active_fit(288)
        if (
            int(
                resource.get(
                    "d59_spd_midpoint_dense_algebra_mac_equivalent_upper_bound", -1
                )
            )
            != expected_extra
        ):
            raise D59ProbeError("D59 dense algebra resource drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row.get("geometry_summary", {}).get(field, {})
            exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d59_probe_arm": ARM,
                "d59_boundary_status": "full_block_spd_geodesic_midpoint_active",
                "d59_midpoint_active": True,
                "d59_class_shared_covariance": True,
                "d59_class_logit_scale_or_intercept_calibration": False,
                "d59_class_id_specific_formula": False,
                "d59_old_new_role_specific_branch": False,
                "d59_scene_receiver_handle_specific_branch": False,
                "d59_uses_outer_held_or_query": False,
                "d59_geodesic_position": "exact_midpoint_no_tunable_weight",
                "d59_hyperparameter_scan_count": 0,
                "d59_single_affine_state_only": True,
                "d59_formula": FORMULA,
                "d59_riccati_formula": RICCATI_FORMULA,
            }
            if any(audit.get(name) != value for name, value in exact.items()):
                raise D59ProbeError("D59 exact fit audit drift")
            hash_fields = (
                "d59_full_covariance_sha256",
                "d59_block_covariance_sha256",
                "d59_midpoint_covariance_sha256",
            )
            if any(
                not isinstance(audit.get(name), str) or len(audit[name]) != 64
                for name in hash_fields
            ):
                raise D59ProbeError("D59 covariance hash evidence drift")
            positive_fields = (
                "d59_full_eigenvalue_min",
                "d59_block_eigenvalue_min",
                "d59_midpoint_eigenvalue_min",
                "d59_full_condition_number",
                "d59_block_condition_number",
                "d59_midpoint_condition_number",
            )
            if any(
                not math.isfinite(float(audit.get(name, float("nan"))))
                or float(audit[name]) <= 0.0
                for name in positive_fields
            ):
                raise D59ProbeError("D59 SPD evidence drift")
            riccati = float(audit["d59_riccati_relative_frobenius_residual"])
            half_error = float(audit["d59_affine_half_distance_max_abs_error"])
            if (
                not math.isfinite(riccati)
                or riccati > RICCATI_RELATIVE_TOLERANCE
                or not math.isfinite(half_error)
                or half_error > 2.0e-7
            ):
                raise D59ProbeError("D59 midpoint closure evidence drift")
            riccati_max = max(riccati_max, riccati)
            half_distance_max = max(half_distance_max, half_error)
            active_audits += 1
    if active_audits != 60:
        raise D59ProbeError("D59 active audit count drift")
    return {
        "verified_d59_training_row_count": len(target_rows),
        "verified_d59_active_fit_audit_count": active_audits,
        "verified_d59_riccati_relative_residual_max": riccati_max,
        "verified_d59_affine_half_distance_error_max": half_distance_max,
    }


def _verify_output(
    output: Path, script_sha: str, helper_sha: str
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if closure.get("d59_d43_helper_sha256") != helper_sha:
        raise D59ProbeError("D59 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_d59_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d59-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D59ProbeError(f"D59 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_sha = d43._sha256(D43_HELPER_PATH)
    helper_hashes = {"d59_d43_helper_sha256": helper_sha}
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_path: tuple[str, ...] = ()
    original_fit = None
    original_top = None
    original_structured = None
    runner_name = "d59_locked_d42_runner"
    exit_code = 1
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d59_fit, original_structured = build_d59_fit(d42)
        d42._fit_equal_prior_lda = d59_fit
        original_top, _ = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D59ProbeError("D59 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d59_arm,
            probe_script_sha256=script_sha,
            extra_source_closure=helper_hashes,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
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
    evidence = _verify_output(output, script_sha, helper_sha)
    metadata = {
        "schema": "cvs.phase2.d59.full_block_spd_geodesic_midpoint_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d59_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "d43_helper_sha256": helper_sha,
        "formula": FORMULA,
        "riccati_formula": RICCATI_FORMULA,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D59_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
