#!/usr/bin/env python3
"""D82 support-only ground-spectrum robust-center Wiener-residual probe."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D80_HELPER_PATH = SCRIPT_DIR / "probe_d80_ground_commonmode_covariance_denoiser.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d82_ground_nuisance_wiener_residual.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D82 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d80 = _load("d82_d80_ground_loader_scaffold", D80_HELPER_PATH)
core = _load("d82_ground_nuisance_wiener_core", CORE_PATH)
d66, d62, d43 = d80.d66, d80.d62, d80.d43

ARM = "ground_nuisance_wiener_residual"
CONFIRMATION_SEEDS = (713102, 713103, 713104, 713105, 713106)
STRUCTURE = "d62_with_ground_spectrum_robust_center_and_wiener_residual"
FORMULA = (
    "dequantize all immutable 84 ground domain-class centers; class-center the "
    "cross-domain drift covariance; retain ceil(participation-ratio effective rank) "
    "directions without a scan; inside every D62 full/block outer and held fit, "
    "compute per-target-class ground-spectrum residual energy and one-step Cauchy "
    "center; set the signal scale to the mean retained ground eigenvalue and apply "
    "the closed-form Wiener retention mean/(direction+mean) to z160 within-class "
    "residuals; preserve FFT96/RF32; fit the locked "
    "target-support D62 metric and compile one INT8 affine head"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D82ProbeError(RuntimeError):
    """Raised when D82 integration or evidence drifts."""


def _install_confirmation_cell_guard(
    runner_module: Any, confirmation_seed: int | None
) -> Callable[..., Any] | None:
    """Allow only preregistered D18 confirmation seeds on the locked D42 cell."""

    if confirmation_seed is None:
        return None
    if confirmation_seed not in CONFIRMATION_SEEDS:
        raise D82ProbeError(
            f"D82 confirmation seed is not preregistered: {confirmation_seed}"
        )
    original = runner_module._require_d42_development_cell

    def require_confirmation_cell(
        before_manifest: dict[str, Any], after_manifest: dict[str, Any]
    ) -> None:
        old_classes = runner_module.legacy._registered_handles(before_manifest)
        all_classes = runner_module.legacy._registered_handles(after_manifest)
        if (
            str(before_manifest.get("receiver"))
            != runner_module.D42_DEVELOPMENT_RECEIVER
            or str(after_manifest.get("receiver"))
            != runner_module.D42_DEVELOPMENT_RECEIVER
            or int(before_manifest.get("seed", -1)) != confirmation_seed
            or int(after_manifest.get("seed", -1)) != confirmation_seed
            or int(before_manifest.get("k_shot", -1)) != 10
            or int(after_manifest.get("k_shot", -1)) != 10
            or all_classes[: len(old_classes)] != old_classes
            or len(all_classes) - len(old_classes)
            != runner_module.D42_DEVELOPMENT_NEW_CLASS_COUNT
        ):
            raise runner_module.D25RunnerError(
                "D82 confirmation cell must be receiver 20-1, "
                f"seed {confirmation_seed}, K10, new5"
            )

    runner_module._require_d42_development_cell = require_confirmation_cell
    return original


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_ground_basis(
    component_dir: Path,
    manifest_sha256: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    covariance, audit = d80.load_ground_covariance(
        component_dir, manifest_sha256, feature_dim
    )
    basis, weights, basis_audit = core.ground_nuisance_basis(
        covariance, audit["quantization_noise_floor"]
    )
    combined = dict(audit)
    combined.update({f"d82_{key}": value for key, value in basis_audit.items()})
    combined.update(
        {
            "ground_bundle_contains_sample_radius": False,
            "ground_bundle_contains_sample_count": False,
            "ground_statistic_semantics": (
                "class_centered_cross_domain_centroid_drift_eigenspectrum"
            ),
            "d82_basis_transient_fp64_bytes": int(
                basis.size * 8 + weights.size * 8
            ),
        }
    )
    return basis, weights, combined


def build_d82_fit(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    ground_audit: dict[str, Any],
) -> tuple[Callable[..., Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Inject robust-center Wiener transforms into every D62 OOF closure."""

    aliases = (
        d62.d43,
        d62.d61.d43,
        d62.d61.d46.d43,
        d62.d61.d46.d45.d43,
    )
    if any(alias is not d43 for alias in aliases):
        raise D82ProbeError("D82 D43 module alias identity drift")
    original_fit = d42._fit_equal_prior_lda
    original_builder = d43.build_structured_fit
    transform_records: list[dict[str, Any]] = []
    basis_audit = {
        "basis_sha256": ground_audit["d82_basis_sha256"],
        "spectral_weight_sha256": ground_audit[
            "d82_spectral_weight_sha256"
        ],
        "participation_ratio_effective_rank": ground_audit[
            "d82_participation_ratio_effective_rank"
        ],
        "retained_rank": ground_audit["d82_retained_rank"],
        "rank_policy": ground_audit["d82_rank_policy"],
    }
    full_fit = core.build_wiener_component_fit(
        original_fit,
        basis,
        spectral_weights,
        basis_audit,
        "full",
        transform_records,
    )

    def structured_builder(d42_arg: Any, arm: str) -> Callable[..., Any]:
        if d42_arg is not d42 or arm != "block3_centered":
            raise D82ProbeError("D82 unexpected structured covariance request")
        base = original_builder(d42_arg, arm)
        return core.build_wiener_component_fit(
            base,
            basis,
            spectral_weights,
            basis_audit,
            arm,
            transform_records,
        )

    try:
        d42._fit_equal_prior_lda = full_fit
        d43.build_structured_fit = structured_builder
        base_fit, call_records = d62.build_d62_fit(d42)
    finally:
        d42._fit_equal_prior_lda = original_fit
        d43.build_structured_fit = original_builder

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, base_audit = base_fit(
            rows, labels, class_count, k_shot
        )
        audit = dict(base_audit)
        audit.update(
            {
                "d82_probe_arm": ARM,
                "d82_structure": STRUCTURE,
                "d82_formula": FORMULA,
                "d82_ground_int8_component_used": True,
                "d82_ground_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "d82_ground_component_update_access": False,
                "d82_ground_statistic_semantics": ground_audit[
                    "ground_statistic_semantics"
                ],
                "d82_ground_bundle_contains_sample_radius": False,
                "d82_ground_bundle_contains_sample_count": False,
                "d82_ground_effective_rank": ground_audit[
                    "d82_participation_ratio_effective_rank"
                ],
                "d82_ground_retained_rank": int(
                    ground_audit["d82_retained_rank"]
                ),
                "d82_ground_rank_policy": ground_audit["d82_rank_policy"],
                "d82_all_full_block_outer_held_fits_transformed": True,
                "d82_target_covariance_preserved_by_class_translation": False,
                "d82_target_covariance_changed_by_fixed_ground_wiener": True,
                "d82_query_metric_source": "target_support_only_d62",
                "d82_old_new_role_specific_branch": False,
                "d82_class_id_specific_formula": False,
                "d82_scene_receiver_handle_specific_branch": False,
                "d82_uses_outer_held_or_query": False,
                "d82_query_rows_used": 0,
                "d82_hyperparameter_count": 0,
                "d82_rank_scan_count": 0,
                "d82_weight_scan_count": 0,
                "d82_optimizer_steps": 0,
                "d82_single_affine_state_only": True,
                "d82_actual_coefficient_fp32": np.asarray(
                    coefficient, dtype=np.float32
                ).tolist(),
                "d82_actual_intercept_fp32": np.asarray(
                    intercept, dtype=np.float32
                ).tolist(),
            }
        )
        return coefficient, intercept, audit

    return fit, call_records, transform_records


def _support_transform_macs(row_count: int, rank: int) -> int:
    rows, retained = int(row_count), int(rank)
    return int(
        4 * rows * core.Z_DIM * retained
        + 7 * rows * retained
        + 3 * rows * core.Z_DIM
    )


def _d62_transform_chain_macs(class_count: int, k_shot: int, rank: int) -> int:
    classes, shots = int(class_count), int(k_shot)
    outer = _support_transform_macs(classes * shots, rank)
    if shots == 1:
        return 2 * outer
    inner = _support_transform_macs(classes * (shots - 1), rank)
    if shots == 2:
        return 2 * outer + 2 * shots * inner
    return 4 * outer + 4 * shots * inner


def _install_resource_accounting(
    d42: Any, ground_audit: dict[str, Any]
) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        rank = int(ground_audit["d82_retained_rank"])
        old_k, new_k = int(resource["old_k_shot"]), int(resource["new_k_shot"])
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        statistics_macs = int(
            ground_audit["ground_covariance_statistics_mac_upper_bound"]
        )
        transform_macs = int(
            _d62_transform_chain_macs(old_count, old_k, rank)
            + _d62_transform_chain_macs(all_count, new_k, rank)
        )
        added_macs = statistics_macs + transform_macs
        component_bytes = int(
            ground_audit["ground_int8_component_logical_state_bytes"]
        )
        resource.update(
            {
                "d82_ground_int8_component_used": True,
                "d82_ground_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "d82_ground_component_update_access": False,
                "d82_ground_component_logical_state_bytes": component_bytes,
                "d82_ground_spectrum_statistics_macs": statistics_macs,
                "d82_support_wiener_transform_mac_upper_bound": transform_macs,
                "d82_total_added_adaptation_macs": added_macs,
                "d82_ground_retained_rank": rank,
                "d82_optimizer_steps_extra": 0,
                "d82_trainable_parameters_extra": 0,
                "d82_query_extra_macs": 0,
                "d82_query_extra_state_bytes": 0,
                "d82_persistent_compiled_transform_bytes": 0,
                "d82_ground_basis_transient_fp64_bytes": int(
                    ground_audit["d82_basis_transient_fp64_bytes"]
                ),
                "d82_single_affine_state_only": True,
                "ground_int8_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "ground_int8_update_access": False,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + added_macs
        )
        resource["persistent_state_bytes"] = int(
            resource["persistent_state_bytes"] + component_bytes
        )
        resource["persistent_state_cap_pass"] = bool(
            resource["persistent_state_bytes"]
            <= resource["persistent_state_cap_bytes"]
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _install_runner_resource_accounting(runner: Any) -> None:
    original_evaluate = runner._evaluate_d42_fold

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_evaluate(*args, **kwargs)
        resource = dict(row["resource"])
        if "d82_ground_component_logical_state_bytes" not in resource:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d82_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d82_compiled_affine_state_bytes": head_bytes,
                "d82_component_inclusive_persistent_state_bytes": total_bytes,
                "persistent_state_bytes": total_bytes,
                "persistent_state_cap_pass": total_bytes
                <= int(resource["persistent_state_cap_bytes"]),
            }
        )
        return {**row, "resource": resource}

    runner._evaluate_d42_fold = evaluate


def _verify_rows(
    rows: list[dict[str, Any]], ground_audit: dict[str, Any]
) -> dict[str, Any]:
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        resource = row.get("resource")
        if not isinstance(resource, dict) or "d82_total_added_adaptation_macs" not in resource:
            continue
        resource["estimated_adaptation_macs"] -= int(
            resource["d82_total_added_adaptation_macs"]
        )
        resource["persistent_state_bytes"] = int(
            resource["d82_compiled_affine_state_bytes"]
        )
        resource["ground_int8_component_input_count"] = 0
        resource["persistent_state_cap_pass"] = True
    base = d62._verify_rows(sanitized)
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    shifts: list[float] = []
    minimum_weights: list[float] = []
    minimum_effective_samples: list[float] = []
    maximum_energy_retention: list[float] = []
    for row in target:
        resource = row["resource"]
        if (
            resource.get("d82_ground_int8_component_used") is not True
            or int(resource.get("d82_ground_component_input_count", -1)) != 84
            or resource.get("d82_ground_component_update_access") is not False
            or int(resource.get("d82_optimizer_steps_extra", -1)) != 0
            or int(resource.get("d82_trainable_parameters_extra", -1)) != 0
            or int(resource.get("d82_query_extra_macs", -1)) != 0
            or resource.get("d82_single_affine_state_only") is not True
            or int(resource.get("persistent_state_bytes", -1))
            != int(resource.get("d82_compiled_affine_state_bytes", -2))
            + int(resource.get("d82_ground_component_logical_state_bytes", -3))
            or not bool(resource.get("persistent_state_cap_pass"))
        ):
            raise D82ProbeError("D82 resource closure drift")
        for field, expected_classes in (
            ("before_covariance_audit", 6),
            ("final_covariance_audit", 11),
        ):
            audit = row["geometry_summary"][field]
            transform = audit.get("d82_transform_audit", {})
            if (
                audit.get("d82_probe_arm") != ARM
                or audit.get("d82_structure") != STRUCTURE
                or audit.get("d82_formula") != FORMULA
                or audit.get("d82_ground_int8_component_used") is not True
                or audit.get("d82_ground_component_update_access") is not False
                or int(audit.get("d82_ground_retained_rank", -1))
                != int(ground_audit["d82_retained_rank"])
                or audit.get("d82_all_full_block_outer_held_fits_transformed") is not True
                or audit.get("d82_target_covariance_preserved_by_class_translation") is not False
                or audit.get("d82_target_covariance_changed_by_fixed_ground_wiener") is not True
                or audit.get("d82_query_metric_source") != "target_support_only_d62"
                or audit.get("d82_old_new_role_specific_branch") is not False
                or audit.get("d82_class_id_specific_formula") is not False
                or audit.get("d82_uses_outer_held_or_query") is not False
                or int(audit.get("d82_query_rows_used", -1)) != 0
                or int(audit.get("d82_hyperparameter_count", -1)) != 0
                or int(audit.get("d82_rank_scan_count", -1)) != 0
                or int(audit.get("d82_weight_scan_count", -1)) != 0
                or int(audit.get("d82_optimizer_steps", -1)) != 0
                or audit.get("d82_single_affine_state_only") is not True
                or int(transform.get("class_count", -1)) != expected_classes
                or int(transform.get("k_shot", -1)) != 8
                or transform.get("transform_scope")
                != "z160_class_common_center_plus_within_class_ground_wiener"
                or transform.get("uses_outer_held_or_query") is not False
                or transform.get("within_class_residual_changed") is not True
                or float(transform.get("wiener_residual_formula_max_abs_error", 1.0)) > 2e-12
                or float(transform.get("robust_center_formula_max_abs_error", 1.0)) > 2e-12
                or float(transform.get("fft96_rf32_max_abs_error", 1.0)) != 0.0
            ):
                raise D82ProbeError("D82 fit closure drift")
            shifts.append(float(transform["center_shift_l2_max"]))
            minimum_weights.append(float(transform["normalized_weight_min"]))
            minimum_effective_samples.append(
                min(float(value) for value in transform["effective_sample_size_by_class"])
            )
            maximum_energy_retention.append(
                max(
                    float(value)
                    for value in transform["nuisance_energy_retention_ratio_by_class"]
                )
            )
    return {
        **base,
        "verified_d82_target_row_count": len(target),
        "verified_d82_fit_audit_count": 2 * len(target),
        "verified_d82_ground_component_input_count": int(
            ground_audit["ground_component_input_count"]
        ),
        "verified_d82_ground_retained_rank": int(
            ground_audit["d82_retained_rank"]
        ),
        "verified_d82_center_shift_l2_min": min(shifts),
        "verified_d82_center_shift_l2_max": max(shifts),
        "verified_d82_normalized_weight_min": min(minimum_weights),
        "verified_d82_effective_sample_size_min": min(minimum_effective_samples),
        "verified_d82_wiener_retention_min": float(
            ground_audit["d82_wiener_retention_min"]
        ),
        "verified_d82_wiener_retention_max": float(
            ground_audit["d82_wiener_retention_max"]
        ),
        "verified_d82_nuisance_energy_retention_max": max(maximum_energy_retention),
    }


def _verify_output(
    output: Path,
    helper_hashes: dict[str, str],
    ground_audit: dict[str, Any],
) -> dict[str, Any]:
    receipt = d43._read_json(output / "RECEIPT.json")
    if int(receipt.get("training_log_row_count", -1)) != 105:
        raise D82ProbeError("D82 receipt row closure drift")
    support = d43._read_json(output / "support_audit.json")
    lock = support.get("candidate_lock", {})
    probe_lock = lock.get("d43_probe_lock", {})
    if (
        probe_lock.get("arm") != ARM
        or probe_lock.get("formal_candidate") is not False
        or probe_lock.get("forced_nonpromotable") is not True
        or probe_lock.get("selected_only_full_k10_refit_allowed") is not False
    ):
        raise D82ProbeError("D82 diagnostic lock drift")
    closure = lock.get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D82ProbeError("D82 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {
        "base_runner_receipt_sha256": d43._sha256(output / "RECEIPT.json"),
        "verified_training_row_count": len(rows),
        "verified_query_opened": False,
        "verified_forced_nonpromotable": True,
        **_verify_rows(rows, ground_audit),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d82-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-component-dir", required=True, type=Path)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--d82-confirmation-seed", type=int)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D82ProbeError(f"D82 output already exists: {output}")
    basis, spectral_weights, ground_audit = load_ground_basis(
        known.ground_component_dir,
        known.ground_manifest_sha256,
        288,
    )
    npz_path = known.ground_component_dir / d66.NPZ_NAME
    manifest_path = known.ground_component_dir / d66.MANIFEST_NAME
    entry_npz_sha = d43._sha256(npz_path)
    entry_manifest_sha = d43._sha256(manifest_path)
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d82_core_sha256": d43._sha256(CORE_PATH),
        "d82_d80_loader_sha256": d43._sha256(D80_HELPER_PATH),
        "d82_d66_loader_sha256": d43._sha256(d80.D66_HELPER_PATH),
        "d82_d62_helper_sha256": d43._sha256(d66.D62_HELPER_PATH),
        "d82_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d82_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d82_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = original_cell_guard = None
    runner_name, exit_code = "d82_locked_d42_runner", 1
    runner_module = None
    call_records: list[dict[str, Any]] = []
    transform_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, call_records, transform_records = build_d82_fit(
            d42, basis, spectral_weights, ground_audit
        )
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(
            d42, ground_audit
        )
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D82ProbeError("D82 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        original_cell_guard = _install_confirmation_cell_guard(
            runner_module, known.d82_confirmation_seed
        )
        _install_runner_resource_accounting(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d82_arm,
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
        if runner_module is not None and original_cell_guard is not None:
            runner_module._require_d42_development_cell = original_cell_guard
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    if len(call_records) != 30 * 36:
        raise D82ProbeError("D82 D62 component-fit count drift")
    if len(transform_records) != 30 * 72:
        raise D82ProbeError(
            f"D82 transform-fit count drift: {len(transform_records)}"
        )
    exit_npz_sha = d43._sha256(npz_path)
    exit_manifest_sha = d43._sha256(manifest_path)
    if exit_npz_sha != entry_npz_sha or exit_manifest_sha != entry_manifest_sha:
        raise D82ProbeError("D82 ground component changed during probe")
    evidence = _verify_output(output, helper_hashes, ground_audit)
    call_record_sha = _sha256_bytes(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    transform_record_sha = _sha256_bytes(
        json.dumps(
            transform_records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    metadata = {
        "schema": "cvs.phase2.d82.ground_nuisance_wiener_residual_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d82_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "evaluation_role": (
            "independent_confirmation"
            if known.d82_confirmation_seed is not None
            else "development"
        ),
        "confirmation_seed": known.d82_confirmation_seed,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": call_record_sha,
        "support_wiener_transform_execution_count": len(transform_records),
        "support_wiener_transform_record_sha256": transform_record_sha,
        "ground_component_entry_npz_sha256": entry_npz_sha,
        "ground_component_exit_npz_sha256": exit_npz_sha,
        "ground_component_entry_manifest_sha256": entry_manifest_sha,
        "ground_component_exit_manifest_sha256": exit_manifest_sha,
        "ground_component_bitwise_unchanged": True,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        "ground_audit": ground_audit,
        **evidence,
    }
    (output / "D82_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



