#!/usr/bin/env python3
"""D80 support-only ground nuisance empirical-Bayes covariance probe."""

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
D66_HELPER_PATH = SCRIPT_DIR / "probe_d66_ground_domain_reliability_residual.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d80_ground_commonmode_denoiser.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D80 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d66 = _load("d80_d66_loader_scaffold", D66_HELPER_PATH)
core = _load("d80_ground_covariance_core", CORE_PATH)
d62, d43 = d66.d62, d66.d43

ARM = "ground_commonmode_covariance_denoiser"
STRUCTURE = "d62_with_ground_classcentered_empirical_bayes_full_and_block_covariance"
FORMULA = (
    "dequantize the immutable 84-cell ground domain-class centers; remove each "
    "ground class mean and pool the 78 cross-domain centroid residual degrees into "
    "one shared z160 covariance with mean(scale^2/12) quantization floor; for every "
    "D62 full/block outer and physical-rank-held fit, trace-match that covariance to "
    "target z160 and combine it with target shrinkage covariance using the fixed "
    "weight (D_eff-1)/(D_eff-1+C*(K-1)); solve equal-prior Mahalanobis rows and "
    "retain the locked D62 crossfitted row splice; compile one INT8 affine head"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D80ProbeError(RuntimeError):
    """Raised when D80 integration or evidence drifts."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_ground_covariance(
    component_dir: Path,
    manifest_sha256: str,
    feature_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse D66's strict v1 validation, then derive only covariance shape."""

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
    covariance, covariance_audit = core.ground_classcentered_covariance(
        prototypes, scales, mask
    )
    active_cells = int(covariance_audit["ground_component_input_count"])
    z_dim = int(covariance_audit["z_dimension"])
    statistics_macs = int(
        4 * active_cells * z_dim * z_dim + 20 * z_dim * z_dim * z_dim
    )
    audit = dict(ground_audit)
    audit.update(covariance_audit)
    audit.update(
        {
            "ground_covariance_statistics_mac_upper_bound": statistics_macs,
            "ground_covariance_transient_fp64_bytes": z_dim * z_dim * 8,
            "ground_bundle_contains_sample_radius": False,
            "ground_bundle_contains_sample_count": False,
            "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_covariance",
        }
    )
    return covariance, audit


def build_d80_fit(
    d42: Any,
    ground_covariance: np.ndarray,
    ground_audit: dict[str, Any],
) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    """Inject D80 into all D62 full/block closures before final-row construction."""

    aliases = (
        d62.d43,
        d62.d61.d43,
        d62.d61.d46.d43,
        d62.d61.d46.d45.d43,
    )
    if any(alias is not d43 for alias in aliases):
        raise D80ProbeError("D80 D43 module alias identity drift")
    original_fit = d42._fit_equal_prior_lda
    original_builder = d43.build_structured_fit
    full_fit = core.build_ground_prior_equal_lda(
        d42, ground_covariance, ground_audit, arm="full"
    )

    def structured_builder(d42_arg: Any, arm: str) -> Callable[..., Any]:
        if d42_arg is not d42 or arm != "block3_centered":
            raise D80ProbeError("D80 unexpected structured covariance request")
        return core.build_ground_prior_equal_lda(
            d42_arg, ground_covariance, ground_audit, arm=arm
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
        classes, shots = int(class_count), int(k_shot)
        domain_degrees = int(
            ground_audit["ground_independent_domain_degrees_of_freedom"]
        )
        target_degrees = classes * (shots - 1)
        audit = dict(base_audit)
        audit.update(
            {
                "d80_probe_arm": ARM,
                "d80_structure": STRUCTURE,
                "d80_formula": FORMULA,
                "d80_ground_int8_component_used": True,
                "d80_ground_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "d80_ground_component_update_access": False,
                "d80_ground_covariance_sha256": ground_audit["covariance_sha256"],
                "d80_ground_statistic_semantics": ground_audit[
                    "ground_statistic_semantics"
                ],
                "d80_ground_bundle_contains_sample_radius": False,
                "d80_ground_bundle_contains_sample_count": False,
                "d80_full_and_block_component_prior_injected": True,
                "d80_target_degrees_of_freedom": target_degrees,
                "d80_ground_independent_domain_degrees_of_freedom": domain_degrees,
                "d80_ground_shrinkage_weight": domain_degrees
                / float(domain_degrees + target_degrees),
                "d80_shared_metric_all_registered_classes": True,
                "d80_old_new_role_specific_branch": False,
                "d80_class_id_specific_formula": False,
                "d80_scene_receiver_handle_specific_branch": False,
                "d80_uses_outer_held_or_query": False,
                "d80_query_rows_used": 0,
                "d80_hyperparameter_count": 0,
                "d80_optimizer_steps": 0,
                "d80_single_affine_state_only": True,
                "d80_actual_coefficient_fp32": np.asarray(
                    coefficient, dtype=np.float32
                ).tolist(),
                "d80_actual_intercept_fp32": np.asarray(
                    intercept, dtype=np.float32
                ).tolist(),
            }
        )
        return coefficient, intercept, audit

    return fit, call_records


def _install_resource_accounting(
    d42: Any, ground_audit: dict[str, Any]
) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        fit_count = int(resource["lda_closed_form_fit_count"])
        dimension = int(d42.FEATURE_DIM)
        stats_macs = int(ground_audit["ground_covariance_statistics_mac_upper_bound"])
        per_fit_macs = int(6 * dimension * dimension + 8 * core.Z_DIM * core.Z_DIM)
        posterior_macs = fit_count * per_fit_macs
        added_macs = stats_macs + posterior_macs
        component_bytes = int(
            ground_audit["ground_int8_component_logical_state_bytes"]
        )
        resource.update(
            {
                "d80_ground_int8_component_used": True,
                "d80_ground_component_input_count": int(
                    ground_audit["ground_component_input_count"]
                ),
                "d80_ground_component_update_access": False,
                "d80_ground_component_logical_state_bytes": component_bytes,
                "d80_ground_covariance_statistics_macs": stats_macs,
                "d80_posterior_covariance_mac_upper_bound": posterior_macs,
                "d80_total_added_adaptation_macs": added_macs,
                "d80_optimizer_steps_extra": 0,
                "d80_trainable_parameters_extra": 0,
                "d80_query_extra_macs": 0,
                "d80_query_extra_state_bytes": 0,
                "d80_persistent_compiled_transform_bytes": 0,
                "d80_ground_covariance_transient_fp64_bytes": int(
                    ground_audit["ground_covariance_transient_fp64_bytes"]
                ),
                "d80_ground_covariance_sha256": ground_audit["covariance_sha256"],
                "d80_single_affine_state_only": True,
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
        if "d80_ground_component_logical_state_bytes" not in resource:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d80_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d80_compiled_affine_state_bytes": head_bytes,
                "d80_component_inclusive_persistent_state_bytes": total_bytes,
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
        if not isinstance(resource, dict) or "d80_total_added_adaptation_macs" not in resource:
            continue
        resource["estimated_adaptation_macs"] -= int(
            resource["d80_total_added_adaptation_macs"]
        )
        resource["persistent_state_bytes"] = int(
            resource["d80_compiled_affine_state_bytes"]
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
    weights: list[float] = []
    for row in target:
        resource = row["resource"]
        if (
            resource.get("d80_ground_int8_component_used") is not True
            or int(resource.get("d80_ground_component_input_count", -1)) != 84
            or resource.get("d80_ground_component_update_access") is not False
            or int(resource.get("d80_optimizer_steps_extra", -1)) != 0
            or int(resource.get("d80_trainable_parameters_extra", -1)) != 0
            or int(resource.get("d80_query_extra_macs", -1)) != 0
            or resource.get("d80_single_affine_state_only") is not True
            or int(resource.get("persistent_state_bytes", -1))
            != int(resource.get("d80_compiled_affine_state_bytes", -2))
            + int(resource.get("d80_ground_component_logical_state_bytes", -3))
            or not bool(resource.get("persistent_state_cap_pass"))
        ):
            raise D80ProbeError("D80 resource closure drift")
        for field, expected_classes in (
            ("before_covariance_audit", 6),
            ("final_covariance_audit", 11),
        ):
            audit = row["geometry_summary"][field]
            expected_weight = 13.0 / (13.0 + expected_classes * 7.0)
            if (
                audit.get("d80_probe_arm") != ARM
                or audit.get("d80_structure") != STRUCTURE
                or audit.get("d80_formula") != FORMULA
                or audit.get("d80_ground_int8_component_used") is not True
                or audit.get("d80_ground_component_update_access") is not False
                or audit.get("d80_full_and_block_component_prior_injected") is not True
                or audit.get("d80_shared_metric_all_registered_classes") is not True
                or audit.get("d80_old_new_role_specific_branch") is not False
                or audit.get("d80_class_id_specific_formula") is not False
                or audit.get("d80_uses_outer_held_or_query") is not False
                or int(audit.get("d80_query_rows_used", -1)) != 0
                or int(audit.get("d80_hyperparameter_count", -1)) != 0
                or int(audit.get("d80_optimizer_steps", -1)) != 0
                or audit.get("d80_single_affine_state_only") is not True
                or not np.isclose(
                    float(audit.get("d80_ground_shrinkage_weight", -1.0)),
                    expected_weight,
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                raise D80ProbeError("D80 covariance fit closure drift")
            weights.append(float(audit["d80_ground_shrinkage_weight"]))
    return {
        **base,
        "verified_d80_target_row_count": len(target),
        "verified_d80_fit_audit_count": 2 * len(target),
        "verified_d80_ground_component_input_count": int(
            ground_audit["ground_component_input_count"]
        ),
        "verified_d80_ground_residual_numerical_rank": int(
            ground_audit["ground_residual_numerical_rank"]
        ),
        "verified_d80_ground_shrinkage_weight_min": min(weights),
        "verified_d80_ground_shrinkage_weight_max": max(weights),
    }


def _verify_output(
    output: Path,
    helper_hashes: dict[str, str],
    ground_audit: dict[str, Any],
) -> dict[str, Any]:
    receipt = d43._read_json(output / "RECEIPT.json")
    if int(receipt.get("training_log_row_count", -1)) != 105:
        raise D80ProbeError("D80 receipt row closure drift")
    support = d43._read_json(output / "support_audit.json")
    lock = support.get("candidate_lock", {})
    probe_lock = lock.get("d43_probe_lock", {})
    if (
        probe_lock.get("arm") != ARM
        or probe_lock.get("formal_candidate") is not False
        or probe_lock.get("forced_nonpromotable") is not True
        or probe_lock.get("selected_only_full_k10_refit_allowed") is not False
    ):
        raise D80ProbeError("D80 diagnostic lock drift")
    closure = lock.get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D80ProbeError("D80 helper source closure drift")
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
    parser.add_argument("--d80-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-component-dir", required=True, type=Path)
    parser.add_argument("--ground-manifest-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D80ProbeError(f"D80 output already exists: {output}")
    covariance, ground_audit = load_ground_covariance(
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
        "d80_core_sha256": d43._sha256(CORE_PATH),
        "d80_d66_loader_sha256": d43._sha256(D66_HELPER_PATH),
        "d80_d62_helper_sha256": d43._sha256(d66.D62_HELPER_PATH),
        "d80_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d80_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d80_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d80_locked_d42_runner", 1
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d80_fit(d42, covariance, ground_audit)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(
            d42, ground_audit
        )
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D80ProbeError("D80 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        _install_runner_resource_accounting(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d80_arm,
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
    if len(call_records) != 30 * 36:
        raise D80ProbeError("D80 component-fit count drift")
    exit_npz_sha = d43._sha256(npz_path)
    exit_manifest_sha = d43._sha256(manifest_path)
    if exit_npz_sha != entry_npz_sha or exit_manifest_sha != entry_manifest_sha:
        raise D80ProbeError("D80 ground component changed during probe")
    evidence = _verify_output(output, helper_hashes, ground_audit)
    record_sha = _sha256_bytes(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    metadata = {
        "schema": "cvs.phase2.d80.ground_covariance_mahalanobis_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d80_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": record_sha,
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
    (output / "D80_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
