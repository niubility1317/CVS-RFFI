#!/usr/bin/env python3
"""D66 support-only ground-domain reliability residual probe."""

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
D62_HELPER_PATH = SCRIPT_DIR / "probe_d62_crossfitted_fisher_row_splice.py"
SPEC = importlib.util.spec_from_file_location("d66_d62_probe_helper", D62_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D66 could not load D62 helper")
d62 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d62)
d43 = d62.d43


ARM = "ground_domain_reliability_residual"
STRUCTURE = "d62_in_shared_phase1_ground_domain_reliability_coordinates"
FORMULA = (
    "r_j=(between_j+eps)/(between_j+within_j+2eps); "
    "s_j=sqrt(1+r_j); x_prime=x*diag(s_z160,ones_fft96_rf32); "
    "fit D62 for every registered class and compile W=W_prime*diag(s)"
)
Z_DIM = 160
NPZ_NAME = "int8_domain_class_prototypes.npz"
MANIFEST_NAME = "manifest.json"
EXPECTED_MEMBERS = {
    "class_registry",
    "domain_class_mask",
    "domain_class_q",
    "domain_class_scale",
    "domain_registry",
    "feature_schema",
}
EXPECTED_FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D66ProbeError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes())


def load_ground_domain_reliability(
    component_dir: Path, expected_manifest_sha256: str, feature_dim: int
) -> tuple[np.ndarray, dict[str, Any]]:
    component_dir = component_dir.resolve()
    manifest_path = component_dir / MANIFEST_NAME
    npz_path = component_dir / NPZ_NAME
    if not manifest_path.is_file() or not npz_path.is_file():
        raise D66ProbeError("D66 ground component member missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = _sha256_bytes(manifest_bytes)
    if manifest_sha != str(expected_manifest_sha256).lower():
        raise D66ProbeError("D66 ground component manifest SHA drift")
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    required = {
        "schema": "phase1_int8_domain_class_centroids_v1",
        "feature_dim": Z_DIM,
        "feature_key": "z_id",
        "phase2_phase1_prototype_component_immutable": True,
        "phase2_phase1_prototype_update_access": False,
        "phase2_phase1_prototype_member_or_exemplar_access": False,
        "phase2_phase1_prototype_sample_reconstruction_access": False,
    }
    if any(manifest.get(name) != value for name, value in required.items()):
        raise D66ProbeError("D66 ground component policy drift")
    if set(manifest.get("member_allowlist", ())) != {NPZ_NAME}:
        raise D66ProbeError("D66 ground component allowlist drift")
    if set(manifest.get("npz_member_allowlist", ())) != EXPECTED_MEMBERS:
        raise D66ProbeError("D66 ground NPZ allowlist drift")
    npz_sha = d43._sha256(npz_path)
    if npz_sha != manifest.get("component_npz_sha256"):
        raise D66ProbeError("D66 ground component payload SHA drift")

    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_MEMBERS:
            raise D66ProbeError("D66 ground component NPZ member drift")
        q = np.asarray(payload["domain_class_q"], dtype=np.int8)
        scales = np.asarray(payload["domain_class_scale"], dtype=np.float16)
        mask = np.asarray(payload["domain_class_mask"], dtype=np.uint8)
        domains = np.asarray(payload["domain_registry"], dtype=np.int16)
        classes = np.asarray(payload["class_registry"]).astype(str)
        schema = str(np.asarray(payload["feature_schema"]).item())
    domain_count = int(manifest.get("domain_count", -1))
    class_count = int(manifest.get("class_count", -1))
    expected_shape = (domain_count, class_count, Z_DIM)
    if (
        q.shape != expected_shape
        or scales.shape != expected_shape[:2]
        or mask.shape != expected_shape[:2]
        or domains.shape != (domain_count,)
        or classes.shape != (class_count,)
        or len(np.unique(domains)) != domain_count
        or len(np.unique(classes)) != class_count
        or schema != EXPECTED_FEATURE_SCHEMA
        or int(feature_dim) < Z_DIM
    ):
        raise D66ProbeError("D66 ground component schema/shape drift")
    domain_order = np.argsort(domains, kind="stable")
    class_order = np.argsort(classes, kind="stable")
    q = q[domain_order][:, class_order]
    scales = scales[domain_order][:, class_order]
    mask = mask[domain_order][:, class_order]
    domains = domains[domain_order]
    classes = classes[class_order]
    active = mask.astype(bool)
    active_count = int(np.sum(active))
    if (
        active_count != int(manifest.get("active_domain_class_cells", -1))
        or np.any(np.sum(active, axis=0) < 2)
        or not np.isfinite(scales).all()
        or np.any(scales[active] <= 0)
    ):
        raise D66ProbeError("D66 ground component active-cell drift")

    dequantized = q.astype(np.float64) * scales.astype(np.float64)[..., None]
    weighted = np.where(active[..., None], dequantized, 0.0)
    class_counts = np.sum(active, axis=0).astype(np.float64)
    class_means = np.sum(weighted, axis=0) / class_counts[:, None]
    residual = np.where(
        active[..., None], dequantized - class_means[None, :, :], 0.0
    )
    within = np.sum(residual * residual, axis=(0, 1)) / float(active_count)
    global_mean = np.mean(class_means, axis=0)
    between = np.mean((class_means - global_mean[None, :]) ** 2, axis=0)
    epsilon = float(np.finfo(np.float32).eps)
    reliability = (between + epsilon) / (between + within + 2.0 * epsilon)
    z_scale = np.sqrt(1.0 + reliability)
    if (
        not np.isfinite(z_scale).all()
        or np.any(reliability <= 0.0)
        or np.any(reliability >= 1.0)
        or np.any(z_scale <= 1.0)
        or np.any(z_scale >= np.sqrt(2.0))
    ):
        raise D66ProbeError("D66 ground reliability numerical boundary drift")
    shared_scale = np.ones(int(feature_dim), dtype=np.float64)
    shared_scale[:Z_DIM] = z_scale
    shared_scale.setflags(write=False)
    logical_state = int(
        manifest.get("resource_audit", {}).get("logical_dense_state_bytes", -1)
    )
    if logical_state <= 0:
        raise D66ProbeError("D66 ground component resource audit missing")
    statistics_macs = int(
        4 * active_count * Z_DIM + 4 * class_count * Z_DIM + 8 * Z_DIM
    )
    audit = {
        "component_manifest_sha256": manifest_sha,
        "component_npz_sha256": npz_sha,
        "component_path": str(npz_path),
        "component_provenance_status": manifest.get("provenance_status"),
        "component_formal_phase2_eligible": bool(
            manifest.get("formal_phase2_eligible", False)
        ),
        "ground_domain_count": domain_count,
        "ground_class_count": class_count,
        "ground_active_domain_class_cells": active_count,
        "ground_class_active_cell_counts": np.sum(active, axis=0).tolist(),
        "ground_domain_reliability_min": float(np.min(reliability)),
        "ground_domain_reliability_mean": float(np.mean(reliability)),
        "ground_domain_reliability_max": float(np.max(reliability)),
        "ground_z_scale_min": float(np.min(z_scale)),
        "ground_z_scale_mean": float(np.mean(z_scale)),
        "ground_z_scale_max": float(np.max(z_scale)),
        "ground_z_scale_condition_number": float(np.max(z_scale) / np.min(z_scale)),
        "ground_z_scale_sha256": _canonical_array_sha256(z_scale.astype(np.float64)),
        "ground_class_registry_sha256": _sha256_bytes(
            json.dumps(classes.tolist(), separators=(",", ":")).encode("utf-8")
        ),
        "ground_int8_component_logical_state_bytes": logical_state,
        "ground_reliability_statistics_scalar_mac_equivalents": statistics_macs,
        "transient_dequantized_ground_bytes": int(active_count * Z_DIM * 4),
        "persistent_full_precision_ground_anchor_count": 0,
        "ground_component_update_access": False,
        "ground_component_class_order_used_for_prediction_branch": False,
        "ground_component_registry_canonical_sort": True,
    }
    return shared_scale, audit


def build_d66_fit(
    d42: Any, shared_scale: np.ndarray, ground_audit: dict[str, Any]
) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    scale = np.asarray(shared_scale, dtype=np.float64)
    if (
        scale.shape != (int(d42.FEATURE_DIM),)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise D66ProbeError("D66 shared scale drift")
    base_fit, call_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x = np.asarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(scale) or not np.isfinite(x).all():
            raise D66ProbeError("D66 support feature drift")
        scaled = x * scale[None, :]
        coefficient_scaled, intercept, base_audit = base_fit(
            scaled, labels, class_count, k_shot
        )
        compiled64 = np.asarray(coefficient_scaled, dtype=np.float64) * scale[None, :]
        compiled = compiled64.astype(np.float32)
        bias = np.asarray(intercept, dtype=np.float32)
        reference_scores = scaled @ np.asarray(coefficient_scaled, dtype=np.float64).T
        reference_scores += np.asarray(intercept, dtype=np.float64)[None, :]
        compiled_scores = x @ compiled.astype(np.float64).T + bias.astype(np.float64)[
            None, :
        ]
        error = float(np.max(np.abs(reference_scores - compiled_scores)))
        tolerance = float(
            32.0
            * np.finfo(np.float32).eps
            * max(1.0, float(np.max(np.abs(reference_scores))))
        )
        if not np.isfinite(error) or error > tolerance:
            raise D66ProbeError("D66 coefficient compilation equivalence drift")
        audit = dict(base_audit)
        audit.update(
            {
                "d66_probe_arm": ARM,
                "d66_structure": STRUCTURE,
                "d66_formula": FORMULA,
                "d66_ground_int8_component_used": True,
                "d66_ground_component_input_count": int(
                    ground_audit["ground_active_domain_class_cells"]
                ),
                "d66_ground_component_update_access": False,
                "d66_ground_z_scale_sha256": ground_audit["ground_z_scale_sha256"],
                "d66_ground_z_scale_min": ground_audit["ground_z_scale_min"],
                "d66_ground_z_scale_max": ground_audit["ground_z_scale_max"],
                "d66_shared_transform_all_registered_classes": True,
                "d66_old_new_role_specific_branch": False,
                "d66_class_id_specific_formula": False,
                "d66_scene_receiver_handle_specific_branch": False,
                "d66_uses_outer_held_or_query": False,
                "d66_query_extra_macs": 0,
                "d66_hyperparameter_count": 0,
                "d66_compiled_single_affine_state_only": True,
                "d66_compilation_max_abs_error": error,
                "d66_compilation_tolerance": tolerance,
                "d66_scaled_coefficient_fp32": np.asarray(
                    coefficient_scaled, dtype=np.float32
                ).tolist(),
                "d66_actual_compiled_coefficient_fp32": compiled.tolist(),
            }
        )
        return compiled, bias, audit

    return fit, call_records


def _install_resource_accounting(
    d42: Any, ground_audit: dict[str, Any]
) -> tuple[Any, Any]:
    original_macs, original_top = d62._install_resource_accounting(d42)
    d62_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d62_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_count = len(result.before_state.classes)
        all_count = len(result.state.classes)
        old_k = int(resource["old_k_shot"])
        new_k = int(resource["new_k_shot"])
        apply_macs = int((old_count * old_k + all_count * new_k) * Z_DIM)
        stats_macs = int(
            ground_audit["ground_reliability_statistics_scalar_mac_equivalents"]
        )
        added_macs = stats_macs + apply_macs
        component_bytes = int(
            ground_audit["ground_int8_component_logical_state_bytes"]
        )
        resource.update(
            {
                "d66_ground_int8_component_used": True,
                "d66_ground_component_input_count": int(
                    ground_audit["ground_active_domain_class_cells"]
                ),
                "d66_ground_component_update_access": False,
                "d66_ground_component_logical_state_bytes": component_bytes,
                "d66_ground_reliability_statistics_macs": stats_macs,
                "d66_shared_transform_application_macs": apply_macs,
                "d66_total_added_adaptation_macs": added_macs,
                "d66_query_extra_macs": 0,
                "d66_persistent_compiled_transform_bytes": 0,
                "d66_transient_dequantized_ground_bytes": int(
                    ground_audit["transient_dequantized_ground_bytes"]
                ),
                "d66_ground_z_scale_sha256": ground_audit["ground_z_scale_sha256"],
                "d66_single_affine_state_only": True,
                "ground_int8_component_input_count": int(
                    ground_audit["ground_active_domain_class_cells"]
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
            resource["persistent_state_bytes"] <= resource["persistent_state_cap_bytes"]
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _install_runner_resource_accounting(runner: Any) -> None:
    """Include the sealed ground component after the runner sizes the affine head."""

    original_evaluate = runner._evaluate_d42_fold

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_evaluate(*args, **kwargs)
        resource = dict(row["resource"])
        if resource.get("d66_ground_int8_component_used") is not True:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d66_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d66_compiled_affine_state_bytes": head_bytes,
                "d66_component_inclusive_persistent_state_bytes": total_bytes,
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
        if not isinstance(resource, dict) or "d66_total_added_adaptation_macs" not in resource:
            continue
        resource["estimated_adaptation_macs"] -= int(
            resource["d66_total_added_adaptation_macs"]
        )
        resource["persistent_state_bytes"] -= int(
            resource["d66_ground_component_logical_state_bytes"]
        )
        resource["persistent_state_cap_pass"] = bool(
            resource["persistent_state_bytes"] <= resource["persistent_state_cap_bytes"]
        )
        resource["ground_int8_component_input_count"] = 0
    base = d62._verify_rows(sanitized)
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    scale_sha = ground_audit["ground_z_scale_sha256"]
    max_error = 0.0
    for row in target:
        resource = row["resource"]
        if (
            resource.get("d66_ground_int8_component_used") is not True
            or int(resource.get("d66_ground_component_input_count", -1))
            != int(ground_audit["ground_active_domain_class_cells"])
            or resource.get("d66_ground_component_update_access") is not False
            or resource.get("d66_ground_z_scale_sha256") != scale_sha
            or int(resource.get("d66_query_extra_macs", -1)) != 0
            or resource.get("d66_single_affine_state_only") is not True
            or not bool(resource.get("persistent_state_cap_pass"))
            or int(resource.get("persistent_state_bytes", -1))
            != int(resource.get("d66_compiled_affine_state_bytes", -2))
            + int(resource.get("d66_ground_component_logical_state_bytes", -3))
            or int(resource.get("d66_component_inclusive_persistent_state_bytes", -1))
            != int(resource.get("persistent_state_bytes", -2))
        ):
            raise D66ProbeError("D66 resource closure drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            if any(
                audit.get(name) != expected
                for name, expected in {
                    "d66_probe_arm": ARM,
                    "d66_structure": STRUCTURE,
                    "d66_formula": FORMULA,
                    "d66_ground_int8_component_used": True,
                    "d66_ground_component_update_access": False,
                    "d66_ground_z_scale_sha256": scale_sha,
                    "d66_shared_transform_all_registered_classes": True,
                    "d66_old_new_role_specific_branch": False,
                    "d66_class_id_specific_formula": False,
                    "d66_scene_receiver_handle_specific_branch": False,
                    "d66_uses_outer_held_or_query": False,
                    "d66_hyperparameter_count": 0,
                    "d66_compiled_single_affine_state_only": True,
                }.items()
            ):
                raise D66ProbeError("D66 exact audit drift")
            error = float(audit["d66_compilation_max_abs_error"])
            tolerance = float(audit["d66_compilation_tolerance"])
            if not np.isfinite(error) or error > tolerance:
                raise D66ProbeError("D66 compilation audit drift")
            max_error = max(max_error, error)
    return {
        **base,
        "verified_d66_target_row_count": len(target),
        "verified_d66_fit_audit_count": 2 * len(target),
        "verified_d66_compilation_max_abs_error": max_error,
    }


def _verify_output(
    output: Path,
    helper_hashes: dict[str, str],
    ground_audit: dict[str, Any],
) -> dict[str, Any]:
    receipt = d43._read_json(output / "RECEIPT.json")
    if int(receipt.get("training_log_row_count", -1)) != 105:
        raise D66ProbeError("D66 receipt row closure drift")
    support = d43._read_json(output / "support_audit.json")
    lock = support.get("candidate_lock", {})
    probe_lock = lock.get("d43_probe_lock", {})
    if (
        probe_lock.get("arm") != ARM
        or probe_lock.get("formal_candidate") is not False
        or probe_lock.get("forced_nonpromotable") is not True
        or probe_lock.get("selected_only_full_k10_refit_allowed") is not False
    ):
        raise D66ProbeError("D66 diagnostic lock drift")
    closure = lock.get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D66ProbeError("D66 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "base_runner_receipt_sha256": d43._sha256(output / "RECEIPT.json"),
        "verified_training_row_count": len(rows),
        "verified_query_opened": False,
        "verified_forced_nonpromotable": True,
        **_verify_rows(rows, ground_audit),
    }


def _component_arguments(arguments: list[str]) -> tuple[Path, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--component-dir", required=True, type=Path)
    parser.add_argument("--component-manifest-sha256", required=True)
    known, _ = parser.parse_known_args(arguments)
    return known.component_dir, known.component_manifest_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d66-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D66ProbeError(f"D66 output already exists: {output}")
    component_dir, component_manifest_sha = _component_arguments(runner_arguments)
    shared_scale, ground_audit = load_ground_domain_reliability(
        component_dir, component_manifest_sha, 288
    )
    entry_sha = d43._sha256(component_dir / NPZ_NAME)
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d66_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d66_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d66_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d66_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d66_locked_d42_runner", 1
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d66_fit(d42, shared_scale, ground_audit)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42, ground_audit)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D66ProbeError("D66 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        _install_runner_resource_accounting(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d66_arm,
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
        raise D66ProbeError(
            f"D66 component-fit count drift: {len(call_records)} != {expected_calls}"
        )
    exit_sha = d43._sha256(component_dir / NPZ_NAME)
    if exit_sha != entry_sha:
        raise D66ProbeError("D66 ground component changed during probe")
    evidence = _verify_output(output, helper_hashes, ground_audit)
    record_sha = _sha256_bytes(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    metadata = {
        "schema": "cvs.phase2.d66.ground_domain_reliability_residual_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d66_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": record_sha,
        "ground_component_entry_sha256": entry_sha,
        "ground_component_exit_sha256": exit_sha,
        "ground_component_bitwise_unchanged": entry_sha == exit_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        "ground_audit": ground_audit,
        **evidence,
    }
    (output / "D66_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
