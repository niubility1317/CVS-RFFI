#!/usr/bin/env python3
"""D87 support-only v2 ground-radius sigma-margin diagnostic probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
D79_PATH = SCRIPT_DIR / "probe_d79_centered_ground_tangent.py"
D85_PATH = SCRIPT_DIR / "probe_d85_ground_radius_calibrated_consensus.py"
CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d87_ground_radius_sigma_margin.py"

ARM = "ground_radius_sigma_margin_centered_head"
STRUCTURE = "d62_plus_centered_v2_radius_sigma_margin_affine_residual"
FORMULA = (
    "load only the immutable pending-joint-seal v2 component; derive 14 "
    "class-agnostic cross-class-consensus directions and fixed amplitudes "
    "sqrt(2*median_class_p90_radius); group original/plus/minus views inside "
    "each physical-rank OOF fold; minimize the all-class smooth-worst "
    "non-quadratic CE with fixed sigma weights 1/2,1/4,1/4 for 20 steps in "
    "the numerical rank-13 span; center class rows and compile delta_b="
    "-delta_W*support_mean into the unchanged single INT8 affine head"
)


class D87ProbeError(RuntimeError):
    """Raised when the D87 integration or evidence closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D87ProbeError(f"D87 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d79 = _load("d87_d79_probe_scaffold", D79_PATH)
d85 = _load("d87_d85_v2_scaffold", D85_PATH)
core = _load("d87_ground_radius_sigma_core", CORE_PATH)
d78, d43 = d79.d78, d79.d43


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_upper_bounds(
    *,
    k_shot: int,
    class_count: int,
    dimension: int,
    lda_macs: int,
    ground_statistics_macs: int,
) -> dict[str, int]:
    inherited = d78._resource_upper_bounds(
        k_shot=k_shot,
        class_count=class_count,
        dimension=dimension,
        lda_macs=lda_macs,
        ground_statistics_macs=ground_statistics_macs,
    )
    held = int(k_shot) * int(class_count)
    domains, rank = 14, 13
    # Conservative explicit plus/minus all-class CE and gradient inventory.
    sigma_macs = int(
        core.OPTIMIZER_STEPS
        * held
        * domains
        * int(class_count)
        * (8 * rank + 14)
    )
    inherited["frank_wolfe_mac_upper_bound"] += sigma_macs
    inherited["non_lda_total"] += sigma_macs
    inherited["total_added"] += sigma_macs
    return inherited


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D87ProbeError("D87 training row closure drift")
    active = 0
    objective_deltas: list[float] = []
    clean_ce_deltas: list[float] = []
    for row in target:
        geometry, resource = row["geometry_summary"], row["resource"]
        audit = geometry["d79_worstclass_margin_audit"]
        tangent = geometry["d79_ground_tangent_audit"]
        if (
            geometry["d79_probe_arm"] != ARM
            or geometry["d79_formula"] != FORMULA
            or int(audit["class_count"]) != 11
            or int(audit["k_shot"]) != 8
            or int(audit["effective_rank"]) != 13
            or int(audit["counterfactual_domain_count"]) != 14
            or int(audit["crossfit_fold_count"]) != 8
            or int(audit["crossfit_held_row_count"]) != 88
            or int(audit["optimizer_iterations"]) != 20
            or len(audit["optimizer_objective_trace"]) != 20
            or float(audit["objective_delta"]) > 1.0e-10
            or float(audit["residual_logit_at_support_center_max_abs"]) > 1.0e-8
            or audit["counterfactual_views_count_as_physical_samples"] is not False
            or audit["physical_group_crossfit_preserved"] is not True
            or audit["old_new_role_specific_branch"] is not False
            or int(audit["query_rows_used"]) != 0
            or int(tangent["effective_rank"]) != 13
            or tangent["ground_target_identity_mapping_access"] is not False
            or int(resource["d79_ground_component_logical_state_bytes"]) != 5816
            or int(resource["d79_component_inclusive_persistent_state_bytes"])
            > 256 * 1024
            or int(resource["d79_optimizer_steps"]) != 20
            or int(resource["d79_query_extra_mac_equivalents"]) != 0
            or int(resource["d79_query_extra_state_bytes"]) != 0
            or int(resource["d79_dense_query_graph_bytes"]) != 0
            or resource["d79_single_affine_state_only"] is not True
        ):
            raise D87ProbeError("D87 row/audit closure drift")
        if any(
            float(item["objective_after"])
            > float(item["objective_before"]) + 1.0e-10
            for item in audit["optimizer_objective_trace"]
        ):
            raise D87ProbeError("D87 optimizer monotonicity drift")
        active += int(audit["residual_active"])
        objective_deltas.append(float(audit["objective_delta"]))
        clean_ce_deltas.append(float(audit["oof_ce_delta_mean"]))
    return {
        **evidence,
        "verified_d87_training_row_count": len(rows),
        "verified_d87_target_row_count": len(target),
        "verified_d87_active_count": active,
        "verified_d87_objective_delta_min": min(objective_deltas),
        "verified_d87_objective_delta_max": max(objective_deltas),
        "verified_d87_clean_ce_delta_min": min(clean_ce_deltas),
        "verified_d87_clean_ce_delta_max": max(clean_ce_deltas),
        "verified_d87_query_rows_used": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d87-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-v2-component-dir", required=True, type=Path)
    parser.add_argument("--ground-v2-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument("--expected-pre-sign-content-root-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    known, runner_arguments = build_parser().parse_known_args(argv)
    output = d43._runner_output(runner_arguments)
    component_root = known.ground_v2_component_dir.resolve()
    if output.exists():
        raise D87ProbeError(f"D87 output already exists: {output}")
    if (
        Path(d43._argument_value(runner_arguments, "--component-dir")).resolve()
        != component_root
        or d43._argument_value(runner_arguments, "--component-manifest-sha256")
        != known.ground_v2_manifest_sha256
    ):
        raise D87ProbeError("D87 runner and v2 geometry components differ")

    component = d85.v2_codec.load_center_lowrank_component(
        component_root,
        expected_checkpoint_sha256=known.expected_checkpoint_sha256,
        expected_class_handle_binding_sha256=(
            known.expected_class_handle_binding_sha256
        ),
        expected_pre_sign_content_root_sha256=(
            known.expected_pre_sign_content_root_sha256
        ),
        allow_pending_outer_joint_seal_development=True,
    )
    prototypes = np.stack(
        [component.reconstruct_domain(domain) for domain in component.domain_registry]
    )
    radius = np.stack(
        [component.radius_for_domain(domain) for domain in component.domain_registry]
    )
    basis, offsets, sigma_audit = core.ground_radius_sigma_geometry(
        prototypes, radius, feature_dim=288
    )
    if basis.shape != (288, 13) or offsets.shape != (14, 288):
        raise D87ProbeError("D87 locked v2 sigma geometry drift")
    resource = component.resource_audit()
    statistics_macs = int(
        resource["all_residual_domain_enrollment_reconstruction_macs"]
        + 14 * 6 * 160 * 12
        + 14 * 160 * 8
    )

    tangent_audit = dict(sigma_audit)
    tangent_audit.update(
        {
            "preconditioner_sha256": sigma_audit["basis_sha256"],
            "tangent_rank": 13,
            "z_dimension": 160,
            "feature_dimension": 288,
            "ground_component_input_count": 84,
            "ground_class_score_access": False,
            "rank_rule": "numerical_rank_without_scan",
        }
    )
    ground_audit = {
        "ground_active_domain_class_cells": 84,
        "ground_tangent_rank": 13,
        "component_formal_phase2_eligible": False,
        "component_provenance_status": "UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        "component_state": str(component.manifest["component_state"]),
        "ground_reliability_statistics_scalar_mac_equivalents": statistics_macs,
        "ground_int8_component_logical_state_bytes": int(
            resource["logical_deployment_state_bytes"]
        ),
        "transient_dequantized_ground_bytes": int(
            prototypes.nbytes + radius.nbytes + basis.nbytes + offsets.nbytes
        ),
    }

    def load_ground_tangent(
        component_dir: Path, manifest_sha256: str, feature_dim: int
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        if (
            component_dir.resolve() != component_root
            or int(feature_dim) != 288
            or _sha256(component_root / d85.v2_codec.MANIFEST_NAME)
            != manifest_sha256
        ):
            raise D87ProbeError("D87 v2 ground loader drift")
        return basis, tangent_audit, ground_audit

    def fit_facade(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
        *,
        base_coefficient: np.ndarray,
        preconditioner: np.ndarray,
        lda_fit: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        delta_w, delta_b, audit = core.fit_ground_radius_sigma_margin(
            rows,
            labels,
            class_count,
            k_shot,
            base_coefficient=base_coefficient,
            tangent_basis=preconditioner,
            counterfactual_offsets=offsets,
            lda_fit=lda_fit,
        )
        d79.facade.set_bias_residual(delta_b)
        return delta_w, audit

    original_install = d79._install_runner_resource_accounting

    def install_resources_and_v2(runner: Any) -> None:
        original_install(runner)

        def load_component(
            component_dir: Path,
            *,
            expected_manifest_sha256: str,
            expected_checkpoint_sha256: str,
            bound_old_handles: Sequence[str],
            class_binding_path: Path,
            expected_class_binding_sha256: str,
        ) -> tuple[Any, dict[str, Any]]:
            if component_dir.resolve() != component_root:
                raise D87ProbeError("D87 runner attempted another component")
            loaded = d85.v2_codec.load_center_lowrank_component(
                component_root,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_class_handle_binding_sha256=(
                    known.expected_class_handle_binding_sha256
                ),
                expected_pre_sign_content_root_sha256=(
                    known.expected_pre_sign_content_root_sha256
                ),
                allow_pending_outer_joint_seal_development=True,
            )
            manifest = dict(loaded.manifest)
            manifest["manifest_sha256"] = _sha256(
                component_root / d85.v2_codec.MANIFEST_NAME
            )
            if manifest["manifest_sha256"] != expected_manifest_sha256:
                raise D87ProbeError("D87 runner manifest drift")
            audit = d85._binding_audit(
                loaded,
                manifest,
                bound_old_handles,
                class_binding_path,
                expected_class_binding_sha256,
            )
            return d85.V2RunnerComponentAdapter(loaded, bound_old_handles), {
                "manifest": manifest,
                "column_binding": audit,
            }

        runner.legacy._load_component = load_component
        runner.legacy.NPZ_NAME = d85.v2_codec.NPZ_NAME

    d79.ARM = ARM
    d79.STRUCTURE = STRUCTURE
    d79.FORMULA = FORMULA
    d79.__file__ = str(Path(__file__).resolve())
    d79.CORE_PATH = CORE_PATH
    d79.facade.fit_ground_preconditioned_common_descent = fit_facade
    d79.facade.OPTIMIZER_STEPS = core.OPTIMIZER_STEPS
    d79.facade.FW_ITERATIONS = core.OPTIMIZER_STEPS
    d79._verify_output = _verify_output
    d79._install_runner_resource_accounting = install_resources_and_v2
    d78.load_ground_tangent = load_ground_tangent
    d78.d77._resource_upper_bounds = _resource_upper_bounds
    d78.d66.NPZ_NAME = d85.v2_codec.NPZ_NAME
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
    if ARM not in d43.ARMS:
        d43.ARMS = tuple((*d43.ARMS, ARM))

    translated = [
        "--d79-arm",
        ARM,
        "--ground-component-dir",
        str(component_root),
        "--ground-manifest-sha256",
        known.ground_v2_manifest_sha256,
        *runner_arguments,
    ]
    exit_code = int(d79.main(translated))
    if exit_code != 0:
        return exit_code
    inherited_path = output / "D79_PROBE_METADATA.json"
    metadata = json.loads(inherited_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "schema": "cvs.phase2.d87.ground_radius_sigma_margin_probe.v1",
            "arm": ARM,
            "formula": FORMULA,
            "ground_v2_component_only": True,
            "component_state": str(component.manifest["component_state"]),
            "outer_joint_seal_verified": False,
            "forced_nonpromotable": True,
            "ground_component_bitwise_unchanged": (
                _sha256(component_root / d85.v2_codec.NPZ_NAME)
                == str(component.manifest["component_npz_sha256"])
            ),
            "sigma_geometry_audit": sigma_audit,
            "d87_core_sha256": _sha256(CORE_PATH),
            "inherited_d79_metadata_sha256": _sha256(inherited_path),
        }
    )
    (output / "D87_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
