#!/usr/bin/env python3
"""D89 v2 radius-reliability D81 Cauchy-center diagnostic probe."""

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
D81_PATH = SCRIPT_DIR / "probe_d81_ground_nuisance_cauchy_center.py"
D85_PATH = SCRIPT_DIR / "probe_d85_ground_radius_calibrated_consensus.py"
CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d89_v2_radius_cauchy_center.py"

ARM = "v2_radius_reliability_cauchy_center"
STRUCTURE = "d62_with_v2_radius_reliability_cauchy_support_center"
FORMULA = (
    "load only the immutable pending-joint-seal v2 component; reconstruct all "
    "84 domain-class cells and read all aggregated p90 radii; assign each cell "
    "fixed reliability rho_dc=s_dc/(s_dc+2*r_dc) from cross-domain cell signal "
    "and p90 cosine radius, normalize rho over domains inside each ground class, "
    "compute class-balanced residuals, discard ground class identity, subtract "
    "the fixed manifest reconstruction-RMSE-squared noise floor, "
    "retain ceil participation-ratio nuisance spectrum without a scan; inside "
    "every D62 full/block OOF fit apply the unchanged D81 one-step per-target-"
    "class Cauchy support-center translation and compile one INT8 affine head"
)


class D89ProbeError(RuntimeError):
    """Raised when D89 integration or evidence closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D89ProbeError(f"D89 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


base = _load("d89_d81_probe_scaffold", D81_PATH)
d85 = _load("d89_d85_v2_scaffold", D85_PATH)
core = _load("d89_v2_radius_cauchy_core", CORE_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d89-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-v2-component-dir", required=True, type=Path)
    parser.add_argument("--ground-v2-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument("--expected-pre-sign-content-root-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    known, runner_arguments = build_parser().parse_known_args(argv)
    output = base.d43._runner_output(runner_arguments)
    component_root = known.ground_v2_component_dir.resolve()
    if (
        Path(base.d43._argument_value(runner_arguments, "--component-dir")).resolve()
        != component_root
        or base.d43._argument_value(
            runner_arguments, "--component-manifest-sha256"
        ) != known.ground_v2_manifest_sha256
    ):
        raise D89ProbeError("D89 runner and v2 component bindings differ")

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
    reconstruction_rmse = float(resource["reconstruction_rmse"])
    basis, weights, spectrum_audit = core.radius_reliability_ground_spectrum(
        prototypes, radius, reconstruction_rmse
    )
    resource = component.resource_audit()
    statistics_macs = int(
        resource["all_residual_domain_enrollment_reconstruction_macs"]
        + radius.size * (10 * 160 + 8)
        + 160 * 160 * 8
    )
    ground_audit = dict(spectrum_audit)
    ground_audit.update({
        **{f"d81_{key}": value for key, value in spectrum_audit.items()},
        "ground_component_input_count": int(radius.size),
        "ground_int8_component_logical_state_bytes": int(
            resource["logical_deployment_state_bytes"]
        ),
        "ground_covariance_statistics_mac_upper_bound": statistics_macs,
        "ground_statistic_semantics": (
            "v2_cell_radius_reliability_ground_spectrum_for_d81_cauchy_center"
        ),
        "ground_bundle_contains_sample_radius": False,
        "ground_bundle_contains_aggregated_p90_radius": True,
        "ground_bundle_contains_sample_count": False,
        "ground_aggregated_center_access": True,
        "ground_aggregated_p90_radius_access": True,
        "ground_sample_radius_access": False,
        "ground_sample_feature_access": False,
        "ground_target_identity_mapping_access": False,
        "ground_class_score_access": False,
        "ground_component_update_access": False,
        "dense_ground_bank_persisted": False,
        "quantization_noise_floor_policy": (
            "manifest_reconstruction_rmse_squared"
        ),
        "ground_component_state": str(component.manifest["component_state"]),
        "d81_basis_transient_fp64_bytes": int(
            basis.nbytes + weights.nbytes + prototypes.nbytes + radius.nbytes
        ),
    })

    def load_ground_basis(
        component_dir: Path, manifest_sha256: str, feature_dim: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if (
            component_dir.resolve() != component_root
            or int(feature_dim) != 288
            or _sha256(component_root / d85.v2_codec.MANIFEST_NAME)
            != manifest_sha256
        ):
            raise D89ProbeError("D89 v2 ground loader drift")
        return basis, weights, ground_audit

    original_install = base._install_runner_resource_accounting

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
                raise D89ProbeError("D89 runner attempted another component")
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
                raise D89ProbeError("D89 runner manifest drift")
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

    base.ARM = ARM
    base.STRUCTURE = STRUCTURE
    base.FORMULA = FORMULA
    base.CORE_PATH = CORE_PATH
    base.core = core
    base.__file__ = str(Path(__file__).resolve())
    base.load_ground_basis = load_ground_basis
    base._install_runner_resource_accounting = install_resources_and_v2
    base.EXTRA_SOURCE_CLOSURE = {
        "d89_probe_sha256": _sha256(Path(__file__).resolve()),
        "d89_v2_codec_sha256": _sha256(d85.V2_CODEC_PATH),
        "d89_d85_scaffold_sha256": _sha256(D85_PATH),
        "d89_d81_scaffold_sha256": _sha256(D81_PATH),
    }
    base.d66.NPZ_NAME = d85.v2_codec.NPZ_NAME
    base.d66.MANIFEST_NAME = d85.v2_codec.MANIFEST_NAME
    base.d43.ARM_STRUCTURES[ARM] = STRUCTURE
    if ARM not in base.d43.ARMS:
        base.d43.ARMS = tuple((*base.d43.ARMS, ARM))

    translated = [
        "--d81-arm", ARM,
        "--ground-component-dir", str(component_root),
        "--ground-manifest-sha256", known.ground_v2_manifest_sha256,
        *runner_arguments,
    ]
    exit_code = int(base.main(translated))
    if exit_code != 0:
        return exit_code
    inherited_path = output / "D81_PROBE_METADATA.json"
    metadata = json.loads(inherited_path.read_text(encoding="utf-8"))
    metadata.update({
        "schema": "cvs.phase2.d89.v2_radius_cauchy_center_probe.v1",
        "arm": ARM,
        "formula": FORMULA,
        "ground_v2_component_only": True,
        "component_state": str(component.manifest["component_state"]),
        "outer_joint_seal_verified": False,
        "forced_nonpromotable": True,
        "d89_core_sha256": _sha256(CORE_PATH),
        "d89_probe_sha256": _sha256(Path(__file__).resolve()),
        "inherited_d81_metadata_sha256": _sha256(inherited_path),
    })
    (output / "D89_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
