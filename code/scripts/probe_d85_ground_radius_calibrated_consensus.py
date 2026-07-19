#!/usr/bin/env python3
"""D85 support-only radius-calibrated ground-consensus diagnostic probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
BASE_PATH = SCRIPT_DIR / "probe_d84_ground_crossclass_consensus_center.py"
V2_CODEC_PATH = CODE_ROOT / "cvsrffi" / "phase1_center_lowrank_prototype_bundle.py"

ARM = "ground_radius_calibrated_consensus_center"
STRUCTURE = "d62_with_v2_radius_calibrated_ground_consensus_center_translation"
FORMULA = (
    "load only the immutable pending-joint-seal v2 center+rank3-residual+p90-radius "
    "component for development; reconstruct 14 domain-class centers transiently; "
    "derive D84 cross-class consensus domain templates; multiply each fixed geometry "
    "weight by median_ground_radius/(median_ground_radius+domain_median_radius), "
    "normalize once without a scan; use the resulting support-only Cauchy center "
    "translation for every registered target class; fit and compile the unchanged "
    "single INT8 D62 head"
)


class D85ProbeError(RuntimeError):
    """Raised when the D85 diagnostic closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D85ProbeError(f"D85 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load("d85_d84_probe_scaffold", BASE_PATH)
v2_codec = _load("d85_v2_component_codec", V2_CODEC_PATH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def radius_calibrated_consensus_templates(
    domain_class_prototypes: np.ndarray,
    domain_class_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reweight D84 consensus templates by parameter-free aggregate precision."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    radius = np.asarray(domain_class_radius, dtype=np.float64)
    if (
        prototypes.ndim != 3
        or prototypes.shape[2] != 160
        or radius.shape != prototypes.shape[:2]
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or not np.isfinite(radius).all()
        or np.any(radius <= 0.0)
    ):
        raise D85ProbeError("D85 v2 ground prototype/radius tensor drift")
    mask = np.ones(prototypes.shape[:2], dtype=np.uint8)
    templates, geometry_weight, base_audit = (
        base.core.ground_crossclass_consensus_templates(prototypes, mask)
    )
    if geometry_weight.shape != (prototypes.shape[0],):
        raise D85ProbeError(
            "D85 requires one retained consensus template per v2 ground domain"
        )
    domain_radius = np.median(radius, axis=1)
    radius_reference = float(np.median(domain_radius))
    if not np.isfinite(radius_reference) or radius_reference <= 0.0:
        raise D85ProbeError("D85 ground radius reference is invalid")
    radius_reliability = radius_reference / (radius_reference + domain_radius)
    raw_weight = geometry_weight * radius_reliability
    weight = raw_weight / np.sum(raw_weight)
    if (
        not np.isfinite(weight).all()
        or np.any(weight <= 0.0)
        or not np.isclose(np.sum(weight), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise D85ProbeError("D85 radius-calibrated weight numerical drift")
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    weight.setflags(write=False)
    audit = dict(base_audit)
    audit.update(
        {
            "schema": "cvs.phase2.d85.ground_radius_calibrated_consensus.v1",
            "template_policy": (
                "class_center_then_domain_crossclass_consensus_with_v2_p90_radius_precision"
            ),
            "weight_policy": (
                "normalized_geometry_reliability_times_median_radius_precision"
            ),
            "ground_radius_definition": (
                "p90_cosine_distance_to_phase1_domain_class_centroid"
            ),
            "ground_radius_overall_min": float(np.min(radius)),
            "ground_radius_overall_median": float(np.median(radius)),
            "ground_radius_overall_mean": float(np.mean(radius)),
            "ground_radius_overall_max": float(np.max(radius)),
            "domain_median_radius_min": float(np.min(domain_radius)),
            "domain_median_radius_reference": radius_reference,
            "domain_median_radius_max": float(np.max(domain_radius)),
            "radius_reliability_min": float(np.min(radius_reliability)),
            "radius_reliability_mean": float(np.mean(radius_reliability)),
            "radius_reliability_max": float(np.max(radius_reliability)),
            "geometry_weight_min": float(np.min(geometry_weight)),
            "geometry_weight_max": float(np.max(geometry_weight)),
            "spectral_weight_min": float(np.min(weight)),
            "spectral_weight_max": float(np.max(weight)),
            "weight_sha256": hashlib.sha256(weight.view(np.uint8)).hexdigest(),
            "radius_scan_count": 0,
            "radius_hyperparameter_count": 0,
            "ground_class_centers_discarded": True,
            "ground_class_score_access": False,
            "ground_target_identity_mapping_access": False,
            "old_new_role_specific_branch": False,
        }
    )
    return templates, weight, audit


class V2RunnerComponentAdapter:
    """Expose only the legacy read-only methods needed by the locked D42 runner."""

    def __init__(self, component: Any, bound_handles: Sequence[str]) -> None:
        self._component = component
        self.class_registry = tuple(str(value) for value in bound_handles)

    @property
    def state_bytes(self) -> int:
        return int(self._component.resource_audit()["logical_deployment_state_bytes"])

    def dequantized_class_anchors(self, class_index: int) -> np.ndarray:
        index = int(class_index)
        if index < 0 or index >= len(self.class_registry):
            raise D85ProbeError("D85 class index is out of range")
        rows = np.stack(
            [
                self._component.reconstruct_domain(domain)[index]
                for domain in self._component.domain_registry
            ]
        ).astype(np.float32)
        norm = np.linalg.norm(rows, axis=1, keepdims=True)
        if bool(np.any(norm <= 1.0e-12)):
            raise D85ProbeError("D85 reconstructed ground anchor is degenerate")
        rows = np.ascontiguousarray(rows / norm, dtype=np.float32)
        rows.setflags(write=False)
        return rows


def _binding_audit(
    component: Any,
    manifest: Mapping[str, Any],
    bound_old_handles: Sequence[str],
    class_binding_path: Path,
    expected_class_binding_sha256: str,
) -> dict[str, Any]:
    if _sha256(class_binding_path) != str(expected_class_binding_sha256):
        raise D85ProbeError("D85 ADV3B02 class binding SHA256 drift")
    binding = json.loads(class_binding_path.read_text(encoding="utf-8"))
    entries = binding.get("entries")
    evidence = binding.get("evidence")
    handles = tuple(str(value) for value in bound_old_handles)
    if (
        binding.get("schema") != "cvs.phase2.d20_adv3b02_class_binding.v2"
        or binding.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or not isinstance(entries, list)
        or not isinstance(evidence, dict)
        or [int(row.get("class_index", -1)) for row in entries]
        != list(range(len(handles)))
        or [int(row.get("direct_logit_index", -1)) for row in entries]
        != list(range(len(handles)))
        or tuple(str(row.get("registered_class_handle", "")) for row in entries)
        != handles
        or tuple(str(row.get("phase1_tx", "")) for row in entries)
        != tuple(component.class_registry)
    ):
        raise D85ProbeError("D85 ADV3B02 class binding contract drift")
    return {
        "status": "DEVELOPMENT_EXACT_V2_PHASE1_TX_TO_REGISTERED_HANDLE_BINDING",
        "formal_mapping_claim_allowed": False,
        "phase1_column_registry": list(component.class_registry),
        "phase2_bound_old_handles": list(handles),
        "phase1_to_phase2_column_index": list(range(len(handles))),
        "direct_logit_indices": [int(row["direct_logit_index"]) for row in entries],
        "direct_logit_to_class_handle_order_bound": True,
        "feature_runtime_sha256": str(evidence["feature_runtime_sha256"]),
        "direct_logit_head_state_key": str(evidence["direct_logit_head_state_key"]),
        "direct_logit_head_tensor_sha256": str(
            evidence["direct_logit_head_tensor_sha256"]
        ),
        "direct_logit_weight_row_sha256": [
            str(row["direct_logit_weight_row_sha256"]) for row in entries
        ],
        "class_binding_sha256": str(expected_class_binding_sha256),
        "strict_replay_class_mapping_required_before_formal_use": True,
        "component_manifest_sha256": str(manifest["manifest_sha256"]),
        "component_npz_sha256": str(manifest["component_npz_sha256"]),
        "component_serialized_bytes": int(manifest["serialized_component_bytes"]),
        "component_logical_state_bytes": int(
            manifest["resource_audit"]["logical_deployment_state_bytes"]
        ),
        "component_provenance_status": str(manifest["provenance_status"]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d85-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-v2-component-dir", required=True, type=Path)
    parser.add_argument("--ground-v2-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument("--expected-pre-sign-content-root-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    known, runner_arguments = build_arg_parser().parse_known_args(argv)
    base.d43._require_locked_runner_arguments(runner_arguments)
    runner_component = Path(
        base.d43._argument_value(runner_arguments, "--component-dir")
    ).resolve()
    runner_manifest_sha = base.d43._argument_value(
        runner_arguments, "--component-manifest-sha256"
    )
    component_root = known.ground_v2_component_dir.resolve()
    if runner_component != component_root or runner_manifest_sha != str(
        known.ground_v2_manifest_sha256
    ):
        raise D85ProbeError("D85 runner and ground v2 component bindings differ")

    def load_ground_basis(
        component_dir: Path, manifest_sha256: str, feature_dim: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if int(feature_dim) != 288 or component_dir.resolve() != component_root:
            raise D85ProbeError("D85 ground component input drift")
        if _sha256(component_root / v2_codec.MANIFEST_NAME) != manifest_sha256:
            raise D85ProbeError("D85 v2 manifest SHA256 drift")
        component = v2_codec.load_center_lowrank_component(
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
        templates, weights, audit = radius_calibrated_consensus_templates(
            prototypes, radius
        )
        resource = component.resource_audit()
        active_cells = int(radius.size)
        statistics_macs = int(
            resource["all_residual_domain_enrollment_reconstruction_macs"]
            + 12 * active_cells * 160
            + 8 * len(component.domain_registry) * 160
            + active_cells
        )
        combined = {f"d84_{key}": value for key, value in audit.items()}
        combined.update(
            {
                "ground_component_input_count": active_cells,
                "ground_int8_component_logical_state_bytes": int(
                    resource["logical_deployment_state_bytes"]
                ),
                "ground_bundle_contains_sample_radius": False,
                "ground_bundle_contains_aggregated_p90_radius": True,
                "ground_bundle_contains_sample_count": False,
                "ground_component_update_access": False,
                "ground_component_state": str(component.manifest["component_state"]),
                "ground_component_formal_phase2_eligible": False,
                "ground_statistic_semantics": (
                    "v2_radius_calibrated_cross_ground_class_consensus_templates"
                ),
                "ground_consensus_statistics_mac_upper_bound": statistics_macs,
                "d84_basis_transient_fp64_bytes": int(
                    templates.nbytes + weights.nbytes + radius.nbytes
                ),
            }
        )
        return templates, weights, combined

    loaded_component: dict[str, Any] = {}

    def install_v2_runner_component(runner: Any) -> None:
        original_load = runner.legacy._load_component

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
                raise D85ProbeError("D85 runner attempted another ground component")
            component = v2_codec.load_center_lowrank_component(
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
            manifest = dict(component.manifest)
            manifest["manifest_sha256"] = _sha256(
                component_root / v2_codec.MANIFEST_NAME
            )
            if manifest["manifest_sha256"] != expected_manifest_sha256:
                raise D85ProbeError("D85 runner component manifest SHA256 drift")
            audit = _binding_audit(
                component,
                manifest,
                bound_old_handles,
                class_binding_path,
                expected_class_binding_sha256,
            )
            adapter = V2RunnerComponentAdapter(component, bound_old_handles)
            loaded_component["value"] = component
            return adapter, {"manifest": manifest, "column_binding": audit}

        runner.legacy._load_component = load_component
        runner.legacy.NPZ_NAME = v2_codec.NPZ_NAME
        runner._d85_original_load_component = original_load

    original_install = base._install_runner_resource_accounting

    def install_resources_and_v2(runner: Any) -> None:
        original_install(runner)
        install_v2_runner_component(runner)

    base.ARM = ARM
    base.STRUCTURE = STRUCTURE
    base.FORMULA = FORMULA
    base.__file__ = str(Path(__file__).resolve())
    base.load_ground_basis = load_ground_basis
    base._install_runner_resource_accounting = install_resources_and_v2
    base.d66.NPZ_NAME = v2_codec.NPZ_NAME
    base.d43.ARM_STRUCTURES[ARM] = STRUCTURE
    if ARM not in base.d43.ARMS:
        base.d43.ARMS = tuple((*base.d43.ARMS, ARM))

    translated = [
        "--d84-arm",
        ARM,
        "--ground-component-dir",
        str(component_root),
        "--ground-manifest-sha256",
        str(known.ground_v2_manifest_sha256),
        *runner_arguments,
    ]
    exit_code = int(base.main(translated))
    if exit_code != 0:
        return exit_code
    output = base.d43._runner_output(runner_arguments)
    inherited_path = output / "D84_PROBE_METADATA.json"
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
    if "value" not in loaded_component:
        raise D85ProbeError("D85 runner did not load the v2 component")
    metadata = dict(inherited)
    metadata.update(
        {
            "schema": "cvs.phase2.d85.ground_radius_calibrated_consensus_probe.v1",
            "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
            "arm": ARM,
            "formula": FORMULA,
            "ground_v2_component_only": True,
            "ground_v1_component_access": False,
            "standalone_component_formal_phase2_eligible": False,
            "outer_joint_seal_verified": False,
            "forced_nonpromotable": True,
            "inherited_d84_probe_metadata_sha256": _sha256(inherited_path),
        }
    )
    (output / "D85_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
