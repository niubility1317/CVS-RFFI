#!/usr/bin/env python3
"""D86 support-only ground-radius counterfactual robust-center probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
D85_PATH = SCRIPT_DIR / "probe_d85_ground_radius_calibrated_consensus.py"
CORE_PATH = (
    CODE_ROOT / "cvsrffi" / "stage2_d86_ground_radius_counterfactual_center.py"
)

ARM = "ground_radius_counterfactual_robust_center"
STRUCTURE = "d62_with_v2_radius_bounded_counterfactual_support_center_translation"
FORMULA = (
    "load only the immutable pending-joint-seal v2 ground component; reconstruct "
    "14 domain-class centers and aggregated p90 radii; discard ground identities "
    "after deriving 14 cross-class-consensus unit domain directions; set each "
    "symmetric feature-space counterfactual amplitude to sqrt(2*median_class_p90_"
    "radius) without a scan; inside every D62 full/block support closure compute "
    "the nearest-rival margin under both signs and every domain direction; apply "
    "one class-symmetric Cauchy weight from that non-quadratic margin risk; "
    "translate only the target class-common z160 center and compile the unchanged "
    "single INT8 D62 head"
)


class D86ProbeError(RuntimeError):
    """Raised when the D86 diagnostic closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D86ProbeError(f"D86 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d85 = _load("d86_d85_probe_scaffold", D85_PATH)
core = _load("d86_ground_radius_counterfactual_core", CORE_PATH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transform_macs(class_count: int, k_shot: int, domain_count: int) -> int:
    classes, shots, domains = int(class_count), int(k_shot), int(domain_count)
    rows = classes * shots
    return int(
        6 * rows * classes * core.Z_DIM
        + 4 * classes * classes * domains * core.Z_DIM
        + 10 * rows * classes * domains
        + 5 * rows * core.Z_DIM
    )


def _d62_counterfactual_chain_macs(
    class_count: int, k_shot: int, domain_count: int
) -> int:
    classes, shots = int(class_count), int(k_shot)
    outer = _transform_macs(classes, shots, domain_count)
    if shots == 1:
        return 2 * outer
    inner = _transform_macs(classes, shots - 1, domain_count)
    if shots == 2:
        return 2 * outer + 2 * shots * inner
    return 4 * outer + 4 * shots * inner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d86-arm", required=True, choices=(ARM,))
    parser.add_argument("--ground-v2-component-dir", required=True, type=Path)
    parser.add_argument("--ground-v2-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument("--expected-pre-sign-content-root-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    known, runner_arguments = build_arg_parser().parse_known_args(argv)

    def chain_macs(class_count: int, k_shot: int, rank: int) -> int:
        return _d62_counterfactual_chain_macs(class_count, k_shot, rank)

    d85.ARM = ARM
    d85.STRUCTURE = STRUCTURE
    d85.FORMULA = FORMULA
    d85.__file__ = str(Path(__file__).resolve())
    d85.radius_calibrated_consensus_templates = (
        core.ground_radius_counterfactual_templates
    )
    d85.base.core.translate_to_consensus_robust_centers = (
        core.translate_to_counterfactual_robust_centers
    )
    d85.base.core.build_consensus_center_component_fit = (
        core.build_counterfactual_center_component_fit
    )
    d85.base._d62_translation_chain_macs = chain_macs
    d85.base.d43.ARM_STRUCTURES[ARM] = STRUCTURE
    if ARM not in d85.base.d43.ARMS:
        d85.base.d43.ARMS = tuple((*d85.base.d43.ARMS, ARM))

    translated = [
        "--d85-arm",
        ARM,
        "--ground-v2-component-dir",
        str(known.ground_v2_component_dir),
        "--ground-v2-manifest-sha256",
        str(known.ground_v2_manifest_sha256),
        "--expected-checkpoint-sha256",
        str(known.expected_checkpoint_sha256),
        "--expected-class-handle-binding-sha256",
        str(known.expected_class_handle_binding_sha256),
        "--expected-pre-sign-content-root-sha256",
        str(known.expected_pre_sign_content_root_sha256),
        *runner_arguments,
    ]
    exit_code = int(d85.main(translated))
    if exit_code != 0:
        return exit_code
    output = d85.base.d43._runner_output(runner_arguments)
    inherited_path = output / "D85_PROBE_METADATA.json"
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
    metadata = dict(inherited)
    metadata.update(
        {
            "schema": "cvs.phase2.d86.ground_radius_counterfactual_probe.v1",
            "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
            "arm": ARM,
            "formula": FORMULA,
            "counterfactual_loss_family": (
                "nearest_rival_softplus_under_symmetric_radius_sigma_points"
            ),
            "counterfactual_views_count_as_physical_samples": False,
            "ground_class_centers_used_for_target_scores": False,
            "ground_v2_component_only": True,
            "standalone_component_formal_phase2_eligible": False,
            "outer_joint_seal_verified": False,
            "forced_nonpromotable": True,
            "inherited_d85_probe_metadata_sha256": _sha256(inherited_path),
            "d86_core_sha256": _sha256(CORE_PATH),
        }
    )
    (output / "D86_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
