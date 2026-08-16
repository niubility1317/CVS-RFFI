#!/usr/bin/env python3
"""Run the frozen D92 QIC K10 three-scene G0 technical check.

The entry invokes the existing E0D evaluator once per arm and reads only the
persisted ``after/fit_audit.json`` receipts.  It has no truth, scorer, or
query-data input surface.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_e0d_query_evaluation as e0d  # noqa: E402


G0_SCHEMA = "cvs.phase2.d92_qic.truth_free_g0_validation.v1"
G0_MARKER = "D92_QIC_G0_ACTIVE_QUANTIZATION_INTERCEPT_CLOSURE_RESOURCE_PASS"
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
G0_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
REFERENCE_ARM = "E0_FULL_ONLY"
CANDIDATE_ARM = "E0_FULL_D42_QUANTIZATION_INTERCEPT_CLOSURE"
ALLOWED_ARMS = (REFERENCE_ARM, CANDIDATE_ARM)
WALL_LIMIT_NS = 150_000_000
WALL_RATIO_LIMIT = 1.50
PEAK_LIMIT_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_QUERY_SUFFIXES = (
    "fit_access",
    "update_access",
    "selection_access",
    "truth_access",
    "role_oracle_access",
    "class_quota_access",
    "global_reassignment",
)
_QIC_ZERO_COUNTS = (
    "additional_full_fit_count",
    "block_fit_count",
    "loo_fit_count",
    "fisher_scan_count",
    "candidate_scan_count",
    "requantize_call_count",
)
_QIC_BYTE_EXACT = (
    "coef1_byte_exact",
    "coef2_byte_exact",
    "scale1_byte_exact",
    "scale2_byte_exact",
    "log_diag_byte_exact",
    "coef_fp32_byte_exact",
    "intercept_fp32_byte_exact",
    "class_registry_byte_exact",
    "state_shape_byte_exact",
)
_PACKAGE_ARGUMENTS = (
    "before_enrollment_package_root",
    "before_enrollment_seal_path",
    "before_enrollment_seal_sha256",
    "before_apply_package_root",
    "before_apply_seal_path",
    "before_apply_seal_sha256",
    "after_enrollment_package_root",
    "after_enrollment_seal_path",
    "after_enrollment_seal_sha256",
    "after_apply_package_root",
    "after_apply_seal_path",
    "after_apply_seal_sha256",
    "ground_component_dir",
    "ground_manifest_sha256",
)


class D92QICG0Error(ValueError):
    """Raised when a persisted G0 receipt is malformed."""


class D92QICG0EntryError(RuntimeError):
    """Raised when the immutable G0 entry boundary would be violated."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise D92QICG0Error(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92QICG0Error(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise D92QICG0Error(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    result = _finite(value, name)
    if result != float(int(result)):
        raise D92QICG0Error(f"{name} must be an integer")
    return int(result)


def _integer_gate(value: Any, name: str, *, expected: int | None = None) -> bool:
    try:
        result = _integer(value, name)
    except D92QICG0Error:
        return False
    return result == expected if expected is not None else result > 0


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise D92QICG0Error(f"{name} must be a lowercase SHA256")
    return value


def _require_new_root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise D92QICG0EntryError(f"{label} already exists; refusing overwrite: {path}")
    return path


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise D92QICG0EntryError(
            f"G0 validation already exists; refusing overwrite: {path}"
        ) from error


def _call_kwargs(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    result = {name: getattr(args, name) for name in _PACKAGE_ARGUMENTS}
    result.update({"output_root": output_root, "device": args.device})
    return result


def _run_arm(args: argparse.Namespace, *, arm_id: str, output_root: Path) -> Mapping[str, Any]:
    """Execute one arm; the persisted fit audit is the only receipt source."""

    return e0d.run_d92_e0d_query_evaluation(
        arm_id=arm_id,
        **_call_kwargs(args, output_root),
    )


def _rows_from_fit_audit(result: Mapping[str, Any], output_root: Path) -> dict[str, Mapping[str, Any]]:
    del result  # Deliberately require the persisted after-state artifact.
    audit_path = output_root / "after" / "fit_audit.json"
    try:
        with audit_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise D92QICG0Error(f"G0 fit audit cannot be read: {audit_path}") from error
    if not isinstance(rows, list) or len(rows) != len(G0_SCENES):
        raise D92QICG0Error("G0 fit audit does not contain three scenes")
    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            raise D92QICG0Error("G0 fit audit row is malformed")
        scene = str(row["scenario"])
        if scene in by_scene:
            raise D92QICG0Error(f"duplicate G0 fit scene: {scene}")
        by_scene[scene] = row
    if set(by_scene) != set(G0_SCENES):
        raise D92QICG0Error("G0 scene set drift")
    return by_scene


def _query_zero(row: Mapping[str, Any], prefix: str = "") -> bool:
    return all(row.get(f"{prefix}query_{suffix}") is False for suffix in _QUERY_SUFFIXES)


def _candidate_scene_gates(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    gates["candidate_active"] = candidate.get("d92_e0d_qic_active") is True
    gates["candidate_nonfallback"] = (
        candidate.get("d92_e0d_qic_fallback_active") is False
        and candidate.get("d92_e0d_qic_fallback_reason") is None
    )
    e0_sha = _sha(candidate.get("d92_e0d_qic_e0_state_sha256"), "qic e0 state sha")
    final_sha = _sha(candidate.get("d92_e0d_qic_final_state_sha256"), "qic final state sha")
    gates["state_transition"] = final_sha != e0_sha
    changed = _integer(
        candidate.get("d92_e0d_qic_intercept_fp16_bit_change_count"),
        "qic intercept bit change",
    )
    candidate_changed = _integer(
        candidate.get("d92_e0d_qic_candidate_intercept_fp16_bit_change_count"),
        "qic candidate intercept bit change",
    )
    gates["intercept_bit_change"] = changed > 0 and candidate_changed == changed
    e0_residual = _finite(candidate.get("d92_e0d_qic_e0_residual_l1"), "qic e0 residual")
    candidate_residual = _finite(
        candidate.get("d92_e0d_qic_candidate_residual_l1"), "qic candidate residual"
    )
    reduction = _finite(
        candidate.get("d92_e0d_qic_residual_reduction_l1"), "qic residual reduction"
    )
    gates["strict_residual_reduction"] = (
        candidate_residual < e0_residual
        and reduction > 0.0
        and reduction == e0_residual - candidate_residual
    )
    gates["intercept_only"] = (
        candidate.get("d92_e0d_qic_modified_state_field_names") == ["intercept_fp16"]
        and candidate.get("d92_e0d_qic_intercept_byte_exact") is False
        and all(candidate.get(f"d92_e0d_qic_{name}") is True for name in _QIC_BYTE_EXACT)
    )
    gates["one_decode"] = (
        _integer(candidate.get("d92_e0d_qic_coefficient_decode_count"), "qic decode count") == 1
    )
    gates["no_extra_fit_or_scan"] = all(
        _integer(candidate.get(f"d92_e0d_qic_{name}"), f"qic {name}") == 0
        for name in _QIC_ZERO_COUNTS
    )
    inventory = candidate.get("after_actual_component_inventory")
    gates["after_actual_full_once"] = (
        isinstance(inventory, Mapping)
        and _integer_gate(
            inventory.get("actual_component_fit_count"),
            "actual component fit count",
            expected=1,
        )
        and candidate.get("after_registered_d_mode_effective") == "full_only"
    )
    gates["base_query_zero"] = _query_zero(candidate)
    gates["qic_query_zero"] = _query_zero(candidate, "d92_e0d_qic_")
    gates["support_only"] = (
        candidate.get("d92_e0d_qic_support_only") is True
        and candidate.get("d92_e0d_qic_clean_sample_access") is False
        and candidate.get("d92_e0d_qic_source_sample_access") is False
    )
    gates["state_delta_zero"] = (
        _integer(candidate.get("d92_e0d_qic_persistent_state_bytes_delta"), "qic state delta") == 0
    )
    gates["query_macs_delta_zero"] = (
        _integer(candidate.get("d92_e0d_qic_query_macs_delta"), "qic query mac delta") == 0
    )
    resource = candidate.get("after_registration_resource")
    reference_resource = reference.get("after_registration_resource")
    if not isinstance(resource, Mapping) or not isinstance(reference_resource, Mapping):
        raise D92QICG0Error("registration resource receipt is missing")
    candidate_wall = _finite(resource.get("registration_wall_time_ns"), "candidate wall")
    reference_wall = _finite(reference_resource.get("registration_wall_time_ns"), "reference wall")
    candidate_peak = _integer(
        resource.get("registration_incremental_peak_working_set_bytes"),
        "candidate registration peak",
    )
    gates["wall_limit"] = 0.0 <= candidate_wall <= WALL_LIMIT_NS
    gates["wall_ratio_limit"] = reference_wall > 0.0 and candidate_wall / reference_wall <= WALL_RATIO_LIMIT
    gates["registration_peak_limit"] = 0 <= candidate_peak <= PEAK_LIMIT_BYTES
    return gates


def _scene_validation(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    gates = _candidate_scene_gates(reference, candidate)
    gates["reference_query_zero"] = _query_zero(reference)
    for label, row in (("reference", reference), ("candidate", candidate)):
        gates[f"{label}_k_and_class_shape"] = all(
            _integer_gate(row.get(field), f"{label} {field}", expected=expected)
            for field, expected in (
                ("k_shot", 10),
                ("old_class_count", 6),
                ("registered_class_count", 11),
            )
        )
    gates["arm_identity"] = (
        reference.get("arm_id") == REFERENCE_ARM and candidate.get("arm_id") == CANDIDATE_ARM
    )
    reference_query_macs = _integer_gate(
        reference.get("query_macs"), "reference query macs", expected=11 * 288
    )
    candidate_query_macs = _integer_gate(
        candidate.get("query_macs"), "candidate query macs", expected=11 * 288
    )
    gates["query_macs_exact"] = (
        reference_query_macs
        and candidate_query_macs
        and candidate.get("query_macs") == reference.get("query_macs")
    )
    reference_state_bytes = _integer_gate(
        reference.get("after_state_bytes"), "reference state bytes"
    )
    candidate_state_bytes = _integer_gate(
        candidate.get("after_state_bytes"), "candidate state bytes"
    )
    gates["persistent_state_bytes_exact"] = (
        reference_state_bytes
        and candidate_state_bytes
        and candidate.get("after_state_bytes") == reference.get("after_state_bytes")
    )
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "reference_query_macs": reference.get("query_macs"),
        "candidate_query_macs": candidate.get("query_macs"),
        "reference_state_bytes": reference.get("after_state_bytes"),
        "candidate_state_bytes": candidate.get("after_state_bytes"),
        "candidate_registration_wall_time_ns": candidate.get(
            "after_registration_resource", {}
        ).get("registration_wall_time_ns"),
        "candidate_registration_incremental_peak_working_set_bytes": candidate.get(
            "after_registration_resource", {}
        ).get("registration_incremental_peak_working_set_bytes"),
    }


def validate_qic_g0(
    reference_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reference_rows) != set(G0_SCENES) or set(candidate_rows) != set(G0_SCENES):
        raise D92QICG0Error("G0 scene set drift")
    scenes = {
        scene: _scene_validation(reference_rows[scene], candidate_rows[scene])
        for scene in G0_SCENES
    }
    passed = all(value["pass"] is True for value in scenes.values())
    return {
        "schema": G0_SCHEMA,
        "marker": G0_MARKER if passed else "D92_QIC_G0_REJECTED",
        "pass": passed,
        "scenes": scenes,
        "scene_gates": {scene: value["pass"] for scene, value in scenes.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.outer_key != G0_OUTER_KEY:
        raise D92QICG0EntryError(f"G0 outer key is frozen: {args.outer_key}")
    if args.reference_arm != REFERENCE_ARM or args.candidate_arm != CANDIDATE_ARM:
        raise D92QICG0EntryError("G0 arm set is frozen")
    if args.reference_arm not in ALLOWED_ARMS or args.candidate_arm not in ALLOWED_ARMS:
        raise D92QICG0EntryError("G0 arm is not allowed")
    reference_root = _require_new_root(args.reference_output_root, "reference E0 output")
    candidate_root = _require_new_root(args.candidate_output_root, "candidate QIC output")
    if reference_root.resolve() == candidate_root.resolve():
        raise D92QICG0EntryError("reference and candidate outputs must differ")
    validation_path = Path(args.g0_validation_path)
    if validation_path.exists() or validation_path.is_symlink():
        raise D92QICG0EntryError(f"G0 validation already exists; refusing overwrite: {validation_path}")

    reference_result = _run_arm(args, arm_id=REFERENCE_ARM, output_root=reference_root)
    candidate_result = _run_arm(args, arm_id=CANDIDATE_ARM, output_root=candidate_root)
    validation = validate_qic_g0(
        _rows_from_fit_audit(reference_result, reference_root),
        _rows_from_fit_audit(candidate_result, candidate_root),
    )
    artifact: dict[str, Any] = {
        "schema": G0_SCHEMA,
        "status": G0_MARKER if validation["pass"] else "D92_QIC_G0_REJECTED",
        "outer_key": G0_OUTER_KEY,
        "reference_arm": REFERENCE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "reference_output_root": str(reference_root.resolve()),
        "candidate_output_root": str(candidate_root.resolve()),
        "validation": validation,
    }
    _write_json_new(validation_path, artifact)
    artifact["g0_validation_path"] = str(validation_path.resolve())
    if not validation["pass"]:
        raise D92QICG0EntryError("D92 QIC G0 validation failed")
    artifact["marker"] = G0_MARKER
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the frozen D92 QIC G0 dual execution.")
    result.add_argument("--outer-key", default=G0_OUTER_KEY, choices=(G0_OUTER_KEY,))
    result.add_argument("--reference-arm", default=REFERENCE_ARM, choices=(REFERENCE_ARM,))
    result.add_argument("--candidate-arm", default=CANDIDATE_ARM, choices=(CANDIDATE_ARM,))
    for name in _PACKAGE_ARGUMENTS:
        result.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    result.add_argument("--reference-output-root", required=True)
    result.add_argument("--candidate-output-root", required=True)
    result.add_argument("--g0-validation-path", required=True)
    result.add_argument("--device", required=True)
    return result


def main() -> int:
    try:
        artifact = run(parser().parse_args())
    except (D92QICG0EntryError, D92QICG0Error, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"D92 QIC G0 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(artifact, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
