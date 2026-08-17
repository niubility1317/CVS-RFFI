#!/usr/bin/env python3
"""Run the frozen D92 AFCP K10 three-scene truth-free G0 check.

The runner owns only immutable execution boundaries and persisted receipt
validation.  AFCP scientific receipt keys are supplied by the scientific
implementation owner before the validation path is enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_e0d_query_evaluation as e0d  # noqa: E402


G0_SCHEMA = "cvs.phase2.d92_afcp.truth_free_g0_validation.v1"
G0_MARKER = "D92_AFCP_G0_ACTIVE_RESOURCE_PASS"
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
G0_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
G0_RECEIVER = "7-7"
G0_SEED = 713106
REFERENCE_ARM = "E0_FULL_ONLY"
CANDIDATE_ARM = "E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE"
ALLOWED_ARMS = (REFERENCE_ARM, CANDIDATE_ARM)
G0_TECHNICAL_FAILURE = "D92_AFCP_G0_TECHNICAL_FAILURE"
WALL_LIMIT_NS = 150_000_000
PEAK_LIMIT_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_ROW_HANDLE_RE = re.compile(r"row_[0-9a-f]{64}\Z")
_QUERY_SUFFIXES = (
    "fit_access",
    "update_access",
    "selection_access",
    "truth_access",
    "role_oracle_access",
    "class_quota_access",
    "global_reassignment",
)
_AFCP_ZERO_COUNTS = (
    "requantize_call_count",
    "additional_full_fit_count",
    "block_fit_count",
    "loo_fit_count",
    "fisher_fit_count",
    "tail_selection_count",
    "rival_pair_selection_count",
    "atomic_candidate_count",
    "prefix_evaluation_count",
    "candidate_scan_count",
    "support_288_square_matrix_bytes",
)
_AFCP_BYTE_EXACT = (
    "coef1_byte_exact",
    "scale1_byte_exact",
    "scale2_byte_exact",
    "intercept_byte_exact",
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


class D92AFCPG0EntryError(RuntimeError):
    """Raised when the frozen AFCP G0 entry boundary is violated."""


class D92AFCPG0Error(ValueError):
    """Raised when an AFCP G0 receipt is malformed."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise D92AFCPG0Error(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92AFCPG0Error(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise D92AFCPG0Error(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    result = _finite(value, name)
    if result != float(int(result)):
        raise D92AFCPG0Error(f"{name} must be an integer")
    return int(result)


def _integer_gate(value: Any, name: str, *, expected: int | None = None) -> bool:
    try:
        result = _integer(value, name)
    except D92AFCPG0Error:
        return False
    return result == expected if expected is not None else result > 0


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise D92AFCPG0Error(f"{name} must be a lowercase SHA256")
    return value


def _require_new_root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise D92AFCPG0EntryError(f"{label} already exists; refusing overwrite: {path}")
    return path


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise D92AFCPG0EntryError(
            f"G0 validation already exists; refusing overwrite: {path}"
        ) from error


def _call_kwargs(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    result = {name: getattr(args, name) for name in _PACKAGE_ARGUMENTS}
    result.update({"output_root": output_root, "device": args.device})
    return result


def _run_arm(args: argparse.Namespace, *, arm_id: str, output_root: Path) -> Mapping[str, Any]:
    """Execute one frozen arm; receipt validation reads persisted artifacts only."""

    return e0d.run_d92_e0d_query_evaluation(
        arm_id=arm_id,
        **_call_kwargs(args, output_root),
    )


def _rows_from_fit_audit(result: Mapping[str, Any], output_root: Path) -> dict[str, Mapping[str, Any]]:
    del result
    audit_path = output_root / "after" / "fit_audit.json"
    try:
        with audit_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise D92AFCPG0Error(f"G0 fit audit cannot be read: {audit_path}") from error
    if not isinstance(rows, list) or len(rows) != len(G0_SCENES):
        raise D92AFCPG0Error("G0 fit audit does not contain three scenes")
    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("scenario"), str):
            raise D92AFCPG0Error("G0 fit audit row is malformed")
        scene = str(row["scenario"])
        if scene in by_scene:
            raise D92AFCPG0Error(f"duplicate G0 fit scene: {scene}")
        by_scene[scene] = row
    if set(by_scene) != set(G0_SCENES):
        raise D92AFCPG0Error("G0 scene set drift")
    return by_scene


def _execution_receipt_from_output(output_root: Path) -> Mapping[str, Any]:
    """Read the published after-state identity receipt, never a CLI label."""

    receipt_path = output_root / "after" / "execution_receipt.json"
    try:
        with receipt_path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise D92AFCPG0Error(
            f"G0 execution receipt cannot be read: {receipt_path}"
        ) from error
    if not isinstance(receipt, Mapping):
        raise D92AFCPG0Error("G0 execution receipt is malformed")
    return receipt


def _query_zero(row: Mapping[str, Any], prefix: str = "") -> bool:
    return all(row.get(f"{prefix}query_{suffix}") is False for suffix in _QUERY_SUFFIXES)


def _three_positive_entries(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        entries = list(value.values())
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return False
    return len(entries) == 3 and all(_integer_gate(entry, name) for entry in entries)


def _three_coordinate_entries(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        entries = list(value.values())
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return False
    try:
        return len(entries) == 3 and all(_integer(entry, name) >= 0 for entry in entries)
    except D92AFCPG0Error:
        return False


def _finite_vector(value: Any, name: str, *, length: int) -> bool:
    """Accept the AFCP core's fixed-length numeric receipt vectors only."""

    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        return all(math.isfinite(_finite(entry, name)) for entry in value)
    except D92AFCPG0Error:
        return False


def _twofold_class_margin_receipt(value: Any) -> bool:
    """Validate the core's [two folds][all eleven classes] margin receipt."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(
        _finite_vector(fold, "afcp fold class margin", length=11) for fold in value
    )


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _exact_scene_receipt(value: Any) -> bool:
    return isinstance(value, list) and tuple(value) == G0_SCENES


def _outer_binding(
    reference_receipt: Mapping[str, Any], candidate_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Close the two persisted after-state receipts over the frozen outer row."""

    def arm_gates(receipt: Mapping[str, Any], label: str) -> dict[str, bool]:
        return {
            f"{label}_registration_state": receipt.get("registration_state") == "after",
            f"{label}_receiver": receipt.get("receiver") == G0_RECEIVER,
            f"{label}_seed": _integer_gate(
                receipt.get("seed"), f"{label} execution seed", expected=G0_SEED
            ),
            f"{label}_k_shot": _integer_gate(
                receipt.get("k_shot"), f"{label} execution k", expected=10
            ),
            f"{label}_registered_class_count": _integer_gate(
                receipt.get("registered_class_count"),
                f"{label} execution class count",
                expected=11,
            ),
            f"{label}_row_handle_shape": isinstance(receipt.get("row_handle"), str)
            and _ROW_HANDLE_RE.fullmatch(str(receipt.get("row_handle"))) is not None,
            f"{label}_support_scenes": _exact_scene_receipt(
                receipt.get("support_scenarios")
            ),
            f"{label}_query_scenes": _exact_scene_receipt(
                receipt.get("query_scenarios")
            ),
        }

    gates = {
        **arm_gates(reference_receipt, "reference"),
        **arm_gates(candidate_receipt, "candidate"),
        "receiver_equal": reference_receipt.get("receiver")
        == candidate_receipt.get("receiver"),
        "seed_equal": reference_receipt.get("seed") == candidate_receipt.get("seed"),
        "k_shot_equal": reference_receipt.get("k_shot")
        == candidate_receipt.get("k_shot"),
        "registered_class_count_equal": reference_receipt.get(
            "registered_class_count"
        )
        == candidate_receipt.get("registered_class_count"),
        "row_handle_equal": reference_receipt.get("row_handle")
        == candidate_receipt.get("row_handle"),
        "support_scenes_equal": reference_receipt.get("support_scenarios")
        == candidate_receipt.get("support_scenarios"),
        "query_scenes_equal": reference_receipt.get("query_scenarios")
        == candidate_receipt.get("query_scenarios"),
    }
    return {"pass": all(gates.values()), "gates": gates}


def _candidate_scene_gates(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, bool]:
    prefix = "d92_e0d_afcp_"
    gates: dict[str, bool] = {}
    gates["candidate_active"] = candidate.get(prefix + "active") is True
    gates["candidate_nonfallback"] = (
        candidate.get(prefix + "fallback_active") is False
        and candidate.get(prefix + "fallback_reason") is None
    )
    gates["lifecycle_receipt"] = (
        _nonempty_text(candidate.get(prefix + "formula_revision"))
        and _nonempty_text(candidate.get(prefix + "state_postprocess_mode"))
        and candidate.get(prefix + "direct_state_publish") is True
        and candidate.get(prefix + "support_only") is True
    )
    gates["candidate_shape"] = all(
        _integer_gate(candidate.get(prefix + field), f"afcp {field}", expected=expected)
        for field, expected in (("class_count", 11), ("old_class_count", 6), ("k_shot", 10))
    )
    e0_sha = _sha(candidate.get(prefix + "e0_state_sha256"), "afcp e0 state sha")
    final_sha = _sha(candidate.get(prefix + "final_state_sha256"), "afcp final state sha")
    modified = candidate.get(prefix + "modified_state_field_names")
    gates["state_transition"] = (
        final_sha != e0_sha
        and candidate.get(prefix + "final_state_non_e0") is True
        and isinstance(modified, list)
        and bool(modified)
        and all(isinstance(name, str) and name for name in modified)
    )
    gates["only_code2_changes"] = (
        candidate.get(prefix + "coef2_byte_exact") is False
        and all(candidate.get(prefix + name) is True for name in _AFCP_BYTE_EXACT)
    )
    gates["state_delta_zero"] = (
        _integer(candidate.get(prefix + "persistent_state_bytes_delta"), "afcp state delta") == 0
    )
    gates["query_macs_delta_zero"] = (
        _integer(candidate.get(prefix + "query_macs_delta"), "afcp query mac delta") == 0
    )
    gates["three_real_code_changes"] = (
        _three_coordinate_entries(candidate.get(prefix + "block_coordinate_indices"), "afcp block coordinate")
        and _three_positive_entries(candidate.get(prefix + "block_changed_code2_counts"), "afcp block changed codes")
        and _integer_gate(candidate.get(prefix + "changed_code2_count"), "afcp changed code count")
        and _integer_gate(candidate.get(prefix + "state_delta_code2_l1"), "afcp state delta l1")
        and candidate.get(prefix + "all_three_blocks_changed") is True
    )
    gates["quantum_support_transition"] = (
        _finite(candidate.get(prefix + "support_margin_delta_max_abs"), "afcp support margin delta") > 0.0
        and candidate.get(prefix + "support_margin_quantum_pass") is True
    )
    gates["twofold_receipt"] = (
        _nonempty_text(candidate.get(prefix + "support_row_canonicalization"))
        and _nonempty_text(candidate.get(prefix + "fold_rule"))
        and _nonempty_text(candidate.get(prefix + "fold_tie_policy"))
        and candidate.get(prefix + "class_permutation_equivariant") is True
        and candidate.get(prefix + "row_permutation_invariant") is True
        and candidate.get(prefix + "task_swap_equivariant") is True
        and candidate.get(prefix + "all_class_symmetric") is True
        and _twofold_class_margin_receipt(
            candidate.get(prefix + "fold_class_all_margin_delta_mean")
        )
        and _finite_vector(
            candidate.get(prefix + "fold_old_to_new_cross_margin_delta_mean"),
            "afcp fold old-to-new cross margin",
            length=2,
        )
        and _finite_vector(
            candidate.get(prefix + "fold_new_to_old_cross_margin_delta_mean"),
            "afcp fold new-to-old cross margin",
            length=2,
        )
    )
    gates["twofold_guard_pass"] = (
        candidate.get(prefix + "twofold_class_guard_pass") is True
        and candidate.get(prefix + "twofold_cross_guard_pass") is True
        and candidate.get(prefix + "support_guard_pass") is True
    )
    gates["no_forbidden_mechanism"] = all(
        _integer(candidate.get(prefix + field), f"afcp {field}") == 0
        for field in _AFCP_ZERO_COUNTS
    )
    gates["base_query_zero"] = _query_zero(candidate)
    gates["afcp_query_zero"] = _query_zero(candidate, prefix)
    gates["support_and_access_boundary"] = (
        _integer(candidate.get(prefix + "query_rows_used"), "afcp query rows") == 0
        and candidate.get(prefix + "clean_sample_access") is False
        and candidate.get(prefix + "source_sample_access") is False
    )
    gates["support_resource_receipt"] = (
        _finite(candidate.get(prefix + "support_macs_upper_bound"), "afcp support macs") >= 0.0
        and _integer(candidate.get(prefix + "support_transient_bytes_upper_bound"), "afcp support bytes") >= 0
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
    resource = candidate.get("after_registration_resource")
    reference_resource = reference.get("after_registration_resource")
    if not isinstance(resource, Mapping) or not isinstance(reference_resource, Mapping):
        raise D92AFCPG0Error("registration resource receipt is missing")
    candidate_wall = _finite(resource.get("registration_wall_time_ns"), "candidate wall")
    candidate_peak = _integer(
        resource.get("registration_incremental_peak_working_set_bytes"),
        "candidate registration peak",
    )
    gates["wall_limit"] = 0.0 <= candidate_wall <= WALL_LIMIT_NS
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
    gates["query_macs_exact"] = (
        _integer_gate(reference.get("query_macs"), "reference query macs", expected=11 * 288)
        and _integer_gate(candidate.get("query_macs"), "candidate query macs", expected=11 * 288)
        and candidate.get("query_macs") == reference.get("query_macs")
    )
    gates["persistent_state_bytes_exact"] = (
        _integer_gate(reference.get("after_state_bytes"), "reference state bytes")
        and _integer_gate(candidate.get("after_state_bytes"), "candidate state bytes")
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


def validate_afcp_g0(
    reference_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    *,
    reference_execution_receipt: Mapping[str, Any],
    candidate_execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reference_rows) != set(G0_SCENES) or set(candidate_rows) != set(G0_SCENES):
        raise D92AFCPG0Error("G0 scene set drift")
    outer_binding = _outer_binding(
        reference_execution_receipt, candidate_execution_receipt
    )
    scenes = {
        scene: _scene_validation(reference_rows[scene], candidate_rows[scene])
        for scene in G0_SCENES
    }
    for scene in scenes.values():
        scene["gates"]["outer_binding"] = outer_binding["pass"]
        scene["pass"] = all(scene["gates"].values())
    passed = all(value["pass"] is True for value in scenes.values())
    return {
        "schema": G0_SCHEMA,
        "marker": G0_MARKER if passed else G0_TECHNICAL_FAILURE,
        "pass": passed,
        "outer_binding": outer_binding,
        "scenes": scenes,
        "scene_gates": {scene: value["pass"] for scene, value in scenes.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.outer_key != G0_OUTER_KEY:
        raise D92AFCPG0EntryError(f"G0 outer key is frozen: {args.outer_key}")
    if args.reference_arm != REFERENCE_ARM or args.candidate_arm != CANDIDATE_ARM:
        raise D92AFCPG0EntryError("G0 arm set is frozen")
    reference_root = _require_new_root(args.reference_output_root, "reference E0 output")
    candidate_root = _require_new_root(args.candidate_output_root, "candidate AFCP output")
    if reference_root.resolve() == candidate_root.resolve():
        raise D92AFCPG0EntryError("reference and candidate outputs must differ")
    validation_path = Path(args.g0_validation_path)
    if validation_path.exists() or validation_path.is_symlink():
        raise D92AFCPG0EntryError(f"G0 validation already exists; refusing overwrite: {validation_path}")

    reference_result = _run_arm(args, arm_id=REFERENCE_ARM, output_root=reference_root)
    candidate_result = _run_arm(args, arm_id=CANDIDATE_ARM, output_root=candidate_root)
    validation = validate_afcp_g0(
        _rows_from_fit_audit(reference_result, reference_root),
        _rows_from_fit_audit(candidate_result, candidate_root),
        reference_execution_receipt=_execution_receipt_from_output(reference_root),
        candidate_execution_receipt=_execution_receipt_from_output(candidate_root),
    )
    status = G0_MARKER if validation["pass"] else G0_TECHNICAL_FAILURE
    artifact: dict[str, Any] = {
        "schema": G0_SCHEMA,
        "status": status,
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
        raise D92AFCPG0EntryError("D92 AFCP G0 validation failed")
    artifact["marker"] = G0_MARKER
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the frozen D92 AFCP G0 dual execution.")
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
    except (D92AFCPG0EntryError, D92AFCPG0Error, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"D92 AFCP G0 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(artifact, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
