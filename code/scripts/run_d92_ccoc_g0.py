#!/usr/bin/env python3
"""Run the frozen D92 CCOC G0 dual execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d92_e0d_query_evaluation as e0d  # noqa: E402
from cvsrffi.stage2_d92_ccoc_g0 import (  # noqa: E402
    ALLOWED_ARMS,
    CANDIDATE_ARM,
    D92CCOCG0Error,
    G0_MARKER,
    G0_OUTER_KEY,
    G0_SCHEMA,
    G0_SCENES,
    REFERENCE_ARM,
    receipt_sha256,
    validate_ccoc_g0,
)


class D92CCOCG0EntryError(RuntimeError):
    """Raised when the immutable G0 entry boundary would be violated."""


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


def _require_new_root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise D92CCOCG0EntryError(
            f"{label} already exists; refusing overwrite: {path}"
        )
    return path


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise D92CCOCG0EntryError(
            f"G0 validation already exists; refusing overwrite: {path}"
        ) from error
    return receipt_sha256(value)


def _call_kwargs(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    result = {
        name: getattr(args, name)
        for name in _PACKAGE_ARGUMENTS
    }
    result.update({"output_root": output_root, "device": args.device})
    return result


def _run_arm(
    args: argparse.Namespace,
    *,
    arm_id: str,
    output_root: Path,
    receipts: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one frozen arm and collect only the support receipt callback."""

    original_allowed = e0d._CCOC_ARM_IDS
    original_ccoc_support_receipt = e0d._ccoc_support_receipt
    # Task2 exposes the callback on the CCOC path.  The reference execution
    # uses the same callback solely to expose its final D42 support receipt;
    # no fit or prediction path is changed by this temporary admission.
    if arm_id == REFERENCE_ARM:
        e0d._CCOC_ARM_IDS = frozenset(set(original_allowed) | {REFERENCE_ARM})

        def reference_ccoc_support_receipt(*call_args: Any, **call_kwargs: Any):
            arm = call_kwargs.get("arm")
            if getattr(arm, "arm_id", None) == REFERENCE_ARM:
                return {}
            return original_ccoc_support_receipt(*call_args, **call_kwargs)

        e0d._ccoc_support_receipt = reference_ccoc_support_receipt
    try:
        return e0d.run_d92_e0d_query_evaluation(
            arm_id=arm_id,
            **_call_kwargs(args, output_root),
            technical_support_receipt_sink=receipts.append,
        )
    finally:
        e0d._CCOC_ARM_IDS = original_allowed
        e0d._ccoc_support_receipt = original_ccoc_support_receipt


def _rows_from_result(
    result: Mapping[str, Any],
    receipts: list[Mapping[str, Any]],
    *,
    arm_id: str,
    output_root: Path,
) -> dict[str, Mapping[str, Any]]:
    fit_audit = result.get("fit_audit")
    if not isinstance(fit_audit, list):
        audit_path = output_root / "after" / "fit_audit.json"
        try:
            with audit_path.open("r", encoding="utf-8") as handle:
                fit_audit = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise D92CCOCG0EntryError(
                f"G0 fit audit cannot be read: {audit_path}"
            ) from error
    if not isinstance(fit_audit, list) or len(fit_audit) != len(G0_SCENES):
        raise D92CCOCG0EntryError("G0 fit audit does not contain three scenes")
    if len(receipts) != len(G0_SCENES):
        raise D92CCOCG0EntryError("G0 support receipt does not contain three scenes")
    by_scene: dict[str, Mapping[str, Any]] = {}
    receipt_by_scene: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or "scene" not in receipt:
            raise D92CCOCG0EntryError("G0 support receipt row is malformed")
        scene = str(receipt["scene"])
        if scene in receipt_by_scene:
            raise D92CCOCG0EntryError(f"duplicate G0 support scene: {scene}")
        receipt_by_scene[scene] = receipt
    for audit in fit_audit:
        if not isinstance(audit, Mapping) or "scenario" not in audit:
            raise D92CCOCG0EntryError("G0 fit audit row is malformed")
        scene = str(audit["scenario"])
        if scene in by_scene:
            raise D92CCOCG0EntryError(f"duplicate G0 fit scene: {scene}")
        receipt = receipt_by_scene.get(scene)
        if receipt is None:
            raise D92CCOCG0EntryError(f"missing G0 support receipt: {scene}")
        row = {**dict(receipt), **dict(audit)}
        inventory = audit.get("after_actual_component_inventory")
        if isinstance(inventory, Mapping):
            row["actual_full_fit_count"] = inventory.get(
                "actual_component_fit_count"
            )
        row["persistent_state_bytes"] = audit.get("after_state_bytes")
        row["query_macs"] = audit.get("query_macs")
        resource = audit.get("after_registration_resource")
        if isinstance(resource, Mapping):
            row.update(resource)
        if arm_id == CANDIDATE_ARM:
            row["active"] = audit.get("d92_e0d_ccoc_active")
            row["fallback_active"] = audit.get(
                "d92_e0d_ccoc_fallback_active"
            )
            row["old_rho"] = audit.get("d92_e0d_ccoc_old_rho")
            row["new_rho"] = audit.get("d92_e0d_ccoc_new_rho")
        else:
            row["active"] = True
            row["fallback_active"] = False
        by_scene[scene] = row
    if set(by_scene) != set(G0_SCENES):
        raise D92CCOCG0EntryError("G0 scene set drift")
    return by_scene


def run(args: argparse.Namespace) -> dict[str, Any]:
    outer_key = getattr(args, "outer_key", G0_OUTER_KEY)
    if outer_key != G0_OUTER_KEY:
        raise D92CCOCG0EntryError(f"G0 outer key is frozen: {outer_key}")
    reference_arm = getattr(args, "reference_arm", REFERENCE_ARM)
    candidate_arm = getattr(args, "candidate_arm", CANDIDATE_ARM)
    if reference_arm != REFERENCE_ARM or candidate_arm != CANDIDATE_ARM:
        raise D92CCOCG0EntryError("G0 arm set is frozen")
    if reference_arm not in ALLOWED_ARMS or candidate_arm not in ALLOWED_ARMS:
        raise D92CCOCG0EntryError("G0 arm is not allowed")
    reference_root = _require_new_root(
        args.reference_output_root,
        "reference E0 output",
    )
    candidate_root = _require_new_root(
        args.candidate_output_root,
        "candidate CCOC output",
    )
    if reference_root.resolve() == candidate_root.resolve():
        raise D92CCOCG0EntryError("reference and candidate outputs must differ")
    validation_path_value = getattr(args, "g0_validation_path", None)
    validation_path = (
        Path(validation_path_value)
        if validation_path_value
        else candidate_root.parent / "g0_validation.json"
    )
    if validation_path.exists() or validation_path.is_symlink():
        raise D92CCOCG0EntryError(
            f"G0 validation already exists; refusing overwrite: {validation_path}"
        )

    reference_receipts: list[Mapping[str, Any]] = []
    candidate_receipts: list[Mapping[str, Any]] = []
    reference_result = _run_arm(
        args,
        arm_id=reference_arm,
        output_root=reference_root,
        receipts=reference_receipts,
    )
    candidate_result = _run_arm(
        args,
        arm_id=candidate_arm,
        output_root=candidate_root,
        receipts=candidate_receipts,
    )
    reference_rows = _rows_from_result(
        reference_result,
        reference_receipts,
        arm_id=reference_arm,
        output_root=reference_root,
    )
    candidate_rows = _rows_from_result(
        candidate_result,
        candidate_receipts,
        arm_id=candidate_arm,
        output_root=candidate_root,
    )
    validation = validate_ccoc_g0(
        {"scenes": reference_rows},
        {"scenes": candidate_rows},
    )
    artifact: dict[str, Any] = {
        "schema": G0_SCHEMA,
        "status": (
            G0_MARKER if validation["pass"] else "D92_CCOC_G0_REJECTED"
        ),
        "outer_key": G0_OUTER_KEY,
        "reference_arm": reference_arm,
        "candidate_arm": candidate_arm,
        "reference_output_root": str(reference_root.resolve()),
        "candidate_output_root": str(candidate_root.resolve()),
        "validation": validation,
    }
    validation_sha256 = _write_json_new(validation_path, artifact)
    artifact["g0_validation_path"] = str(validation_path.resolve())
    artifact["g0_validation_sha256"] = validation_sha256
    if not validation["pass"]:
        raise D92CCOCG0EntryError("D92 CCOC G0 validation failed")
    artifact["marker"] = G0_MARKER
    return artifact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the frozen D92 CCOC G0 dual execution."
    )
    result.add_argument(
        "--outer-key",
        default=G0_OUTER_KEY,
        choices=(G0_OUTER_KEY,),
    )
    result.add_argument(
        "--reference-arm",
        default=REFERENCE_ARM,
        choices=(REFERENCE_ARM,),
    )
    result.add_argument(
        "--candidate-arm",
        default=CANDIDATE_ARM,
        choices=(CANDIDATE_ARM,),
    )
    for name in _PACKAGE_ARGUMENTS:
        option = "--" + name.replace("_", "-")
        result.add_argument(option, dest=name, required=True)
    result.add_argument("--reference-output-root", required=True)
    result.add_argument("--candidate-output-root", required=True)
    result.add_argument("--g0-validation-path", default=None)
    result.add_argument("--device", required=True)
    return result


def main() -> int:
    try:
        artifact = run(parser().parse_args())
    except (D92CCOCG0EntryError, D92CCOCG0Error, ValueError) as error:
        print(f"D92 CCOC G0 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(artifact, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
