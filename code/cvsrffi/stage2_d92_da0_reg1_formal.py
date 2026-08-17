"""Formal ERTB-IDR ``DA0_REG1`` prediction consumer.

This entry consumes four signed, physically isolated SOMP-H packages.  It
uses the frozen D92 ``E0_FULL_ONLY`` joint old/new registration unchanged:
``before`` is ``DA0_REG0`` and ``after`` is ``DA0_REG1``.  The only added
boundary is authorization and ordering: support packages are atomically
authorized/materialized first, the two deployed states are sealed, and only
then can apply-only query packages be materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cvsrffi.somph_predictor_bundle import (
    APPLY_ONLY,
    ENROLLMENT_ONLY,
    FORMAL_LEO_WEAK_SCENARIOS,
    finalize_somph_predictor_bundle_authority_after_materialization,
    materialize_somph_predictor_bundle_with_signed_authority,
)
from cvsrffi.stage2_d92_e0d_query_evaluation import (
    SCHEMA_BY_ARM,
    run_d92_e0d_query_evaluation,
)
from cvsrffi.stage2_diag_cosine_exploration import _write_json_new


ARM_ID = "E0_FULL_ONLY"
CANDIDATE_ID = "d92_e0d_e0_full_only"
DA0_REG0 = "DA0_REG0"
DA0_REG1 = "DA0_REG1"
RECEIPT_SCHEMA = "cvs.phase2.ertb_idr.da0_reg1.formal_prediction.v1"
STATE_LOCK_SCHEMA = "cvs.phase2.ertb_idr.da0_reg1.state_lock.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_SLOTS = (
    "before_enrollment",
    "before_apply",
    "after_enrollment",
    "after_apply",
)


class D92DA0REG1FormalError(RuntimeError):
    """Raised when a formal DA0 registration input or receipt drifts."""


@dataclass(frozen=True)
class SignedPackageInput:
    """External, sealed authorization inputs for exactly one package profile."""

    package_root: str | Path
    detached_seal_path: str | Path
    detached_seal_sha256: str
    formal_policy_path: str | Path
    formal_policy_authorization_path: str | Path
    signed_policy_authorization_envelope_path: str | Path
    signed_policy_authorization_envelope_sha256: str


def _canonical_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise D92DA0REG1FormalError(f"formal D92 {field} SHA256 drift")
    return normalized


def _regular_path(path: str | Path, *, field: str) -> Path:
    raw = Path(path)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise D92DA0REG1FormalError(f"formal D92 {field} is unavailable") from error
    if raw.is_symlink() or not resolved.is_file() and field != "package root":
        raise D92DA0REG1FormalError(f"formal D92 {field} must be a regular input")
    if field == "package root" and (resolved.is_symlink() or not resolved.is_dir()):
        raise D92DA0REG1FormalError("formal D92 package root must be a directory")
    return resolved


def _package_key(
    package_root: str | Path,
    detached_seal_path: str | Path,
    detached_seal_sha256: str,
) -> tuple[Path, Path, str]:
    return (
        _regular_path(package_root, field="package root"),
        _regular_path(detached_seal_path, field="detached seal"),
        _require_sha256(detached_seal_sha256, field="detached seal"),
    )


def _validate_input(value: SignedPackageInput, *, slot: str) -> tuple[Path, Path, str]:
    if slot not in _PACKAGE_SLOTS:
        raise D92DA0REG1FormalError("formal D92 package slot drift")
    key = _package_key(
        value.package_root, value.detached_seal_path, value.detached_seal_sha256
    )
    _regular_path(value.formal_policy_path, field=f"{slot} formal policy")
    _regular_path(
        value.formal_policy_authorization_path,
        field=f"{slot} formal policy authorization",
    )
    _regular_path(
        value.signed_policy_authorization_envelope_path,
        field=f"{slot} signed policy envelope",
    )
    _require_sha256(
        value.signed_policy_authorization_envelope_sha256,
        field=f"{slot} signed policy envelope",
    )
    return key


def _manifest_binding(
    manifest: Mapping[str, Any],
    final_audit: Mapping[str, Any],
    package: SignedPackageInput,
    *,
    expected_profile: str,
    expected_state: str,
) -> dict[str, Any]:
    if manifest.get("profile") != expected_profile:
        raise D92DA0REG1FormalError("formal D92 package profile drift")
    if manifest.get("registration_state") != expected_state:
        raise D92DA0REG1FormalError("formal D92 package registration-state drift")
    for field in (
        "package_root_sha256",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
    ):
        _require_sha256(manifest.get(field), field=field)
    registry = manifest.get("registered_classes")
    if not isinstance(registry, list) or not registry:
        raise D92DA0REG1FormalError("formal D92 class registry drift")
    handles = tuple(
        str(row.get("class_handle", "")) if isinstance(row, Mapping) else ""
        for row in registry
    )
    if not all(handles) or len(set(handles)) != len(handles):
        raise D92DA0REG1FormalError("formal D92 class-handle drift")
    registry_sha256 = hashlib.sha256(_canonical_bytes(registry)).hexdigest()
    if _require_sha256(
        final_audit.get("package_class_registry_sha256"),
        field="package class registry",
    ) != registry_sha256:
        raise D92DA0REG1FormalError("formal D92 class registry authorization drift")
    required_audit_sha = (
        "package_root_sha256",
        "package_detached_seal_sha256",
        "manifest_sha256",
        "artifact_member_allowlist_sha256",
        "formal_policy_sha256",
        "formal_policy_authorization_sha256",
        "signed_policy_authorization_envelope_sha256",
        "authority_commit_sha256",
        "code_closure_sha256",
        "post_materialization_audit_sha256",
    )
    for field in required_audit_sha:
        _require_sha256(final_audit.get(field), field=field)
    if (
        final_audit.get("package_root_sha256") != manifest.get("package_root_sha256")
        or final_audit.get("package_detached_seal_sha256")
        != _require_sha256(package.detached_seal_sha256, field="detached seal")
        or final_audit.get("signed_policy_authorization_envelope_sha256")
        != _require_sha256(
            package.signed_policy_authorization_envelope_sha256,
            field="signed policy envelope",
        )
    ):
        raise D92DA0REG1FormalError("formal D92 detached authority binding drift")
    if (
        final_audit.get("status") != "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
        or final_audit.get("control_state")
        != "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
        or final_audit.get("formal_launch_authority") is not True
        or final_audit.get("formal_metric_claim_allowed") is not False
        or final_audit.get("iq_payload_materialized") is not True
    ):
        raise D92DA0REG1FormalError("formal D92 atomic materialization audit drift")
    result: dict[str, Any] = {
        "profile": expected_profile,
        "registration_state": expected_state,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": final_audit[
            "package_detached_seal_sha256"
        ],
        "manifest_sha256": final_audit["manifest_sha256"],
        "artifact_member_allowlist_sha256": final_audit[
            "artifact_member_allowlist_sha256"
        ],
        "formal_policy_sha256": final_audit["formal_policy_sha256"],
        "formal_policy_authorization_sha256": final_audit[
            "formal_policy_authorization_sha256"
        ],
        "signed_policy_authorization_envelope_sha256": final_audit[
            "signed_policy_authorization_envelope_sha256"
        ],
        "authority_commit_sha256": final_audit["authority_commit_sha256"],
        "code_closure_sha256": final_audit["code_closure_sha256"],
        "post_materialization_audit_sha256": final_audit[
            "post_materialization_audit_sha256"
        ],
        "phase1_checkpoint_sha256": manifest["phase1_checkpoint_sha256"],
        "feature_runtime_sha256": manifest["feature_runtime_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "class_registry_sha256": registry_sha256,
        "class_handles": list(handles),
        "receiver": manifest.get("receiver"),
        "seed": manifest.get("seed"),
        "k_shot": manifest.get("k_shot"),
    }
    if expected_profile == APPLY_ONLY:
        for field in (
            "head_enrollment_binding_sha256",
            "head_capsule_sha256",
            "row_manifest_sha256",
        ):
            result[field] = _require_sha256(manifest.get(field), field=field)
        row_handle = manifest.get("row_handle")
        if not isinstance(row_handle, str) or not row_handle:
            raise D92DA0REG1FormalError("formal D92 apply row-handle drift")
        result["row_handle"] = row_handle
    return result


def _make_package_loader(
    packages: Mapping[str, SignedPackageInput],
    *,
    expected_profile: str,
    opened: dict[str, dict[str, Any]],
) -> Callable[..., tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    expected_slots = {
        "before": "before_enrollment" if expected_profile == ENROLLMENT_ONLY else "before_apply",
        "after": "after_enrollment" if expected_profile == ENROLLMENT_ONLY else "after_apply",
    }
    by_key = {
        _package_key(
            packages[slot].package_root,
            packages[slot].detached_seal_path,
            packages[slot].detached_seal_sha256,
        ): slot
        for slot in expected_slots.values()
    }

    def load(
        package_root: str | Path,
        *,
        detached_seal_path: str | Path,
        expected_seal_sha256: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
        try:
            slot = by_key[_package_key(
                package_root, detached_seal_path, expected_seal_sha256
            )]
        except KeyError as error:
            raise D92DA0REG1FormalError("formal D92 package input mismatch") from error
        if slot in opened:
            raise D92DA0REG1FormalError("formal D92 package reopened")
        package = packages[slot]
        evidence = materialize_somph_predictor_bundle_with_signed_authority(
            package.package_root,
            detached_seal_path=package.detached_seal_path,
            expected_seal_sha256=_require_sha256(
                package.detached_seal_sha256, field="detached seal"
            ),
            formal_policy_path=package.formal_policy_path,
            formal_policy_authorization_path=package.formal_policy_authorization_path,
            signed_policy_authorization_envelope_path=(
                package.signed_policy_authorization_envelope_path
            ),
            expected_signed_policy_authorization_envelope_sha256=(
                _require_sha256(
                    package.signed_policy_authorization_envelope_sha256,
                    field="signed policy envelope",
                )
            ),
        )
        final_audit = finalize_somph_predictor_bundle_authority_after_materialization(
            evidence
        )
        manifest = evidence.manifest
        expected_state = "before" if slot.startswith("before_") else "after"
        binding = _manifest_binding(
            manifest,
            final_audit,
            package,
            expected_profile=expected_profile,
            expected_state=expected_state,
        )
        opened[slot] = binding
        return (
            {
                str(scenario): dict(arrays)
                for scenario, arrays in evidence.materialized_payloads.items()
            },
            manifest,
            dict(final_audit),
        )

    return load


def _state_lock_sink(output_root: Path) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def seal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not output_root.is_dir() or output_root.is_symlink():
            raise D92DA0REG1FormalError("formal D92 state-lock output root drift")
        required = {
            "schema",
            "candidate",
            "receiver",
            "seed",
            "k_shot",
            "old_class_count",
            "registered_class_count",
            "phase1_checkpoint_sha256",
            "feature_runtime_sha256",
            "method_lock_sha256",
            "old_class_handles",
            "registered_class_handles",
            "state_fingerprints",
            "query_opened",
        }
        if set(payload) != required or payload.get("query_opened") is not False:
            raise D92DA0REG1FormalError("formal D92 support-state lock payload drift")
        if (
            payload.get("candidate") != CANDIDATE_ID
            or int(payload.get("old_class_count", -1)) != 6
            or int(payload.get("registered_class_count", -1)) <= 6
            or set(payload.get("state_fingerprints", {}))
            != set(FORMAL_LEO_WEAK_SCENARIOS)
        ):
            raise D92DA0REG1FormalError("formal D92 support-state identity drift")
        receipt = {
            "schema": STATE_LOCK_SCHEMA,
            "status": "ERTB_IDR_DA0_REG1_SUPPORT_STATE_LOCKED",
            "state_labels": {"before": DA0_REG0, "after": DA0_REG1},
            "independent_da_preadaptation_applied": False,
            "mrior_sda_preadaptation_applied": False,
            "native_joint_old_new_registration": True,
            "support_state": dict(payload),
        }
        path = output_root / "ERTB_DA0_REG1_STATE_LOCK.json"
        digest = _write_json_new(path, receipt)
        return {
            "schema": STATE_LOCK_SCHEMA,
            "state_lock_sha256": digest,
            "state_lock_member": path.name,
            "query_opened_after_state_lock": True,
        }

    return seal


def _publication(state_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "result_status": "ERTB_IDR_DA0_REG1_TRUTH_FREE_PREDICTIONS_COMPLETE",
        "receipt_status": "ERTB_IDR_DA0_REG1_FORMAL_STATE_COMPLETE",
        "claim_scope": "ertb_idr_da0_native_registration_no_independent_da_preadaptation",
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": False,
        "ground_component_status": "SEALED_MANIFEST_BOUND",
        "state_labels": {"before": DA0_REG0, "after": DA0_REG1},
        # Keep the sink-owned mapping live until D81 publishes its receipts;
        # it is populated only before apply-only package materialization.
        "state_lock": state_lock,
    }


def run_d92_da0_reg1_formal(
    *,
    before_enrollment: SignedPackageInput,
    before_apply: SignedPackageInput,
    after_enrollment: SignedPackageInput,
    after_apply: SignedPackageInput,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    """Run one formal ERTB-IDR DA0 registration artifact without MRIOR."""

    packages = {
        "before_enrollment": before_enrollment,
        "before_apply": before_apply,
        "after_enrollment": after_enrollment,
        "after_apply": after_apply,
    }
    if set(packages) != set(_PACKAGE_SLOTS):
        raise D92DA0REG1FormalError("formal D92 package collection drift")
    for slot, package in packages.items():
        _validate_input(package, slot=slot)
    output = Path(output_root)
    if output.exists():
        raise D92DA0REG1FormalError("formal D92 output already exists")
    _require_sha256(ground_manifest_sha256, field="ground manifest")
    opened: dict[str, dict[str, Any]] = {}
    support_loader = _make_package_loader(
        packages, expected_profile=ENROLLMENT_ONLY, opened=opened
    )
    query_loader = _make_package_loader(
        packages, expected_profile=APPLY_ONLY, opened=opened
    )
    state_lock_holder: dict[str, Any] = {}
    raw_sink = _state_lock_sink(output)

    def sink(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(opened) != {"before_enrollment", "after_enrollment"}:
            raise D92DA0REG1FormalError("formal D92 query opened before support lock")
        receipt = dict(raw_sink(payload))
        state_lock_holder.update(receipt)
        return receipt

    result = run_d92_e0d_query_evaluation(
        arm_id=ARM_ID,
        before_enrollment_package_root=before_enrollment.package_root,
        before_enrollment_seal_path=before_enrollment.detached_seal_path,
        before_enrollment_seal_sha256=before_enrollment.detached_seal_sha256,
        before_apply_package_root=before_apply.package_root,
        before_apply_seal_path=before_apply.detached_seal_path,
        before_apply_seal_sha256=before_apply.detached_seal_sha256,
        after_enrollment_package_root=after_enrollment.package_root,
        after_enrollment_seal_path=after_enrollment.detached_seal_path,
        after_enrollment_seal_sha256=after_enrollment.detached_seal_sha256,
        after_apply_package_root=after_apply.package_root,
        after_apply_seal_path=after_apply.detached_seal_path,
        after_apply_seal_sha256=after_apply.detached_seal_sha256,
        ground_component_dir=ground_component_dir,
        ground_manifest_sha256=ground_manifest_sha256,
        output_root=output,
        device=device,
        package_loader=support_loader,
        query_package_loader=query_loader,
        state_lock_sink=sink,
        publication=_publication(state_lock_holder),
    )
    if (
        result.get("candidate") != CANDIDATE_ID
        or result.get("arm_id") != ARM_ID
        or result.get("schema") != SCHEMA_BY_ARM[ARM_ID]
        or set(opened) != set(_PACKAGE_SLOTS)
        or not state_lock_holder
    ):
        raise D92DA0REG1FormalError("formal D92 native registration result drift")
    if result.get("state_lock") != state_lock_holder:
        raise D92DA0REG1FormalError("formal D92 state-lock receipt drift")
    states = result.get("states")
    if not isinstance(states, Mapping) or set(states) != {"before", "after"}:
        raise D92DA0REG1FormalError("formal D92 state result drift")
    commits: dict[str, str] = {}
    for state in ("before", "after"):
        row = states[state]
        if not isinstance(row, Mapping):
            raise D92DA0REG1FormalError("formal D92 state receipt drift")
        commits[state] = _require_sha256(row.get("commit_sha256"), field=f"{state} commit")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "ERTB_IDR_DA0_REG1_TRUTH_FREE_PREDICTIONS_COMPLETE",
        "arm_id": ARM_ID,
        "candidate": CANDIDATE_ID,
        "state_labels": {"before": DA0_REG0, "after": DA0_REG1},
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": False,
        "independent_da_preadaptation_applied": False,
        "mrior_sda_preadaptation_applied": False,
        "native_joint_old_new_registration": True,
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
        "ground_manifest_sha256": _require_sha256(
            ground_manifest_sha256, field="ground manifest"
        ),
        "ground_component_status": "SEALED_MANIFEST_BOUND",
        "package_authority": opened,
        "state_lock": {
            **state_lock_holder,
            "query_opened_after_state_lock": True,
        },
        "member_hashes": {
            "before_COMMIT.json": commits["before"],
            "after_COMMIT.json": commits["after"],
            "ERTB_DA0_REG1_STATE_LOCK.json": state_lock_holder["state_lock_sha256"],
        },
    }
    receipt_sha256 = _write_json_new(output / "ERTB_DA0_REG1_RECEIPT.json", receipt)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": receipt["status"],
        "candidate": CANDIDATE_ID,
        "arm_id": ARM_ID,
        "state_labels": receipt["state_labels"],
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": False,
        "output_root": str(output.resolve()),
        "receipt_sha256": receipt_sha256,
        "state_lock_sha256": state_lock_holder["state_lock_sha256"],
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }


def _package_from_args(args: argparse.Namespace, prefix: str) -> SignedPackageInput:
    normalized = prefix.replace("-", "_")
    return SignedPackageInput(
        package_root=getattr(args, f"{normalized}_package_root"),
        detached_seal_path=getattr(args, f"{normalized}_seal_path"),
        detached_seal_sha256=getattr(args, f"{normalized}_seal_sha256"),
        formal_policy_path=getattr(args, f"{normalized}_formal_policy"),
        formal_policy_authorization_path=getattr(
            args, f"{normalized}_formal_policy_authorization"
        ),
        signed_policy_authorization_envelope_path=getattr(
            args, f"{normalized}_signed_policy_authorization_envelope"
        ),
        signed_policy_authorization_envelope_sha256=getattr(
            args, f"{normalized}_signed_policy_authorization_envelope_sha256"
        ),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for prefix in _PACKAGE_SLOTS:
        cli = prefix.replace("_", "-")
        result.add_argument(f"--{cli}-package-root", required=True)
        result.add_argument(f"--{cli}-seal-path", required=True)
        result.add_argument(f"--{cli}-seal-sha256", required=True)
        result.add_argument(f"--{cli}-formal-policy", required=True)
        result.add_argument(f"--{cli}-formal-policy-authorization", required=True)
        result.add_argument(
            f"--{cli}-signed-policy-authorization-envelope", required=True
        )
        result.add_argument(
            f"--{cli}-signed-policy-authorization-envelope-sha256", required=True
        )
    result.add_argument("--ground-component-dir", required=True)
    result.add_argument("--ground-manifest-sha256", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", required=True)
    return result


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return run_d92_da0_reg1_formal(
        before_enrollment=_package_from_args(args, "before_enrollment"),
        before_apply=_package_from_args(args, "before_apply"),
        after_enrollment=_package_from_args(args, "after_enrollment"),
        after_apply=_package_from_args(args, "after_apply"),
        ground_component_dir=args.ground_component_dir,
        ground_manifest_sha256=args.ground_manifest_sha256,
        output_root=args.output_root,
        device=args.device,
    )


__all__ = [
    "ARM_ID",
    "CANDIDATE_ID",
    "DA0_REG0",
    "DA0_REG1",
    "D92DA0REG1FormalError",
    "SignedPackageInput",
    "parser",
    "run_d92_da0_reg1_formal",
    "run_from_args",
]
