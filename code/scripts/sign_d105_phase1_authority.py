#!/usr/bin/env python3
"""Externally sign D105 authority envelopes and D102 revocation manifests.

The private Ed25519 seed is read only for this invocation.  The command refuses
to sign unless it derives the already-pinned public key, and it never copies
the seed into a bundle, report, manifest, or output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d105_cbrc import (  # noqa: E402
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
)
from cvsrffi.stage2_d105_phase1_authority import (  # noqa: E402
    AUTHORITY_SIGNATURE_DOMAIN,
    D102_REVOCATION_SIGNATURE_DOMAIN,
    D105AuthorityError,
    TARGET25_PREPARE_SIGNATURE_DOMAIN,
    build_d105_authority_envelope,
    build_d105_target25_prepare_envelope,
    canonical_bytes,
    load_independent_review_receipt,
    load_signed_d102_revocation_manifest,
    sha256_bytes,
    sign_d105_detached,
    validate_d102_revocation_manifest,
)
from cvsrffi.stage2_d105_phase1_bundle import (  # noqa: E402
    COMPONENT_STATUS,
    D105Phase1BundleError,
    load_d105_phase1_asset,
    sha256_file,
)


def _regular(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise D105AuthorityError(f"{name} must be a regular non-symlink file")
    return path


def _load_private_seed(path: Path) -> bytes:
    raw = _regular(path, "authority private seed").read_bytes()
    if len(raw) == 32:
        return raw
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise D105AuthorityError("authority private seed must be 32 raw bytes or hex") from error
    if len(text) != 64:
        raise D105AuthorityError("authority private seed hex must contain 64 characters")
    try:
        return bytes.fromhex(text)
    except ValueError as error:
        raise D105AuthorityError("authority private seed hex is malformed") from error


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise D105AuthorityError("signature output must be a new child file")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105AuthorityError("signature output remained writable")


def _load_canonical_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    payload = _regular(path, name).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105AuthorityError(f"{name} is not valid UTF-8 JSON") from error
    if type(value) is not dict or canonical_bytes(value) != payload:
        raise D105AuthorityError(f"{name} must be canonical JSON")
    return value, payload


def _load_canonical_json_with_optional_newline(
    path: Path, name: str
) -> tuple[dict[str, Any], bytes]:
    """Read a Target25 input written with either canonical JSON form."""

    payload = _regular(path, name).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105AuthorityError(f"{name} is not valid UTF-8 JSON") from error
    canonical = canonical_bytes(value)
    if type(value) is not dict or payload not in (canonical, canonical + b"\n"):
        raise D105AuthorityError(f"{name} must be canonical JSON")
    return value, payload


_TARGET25_PLAN_MANIFEST_FIELDS = {
    "schema",
    "plan_payload",
    "candidate_identity_sources",
    "plan_receipt_sha256",
    "plan_manifest_receipt_sha256",
}
_TARGET25_PLAN_PAYLOAD_FIELDS = {
    "schema",
    "seed",
    "claim_scope",
    "formal_launch_authority",
    "authority_envelope_root_sha256",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "arms",
    "leo_scenarios",
    "target25_slices",
    "rows",
}
_TARGET25_CONTEXT_FIELDS = {
    "schema",
    "plan_receipt_sha256",
    "claim_scope",
    "formal_launch_authority",
    "authority_envelope_root_sha256",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "rows",
    "context_manifest_receipt_sha256",
}
_TARGET25_PREPARE_RECEIPT_FIELDS = {
    "schema",
    "status",
    "claim_scope",
    "formal_launch_authority",
    "promotable",
    "matrix_index_sha256",
    "plan_manifest_sha256",
    "context_manifest_sha256",
    "plan_receipt_sha256",
    "authority_envelope_root_sha256",
    "data_feature_runtime_sha256",
    "data_materialization_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "outer_row_count",
    "prepare_receipt_sha256",
}


def _validate_target25_plan_document(plan: dict[str, Any]) -> dict[str, Any]:
    if (
        set(plan) != _TARGET25_PLAN_MANIFEST_FIELDS
        or plan.get("schema") != "cvs.phase2.d105.target25_runner.v1.plan_manifest"
        or plan.get("plan_manifest_receipt_sha256")
        != sha256_bytes(
            canonical_bytes(
                {
                    key: value
                    for key, value in plan.items()
                    if key != "plan_manifest_receipt_sha256"
                }
            )
        )
    ):
        raise D105AuthorityError("Target25 plan manifest closure or receipt drift")
    payload = plan.get("plan_payload")
    sources = plan.get("candidate_identity_sources")
    if (
        type(payload) is not dict
        or set(payload) != _TARGET25_PLAN_PAYLOAD_FIELDS
        or payload.get("schema") != "cvs.phase2.d105.target25_runner.v1.plan"
        or type(sources) is not dict
        or set(sources)
        != {"candidate_runtime_manifest_path", "candidate_method_lock_path"}
        or plan.get("plan_receipt_sha256") != sha256_bytes(canonical_bytes(payload))
    ):
        raise D105AuthorityError("Target25 plan payload closure or receipt drift")
    return payload


def _validate_target25_context_document(context: dict[str, Any]) -> None:
    if (
        set(context) != _TARGET25_CONTEXT_FIELDS
        or context.get("schema")
        != "cvs.phase2.d105.target25_context_manifest.v1"
        or context.get("context_manifest_receipt_sha256")
        != sha256_bytes(
            canonical_bytes(
                {
                    key: value
                    for key, value in context.items()
                    if key != "context_manifest_receipt_sha256"
                }
            )
        )
    ):
        raise D105AuthorityError("Target25 context manifest closure or receipt drift")


def _component_identity(component_dir: Path, validated_bundle_id_sha256: str) -> dict[str, str]:
    asset = load_d105_phase1_asset(component_dir, require_formal_phase2_eligible=False)
    if asset.formal_phase2_eligible or asset.manifest["status"] != COMPONENT_STATUS:
        raise D105AuthorityError("authority signing requires a gate-complete unsealed component")
    manifest = asset.manifest
    if set(manifest["formal_phase2_eligibility_missing"]) != {
        "independent_review_p0_0_p1_0",
        "independent_phase2_authority_seal",
    }:
        raise D105AuthorityError("component prerequisites are not ready for authority signing")
    validator = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=validated_bundle_id_sha256,
        expected_content_root_sha256=asset.bundle.content_root_sha256,
        checkpoint_sha256=asset.bundle.checkpoint_sha256,
        runtime_sha256=asset.bundle.runtime_sha256,
        method_lock_sha256=asset.bundle.method_lock_sha256,
        receipt_root_sha256=compute_d105_bundle_receipt_root(asset.bundle),
    )
    if validated_bundle_id_sha256 == asset.bundle.content_root_sha256:
        raise D105AuthorityError("validated bundle ID may not equal content root")
    return {
        "candidate_id": str(manifest["candidate_id"]),
        "component_manifest_sha256": asset.manifest_sha256,
        "bundle_wire_sha256": str(manifest["bundle_wire_sha256"]),
        "bundle_content_root_sha256": asset.bundle.content_root_sha256,
        "bundle_receipt_root_sha256": str(manifest["bundle_receipt_root_sha256"]),
        "checkpoint_sha256": asset.bundle.checkpoint_sha256,
        "runtime_sha256": asset.bundle.runtime_sha256,
        "method_lock_sha256": asset.bundle.method_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": str(
            manifest["d105_candidate_runtime_manifest_sha256"]
        ),
        "d105_candidate_method_lock_sha256": str(
            manifest["d105_candidate_method_lock_sha256"]
        ),
        "strict_tap_receipt_sha256": str(manifest["strict_tap_receipt_sha256"]),
        "source_held_gate_receipt_sha256": str(
            manifest["source_held_gate_receipt_sha256"]
        ),
        "validated_bundle_id_sha256": validated_bundle_id_sha256,
        "validator_receipt_sha256": validator,
        "d102_revocation_manifest_sha256": str(
            manifest["d102_revocation_manifest_sha256"]
        ),
    }


def _sign_revocation(args: argparse.Namespace) -> dict[str, object]:
    manifest, payload = _load_canonical_json(
        args.revocation_manifest, "D102 revocation manifest"
    )
    signature = sign_d105_detached(
        domain=D102_REVOCATION_SIGNATURE_DOMAIN,
        payload=payload,
        private_seed=_load_private_seed(args.private_seed_file),
    )
    validate_d102_revocation_manifest(manifest, signature)
    _write_new(args.output_signature, signature)
    return {
        "d102_revocation_manifest": str(args.revocation_manifest),
        "d102_revocation_signature": str(args.output_signature),
    }


def _sign_authority(args: argparse.Namespace) -> dict[str, object]:
    revocation = load_signed_d102_revocation_manifest(
        _regular(args.d102_revocation_manifest, "D102 revocation manifest"),
        _regular(args.d102_revocation_signature, "D102 revocation signature"),
    )
    identity = _component_identity(args.component_dir, args.validated_bundle_id_sha256)
    if identity["d102_revocation_manifest_sha256"] != revocation["manifest_sha256"]:
        raise D105AuthorityError("external D102 revocation does not match component")
    review = load_independent_review_receipt(
        _regular(args.independent_review_receipt, "independent review receipt"),
        identity=identity,
    )
    envelope = build_d105_authority_envelope(
        identity=identity,
        independent_review_receipt_sha256=review["receipt_sha256"],
        d102_revocation_manifest_sha256=revocation["manifest_sha256"],
        nonce_ledger_identity_sha256=args.nonce_ledger_identity_sha256,
        issued_at=args.issued_at,
        not_before=args.not_before,
        expires_at=args.expires_at,
        nonce=args.nonce,
        run_id=args.run_id,
        git_commit=args.git_commit,
    )
    payload = canonical_bytes(envelope)
    signature = sign_d105_detached(
        domain=AUTHORITY_SIGNATURE_DOMAIN,
        payload=payload,
        private_seed=_load_private_seed(args.private_seed_file),
    )
    _write_new(args.output_envelope, payload)
    _write_new(args.output_signature, signature)
    return {
        "authority_envelope": str(args.output_envelope),
        "authority_signature": str(args.output_signature),
        "independent_review_receipt_sha256": review["receipt_sha256"],
        "d102_revocation_manifest_sha256": revocation["manifest_sha256"],
        "run_id": args.run_id,
    }


def _target25_prepare_binding(args: argparse.Namespace) -> dict[str, Any]:
    """Recompute every signable Target25 prepare identity from immutable files."""

    receipt, _ = _load_canonical_json_with_optional_newline(
        args.prepare_receipt, "Target25 prepare receipt"
    )
    if (
        set(receipt) != _TARGET25_PREPARE_RECEIPT_FIELDS
        or receipt.get("schema") != "cvs.phase2.d105.target25_prepare_receipt.v1"
        or receipt.get("status") != "TARGET25_INPUTS_PREPARED"
        or receipt.get("promotable") is not False
    ):
        raise D105AuthorityError("Target25 prepare receipt closure drift")
    unsigned = dict(receipt)
    declared_receipt_sha = unsigned.pop("prepare_receipt_sha256", None)
    if declared_receipt_sha != sha256_bytes(canonical_bytes(unsigned)):
        raise D105AuthorityError("Target25 prepare receipt self-hash drift")
    plan, _ = _load_canonical_json_with_optional_newline(
        args.plan_manifest, "Target25 plan manifest"
    )
    plan_payload = _validate_target25_plan_document(plan)
    plan_binding = {**plan_payload, "plan_receipt_sha256": plan["plan_receipt_sha256"]}
    context, _ = _load_canonical_json_with_optional_newline(
        args.context_manifest, "Target25 context manifest"
    )
    _validate_target25_context_document(context)
    expected_file_hashes = {
        "matrix_index_sha256": sha256_file(
            _regular(args.matrix_index, "Target25 matrix index")
        ),
        "plan_manifest_sha256": sha256_file(
            _regular(args.plan_manifest, "Target25 plan manifest")
        ),
        "context_manifest_sha256": sha256_file(
            _regular(args.context_manifest, "Target25 context manifest")
        ),
    }
    for field, observed in expected_file_hashes.items():
        if receipt.get(field) != observed:
            raise D105AuthorityError(f"Target25 prepare receipt {field} drift")
    shared_fields = (
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
    )
    for field in shared_fields:
        if (
            receipt.get(field) != plan_binding.get(field)
            or receipt.get(field) != context.get(field)
        ):
            raise D105AuthorityError(
                f"Target25 prepare receipt/plan/context {field} drift"
            )
    return {
        "run_id": args.run_id,
        "git_commit": args.git_commit,
        **expected_file_hashes,
        "prepare_receipt_file_sha256": sha256_file(
            _regular(args.prepare_receipt, "Target25 prepare receipt")
        ),
        "prepare_receipt_sha256": declared_receipt_sha,
        "nonce_ledger_identity_sha256": args.nonce_ledger_identity_sha256,
        **{field: receipt[field] for field in shared_fields},
    }


def _sign_target25_prepare(args: argparse.Namespace) -> dict[str, object]:
    binding = _target25_prepare_binding(args)
    envelope = build_d105_target25_prepare_envelope(
        binding=binding,
        issued_at=args.issued_at,
        not_before=args.not_before,
        expires_at=args.expires_at,
        nonce=args.nonce,
    )
    payload = canonical_bytes(envelope)
    signature = sign_d105_detached(
        domain=TARGET25_PREPARE_SIGNATURE_DOMAIN,
        payload=payload,
        private_seed=_load_private_seed(args.private_seed_file),
    )
    _write_new(args.output_envelope, payload)
    _write_new(args.output_signature, signature)
    return {
        "target25_prepare_authority_envelope": str(args.output_envelope),
        "target25_prepare_authority_signature": str(args.output_signature),
        "prepare_receipt_file_sha256": binding["prepare_receipt_file_sha256"],
        "run_id": binding["run_id"],
    }


def _add_revocation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--revocation-manifest", type=Path, required=True)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    parser.add_argument("--output-signature", type=Path, required=True)


def _add_authority_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--independent-review-receipt", type=Path, required=True)
    parser.add_argument("--d102-revocation-manifest", type=Path, required=True)
    parser.add_argument("--d102-revocation-signature", type=Path, required=True)
    parser.add_argument("--validated-bundle-id-sha256", required=True)
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--nonce-ledger-identity-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    parser.add_argument("--output-envelope", type=Path, required=True)
    parser.add_argument("--output-signature", type=Path, required=True)


def _add_target25_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepare-receipt", type=Path, required=True)
    parser.add_argument("--matrix-index", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--nonce-ledger-identity-sha256", required=True)
    parser.add_argument("--private-seed-file", type=Path, required=True)
    parser.add_argument("--output-envelope", type=Path, required=True)
    parser.add_argument("--output-signature", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    revocation = commands.add_parser("sign-d102-revocation")
    authority = commands.add_parser("sign-authority")
    target25_prepare = commands.add_parser("sign-target25-prepare")
    _add_revocation_arguments(revocation)
    _add_authority_arguments(authority)
    _add_target25_prepare_arguments(target25_prepare)
    args = parser.parse_args(argv)
    try:
        if args.command == "sign-d102-revocation":
            result = _sign_revocation(args)
        elif args.command == "sign-authority":
            result = _sign_authority(args)
        elif args.command == "sign-target25-prepare":
            result = _sign_target25_prepare(args)
        else:
            raise AssertionError("authority signer command closure drift")
    except (D105AuthorityError, D105Phase1BundleError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
