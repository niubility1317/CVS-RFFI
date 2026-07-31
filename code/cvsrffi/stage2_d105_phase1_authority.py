"""Fail-closed D105 Phase1 authority and D102 revocation primitives.

The module deliberately has no dataset, bundle-builder, NumPy, or runtime
evaluator dependency.  Its trust root is the already-pinned Ed25519 public
key in :mod:`cvsrffi.somph_runtime_trust`; a D105 signature is therefore not
interchangeable with a prior SOMP-H or revocation signature.

Only the external signing CLI should call the small signing helpers below.
They accept a private seed supplied at invocation time and refuse to sign
unless its derived public key is exactly the pinned production key.  No
private material is stored by this module or by a formal asset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, MutableSet, Sequence

from cvsrffi import somph_runtime_trust


AUTHORITY_ENVELOPE_SCHEMA = "cvs.phase1.d105.cbrc.authority_envelope.v2"
AUTHORITY_SIGNATURE_DOMAIN = "CVS-RFFI/D105-PHASE1-AUTHORITY/ED25519/V1"
D102_REVOCATION_SCHEMA = "cvs.phase1.d105.d102_revocation_manifest.v1"
D102_REVOCATION_SIGNATURE_DOMAIN = "CVS-RFFI/D105-D102-REVOCATION/ED25519/V1"
INDEPENDENT_REVIEW_SCHEMA = "cvs.phase1.d105.cbrc.independent_review_receipt.v1"
NONCE_CONSUMPTION_SCHEMA = "cvs.phase1.d105.cbrc.authority_nonce_consumption.v1"
NONCE_LEDGER_IDENTITY_SCHEMA = "cvs.phase1.d105.nonce_ledger_identity.v1"
TARGET25_PREPARE_ENVELOPE_SCHEMA = (
    "cvs.phase2.d105.target25_prepare_authority_envelope.v1"
)
TARGET25_PREPARE_SIGNATURE_DOMAIN = "CVS-RFFI/D105-TARGET25-PREPARE/ED25519/V1"
TARGET25_DEVELOPMENT_CLAIM_SCOPE = "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE"

AUTHORITY_ENVELOPE_NAME = "d105_phase1_authority_envelope.json"
AUTHORITY_SIGNATURE_NAME = "d105_phase1_authority_envelope.ed25519"
D102_REVOCATION_MANIFEST_NAME = "d105_d102_revocation_manifest.json"
D102_REVOCATION_SIGNATURE_NAME = "d105_d102_revocation_manifest.ed25519"
INDEPENDENT_REVIEW_RECEIPT_NAME = "d105_independent_review_receipt.json"
TARGET25_PREPARE_ENVELOPE_NAME = "d105_target25_prepare_authority_envelope.json"
TARGET25_PREPARE_SIGNATURE_NAME = "d105_target25_prepare_authority_envelope.ed25519"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class D105AuthorityError(ValueError):
    """Raised when a D105 authority or revocation artifact is untrusted."""


def canonical_bytes(value: Any) -> bytes:
    """Return the one permitted canonical JSON encoding for authority data."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise D105AuthorityError(f"{name} must be a lowercase SHA256")
    return text


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise D105AuthorityError(f"{name} exact schema drift")


def _require_nonempty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise D105AuthorityError(f"{name} must be a non-empty string")
    return value


def _parse_utc(value: Any, name: str) -> datetime:
    text = _require_nonempty_text(value, name)
    if not _UTC_RE.fullmatch(text):
        raise D105AuthorityError(f"{name} must be UTC RFC3339 seconds")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise D105AuthorityError(f"{name} is not a valid UTC timestamp") from error


def _normalise_now(now_utc: datetime | None) -> datetime:
    now = datetime.now(timezone.utc) if now_utc is None else now_utc
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise D105AuthorityError("authority verification time must be timezone-aware")
    return now.astimezone(timezone.utc)


def _read_canonical_json(path: str | Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D105AuthorityError(f"{name} must be a regular non-symlink file")
    payload = source.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105AuthorityError(f"{name} is not valid UTF-8 JSON") from error
    if type(value) is not dict or canonical_bytes(value) != payload:
        raise D105AuthorityError(f"{name} must be canonical JSON")
    return value, payload


def _read_signature(path: str | Path, *, name: str) -> bytes:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise D105AuthorityError(f"{name} must be a regular non-symlink file")
    signature = source.read_bytes()
    if len(signature) != 64:
        raise D105AuthorityError(f"{name} must contain one raw 64-byte Ed25519 signature")
    return signature


def _signature_message(domain: str, payload: bytes) -> bytes:
    domain_bytes = domain.encode("ascii")
    return domain_bytes + b"\0" + payload


def _pinned_public_key() -> bytes:
    key_id = somph_runtime_trust.PINNED_AUTHORITY_KEY_ID
    key_hex = somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_HEX
    key_sha = somph_runtime_trust.PINNED_AUTHORITY_PUBLIC_KEY_SHA256
    if type(key_id) is not str or type(key_hex) is not str or type(key_sha) is not str:
        raise D105AuthorityError("pinned authority configuration drift")
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as error:
        raise D105AuthorityError("pinned authority public key is malformed") from error
    if len(key) != 32 or sha256_bytes(key) != key_sha:
        raise D105AuthorityError("pinned authority public key SHA256 drift")
    return key


def _verify_pinned_signature(*, domain: str, payload: bytes, signature: bytes, issuer_key_id: Any) -> None:
    if issuer_key_id != somph_runtime_trust.PINNED_AUTHORITY_KEY_ID:
        raise D105AuthorityError("authority issuer key ID is not pinned")
    try:
        somph_runtime_trust.verify_ed25519(
            _pinned_public_key(), _signature_message(domain, payload), signature
        )
    except ValueError as error:
        raise D105AuthorityError("D105 detached authority signature invalid") from error


_REVOCATION_ENTRY_FIELDS = {
    "run_id",
    "bundle_manifest_sha256",
    "bundle_payload_sha256",
    "bundle_seal_sha256",
    "bundle_content_root_sha256",
    "checkpoint_sha256",
    "method_lock_sha256",
    "runtime_sha256",
    "held_score_sha256",
    "tap_archive_sha256",
    "status",
}

_REVOCATION_FIELDS = {
    "schema",
    "signature_domain",
    "issuer_key_id",
    "issued_at",
    "not_before",
    "expires_at",
    "revocation_id",
    "revoked_artifacts",
}

_REVIEW_FIELDS = {
    "schema",
    "candidate_id",
    "component_manifest_sha256",
    "bundle_content_root_sha256",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "reviewer_id",
    "reviewed_at",
    "review_p0",
    "review_p1",
}

_AUTHORITY_FIELDS = {
    "schema",
    "signature_domain",
    "candidate_id",
    "issuer_key_id",
    "component_manifest_sha256",
    "bundle_wire_sha256",
    "bundle_content_root_sha256",
    "bundle_receipt_root_sha256",
    "checkpoint_sha256",
    "runtime_sha256",
    "method_lock_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "strict_tap_receipt_sha256",
    "source_held_gate_receipt_sha256",
    "validated_bundle_id_sha256",
    "validator_receipt_sha256",
    "independent_review_receipt_sha256",
    "d102_revocation_manifest_sha256",
    "nonce_ledger_identity_sha256",
    "issued_at",
    "not_before",
    "expires_at",
    "nonce",
    "run_id",
    "git_commit",
}

_TARGET25_PREPARE_BINDING_FIELDS = {
    "run_id",
    "git_commit",
    "matrix_index_sha256",
    "prepare_receipt_file_sha256",
    "prepare_receipt_sha256",
    "plan_manifest_sha256",
    "context_manifest_sha256",
    "plan_receipt_sha256",
    "authority_envelope_root_sha256",
    "d105_candidate_runtime_manifest_sha256",
    "d105_candidate_method_lock_sha256",
    "claim_scope",
    "formal_launch_authority",
    "nonce_ledger_identity_sha256",
}

_TARGET25_PREPARE_FIELDS = _TARGET25_PREPARE_BINDING_FIELDS | {
    "schema",
    "signature_domain",
    "issuer_key_id",
    "issued_at",
    "not_before",
    "expires_at",
    "nonce",
}


def _validate_time_window(
    value: Mapping[str, Any], *, now_utc: datetime | None, name: str
) -> None:
    issued = _parse_utc(value["issued_at"], f"{name}.issued_at")
    not_before = _parse_utc(value["not_before"], f"{name}.not_before")
    expires = _parse_utc(value["expires_at"], f"{name}.expires_at")
    if not_before > issued or issued > expires:
        raise D105AuthorityError(f"{name} issuance window ordering drift")
    now = _normalise_now(now_utc)
    if now < not_before:
        raise D105AuthorityError(f"{name} is not yet valid")
    if now > expires:
        raise D105AuthorityError(f"{name} is expired")


def validate_d102_revocation_manifest(
    manifest: Mapping[str, Any],
    signature: bytes,
    *,
    now_utc: datetime | None = None,
) -> str:
    """Verify a pinned, domain-separated D102 immutable revocation list."""

    _require_exact_keys(manifest, _REVOCATION_FIELDS, "D102 revocation manifest")
    if (
        manifest["schema"] != D102_REVOCATION_SCHEMA
        or manifest["signature_domain"] != D102_REVOCATION_SIGNATURE_DOMAIN
    ):
        raise D105AuthorityError("D102 revocation manifest domain/schema drift")
    _require_sha256(manifest["revocation_id"], "revocation_id")
    _validate_time_window(manifest, now_utc=now_utc, name="D102 revocation manifest")
    rows = manifest["revoked_artifacts"]
    if type(rows) is not list or not rows:
        raise D105AuthorityError("D102 revocation manifest must contain revoked artifacts")
    seen: set[bytes] = set()
    for row in rows:
        _require_exact_keys(row, _REVOCATION_ENTRY_FIELDS, "D102 revocation entry")
        run_id = _require_nonempty_text(row["run_id"], "D102 revocation run_id")
        if not _RUN_ID_RE.fullmatch(run_id):
            raise D105AuthorityError("D102 revocation run_id drift")
        if row["status"] != "PHASE1_HELD_FALSIFIER_REJECT":
            raise D105AuthorityError("D102 revocation status must be immutable rejection")
        for field in _REVOCATION_ENTRY_FIELDS - {"run_id", "status"}:
            _require_sha256(row[field], f"D102 revocation {field}")
        identity = canonical_bytes(row)
        if identity in seen:
            raise D105AuthorityError("D102 revocation entry is duplicated")
        seen.add(identity)
    payload = canonical_bytes(manifest)
    _verify_pinned_signature(
        domain=D102_REVOCATION_SIGNATURE_DOMAIN,
        payload=payload,
        signature=signature,
        issuer_key_id=manifest["issuer_key_id"],
    )
    return sha256_bytes(payload)


def load_signed_d102_revocation_manifest(
    manifest_path: str | Path,
    signature_path: str | Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Load and verify externally signed D102 revocation material."""

    manifest, payload = _read_canonical_json(
        manifest_path, name="D102 revocation manifest"
    )
    signature = _read_signature(signature_path, name="D102 revocation signature")
    digest = validate_d102_revocation_manifest(
        manifest, signature, now_utc=now_utc
    )
    if digest != sha256_bytes(payload):
        raise D105AuthorityError("D102 revocation canonical SHA256 drift")
    return {
        "manifest": manifest,
        "manifest_bytes": payload,
        "signature": signature,
        "manifest_sha256": digest,
        "signature_sha256": sha256_bytes(signature),
    }


def reject_revoked_d102_identity(
    revocation_manifest: Mapping[str, Any],
    *,
    run_id: str | None = None,
    bundle_manifest_sha256: str | None = None,
    bundle_payload_sha256: str | None = None,
    bundle_seal_sha256: str | None = None,
    bundle_content_root_sha256: str | None = None,
    held_score_sha256: str | None = None,
    tap_archive_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    method_lock_sha256: str | None = None,
    runtime_sha256: str | None = None,
) -> None:
    """Reject content identity, not a filename or caller self-description.

    Direct manifest/payload/seal/content/held/tap equality blocks renamed D102
    copies.  The checkpoint/runtime/method triple is checked only together so
    a legitimate D105 component can continue to use the same frozen Phase1
    checkpoint without being rejected for a single shared digest.
    """

    for row in revocation_manifest["revoked_artifacts"]:
        direct = (
            (run_id is not None and run_id == row["run_id"])
            or (
                bundle_manifest_sha256 is not None
                and bundle_manifest_sha256 == row["bundle_manifest_sha256"]
            )
            or (
                bundle_payload_sha256 is not None
                and bundle_payload_sha256 == row["bundle_payload_sha256"]
            )
            or (
                bundle_seal_sha256 is not None
                and bundle_seal_sha256 == row["bundle_seal_sha256"]
            )
            or (
                bundle_content_root_sha256 is not None
                and bundle_content_root_sha256 == row["bundle_content_root_sha256"]
            )
            or (
                held_score_sha256 is not None
                and held_score_sha256 == row["held_score_sha256"]
            )
            or (
                tap_archive_sha256 is not None
                and tap_archive_sha256 == row["tap_archive_sha256"]
            )
        )
        triple = (
            checkpoint_sha256 is not None
            and method_lock_sha256 is not None
            and runtime_sha256 is not None
            and checkpoint_sha256 == row["checkpoint_sha256"]
            and method_lock_sha256 == row["method_lock_sha256"]
            and runtime_sha256 == row["runtime_sha256"]
        )
        if direct or triple:
            raise D105AuthorityError("D102 revoked immutable content identity detected")


def validate_independent_review_receipt(
    receipt: Mapping[str, Any], *, identity: Mapping[str, Any]
) -> str:
    """Check a signed-by-envelope independent review receipt, including P0/P1."""

    _require_exact_keys(receipt, _REVIEW_FIELDS, "independent review receipt")
    if receipt["schema"] != INDEPENDENT_REVIEW_SCHEMA:
        raise D105AuthorityError("independent review receipt schema drift")
    _require_nonempty_text(receipt["reviewer_id"], "independent reviewer ID")
    _parse_utc(receipt["reviewed_at"], "independent review reviewed_at")
    for field in (
        "component_manifest_sha256",
        "bundle_content_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
    ):
        _require_sha256(receipt[field], f"independent review {field}")
        if receipt[field] != identity[field]:
            raise D105AuthorityError("independent review/component binding drift")
    if receipt["candidate_id"] != identity["candidate_id"]:
        raise D105AuthorityError("independent review candidate binding drift")
    for field in ("review_p0", "review_p1"):
        if type(receipt[field]) is not int or receipt[field] < 0:
            raise D105AuthorityError("independent review P0/P1 must be non-negative ints")
    if receipt["review_p0"] != 0 or receipt["review_p1"] != 0:
        raise D105AuthorityError("independent review P0/P1 must both be zero")
    return sha256_bytes(canonical_bytes(receipt))


def load_independent_review_receipt(path: str | Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    receipt, payload = _read_canonical_json(path, name="independent review receipt")
    digest = validate_independent_review_receipt(receipt, identity=identity)
    if digest != sha256_bytes(payload):
        raise D105AuthorityError("independent review canonical SHA256 drift")
    return {"receipt": receipt, "receipt_bytes": payload, "receipt_sha256": digest}


def validate_d105_authority_envelope(
    envelope: Mapping[str, Any],
    signature: bytes,
    *,
    identity: Mapping[str, Any],
    independent_review_receipt_sha256: str,
    d102_revocation_manifest_sha256: str,
    now_utc: datetime | None = None,
    nonce_guard: MutableSet[tuple[str, str]] | None = None,
) -> str:
    """Verify the full D105 authority envelope against immutable identities.

    ``nonce_guard`` is a caller-owned consumed-nonce set.  When supplied,
    validation is intentionally one-shot: a second envelope with the same
    pinned-key/nonce pair is rejected before it can be used to create another
    formal release.
    """

    _require_exact_keys(envelope, _AUTHORITY_FIELDS, "D105 authority envelope")
    if (
        envelope["schema"] != AUTHORITY_ENVELOPE_SCHEMA
        or envelope["signature_domain"] != AUTHORITY_SIGNATURE_DOMAIN
    ):
        raise D105AuthorityError("D105 authority envelope domain/schema drift")
    for field in (
        "component_manifest_sha256",
        "bundle_wire_sha256",
        "bundle_content_root_sha256",
        "bundle_receipt_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_held_gate_receipt_sha256",
        "validated_bundle_id_sha256",
        "validator_receipt_sha256",
        "independent_review_receipt_sha256",
        "d102_revocation_manifest_sha256",
        "nonce_ledger_identity_sha256",
        "nonce",
    ):
        _require_sha256(envelope[field], f"D105 authority {field}")
    for field in (
        "candidate_id",
        "component_manifest_sha256",
        "bundle_wire_sha256",
        "bundle_content_root_sha256",
        "bundle_receipt_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_held_gate_receipt_sha256",
        "validated_bundle_id_sha256",
        "validator_receipt_sha256",
    ):
        if envelope[field] != identity[field]:
            raise D105AuthorityError("D105 authority envelope/component binding drift")
    if envelope["independent_review_receipt_sha256"] != _require_sha256(
        independent_review_receipt_sha256, "independent_review_receipt_sha256"
    ):
        raise D105AuthorityError("D105 authority review receipt SHA256 drift")
    if envelope["d102_revocation_manifest_sha256"] != _require_sha256(
        d102_revocation_manifest_sha256, "d102_revocation_manifest_sha256"
    ):
        raise D105AuthorityError("D105 authority revocation SHA256 drift")
    run_id = _require_nonempty_text(envelope["run_id"], "D105 authority run_id")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise D105AuthorityError("D105 authority run_id drift")
    if not _GIT_COMMIT_RE.fullmatch(_require_nonempty_text(envelope["git_commit"], "git_commit")):
        raise D105AuthorityError("D105 authority git commit drift")
    _validate_time_window(envelope, now_utc=now_utc, name="D105 authority envelope")
    payload = canonical_bytes(envelope)
    _verify_pinned_signature(
        domain=AUTHORITY_SIGNATURE_DOMAIN,
        payload=payload,
        signature=signature,
        issuer_key_id=envelope["issuer_key_id"],
    )
    if nonce_guard is not None:
        nonce_key = (str(envelope["issuer_key_id"]), str(envelope["nonce"]))
        if nonce_key in nonce_guard:
            raise D105AuthorityError("D105 authority nonce replay detected")
        nonce_guard.add(nonce_key)
    return sha256_bytes(payload)


def load_signed_d105_authority_envelope(
    envelope_path: str | Path,
    signature_path: str | Path,
    *,
    identity: Mapping[str, Any],
    independent_review_receipt_sha256: str,
    d102_revocation_manifest_sha256: str,
    now_utc: datetime | None = None,
    nonce_guard: MutableSet[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Load a pinned D105 envelope and its detached signature."""

    envelope, payload = _read_canonical_json(envelope_path, name="D105 authority envelope")
    signature = _read_signature(signature_path, name="D105 authority signature")
    digest = validate_d105_authority_envelope(
        envelope,
        signature,
        identity=identity,
        independent_review_receipt_sha256=independent_review_receipt_sha256,
        d102_revocation_manifest_sha256=d102_revocation_manifest_sha256,
        now_utc=now_utc,
        nonce_guard=nonce_guard,
    )
    if digest != sha256_bytes(payload):
        raise D105AuthorityError("D105 authority canonical SHA256 drift")
    return {
        "envelope": envelope,
        "envelope_bytes": payload,
        "signature": signature,
        "envelope_sha256": digest,
        "signature_sha256": sha256_bytes(signature),
    }


def _validate_target25_prepare_binding(
    binding: Mapping[str, Any], *, name: str
) -> None:
    _require_exact_keys(binding, _TARGET25_PREPARE_BINDING_FIELDS, name)
    run_id = _require_nonempty_text(binding["run_id"], f"{name}.run_id")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise D105AuthorityError(f"{name}.run_id drift")
    git_commit = _require_nonempty_text(binding["git_commit"], f"{name}.git_commit")
    if not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise D105AuthorityError(f"{name}.git_commit drift")
    for field in (
        "matrix_index_sha256",
        "prepare_receipt_file_sha256",
        "prepare_receipt_sha256",
        "plan_manifest_sha256",
        "context_manifest_sha256",
        "plan_receipt_sha256",
        "authority_envelope_root_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "nonce_ledger_identity_sha256",
    ):
        _require_sha256(binding[field], f"{name}.{field}")
    if binding["claim_scope"] != TARGET25_DEVELOPMENT_CLAIM_SCOPE:
        raise D105AuthorityError(f"{name} claim scope is not development-only")
    if binding["formal_launch_authority"] is not False:
        raise D105AuthorityError(f"{name} must deny formal launch authority")


def build_d105_target25_prepare_envelope(
    *,
    binding: Mapping[str, Any],
    issued_at: str,
    not_before: str,
    expires_at: str,
    nonce: str,
) -> dict[str, Any]:
    """Build, but do not sign, one strict Target25 prepare authority envelope.

    ``binding`` is deliberately exact rather than open-ended.  The signature
    is meant to make a locally generated Target25 prepare receipt executable
    only for the one reviewed run, frozen matrix, implementation pair, and
    development-only claim scope named here.
    """

    _validate_target25_prepare_binding(binding, name="Target25 prepare binding")
    envelope = {
        "schema": TARGET25_PREPARE_ENVELOPE_SCHEMA,
        "signature_domain": TARGET25_PREPARE_SIGNATURE_DOMAIN,
        "issuer_key_id": somph_runtime_trust.PINNED_AUTHORITY_KEY_ID,
        **dict(binding),
        "issued_at": issued_at,
        "not_before": not_before,
        "expires_at": expires_at,
        "nonce": nonce,
    }
    _require_exact_keys(
        envelope, _TARGET25_PREPARE_FIELDS, "Target25 prepare authority envelope"
    )
    _require_sha256(envelope["nonce"], "Target25 prepare authority nonce")
    _validate_time_window(
        envelope,
        now_utc=_parse_utc(issued_at, "issued_at"),
        name="Target25 prepare authority envelope",
    )
    return envelope


def validate_d105_target25_prepare_envelope(
    envelope: Mapping[str, Any],
    signature: bytes,
    *,
    expected_binding: Mapping[str, Any],
    now_utc: datetime | None = None,
    nonce_guard: MutableSet[tuple[str, str]] | None = None,
) -> str:
    """Verify one externally signed, exact Target25 prepare authorization."""

    _require_exact_keys(
        envelope, _TARGET25_PREPARE_FIELDS, "Target25 prepare authority envelope"
    )
    if (
        envelope["schema"] != TARGET25_PREPARE_ENVELOPE_SCHEMA
        or envelope["signature_domain"] != TARGET25_PREPARE_SIGNATURE_DOMAIN
    ):
        raise D105AuthorityError("Target25 prepare authority domain/schema drift")
    _require_sha256(envelope["nonce"], "Target25 prepare authority nonce")
    observed_binding = {
        field: envelope[field] for field in _TARGET25_PREPARE_BINDING_FIELDS
    }
    _validate_target25_prepare_binding(
        observed_binding, name="Target25 prepare authority binding"
    )
    _validate_target25_prepare_binding(
        expected_binding, name="expected Target25 prepare binding"
    )
    if observed_binding != dict(expected_binding):
        raise D105AuthorityError("Target25 prepare authority binding drift")
    _validate_time_window(
        envelope, now_utc=now_utc, name="Target25 prepare authority envelope"
    )
    payload = canonical_bytes(envelope)
    _verify_pinned_signature(
        domain=TARGET25_PREPARE_SIGNATURE_DOMAIN,
        payload=payload,
        signature=signature,
        issuer_key_id=envelope["issuer_key_id"],
    )
    if nonce_guard is not None:
        nonce_key = (str(envelope["issuer_key_id"]), str(envelope["nonce"]))
        if nonce_key in nonce_guard:
            raise D105AuthorityError("Target25 prepare authority nonce replay detected")
        nonce_guard.add(nonce_key)
    return sha256_bytes(payload)


def load_signed_d105_target25_prepare_envelope(
    envelope_path: str | Path,
    signature_path: str | Path,
    *,
    expected_binding: Mapping[str, Any],
    now_utc: datetime | None = None,
    nonce_guard: MutableSet[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Load and verify a Target25 prepare authority envelope from disk."""

    envelope, payload = _read_canonical_json(
        envelope_path, name="Target25 prepare authority envelope"
    )
    signature = _read_signature(
        signature_path, name="Target25 prepare authority signature"
    )
    digest = validate_d105_target25_prepare_envelope(
        envelope,
        signature,
        expected_binding=expected_binding,
        now_utc=now_utc,
        nonce_guard=nonce_guard,
    )
    if digest != sha256_bytes(payload):
        raise D105AuthorityError("Target25 prepare authority canonical SHA256 drift")
    return {
        "envelope": envelope,
        "envelope_bytes": payload,
        "signature": signature,
        "envelope_sha256": digest,
        "signature_sha256": sha256_bytes(signature),
    }


def compute_d105_nonce_ledger_identity(
    ledger_dir: str | Path,
    *,
    run_id: str,
    signature_domain: str,
) -> str:
    """Hash one pre-existing, host-local nonce-ledger namespace.

    The authority signer receives this resulting SHA256 out of band.  It does
    not resolve a local path itself, so an offline signer can bind a future
    N607 ledger without leaking or accidentally substituting its own host
    path.  At consumption the N607 process recomputes this identity from its
    own resolved ledger path, signed run ID and domain.
    """

    if signature_domain not in (
        AUTHORITY_SIGNATURE_DOMAIN,
        TARGET25_PREPARE_SIGNATURE_DOMAIN,
    ):
        raise D105AuthorityError("nonce ledger signature domain is not consumable")
    safe_run_id = _require_nonempty_text(run_id, "nonce ledger run_id")
    if not _RUN_ID_RE.fullmatch(safe_run_id):
        raise D105AuthorityError("nonce ledger run_id drift")
    root = Path(ledger_dir)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise D105AuthorityError(
            "authority nonce ledger must be a pre-existing normal directory"
        )
    resolved = root.resolve(strict=True)
    ledger_root = os.path.normcase(str(resolved)).replace("\\", "/")
    if not ledger_root:
        raise D105AuthorityError("authority nonce ledger resolved path drift")
    return sha256_bytes(
        canonical_bytes(
            {
                "schema": NONCE_LEDGER_IDENTITY_SCHEMA,
                "ledger_root": ledger_root,
                "run_id": safe_run_id,
                "signature_domain": signature_domain,
            }
        )
    )


def _consume_d105_nonce_once(
    ledger_dir: str | Path,
    *,
    envelope: Mapping[str, Any],
    envelope_sha256: str,
    signature_domain: str,
    name: str,
) -> Path:
    if envelope.get("signature_domain") != signature_domain:
        raise D105AuthorityError(f"{name} signature domain drift")
    run_id = _require_nonempty_text(envelope.get("run_id"), f"{name} run_id")
    nonce = _require_sha256(envelope.get("nonce"), f"{name} nonce")
    issuer_key_id = _require_nonempty_text(
        envelope.get("issuer_key_id"), f"{name} issuer_key_id"
    )
    expected_ledger_identity = _require_sha256(
        envelope.get("nonce_ledger_identity_sha256"),
        f"{name} nonce_ledger_identity_sha256",
    )
    observed_ledger_identity = compute_d105_nonce_ledger_identity(
        ledger_dir, run_id=run_id, signature_domain=signature_domain
    )
    if observed_ledger_identity != expected_ledger_identity:
        raise D105AuthorityError(f"{name} nonce ledger identity drift")
    root = Path(ledger_dir).resolve(strict=True)
    token = sha256_bytes(
        canonical_bytes(
            {
                "signature_domain": signature_domain,
                "issuer_key_id": issuer_key_id,
                "nonce": nonce,
            }
        )
    )
    marker = root / f"{token}.json"
    if marker.exists() or marker.is_symlink():
        raise D105AuthorityError(f"{name} nonce replay detected")
    value = {
        "schema": NONCE_CONSUMPTION_SCHEMA,
        "authority_envelope_sha256": _require_sha256(
            envelope_sha256, "authority_envelope_sha256"
        ),
        "issuer_key_id": issuer_key_id,
        "signature_domain": signature_domain,
        "nonce": nonce,
        "run_id": run_id,
        "nonce_ledger_identity_sha256": observed_ledger_identity,
    }
    payload = canonical_bytes(value)
    try:
        with marker.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise D105AuthorityError(f"{name} nonce replay detected") from error
    os.chmod(marker, stat.S_IREAD)
    if marker.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105AuthorityError(f"{name} nonce marker remained writable")
    return marker


def consume_authority_nonce_once(
    ledger_dir: str | Path,
    *,
    envelope: Mapping[str, Any],
    envelope_sha256: str,
) -> Path:
    """Atomically consume one verified formal Phase1 authority nonce."""

    return _consume_d105_nonce_once(
        ledger_dir,
        envelope=envelope,
        envelope_sha256=envelope_sha256,
        signature_domain=AUTHORITY_SIGNATURE_DOMAIN,
        name="D105 authority",
    )


def consume_target25_prepare_nonce_once(
    ledger_dir: str | Path,
    *,
    envelope: Mapping[str, Any],
    envelope_sha256: str,
) -> Path:
    """Atomically consume one verified Target25 prepare authorization nonce."""

    return _consume_d105_nonce_once(
        ledger_dir,
        envelope=envelope,
        envelope_sha256=envelope_sha256,
        signature_domain=TARGET25_PREPARE_SIGNATURE_DOMAIN,
        name="Target25 prepare authority",
    )


def ed25519_public_key_from_seed(seed: bytes) -> bytes:
    """Derive an Ed25519 public key from one 32-byte seed without dependencies."""

    if not isinstance(seed, bytes) or len(seed) != 32:
        raise D105AuthorityError("Ed25519 private seed must be exactly 32 bytes")
    digest = hashlib.sha512(seed).digest()
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    return somph_runtime_trust._ed_encode(
        somph_runtime_trust._ed_scalar_mult(somph_runtime_trust._ED_B, scalar)
    )


def sign_d105_detached(*, domain: str, payload: bytes, private_seed: bytes) -> bytes:
    """Sign one D105 domain-separated payload with an externally supplied seed."""

    if domain not in (
        AUTHORITY_SIGNATURE_DOMAIN,
        D102_REVOCATION_SIGNATURE_DOMAIN,
        TARGET25_PREPARE_SIGNATURE_DOMAIN,
    ):
        raise D105AuthorityError("unrecognised D105 signing domain")
    public = ed25519_public_key_from_seed(private_seed)
    if public != _pinned_public_key():
        raise D105AuthorityError("private signing seed does not match pinned authority key")
    digest = hashlib.sha512(private_seed).digest()
    scalar_bytes = bytearray(digest[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 63
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    message = _signature_message(domain, payload)
    nonce_scalar = int.from_bytes(hashlib.sha512(digest[32:] + message).digest(), "little") % somph_runtime_trust._ED_L
    encoded_r = somph_runtime_trust._ed_encode(
        somph_runtime_trust._ed_scalar_mult(somph_runtime_trust._ED_B, nonce_scalar)
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public + message).digest(), "little"
    ) % somph_runtime_trust._ED_L
    response = (nonce_scalar + challenge * scalar) % somph_runtime_trust._ED_L
    return encoded_r + response.to_bytes(32, "little")


def build_d105_authority_envelope(
    *,
    identity: Mapping[str, Any],
    independent_review_receipt_sha256: str,
    d102_revocation_manifest_sha256: str,
    nonce_ledger_identity_sha256: str,
    issued_at: str,
    not_before: str,
    expires_at: str,
    nonce: str,
    run_id: str,
    git_commit: str,
) -> dict[str, Any]:
    """Build, but do not sign, a strict D105 authority envelope."""

    envelope = {
        "schema": AUTHORITY_ENVELOPE_SCHEMA,
        "signature_domain": AUTHORITY_SIGNATURE_DOMAIN,
        "candidate_id": identity["candidate_id"],
        "issuer_key_id": somph_runtime_trust.PINNED_AUTHORITY_KEY_ID,
        "component_manifest_sha256": identity["component_manifest_sha256"],
        "bundle_wire_sha256": identity["bundle_wire_sha256"],
        "bundle_content_root_sha256": identity["bundle_content_root_sha256"],
        "bundle_receipt_root_sha256": identity["bundle_receipt_root_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "method_lock_sha256": identity["method_lock_sha256"],
        "d105_candidate_runtime_manifest_sha256": identity[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": identity[
            "d105_candidate_method_lock_sha256"
        ],
        "strict_tap_receipt_sha256": identity["strict_tap_receipt_sha256"],
        "source_held_gate_receipt_sha256": identity[
            "source_held_gate_receipt_sha256"
        ],
        "validated_bundle_id_sha256": identity["validated_bundle_id_sha256"],
        "validator_receipt_sha256": identity["validator_receipt_sha256"],
        "independent_review_receipt_sha256": independent_review_receipt_sha256,
        "d102_revocation_manifest_sha256": d102_revocation_manifest_sha256,
        "nonce_ledger_identity_sha256": nonce_ledger_identity_sha256,
        "issued_at": issued_at,
        "not_before": not_before,
        "expires_at": expires_at,
        "nonce": nonce,
        "run_id": run_id,
        "git_commit": git_commit,
    }
    # Signature verification is intentionally deferred until after signing,
    # but an external signer still receives only a well-formed payload.
    _require_exact_keys(envelope, _AUTHORITY_FIELDS, "D105 authority envelope")
    for field in (
        "component_manifest_sha256",
        "bundle_wire_sha256",
        "bundle_content_root_sha256",
        "bundle_receipt_root_sha256",
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "strict_tap_receipt_sha256",
        "source_held_gate_receipt_sha256",
        "validated_bundle_id_sha256",
        "validator_receipt_sha256",
        "independent_review_receipt_sha256",
        "d102_revocation_manifest_sha256",
        "nonce_ledger_identity_sha256",
        "nonce",
    ):
        _require_sha256(envelope[field], field)
    _validate_time_window(envelope, now_utc=_parse_utc(issued_at, "issued_at"), name="D105 authority envelope")
    if not _RUN_ID_RE.fullmatch(run_id) or not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise D105AuthorityError("D105 authority run_id or git commit drift")
    return envelope


__all__ = [
    "AUTHORITY_ENVELOPE_NAME",
    "AUTHORITY_ENVELOPE_SCHEMA",
    "AUTHORITY_SIGNATURE_DOMAIN",
    "AUTHORITY_SIGNATURE_NAME",
    "D102_REVOCATION_MANIFEST_NAME",
    "D102_REVOCATION_SCHEMA",
    "D102_REVOCATION_SIGNATURE_DOMAIN",
    "D102_REVOCATION_SIGNATURE_NAME",
    "D105AuthorityError",
    "INDEPENDENT_REVIEW_RECEIPT_NAME",
    "INDEPENDENT_REVIEW_SCHEMA",
    "NONCE_LEDGER_IDENTITY_SCHEMA",
    "TARGET25_DEVELOPMENT_CLAIM_SCOPE",
    "TARGET25_PREPARE_ENVELOPE_NAME",
    "TARGET25_PREPARE_ENVELOPE_SCHEMA",
    "TARGET25_PREPARE_SIGNATURE_DOMAIN",
    "TARGET25_PREPARE_SIGNATURE_NAME",
    "build_d105_authority_envelope",
    "build_d105_target25_prepare_envelope",
    "canonical_bytes",
    "compute_d105_nonce_ledger_identity",
    "consume_authority_nonce_once",
    "consume_target25_prepare_nonce_once",
    "ed25519_public_key_from_seed",
    "load_independent_review_receipt",
    "load_signed_d102_revocation_manifest",
    "load_signed_d105_authority_envelope",
    "load_signed_d105_target25_prepare_envelope",
    "reject_revoked_d102_identity",
    "sha256_bytes",
    "sign_d105_detached",
    "validate_d102_revocation_manifest",
    "validate_d105_authority_envelope",
    "validate_d105_target25_prepare_envelope",
    "validate_independent_review_receipt",
]
