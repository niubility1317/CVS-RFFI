#!/usr/bin/env python
"""Bridge one real IQ record through the frozen Phase3 local-evidence path.

This entry point is deliberately a small technical vertical slice.  It loads
an externally anchored single-control bundle, validates one registered IQ
record and one truth-free deployment context, emits the immutable
``cvs.phase3.local_evidence.v3`` artifact, then proves the CARE N=1 identity
and CIRF byte-preserving N=1 passthrough.  It never invokes a scorer, creates
an N>1 event, or joins records by TX/sig/RX fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_single_control_bundle_v1 as scb
from cvsrffi import phase3_cirf_track_v3 as cirf


BRIDGE_SCHEMA = "cvs.phase3.scb_real_n1_bridge.v1"
LOCAL_EVIDENCE_SCHEMA = "cvs.phase3.local_evidence.v3"
CARE_RECEIPT_SCHEMA = "cvs.phase3.scb_real_n1_bridge.care_identity_receipt.v1"
CIRF_RECEIPT_SCHEMA = "cvs.phase3.scb_real_n1_bridge.cirf_passthrough_receipt.v1"
MANIFEST_SCHEMA = "cvs.phase3.scb_real_n1_bridge.manifest.v1"

CONTEXT_FIELDS = frozenset(
    {
        "linkage_mode",
        "proxy_group_id",
        "satellite_reception_id",
        "node_id",
        "base_manifest_id",
        "correlation_group_id",
        "delay_ms",
        "deadline_ms",
        "sealed_at_ms",
    }
)
CONTEXT_IDENTIFIER_FIELDS = (
    "proxy_group_id",
    "satellite_reception_id",
    "node_id",
    "base_manifest_id",
    "correlation_group_id",
)
CONTEXT_NUMERIC_FIELDS = ("delay_ms", "deadline_ms", "sealed_at_ms")
OUTPUT_NAMES = (
    "local_evidence.json",
    "care_n1_identity_receipt.json",
    "cirf_n1_passthrough_receipt.json",
    "bridge_manifest.json",
)


class BridgeContractError(ValueError):
    """Raised when the real-N=1 bridge contract is violated."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_hex(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BridgeContractError(f"{field} must be a 64-character SHA256")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise BridgeContractError(f"{field} must be hexadecimal SHA256")
    return lowered


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return cirf.canonical_json(dict(value)).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BridgeContractError("bridge artifact is not canonical JSON") from exc


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BridgeContractError(f"context {field} must be a non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise BridgeContractError(f"context {field} must be Unicode NFC")
    return value


def _require_nonnegative_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise BridgeContractError(f"context {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise BridgeContractError(f"context {field} must be finite and non-negative")
    return number


def validate_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact nine-field, proxy-unverified context allowlist."""

    if not isinstance(context, Mapping):
        raise BridgeContractError("context JSON must be an object")
    keys = set(context)
    if keys != set(CONTEXT_FIELDS):
        unexpected = sorted(keys.difference(CONTEXT_FIELDS))
        missing = sorted(CONTEXT_FIELDS.difference(keys))
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if unexpected:
            detail.append(f"unexpected={unexpected}")
        raise BridgeContractError("context exact allowlist mismatch: " + ", ".join(detail))
    if context.get("linkage_mode") != "proxy_unverified":
        raise BridgeContractError("context linkage_mode must be proxy_unverified")
    normalized: dict[str, Any] = {"linkage_mode": "proxy_unverified"}
    for field in CONTEXT_IDENTIFIER_FIELDS:
        normalized[field] = _require_identifier(context[field], field=field)
    for field in CONTEXT_NUMERIC_FIELDS:
        normalized[field] = _require_nonnegative_finite(context[field], field=field)
    return normalized


def _read_context(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise BridgeContractError("context JSON must be a regular file")
    try:
        parsed = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeContractError("context JSON is unreadable") from exc
    return validate_context(parsed)


def validate_iq_file(iq_path: str | Path, expected_iq_sha256: str) -> np.ndarray:
    """Read one finite ``.npy`` IQ array and close its raw-file SHA256."""

    source = Path(iq_path)
    if source.is_symlink() or not source.is_file():
        raise BridgeContractError("iq.npy must be a regular file")
    expected = _sha256_hex(expected_iq_sha256, field="expected IQ SHA256")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise BridgeContractError("iq.npy is unreadable") from exc
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise BridgeContractError("IQ SHA256 does not match the pre-registered digest")
    try:
        array = np.load(source, allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise BridgeContractError("iq.npy must be a readable non-object NumPy array") from exc
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise BridgeContractError("IQ must have shape [T,2] or [2,T]")
    if array.shape[0] != 2 and array.shape[1] != 2:
        raise BridgeContractError("IQ must have shape [T,2] or [2,T]")
    time_length = int(array.shape[1] if array.shape[0] == 2 else array.shape[0])
    if time_length <= 0:
        raise BridgeContractError("IQ time dimension must be non-empty")
    # bool/object/string arrays are not IQ numeric samples.  Complex IQ is
    # also rejected because the SCB contract consumes two real I/Q channels.
    if array.dtype.kind not in "iuf":
        raise BridgeContractError("IQ must be a real numeric array")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise BridgeContractError("IQ must contain finite numeric samples") from exc
    if not bool(np.all(finite)):
        raise BridgeContractError("IQ must contain only finite samples")
    return array


def _ensure_new_output_root(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite an existing bridge output root")
    if target.name in {"", ".", ".."} or not target.parent.is_dir():
        raise BridgeContractError("bridge output parent must exist and root name must be concrete")
    staging = target.parent / f".{target.name}.scb-n1-staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("refusing to reuse an SCB N=1 staging root")
    return target


def _write_json(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite bridge artifact {path.name}")
    path.write_bytes(raw)


def _atomic_write_outputs(output_dir: Path, artifacts: Mapping[str, bytes]) -> None:
    if tuple(artifacts) != OUTPUT_NAMES:
        raise BridgeContractError("bridge artifact allowlist drift")
    staging = output_dir.parent / f".{output_dir.name}.scb-n1-staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("refusing to reuse an SCB N=1 staging root")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        for name in OUTPUT_NAMES:
            _write_json(staging / name, artifacts[name])
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists() and staging.is_dir():
            # The staging path is this process's exact, newly-created root.
            for child in staging.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
            staging.rmdir()
        raise


def run_bridge(
    *,
    bundle_root: str | Path,
    expected_content_root: str,
    iq_path: str | Path,
    expected_iq_sha256: str,
    context_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute the complete real single-IQ bridge and atomically emit four files."""

    target = _ensure_new_output_root(output_dir)
    expected_root = _sha256_hex(expected_content_root, field="external expected content root")
    context = _read_context(context_json)
    iq = validate_iq_file(iq_path, expected_iq_sha256)

    try:
        bundle = scb.load_bundle(
            bundle_root,
            expected_content_root=expected_root,
            device="cpu",
            expected_bundle_status=scb.BUNDLE_STATUS,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise BridgeContractError("SCB bundle failed the external-root or status contract") from exc

    try:
        local_evidence = scb.local_evidence_from_bundle(bundle, raw_iq=iq, context=context)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise BridgeContractError("SCB local evidence generation failed closed") from exc
    if local_evidence.get("schema_version") != LOCAL_EVIDENCE_SCHEMA:
        raise BridgeContractError("SCB did not emit cvs.phase3.local_evidence.v3")
    local_bytes = _canonical_json_bytes(local_evidence)

    try:
        care_receipt = scb.care_n1_parity(local_evidence)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        # The existing CARE fusion contract drops every late reception before
        # its N=1 identity branch and returns ``NO_VALID_RECEPTION``.  SCB has
        # already converted this exact condition to ``SCB_CONTEXT_DEFER``;
        # do not invent a second timeout reason or write a misleading receipt.
        # Fail closed for the late row, which is explicitly allowed by the
        # bridge contract.
        if local_evidence.get("local_decision") == "defer":
            raise BridgeContractError("late N=1 context cannot satisfy CARE identity; fail-closed") from exc
        raise BridgeContractError("CARE N=1 identity parity failed") from exc
    if (
        care_receipt.get("decision") != local_evidence.get("local_decision")
        or care_receipt.get("label") != local_evidence.get("local_label")
        or care_receipt.get("reason_code") != local_evidence.get("reason_code")
        or care_receipt.get("p_fused") != local_evidence.get("p_local")
    ):
        raise BridgeContractError("CARE N=1 p_local/decision/label/reason identity failed")
    care_output = {
        "schema": CARE_RECEIPT_SCHEMA,
        "source_schema": LOCAL_EVIDENCE_SCHEMA,
        "technical_only": True,
        "performance_result": False,
        "n_sat": 1,
        "p_local_identity": True,
        "decision_identity": True,
        "label_identity": True,
        "reason_identity": True,
        "identity_receipt": care_receipt,
    }

    try:
        passthrough = cirf.n1_passthrough_bytes(local_bytes)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise BridgeContractError("CIRF N=1 passthrough validation failed") from exc
    if passthrough != local_bytes:
        raise BridgeContractError("CIRF N=1 passthrough changed LocalEvidence bytes")
    local_sha = _sha256_bytes(local_bytes)
    passthrough_sha = _sha256_bytes(passthrough)
    cirf_output = {
        "schema": CIRF_RECEIPT_SCHEMA,
        "source_schema": LOCAL_EVIDENCE_SCHEMA,
        "technical_only": True,
        "performance_result": False,
        "truth_sidecar_opened": False,
        "n_sat": 1,
        "shot_count": 1,
        "byte_identical": True,
        "local_evidence_sha256": local_sha,
        "passthrough_sha256": passthrough_sha,
        "local_evidence_bytes": len(local_bytes),
        "passthrough_bytes": len(passthrough),
    }

    artifact_payloads: dict[str, bytes] = {
        "local_evidence.json": local_bytes,
        "care_n1_identity_receipt.json": _canonical_json_bytes(care_output),
        "cirf_n1_passthrough_receipt.json": _canonical_json_bytes(cirf_output),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "bundle_status": scb.BUNDLE_STATUS,
        "bundle_content_root": bundle.content_root,
        "iq_file_sha256": _sha256_hex(expected_iq_sha256, field="expected IQ SHA256"),
        "context_fields": sorted(CONTEXT_FIELDS),
        "local_evidence_schema": LOCAL_EVIDENCE_SCHEMA,
        "technical_only": True,
        "performance_result": False,
        "truth_sidecar_opened": False,
        "same_event_claim": False,
        "collaborative_gain_claim": False,
        "n_sat": 1,
        "shot_count": 1,
        "records_emitted": 1,
        "scorer_called": False,
        "tx_sig_rx_grouping": False,
        "artifact_files": {
            name: _sha256_bytes(raw) for name, raw in artifact_payloads.items()
        },
    }
    artifact_payloads["bridge_manifest.json"] = _canonical_json_bytes(manifest)
    _atomic_write_outputs(target, artifact_payloads)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", "--bundle", dest="bundle_root", required=True, help="SCB bundle directory")
    parser.add_argument(
        "--expected-content-root",
        "--external-expected-content-root",
        dest="expected_content_root",
        required=True,
        help="externally pre-registered SCB content root SHA256",
    )
    parser.add_argument("--iq", "--iq-path", dest="iq_path", required=True, help="one IQ .npy file")
    parser.add_argument(
        "--expected-iq-sha256",
        "--iq-sha256",
        dest="expected_iq_sha256",
        required=True,
        help="pre-registered raw iq.npy SHA256",
    )
    parser.add_argument(
        "--context-json", "--context", dest="context_json", required=True, help="exact nine-field truth-free context JSON"
    )
    parser.add_argument("--output-dir", required=True, help="new, non-existing output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_bridge(
            bundle_root=args.bundle_root,
            expected_content_root=args.expected_content_root,
            iq_path=args.iq_path,
            expected_iq_sha256=args.expected_iq_sha256,
            context_json=args.context_json,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"phase3_scb_real_n1_bridge: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
