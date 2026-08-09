"""CIRF-Track v3 G0 technical-contract closure.

This module deliberately implements *only* the frozen, truth-free G0
properties from ``phase3_cirf_track_v3_multiaxis_revision_20260809.md``.
It is not a scorer and never accepts a query truth/role sidecar.  The small
CPU-only implementation is intentionally explicit: immutable JSON contracts
are more useful than a convenient but opaque end-to-end simulator at this
stage.

The public functions are split into four planes:

* event ledger and replay/counter protection;
* factorised local opinions, reference-prior restoration, correlation and QP;
* four-way split, interval/conformal/risk and finite transcript contracts;
* a structural anonymous-track state machine plus capacity preflight.

All outputs carry ``TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`` when they are
created by the companion G0 entry point.  Nothing here makes an operational,
performance, unknown-FAR, or identity-authorization claim.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .phase3_care_poe import (
    EvidenceError,
    canonical_json,
    sha256_json,
    validate_local_evidence,
)


SCHEMA = "cvs.phase3.cirf_track_v3.g0.v1"
LEDGER_SCHEMA = "cvs.phase3.cirf_track_v3.event_ledger.v1"
FACTOR_SCHEMA = "cvs.phase3.cirf_track_v3.factorized_evidence.v1"
KERNEL_SCHEMA = "cvs.phase3.cirf_track_v3.kernel_contract.v1"
SPLIT_SCHEMA = "cvs.phase3.cirf_track_v3.four_split_ledger.v1"
INTERVAL_SCHEMA = "cvs.phase3.cirf_track_v3.interval_contract.v1"
TRANSCRIPT_SCHEMA = "cvs.phase3.cirf_track_v3.transcript_contract.v2"
SCHEDULER_REPLAY_SCHEMA = "cvs.phase3.cirf_track_v3.scheduler_replay_authority.v1"
SCHEDULER_LEDGER_SCHEMA = "cvs.phase3.cirf_track_v3.scheduler_session_ledger.v1"
TRACK_SCHEMA = "cvs.phase3.cirf_track_v3.track_revision.v1"
EVENT_AUTHORITY_SCHEMA = "cvs.phase3.cirf_track_v3.event_authority_receipt.v2"
EVENT_CANDIDATE_SCHEMA = "cvs.phase3.cirf_track_v3.event_candidate_receipt.v1"
FUSION_PLAN_SCHEMA = "cvs.phase3.cirf_track_v3.fusion_plan.v1"
KERNEL_FIT_CELL_SCHEMA = "cvs.phase3.cirf_track_v3.kernel_fit_cell_receipt.v1"
FALLBACK_MAD_SCHEMA = "cvs.phase3.cirf_track_v3.fallback_mad_receipt.v1"
RISK_RECEIPT_SCHEMA = "cvs.phase3.cirf_track_v3.risk_receipt.v1"
MANIFEST_SCHEMA = "cvs.phase3.cirf_track_v3.g0_manifest.v2"
TECHNICAL_NO_PERFORMANCE = "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT"

MAX_ORIGINS = 5
MAX_CLASS_PLUS_UNKNOWN = 32
MAX_CONTEXT_BUCKETS = 12
MAX_STOCHASTIC_ERROR_SOURCES = 4
MAX_PRIORS = 4
MAX_TRANSCRIPTS = 65536
MAX_MEMORY_BYTES = 8 * 1024**3
MAX_PRIMITIVE_OPERATIONS = 2_000_000_000
EPS = 1e-12
FEASIBILITY_TOL = 1e-10
SCHEDULER_TEMPLATE_BINDING_HASH = hashlib.sha256(b"CIRF_TRACK_V3_UNBOUND_SCHEDULER_TEMPLATE").hexdigest()


class CIRFContractError(EvidenceError):
    """Raised when a frozen CIRF-Track v3 G0 contract is violated."""


class CapacityRejected(CIRFContractError):
    """Raised only by :func:`require_capacity` after deterministic preflight."""


# These must be rejected before copying a mapping: guarded test mappings can
# therefore prove that no truth or role value was read by the predictor.
FORBIDDEN_PREDICTOR_FIELDS = frozenset(
    {
        "role",
        "true_label",
        "query_truth",
        "query_role",
        "batch_class_counts",
        "class_quota",
        "quota",
        "global_assignment",
        "global_reassignment",
        "hungarian_assignment",
        "optimal_transport",
        "credential",
        "registration_authorized",
        "track_identity",
        "z_track",
        "source_features",
        "clean_iq",
        "raw_iq",
    }
)

IDENTITY_BLIND_FIELDS = frozenset(
    {
        "tx",
        "tx_id",
        "class",
        "class_id",
        "class_handle",
        "label",
        "identity",
        "credential",
        "role",
        "prediction",
        "local_decision",
        "local_label",
        "z_id",
        "z_track",
        "mac_address",
        "device_address",
        "payload",
    }
)


def _mapping_keys(value: Mapping[str, Any], name: str) -> set[str]:
    if not isinstance(value, Mapping):
        raise CIRFContractError(f"{name} must be a mapping")
    # Mapping.keys() does not need to dereference values.  Keep this first.
    keys = set(value.keys())
    forbidden = sorted(keys.intersection(FORBIDDEN_PREDICTOR_FIELDS))
    if forbidden:
        raise CIRFContractError(f"forbidden predictor fields in {name}: {forbidden}")
    return keys


def _copy_allowed(
    value: Mapping[str, Any],
    *,
    name: str,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
) -> dict[str, Any]:
    keys = _mapping_keys(value, name)
    unexpected = sorted(keys.difference(allowed))
    missing = sorted(required.difference(keys))
    if unexpected:
        raise CIRFContractError(f"unexpected {name} fields: {unexpected}")
    if missing:
        raise CIRFContractError(f"missing {name} fields: {missing}")
    return {key: value[key] for key in keys}


def _copy_without_hash(value: Mapping[str, Any], *, name: str, hash_field: str) -> dict[str, Any]:
    """Copy only after the no-truth/no-role key guard has run."""

    keys = _mapping_keys(value, name)
    return {key: value[key] for key in keys if key != hash_field}


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CIRFContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise CIRFContractError(f"{name} must be finite")
    return result


def _sha256_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CIRFContractError(f"{name} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CIRFContractError(f"{name} must be a SHA256 hex digest") from exc
    return value.lower()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CIRFContractError(f"{name} must be a non-empty string")
    return value


def _interval(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise CIRFContractError(f"{name} must be a length-two interval")
    lo, hi = _finite(value[0], name), _finite(value[1], name)
    if lo > hi:
        raise CIRFContractError(f"{name} lower bound exceeds upper bound")
    return [lo, hi]


def _intersection(intervals: Sequence[Sequence[float]]) -> list[float] | None:
    if not intervals:
        return None
    lo = max(float(item[0]) for item in intervals)
    hi = min(float(item[1]) for item in intervals)
    return [lo, hi] if lo <= hi else None


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise CIRFContractError("logsumexp requires values")
    maximum = max(values)
    if not math.isfinite(maximum):
        raise CIRFContractError("logsumexp input must be finite")
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _logsigmoid(value: float) -> float:
    # Both branches remain finite for the frozen clipped range.
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _log1m_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -value - math.log1p(math.exp(-value))
    return -math.log1p(math.exp(value))


def _logsoftmax(values: Sequence[float]) -> list[float]:
    normalizer = _logsumexp(values)
    return [float(value) - normalizer for value in values]


def _stable_matrix_hash(matrix: Sequence[Sequence[float]]) -> str:
    return sha256_json([[float(value) for value in row] for row in matrix])


# ---------------------------------------------------------------------------
# Event plane: identity-blind ledger and counter/replay guard
# ---------------------------------------------------------------------------

LEDGER_REQUIRED = frozenset(
    {
        "schema_version",
        "reception_id",
        "node_id",
        "roster_epoch",
        "clock_state_id",
        "clock_error_bound",
        "drift_bound",
        "capture_time_interval",
        "receiver_ephemeris_interval",
        "propagation_delay_interval",
        "carrier_frequency_interval",
        "doppler_residual_interval",
        "beam_id",
        "visibility_cell",
        "transmission_opportunity",
        "waveform_digest",
        "nonce",
        "monotonic_counter",
        "key_epoch",
        "revocation_epoch",
        "evidence_origin_id",
        "revoked",
    }
)
LEDGER_ALLOWED = LEDGER_REQUIRED | {"ledger_hash"}
OPPORTUNITY_FIELDS = frozenset(
    {"roster_epoch", "time_slot", "band", "beam", "visibility_cell", "schedule_epoch"}
)


def _identity_blind_mapping(
    value: Mapping[str, Any], *, name: str, required: set[str] | frozenset[str]
) -> dict[str, Any]:
    # Opportunity and waveform descriptors are deliberately much more strict
    # than generic predictor records.  A field name alone is enough to reject.
    if not isinstance(value, Mapping):
        raise CIRFContractError(f"{name} must be a mapping")
    keys = set(value.keys())
    bad = sorted(keys.intersection(IDENTITY_BLIND_FIELDS | FORBIDDEN_PREDICTOR_FIELDS))
    if bad:
        raise CIRFContractError(f"{name} contains identity-bearing fields: {bad}")
    unexpected = sorted(keys.difference(required))
    missing = sorted(set(required).difference(keys))
    if unexpected or missing:
        raise CIRFContractError(f"{name} fields mismatch: unexpected={unexpected}, missing={missing}")
    result = {key: value[key] for key in keys}
    for key in required:
        _nonempty_string(result[key], f"{name}.{key}")
    return result


def _validate_ledger(payload: Mapping[str, Any], *, require_hash: bool) -> dict[str, Any]:
    raw = _copy_allowed(payload, name="event ledger", allowed=LEDGER_ALLOWED, required=LEDGER_REQUIRED)
    if require_hash:
        digest = _sha256_hex(raw.get("ledger_hash"), "ledger_hash")
        unsigned = dict(raw)
        unsigned.pop("ledger_hash", None)
        if sha256_json(unsigned) != digest:
            raise CIRFContractError("event ledger hash mismatch")
    if raw["schema_version"] != LEDGER_SCHEMA:
        raise CIRFContractError("unsupported event ledger schema")
    for field_name in (
        "reception_id",
        "node_id",
        "roster_epoch",
        "clock_state_id",
        "beam_id",
        "visibility_cell",
        "nonce",
        "key_epoch",
        "revocation_epoch",
        "evidence_origin_id",
    ):
        _nonempty_string(raw[field_name], field_name)
    raw["clock_error_bound"] = _finite(raw["clock_error_bound"], "clock_error_bound")
    raw["drift_bound"] = _finite(raw["drift_bound"], "drift_bound")
    if raw["clock_error_bound"] < 0.0 or raw["drift_bound"] < 0.0:
        raise CIRFContractError("clock bounds must be non-negative")
    for field_name in (
        "capture_time_interval",
        "receiver_ephemeris_interval",
        "propagation_delay_interval",
        "carrier_frequency_interval",
        "doppler_residual_interval",
    ):
        raw[field_name] = _interval(raw[field_name], field_name)
    opportunity = _identity_blind_mapping(
        raw["transmission_opportunity"], name="transmission_opportunity", required=OPPORTUNITY_FIELDS
    )
    if opportunity["roster_epoch"] != raw["roster_epoch"]:
        raise CIRFContractError("opportunity roster epoch mismatch")
    if opportunity["beam"] != raw["beam_id"] or opportunity["visibility_cell"] != raw["visibility_cell"]:
        raise CIRFContractError("opportunity beam/visibility mismatch")
    raw["transmission_opportunity"] = opportunity
    digest = raw["waveform_digest"]
    if digest is not None:
        raw["waveform_digest"] = _sha256_hex(digest, "waveform_digest")
    if isinstance(raw["monotonic_counter"], bool) or not isinstance(raw["monotonic_counter"], int):
        raise CIRFContractError("monotonic_counter must be a non-negative integer")
    if raw["monotonic_counter"] < 0:
        raise CIRFContractError("monotonic_counter must be non-negative")
    if raw["revoked"] not in {True, False}:
        raise CIRFContractError("revoked must be boolean")
    return raw


def seal_event_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal a strictly identity-blind pre-query ledger receipt."""

    unsigned = _copy_without_hash(payload, name="event ledger", hash_field="ledger_hash")
    normalized = _validate_ledger(unsigned, require_hash=False)
    normalized["ledger_hash"] = sha256_json(normalized)
    return normalized


def validate_event_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_ledger(payload, require_hash=True)


def transmission_opportunity_id(ledger: Mapping[str, Any]) -> str:
    """Canonical identity-blind opportunity identifier for a sealed ledger."""

    valid = validate_event_ledger(ledger)
    return sha256_json(valid["transmission_opportunity"])


def corrected_emission_interval(ledger: Mapping[str, Any]) -> list[float]:
    """Return capture-minus-propagation interval including clock uncertainty."""

    valid = validate_event_ledger(ledger)
    capture = valid["capture_time_interval"]
    propagation = valid["propagation_delay_interval"]
    uncertainty = valid["clock_error_bound"] + valid["drift_bound"]
    return [capture[0] - uncertainty - propagation[1], capture[1] + uncertainty - propagation[0]]


@dataclass
class ReplayGuard:
    """Monotonic counter/nonce guard used before same-event certification."""

    _by_counter: dict[tuple[str, str, int], str] = field(default_factory=dict)
    _high_counter: dict[tuple[str, str], int] = field(default_factory=dict)
    _nonces: set[tuple[str, str, str]] = field(default_factory=set)

    def accept(self, ledger: Mapping[str, Any]) -> None:
        valid = validate_event_ledger(ledger)
        if valid["revoked"]:
            raise CIRFContractError("REVOKED_LEDGER")
        epoch_key = (valid["node_id"], valid["key_epoch"])
        counter_key = (*epoch_key, valid["monotonic_counter"])
        previous = self._by_counter.get(counter_key)
        if previous is not None:
            if previous == valid["ledger_hash"]:
                raise CIRFContractError("REPLAY_DETECTED")
            raise CIRFContractError("COUNTER_FORK_DETECTED")
        high = self._high_counter.get(epoch_key)
        if high is not None and valid["monotonic_counter"] <= high:
            raise CIRFContractError("COUNTER_ROLLBACK_DETECTED")
        nonce_key = (*epoch_key, valid["nonce"])
        if nonce_key in self._nonces:
            raise CIRFContractError("NONCE_REPLAY_DETECTED")
        self._by_counter[counter_key] = valid["ledger_hash"]
        self._high_counter[epoch_key] = valid["monotonic_counter"]
        self._nonces.add(nonce_key)

    def accept_many(self, ledgers: Sequence[Mapping[str, Any]]) -> None:
        """Atomically accept a certificate-sized batch of ledgers."""

        trial = ReplayGuard(
            _by_counter=dict(self._by_counter),
            _high_counter=dict(self._high_counter),
            _nonces=set(self._nonces),
        )
        for ledger in ledgers:
            trial.accept(ledger)
        self._by_counter = trial._by_counter
        self._high_counter = trial._high_counter
        self._nonces = trial._nonces


def _assert_identity_blind_tree(value: Any, name: str) -> None:
    """Reject identity/truth keys at every depth without reading bad values."""

    if isinstance(value, Mapping):
        keys = set(value.keys())
        non_strings = [key for key in keys if not isinstance(key, str)]
        if non_strings:
            raise CIRFContractError(f"{name} keys must be strings")
        bad = sorted(keys.intersection(IDENTITY_BLIND_FIELDS | FORBIDDEN_PREDICTOR_FIELDS))
        if bad:
            raise CIRFContractError(f"{name} contains forbidden fields: {bad}")
        for key in sorted(keys):
            _assert_identity_blind_tree(value[key], f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_identity_blind_tree(item, f"{name}[{index}]")


def _candidate_hypothesis_hash(candidate: Mapping[str, Any]) -> str:
    # The event-plane candidate is allowed only physical, identity-blind facts.
    if not isinstance(candidate, Mapping):
        raise CIRFContractError("candidate hypothesis must be a mapping")
    keys = set(candidate.keys())
    if not keys:
        raise CIRFContractError("candidate hypothesis must not be empty")
    _assert_identity_blind_tree(candidate, "candidate hypothesis")
    return sha256_json({key: candidate[key] for key in keys})


EVENT_AUTHORITY_REQUIRED = frozenset(
    {
        "schema_version",
        "roster_epoch",
        "opportunity_id",
        "ledger_hashes",
        "physical_constraint_hash",
        "collision_receipt_hash",
        "revocation_receipt_hash",
        "replay_policy_hash",
        "scheduler_replay_store_id",
        "collision_gate_passed",
    }
)
EVENT_CANDIDATE_REQUIRED = frozenset(
    {
        "schema_version",
        "candidate_id",
        "opportunity_id",
        "ledger_hashes",
        "physical_constraint_hash",
        "collision_receipt_hash",
    }
)


def _hash_list(values: Any, name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise CIRFContractError(f"{name} must be a non-empty list")
    normalized = [_sha256_hex(value, name) for value in values]
    if len(normalized) != len(set(normalized)):
        raise CIRFContractError(f"{name} must be unique")
    return sorted(normalized)


def seal_event_authority_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal pre-query replay/revocation/collision authority for one event cell."""

    raw = _copy_without_hash(payload, name="event authority receipt", hash_field="authority_receipt_hash")
    raw = _copy_allowed(
        raw,
        name="event authority receipt",
        allowed=EVENT_AUTHORITY_REQUIRED,
        required=EVENT_AUTHORITY_REQUIRED,
    )
    if raw["schema_version"] != EVENT_AUTHORITY_SCHEMA:
        raise CIRFContractError("unsupported event authority receipt schema")
    raw["roster_epoch"] = _nonempty_string(raw["roster_epoch"], "authority roster_epoch")
    raw["opportunity_id"] = _sha256_hex(raw["opportunity_id"], "authority opportunity_id")
    raw["ledger_hashes"] = _hash_list(raw["ledger_hashes"], "authority ledger_hashes")
    for name in (
        "physical_constraint_hash",
        "collision_receipt_hash",
        "revocation_receipt_hash",
        "replay_policy_hash",
        "scheduler_replay_store_id",
    ):
        raw[name] = _sha256_hex(raw[name], name)
    if raw["collision_gate_passed"] is not True:
        raise CIRFContractError("authority collision gate must be explicitly passed")
    raw["authority_receipt_hash"] = sha256_json(raw)
    return raw


def validate_event_authority_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="event authority receipt",
        allowed=EVENT_AUTHORITY_REQUIRED | {"authority_receipt_hash"},
        required=EVENT_AUTHORITY_REQUIRED | {"authority_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("authority_receipt_hash"), "authority_receipt_hash")
    sealed = seal_event_authority_receipt(raw)
    if sealed["authority_receipt_hash"] != digest:
        raise CIRFContractError("event authority receipt hash mismatch")
    sealed["authority_receipt_hash"] = digest
    return sealed


def seal_same_event_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one constrained, identity-blind physical-event candidate receipt."""

    raw = _copy_without_hash(payload, name="same-event candidate", hash_field="candidate_receipt_hash")
    raw = _copy_allowed(
        raw,
        name="same-event candidate",
        allowed=EVENT_CANDIDATE_REQUIRED,
        required=EVENT_CANDIDATE_REQUIRED,
    )
    if raw["schema_version"] != EVENT_CANDIDATE_SCHEMA:
        raise CIRFContractError("unsupported same-event candidate schema")
    raw["candidate_id"] = _nonempty_string(raw["candidate_id"], "candidate_id")
    raw["opportunity_id"] = _sha256_hex(raw["opportunity_id"], "candidate opportunity_id")
    raw["ledger_hashes"] = _hash_list(raw["ledger_hashes"], "candidate ledger_hashes")
    raw["physical_constraint_hash"] = _sha256_hex(raw["physical_constraint_hash"], "candidate physical_constraint_hash")
    raw["collision_receipt_hash"] = _sha256_hex(raw["collision_receipt_hash"], "candidate collision_receipt_hash")
    _assert_identity_blind_tree(raw, "same-event candidate")
    raw["candidate_receipt_hash"] = sha256_json(raw)
    return raw


def validate_same_event_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="same-event candidate",
        allowed=EVENT_CANDIDATE_REQUIRED | {"candidate_receipt_hash"},
        required=EVENT_CANDIDATE_REQUIRED | {"candidate_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("candidate_receipt_hash"), "candidate_receipt_hash")
    sealed = seal_same_event_candidate(raw)
    if sealed["candidate_receipt_hash"] != digest:
        raise CIRFContractError("same-event candidate receipt hash mismatch")
    sealed["candidate_receipt_hash"] = digest
    return sealed


def same_event_certificate(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    candidate_hypotheses: Sequence[Mapping[str, Any]],
    authority_receipt: Mapping[str, Any] | None = None,
    replay_guard: ReplayGuard | None = None,
) -> dict[str, Any]:
    """Create a certificate only for one physically feasible, blinded event.

    Ambiguity is a normal negative outcome, not a guess.  The return payload is
    always technical and never attaches an identity or truth label.
    """

    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "certificate_status": "NO_CERTIFICATE",
        "reason_code": "UNINITIALIZED",
        "shot_count": 1,
    }
    try:
        if replay_guard is None:
            raise CIRFContractError("REPLAY_GUARD_REQUIRED")
        if authority_receipt is None:
            raise CIRFContractError("EVENT_AUTHORITY_RECEIPT_REQUIRED")
        valid = [validate_event_ledger(row) for row in ledgers]
        if len(valid) < 2:
            raise CIRFContractError("INSUFFICIENT_RECEPTIONS")
        receptions = [row["reception_id"] for row in valid]
        nodes = [row["node_id"] for row in valid]
        if len(set(receptions)) != len(receptions) or len(set(nodes)) != len(nodes):
            raise CIRFContractError("DUPLICATE_RECEPTION_OR_NODE")
        if any(row["revoked"] for row in valid):
            raise CIRFContractError("REVOKED_LEDGER")
        if len({transmission_opportunity_id(row) for row in valid}) != 1:
            raise CIRFContractError("OPPORTUNITY_MISMATCH")
        if len({row["roster_epoch"] for row in valid}) != 1:
            raise CIRFContractError("ROSTER_EPOCH_MISMATCH")
        emission = _intersection([corrected_emission_interval(row) for row in valid])
        if emission is None:
            raise CIRFContractError("CLOCK_PROPAGATION_DISJOINT")
        frequency_intervals = [
            [
                row["carrier_frequency_interval"][0] - row["doppler_residual_interval"][1],
                row["carrier_frequency_interval"][1] - row["doppler_residual_interval"][0],
            ]
            for row in valid
        ]
        frequency = _intersection(frequency_intervals)
        if frequency is None:
            raise CIRFContractError("FREQUENCY_DOPPLER_DISJOINT")
        waveform = {row["waveform_digest"] for row in valid}
        if len(waveform) > 1:
            raise CIRFContractError("WAVEFORM_DIGEST_MISMATCH")
        authority = validate_event_authority_receipt(authority_receipt)
        ledger_hashes = sorted(row["ledger_hash"] for row in valid)
        opportunity_id = transmission_opportunity_id(valid[0])
        if authority["ledger_hashes"] != ledger_hashes:
            raise CIRFContractError("AUTHORITY_LEDGER_BINDING_MISMATCH")
        if authority["roster_epoch"] != valid[0]["roster_epoch"]:
            raise CIRFContractError("AUTHORITY_ROSTER_BINDING_MISMATCH")
        if authority["opportunity_id"] != opportunity_id:
            raise CIRFContractError("AUTHORITY_OPPORTUNITY_BINDING_MISMATCH")
        if authority["collision_gate_passed"] is not True:
            raise CIRFContractError("AUTHORITY_COLLISION_GATE_FAILED")
        candidates = [validate_same_event_candidate(row) for row in candidate_hypotheses]
        if len(candidates) != 1:
            raise CIRFContractError("AMBIGUOUS_EVENT_HYPOTHESES")
        candidate = candidates[0]
        if (
            candidate["ledger_hashes"] != ledger_hashes
            or candidate["opportunity_id"] != opportunity_id
            or candidate["physical_constraint_hash"] != authority["physical_constraint_hash"]
            or candidate["collision_receipt_hash"] != authority["collision_receipt_hash"]
        ):
            raise CIRFContractError("CANDIDATE_AUTHORITY_BINDING_MISMATCH")
        replay_guard.accept_many(sorted(valid, key=lambda item: (item["node_id"], item["monotonic_counter"])))
        result.update(
            {
                "certificate_status": "CERTIFIED",
                "reason_code": "UNIQUE_PHYSICAL_HYPOTHESIS",
                "roster_epoch": valid[0]["roster_epoch"],
                "opportunity_id": opportunity_id,
                "emission_interval": emission,
                "frequency_interval": frequency,
                "candidate_hypothesis_hash": candidate["candidate_receipt_hash"],
                "authority_receipt_hash": authority["authority_receipt_hash"],
                "ledger_hashes": ledger_hashes,
            }
        )
    except CIRFContractError as exc:
        result["reason_code"] = str(exc)
        result["ledger_hashes"] = sorted(
            row.get("ledger_hash", "") for row in ledgers if isinstance(row, Mapping)
        )
    unsigned = dict(result)
    result["certificate_hash"] = sha256_json(unsigned)
    return result


def validate_same_event_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "evidence_level",
        "certificate_status",
        "reason_code",
        "shot_count",
        "certificate_hash",
    }
    raw = _copy_allowed(
        certificate,
        name="same-event certificate",
        allowed=required
        | {
            "roster_epoch",
            "opportunity_id",
            "emission_interval",
            "frequency_interval",
            "candidate_hypothesis_hash",
            "authority_receipt_hash",
            "ledger_hashes",
        },
        required=required,
    )
    if raw["schema_version"] != SCHEMA or raw["evidence_level"] != TECHNICAL_NO_PERFORMANCE:
        raise CIRFContractError("invalid same-event certificate schema/evidence level")
    digest = _sha256_hex(raw["certificate_hash"], "certificate_hash")
    unsigned = dict(raw)
    unsigned.pop("certificate_hash", None)
    if sha256_json(unsigned) != digest:
        raise CIRFContractError("same-event certificate hash mismatch")
    if raw["certificate_status"] not in {"CERTIFIED", "NO_CERTIFICATE"}:
        raise CIRFContractError("invalid certificate status")
    if raw["shot_count"] != 1:
        raise CIRFContractError("same-event certificate must count one shot")
    if raw["certificate_status"] == "CERTIFIED":
        for name in ("authority_receipt_hash", "candidate_hypothesis_hash", "opportunity_id", "ledger_hashes"):
            if name not in raw:
                raise CIRFContractError("certified event receipt lacks authority binding")
    return raw


# ---------------------------------------------------------------------------
# Decision plane: factorised finite opinions and reference-prior restoration
# ---------------------------------------------------------------------------

ASSET_HASH_FIELDS = frozenset(
    {
        "base_checkpoint_hash",
        "class_registry_hash",
        "unknown_converter_hash",
        "gate_calibrator_hash",
        "registered_temperature_hash",
        "reference_prior_hash",
        "tier_codec_hash",
        "receiver_state_hash",
    }
)
FACTOR_REQUIRED = frozenset(
    {
        "schema_version",
        "local_evidence",
        "event_ledger",
        "evidence_origin_id",
        "context_id",
        "unknown_gate_raw_logit",
        "registered_raw_logits",
        "asset_hashes",
        "gate_calibrator",
        "registered_temperature",
        "reference_prior_id",
        "reference_prior",
        "tier_codec",
    }
)
FACTOR_ALLOWED = FACTOR_REQUIRED | {"factor_hash"}


def _probability_vector(prior: Mapping[str, Any], labels: Sequence[str], name: str) -> dict[str, float]:
    if not isinstance(prior, Mapping):
        raise CIRFContractError(f"{name} must be a mapping")
    keys = _mapping_keys(prior, name)
    expected = set(labels) | {"__unknown__"}
    if keys != expected:
        raise CIRFContractError(f"{name} keys must equal registered labels plus __unknown__")
    normalized = {key: _finite(prior[key], f"{name}.{key}") for key in expected}
    if any(value <= 0.0 for value in normalized.values()):
        raise CIRFContractError(f"{name} must be strictly positive")
    if abs(sum(normalized.values()) - 1.0) > 1e-12:
        raise CIRFContractError(f"{name} must sum to one")
    return {key: normalized[key] for key in list(labels) + ["__unknown__"]}


def _validate_factorized(payload: Mapping[str, Any], *, require_hash: bool) -> dict[str, Any]:
    raw = _copy_allowed(payload, name="factorized evidence", allowed=FACTOR_ALLOWED, required=FACTOR_REQUIRED)
    if require_hash:
        digest = _sha256_hex(raw.get("factor_hash"), "factor_hash")
        unsigned = dict(raw)
        unsigned.pop("factor_hash", None)
        if sha256_json(unsigned) != digest:
            raise CIRFContractError("factorized evidence hash mismatch")
    if raw["schema_version"] != FACTOR_SCHEMA:
        raise CIRFContractError("unsupported factorized evidence schema")
    # Preserve and strengthen the CARE-PoE LocalEvidence contract rather than
    # accepting a look-alike mapping.
    local_keys = _mapping_keys(raw["local_evidence"], "local_evidence")
    if local_keys.intersection(FORBIDDEN_PREDICTOR_FIELDS):
        raise CIRFContractError("forbidden LocalEvidence field")
    local = validate_local_evidence(raw["local_evidence"])
    ledger = validate_event_ledger(raw["event_ledger"])
    if local["satellite_reception_id"] != ledger["reception_id"]:
        raise CIRFContractError("LocalEvidence/ledger reception mismatch")
    if local["node_id"] != ledger["node_id"]:
        raise CIRFContractError("LocalEvidence/ledger node mismatch")
    raw["local_evidence"] = local
    raw["event_ledger"] = ledger
    raw["evidence_origin_id"] = _nonempty_string(raw["evidence_origin_id"], "evidence_origin_id")
    if raw["evidence_origin_id"] != ledger["evidence_origin_id"]:
        raise CIRFContractError("evidence origin must match ledger")
    raw["context_id"] = _nonempty_string(raw["context_id"], "context_id")
    raw["unknown_gate_raw_logit"] = _finite(raw["unknown_gate_raw_logit"], "unknown_gate_raw_logit")
    logits = raw["registered_raw_logits"]
    if not isinstance(logits, list) or len(logits) != len(local["class_handles"]):
        raise CIRFContractError("registered_raw_logits dimension mismatch")
    raw["registered_raw_logits"] = [_finite(value, "registered_raw_logits") for value in logits]
    if not isinstance(raw["asset_hashes"], Mapping):
        raise CIRFContractError("asset_hashes must be a mapping")
    asset_keys = _mapping_keys(raw["asset_hashes"], "asset_hashes")
    if asset_keys != ASSET_HASH_FIELDS:
        raise CIRFContractError("asset_hashes keys mismatch")
    raw["asset_hashes"] = {key: _sha256_hex(raw["asset_hashes"][key], key) for key in sorted(asset_keys)}
    if raw["asset_hashes"]["class_registry_hash"] != sha256_json(local["class_handles"]):
        raise CIRFContractError("class registry hash mismatch")
    raw["reference_prior_id"] = _nonempty_string(raw["reference_prior_id"], "reference_prior_id")
    raw["reference_prior"] = _probability_vector(raw["reference_prior"], local["class_handles"], "reference_prior")
    if raw["asset_hashes"]["reference_prior_hash"] != sha256_json(raw["reference_prior"]):
        raise CIRFContractError("reference prior hash mismatch")
    gate = _copy_allowed(
        raw["gate_calibrator"],
        name="gate_calibrator",
        allowed={"a", "b", "fit_asset_hash"},
        required={"a", "b", "fit_asset_hash"},
    )
    gate["a"] = _finite(gate["a"], "gate_calibrator.a")
    gate["b"] = _finite(gate["b"], "gate_calibrator.b")
    if not 1e-3 <= gate["a"] <= 1e3 or not -30.0 <= gate["b"] <= 30.0:
        raise CIRFContractError("gate calibrator parameter domain violation")
    gate["fit_asset_hash"] = _sha256_hex(gate["fit_asset_hash"], "gate_calibrator.fit_asset_hash")
    if raw["asset_hashes"]["gate_calibrator_hash"] != sha256_json(gate):
        raise CIRFContractError("gate calibrator hash mismatch")
    raw["gate_calibrator"] = gate
    temperature = _copy_allowed(
        raw["registered_temperature"],
        name="registered_temperature",
        allowed={"tau", "fit_asset_hash"},
        required={"tau", "fit_asset_hash"},
    )
    temperature["tau"] = _finite(temperature["tau"], "registered_temperature.tau")
    if not 1e-3 <= temperature["tau"] <= 1e3:
        raise CIRFContractError("registered temperature parameter domain violation")
    temperature["fit_asset_hash"] = _sha256_hex(
        temperature["fit_asset_hash"], "registered_temperature.fit_asset_hash"
    )
    if raw["asset_hashes"]["registered_temperature_hash"] != sha256_json(temperature):
        raise CIRFContractError("registered temperature hash mismatch")
    raw["registered_temperature"] = temperature
    codec = _copy_allowed(
        raw["tier_codec"],
        name="tier_codec",
        allowed={"codec_id", "codec_hash"},
        required={"codec_id", "codec_hash"},
    )
    codec["codec_id"] = _nonempty_string(codec["codec_id"], "tier_codec.codec_id")
    codec["codec_hash"] = _sha256_hex(codec["codec_hash"], "tier_codec.codec_hash")
    if raw["asset_hashes"]["tier_codec_hash"] != sha256_json(codec):
        raise CIRFContractError("tier codec hash mismatch")
    raw["tier_codec"] = codec
    return raw


def seal_factorized_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal finite raw-logit evidence bound to CARE-PoE LocalEvidence assets."""

    unsigned = _copy_without_hash(payload, name="factorized evidence", hash_field="factor_hash")
    normalized = _validate_factorized(unsigned, require_hash=False)
    normalized["factor_hash"] = sha256_json(normalized)
    return normalized


def validate_factorized_evidence(
    payload: Mapping[str, Any],
    *,
    expected_assets: Mapping[str, str] | None = None,
    expected_roster_state_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a sealed factor artifact and optional pre-sealed roster assets."""

    valid = _validate_factorized(payload, require_hash=True)
    if expected_assets is not None:
        keys = set(expected_assets.keys())
        if keys != ASSET_HASH_FIELDS:
            raise CIRFContractError("expected asset manifest keys mismatch")
        for key in ASSET_HASH_FIELDS:
            if valid["asset_hashes"][key] != _sha256_hex(expected_assets[key], f"expected_assets.{key}"):
                raise CIRFContractError(f"asset hash drift: {key}")
    if expected_roster_state_hashes is not None:
        node = valid["local_evidence"]["node_id"]
        if node not in expected_roster_state_hashes:
            raise CIRFContractError("node absent from pre-sealed roster")
        expected = _sha256_hex(expected_roster_state_hashes[node], "expected roster receiver state hash")
        if valid["asset_hashes"]["receiver_state_hash"] != expected:
            raise CIRFContractError("receiver state hash drift")
    return valid


def factorized_log_opinion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable finite log probabilities and prior-corrected evidence.

    ``unknown_gate_raw_logit`` is never treated as a C+1 class logit.  It is
    calibrated independently and only then combined with the conditional
    registered softmax branch.
    """

    valid = validate_factorized_evidence(payload)
    gate = valid["gate_calibrator"]
    raw_gate = min(30.0, max(-30.0, valid["unknown_gate_raw_logit"]))
    gate_value = gate["a"] * raw_gate + gate["b"]
    log_u = _logsigmoid(gate_value)
    log_not_u = _log1m_sigmoid(gate_value)
    raw_logits = valid["registered_raw_logits"]
    maximum = max(raw_logits)
    clipped = [min(0.0, max(-30.0, value - maximum)) for value in raw_logits]
    # The frozen parameter is inverse temperature tau=1/T, so it multiplies
    # logits.  Dividing by tau would silently invert the intended calibration.
    log_q = _logsoftmax([value * valid["registered_temperature"]["tau"] for value in clipped])
    log_p = [log_not_u + value for value in log_q] + [log_u]
    if any(not math.isfinite(value) for value in log_p) or abs(_logsumexp(log_p)) > 1e-12:
        raise CIRFContractError("factorized log opinion is not a finite simplex")
    labels = list(valid["local_evidence"]["class_handles"])
    prior = valid["reference_prior"]
    prior_corrected = {
        label: log_p[index] - math.log(prior[label]) for index, label in enumerate(labels)
    }
    prior_corrected["__unknown__"] = log_p[-1] - math.log(prior["__unknown__"])
    return {
        "schema_version": SCHEMA,
        "factor_hash": valid["factor_hash"],
        "evidence_origin_id": valid["evidence_origin_id"],
        "context_id": valid["context_id"],
        "labels": labels,
        "log_probability": {label: log_p[index] for index, label in enumerate(labels)}
        | {"__unknown__": log_p[-1]},
        "prior_corrected_log_evidence": prior_corrected,
        "reference_prior_id": valid["reference_prior_id"],
        "reference_prior_hash": valid["asset_hashes"]["reference_prior_hash"],
        "asset_hashes": dict(valid["asset_hashes"]),
    }


def restore_reference_prior(
    opinion: Mapping[str, Any],
    *,
    common_reference_prior_id: str,
    frozen_transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attest that prior-corrected evidence is comparable in one reference frame.

    A source already using the common reference passes unchanged.  A different
    source requires an explicit frozen transform with an asset hash; no runtime
    inferred conversion or query-batch prior is accepted.
    """

    raw = _copy_allowed(
        opinion,
        name="factorized opinion",
        allowed={
            "schema_version",
            "factor_hash",
            "evidence_origin_id",
            "context_id",
            "labels",
            "log_probability",
            "prior_corrected_log_evidence",
            "reference_prior_id",
            "reference_prior_hash",
            "asset_hashes",
        },
        required={
            "factor_hash",
            "evidence_origin_id",
            "labels",
            "prior_corrected_log_evidence",
            "reference_prior_id",
            "reference_prior_hash",
        },
    )
    common_reference_prior_id = _nonempty_string(common_reference_prior_id, "common_reference_prior_id")
    corrected = dict(raw["prior_corrected_log_evidence"])
    if raw["reference_prior_id"] == common_reference_prior_id:
        transform_hash = None
    else:
        if frozen_transform is None:
            raise CIRFContractError("reference-prior restoration is unavailable")
        transform = _copy_allowed(
            frozen_transform,
            name="reference prior transform",
            allowed={"from_id", "to_id", "from_hash", "offsets", "transform_hash"},
            required={"from_id", "to_id", "from_hash", "offsets", "transform_hash"},
        )
        unsigned = dict(transform)
        digest = _sha256_hex(unsigned.pop("transform_hash"), "reference prior transform hash")
        if sha256_json(unsigned) != digest:
            raise CIRFContractError("reference prior transform hash mismatch")
        if (
            transform["from_id"] != raw["reference_prior_id"]
            or transform["to_id"] != common_reference_prior_id
            or _sha256_hex(transform["from_hash"], "reference prior transform from_hash")
            != raw["reference_prior_hash"]
        ):
            raise CIRFContractError("reference prior transform does not bind the source")
        offsets = transform["offsets"]
        labels = list(raw["labels"]) + ["__unknown__"]
        if not isinstance(offsets, Mapping) or set(offsets.keys()) != set(labels):
            raise CIRFContractError("reference prior transform offsets mismatch")
        # The offsets are a frozen semantic conversion, not a fitted query-time
        # recalibration.  They are permitted only because their hash is sealed.
        corrected = {label: corrected[label] + _finite(offsets[label], "reference prior offset") for label in labels}
        transform_hash = digest
    return {
        "factor_hash": raw["factor_hash"],
        "evidence_origin_id": raw["evidence_origin_id"],
        "labels": list(raw["labels"]),
        "common_reference_prior_id": common_reference_prior_id,
        "prior_corrected_log_evidence": corrected,
        "reference_transform_hash": transform_hash,
    }


def validate_operational_prior_set(
    priors: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> list[dict[str, float]]:
    """Validate a finite, positive, query-predeclared operational prior set."""

    if not priors or len(priors) > MAX_PRIORS:
        raise CIRFContractError("operational prior set must contain 1..4 priors")
    return [_probability_vector(prior, labels, "operational_prior") for prior in priors]


# ---------------------------------------------------------------------------
# Sealed fusion plan: binds one opportunity/availability/context execution
# ---------------------------------------------------------------------------

FUSION_PLAN_REQUIRED = frozenset(
    {
        "schema_version",
        "opportunity_id",
        "availability",
        "context_id",
        "common_reference_prior_id",
        "expected_assets_by_origin",
        "expected_roster_state_hashes",
        "operational_priors",
        "kernel_r_hash",
        "kernel_u_hash",
        "cap_by_origin",
        "component_caps",
        "frozen_transforms",
    }
)


def _validated_asset_manifest(value: Mapping[str, Any], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value.keys()) != ASSET_HASH_FIELDS:
        raise CIRFContractError(f"{name} asset manifest keys mismatch")
    return {key: _sha256_hex(value[key], f"{name}.{key}") for key in sorted(ASSET_HASH_FIELDS)}


def seal_fusion_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal all pre-query objects consumed by a multi-origin fusion call."""

    raw = _copy_without_hash(payload, name="fusion plan", hash_field="fusion_plan_hash")
    raw = _copy_allowed(raw, name="fusion plan", allowed=FUSION_PLAN_REQUIRED, required=FUSION_PLAN_REQUIRED)
    if raw["schema_version"] != FUSION_PLAN_SCHEMA:
        raise CIRFContractError("unsupported fusion plan schema")
    raw["opportunity_id"] = _sha256_hex(raw["opportunity_id"], "fusion plan opportunity_id")
    availability = raw["availability"]
    if not isinstance(availability, list) or not availability or len(availability) != len(set(availability)):
        raise CIRFContractError("fusion plan availability must be unique and non-empty")
    if availability != sorted(availability):
        raise CIRFContractError("fusion plan availability must use canonical sort order")
    raw["availability"] = [_nonempty_string(value, "fusion plan origin") for value in availability]
    raw["context_id"] = _nonempty_string(raw["context_id"], "fusion plan context")
    raw["common_reference_prior_id"] = _nonempty_string(
        raw["common_reference_prior_id"], "fusion plan common reference prior"
    )
    assets = raw["expected_assets_by_origin"]
    if not isinstance(assets, Mapping) or set(assets.keys()) != set(availability):
        raise CIRFContractError("fusion plan assets must exactly cover availability")
    raw["expected_assets_by_origin"] = {
        origin: _validated_asset_manifest(assets[origin], f"fusion plan assets {origin}") for origin in availability
    }
    roster = raw["expected_roster_state_hashes"]
    if not isinstance(roster, Mapping) or not roster:
        raise CIRFContractError("fusion plan roster state manifest must not be empty")
    raw["expected_roster_state_hashes"] = {
        _nonempty_string(node, "fusion plan roster node"): _sha256_hex(value, "fusion plan roster state")
        for node, value in sorted(roster.items())
    }
    if not isinstance(raw["operational_priors"], list) or not raw["operational_priors"] or len(raw["operational_priors"]) > MAX_PRIORS:
        raise CIRFContractError("fusion plan operational priors must contain 1..4 entries")
    priors: list[dict[str, Any]] = []
    for prior in raw["operational_priors"]:
        if not isinstance(prior, Mapping):
            raise CIRFContractError("fusion plan operational prior must be a mapping")
        keys = _mapping_keys(prior, "fusion plan operational prior")
        priors.append({key: _finite(prior[key], "fusion plan operational prior") for key in sorted(keys)})
    raw["operational_priors"] = priors
    raw["kernel_r_hash"] = _sha256_hex(raw["kernel_r_hash"], "fusion plan kernel_r_hash")
    raw["kernel_u_hash"] = _sha256_hex(raw["kernel_u_hash"], "fusion plan kernel_u_hash")
    caps = raw["cap_by_origin"]
    if not isinstance(caps, Mapping) or set(caps.keys()) != set(availability):
        raise CIRFContractError("fusion plan origin caps must exactly cover availability")
    raw["cap_by_origin"] = {origin: _finite(caps[origin], "fusion plan origin cap") for origin in availability}
    if any(not 0.0 <= value <= 1.0 for value in raw["cap_by_origin"].values()):
        raise CIRFContractError("fusion plan origin cap must lie in [0,1]")
    component_caps = raw["component_caps"]
    if not isinstance(component_caps, Mapping) or not component_caps:
        raise CIRFContractError("fusion plan component caps must not be empty")
    raw["component_caps"] = {
        _nonempty_string(component, "fusion plan component"): _finite(value, "fusion plan component cap")
        for component, value in sorted(component_caps.items())
    }
    if any(not 0.0 <= value <= 1.0 for value in raw["component_caps"].values()):
        raise CIRFContractError("fusion plan component cap must lie in [0,1]")
    transforms = raw["frozen_transforms"]
    if not isinstance(transforms, Mapping):
        raise CIRFContractError("fusion plan frozen transforms must be a mapping")
    raw["frozen_transforms"] = {str(key): transforms[key] for key in sorted(transforms)}
    raw["fusion_plan_hash"] = sha256_json(raw)
    return raw


def validate_fusion_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="fusion plan",
        allowed=FUSION_PLAN_REQUIRED | {"fusion_plan_hash"},
        required=FUSION_PLAN_REQUIRED | {"fusion_plan_hash"},
    )
    digest = _sha256_hex(raw.pop("fusion_plan_hash"), "fusion_plan_hash")
    sealed = seal_fusion_plan(raw)
    if sealed["fusion_plan_hash"] != digest:
        raise CIRFContractError("fusion plan hash mismatch")
    sealed["fusion_plan_hash"] = digest
    return sealed


# ---------------------------------------------------------------------------
# Correlation kernels and small deterministic non-negative QPs
# ---------------------------------------------------------------------------


def topology_kernel(
    origins: Sequence[str], component_by_origin: Mapping[str, str]
) -> tuple[list[list[float]], list[list[str]]]:
    """Build the only allowed 0/1 topology kernel: an equivalence partition."""

    ordered = list(origins)
    if not ordered or len(ordered) != len(set(ordered)) or len(ordered) > MAX_ORIGINS:
        raise CIRFContractError("origins must be a unique non-empty roster of at most five")
    if set(component_by_origin.keys()) != set(ordered):
        raise CIRFContractError("topology partition must cover exactly the origins")
    for origin in ordered:
        _nonempty_string(origin, "origin")
        _nonempty_string(component_by_origin[origin], "component")
    kernel = [
        [1.0 if component_by_origin[left] == component_by_origin[right] else 0.0 for right in ordered]
        for left in ordered
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for origin in ordered:
        groups[component_by_origin[origin]].append(origin)
    return kernel, [groups[key] for key in sorted(groups)]


def _mad(values: Sequence[float]) -> float:
    if not values:
        raise CIRFContractError("MAD requires values")
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    deviations = sorted(abs(value - median) for value in ordered)
    mid = len(deviations) // 2
    mad = deviations[mid] if len(deviations) % 2 else (deviations[mid - 1] + deviations[mid]) / 2.0
    return max(1e-3, 1.4826 * mad)


def _symmetric(matrix: Sequence[Sequence[float]], tolerance: float = 1e-12) -> bool:
    return bool(matrix) and all(
        len(row) == len(matrix)
        and all(math.isfinite(float(value)) for value in row)
        and abs(float(matrix[i][j]) - float(matrix[j][i])) <= tolerance
        for i, row in enumerate(matrix)
        for j, _ in enumerate(row)
    )


def _jacobi_eigen(matrix: Sequence[Sequence[float]], iterations: int = 256) -> tuple[list[float], list[list[float]]]:
    """Dependency-free symmetric eigensolver adequate for frozen ``M<=5``."""

    n = len(matrix)
    if n == 0 or not _symmetric(matrix, tolerance=1e-8):
        raise CIRFContractError("Jacobi eigensolver requires a finite symmetric matrix")
    a = [[float(value) for value in row] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iterations):
        p, q, magnitude = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > magnitude:
                    p, q, magnitude = i, j, abs(a[i][j])
        if magnitude <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for index in range(n):
            if index in (p, q):
                continue
            aip, aiq = a[index][p], a[index][q]
            a[index][p] = a[p][index] = c * aip - s * aiq
            a[index][q] = a[q][index] = s * aip + c * aiq
        for index in range(n):
            vip, viq = vectors[index][p], vectors[index][q]
            vectors[index][p] = c * vip - s * viq
            vectors[index][q] = s * vip + c * viq
    return [a[i][i] for i in range(n)], vectors


def is_psd(matrix: Sequence[Sequence[float]], tolerance: float = 1e-9) -> bool:
    try:
        values, _ = _jacobi_eigen(matrix)
    except CIRFContractError:
        return False
    return min(values) >= -tolerance


def _project_psd(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    values, vectors = _jacobi_eigen(matrix)
    n = len(matrix)
    clipped = [max(0.0, value) for value in values]
    return [
        [sum(vectors[i][k] * clipped[k] * vectors[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def constrained_psd_completion(
    matrix: Sequence[Sequence[float]], *, components: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Repair a correlation matrix without weakening complete-correlation blocks.

    Failure returns an all-one matrix.  This is deliberately conservative: it
    merges evidence rather than inventing an independent component.
    """

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise CIRFContractError("PSD completion requires a square non-empty matrix")
    if any(index < 0 or index >= n for component in components for index in component):
        raise CIRFContractError("PSD completion component index out of range")
    required_one = {(i, j) for component in components for i in component for j in component}
    current = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    if any(not math.isfinite(value) for row in current for value in row):
        return {
            "kernel": [[1.0] * n for _ in range(n)],
            "mode": "FULLY_CORRELATED_DEGENERACY",
            "completion_hash": _stable_matrix_hash([[1.0] * n for _ in range(n)]),
        }
    for _ in range(256):
        current = _project_psd([[0.5 * (current[i][j] + current[j][i]) for j in range(n)] for i in range(n)])
        for i in range(n):
            for j in range(n):
                current[i][j] = min(1.0, max(0.0, current[i][j]))
        for i in range(n):
            current[i][i] = 1.0
        for i, j in required_one:
            current[i][j] = 1.0
        if (
            _symmetric(current, tolerance=1e-8)
            and is_psd(current, tolerance=1e-8)
            and all(abs(current[i][i] - 1.0) <= 1e-10 for i in range(n))
            and all(0.0 <= current[i][j] <= 1.0 for i in range(n) for j in range(n))
            and all(abs(current[i][j] - 1.0) <= 1e-10 for i, j in required_one)
        ):
            return {
                "kernel": current,
                "mode": "CONSTRAINED_PSD_COMPLETION",
                "completion_hash": _stable_matrix_hash(current),
            }
    fallback = [[1.0] * n for _ in range(n)]
    return {
        "kernel": fallback,
        "mode": "FULLY_CORRELATED_DEGENERACY",
        "completion_hash": _stable_matrix_hash(fallback),
    }


def _correlation_from_blocks(blocks: Sequence[Mapping[str, float]], origins: Sequence[str]) -> list[list[float]]:
    values = [[_finite(block[origin], f"fit residual {origin}") for block in blocks] for origin in origins]
    standardized: list[list[float]] = []
    for row in values:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / max(1, len(row) - 1)
        scale = math.sqrt(variance)
        standardized.append([0.0 for _ in row] if scale <= EPS else [(value - mean) / scale for value in row])
    result: list[list[float]] = []
    for i in range(len(origins)):
        row: list[float] = []
        for j in range(len(origins)):
            if i == j:
                row.append(1.0)
            else:
                numerator = sum(a * b for a, b in zip(standardized[i], standardized[j]))
                row.append(max(-1.0, min(1.0, numerator / max(1, len(blocks) - 1))))
        result.append(row)
    return result


KERNEL_FIT_CELL_REQUIRED = frozenset(
    {"schema_version", "axis", "availability", "context_id", "fit_event_hash_by_block"}
)
FALLBACK_MAD_REQUIRED = frozenset(
    {"schema_version", "axis", "availability", "context_id", "mad_scales", "source_fit_cell_receipt_hashes"}
)


def seal_kernel_fit_cell_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the exact fit availability/context/axis/event membership of a cell."""

    raw = _copy_without_hash(payload, name="kernel fit-cell receipt", hash_field="fit_cell_receipt_hash")
    raw = _copy_allowed(raw, name="kernel fit-cell receipt", allowed=KERNEL_FIT_CELL_REQUIRED, required=KERNEL_FIT_CELL_REQUIRED)
    if raw["schema_version"] != KERNEL_FIT_CELL_SCHEMA or raw["axis"] not in {"R", "U"}:
        raise CIRFContractError("invalid kernel fit-cell schema/axis")
    availability = raw["availability"]
    if not isinstance(availability, list) or not availability or len(availability) != len(set(availability)) or availability != sorted(availability):
        raise CIRFContractError("kernel fit-cell availability must be canonical")
    raw["availability"] = [_nonempty_string(origin, "kernel fit-cell origin") for origin in availability]
    raw["context_id"] = _nonempty_string(raw["context_id"], "kernel fit-cell context")
    event_by_block = raw["fit_event_hash_by_block"]
    if not isinstance(event_by_block, Mapping) or not event_by_block:
        raise CIRFContractError("kernel fit-cell event bindings must be a non-empty mapping")
    raw["fit_event_hash_by_block"] = {
        _nonempty_string(block_id, "kernel fit-cell block id"): _sha256_hex(event_hash, "kernel fit-cell sealed event hash")
        for block_id, event_hash in sorted(event_by_block.items())
    }
    raw["fit_cell_receipt_hash"] = sha256_json(raw)
    return raw


def validate_kernel_fit_cell_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="kernel fit-cell receipt",
        allowed=KERNEL_FIT_CELL_REQUIRED | {"fit_cell_receipt_hash"},
        required=KERNEL_FIT_CELL_REQUIRED | {"fit_cell_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("fit_cell_receipt_hash"), "fit_cell_receipt_hash")
    sealed = seal_kernel_fit_cell_receipt(raw)
    if sealed["fit_cell_receipt_hash"] != digest:
        raise CIRFContractError("kernel fit-cell receipt hash mismatch")
    sealed["fit_cell_receipt_hash"] = digest
    return sealed


def seal_fallback_mad_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal conservative MAD provenance used only on topology fallback cells."""

    raw = _copy_without_hash(payload, name="fallback MAD receipt", hash_field="fallback_mad_receipt_hash")
    raw = _copy_allowed(raw, name="fallback MAD receipt", allowed=FALLBACK_MAD_REQUIRED, required=FALLBACK_MAD_REQUIRED)
    if raw["schema_version"] != FALLBACK_MAD_SCHEMA or raw["axis"] not in {"R", "U"}:
        raise CIRFContractError("invalid fallback MAD receipt schema/axis")
    availability = raw["availability"]
    if not isinstance(availability, list) or not availability or availability != sorted(availability) or len(availability) != len(set(availability)):
        raise CIRFContractError("fallback MAD availability must be canonical")
    raw["availability"] = [_nonempty_string(origin, "fallback MAD origin") for origin in availability]
    raw["context_id"] = _nonempty_string(raw["context_id"], "fallback MAD context")
    scales = raw["mad_scales"]
    if not isinstance(scales, Mapping) or set(scales.keys()) != set(availability):
        raise CIRFContractError("fallback MAD scales must cover availability")
    raw["mad_scales"] = {origin: _finite(scales[origin], "fallback MAD scale") for origin in availability}
    if any(value < 1e-3 for value in raw["mad_scales"].values()):
        raise CIRFContractError("fallback MAD scale below frozen floor")
    raw["source_fit_cell_receipt_hashes"] = _hash_list(
        raw["source_fit_cell_receipt_hashes"], "fallback MAD provenance"
    )
    raw["fallback_mad_receipt_hash"] = sha256_json(raw)
    return raw


def validate_fallback_mad_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="fallback MAD receipt",
        allowed=FALLBACK_MAD_REQUIRED | {"fallback_mad_receipt_hash"},
        required=FALLBACK_MAD_REQUIRED | {"fallback_mad_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("fallback_mad_receipt_hash"), "fallback_mad_receipt_hash")
    sealed = seal_fallback_mad_receipt(raw)
    if sealed["fallback_mad_receipt_hash"] != digest:
        raise CIRFContractError("fallback MAD receipt hash mismatch")
    sealed["fallback_mad_receipt_hash"] = digest
    return sealed


def build_kernel_contract(
    *,
    axis: str,
    availability: Sequence[str],
    context_id: str,
    component_by_origin: Mapping[str, str],
    fit_blocks: Sequence[Mapping[str, Any]],
    fit_cell_receipt: Mapping[str, Any],
    min_joint_blocks: int = 2,
    fallback_mad_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one ``availability × context × axis`` kernel/MAD contract.

    Only listwise-complete fit blocks for the requested availability set may
    influence the empirical residual kernel.  Insufficient blocks fall back to
    the topology kernel; absent scales defer rather than borrowing calibration.
    """

    if axis not in {"R", "U"}:
        raise CIRFContractError("kernel axis must be R or U")
    origins = list(availability)
    if not origins or len(origins) != len(set(origins)) or origins != sorted(origins):
        raise CIRFContractError("availability must be a canonical unique non-empty origin set")
    _nonempty_string(context_id, "context_id")
    if min_joint_blocks < 2:
        raise CIRFContractError("min_joint_blocks must be at least two")
    fit_receipt = validate_kernel_fit_cell_receipt(fit_cell_receipt)
    if (
        fit_receipt["axis"] != axis
        or fit_receipt["availability"] != origins
        or fit_receipt["context_id"] != context_id
    ):
        raise CIRFContractError("kernel fit-cell receipt availability/context/axis mismatch")
    top, groups = topology_kernel(origins, {origin: component_by_origin[origin] for origin in origins})
    complete: list[dict[str, float]] = []
    block_ids: list[str] = []
    for raw in fit_blocks:
        allowed = {"block_id", "residuals", "split", "axis", "context_id", "availability", "sealed_event_hash"}
        item = _copy_allowed(raw, name="fit residual block", allowed=allowed, required=allowed)
        if item["split"] != "fit":
            raise CIRFContractError("kernel fit blocks must come from fit split")
        if item["axis"] != axis or item["context_id"] != context_id or item["availability"] != origins:
            raise CIRFContractError("fit block cell mismatch")
        block_id = _nonempty_string(item["block_id"], "fit block id")
        sealed_event_hash = _sha256_hex(item["sealed_event_hash"], "fit block sealed_event_hash")
        residuals = item["residuals"]
        if not isinstance(residuals, Mapping):
            raise CIRFContractError("fit block residuals must be a mapping")
        if not set(origins).issubset(set(residuals.keys())):
            continue
        complete.append({origin: _finite(residuals[origin], "fit residual") for origin in origins})
        block_ids.append(block_id)
        if fit_receipt["fit_event_hash_by_block"].get(block_id) != sealed_event_hash:
            raise CIRFContractError("fit block is not bound by fit-cell receipt")
    if len(block_ids) != len(set(block_ids)):
        raise CIRFContractError("duplicate fit block id")
    if sorted(block_ids) != sorted(fit_receipt["fit_event_hash_by_block"]):
        raise CIRFContractError("fit-cell receipt block membership mismatch")
    if not complete:
        raise CIRFContractError("no listwise-complete fit block for availability/context cell")
    if len(complete) < min_joint_blocks:
        # A one-block cell cannot estimate its own conservative error scale.
        # The caller must supply the pre-frozen maximum from all fit-eligible
        # cells, exactly as required by the availability/context contract.
        if fallback_mad_receipt is None:
            raise CIRFContractError("insufficient joint blocks require sealed fallback MAD provenance")
        fallback = validate_fallback_mad_receipt(fallback_mad_receipt)
        if fallback["axis"] != axis or fallback["availability"] != origins or fallback["context_id"] != context_id:
            raise CIRFContractError("fallback MAD receipt availability/context/axis mismatch")
        scales = dict(fallback["mad_scales"])
        empirical = None
        lam = 1.0
        kernel = top
        mode = "TOPOLOGY_FALLBACK_INSUFFICIENT_BLOCKS"
    else:
        scales = {origin: _mad([block[origin] for block in complete]) for origin in origins}
        empirical = _correlation_from_blocks(complete, origins)
        b = len(complete)
        variance_sum = 0.0
        denominator = 0.0
        for i in range(len(origins)):
            for j in range(len(origins)):
                if i == j:
                    continue
                samples = [block[origins[i]] for block in complete]
                samples_j = [block[origins[j]] for block in complete]
                mean_i = sum(samples) / b
                mean_j = sum(samples_j) / b
                std_i = math.sqrt(sum((value - mean_i) ** 2 for value in samples) / max(1, b - 1))
                std_j = math.sqrt(sum((value - mean_j) ** 2 for value in samples_j) / max(1, b - 1))
                products = [
                    0.0 if std_i <= EPS or std_j <= EPS else (x - mean_i) / std_i * (y - mean_j) / std_j
                    for x, y in zip(samples, samples_j)
                ]
                variance_sum += sum((value - empirical[i][j]) ** 2 for value in products) / (b * (b - 1))
                denominator += (empirical[i][j] - top[i][j]) ** 2
        lam = 1.0 if denominator <= EPS else min(1.0, max(0.0, variance_sum / denominator))
        proposed = [
            [(1.0 - lam) * empirical[i][j] + lam * top[i][j] for j in range(len(origins))]
            for i in range(len(origins))
        ]
        index_groups = [[origins.index(item) for item in group] for group in groups]
        completion = constrained_psd_completion(proposed, components=index_groups)
        kernel = completion["kernel"]
        mode = completion["mode"]
    contract = {
        "schema_version": KERNEL_SCHEMA,
        "axis": axis,
        "availability": origins,
        "context_id": context_id,
        "fit_block_ids": sorted(block_ids),
        "joint_block_count": len(complete),
        "min_joint_blocks": min_joint_blocks,
        "topology_kernel": top,
        "empirical_kernel": empirical,
        "shrinkage_lambda": lam,
        "kernel": kernel,
        "mad_scales": scales,
        "component_by_origin": {origin: component_by_origin[origin] for origin in origins},
        "mode": mode,
        "fit_cell_receipt_hash": fit_receipt["fit_cell_receipt_hash"],
        "fallback_mad_receipt_hash": None if fallback_mad_receipt is None else validate_fallback_mad_receipt(fallback_mad_receipt)["fallback_mad_receipt_hash"],
    }
    contract["kernel_hash"] = sha256_json(contract)
    return contract


def validate_kernel_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "axis",
        "availability",
        "context_id",
        "fit_block_ids",
        "joint_block_count",
        "min_joint_blocks",
        "topology_kernel",
        "empirical_kernel",
        "shrinkage_lambda",
        "kernel",
        "mad_scales",
        "component_by_origin",
        "mode",
        "fit_cell_receipt_hash",
        "fallback_mad_receipt_hash",
        "kernel_hash",
    }
    raw = _copy_allowed(contract, name="kernel contract", allowed=required, required=required)
    digest = _sha256_hex(raw["kernel_hash"], "kernel_hash")
    unsigned = dict(raw)
    unsigned.pop("kernel_hash", None)
    if sha256_json(unsigned) != digest:
        raise CIRFContractError("kernel contract hash mismatch")
    if raw["schema_version"] != KERNEL_SCHEMA or raw["axis"] not in {"R", "U"}:
        raise CIRFContractError("invalid kernel contract schema/axis")
    origins = raw["availability"]
    if not isinstance(origins, list) or not origins or len(origins) != len(set(origins)):
        raise CIRFContractError("kernel availability invalid")
    if origins != sorted(origins):
        raise CIRFContractError("kernel availability must be canonical")
    _sha256_hex(raw["fit_cell_receipt_hash"], "fit_cell_receipt_hash")
    if raw["fallback_mad_receipt_hash"] is not None:
        _sha256_hex(raw["fallback_mad_receipt_hash"], "fallback_mad_receipt_hash")
    kernel = raw["kernel"]
    if not _symmetric(kernel, tolerance=1e-8) or len(kernel) != len(origins):
        raise CIRFContractError("kernel must be symmetric and match availability")
    if not is_psd(kernel, tolerance=1e-8):
        raise CIRFContractError("kernel must be PSD")
    if any(abs(kernel[i][i] - 1.0) > 1e-8 for i in range(len(origins))):
        raise CIRFContractError("kernel diagonal must equal one")
    if any(value < -1e-10 or value > 1.0 + 1e-10 for row in kernel for value in row):
        raise CIRFContractError("kernel entries must be in [0,1]")
    top, groups = topology_kernel(origins, raw["component_by_origin"])
    for group in groups:
        for left in group:
            for right in group:
                if abs(kernel[origins.index(left)][origins.index(right)] - 1.0) > 1e-8:
                    raise CIRFContractError("complete-correlation component was weakened")
    if set(raw["mad_scales"].keys()) != set(origins):
        raise CIRFContractError("MAD scales must cover availability")
    for value in raw["mad_scales"].values():
        if _finite(value, "MAD scale") < 1e-3:
            raise CIRFContractError("MAD scale below frozen floor")
    return raw


def deduplicate_evidence_units(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_assets_by_origin: Mapping[str, Mapping[str, str]] | None = None,
    expected_roster_state_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Drop byte-identical replays; reject different evidence for one origin/reception."""

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in records:
        origin_hint = item.get("evidence_origin_id") if isinstance(item, Mapping) else None
        expected_assets = None
        if expected_assets_by_origin is not None:
            if not isinstance(origin_hint, str) or origin_hint not in expected_assets_by_origin:
                raise CIRFContractError("origin absent from pre-sealed asset manifest")
            expected_assets = expected_assets_by_origin[origin_hint]
        valid = validate_factorized_evidence(
            item,
            expected_assets=expected_assets,
            expected_roster_state_hashes=expected_roster_state_hashes,
        )
        key = (valid["local_evidence"]["satellite_reception_id"], valid["evidence_origin_id"])
        previous = result.get(key)
        if previous is not None and previous["factor_hash"] != valid["factor_hash"]:
            raise CIRFContractError("conflicting duplicate evidence unit")
        result[key] = valid
    return [result[key] for key in sorted(result)]


def _solve_linear(system: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float] | None:
    n = len(rhs)
    if len(system) != n or any(len(row) != n for row in system):
        return None
    a = [[float(value) for value in row] + [float(rhs[i])] for i, row in enumerate(system)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) <= 1e-12:
            return None
        a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        a[column] = [value / scale for value in a[column]]
        for row in range(n):
            if row == column:
                continue
            factor = a[row][column]
            if factor:
                a[row] = [left - factor * right for left, right in zip(a[row], a[column])]
    return [a[i][-1] for i in range(n)]


def _constraints(
    origins: Sequence[str], cap_by_origin: Mapping[str, float], component_by_origin: Mapping[str, str], component_caps: Mapping[str, float]
) -> tuple[list[list[float]], list[float]]:
    n = len(origins)
    rows: list[list[float]] = []
    bounds: list[float] = []
    for i, origin in enumerate(origins):
        cap = _finite(cap_by_origin[origin], f"cap {origin}")
        if cap < 0.0:
            raise CIRFContractError("origin cap must be non-negative")
        lower = [0.0] * n
        lower[i] = -1.0
        rows.append(lower)
        bounds.append(0.0)
        upper = [0.0] * n
        upper[i] = 1.0
        rows.append(upper)
        bounds.append(cap)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, origin in enumerate(origins):
        grouped[component_by_origin[origin]].append(index)
    for component, indices in sorted(grouped.items()):
        if component not in component_caps:
            raise CIRFContractError("component cap missing")
        cap = _finite(component_caps[component], f"component cap {component}")
        if cap < 0.0:
            raise CIRFContractError("component cap must be non-negative")
        row = [0.0] * n
        for index in indices:
            row[index] = 1.0
        rows.append(row)
        bounds.append(cap)
    return rows, bounds


def _feasible_caps(
    origins: Sequence[str], cap_by_origin: Mapping[str, float], component_by_origin: Mapping[str, str], component_caps: Mapping[str, float]
) -> bool:
    grouped: dict[str, float] = defaultdict(float)
    for origin in origins:
        cap = float(cap_by_origin[origin])
        if cap < -FEASIBILITY_TOL:
            return False
        grouped[component_by_origin[origin]] += max(0.0, cap)
    return sum(min(total, float(component_caps[component])) for component, total in grouped.items()) >= 1.0 - FEASIBILITY_TOL


def _quadratic_value(matrix: Sequence[Sequence[float]], linear: Sequence[float], point: Sequence[float]) -> float:
    return sum(point[i] * matrix[i][j] * point[j] for i in range(len(point)) for j in range(len(point))) + sum(
        linear[i] * point[i] for i in range(len(point))
    )


def _exact_simplex_quadratic_value(
    matrix: Sequence[Sequence[float]], linear: Sequence[float], point: Sequence[float]
) -> Decimal:
    """Compare objectives without losing sub-ulp curvature or simplex drift.

    ``Decimal.from_float`` preserves the exact input floats.  A positive
    eigen-direction whose contribution is smaller than one ulp of a common
    baseline must still remain a stage-1 distinction; converting only the
    final rounded float objective would erase it.  The last coordinate is
    reconstructed from the exact simplex equality, so harmless floating-point
    summation drift cannot make one active face look better than another.
    """

    with localcontext() as context:
        context.prec = 120
        values = [Decimal.from_float(float(value)) for value in point]
        if values:
            values[-1] = Decimal(1) - sum(values[:-1], Decimal(0))
        result = Decimal(0)
        for row in range(len(values)):
            for column in range(len(values)):
                result += values[row] * Decimal.from_float(float(matrix[row][column])) * values[column]
            result += Decimal.from_float(float(linear[row])) * values[row]
        return +result


def _solve_regularized_qp(
    matrix: Sequence[Sequence[float]],
    linear: Sequence[float],
    *,
    origins: Sequence[str],
    cap_by_origin: Mapping[str, float],
    component_by_origin: Mapping[str, str],
    component_caps: Mapping[str, float],
) -> list[float]:
    """Enumerate active sets for a tiny positive-definite QP (``M<=5``)."""

    n = len(origins)
    if n == 0 or n > MAX_ORIGINS:
        raise CIRFContractError("QP dimension must be 1..5")
    if not _feasible_caps(origins, cap_by_origin, component_by_origin, component_caps):
        raise CIRFContractError("QP feasible set is empty")
    inequalities, bounds = _constraints(origins, cap_by_origin, component_by_origin, component_caps)
    candidates: list[list[float]] = []
    # The simplex equality plus at most n-1 independent active inequalities
    # identifies every face minimiser of a strictly convex objective.
    for active_count in range(0, min(n - 1, len(inequalities)) + 1):
        for active in itertools.combinations(range(len(inequalities)), active_count):
            aeq = [[1.0] * n] + [inequalities[index] for index in active]
            beq = [1.0] + [bounds[index] for index in active]
            m = len(aeq)
            size = n + m
            system = [[0.0] * size for _ in range(size)]
            rhs = [0.0] * size
            for i in range(n):
                rhs[i] = -linear[i]
                for j in range(n):
                    system[i][j] = 2.0 * matrix[i][j]
                for j in range(m):
                    system[i][n + j] = aeq[j][i]
                    system[n + j][i] = aeq[j][i]
            for j in range(m):
                rhs[n + j] = beq[j]
            solution = _solve_linear(system, rhs)
            if solution is None:
                continue
            point = solution[:n]
            if abs(sum(point) - 1.0) > 1e-8:
                continue
            if all(
                sum(row[i] * point[i] for i in range(n)) <= bound + 1e-8
                for row, bound in zip(inequalities, bounds)
            ):
                candidates.append(point)
    if not candidates:
        raise CIRFContractError("QP solver found no feasible active-set solution")
    return min(candidates, key=lambda point: (_quadratic_value(matrix, linear, point), tuple(round(x, 15) for x in point)))


def project_to_feasible_weights(
    target: Sequence[float],
    *,
    origins: Sequence[str],
    cap_by_origin: Mapping[str, float],
    component_by_origin: Mapping[str, str],
    component_caps: Mapping[str, float],
) -> list[float]:
    if len(target) != len(origins):
        raise CIRFContractError("projection target dimension mismatch")
    matrix = [[1.0 if i == j else 0.0 for j in range(len(origins))] for i in range(len(origins))]
    linear = [-2.0 * _finite(value, "projection target") for value in target]
    return _solve_regularized_qp(
        matrix,
        linear,
        origins=origins,
        cap_by_origin=cap_by_origin,
        component_by_origin=component_by_origin,
        component_caps=component_caps,
    )


def _affine_space(rows: Sequence[Sequence[float]], values: Sequence[float], n: int) -> tuple[list[float], list[list[float]]] | None:
    """Return one solution and a null-space basis of ``rows*x=values``."""

    if len(rows) != len(values) or any(len(row) != n for row in rows):
        return None
    augmented = [[float(value) for value in row] + [float(values[index])] for index, row in enumerate(rows)]
    pivot_columns: list[int] = []
    row_index = 0
    for column in range(n):
        pivot = next((index for index in range(row_index, len(augmented)) if abs(augmented[index][column]) > 1e-12), None)
        if pivot is None:
            continue
        augmented[row_index], augmented[pivot] = augmented[pivot], augmented[row_index]
        divisor = augmented[row_index][column]
        augmented[row_index] = [value / divisor for value in augmented[row_index]]
        for index in range(len(augmented)):
            if index == row_index:
                continue
            multiplier = augmented[index][column]
            if multiplier:
                augmented[index] = [left - multiplier * right for left, right in zip(augmented[index], augmented[row_index])]
        pivot_columns.append(column)
        row_index += 1
        if row_index == len(augmented):
            break
    if any(all(abs(value) <= 1e-10 for value in row[:n]) and abs(row[-1]) > 1e-10 for row in augmented):
        return None
    free_columns = [column for column in range(n) if column not in pivot_columns]
    particular = [0.0] * n
    for index, column in enumerate(pivot_columns):
        particular[column] = augmented[index][-1]
    basis: list[list[float]] = []
    for free in free_columns:
        vector = [0.0] * n
        vector[free] = 1.0
        for index, column in enumerate(pivot_columns):
            vector[column] = -augmented[index][free]
        basis.append(vector)
    return particular, basis


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(value) * vector[column] for column, value in enumerate(row)) for row in matrix]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _primary_nullspace_solution(
    matrix: Sequence[Sequence[float]], linear: Sequence[float]
) -> tuple[list[float], list[list[float]]] | None:
    """Solve PSD ``A*y=-c`` and return one solution plus its null basis."""

    dimension = len(linear)
    if dimension == 0:
        return [], []
    values, vectors = _jacobi_eigen(matrix)
    scale = max(1.0, max(abs(value) for value in values))
    negative_tolerance = 32.0 * math.ulp(scale) * max(1, dimension)
    solution = [0.0] * dimension
    null_basis: list[list[float]] = []
    for index, eigenvalue in enumerate(values):
        vector = [vectors[row][index] for row in range(dimension)]
        component = _dot(vector, linear)
        # A strictly positive direction remains part of the primary objective,
        # however small its curvature is.  Moving it to the secondary
        # null-space changes the lexicographic optimum.  Only exact zero or a
        # tiny *negative* Jacobi round-off is accepted as numerical null-space.
        if eigenvalue > 0.0:
            for row in range(dimension):
                solution[row] -= component * vector[row] / eigenvalue
        elif eigenvalue >= -negative_tolerance and abs(component) <= 1e-9 * max(
            1.0, max(abs(value) for value in linear)
        ):
            null_basis.append(vector)
        else:
            return None
    return solution, null_basis


def _lexicographic_face_candidate(
    covariance: Sequence[Sequence[float]],
    beta0: Sequence[float],
    equality_rows: Sequence[Sequence[float]],
    equality_values: Sequence[float],
) -> list[float] | None:
    """Exact stage-1/stage-2 solve on one affine active-set face."""

    affine = _affine_space(equality_rows, equality_values, len(beta0))
    if affine is None:
        return None
    base, basis = affine
    dimension = len(basis)
    if dimension == 0:
        return base
    h_basis = [_mat_vec(covariance, vector) for vector in basis]
    reduced = [[_dot(basis[i], h_basis[j]) for j in range(dimension)] for i in range(dimension)]
    linear = [_dot(vector, _mat_vec(covariance, base)) for vector in basis]
    primary = _primary_nullspace_solution(reduced, linear)
    if primary is None:
        return None
    y0, null_basis = primary
    primary_point = [base[index] + sum(basis[column][index] * y0[column] for column in range(dimension)) for index in range(len(base))]
    if not null_basis:
        return primary_point
    directions = [
        [sum(basis[column][index] * null_vector[column] for column in range(dimension)) for index in range(len(base))]
        for null_vector in null_basis
    ]
    gram = [[_dot(left, right) for right in directions] for left in directions]
    rhs = [-_dot(direction, [primary_point[index] - beta0[index] for index in range(len(base))]) for direction in directions]
    correction = _solve_linear(gram, rhs)
    if correction is None:
        return None
    return [primary_point[index] + sum(directions[column][index] * correction[column] for column in range(len(directions))) for index in range(len(base))]


def solve_lexicographic_qp(
    *,
    covariance: Sequence[Sequence[float]],
    reference_weights: Sequence[float],
    origins: Sequence[str],
    cap_by_origin: Mapping[str, float],
    component_by_origin: Mapping[str, str],
    component_caps: Mapping[str, float],
) -> dict[str, Any]:
    """Solve the frozen primary-QP/secondary-distance lexicographic contract.

    Stage 1 enumerates every active-set affine face and minimizes the
    unregularised PSD covariance objective.  Stage 2 then minimizes distance
    to ``beta0`` over the exact stage-1 nullspace.  No epsilon perturbation or
    hash tie-break is used for continuous weights.
    """

    n = len(origins)
    if len(covariance) != n or any(len(row) != n for row in covariance):
        raise CIRFContractError("covariance dimension mismatch")
    if len(reference_weights) != n:
        raise CIRFContractError("reference weight dimension mismatch")
    if not _symmetric(covariance, tolerance=1e-8) or not is_psd(covariance, tolerance=1e-8):
        raise CIRFContractError("covariance must be symmetric PSD")
    beta0 = project_to_feasible_weights(
        reference_weights,
        origins=origins,
        cap_by_origin=cap_by_origin,
        component_by_origin=component_by_origin,
        component_caps=component_caps,
    )
    if not _feasible_caps(origins, cap_by_origin, component_by_origin, component_caps):
        raise CIRFContractError("QP feasible set is empty")
    inequalities, bounds = _constraints(origins, cap_by_origin, component_by_origin, component_caps)
    candidates: list[tuple[Decimal, float, float, list[float]]] = []
    for active_count in range(0, min(n - 1, len(inequalities)) + 1):
        for active in itertools.combinations(range(len(inequalities)), active_count):
            rows = [[1.0] * n] + [inequalities[index] for index in active]
            values = [1.0] + [bounds[index] for index in active]
            candidate = _lexicographic_face_candidate(covariance, beta0, rows, values)
            if candidate is None:
                continue
            total = sum(candidate)
            if abs(total - 1.0) > FEASIBILITY_TOL or total == 0.0:
                continue
            # Canonicalise the affine equality before comparing primary
            # objectives across duplicate active-set faces.  Without this,
            # one-ulp sum drift can make two mathematically identical points
            # appear lexicographically different.
            candidate = [value / total for value in candidate]
            if not all(_dot(row, candidate) <= bound + FEASIBILITY_TOL for row, bound in zip(inequalities, bounds)):
                continue
            primary = _quadratic_value(covariance, [0.0] * n, candidate)
            exact_primary = _exact_simplex_quadratic_value(covariance, [0.0] * n, candidate)
            secondary = sum((left - right) ** 2 for left, right in zip(candidate, beta0))
            candidates.append((exact_primary, primary, secondary, candidate))
    if not candidates:
        raise CIRFContractError("exact lexicographic QP found no feasible face")
    best_exact_primary = min(item[0] for item in candidates)
    best_item = min(candidates, key=lambda item: item[0])
    best_point = best_item[3]
    spectral_values, spectral_vectors = _jacobi_eigen(covariance)

    def same_primary_face(point: Sequence[float]) -> bool:
        difference = [left - right for left, right in zip(point, best_point)]
        # Reuse one global spectral decomposition for every active face.
        # Stage 2 may move only in eigen-directions whose represented
        # eigenvalue is exactly zero (or tiny negative Jacobi round-off).
        # Any non-zero component in a strictly positive direction is excluded,
        # even if its objective increment lies below one ulp of the common
        # baseline.
        for index, eigenvalue in enumerate(spectral_values):
            if eigenvalue <= 0.0:
                continue
            vector = [spectral_vectors[row][index] for row in range(n)]
            if math.fsum(vector[row] * difference[row] for row in range(n)) != 0.0:
                return False
        return True

    eligible = [
        item
        for item in candidates
        if item is best_item or (item[0] == best_exact_primary and same_primary_face(item[3]))
    ]
    exact_primary, primary, secondary, beta = min(
        eligible,
        key=lambda item: (item[2], tuple(round(value, 15) for value in item[3])),
    )
    return {
        "origins": list(origins),
        "beta": beta,
        "beta0": beta0,
        "primary_objective": primary,
        "secondary_objective": secondary,
        "feasibility_tol": FEASIBILITY_TOL,
        "primary_optimality_tol": 0.0,
        "solver": "two_stage_active_set_exact_float_spectrum_v3",
    }


def dual_axis_qp(
    kernel_r: Mapping[str, Any],
    kernel_u: Mapping[str, Any],
    *,
    active_origins: Sequence[str],
    cap_by_origin: Mapping[str, float],
    component_caps: Mapping[str, float],
) -> dict[str, Any]:
    """Rebuild the availability-cell QP independently for R and U axes."""

    r = validate_kernel_contract(kernel_r)
    u = validate_kernel_contract(kernel_u)
    active = list(active_origins)
    if not active or len(active) != len(set(active)):
        raise CIRFContractError("active origins must be a unique non-empty set")
    if r["availability"] != active or u["availability"] != active:
        raise CIRFContractError("must use the exact availability/context kernel cell")
    if r["context_id"] != u["context_id"]:
        raise CIRFContractError("R/U context mismatch")
    if r["component_by_origin"] != u["component_by_origin"]:
        raise CIRFContractError("R/U topology partition mismatch")
    components = r["component_by_origin"]
    if set(cap_by_origin.keys()) != set(active):
        raise CIRFContractError("origin cap set mismatch")
    if len(active) == 1:
        return {
            "mode": "DEGRADED_N1_NONCOLLABORATIVE",
            "active_origins": active,
            "R": None,
            "U": None,
        }

    def solve(contract: Mapping[str, Any]) -> dict[str, Any]:
        scales = [float(contract["mad_scales"][origin]) for origin in active]
        covariance = [
            [scales[i] * float(contract["kernel"][i][j]) * scales[j] for j in range(len(active))]
            for i in range(len(active))
        ]
        inverse = [scale ** -2 for scale in scales]
        total = sum(inverse)
        result = solve_lexicographic_qp(
            covariance=covariance,
            reference_weights=[value / total for value in inverse],
            origins=active,
            cap_by_origin=cap_by_origin,
            component_by_origin=components,
            component_caps=component_caps,
        )
        beta = result["beta"]
        kernel = contract["kernel"]
        denominator = sum(beta[i] * float(kernel[i][j]) * beta[j] for i in range(len(active)) for j in range(len(active)))
        if denominator <= 0.0:
            raise CIRFContractError("correlation QP effective-count denominator invalid")
        result["nu"] = min(float(len(active)), 1.0 / denominator)
        return result

    return {
        "mode": "DUAL_AXIS_QP",
        "active_origins": active,
        "context_id": r["context_id"],
        "R": solve(r),
        "U": solve(u),
    }


def n1_passthrough_bytes(local_evidence: bytes | Mapping[str, Any]) -> bytes:
    """Validate a frozen LocalEvidence artifact and return its original bytes."""

    if isinstance(local_evidence, bytes):
        try:
            decoded = local_evidence.decode("utf-8")
            parsed = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CIRFContractError("N1 LocalEvidence bytes must be UTF-8 JSON") from exc
        if not isinstance(parsed, Mapping):
            raise CIRFContractError("N1 LocalEvidence JSON must be a mapping")
        _mapping_keys(parsed, "LocalEvidence")
        validate_local_evidence(parsed)
        return local_evidence
    _mapping_keys(local_evidence, "LocalEvidence")
    valid = validate_local_evidence(local_evidence)
    return canonical_json(valid).encode("utf-8")


def fuse_factorized_event(
    records: Sequence[Mapping[str, Any]],
    *,
    certificate: Mapping[str, Any] | None,
    fusion_plan: Mapping[str, Any] | None = None,
    kernel_r: Mapping[str, Any] | None = None,
    kernel_u: Mapping[str, Any] | None = None,
    n1_local_artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Truth-free technical fusion witness for G0.

    This returns per-operational-prior technical candidates only.  It does not
    run a scorer, select a metric, authorize registration, or expose a
    performance claim.  The N=1 branch intentionally bypasses every v3 layer.
    """

    if not records:
        raise CIRFContractError("fusion requires evidence")
    # The N=1 path is byte-preserving and validates only the frozen legacy
    # LocalEvidence artifact; no v3 factor, prior, certificate, kernel or cap is
    # consulted.
    if len(records) == 1:
        if n1_local_artifact_bytes is None:
            raise CIRFContractError("N1 fusion requires original LocalEvidence artifact bytes")
        raw_artifact = n1_passthrough_bytes(n1_local_artifact_bytes)
        return {
            "schema_version": SCHEMA,
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "technical_status": "DEGRADED_N1_NONCOLLABORATIVE",
            "n1_local_artifact": raw_artifact.decode("utf-8"),
            "n1_local_artifact_sha256": hashlib.sha256(raw_artifact).hexdigest(),
            "shot_count": 1,
        }
    if certificate is None:
        raise CIRFContractError("multi-origin fusion requires a same-event certificate")
    if fusion_plan is None:
        raise CIRFContractError("multi-origin fusion requires a sealed FusionPlan receipt")
    plan = validate_fusion_plan(fusion_plan)
    unique = deduplicate_evidence_units(
        records,
        expected_assets_by_origin=plan["expected_assets_by_origin"],
        expected_roster_state_hashes=plan["expected_roster_state_hashes"],
    )
    if len(unique) == 1:
        raise CIRFContractError("multi-record fusion collapsed to one unique evidence unit")
    if len({item["evidence_origin_id"] for item in unique}) != len(unique):
        raise CIRFContractError("multi-origin fusion requires exactly one evidence unit per origin")
    cert = validate_same_event_certificate(certificate)
    if cert["certificate_status"] != "CERTIFIED":
        return {
            "schema_version": SCHEMA,
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "technical_status": "DEFER",
            "reason_code": "NO_SAME_EVENT_CERTIFICATE",
        }
    hashes = sorted(item["event_ledger"]["ledger_hash"] for item in unique)
    if hashes != sorted(cert.get("ledger_hashes", [])):
        raise CIRFContractError("certificate/evidence ledger hash mismatch")
    if cert.get("opportunity_id") != plan["opportunity_id"]:
        raise CIRFContractError("FusionPlan opportunity binding mismatch")
    by_origin = {item["evidence_origin_id"]: item for item in unique}
    if set(by_origin) != set(plan["availability"]):
        raise CIRFContractError("FusionPlan availability binding mismatch")
    unique = [by_origin[origin] for origin in plan["availability"]]
    if set(plan["expected_roster_state_hashes"]) != {item["local_evidence"]["node_id"] for item in unique}:
        raise CIRFContractError("FusionPlan roster-state binding mismatch")
    labels = list(unique[0]["local_evidence"]["class_handles"])
    if any(item["local_evidence"]["class_handles"] != labels for item in unique):
        raise CIRFContractError("factorized evidence class order mismatch")
    if any(item["context_id"] != plan["context_id"] for item in unique):
        raise CIRFContractError("FusionPlan context binding mismatch")
    opinions = [factorized_log_opinion(item) for item in unique]
    transforms = dict(plan["frozen_transforms"])
    restored = [
        restore_reference_prior(
            opinion,
            common_reference_prior_id=plan["common_reference_prior_id"],
            frozen_transform=transforms.get(opinion["reference_prior_id"]),
        )
        for opinion in opinions
    ]
    origins = [item["evidence_origin_id"] for item in unique]
    if len(origins) != len(set(origins)):
        raise CIRFContractError("one active evidence unit per origin is required")
    if kernel_r is None or kernel_u is None:
        raise CIRFContractError("multi-origin fusion requires FusionPlan-bound R/U kernels")
    valid_kernel_r = validate_kernel_contract(kernel_r)
    valid_kernel_u = validate_kernel_contract(kernel_u)
    if valid_kernel_r["kernel_hash"] != plan["kernel_r_hash"] or valid_kernel_u["kernel_hash"] != plan["kernel_u_hash"]:
        raise CIRFContractError("FusionPlan kernel binding mismatch")
    if valid_kernel_r["availability"] != plan["availability"] or valid_kernel_u["availability"] != plan["availability"]:
        raise CIRFContractError("FusionPlan/kernel availability binding mismatch")
    if valid_kernel_r["context_id"] != plan["context_id"] or valid_kernel_u["context_id"] != plan["context_id"]:
        raise CIRFContractError("FusionPlan/kernel context binding mismatch")
    expected_components = set(valid_kernel_r["component_by_origin"].values())
    if set(plan["component_caps"]) != expected_components:
        raise CIRFContractError("FusionPlan component-cap binding mismatch")
    qp = dual_axis_qp(
        valid_kernel_r,
        valid_kernel_u,
        active_origins=origins,
        cap_by_origin=plan["cap_by_origin"],
        component_caps=plan["component_caps"],
    )
    if qp["mode"] != "DUAL_AXIS_QP":
        raise CIRFContractError("unexpected N1 QP branch")
    reference = labels[0]
    r_beta = qp["R"]["beta"]
    u_beta = qp["U"]["beta"]
    l_r = {
        label: qp["R"]["nu"]
        * sum(r_beta[i] * (restored[i]["prior_corrected_log_evidence"][label] - restored[i]["prior_corrected_log_evidence"][reference]) for i in range(len(restored)))
        for label in labels
    }
    l_u = qp["U"]["nu"] * sum(
        u_beta[i]
        * (
            restored[i]["prior_corrected_log_evidence"]["__unknown__"]
            - restored[i]["prior_corrected_log_evidence"][reference]
        )
        for i in range(len(restored))
    )
    candidates: list[dict[str, Any]] = []
    for prior in validate_operational_prior_set(plan["operational_priors"], labels):
        scores = {reference: math.log(prior[reference])}
        scores.update({label: math.log(prior[label]) + l_r[label] for label in labels if label != reference})
        scores["__unknown__"] = math.log(prior["__unknown__"]) + l_u
        if any(not math.isfinite(value) for value in scores.values()):
            raise CIRFContractError("non-finite prior-propagated technical score")
        winners = [label for label, value in scores.items() if abs(value - max(scores.values())) <= 1e-12]
        candidates.append({"prior_hash": sha256_json(prior), "scores": scores, "winner_set": sorted(winners)})
    unanimous = len({tuple(item["winner_set"]) for item in candidates}) == 1
    return {
        "schema_version": SCHEMA,
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "technical_status": "UNANIMOUS_PRIOR_CANDIDATE" if unanimous else "DEFER_PRIOR_DISAGREEMENT",
        "shot_count": 1,
        "reference_class": reference,
        "registered_relative_evidence": l_r,
        "unknown_relative_evidence": l_u,
        "dual_axis_qp": qp,
        "operational_prior_candidates": candidates,
        "fusion_plan_hash": plan["fusion_plan_hash"],
    }


# ---------------------------------------------------------------------------
# Four-way split, simultaneous interval and non-interchangeable risk contracts
# ---------------------------------------------------------------------------

SPLIT_NAMES = ("fit", "interval_calibration", "conformal_calibration", "formal_test")
SPLIT_ASSIGNMENT_REQUIRED = frozenset(
    {
        "split",
        "emission_event_id",
        "physical_sample_id",
        "risk_cluster_id",
        "event_opportunity_block_id",
        "identity_handle",
        "population",
    }
)


def seal_four_split_ledger(assignments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seal four mutually isolated data layers without accepting truth labels.

    ``population`` is acquisition metadata (``registered`` or ``unknown``),
    not a query role input to the predictor.  It is used only to enforce the
    stricter unknown identity four-way isolation in a data-contract artifact.
    """

    normalized: list[dict[str, Any]] = []
    split_by_event: dict[str, str] = {}
    split_by_physical: dict[str, str] = {}
    split_by_risk: dict[str, str] = {}
    split_by_block: dict[str, str] = {}
    split_by_unknown_identity: dict[str, str] = {}
    for item in assignments:
        raw = _copy_allowed(
            item,
            name="four split assignment",
            allowed=SPLIT_ASSIGNMENT_REQUIRED,
            required=SPLIT_ASSIGNMENT_REQUIRED,
        )
        if raw["split"] not in SPLIT_NAMES:
            raise CIRFContractError("invalid four-split assignment")
        for field_name in SPLIT_ASSIGNMENT_REQUIRED.difference({"split", "population"}):
            raw[field_name] = _nonempty_string(raw[field_name], field_name)
        if raw["population"] not in {"registered", "unknown"}:
            raise CIRFContractError("population must be registered or unknown")
        for field_name, split_map in (
            ("emission_event_id", split_by_event),
            ("physical_sample_id", split_by_physical),
            ("risk_cluster_id", split_by_risk),
            ("event_opportunity_block_id", split_by_block),
        ):
            previous_split = split_map.get(raw[field_name])
            if previous_split is not None and previous_split != raw["split"]:
                raise CIRFContractError(f"four-split isolation violated by {field_name}")
            split_map[raw[field_name]] = raw["split"]
        if raw["population"] == "unknown":
            previous_split = split_by_unknown_identity.get(raw["identity_handle"])
            if previous_split is not None and previous_split != raw["split"]:
                raise CIRFContractError("unknown identity must be four-way isolated")
            split_by_unknown_identity[raw["identity_handle"]] = raw["split"]
        normalized.append(raw)
    if not normalized:
        raise CIRFContractError("four split ledger requires assignments")
    payload = {
        "schema_version": SPLIT_SCHEMA,
        "splits": {split: sorted([row for row in normalized if row["split"] == split], key=lambda row: row["emission_event_id"]) for split in SPLIT_NAMES},
    }
    payload["split_hashes"] = {split: sha256_json(payload["splits"][split]) for split in SPLIT_NAMES}
    payload["ledger_hash"] = sha256_json(payload)
    return payload


def validate_four_split_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "splits", "split_hashes", "ledger_hash"}
    raw = _copy_allowed(ledger, name="four split ledger", allowed=required, required=required)
    digest = _sha256_hex(raw["ledger_hash"], "four split ledger hash")
    unsigned = dict(raw)
    unsigned.pop("ledger_hash", None)
    if sha256_json(unsigned) != digest or raw["schema_version"] != SPLIT_SCHEMA:
        raise CIRFContractError("four split ledger hash/schema mismatch")
    if not isinstance(raw["splits"], Mapping) or set(raw["splits"].keys()) != set(SPLIT_NAMES):
        raise CIRFContractError("four split ledger split names mismatch")
    flattened: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        rows = raw["splits"][split]
        if not isinstance(rows, list):
            raise CIRFContractError("split rows must be a list")
        expected_hash = _sha256_hex(raw["split_hashes"].get(split), f"split hash {split}")
        if sha256_json(rows) != expected_hash:
            raise CIRFContractError("split hash drift")
        flattened.extend(rows)
    rebuilt = seal_four_split_ledger(flattened)
    if rebuilt["ledger_hash"] != raw["ledger_hash"]:
        raise CIRFContractError("four split ledger isolation/hash mismatch")
    return raw


def n_min_zero_failure(alpha: float, delta_cell: float) -> int:
    """Independent-block minimum required for a zero-failure CP-style gate."""

    alpha = _finite(alpha, "alpha")
    delta_cell = _finite(delta_cell, "delta_cell")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta_cell < 1.0:
        raise CIRFContractError("alpha and delta_cell must lie in (0,1)")
    return int(math.ceil(math.log(delta_cell) / math.log1p(-alpha)))


def seal_interval_contract(
    *,
    four_split_ledger: Mapping[str, Any],
    delta_event: float,
    origin_count: int,
    class_plus_unknown_count: int,
    context_count: int,
    stochastic_error_sources: int,
    deterministic_envelope: Mapping[str, Any],
    p_lower_function_hash: str,
) -> dict[str, Any]:
    """Seal event-level simultaneous interval accounting from fit/interval only."""

    ledger = validate_four_split_ledger(four_split_ledger)
    delta_event = _finite(delta_event, "delta_event")
    if not 0.0 < delta_event < 1.0:
        raise CIRFContractError("delta_event must lie in (0,1)")
    counts = [origin_count, class_plus_unknown_count, context_count, stochastic_error_sources]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise CIRFContractError("interval atomic dimensions must be positive integers")
    n_atomic = math.prod(counts)
    if origin_count > MAX_ORIGINS or class_plus_unknown_count > MAX_CLASS_PLUS_UNKNOWN or context_count > MAX_CONTEXT_BUCKETS or stochastic_error_sources > MAX_STOCHASTIC_ERROR_SOURCES or n_atomic > 7680:
        raise CIRFContractError("interval capacity exceeds frozen v3 limits")
    if not isinstance(deterministic_envelope, Mapping):
        raise CIRFContractError("deterministic envelope must be a mapping")
    envelope_keys = _mapping_keys(deterministic_envelope, "deterministic envelope")
    if not envelope_keys:
        raise CIRFContractError("deterministic envelope must not be empty")
    payload = {
        "schema_version": INTERVAL_SCHEMA,
        "fit_split_hash": ledger["split_hashes"]["fit"],
        "interval_calibration_split_hash": ledger["split_hashes"]["interval_calibration"],
        "split_isolation_contract": "FULL_LEDGER_VALIDATED_BUT_CONFORMAL_AND_FORMAL_EXCLUDED_FROM_INTERVAL_HASH",
        "delta_event": delta_event,
        "N_atomic": n_atomic,
        "delta_atomic": delta_event / n_atomic,
        "deterministic_envelope": {key: deterministic_envelope[key] for key in sorted(envelope_keys)},
        "p_lower_function_hash": _sha256_hex(p_lower_function_hash, "p_lower_function_hash"),
    }
    payload["interval_hash"] = sha256_json(payload)
    return payload


def validate_interval_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "fit_split_hash",
        "interval_calibration_split_hash",
        "split_isolation_contract",
        "delta_event",
        "N_atomic",
        "delta_atomic",
        "deterministic_envelope",
        "p_lower_function_hash",
        "interval_hash",
    }
    raw = _copy_allowed(contract, name="interval contract", allowed=required, required=required)
    digest = _sha256_hex(raw["interval_hash"], "interval hash")
    unsigned = dict(raw)
    unsigned.pop("interval_hash", None)
    if raw["schema_version"] != INTERVAL_SCHEMA or sha256_json(unsigned) != digest:
        raise CIRFContractError("interval contract hash/schema mismatch")
    if raw["split_isolation_contract"] != "FULL_LEDGER_VALIDATED_BUT_CONFORMAL_AND_FORMAL_EXCLUDED_FROM_INTERVAL_HASH":
        raise CIRFContractError("interval split-isolation contract mismatch")
    if raw["N_atomic"] < 1 or raw["N_atomic"] > 7680:
        raise CIRFContractError("invalid interval atomic count")
    if abs(_finite(raw["delta_atomic"], "delta_atomic") - _finite(raw["delta_event"], "delta_event") / raw["N_atomic"]) > 1e-18:
        raise CIRFContractError("interval Bonferroni accounting mismatch")
    return raw


def simultaneous_lower_envelope(
    atomic_lower_bounds: Mapping[str, float], *, interval_contract: Mapping[str, Any]
) -> float:
    """Use the deterministic maximum envelope; never split delta twice."""

    validate_interval_contract(interval_contract)
    keys = _mapping_keys(atomic_lower_bounds, "atomic lower bounds")
    if not keys:
        raise CIRFContractError("atomic lower bounds must not be empty")
    values = [_finite(atomic_lower_bounds[key], "atomic lower bound") for key in keys]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise CIRFContractError("lower bounds must be probabilities")
    return min(values)


def split_conformal_quantile(nonconformity: Sequence[float], alpha: float) -> dict[str, Any]:
    values = sorted(_finite(value, "nonconformity") for value in nonconformity)
    alpha = _finite(alpha, "conformal alpha")
    if not values or not 0.0 < alpha < 1.0:
        raise CIRFContractError("conformal requires non-empty values and alpha in (0,1)")
    index = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    if index > len(values):
        raise CIRFContractError("conformal cell has no finite certificate")
    return {"n": len(values), "order_statistic": index, "q": values[index - 1]}


def nested_prediction_set(
    nonconformity_by_time: Sequence[Mapping[str, float]], quantiles: Mapping[str, float]
) -> list[list[str]]:
    """Compute the required monotone-shrinking anytime conformal sets."""

    running: dict[str, float] = {}
    result: list[list[str]] = []
    for state in nonconformity_by_time:
        if set(state.keys()) != set(quantiles.keys()):
            raise CIRFContractError("conformal labels/quantiles mismatch")
        for label, value in state.items():
            running[label] = max(running.get(label, -math.inf), _finite(value, "nonconformity"))
        result.append(sorted(label for label, value in running.items() if value <= _finite(quantiles[label], "quantile")))
    if any(not set(later).issubset(set(earlier)) for earlier, later in zip(result, result[1:])):
        raise CIRFContractError("prediction set is not nested")
    return result


def class_conditional_block_max_nonconformity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Construct the frozen class/context block maxima for split conformal.

    This is a calibration-only helper.  It takes already frozen nonconformity
    values, never a local model, interval fitting input, scheduler decision, or
    formal-test output.  The returned values are exactly the block-max atoms
    used by the order statistic, so multiple receptions in one opportunity do
    not become pseudo-independent calibration examples.
    """

    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for source in rows:
        raw = _copy_allowed(
            source,
            name="conformal block row",
            allowed={"block_id", "class_label", "context_id", "nonconformity", "split"},
            required={"block_id", "class_label", "context_id", "nonconformity", "split"},
        )
        if raw["split"] != "conformal_calibration":
            raise CIRFContractError("block-max conformal rows must use conformal_calibration split")
        block = _nonempty_string(raw["block_id"], "conformal block id")
        label = _nonempty_string(raw["class_label"], "conformal class label")
        context = _nonempty_string(raw["context_id"], "conformal context id")
        value = _finite(raw["nonconformity"], "conformal nonconformity")
        if not 0.0 <= value <= 1.0:
            raise CIRFContractError("conformal nonconformity must lie in [0,1]")
        key = (label, context)
        grouped[key][block] = max(grouped[key].get(block, -math.inf), value)
    if not grouped:
        raise CIRFContractError("conformal block-max requires rows")
    return {key: {block: values[block] for block in sorted(values)} for key, values in sorted(grouped.items())}


def top_l_worst_omission_envelope(
    retained_lower: Mapping[str, float], *, omitted_probability_upper: float, interval_contract: Mapping[str, Any]
) -> dict[str, float]:
    """Apply a deterministic worst-case top-L omission envelope once.

    It intentionally does not allocate a second confidence budget.  Every
    retained lower bound is reduced by the same pre-frozen omitted-mass upper
    envelope; this operation is deterministic after interval calibration.
    """

    validate_interval_contract(interval_contract)
    omitted = _finite(omitted_probability_upper, "omitted_probability_upper")
    if not 0.0 <= omitted <= 1.0:
        raise CIRFContractError("omitted probability envelope must lie in [0,1]")
    keys = _mapping_keys(retained_lower, "retained top-L lower bounds")
    if not keys:
        raise CIRFContractError("retained top-L lower bounds must not be empty")
    output: dict[str, float] = {}
    for key in sorted(keys):
        lower = _finite(retained_lower[key], "retained top-L lower bound")
        if not 0.0 <= lower <= 1.0:
            raise CIRFContractError("retained top-L lower bound must lie in [0,1]")
        output[key] = max(0.0, lower - omitted)
    return output


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(math.comb(n, i) * probability**i * (1.0 - probability) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper_bound(failures: int, total: int, delta: float) -> float:
    """One-sided exact binomial upper confidence bound using deterministic bisection."""

    if isinstance(failures, bool) or isinstance(total, bool) or not isinstance(failures, int) or not isinstance(total, int):
        raise CIRFContractError("failures and total must be integers")
    delta = _finite(delta, "delta")
    if total < 1 or failures < 0 or failures > total or not 0.0 < delta < 1.0:
        raise CIRFContractError("invalid Clopper-Pearson inputs")
    if failures == total:
        return 1.0
    if failures == 0:
        return 1.0 - delta ** (1.0 / total)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        # P(X <= failures | p) is decreasing in p.
        if _binomial_cdf(failures, total, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


RISK_NAMES = frozenset(
    {"R_known_id", "R_unknown_FA", "R_unknown_safe", "R_false_binding", "R_false_nonopportunity", "R_deadline"}
)
RISK_RECEIPT_REQUIRED = frozenset(
    {"schema_version", "risk_name", "split", "alpha", "delta", "loss_range", "block_max_losses"}
)


def seal_decision_risk_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a formal-test, block-max decision-risk audit receipt.

    The receipt contains only already-scored bounded loss by independent block;
    it is not a predictor input and cannot be traded against another risk.
    """

    raw = _copy_without_hash(payload, name="decision risk receipt", hash_field="risk_receipt_hash")
    raw = _copy_allowed(
        raw,
        name="decision risk receipt",
        allowed=RISK_RECEIPT_REQUIRED,
        required=RISK_RECEIPT_REQUIRED,
    )
    if raw["schema_version"] != RISK_RECEIPT_SCHEMA:
        raise CIRFContractError("unsupported decision risk receipt schema")
    if raw["risk_name"] not in RISK_NAMES:
        raise CIRFContractError("unknown decision risk name")
    if raw["split"] != "formal_test":
        raise CIRFContractError("decision risk receipt must use formal_test blocks")
    alpha = _finite(raw["alpha"], "risk alpha")
    delta = _finite(raw["delta"], "risk delta")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise CIRFContractError("risk alpha/delta must lie in (0,1)")
    loss_range = _interval(raw["loss_range"], "risk loss range")
    if loss_range != [0.0, 1.0]:
        raise CIRFContractError("decision risk loss range must be exactly [0,1]")
    losses = raw["block_max_losses"]
    if not isinstance(losses, Mapping) or not losses:
        raise CIRFContractError("decision risk block-max losses must be a non-empty mapping")
    normalized_losses: dict[str, float] = {}
    for block_id, loss in sorted(losses.items()):
        block_id = _nonempty_string(block_id, "risk block id")
        loss = _finite(loss, "risk block-max loss")
        if loss not in {0.0, 1.0}:
            raise CIRFContractError("risk block-max loss must be binary in [0,1]")
        normalized_losses[block_id] = loss
    failures = sum(1 for loss in normalized_losses.values() if loss > 0.0)
    independent_blocks = len(normalized_losses)
    bound = clopper_pearson_upper_bound(failures, independent_blocks, delta)
    raw.update(
        {
            "alpha": alpha,
            "delta": delta,
            "loss_range": loss_range,
            "block_max_losses": normalized_losses,
            "failures": failures,
            "independent_blocks": independent_blocks,
            "upper_bound": bound,
            "passes": bound <= alpha,
            "method": "one_sided_exact_clopper_pearson_block_max",
        }
    )
    raw["risk_receipt_hash"] = sha256_json(raw)
    return raw


def validate_decision_risk_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = RISK_RECEIPT_REQUIRED | {
        "failures",
        "independent_blocks",
        "upper_bound",
        "passes",
        "method",
        "risk_receipt_hash",
    }
    raw = _copy_allowed(payload, name="decision risk receipt", allowed=fields, required=fields)
    digest = _sha256_hex(raw.pop("risk_receipt_hash"), "risk_receipt_hash")
    # Re-seal only the stable source fields; derived values are then compared
    # exactly enough for deterministic binary-search arithmetic.
    source = {key: raw[key] for key in RISK_RECEIPT_REQUIRED}
    sealed = seal_decision_risk_receipt(source)
    if sealed["risk_receipt_hash"] != digest:
        raise CIRFContractError("decision risk receipt hash mismatch")
    for key in ("failures", "independent_blocks", "method", "passes"):
        if raw[key] != sealed[key]:
            raise CIRFContractError("decision risk receipt derived field mismatch")
    if abs(_finite(raw["upper_bound"], "risk upper bound") - sealed["upper_bound"]) > 1e-12:
        raise CIRFContractError("decision risk receipt upper bound mismatch")
    sealed["risk_receipt_hash"] = digest
    return sealed


def noncompensating_decision_gates(
    *,
    conformal_singleton: bool,
    risk_receipts: Mapping[str, Mapping[str, Any]],
    required_risks: Sequence[str] = ("R_known_id", "R_unknown_FA", "R_false_binding", "R_false_nonopportunity", "R_deadline"),
) -> dict[str, Any]:
    """Evaluate separately sealed risk gates without averaging or substitution."""

    if not isinstance(conformal_singleton, bool):
        raise CIRFContractError("conformal_singleton must be boolean")
    required = [_nonempty_string(value, "required risk name") for value in required_risks]
    if not required or len(required) != len(set(required)) or set(required).difference(RISK_NAMES):
        raise CIRFContractError("required risks must be a unique non-empty frozen risk subset")
    if not isinstance(risk_receipts, Mapping):
        raise CIRFContractError("risk_receipts must be a mapping")
    unknown = set(risk_receipts).difference(required)
    if unknown:
        raise CIRFContractError("risk receipt set differs from required non-compensating gates")
    gate_details: dict[str, dict[str, Any]] = {}
    for risk_name in required:
        receipt_payload = risk_receipts.get(risk_name)
        if receipt_payload is None:
            gate_details[risk_name] = {"present": False, "passes": False, "reason_code": "MISSING_RISK_RECEIPT"}
            continue
        receipt = validate_decision_risk_receipt(receipt_payload)
        if receipt["risk_name"] != risk_name:
            raise CIRFContractError("risk receipt name/key mismatch")
        gate_details[risk_name] = {
            "present": True,
            "passes": receipt["passes"],
            "upper_bound": receipt["upper_bound"],
            "risk_receipt_hash": receipt["risk_receipt_hash"],
        }
    all_risk_gates_pass = all(item["passes"] for item in gate_details.values())
    all_gates_pass = conformal_singleton and all_risk_gates_pass
    return {
        "schema_version": SCHEMA,
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "conformal_singleton": conformal_singleton,
        "risk_gates": gate_details,
        "all_risk_gates_pass": all_risk_gates_pass,
        "all_gates_pass": all_gates_pass,
        "technical_status": "ALL_NONCOMPENSATING_GATES_PASS" if all_gates_pass else "DEFER_NONCOMPENSATING_GATE_FAILURE",
    }


def decision_risk_gate(*, failures: int, independent_blocks: int, alpha: float, delta: float) -> dict[str, Any]:
    bound = clopper_pearson_upper_bound(failures, independent_blocks, delta)
    alpha = _finite(alpha, "risk alpha")
    if not 0.0 < alpha < 1.0:
        raise CIRFContractError("risk alpha must lie in (0,1)")
    return {
        "method": "one_sided_exact_clopper_pearson",
        "failures": failures,
        "independent_blocks": independent_blocks,
        "alpha": alpha,
        "delta": _finite(delta, "risk delta"),
        "upper_bound": bound,
        "passes": bound <= alpha,
    }


def conformal_vs_unknown_far_counterexample() -> dict[str, Any]:
    """A deterministic G0 counterexample proving the two gates are distinct."""

    conformal = split_conformal_quantile([0.01, 0.02, 0.03, 0.04, 0.05], 0.20)
    risk = decision_risk_gate(failures=2, independent_blocks=20, alpha=0.05, delta=0.05)
    return {
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "conformal_cell_has_finite_q": conformal["q"] <= 0.05,
        "unknown_far_gate_passes": risk["passes"],
        "risk_upper_bound": risk["upper_bound"],
        "conclusion": "CONFORMAL_AND_UNKNOWN_FAR_GATES_ARE_NOT_INTERCHANGEABLE",
    }


def registered_defer_is_error(decision: str) -> int:
    """Structural scorer-side rule: defer/unknown cannot pass registered ID."""

    return 0 if decision == "registered" else 1


def reject_all_is_not_a_safe_claim(decisions: Sequence[str]) -> bool:
    """Reject the degenerate all-defer/all-unknown shortcut in G0 assertions."""

    if not decisions:
        raise CIRFContractError("decisions must not be empty")
    return all(decision in {"unknown", "defer"} for decision in decisions)


# ---------------------------------------------------------------------------
# Network plane: finite transcript grammar and hard pre-send budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerContract:
    """Fully finite, pre-sealed scheduler grammar and message envelope.

    A catalog entry is the only source of request bytes, energy and worst-case
    delay.  Callers may report which *catalog ids* are physically available,
    but cannot insert an action or change its resource envelope at run time.
    """

    origins: tuple[str, ...]
    component_by_origin: Mapping[str, str]
    roster_epoch: str = "SYNTHETIC-ROSTER-1"
    delay_bucket_count: int = 1
    quantization_levels: int = 1
    interval_bin_counts: tuple[int, int, int] = (1, 1, 1)
    max_retransmissions: int = 0
    tier_paths: tuple[tuple[str, ...], ...] = (("T0",), ("T0", "T1"), ("T0", "T1", "T2"))
    hard_deadline_ms: float = 100.0
    seal_ack_bytes: int = 1
    seal_ack_energy: float = 0.0
    seal_ack_delay_ms: float = 0.0
    message_catalog: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def _catalog(self) -> dict[str, dict[str, Any]]:
        """Return one canonical entry for every finite ``origin×tier×retry`` action."""

        if not isinstance(self.message_catalog, Mapping) or not self.message_catalog:
            raise CIRFContractError("scheduler requires a frozen non-empty message catalog")
        catalog: dict[str, dict[str, Any]] = {}
        seen_slots: set[tuple[str, str, int]] = set()
        required_slots = {
            (origin, tier, retry)
            for origin in self.origins
            for tier in ("T1", "T2")
            for retry in range(self.max_retransmissions + 1)
        }
        for message_id, payload in self.message_catalog.items():
            message_id = _nonempty_string(message_id, "scheduler message id")
            raw = _copy_allowed(
                payload,
                name="scheduler catalog entry",
                allowed={"origin_id", "tier", "retransmission_index", "message_bytes", "energy_upper", "worst_delay_ms", "shrinkage_lower_bound"},
                required={"origin_id", "tier", "retransmission_index", "message_bytes", "energy_upper", "worst_delay_ms", "shrinkage_lower_bound"},
            )
            origin = _nonempty_string(raw["origin_id"], "catalog origin")
            if origin not in self.origins:
                raise CIRFContractError("scheduler catalog origin is outside frozen roster")
            if raw["tier"] not in {"T1", "T2"}:
                raise CIRFContractError("scheduler catalog tier must be T1 or T2")
            retry = raw["retransmission_index"]
            if isinstance(retry, bool) or not isinstance(retry, int) or not 0 <= retry <= self.max_retransmissions:
                raise CIRFContractError("scheduler catalog retransmission index out of range")
            if isinstance(raw["message_bytes"], bool) or not isinstance(raw["message_bytes"], int) or raw["message_bytes"] < 0:
                raise CIRFContractError("scheduler catalog message bytes invalid")
            energy = _finite(raw["energy_upper"], "scheduler catalog energy upper")
            delay = _finite(raw["worst_delay_ms"], "scheduler catalog worst delay")
            shrinkage = _finite(raw["shrinkage_lower_bound"], "scheduler catalog shrinkage lower bound")
            if energy < 0.0 or delay < 0.0 or shrinkage < 0.0:
                raise CIRFContractError("scheduler catalog resource/shrinkage value must be non-negative")
            slot = (origin, raw["tier"], retry)
            if slot in seen_slots:
                raise CIRFContractError("scheduler catalog has duplicate origin/tier/retry action")
            seen_slots.add(slot)
            catalog[message_id] = {
                "origin_id": origin,
                "tier": raw["tier"],
                "retransmission_index": retry,
                "message_bytes": raw["message_bytes"],
                "energy_upper": energy,
                "worst_delay_ms": delay,
                "shrinkage_lower_bound": shrinkage,
            }
        if seen_slots != required_slots:
            raise CIRFContractError("scheduler catalog must cover exactly every frozen origin/tier/retry action")
        return {message_id: catalog[message_id] for message_id in sorted(catalog)}

    def resource_envelope(self) -> dict[str, float | int]:
        catalog = self._catalog()
        return {
            "max_message_bytes": max(entry["message_bytes"] for entry in catalog.values()),
            "max_energy_upper": max(entry["energy_upper"] for entry in catalog.values()),
            "max_worst_delay_ms": max(entry["worst_delay_ms"] for entry in catalog.values()),
            "max_request_plus_ack_bytes": max(entry["message_bytes"] + self.seal_ack_bytes for entry in catalog.values()),
            "max_request_plus_ack_energy": max(entry["energy_upper"] + self.seal_ack_energy for entry in catalog.values()),
            "max_request_plus_ack_delay_ms": max(entry["worst_delay_ms"] + self.seal_ack_delay_ms for entry in catalog.values()),
        }

    def validate(self) -> None:
        origins = list(self.origins)
        if not origins or len(origins) != len(set(origins)) or len(origins) > MAX_ORIGINS:
            raise CIRFContractError("scheduler origins must be a unique roster of at most five")
        _nonempty_string(self.roster_epoch, "scheduler roster_epoch")
        if set(self.component_by_origin.keys()) != set(origins):
            raise CIRFContractError("scheduler component partition must cover exactly origins")
        for origin in origins:
            _nonempty_string(origin, "scheduler origin")
            _nonempty_string(self.component_by_origin[origin], "scheduler component")
        if not 1 <= self.delay_bucket_count <= 4 or not 1 <= self.quantization_levels <= 3:
            raise CIRFContractError("scheduler delay/quantization bounds exceed frozen limits")
        if len(self.interval_bin_counts) != 3 or any(not 1 <= value <= 4 for value in self.interval_bin_counts):
            raise CIRFContractError("scheduler requires three interval-bin counts in 1..4")
        if self.max_retransmissions not in {0, 1}:
            raise CIRFContractError("scheduler retransmission count must be 0 or 1")
        canonical_paths = (("T0",), ("T0", "T1"), ("T0", "T1", "T2"))
        if tuple(self.tier_paths) != canonical_paths:
            raise CIRFContractError("scheduler tier paths differ from frozen grammar")
        if _finite(self.hard_deadline_ms, "hard_deadline_ms") < 0.0:
            raise CIRFContractError("hard deadline must be non-negative")
        if isinstance(self.seal_ack_bytes, bool) or not isinstance(self.seal_ack_bytes, int) or self.seal_ack_bytes < 0:
            raise CIRFContractError("seal acknowledgment bytes must be non-negative integer")
        for value, name in (
            (self.seal_ack_energy, "seal_ack_energy"),
            (self.seal_ack_delay_ms, "seal_ack_delay_ms"),
        ):
            if _finite(value, name) < 0.0:
                raise CIRFContractError(f"{name} must be non-negative")
        self._catalog()

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": TRANSCRIPT_SCHEMA,
            "origins": list(self.origins),
            "component_by_origin": {key: self.component_by_origin[key] for key in sorted(self.component_by_origin)},
            "roster_epoch": self.roster_epoch,
            "delay_bucket_count": self.delay_bucket_count,
            "quantization_levels": self.quantization_levels,
            "interval_bin_counts": list(self.interval_bin_counts),
            "max_retransmissions": self.max_retransmissions,
            "tier_paths": [list(path) for path in self.tier_paths],
            "hard_deadline_ms": float(self.hard_deadline_ms),
            "seal_ack_bytes": self.seal_ack_bytes,
            "seal_ack_energy": float(self.seal_ack_energy),
            "seal_ack_delay_ms": float(self.seal_ack_delay_ms),
            "message_catalog": self._catalog(),
            "max_resources": self.resource_envelope(),
        }

    @property
    def contract_hash(self) -> str:
        return sha256_json(self.canonical())


def reachable_transcript_breakdown(contract: SchedulerContract) -> dict[str, Any]:
    """Exact count under the frozen finite grammar, without sampling traces."""

    contract.validate()
    m = len(contract.origins)
    interval_states = math.prod(contract.interval_bin_counts)
    path_states = len(contract.tier_paths)
    by_active_count: dict[int, int] = {}
    for active_count in range(1, m + 1):
        # Ordered active subset encodes active origins, arrival order and the
        # complement dropout mask.  Per-origin delay/retry/quantization states
        # and global interval/tier state complete the finite transcript key.
        count = math.perm(m, active_count)
        count *= contract.delay_bucket_count**active_count
        count *= (contract.max_retransmissions + 1) ** active_count
        count *= contract.quantization_levels**active_count
        count *= interval_states * path_states
        by_active_count[active_count] = count
    return {
        "contract_hash": contract.contract_hash,
        "by_active_count": by_active_count,
        "reachable_transcript_count": sum(by_active_count.values()),
    }


TRANSCRIPT_STATE_REQUIRED = frozenset(
    {
        "schema_version",
        "contract_hash",
        "event_authority_receipt_hash",
        "opportunity_id",
        "scheduler_replay_store_id",
        "scheduler_session_hash",
        "initial_budget_hash",
        "roster_epoch",
        "active_origins",
        "arrival_order",
        "delay_buckets",
        "dropout_mask",
        "quantization_levels",
        "retransmissions",
        "interval_bins",
        "tier_path",
        "stop_reason",
        "elapsed_ms",
        "parent_state_hash",
        "request_history",
        "cumulative_reserved_resources",
        "state_hash",
    }
)


def _transcript_state_hash(state: Mapping[str, Any]) -> str:
    payload = {key: state[key] for key in sorted(TRANSCRIPT_STATE_REQUIRED.difference({"state_hash"}))}
    return sha256_json(payload)


def _seal_transcript_state(state: Mapping[str, Any], contract: SchedulerContract) -> dict[str, Any]:
    sealed = dict(state)
    sealed["contract_hash"] = contract.contract_hash
    sealed["state_hash"] = _transcript_state_hash(sealed)
    return sealed


def enumerate_transcript_prefixes(contract: SchedulerContract, *, limit: int = MAX_TRANSCRIPTS) -> list[dict[str, Any]]:
    """Enumerate every reachable prefix exactly, or refuse before partial work."""

    breakdown = reachable_transcript_breakdown(contract)
    count = breakdown["reachable_transcript_count"]
    if count > limit:
        raise CapacityRejected(f"reachable transcript count {count} exceeds enumeration limit {limit}")
    origins = list(contract.origins)
    result: list[dict[str, Any]] = []
    for active_count in range(1, len(origins) + 1):
        for arrival_order in itertools.permutations(origins, active_count):
            active = list(arrival_order)
            dropout = sorted(set(origins).difference(active))
            for delays in itertools.product(range(contract.delay_bucket_count), repeat=active_count):
                for retransmissions in itertools.product(range(contract.max_retransmissions + 1), repeat=active_count):
                    for quantization in itertools.product(range(contract.quantization_levels), repeat=active_count):
                        for interval_bins in itertools.product(*[range(value) for value in contract.interval_bin_counts]):
                            for path in contract.tier_paths:
                                result.append(
                                    _seal_transcript_state(
                                         {
                                             "schema_version": TRANSCRIPT_SCHEMA,
                                             "event_authority_receipt_hash": SCHEDULER_TEMPLATE_BINDING_HASH,
                                             "opportunity_id": SCHEDULER_TEMPLATE_BINDING_HASH,
                                             "scheduler_replay_store_id": SCHEDULER_TEMPLATE_BINDING_HASH,
                                             "scheduler_session_hash": SCHEDULER_TEMPLATE_BINDING_HASH,
                                             "initial_budget_hash": SCHEDULER_TEMPLATE_BINDING_HASH,
                                             "roster_epoch": contract.roster_epoch,
                                            "active_origins": active,
                                            "arrival_order": list(arrival_order),
                                            "delay_buckets": list(delays),
                                            "dropout_mask": dropout,
                                            "quantization_levels": list(quantization),
                                            "retransmissions": list(retransmissions),
                                            "interval_bins": list(interval_bins),
                                            "tier_path": list(path),
                                            "stop_reason": "ENUMERATED_PREFIX",
                                            "elapsed_ms": 0.0,
                                            "parent_state_hash": None,
                                            "request_history": [],
                                            "cumulative_reserved_resources": {
                                                "bytes": 0,
                                                "energy": 0.0,
                                                "delay_ms": 0.0,
                                            },
                                        },
                                        contract,
                                    )
                                )
    if len(result) != count:
        raise CIRFContractError("finite transcript enumerator/count disagreement")
    return result


def validate_transcript_state(state: Mapping[str, Any], contract: SchedulerContract) -> dict[str, Any]:
    contract.validate()
    raw = _copy_allowed(
        state, name="network transcript state", allowed=TRANSCRIPT_STATE_REQUIRED, required=TRANSCRIPT_STATE_REQUIRED
    )
    if raw["schema_version"] != TRANSCRIPT_SCHEMA:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    if raw["contract_hash"] != contract.contract_hash:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    for field_name in (
        "event_authority_receipt_hash",
        "opportunity_id",
        "scheduler_replay_store_id",
        "scheduler_session_hash",
        "initial_budget_hash",
    ):
        raw[field_name] = _sha256_hex(raw[field_name], f"network {field_name}")
    if raw["roster_epoch"] != contract.roster_epoch:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    origins = list(contract.origins)
    active = raw["active_origins"]
    order = raw["arrival_order"]
    if not isinstance(active, list) or not active or len(active) != len(set(active)) or set(active).difference(origins):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    if order != active or len(order) != len(set(order)):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    if sorted(raw["dropout_mask"]) != sorted(set(origins).difference(active)):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    length = len(active)
    for field_name, upper in (
        ("delay_buckets", contract.delay_bucket_count),
        ("quantization_levels", contract.quantization_levels),
        ("retransmissions", contract.max_retransmissions + 1),
    ):
        values = raw[field_name]
        if not isinstance(values, list) or len(values) != length or any(not isinstance(value, int) or not 0 <= value < upper for value in values):
            raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    bins = raw["interval_bins"]
    if not isinstance(bins, list) or len(bins) != 3 or any(
        not isinstance(value, int) or not 0 <= value < contract.interval_bin_counts[index]
        for index, value in enumerate(bins)
    ):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    if tuple(raw["tier_path"]) not in contract.tier_paths or raw["stop_reason"] not in {
        "ENUMERATED_PREFIX",
        "REQUESTED",
        "SEALED",
        "DEFERRED",
        "DEADLINE_EXPIRED",
    }:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    elapsed = _finite(raw["elapsed_ms"], "network elapsed_ms")
    if elapsed < 0.0 or elapsed > contract.hard_deadline_ms + FEASIBILITY_TOL:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    raw["elapsed_ms"] = elapsed
    parent = raw["parent_state_hash"]
    if parent is not None:
        parent = _sha256_hex(parent, "network parent_state_hash")
    raw["parent_state_hash"] = parent
    history = raw["request_history"]
    if not isinstance(history, list) or any(not isinstance(message_id, str) or not message_id for message_id in history):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    catalog = contract._catalog()
    if any(message_id not in catalog for message_id in history) or len(history) != len(set(history)):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    cumulative = _copy_allowed(
        raw["cumulative_reserved_resources"],
        name="network cumulative reserved resources",
        allowed={"bytes", "energy", "delay_ms"},
        required={"bytes", "energy", "delay_ms"},
    )
    if isinstance(cumulative["bytes"], bool) or not isinstance(cumulative["bytes"], int) or cumulative["bytes"] < 0:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    cumulative["energy"] = _finite(cumulative["energy"], "network cumulative energy")
    cumulative["delay_ms"] = _finite(cumulative["delay_ms"], "network cumulative delay")
    if cumulative["energy"] < 0.0 or cumulative["delay_ms"] < 0.0:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    expected_bytes = sum(catalog[message_id]["message_bytes"] + contract.seal_ack_bytes for message_id in history)
    expected_energy = sum(catalog[message_id]["energy_upper"] + contract.seal_ack_energy for message_id in history)
    expected_delay = sum(catalog[message_id]["worst_delay_ms"] + contract.seal_ack_delay_ms for message_id in history)
    if (
        cumulative["bytes"] != expected_bytes
        or not math.isclose(cumulative["energy"], expected_energy, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(cumulative["delay_ms"], expected_delay, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(elapsed, expected_delay, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    raw["cumulative_reserved_resources"] = cumulative
    if history:
        if parent is None or raw["stop_reason"] != "REQUESTED":
            raise CIRFContractError("UNKNOWN_NETWORK_STATE")
        if [catalog[message_id]["tier"] for message_id in history] != raw["tier_path"][1:]:
            raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    elif parent is not None:
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    raw["state_hash"] = _sha256_hex(raw["state_hash"], "network state_hash")
    if raw["state_hash"] != _transcript_state_hash(raw):
        raise CIRFContractError("UNKNOWN_NETWORK_STATE")
    return raw


@dataclass(frozen=True)
class ResourceBudget:
    bytes_remaining: int
    energy_remaining: float
    deadline_slack_ms: float

    def validate(self) -> None:
        if isinstance(self.bytes_remaining, bool) or not isinstance(self.bytes_remaining, int) or self.bytes_remaining < 0:
            raise CIRFContractError("bytes_remaining must be a non-negative integer")
        if _finite(self.energy_remaining, "energy_remaining") < 0.0:
            raise CIRFContractError("energy_remaining must be non-negative")
        if _finite(self.deadline_slack_ms, "deadline_slack_ms") < 0.0:
            raise CIRFContractError("deadline_slack_ms must be non-negative")


def _resource_budget_payload(budget: ResourceBudget) -> dict[str, Any]:
    budget.validate()
    return {
        "bytes_remaining": budget.bytes_remaining,
        "energy_remaining": float(budget.energy_remaining),
        "deadline_slack_ms": float(budget.deadline_slack_ms),
    }


def _resource_budget_hash(budget: ResourceBudget) -> str:
    return sha256_json(_resource_budget_payload(budget))


def scheduler_replay_store_id(root: str | Path) -> str:
    """Bind an event-authority receipt to one canonical persistent store."""

    canonical_root = Path(root).resolve().as_posix()
    return sha256_json(
        {
            "schema_version": SCHEDULER_LEDGER_SCHEMA,
            "canonical_store_root": canonical_root,
        }
    )


SCHEDULER_REPLAY_RECEIPT_REQUIRED = frozenset(
    {
        "schema_version",
        "store_id",
        "contract_hash",
        "event_authority_receipt_hash",
        "opportunity_id",
        "scheduler_session_hash",
        "root_state_hash",
        "initial_budget_hash",
        "ledger_generation",
        "ledger_snapshot_hash",
        "issued_state_hashes",
        "consumed_reason_by_state_hash",
        "active_state_hashes",
    }
)


def validate_scheduler_replay_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="scheduler replay receipt",
        allowed=SCHEDULER_REPLAY_RECEIPT_REQUIRED | {"replay_receipt_hash"},
        required=SCHEDULER_REPLAY_RECEIPT_REQUIRED | {"replay_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("replay_receipt_hash"), "scheduler replay_receipt_hash")
    if raw["schema_version"] != SCHEDULER_REPLAY_SCHEMA:
        raise CIRFContractError("unsupported scheduler replay receipt schema")
    for field_name in (
        "store_id",
        "contract_hash",
        "event_authority_receipt_hash",
        "opportunity_id",
        "scheduler_session_hash",
        "root_state_hash",
        "initial_budget_hash",
        "ledger_snapshot_hash",
    ):
        raw[field_name] = _sha256_hex(raw[field_name], f"scheduler replay {field_name}")
    if isinstance(raw["ledger_generation"], bool) or not isinstance(raw["ledger_generation"], int) or raw["ledger_generation"] < 0:
        raise CIRFContractError("scheduler replay ledger_generation must be non-negative")
    issued = raw["issued_state_hashes"]
    active = raw["active_state_hashes"]
    consumed = raw["consumed_reason_by_state_hash"]
    if not isinstance(issued, list) or not issued or not isinstance(active, list) or not isinstance(consumed, Mapping):
        raise CIRFContractError("scheduler replay state sets malformed")
    issued = sorted(_sha256_hex(value, "scheduler issued state hash") for value in issued)
    active = sorted(_sha256_hex(value, "scheduler active state hash") for value in active)
    if len(issued) != len(set(issued)) or len(active) != len(set(active)):
        raise CIRFContractError("scheduler replay state hashes must be unique")
    normalized_consumed = {
        _sha256_hex(state_hash, "scheduler consumed state hash"): _nonempty_string(reason, "scheduler consumed reason")
        for state_hash, reason in consumed.items()
    }
    if set(active).intersection(normalized_consumed) or set(active).union(normalized_consumed) != set(issued):
        raise CIRFContractError("scheduler replay active/consumed partition mismatch")
    raw["issued_state_hashes"] = issued
    raw["active_state_hashes"] = active
    raw["consumed_reason_by_state_hash"] = dict(sorted(normalized_consumed.items()))
    if sha256_json(raw) != digest:
        raise CIRFContractError("scheduler replay receipt hash mismatch")
    raw["replay_receipt_hash"] = digest
    return raw


class SchedulerEventAuthority:
    """Cross-process, append-only one-event scheduler authority.

    The authority store is fixed in the pre-query event-authority receipt.  A
    scheduler process may be restarted, but it must recover the latest sealed
    generation for the same session and can never reactivate a consumed root.
    Every transition appends a new generation under an exclusive session lock;
    existing generations are never overwritten.
    """

    def __init__(
        self,
        contract: SchedulerContract,
        *,
        event_authority_receipt: Mapping[str, Any],
        root_template: Mapping[str, Any],
        initial_budget: ResourceBudget,
        replay_store_root: str | Path,
        prior_replay_receipt: Mapping[str, Any] | None = None,
    ):
        contract.validate()
        authority = validate_event_authority_receipt(event_authority_receipt)
        if authority["roster_epoch"] != contract.roster_epoch:
            raise CIRFContractError("scheduler/event-authority roster mismatch")
        store_root = Path(replay_store_root).resolve()
        store_id = scheduler_replay_store_id(store_root)
        if authority["scheduler_replay_store_id"] != store_id:
            raise CIRFContractError("scheduler replay store is not the pre-sealed event-authority store")
        template = validate_transcript_state(root_template, contract)
        if any(
            template[field_name] != SCHEDULER_TEMPLATE_BINDING_HASH
            for field_name in (
                "event_authority_receipt_hash",
                "opportunity_id",
                "scheduler_replay_store_id",
                "scheduler_session_hash",
                "initial_budget_hash",
            )
        ):
            raise CIRFContractError("scheduler root must be issued from an unbound enumerated template")
        if not (
            template["parent_state_hash"] is None
            and template["request_history"] == []
            and template["tier_path"] == ["T0"]
            and template["stop_reason"] == "ENUMERATED_PREFIX"
            and template["elapsed_ms"] == 0.0
            and template["cumulative_reserved_resources"] == {"bytes": 0, "energy": 0.0, "delay_ms": 0.0}
        ):
            raise CIRFContractError("scheduler authority requires a T0 root template")
        budget_payload = _resource_budget_payload(initial_budget)
        initial_budget_hash = sha256_json(budget_payload)
        session_hash = sha256_json(
            {
                "schema_version": SCHEDULER_REPLAY_SCHEMA,
                "store_id": store_id,
                "contract_hash": contract.contract_hash,
                "event_authority_receipt_hash": authority["authority_receipt_hash"],
                "opportunity_id": authority["opportunity_id"],
                "root_template_hash": template["state_hash"],
                "initial_budget_hash": initial_budget_hash,
            }
        )
        root = copy.deepcopy(template)
        root.update(
            {
                "event_authority_receipt_hash": authority["authority_receipt_hash"],
                "opportunity_id": authority["opportunity_id"],
                "scheduler_replay_store_id": store_id,
                "scheduler_session_hash": session_hash,
                "initial_budget_hash": initial_budget_hash,
            }
        )
        root = validate_transcript_state(_seal_transcript_state(root, contract), contract)

        self.contract = contract
        self.contract_hash = contract.contract_hash
        self.store_id = store_id
        self.event_authority_receipt_hash = authority["authority_receipt_hash"]
        self.opportunity_id = authority["opportunity_id"]
        self.scheduler_session_hash = session_hash
        self.initial_budget_hash = initial_budget_hash
        self._root_state_hash = root["state_hash"]
        self._initial_budget = ResourceBudget(**budget_payload)
        self._store_root = store_root
        self._session_dir = store_root / session_hash
        self._lock_path = store_root / f"{session_hash}.lock"
        self._open_or_recover(root, prior_replay_receipt=prior_replay_receipt)

    @staticmethod
    def _same_budget(left: ResourceBudget, right: ResourceBudget) -> bool:
        return (
            left.bytes_remaining == right.bytes_remaining
            and math.isclose(left.energy_remaining, right.energy_remaining, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(left.deadline_slack_ms, right.deadline_slack_ms, rel_tol=0.0, abs_tol=1e-12)
        )

    def _initial_snapshot(self, root: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEDULER_LEDGER_SCHEMA,
            "store_id": self.store_id,
            "contract_hash": self.contract_hash,
            "event_authority_receipt_hash": self.event_authority_receipt_hash,
            "opportunity_id": self.opportunity_id,
            "scheduler_session_hash": self.scheduler_session_hash,
            "root_state_hash": self._root_state_hash,
            "initial_budget_hash": self.initial_budget_hash,
            "generation": 0,
            "parent_snapshot_hash": None,
            "issued_states": {self._root_state_hash: copy.deepcopy(dict(root))},
            "budget_by_state_hash": {self._root_state_hash: _resource_budget_payload(self._initial_budget)},
            "consumed_reason_by_state_hash": {},
            "transition_history": [],
        }
        payload["ledger_snapshot_hash"] = sha256_json(payload)
        return payload

    def _validate_snapshot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "schema_version",
            "store_id",
            "contract_hash",
            "event_authority_receipt_hash",
            "opportunity_id",
            "scheduler_session_hash",
            "root_state_hash",
            "initial_budget_hash",
            "generation",
            "parent_snapshot_hash",
            "issued_states",
            "budget_by_state_hash",
            "consumed_reason_by_state_hash",
            "transition_history",
            "ledger_snapshot_hash",
        }
        raw = _copy_allowed(payload, name="scheduler session ledger", allowed=required, required=required)
        digest = _sha256_hex(raw.pop("ledger_snapshot_hash"), "scheduler ledger_snapshot_hash")
        if raw["schema_version"] != SCHEDULER_LEDGER_SCHEMA:
            raise CIRFContractError("unsupported scheduler ledger schema")
        expected_bindings = {
            "store_id": self.store_id,
            "contract_hash": self.contract_hash,
            "event_authority_receipt_hash": self.event_authority_receipt_hash,
            "opportunity_id": self.opportunity_id,
            "scheduler_session_hash": self.scheduler_session_hash,
            "root_state_hash": self._root_state_hash,
            "initial_budget_hash": self.initial_budget_hash,
        }
        if any(raw[name] != value for name, value in expected_bindings.items()):
            raise CIRFContractError("scheduler ledger session binding mismatch")
        if isinstance(raw["generation"], bool) or not isinstance(raw["generation"], int) or raw["generation"] < 0:
            raise CIRFContractError("scheduler ledger generation invalid")
        parent = raw["parent_snapshot_hash"]
        if parent is not None:
            raw["parent_snapshot_hash"] = _sha256_hex(parent, "scheduler parent_snapshot_hash")
        issued = raw["issued_states"]
        budgets = raw["budget_by_state_hash"]
        consumed = raw["consumed_reason_by_state_hash"]
        history = raw["transition_history"]
        if not isinstance(issued, Mapping) or not issued or not isinstance(budgets, Mapping) or not isinstance(consumed, Mapping) or not isinstance(history, list):
            raise CIRFContractError("scheduler ledger state maps malformed")
        normalized_issued: dict[str, dict[str, Any]] = {}
        normalized_budgets: dict[str, dict[str, Any]] = {}
        for state_hash, state in issued.items():
            state_hash = _sha256_hex(state_hash, "scheduler issued state hash")
            validated_state = validate_transcript_state(state, self.contract)
            if validated_state["state_hash"] != state_hash:
                raise CIRFContractError("scheduler ledger issued-state key mismatch")
            normalized_issued[state_hash] = validated_state
        for state_hash, budget_payload in budgets.items():
            state_hash = _sha256_hex(state_hash, "scheduler budget state hash")
            if not isinstance(budget_payload, Mapping):
                raise CIRFContractError("scheduler ledger budget malformed")
            budget = ResourceBudget(**dict(budget_payload))
            normalized_budgets[state_hash] = _resource_budget_payload(budget)
        normalized_consumed = {
            _sha256_hex(state_hash, "scheduler consumed state hash"): _nonempty_string(reason, "scheduler consumed reason")
            for state_hash, reason in consumed.items()
        }
        if set(normalized_issued) != set(normalized_budgets) or set(normalized_consumed).difference(normalized_issued):
            raise CIRFContractError("scheduler ledger issued/budget/consumed mismatch")
        if len(history) != raw["generation"]:
            raise CIRFContractError("scheduler ledger transition count mismatch")
        normalized_history: list[dict[str, Any]] = []
        for expected_generation, transition in enumerate(history, start=1):
            if not isinstance(transition, Mapping):
                raise CIRFContractError("scheduler ledger transition malformed")
            item = _copy_allowed(
                transition,
                name="scheduler ledger transition",
                allowed={"generation", "consumed_state_hash", "reason_code", "issued_state_hash", "transition_hash"},
                required={"generation", "consumed_state_hash", "reason_code", "issued_state_hash", "transition_hash"},
            )
            transition_hash = _sha256_hex(item.pop("transition_hash"), "scheduler transition_hash")
            if item["generation"] != expected_generation:
                raise CIRFContractError("scheduler ledger transition generation mismatch")
            item["consumed_state_hash"] = _sha256_hex(item["consumed_state_hash"], "scheduler transition consumed hash")
            if item["issued_state_hash"] is not None:
                item["issued_state_hash"] = _sha256_hex(item["issued_state_hash"], "scheduler transition issued hash")
            item["reason_code"] = _nonempty_string(item["reason_code"], "scheduler transition reason")
            if sha256_json(item) != transition_hash:
                raise CIRFContractError("scheduler transition hash mismatch")
            item["transition_hash"] = transition_hash
            normalized_history.append(item)
        raw["issued_states"] = normalized_issued
        raw["budget_by_state_hash"] = normalized_budgets
        raw["consumed_reason_by_state_hash"] = dict(sorted(normalized_consumed.items()))
        raw["transition_history"] = normalized_history
        if sha256_json(raw) != digest:
            raise CIRFContractError("scheduler ledger snapshot hash mismatch")
        raw["ledger_snapshot_hash"] = digest
        return raw

    def _generation_files(self) -> list[Path]:
        if not self._session_dir.exists():
            return []
        files = sorted(self._session_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json"))
        return files

    def _load_latest(self) -> dict[str, Any] | None:
        files = self._generation_files()
        if not files:
            return None
        expected_names = [f"{index:020d}.json" for index in range(len(files))]
        if [path.name for path in files] != expected_names:
            raise CIRFContractError("scheduler ledger generation sequence has a gap")
        previous_hash: str | None = None
        latest: dict[str, Any] | None = None
        for index, path in enumerate(files):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CIRFContractError("scheduler ledger snapshot is unreadable") from exc
            latest = self._validate_snapshot(payload)
            if latest["generation"] != index or latest["parent_snapshot_hash"] != previous_hash:
                raise CIRFContractError("scheduler ledger append chain mismatch")
            previous_hash = latest["ledger_snapshot_hash"]
        return latest

    def _write_generation(self, snapshot: Mapping[str, Any]) -> None:
        generation = snapshot["generation"]
        target = self._session_dir / f"{generation:020d}.json"
        temporary = self._session_dir / f".{generation:020d}.pending"
        content = canonical_json(dict(snapshot)) + "\n"
        if target.exists() or temporary.exists():
            raise CIRFContractError("scheduler ledger refuses generation overwrite")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            temporary.replace(target)
        except (FileExistsError, OSError) as exc:
            raise CIRFContractError("scheduler ledger generation append failed") from exc

    def _acquire_lock(self):
        self._store_root.mkdir(parents=True, exist_ok=True)
        try:
            handle = self._lock_path.open("x", encoding="utf-8", newline="\n")
            handle.write(self.scheduler_session_hash + "\n")
            handle.flush()
            return handle
        except FileExistsError as exc:
            raise CIRFContractError("scheduler replay session is locked") from exc

    def _release_lock(self, handle) -> None:
        handle.close()
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def _receipt_from_snapshot(self, latest: Mapping[str, Any]) -> dict[str, Any]:
        issued = sorted(latest["issued_states"])
        consumed = dict(sorted(latest["consumed_reason_by_state_hash"].items()))
        payload = {
            "schema_version": SCHEDULER_REPLAY_SCHEMA,
            "store_id": self.store_id,
            "contract_hash": self.contract_hash,
            "event_authority_receipt_hash": self.event_authority_receipt_hash,
            "opportunity_id": self.opportunity_id,
            "scheduler_session_hash": self.scheduler_session_hash,
            "root_state_hash": self._root_state_hash,
            "initial_budget_hash": self.initial_budget_hash,
            "ledger_generation": latest["generation"],
            "ledger_snapshot_hash": latest["ledger_snapshot_hash"],
            "issued_state_hashes": issued,
            "consumed_reason_by_state_hash": consumed,
            "active_state_hashes": sorted(set(issued).difference(consumed)),
        }
        payload["replay_receipt_hash"] = sha256_json(payload)
        return validate_scheduler_replay_receipt(payload)

    def _open_or_recover(
        self,
        root: Mapping[str, Any],
        *,
        prior_replay_receipt: Mapping[str, Any] | None,
    ) -> None:
        handle = self._acquire_lock()
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            latest = self._load_latest()
            if latest is None:
                if prior_replay_receipt is not None:
                    raise CIRFContractError("scheduler replay receipt supplied for a missing session")
                self._write_generation(self._initial_snapshot(root))
                latest = self._load_latest()
            else:
                if prior_replay_receipt is None:
                    raise CIRFContractError("existing scheduler session requires its latest external replay receipt")
                validated_prior = validate_scheduler_replay_receipt(prior_replay_receipt)
                if validated_prior != self._receipt_from_snapshot(latest):
                    raise CIRFContractError("scheduler replay receipt does not anchor the latest ledger head")
            if latest is None or latest["root_state_hash"] != root["state_hash"]:
                raise CIRFContractError("scheduler ledger root recovery mismatch")
        finally:
            self._release_lock(handle)

    def _append_transition(
        self,
        *,
        current_state_hash: str,
        reason_code: str,
        next_state: Mapping[str, Any] | None,
        next_budget: ResourceBudget | None,
    ) -> bool:
        reason = _nonempty_string(reason_code, "scheduler transition reason")
        handle = self._acquire_lock()
        try:
            latest = self._load_latest()
            if latest is None:
                raise CIRFContractError("scheduler ledger session missing")
            issued = latest["issued_states"]
            consumed = latest["consumed_reason_by_state_hash"]
            if current_state_hash not in issued or current_state_hash in consumed:
                return False
            next_hash: str | None = None
            if next_state is not None:
                if next_budget is None:
                    raise CIRFContractError("scheduler next budget missing")
                next_copy = validate_transcript_state(copy.deepcopy(dict(next_state)), self.contract)
                next_hash = next_copy["state_hash"]
                if next_hash in issued or next_hash in consumed:
                    return False
                issued[next_hash] = next_copy
                latest["budget_by_state_hash"][next_hash] = _resource_budget_payload(next_budget)
            consumed[current_state_hash] = reason
            generation = latest["generation"] + 1
            transition = {
                "generation": generation,
                "consumed_state_hash": current_state_hash,
                "reason_code": reason,
                "issued_state_hash": next_hash,
            }
            transition["transition_hash"] = sha256_json(transition)
            new_snapshot = {
                key: copy.deepcopy(value)
                for key, value in latest.items()
                if key not in {"ledger_snapshot_hash"}
            }
            new_snapshot["generation"] = generation
            new_snapshot["parent_snapshot_hash"] = latest["ledger_snapshot_hash"]
            new_snapshot["transition_history"] = [*latest["transition_history"], transition]
            new_snapshot["ledger_snapshot_hash"] = sha256_json(new_snapshot)
            self._write_generation(new_snapshot)
            return True
        finally:
            self._release_lock(handle)

    @property
    def root_state(self) -> dict[str, Any]:
        latest = self._load_latest()
        if latest is None:
            raise CIRFContractError("scheduler ledger session missing")
        return copy.deepcopy(latest["issued_states"][self._root_state_hash])

    @property
    def initial_budget(self) -> ResourceBudget:
        return ResourceBudget(**_resource_budget_payload(self._initial_budget))

    def authorize(self, state: Mapping[str, Any], budget: ResourceBudget) -> bool:
        latest = self._load_latest()
        if latest is None:
            return False
        state_hash = state["state_hash"]
        expected_state = latest["issued_states"].get(state_hash)
        budget_payload = latest["budget_by_state_hash"].get(state_hash)
        if expected_state is None or budget_payload is None or state_hash in latest["consumed_reason_by_state_hash"]:
            return False
        expected_budget = ResourceBudget(**budget_payload)
        return expected_state == state and self._same_budget(expected_budget, budget)

    def consume_terminal(self, state_hash: str, reason_code: str) -> bool:
        return self._append_transition(
            current_state_hash=state_hash,
            reason_code=reason_code,
            next_state=None,
            next_budget=None,
        )

    def consume_and_issue(
        self,
        *,
        current_state_hash: str,
        next_state: Mapping[str, Any],
        next_budget: ResourceBudget,
        reason_code: str,
    ) -> bool:
        return self._append_transition(
            current_state_hash=current_state_hash,
            reason_code=reason_code,
            next_state=next_state,
            next_budget=next_budget,
        )

    def replay_receipt(self) -> dict[str, Any]:
        latest = self._load_latest()
        if latest is None:
            raise CIRFContractError("scheduler ledger session missing")
        return self._receipt_from_snapshot(latest)


class FrozenScheduler:
    """A one-event finite scheduler with an internally issued state chain."""

    def __init__(self, contract: SchedulerContract, *, authority: SchedulerEventAuthority):
        contract.validate()
        if not isinstance(authority, SchedulerEventAuthority) or authority.contract_hash != contract.contract_hash:
            raise CIRFContractError("scheduler requires the matching persistent event authority")
        self.contract = contract
        self.authority = authority

    def _authorize_state(self, state: dict[str, Any], budget: ResourceBudget) -> bool:
        return self.authority.authorize(state, budget)

    def request_next(
        self,
        state: Mapping[str, Any],
        budget: ResourceBudget,
        available_message_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Choose only a catalogued action and reserve its full worst-case cost.

        The public input is intentionally restricted to message identifiers.
        It cannot carry a query-time byte cost, score, winner, or newly-created
        action.  The returned state closes elapsed time at the same instant the
        bounded request and its seal acknowledgement are reserved.
        """

        try:
            valid_state = validate_transcript_state(state, self.contract)
        except CIRFContractError:
            return {"action": "DEFER", "reason_code": "UNKNOWN_NETWORK_STATE"}
        budget.validate()
        if not self._authorize_state(valid_state, budget):
            return {"action": "DEFER", "reason_code": "UNKNOWN_NETWORK_STATE"}

        def terminal(payload: dict[str, Any]) -> dict[str, Any]:
            if not self.authority.consume_terminal(valid_state["state_hash"], payload["reason_code"]):
                return {"action": "DEFER", "reason_code": "UNKNOWN_NETWORK_STATE"}
            return payload

        if valid_state["elapsed_ms"] >= self.contract.hard_deadline_ms - FEASIBILITY_TOL:
            return terminal({"action": "SEAL_OR_DEFER", "reason_code": "HARD_DEADLINE_EXPIRED"})
        if not isinstance(available_message_ids, Sequence) or isinstance(available_message_ids, (str, bytes)):
            raise CIRFContractError("available scheduler messages must be a sequence of frozen ids")
        requested_ids = [_nonempty_string(value, "available scheduler message id") for value in available_message_ids]
        if len(requested_ids) != len(set(requested_ids)):
            raise CIRFContractError("available scheduler message ids must be unique")
        catalog = self.contract._catalog()
        unknown_ids = sorted(set(requested_ids).difference(catalog))
        if unknown_ids:
            raise CIRFContractError("available scheduler message is absent from frozen catalog")
        active_components = {self.contract.component_by_origin[origin] for origin in valid_state["active_origins"]}
        current_path = tuple(valid_state["tier_path"])
        next_tier = "T1" if current_path == ("T0",) else "T2" if current_path == ("T0", "T1") else None
        valid_candidates: list[tuple[str, dict[str, Any]]] = []
        for message_id in requested_ids:
            item = catalog[message_id]
            origin = item["origin_id"]
            if item["shrinkage_lower_bound"] <= 0.0 or item["tier"] != next_tier:
                continue
            if origin in valid_state["active_origins"]:
                # A present origin may only use the next frozen retry slot.
                index = valid_state["active_origins"].index(origin)
                if item["retransmission_index"] != valid_state["retransmissions"][index] + 1:
                    continue
            elif item["retransmission_index"] != 0:
                continue
            copied = dict(item)
            copied["independent_component"] = self.contract.component_by_origin[origin] not in active_components
            cost = max(1, copied["message_bytes"]) + copied["energy_upper"] + copied["worst_delay_ms"]
            copied["priority"] = (
                0 if copied["independent_component"] else 1,
                -copied["shrinkage_lower_bound"] / cost,
                hashlib.sha256(origin.encode("utf-8")).hexdigest(),
                hashlib.sha256(message_id.encode("utf-8")).hexdigest(),
            )
            valid_candidates.append((message_id, copied))
        if not valid_candidates:
            return terminal({"action": "SEAL_OR_DEFER", "reason_code": "NO_POSITIVE_FROZEN_SHRINKAGE"})
        message_id, chosen = min(valid_candidates, key=lambda pair: pair[1]["priority"])
        required_bytes = chosen["message_bytes"] + self.contract.seal_ack_bytes
        required_energy = chosen["energy_upper"] + self.contract.seal_ack_energy
        required_delay = chosen["worst_delay_ms"] + self.contract.seal_ack_delay_ms
        if (
            budget.bytes_remaining < required_bytes
            or budget.energy_remaining < required_energy
            or budget.deadline_slack_ms < required_delay
            or valid_state["elapsed_ms"] + required_delay > self.contract.hard_deadline_ms + FEASIBILITY_TOL
        ):
            return terminal(
                {
                    "action": "SEAL_OR_DEFER",
                    "reason_code": "HARD_BUDGET_PRE_SEND_REFUSAL",
                    "required": {"bytes": required_bytes, "energy": required_energy, "delay_ms": required_delay},
                }
            )
        next_state = copy.deepcopy(valid_state)
        origin = chosen["origin_id"]
        if origin not in next_state["active_origins"]:
            next_state["active_origins"] = [*next_state["active_origins"], origin]
            next_state["arrival_order"] = [*next_state["arrival_order"], origin]
            next_state["delay_buckets"] = [*next_state["delay_buckets"], 0]
            next_state["quantization_levels"] = [*next_state["quantization_levels"], 0]
            next_state["retransmissions"] = [*next_state["retransmissions"], chosen["retransmission_index"]]
            next_state["dropout_mask"] = sorted(set(self.contract.origins).difference(next_state["active_origins"]))
        else:
            index = next_state["active_origins"].index(origin)
            retries = list(next_state["retransmissions"])
            retries[index] = chosen["retransmission_index"]
            next_state["retransmissions"] = retries
        next_state["tier_path"] = [*next_state["tier_path"], chosen["tier"]]
        next_state["stop_reason"] = "REQUESTED"
        next_state["elapsed_ms"] = valid_state["elapsed_ms"] + required_delay
        next_state["parent_state_hash"] = valid_state["state_hash"]
        next_state["request_history"] = [*valid_state["request_history"], message_id]
        cumulative = valid_state["cumulative_reserved_resources"]
        next_state["cumulative_reserved_resources"] = {
            "bytes": cumulative["bytes"] + required_bytes,
            "energy": cumulative["energy"] + required_energy,
            "delay_ms": cumulative["delay_ms"] + required_delay,
        }
        next_state = _seal_transcript_state(next_state, self.contract)
        validate_transcript_state(next_state, self.contract)
        next_budget = ResourceBudget(
            bytes_remaining=budget.bytes_remaining - required_bytes,
            energy_remaining=budget.energy_remaining - required_energy,
            deadline_slack_ms=budget.deadline_slack_ms - required_delay,
        )
        if not self.authority.consume_and_issue(
            current_state_hash=valid_state["state_hash"],
            next_state=next_state,
            next_budget=next_budget,
            reason_code=f"REQUEST:{message_id}",
        ):
            return {"action": "DEFER", "reason_code": "UNKNOWN_NETWORK_STATE"}
        return {
            "action": "REQUEST",
            "message_id": message_id,
            "origin_id": origin,
            "reason_code": "FROZEN_CATALOG_COMPONENT_FIRST_LOOKUP",
            "reserved_max_resources": {"bytes": required_bytes, "energy": required_energy, "delay_ms": required_delay},
            "remaining_after_request": {
                "bytes": next_budget.bytes_remaining,
                "energy": next_budget.energy_remaining,
                "deadline_slack_ms": next_budget.deadline_slack_ms,
            },
            "next_state": copy.deepcopy(next_state),
        }


def fault_response_contract(
    *,
    fault_kind: str,
    independent_component_count: int,
    conflicting_components: int = 0,
    leave_one_component_invariant: bool = False,
) -> dict[str, Any]:
    """Frozen G0 response for fail-silent, numeric, and Byzantine faults.

    Signatures are deliberately not upgraded into honesty.  This helper makes
    the threat boundary executable without assigning identities or reading a
    majority's class decision.
    """

    if fault_kind not in {"fail_silent", "bounded_numeric", "authenticated_byzantine"}:
        raise CIRFContractError("unknown fault kind")
    if isinstance(independent_component_count, bool) or not isinstance(independent_component_count, int) or independent_component_count < 0:
        raise CIRFContractError("independent component count invalid")
    if isinstance(conflicting_components, bool) or not isinstance(conflicting_components, int) or conflicting_components < 0:
        raise CIRFContractError("conflicting component count invalid")
    if fault_kind == "fail_silent":
        action = "REBUILD_ACTIVE_SET_OR_DEFER"
    elif fault_kind == "bounded_numeric":
        action = "USE_FROZEN_INTERVAL_OR_DEFER"
    elif independent_component_count >= 3 and conflicting_components == 0 and leave_one_component_invariant:
        action = "TECHNICAL_BYZANTINE_CONTRACT_SATISFIED"
    else:
        action = "CONFLICT_DEFER_NO_BYZANTINE_CLAIM"
    return {
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "fault_kind": fault_kind,
        "action": action,
        "independent_component_count": independent_component_count,
        "conflicting_components": conflicting_components,
    }


# ---------------------------------------------------------------------------
# Track plane: structural, identity-blind G0 MHT state machine only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TechnicalMHTConfig:
    h_max: int = 32
    fixed_lag_s: float = 120.0
    fixed_lag_opportunities: int = 5
    n_scan: int = 3
    max_association_branches: int = 3
    death_after_observable_misses: int = 4
    archive_ttl_s: float = 24.0 * 60.0 * 60.0
    prune_log_mass_gap: float = 20.0

    def validate(self) -> None:
        if (
            self.h_max != 32
            or self.fixed_lag_opportunities != 5
            or self.n_scan != 3
            or self.max_association_branches != 3
            or self.death_after_observable_misses != 4
        ):
            raise CIRFContractError("TechnicalMHTConfig differs from frozen v3 G0 state machine")
        if abs(_finite(self.fixed_lag_s, "fixed_lag_s") - 120.0) > EPS:
            raise CIRFContractError("fixed lag must be 120 seconds")
        if abs(_finite(self.archive_ttl_s, "archive_ttl_s") - 86400.0) > EPS:
            raise CIRFContractError("archive TTL must be 24 hours")
        if abs(_finite(self.prune_log_mass_gap, "prune_log_mass_gap") - 20.0) > EPS:
            raise CIRFContractError("prune log-mass gap must be 20")


TRACK_EVENT_REQUIRED = frozenset(
    {
        "event_hash",
        "decision",
        "event_time",
        "arrival_time",
        "opportunity_index",
        "possible_independent_components",
        "associations",
    }
)


def _validate_track_event(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(event, name="technical MHT event", allowed=TRACK_EVENT_REQUIRED, required=TRACK_EVENT_REQUIRED)
    raw["event_hash"] = _sha256_hex(raw["event_hash"], "track event hash")
    if raw["decision"] not in {"registered", "unknown", "defer"}:
        raise CIRFContractError("technical MHT event decision invalid")
    raw["event_time"] = _finite(raw["event_time"], "event_time")
    raw["arrival_time"] = _finite(raw["arrival_time"], "arrival_time")
    if raw["arrival_time"] < raw["event_time"]:
        raise CIRFContractError("arrival_time must not precede event_time")
    if isinstance(raw["opportunity_index"], bool) or not isinstance(raw["opportunity_index"], int) or raw["opportunity_index"] < 0:
        raise CIRFContractError("opportunity_index must be a non-negative integer")
    if isinstance(raw["possible_independent_components"], bool) or not isinstance(raw["possible_independent_components"], int) or raw["possible_independent_components"] < 0:
        raise CIRFContractError("possible_independent_components invalid")
    if not isinstance(raw["associations"], list):
        raise CIRFContractError("associations must be a list")
    associations: list[dict[str, Any]] = []
    for candidate in raw["associations"]:
        item = _copy_allowed(
            candidate,
            name="physical association candidate",
            allowed={"track_id", "physical_log_likelihood"},
            required={"track_id", "physical_log_likelihood"},
        )
        item["track_id"] = _nonempty_string(item["track_id"], "association track_id")
        item["physical_log_likelihood"] = _finite(item["physical_log_likelihood"], "physical log likelihood")
        associations.append(item)
    raw["associations"] = associations
    return raw


class TechnicalMHT:
    """G0-only anonymous-track state machine with no identity representation."""

    def __init__(self, config: TechnicalMHTConfig | None = None):
        self.config = config or TechnicalMHTConfig()
        self.config.validate()
        self.tracks: dict[str, dict[str, Any]] = {}
        self.hypotheses: list[dict[str, Any]] = []
        self._included_events: list[str] = []
        self._latest_event_time = -math.inf
        self._latest_opportunity_index = -1
        self._processing_watermark = -math.inf
        self._event_history: dict[str, dict[str, Any]] = {}
        self._n_scan_finalized_events: set[str] = set()
        self._last_revision_hash: str | None = None
        self.revisions: list[dict[str, Any]] = []
        self.audit_only_events: list[str] = []

    def _seal_revision(self, event: Mapping[str, Any], *, oos: bool, status: str) -> dict[str, Any]:
        included = sorted(set(self._included_events))
        revision = {
            "schema_version": TRACK_SCHEMA,
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "event_time": event["event_time"],
            "arrival_time": event["arrival_time"],
            "processing_watermark": self._processing_watermark,
            "parent_revision_hash": self._last_revision_hash,
            "included_event_hashes": included,
            "sealed_at": event["arrival_time"],
            "online_as_of": self._processing_watermark,
            "revision_scope": "ONLINE_AS_OF",
            "oos_within_lag": oos,
            "fixed_lag": {
                "max_opportunities": self.config.fixed_lag_opportunities,
                "max_seconds": self.config.fixed_lag_s,
            },
            "n_scan": self.config.n_scan,
            "n_scan_finalized_event_hashes": sorted(self._n_scan_finalized_events),
            "status": status,
        }
        revision["revision_hash"] = sha256_json(revision)
        self._last_revision_hash = revision["revision_hash"]
        self.revisions.append(revision)
        return revision

    def process_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        valid = _validate_track_event(event)
        # Registered events are a hard no-op: zero birth/association/extension.
        if valid["decision"] == "registered":
            return {
                "evidence_level": TECHNICAL_NO_PERFORMANCE,
                "status": "REGISTERED_EVENT_NO_TRACK_MUTATION",
                "track_count": len(self.tracks),
                "hypothesis_count": len(self.hypotheses),
            }
        n_scan_cutoff = self._latest_opportunity_index - self.config.n_scan
        if valid["event_hash"] in self._n_scan_finalized_events or (
            self._latest_opportunity_index >= 0 and valid["opportunity_index"] <= n_scan_cutoff
        ):
            self.audit_only_events.append(valid["event_hash"])
            return {
                "evidence_level": TECHNICAL_NO_PERFORMANCE,
                "status": "N_SCAN_FINALIZED_AUDIT_ONLY",
                "event_hash": valid["event_hash"],
                "event_decision_rewritten": False,
            }
        candidate_watermark = max(self._processing_watermark, valid["arrival_time"])
        # The lag window is evaluated at the processing/arrival watermark,
        # not just the event's own declared arrival stamp.  This rejects a
        # caller that attempts to inject a past event after later arrivals
        # while presenting an old arrival timestamp.
        lateness_s = candidate_watermark - valid["event_time"]
        opportunity_lag = max(0, self._latest_opportunity_index - valid["opportunity_index"])
        if lateness_s > self.config.fixed_lag_s or opportunity_lag > self.config.fixed_lag_opportunities:
            self.audit_only_events.append(valid["event_hash"])
            return {
                "evidence_level": TECHNICAL_NO_PERFORMANCE,
                "status": "LATE_EVENT_AUDIT_ONLY",
                "event_hash": valid["event_hash"],
                "lag_seconds": lateness_s,
                "lag_opportunities": opportunity_lag,
                "event_decision_rewritten": False,
            }
        if valid["event_hash"] in self._event_history:
            self.audit_only_events.append(valid["event_hash"])
            return {
                "evidence_level": TECHNICAL_NO_PERFORMANCE,
                "status": "DUPLICATE_EVENT_AUDIT_ONLY",
                "event_hash": valid["event_hash"],
                "event_decision_rewritten": False,
            }
        self._processing_watermark = candidate_watermark
        oos = valid["event_time"] < self._latest_event_time or valid["opportunity_index"] < self._latest_opportunity_index
        ranked = sorted(
            valid["associations"],
            key=lambda item: (-item["physical_log_likelihood"], item["track_id"]),
        )[: self.config.max_association_branches]
        branches: list[dict[str, Any]] = []
        for rank, association in enumerate(ranked):
            if association["track_id"] not in self.tracks:
                continue
            track = self.tracks[association["track_id"]]
            # Cold-storage expiry deliberately prevents re-identification.  A
            # later event may birth a new anonymous track but cannot reactivate
            # an archived or scientifically dead one.
            if not track["alive"] or track["archived"]:
                continue
            track["last_event_time"] = max(track["last_event_time"], valid["event_time"])
            track["observable_misses"] = 0
            branches.append(
                {
                    "track_id": association["track_id"],
                    "log_mass": association["physical_log_likelihood"],
                    "event_hash": valid["event_hash"],
                    "branch_rank": rank,
                }
            )
        birth_id = sha256_json({"first_event_hash": valid["event_hash"], "branch_rank": len(branches)})
        self.tracks.setdefault(
            birth_id,
            {
                "track_id": birth_id,
                "first_event_hash": valid["event_hash"],
                "last_event_time": valid["event_time"],
                "observable_misses": 0,
                "alive": True,
                "archived": False,
            },
        )
        branches.append({"track_id": birth_id, "log_mass": 0.0, "event_hash": valid["event_hash"], "branch_rank": len(branches)})
        self.hypotheses.extend(branches)
        self.hypotheses.sort(key=lambda item: (-item["log_mass"], item["event_hash"], item["track_id"], item["branch_rank"]))
        capacity_overflow = len(self.hypotheses) > self.config.h_max
        if self.hypotheses:
            best = self.hypotheses[0]["log_mass"]
            self.hypotheses = [
                item for item in self.hypotheses if item["log_mass"] >= best - self.config.prune_log_mass_gap
            ][: self.config.h_max]
        self._included_events.append(valid["event_hash"])
        self._event_history[valid["event_hash"]] = valid
        self._latest_event_time = max(self._latest_event_time, valid["event_time"])
        self._latest_opportunity_index = max(self._latest_opportunity_index, valid["opportunity_index"])
        n_scan_cutoff = self._latest_opportunity_index - self.config.n_scan
        self._n_scan_finalized_events.update(
            event_hash
            for event_hash, prior in self._event_history.items()
            if prior["opportunity_index"] <= n_scan_cutoff
        )
        revision = self._seal_revision(valid, oos=oos, status="OOS_REVISION" if oos else "ONLINE_UPDATE")
        return {
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "status": "CAPACITY_PRUNED" if capacity_overflow else "TRACK_TECHNICAL_UPDATE",
            "birth_track_id": birth_id,
            "oos_within_lag": oos,
            "revision": revision,
            "event_decision_rewritten": False,
        }

    def visibility_opportunity(self, *, possible_independent_components: int, now: float) -> dict[str, Any]:
        """Count a miss only if at least one independent component could see it."""

        now = _finite(now, "opportunity now")
        if isinstance(possible_independent_components, bool) or not isinstance(possible_independent_components, int) or possible_independent_components < 0:
            raise CIRFContractError("possible_independent_components invalid")
        updated: list[str] = []
        if possible_independent_components >= 1:
            for track in self.tracks.values():
                if not track["alive"] or track["archived"]:
                    continue
                track["observable_misses"] += 1
                if track["observable_misses"] >= self.config.death_after_observable_misses:
                    track["alive"] = False
                updated.append(track["track_id"])
        # Non-opportunities cannot lower existence probability or cause death.
        for track in self.tracks.values():
            if track["alive"] and now - track["last_event_time"] >= self.config.archive_ttl_s:
                track["archived"] = True
        return {
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "status": "OBSERVABLE_MISS_COUNTED" if possible_independent_components >= 1 else "NON_OPPORTUNITY_NO_MISS",
            "updated_tracks": sorted(updated),
            "archived_tracks": sorted(track["track_id"] for track in self.tracks.values() if track["archived"]),
        }

    def lag_final_audit_hash(self) -> str:
        """Canonical, arrival-order-independent audit signature of sealed events."""

        return sha256_json(sorted(self._included_events))

    def online_as_of_revision(self, as_of: float) -> dict[str, Any] | None:
        """Return the newest immutable online revision available at ``as_of``.

        This is intentionally an arrival/processing-time query, not an
        event-time rewrite.  It therefore cannot leak a late OOS correction
        into a past online state.
        """

        as_of = _finite(as_of, "online as_of")
        available = [revision for revision in self.revisions if revision["processing_watermark"] <= as_of]
        if not available:
            return None
        return max(available, key=lambda revision: (revision["processing_watermark"], revision["revision_hash"]))


# ---------------------------------------------------------------------------
# Capacity preflight: exact finite-contract accounting before any run
# ---------------------------------------------------------------------------


def _risk_cells_from_mapping(cells: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, int], list[str]]:
    if not isinstance(cells, Mapping) or not cells:
        raise CIRFContractError("risk_cells must be a non-empty mapping")
    n_min: dict[str, int] = {}
    errors: list[str] = []
    for name in sorted(cells):
        _nonempty_string(name, "risk cell name")
        cell = _copy_allowed(
            cells[name], name=f"risk cell {name}", allowed={"alpha", "delta", "available_blocks"}, required={"alpha", "delta", "available_blocks"}
        )
        needed = n_min_zero_failure(cell["alpha"], cell["delta"])
        if isinstance(cell["available_blocks"], bool) or not isinstance(cell["available_blocks"], int) or cell["available_blocks"] < 0:
            raise CIRFContractError("available_blocks must be a non-negative integer")
        n_min[name] = needed
        if cell["available_blocks"] < needed:
            errors.append(f"INSUFFICIENT_INDEPENDENT_BLOCKS:{name}:{cell['available_blocks']}<{needed}")
    return n_min, errors


def capacity_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate and enforce every frozen G0 capacity quantity exactly.

    The primitive-operation and memory values are exact under the documented
    finite implementation accounting formula below; they are not benchmarked
    performance measurements.
    """

    required = {
        "origin_count",
        "class_plus_unknown_count",
        "context_bucket_count",
        "stochastic_error_source_count",
        "prior_count",
        "scheduler_contract",
        "risk_cells",
    }
    raw = _copy_allowed(config, name="capacity preflight", allowed=required, required=required)
    counts: dict[str, int] = {}
    for field_name in (
        "origin_count",
        "class_plus_unknown_count",
        "context_bucket_count",
        "stochastic_error_source_count",
        "prior_count",
    ):
        value = raw[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CIRFContractError(f"{field_name} must be a positive integer")
        counts[field_name] = value
    scheduler_data = raw["scheduler_contract"]
    if not isinstance(scheduler_data, SchedulerContract):
        raise CIRFContractError("scheduler_contract must be SchedulerContract")
    scheduler_data.validate()
    if counts["origin_count"] != len(scheduler_data.origins):
        raise CIRFContractError("origin_count/scheduler roster mismatch")
    n_atomic = (
        counts["origin_count"]
        * counts["class_plus_unknown_count"]
        * counts["context_bucket_count"]
        * counts["stochastic_error_source_count"]
    )
    breakdown = reachable_transcript_breakdown(scheduler_data)
    n_min, reasons = _risk_cells_from_mapping(raw["risk_cells"])
    qp_solves = sum(
        count * 2 * counts["prior_count"]
        for active, count in breakdown["by_active_count"].items()
        if active >= 2
    )
    # Frozen operation/memory estimator: 96 scalar scheduler operations per
    # prefix; a dual-axis prior QP costs 64+16*M^3 primitive operations; each
    # stored prefix costs 128 bytes and each QP receipt 512 bytes.
    operations = sum(96 * count for count in breakdown["by_active_count"].values())
    operations += sum(
        count * 2 * counts["prior_count"] * (64 + 16 * active**3)
        for active, count in breakdown["by_active_count"].items()
        if active >= 2
    )
    peak_memory = 128 * breakdown["reachable_transcript_count"] + 512 * qp_solves + 160 * n_atomic
    if counts["origin_count"] > MAX_ORIGINS:
        reasons.append("ORIGIN_COUNT_CAP")
    if counts["class_plus_unknown_count"] > MAX_CLASS_PLUS_UNKNOWN:
        reasons.append("CLASS_PLUS_UNKNOWN_CAP")
    if counts["context_bucket_count"] > MAX_CONTEXT_BUCKETS:
        reasons.append("CONTEXT_BUCKET_CAP")
    if counts["stochastic_error_source_count"] > MAX_STOCHASTIC_ERROR_SOURCES:
        reasons.append("STOCHASTIC_ERROR_SOURCE_CAP")
    if counts["prior_count"] > MAX_PRIORS:
        reasons.append("PRIOR_SET_CAP")
    if n_atomic > 7680:
        reasons.append("N_ATOMIC_CAP")
    if breakdown["reachable_transcript_count"] > MAX_TRANSCRIPTS:
        reasons.append("TRANSCRIPT_CAP")
    if qp_solves < 0:  # Defensive invariant kept visible in the artifact.
        reasons.append("QP_COUNT_INVALID")
    if operations > MAX_PRIMITIVE_OPERATIONS:
        reasons.append("PRIMITIVE_OPERATION_CAP")
    if peak_memory > MAX_MEMORY_BYTES:
        reasons.append("PEAK_MEMORY_CAP")
    return {
        "schema_version": SCHEMA,
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "status": "CAPACITY_APPROVED" if not reasons else "CAPACITY_REJECTED_PRE_RUN",
        "refusal_reasons": reasons,
        "N_atomic": n_atomic,
        "n_min_by_risk_cell": n_min,
        "reachable_transcript_count": breakdown["reachable_transcript_count"],
        "reachable_transcript_by_active_count": breakdown["by_active_count"],
        "QP_solve_count": qp_solves,
        "primitive_operation_upper_bound": operations,
        "peak_memory_upper_bound_bytes": peak_memory,
        "scheduler_contract_hash": scheduler_data.contract_hash,
        "scheduler_max_resources": scheduler_data.resource_envelope(),
        "operation_estimator": "96*prefix + 2*prior_count*(64+16*M^3)*multi_origin_prefix",
        "memory_estimator": "128*prefix + 512*QP + 160*N_atomic",
    }


def require_capacity(config: Mapping[str, Any]) -> dict[str, Any]:
    report = capacity_preflight(config)
    if report["status"] != "CAPACITY_APPROVED":
        raise CapacityRejected(";".join(report["refusal_reasons"]))
    return report


G0_MANIFEST_REQUIRED = frozenset(
    {
        "schema_version",
        "evidence_level",
        "technical_synthetic",
        "performance_result",
        "operational_claim",
        "truth_sidecar_opened",
        "cpu_only",
        "output_non_overwriting",
        "artifact_files",
        "external_receipt_filename",
    }
)
MANIFEST_RECEIPT_REQUIRED = frozenset(
    {"schema_version", "evidence_level", "manifest_filename", "manifest_content_sha256", "manifest_self_hash"}
)


def seal_g0_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the manifest's semantic self-hash before writing its byte receipt."""

    raw = _copy_without_hash(payload, name="G0 manifest", hash_field="manifest_self_hash")
    raw = _copy_allowed(raw, name="G0 manifest", allowed=G0_MANIFEST_REQUIRED, required=G0_MANIFEST_REQUIRED)
    if raw["schema_version"] != MANIFEST_SCHEMA or raw["evidence_level"] != TECHNICAL_NO_PERFORMANCE:
        raise CIRFContractError("invalid G0 manifest schema/evidence level")
    for field_name in (
        "technical_synthetic",
        "performance_result",
        "operational_claim",
        "truth_sidecar_opened",
        "cpu_only",
        "output_non_overwriting",
    ):
        if not isinstance(raw[field_name], bool):
            raise CIRFContractError(f"G0 manifest {field_name} must be boolean")
    if (
        raw["technical_synthetic"] is not True
        or raw["performance_result"] is not False
        or raw["operational_claim"] is not False
        or raw["truth_sidecar_opened"] is not False
        or raw["cpu_only"] is not True
        or raw["output_non_overwriting"] is not True
    ):
        raise CIRFContractError("G0 manifest must remain technical synthetic and non-operational")
    artifacts = raw["artifact_files"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise CIRFContractError("G0 manifest requires non-empty artifact file hashes")
    normalized_artifacts: dict[str, str] = {}
    for filename, digest in sorted(artifacts.items()):
        filename = _nonempty_string(filename, "artifact filename")
        if Path(filename).name != filename:
            raise CIRFContractError("artifact filename must not contain a path")
        normalized_artifacts[filename] = _sha256_hex(digest, "artifact content hash")
    raw["artifact_files"] = normalized_artifacts
    receipt_filename = _nonempty_string(raw["external_receipt_filename"], "external receipt filename")
    if Path(receipt_filename).name != receipt_filename or receipt_filename == "g0_manifest.json":
        raise CIRFContractError("G0 manifest external receipt filename invalid")
    raw["external_receipt_filename"] = receipt_filename
    raw["manifest_self_hash"] = sha256_json(raw)
    return raw


def validate_g0_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="G0 manifest",
        allowed=G0_MANIFEST_REQUIRED | {"manifest_self_hash"},
        required=G0_MANIFEST_REQUIRED | {"manifest_self_hash"},
    )
    digest = _sha256_hex(raw.pop("manifest_self_hash"), "manifest_self_hash")
    sealed = seal_g0_manifest(raw)
    if sealed["manifest_self_hash"] != digest:
        raise CIRFContractError("G0 manifest self hash mismatch")
    sealed["manifest_self_hash"] = digest
    return sealed


def seal_manifest_external_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a separate content-hash receipt for an already-written manifest."""

    raw = _copy_without_hash(payload, name="manifest external receipt", hash_field="external_receipt_hash")
    raw = _copy_allowed(
        raw,
        name="manifest external receipt",
        allowed=MANIFEST_RECEIPT_REQUIRED,
        required=MANIFEST_RECEIPT_REQUIRED,
    )
    if raw["schema_version"] != MANIFEST_SCHEMA or raw["evidence_level"] != TECHNICAL_NO_PERFORMANCE:
        raise CIRFContractError("invalid manifest external receipt schema/evidence level")
    if raw["manifest_filename"] != "g0_manifest.json":
        raise CIRFContractError("external receipt must bind g0_manifest.json")
    raw["manifest_content_sha256"] = _sha256_hex(raw["manifest_content_sha256"], "manifest content hash")
    raw["manifest_self_hash"] = _sha256_hex(raw["manifest_self_hash"], "manifest self hash")
    raw["external_receipt_hash"] = sha256_json(raw)
    return raw


def validate_manifest_external_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _copy_allowed(
        payload,
        name="manifest external receipt",
        allowed=MANIFEST_RECEIPT_REQUIRED | {"external_receipt_hash"},
        required=MANIFEST_RECEIPT_REQUIRED | {"external_receipt_hash"},
    )
    digest = _sha256_hex(raw.pop("external_receipt_hash"), "external_receipt_hash")
    sealed = seal_manifest_external_receipt(raw)
    if sealed["external_receipt_hash"] != digest:
        raise CIRFContractError("manifest external receipt hash mismatch")
    sealed["external_receipt_hash"] = digest
    return sealed


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write one immutable JSON artifact and return its SHA256 content hash."""

    target = Path(path)
    if target.exists():
        raise CIRFContractError(f"refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(dict(payload)) + "\n"
    # Keep the artifact bytes canonical across Windows and Linux; otherwise
    # TextIO may turn the terminal LF into CRLF after its SHA256 is recorded.
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise CIRFContractError(f"refusing to overwrite artifact: {target}") from exc
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_g0_manifest(output_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write a self-hashed manifest and a separate immutable content receipt."""

    root = Path(output_dir)
    manifest = seal_g0_manifest(payload)
    manifest_path = root / "g0_manifest.json"
    manifest_content_sha256 = write_json_artifact(manifest_path, manifest)
    receipt = seal_manifest_external_receipt(
        {
            "schema_version": MANIFEST_SCHEMA,
            "evidence_level": TECHNICAL_NO_PERFORMANCE,
            "manifest_filename": manifest_path.name,
            "manifest_content_sha256": manifest_content_sha256,
            "manifest_self_hash": manifest["manifest_self_hash"],
        }
    )
    expected_receipt = root / manifest["external_receipt_filename"]
    write_json_artifact(expected_receipt, receipt)
    return {
        "manifest": manifest,
        "manifest_content_sha256": manifest_content_sha256,
        "external_receipt": receipt,
        "external_receipt_filename": expected_receipt.name,
    }
