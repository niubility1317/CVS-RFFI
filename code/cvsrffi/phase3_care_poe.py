"""Truth-free Phase3 CARE-PoE technical closure.

This module deliberately contains no dataset loader. Predictor inputs are sealed
local evidence records; truth is accepted only by :func:`score_predictions`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "cvs.phase3.local_evidence.v2"
FORBIDDEN_PREDICTOR_FIELDS = {
    "role",
    "true_label",
    "registration_authorized",
    "credential",
}
ALLOWED_DECISIONS = {"registered", "unknown", "defer"}


class EvidenceError(ValueError):
    """Raised when a local evidence artifact violates the frozen contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def event_key(record: Mapping[str, Any]) -> str:
    mode = record.get("linkage_mode")
    if mode == "verified_physical":
        return str(record.get("emission_event_id", ""))
    if mode == "proxy_unverified":
        return str(record.get("proxy_group_id", ""))
    raise EvidenceError("linkage_mode must be verified_physical or proxy_unverified")


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{name} must be finite")
    return result


def _validate_payload(payload: Mapping[str, Any], *, require_hash: bool) -> dict[str, Any]:
    raw = dict(payload)
    if require_hash:
        digest = raw.get("evidence_hash")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EvidenceError("missing or malformed evidence_hash")
        unsigned_received = dict(raw)
        unsigned_received.pop("evidence_hash", None)
        if sha256_json(unsigned_received) != digest:
            raise EvidenceError("evidence_hash mismatch")
    forbidden = FORBIDDEN_PREDICTOR_FIELDS.intersection(raw)
    if forbidden:
        raise EvidenceError(f"forbidden predictor fields: {sorted(forbidden)}")
    required = {
        "schema_version",
        "linkage_mode",
        "satellite_reception_id",
        "node_id",
        "base_manifest_id",
        "bundle_id",
        "class_handles",
        "p_local",
        "q",
        "correlation_group_id",
        "delay_ms",
        "deadline_ms",
        "local_decision",
        "reason_code",
        "sealed_at_ms",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise EvidenceError(f"missing local evidence fields: {missing}")
    if raw["schema_version"] != SCHEMA:
        raise EvidenceError(f"unsupported schema_version={raw['schema_version']!r}")
    mode = raw["linkage_mode"]
    if mode == "verified_physical":
        if not raw.get("emission_event_id") or raw.get("proxy_group_id"):
            raise EvidenceError("verified_physical requires only emission_event_id")
    elif mode == "proxy_unverified":
        if not raw.get("proxy_group_id") or raw.get("emission_event_id"):
            raise EvidenceError("proxy_unverified requires only proxy_group_id")
    else:
        raise EvidenceError("invalid linkage_mode")
    for name in (
        "satellite_reception_id",
        "node_id",
        "base_manifest_id",
        "bundle_id",
        "correlation_group_id",
        "reason_code",
    ):
        if not isinstance(raw[name], str) or not raw[name]:
            raise EvidenceError(f"{name} must be a non-empty string")
    handles = raw["class_handles"]
    if not isinstance(handles, list) or not handles or len(handles) != len(set(handles)):
        raise EvidenceError("class_handles must be a non-empty unique list")
    if any(not isinstance(handle, str) or not handle for handle in handles):
        raise EvidenceError("class_handles entries must be non-empty strings")
    probabilities = raw["p_local"]
    if not isinstance(probabilities, list) or len(probabilities) != len(handles) + 1:
        raise EvidenceError("p_local must contain C registered values plus one unknown value")
    probabilities = [_finite_float(value, "p_local") for value in probabilities]
    if any(value < 0.0 for value in probabilities):
        raise EvidenceError("p_local values must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise EvidenceError("p_local must have positive mass")
    raw["p_local"] = [value / total for value in probabilities]
    raw["q"] = _finite_float(raw["q"], "q")
    if not 0.0 <= raw["q"] <= 1.0:
        raise EvidenceError("q must be in [0,1]")
    for name in ("delay_ms", "deadline_ms", "sealed_at_ms"):
        raw[name] = _finite_float(raw[name], name)
        if raw[name] < 0.0:
            raise EvidenceError(f"{name} must be non-negative")
    if raw["local_decision"] not in ALLOWED_DECISIONS:
        raise EvidenceError("invalid local_decision")
    label = raw.get("local_label")
    if raw["local_decision"] == "registered":
        if label not in handles:
            raise EvidenceError("registered local_decision requires a class handle")
    elif label is not None:
        raise EvidenceError("unknown/defer local_decision must not carry local_label")
    return raw


def seal_local_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, normalize, and seal one truth-free local evidence record."""

    unsigned = dict(payload)
    unsigned.pop("evidence_hash", None)
    normalized = _validate_payload(unsigned, require_hash=False)
    normalized["evidence_hash"] = sha256_json(normalized)
    return normalized


def validate_local_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_payload(payload, require_hash=True)


@dataclass(frozen=True)
class FusionConfig:
    lambda_delay: float = 0.002
    tau_unknown_max: float = 0.35
    tau_registered_margin: float = 0.20
    tau_conflict: float = 0.25
    tau_reject: float = 0.65
    tau_group_quality: float = 1.0
    epsilon: float = 1e-8

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"invalid FusionConfig {name}")
        if self.tau_unknown_max > 1.0 or self.tau_reject > 1.0:
            raise ValueError("probability thresholds must be <=1")


def _defer_result(key: str, reason: str, *, node_budget: int | None = None) -> dict[str, Any]:
    result = {
        "event_key": key,
        "decision": "defer",
        "label": None,
        "reason_code": reason,
        "shot_count": 1,
        "p_fused": None,
        "valid_reception_count": 0,
        "correlation_group_count": 0,
    }
    if node_budget is not None:
        result["node_budget"] = node_budget
    return result


def _normalize(values: Sequence[float], epsilon: float) -> list[float]:
    clipped = [max(float(value), epsilon) for value in values]
    total = sum(clipped)
    return [value / total for value in clipped]


def _js_divergence(left: Sequence[float], right: Sequence[float], epsilon: float) -> float:
    p = _normalize(left, epsilon)
    q = _normalize(right, epsilon)
    middle = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b))

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def fuse_event(
    records: Sequence[Mapping[str, Any]],
    config: FusionConfig,
    *,
    node_priors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fuse one event without opening a truth sidecar."""

    config.validate()
    if not records:
        return _defer_result("", "NO_RECEPTION")
    try:
        valid = [validate_local_evidence(record) for record in records]
        keys = {event_key(record) for record in valid}
        if len(keys) != 1:
            raise EvidenceError("mixed event keys")
        key = next(iter(keys))
        invariant_fields = ("linkage_mode", "base_manifest_id", "bundle_id")
        for name in invariant_fields:
            if len({record[name] for record in valid}) != 1:
                raise EvidenceError(f"mixed {name}")
        class_orders = {tuple(record["class_handles"]) for record in valid}
        if len(class_orders) != 1:
            raise EvidenceError("mixed class_handles")
        reception_ids = [record["satellite_reception_id"] for record in valid]
        node_ids = [record["node_id"] for record in valid]
        if len(reception_ids) != len(set(reception_ids)) or len(node_ids) != len(set(node_ids)):
            raise EvidenceError("duplicate reception or node identity")
    except EvidenceError:
        key = ""
        try:
            key = event_key(records[0])
        except (EvidenceError, KeyError, TypeError):
            pass
        return _defer_result(key, "EVENT_INTEGRITY_FAILURE")

    on_time = [record for record in valid if record["delay_ms"] <= record["deadline_ms"]]
    if not on_time:
        return _defer_result(key, "NO_VALID_RECEPTION")
    handles = list(on_time[0]["class_handles"])
    if len(on_time) == 1:
        record = on_time[0]
        return {
            "event_key": key,
            "decision": record["local_decision"],
            "label": record.get("local_label"),
            "reason_code": record["reason_code"],
            "shot_count": 1,
            "class_handles": handles,
            "p_fused": list(record["p_local"]),
            "valid_reception_count": 1,
            "correlation_group_count": 1,
            "evidence_hashes": [record["evidence_hash"]],
        }

    priors = dict(node_priors or {})
    weighted: list[tuple[dict[str, Any], float]] = []
    for record in on_time:
        prior = _finite_float(priors.get(record["node_id"], 1.0), "node_prior")
        if prior < 0.0:
            return _defer_result(key, "EVENT_INTEGRITY_FAILURE")
        reliability = min(1.0, max(0.0, record["q"] * prior * math.exp(-config.lambda_delay * record["delay_ms"])))
        if reliability > 0.0:
            weighted.append((record, reliability))
    if not weighted:
        return _defer_result(key, "NO_VALID_RECEPTION")

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for record, reliability in weighted:
        grouped[record["correlation_group_id"]].append((record, reliability))
    group_probabilities: list[list[float]] = []
    gammas: list[float] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        denominator = sum(reliability for _, reliability in members)
        mixture = [0.0] * (len(handles) + 1)
        for record, reliability in members:
            alpha = reliability / denominator
            for index, probability in enumerate(record["p_local"]):
                mixture[index] += alpha * probability
        group_probabilities.append(_normalize(mixture, config.epsilon))
        gammas.append(max(reliability for _, reliability in members))

    logits = [-math.log(len(handles) + 1)] * (len(handles) + 1)
    for probabilities, gamma in zip(group_probabilities, gammas):
        for index, probability in enumerate(_normalize(probabilities, config.epsilon)):
            logits[index] += gamma * math.log(max(probability, config.epsilon))
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    fused = _normalize(exp_values, config.epsilon)
    unknown_probability = fused[-1]
    top_index = max(range(len(handles)), key=lambda index: fused[index])
    competitors = [fused[-1]] + [value for index, value in enumerate(fused[:-1]) if index != top_index]
    margin = fused[top_index] - max(competitors)
    max_conflict = max(_js_divergence(probabilities, fused, config.epsilon) for probabilities in group_probabilities)

    if (
        unknown_probability >= config.tau_reject
        and len(group_probabilities) >= 2
        and sum(gammas) >= config.tau_group_quality
    ):
        decision, label, reason = "unknown", None, "CARE_UNKNOWN_CONSENSUS"
    elif (
        unknown_probability < config.tau_unknown_max
        and margin >= config.tau_registered_margin
        and max_conflict <= config.tau_conflict
    ):
        decision, label, reason = "registered", handles[top_index], "CARE_REGISTERED_CONSENSUS"
    else:
        decision, label, reason = "defer", None, "CARE_INSUFFICIENT_CONSENSUS"
    return {
        "event_key": key,
        "decision": decision,
        "label": label,
        "reason_code": reason,
        "shot_count": 1,
        "class_handles": handles,
        "p_fused": fused,
        "valid_reception_count": len(weighted),
        "correlation_group_count": len(group_probabilities),
        "group_quality_sum": sum(gammas),
        "max_group_js": max_conflict,
        "evidence_hashes": sorted(record["evidence_hash"] for record, _ in weighted),
    }


def _group_events(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[event_key(record)].append(record)
    return dict(grouped)


def run_abcd_matrix(
    base_records: Sequence[Mapping[str, Any]],
    new_records: Sequence[Mapping[str, Any]],
    config: FusionConfig,
    *,
    node_roster: Sequence[str],
    budgets: Sequence[int] = (1, 2, 3, 4, 5),
    node_priors: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Run the same-input A/B/C/D intervention matrix."""

    if len(node_roster) < max(budgets) or len(set(node_roster)) != len(node_roster):
        raise ValueError("node_roster must uniquely cover every budget")
    base_events = _group_events(base_records)
    new_events = _group_events(new_records)
    if set(base_events) != set(new_events):
        raise EvidenceError("base/new event sets differ")
    rows: list[dict[str, Any]] = []
    for key in sorted(base_events):
        if {record["base_manifest_id"] for record in base_events[key] + new_events[key]}.__len__() != 1:
            raise EvidenceError("base/new base_manifest_id differs")
        for budget in budgets:
            selected = set(node_roster[:budget])
            base_selected = [record for record in base_events[key] if record["node_id"] in selected]
            new_selected = [record for record in new_events[key] if record["node_id"] in selected]
            leader = node_roster[0]
            arms = {
                "A": [record for record in base_selected if record["node_id"] == leader],
                "B": [record for record in new_selected if record["node_id"] == leader],
                "C": base_selected,
                "D": new_selected,
            }
            for arm, arm_records in arms.items():
                result = fuse_event(arm_records, config, node_priors=node_priors)
                result.update({"arm": arm, "node_budget": int(budget), "event_key": key})
                rows.append(result)
    return rows


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    truth_sidecar: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score sealed predictions; this is the only truth-reading function."""

    truth: dict[str, Mapping[str, Any]] = {}
    for row in truth_sidecar:
        key = str(row.get("event_key", ""))
        if not key or key in truth or "true_label" not in row or "role" not in row:
            raise ValueError("invalid truth sidecar")
        truth[key] = row
    buckets: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        key = str(prediction.get("event_key", ""))
        if key not in truth:
            raise ValueError(f"missing truth for event {key!r}")
        buckets[(str(prediction["arm"]), int(prediction["node_budget"]))].append(prediction)
    metrics: dict[str, Any] = {}
    for (arm, budget), rows in sorted(buckets.items()):
        known_total = known_correct = 0
        unknown_total = false_accept = safe_reject = unknown_defer = 0
        class_totals: dict[str, int] = defaultdict(int)
        class_correct: dict[str, int] = defaultdict(int)
        for prediction in rows:
            sidecar = truth[prediction["event_key"]]
            if sidecar["role"] == "unknown":
                unknown_total += 1
                if prediction["decision"] == "registered":
                    false_accept += 1
                elif prediction["decision"] == "unknown":
                    safe_reject += 1
                else:
                    unknown_defer += 1
            else:
                label = str(sidecar["true_label"])
                known_total += 1
                class_totals[label] += 1
                correct = prediction["decision"] == "registered" and prediction.get("label") == label
                if correct:
                    known_correct += 1
                    class_correct[label] += 1
        class_accuracy = {
            label: class_correct[label] / total for label, total in sorted(class_totals.items())
        }
        metrics[f"{arm}:N{budget}"] = {
            "known_total": known_total,
            "known_accuracy": known_correct / known_total if known_total else None,
            "min_class_old_accuracy": min(class_accuracy.values()) if class_accuracy else None,
            "per_class_old_accuracy": class_accuracy,
            "unknown_total": unknown_total,
            "unknown_far": false_accept / unknown_total if unknown_total else None,
            "safe_reject_rate": safe_reject / unknown_total if unknown_total else None,
            "unknown_defer_rate": unknown_defer / unknown_total if unknown_total else None,
        }
    return {"evidence_level": "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT", "rows": metrics}


def create_anonymous_entity(prediction: Mapping[str, Any]) -> dict[str, Any]:
    if prediction.get("decision") != "unknown":
        raise ValueError("only an unknown decision can create an anonymous entity")
    source = {
        "event_key": prediction.get("event_key"),
        "evidence_hashes": sorted(prediction.get("evidence_hashes", [])),
    }
    return {
        "state": "ANONYMOUS",
        "anonymous_entity_id": f"anon-{sha256_json(source)[:24]}",
        "historical_event_ids": [str(prediction.get("event_key"))],
        "semantic_identity": None,
    }


def authorize_registration(
    anonymous: Mapping[str, Any],
    credential: Mapping[str, Any],
    *,
    now_ms: float,
    min_confidence: float = 0.90,
) -> dict[str, Any]:
    required = {
        "anonymous_entity_id",
        "candidate_identity",
        "evidence_sources",
        "independent_sources",
        "conflicts",
        "confidence",
        "valid_until_ms",
        "registration_authorized",
        "signature",
    }
    if required.difference(credential):
        raise ValueError("credential is incomplete")
    if anonymous.get("state") != "ANONYMOUS" or credential["anonymous_entity_id"] != anonymous.get("anonymous_entity_id"):
        raise ValueError("credential binding mismatch")
    sources = credential["evidence_sources"]
    if not isinstance(sources, list) or len(set(sources)) < 2 or not credential["independent_sources"]:
        raise ValueError("at least two independent credential sources are required")
    if credential["conflicts"]:
        raise ValueError("credential conflicts must be empty")
    if _finite_float(credential["confidence"], "confidence") < min_confidence:
        raise ValueError("credential confidence is insufficient")
    if _finite_float(credential["valid_until_ms"], "valid_until_ms") < float(now_ms):
        raise ValueError("credential has expired")
    if credential["registration_authorized"] is not True or not credential["signature"]:
        raise ValueError("registration is not externally authorized")
    receipt = {
        "state": "REGISTRATION_AUTHORIZED",
        "anonymous_entity_id": anonymous["anonymous_entity_id"],
        "candidate_identity": str(credential["candidate_identity"]),
        "credential_hash": sha256_json(dict(credential)),
        "historical_event_ids": list(anonymous.get("historical_event_ids", [])),
    }
    receipt["authorization_receipt_hash"] = sha256_json(receipt)
    return receipt


def build_fresh_k_bridge(
    authorization: Mapping[str, Any],
    support_records: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    if authorization.get("state") != "REGISTRATION_AUTHORIZED":
        raise ValueError("registration authorization is required")
    if len(support_records) != int(k) or k <= 0:
        raise ValueError("fresh support must contain exactly K records")
    event_ids: list[str] = []
    physical_ids: list[str] = []
    for record in support_records:
        if record.get("linkage_mode") != "verified_physical":
            raise ValueError("fresh support requires verified_physical linkage")
        if record.get("candidate_identity") != authorization.get("candidate_identity"):
            raise ValueError("fresh support identity mismatch")
        event_ids.append(str(record.get("emission_event_id", "")))
        physical_ids.append(str(record.get("physical_sample_id", "")))
    if "" in event_ids or "" in physical_ids or len(set(event_ids)) != k or len(set(physical_ids)) != k:
        raise ValueError("fresh support events and physical samples must be unique")
    historical = set(map(str, authorization.get("historical_event_ids", [])))
    if historical.intersection(event_ids):
        raise ValueError("historical unknown evidence cannot become support")
    split_payload = {
        "candidate_identity": authorization["candidate_identity"],
        "authorization_receipt_hash": authorization["authorization_receipt_hash"],
        "k": int(k),
        "emission_event_ids": sorted(event_ids),
        "physical_sample_ids": sorted(physical_ids),
    }
    return {
        "state": "FRESH_K_READY_FOR_STAGE2_C",
        "split_id": f"phase3-fresh-k-{sha256_json(split_payload)[:24]}",
        **split_payload,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
