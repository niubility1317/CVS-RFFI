from __future__ import annotations

import copy
import json

import pytest

from cvsrffi.phase3_care_poe import (
    EvidenceError,
    FusionConfig,
    SCHEMA,
    authorize_registration,
    build_fresh_k_bridge,
    canonical_json,
    create_anonymous_entity,
    fuse_event,
    run_abcd_matrix,
    score_predictions,
    seal_local_evidence,
    validate_local_evidence,
)


NODES = [f"SAT-{index:02d}" for index in range(1, 6)]


def evidence(
    *,
    event_id: str = "EVT-1",
    node: str = "SAT-01",
    bundle: str = "base",
    probabilities=(0.8, 0.1, 0.1),
    decision: str = "registered",
    label: str | None = "a",
    group: str = "G1",
    delay: float = 1.0,
    q: float = 0.9,
):
    return seal_local_evidence(
        {
            "schema_version": SCHEMA,
            "linkage_mode": "verified_physical",
            "emission_event_id": event_id,
            "satellite_reception_id": f"{event_id}-{node}",
            "node_id": node,
            "base_manifest_id": "M1",
            "bundle_id": bundle,
            "class_handles": ["a", "b"],
            "p_local": list(probabilities),
            "q": q,
            "correlation_group_id": group,
            "delay_ms": delay,
            "deadline_ms": 10.0,
            "local_decision": decision,
            "local_label": label,
            "reason_code": "LOCAL",
            "sealed_at_ms": 1.0,
        }
    )


def event_bundle(bundle: str, probabilities=(0.8, 0.1, 0.1), decision="registered", label="a"):
    rows = []
    for index, node in enumerate(NODES):
        rows.append(
            evidence(
                node=node,
                bundle=bundle,
                probabilities=probabilities,
                decision=decision,
                label=label,
                group=f"G{min(index, 2) + 1}",
            )
        )
    return rows


def test_forbidden_truth_and_role_are_rejected():
    raw = evidence()
    raw.pop("evidence_hash")
    raw["true_label"] = "a"
    with pytest.raises(EvidenceError, match="forbidden"):
        seal_local_evidence(raw)


def test_linkage_contract_and_hash_tamper_fail_closed():
    raw = evidence()
    tampered = copy.deepcopy(raw)
    tampered["q"] = 0.1
    with pytest.raises(EvidenceError, match="hash"):
        validate_local_evidence(tampered)
    unsigned = copy.deepcopy(raw)
    unsigned.pop("evidence_hash")
    unsigned["proxy_group_id"] = "P1"
    with pytest.raises(EvidenceError, match="only emission"):
        seal_local_evidence(unsigned)


def test_single_node_is_exact_identity_and_one_shot():
    row = evidence()
    result = fuse_event([row], FusionConfig())
    assert result["p_fused"] == row["p_local"]
    assert result["decision"] == row["local_decision"]
    assert result["label"] == row["local_label"]
    assert result["shot_count"] == 1


def test_same_correlation_copy_does_not_add_strength():
    first = evidence(node="SAT-01", group="G1", probabilities=(0.15, 0.10, 0.75), decision="unknown", label=None)
    second = evidence(node="SAT-02", group="G1", probabilities=(0.15, 0.10, 0.75), decision="unknown", label=None)
    third = evidence(node="SAT-03", group="G1", probabilities=(0.80, 0.10, 0.10), decision="registered", label="a")
    two = fuse_event([first, second], FusionConfig())
    three = fuse_event([first, second, third], FusionConfig())
    assert two["p_fused"] == pytest.approx(three["p_fused"])
    assert two["correlation_group_count"] == three["correlation_group_count"] == 1
    assert two["decision"] != "unknown" and three["decision"] != "unknown"


def test_independent_groups_can_support_unknown_consensus():
    rows = [
        evidence(node="SAT-01", group="G1", probabilities=(0.05, 0.05, 0.90), decision="unknown", label=None, q=1.0),
        evidence(node="SAT-02", group="G2", probabilities=(0.05, 0.05, 0.90), decision="unknown", label=None, q=1.0),
    ]
    result = fuse_event(rows, FusionConfig(tau_group_quality=1.0))
    assert result["decision"] == "unknown"
    assert result["correlation_group_count"] == 2


def test_late_missing_and_integrity_failure_defer():
    late = evidence(delay=20.0)
    assert fuse_event([late], FusionConfig())["reason_code"] == "NO_VALID_RECEPTION"
    mixed = [evidence(event_id="E1"), evidence(event_id="E2", node="SAT-02")]
    assert fuse_event(mixed, FusionConfig())["reason_code"] == "EVENT_INTEGRITY_FAILURE"
    assert fuse_event([], FusionConfig())["decision"] == "defer"


def test_node_and_class_permutation_are_semantically_invariant():
    rows = [
        evidence(node="SAT-01", group="G1", probabilities=(0.8, 0.1, 0.1)),
        evidence(node="SAT-02", group="G2", probabilities=(0.7, 0.2, 0.1)),
    ]
    forward = fuse_event(rows, FusionConfig())
    reverse = fuse_event(list(reversed(rows)), FusionConfig())
    assert forward["decision"] == reverse["decision"]
    assert forward["label"] == reverse["label"]
    assert forward["p_fused"] == pytest.approx(reverse["p_fused"])
    swapped = []
    for row in rows:
        unsigned = copy.deepcopy(row)
        unsigned.pop("evidence_hash")
        unsigned["class_handles"] = ["b", "a"]
        unsigned["p_local"] = [row["p_local"][1], row["p_local"][0], row["p_local"][2]]
        unsigned["local_label"] = "a"
        swapped.append(seal_local_evidence(unsigned))
    permuted = fuse_event(swapped, FusionConfig())
    assert permuted["decision"] == forward["decision"]
    assert permuted["label"] == forward["label"]
    assert permuted["p_fused"] == pytest.approx([forward["p_fused"][1], forward["p_fused"][0], forward["p_fused"][2]])


def test_abcd_same_input_all_budgets_and_n1_parity():
    base = event_bundle("base")
    new = event_bundle("new", probabilities=(0.85, 0.08, 0.07))
    rows = run_abcd_matrix(base, new, FusionConfig(), node_roster=NODES)
    assert len(rows) == 20
    assert {row["node_budget"] for row in rows} == {1, 2, 3, 4, 5}
    lookup = {(row["arm"], row["node_budget"]): row for row in rows}
    for left, right in (("A", "C"), ("B", "D")):
        assert lookup[(left, 1)]["p_fused"] == lookup[(right, 1)]["p_fused"]
        assert lookup[(left, 1)]["decision"] == lookup[(right, 1)]["decision"]
    assert all(row["shot_count"] == 1 for row in rows)


def test_abcd_rejects_different_physical_reception_binding():
    base = event_bundle("base")
    new = event_bundle("new")
    unsigned = copy.deepcopy(new[0])
    unsigned.pop("evidence_hash")
    unsigned["satellite_reception_id"] = "DIFFERENT-RECEPTION"
    new[0] = seal_local_evidence(unsigned)
    with pytest.raises(EvidenceError, match="physical reception binding"):
        run_abcd_matrix(base, new, FusionConfig(), node_roster=NODES)


def test_scorer_counts_known_defer_as_error_and_unknown_defer_unresolved():
    predictions = [
        {"event_key": "K1", "arm": "A", "node_budget": 1, "decision": "defer", "label": None},
        {"event_key": "U1", "arm": "A", "node_budget": 1, "decision": "defer", "label": None},
        {"event_key": "U2", "arm": "A", "node_budget": 1, "decision": "unknown", "label": None},
    ]
    truth = [
        {"event_key": "K1", "role": "registered", "true_label": "a"},
        {"event_key": "U1", "role": "unknown", "true_label": "u"},
        {"event_key": "U2", "role": "unknown", "true_label": "u"},
    ]
    row = score_predictions(predictions, truth)["rows"]["A:N1"]
    assert row["known_accuracy"] == 0.0
    assert row["safe_reject_rate"] == 0.5
    assert row["unknown_defer_rate"] == 0.5


def test_scorer_rejects_duplicate_prediction_and_illegal_role():
    prediction = {"event_key": "E1", "arm": "A", "node_budget": 1, "decision": "defer", "label": None}
    truth = [{"event_key": "E1", "role": "registered", "true_label": "a"}]
    with pytest.raises(ValueError, match="duplicate prediction"):
        score_predictions([prediction, dict(prediction)], truth)
    with pytest.raises(ValueError, match="invalid truth role"):
        score_predictions([prediction], [{"event_key": "E1", "role": "oracle_proxy", "true_label": "a"}])


def test_truth_sidecar_never_changes_prediction_bytes():
    rows = run_abcd_matrix(event_bundle("base"), event_bundle("new"), FusionConfig(), node_roster=NODES)
    before = canonical_json(rows)
    score_predictions(rows, [{"event_key": "EVT-1", "role": "registered", "true_label": "a"}])
    score_predictions(rows, [{"event_key": "EVT-1", "role": "unknown", "true_label": "x"}])
    assert canonical_json(rows) == before


def test_anonymous_cannot_self_authorize_and_credential_fails_closed():
    prediction = {"event_key": "U1", "decision": "unknown", "evidence_hashes": ["a"]}
    anonymous = create_anonymous_entity(prediction)
    assert anonymous["semantic_identity"] is None
    credential = {
        "anonymous_entity_id": anonymous["anonymous_entity_id"],
        "candidate_identity": "new-tx",
        "evidence_sources": ["s1", "s2"],
        "independent_sources": True,
        "conflicts": [],
        "confidence": 0.99,
        "valid_until_ms": 100.0,
        "registration_authorized": True,
        "signature": "external",
    }
    expired = dict(credential)
    with pytest.raises(ValueError, match="expired"):
        authorize_registration(anonymous, expired, now_ms=101.0)
    conflict = dict(credential, conflicts=["identity collision"], valid_until_ms=1000.0)
    with pytest.raises(ValueError, match="conflicts"):
        authorize_registration(anonymous, conflict, now_ms=1.0)


def test_fresh_k_requires_new_unique_events_and_never_reuses_unknown():
    anonymous = create_anonymous_entity({"event_key": "U1", "decision": "unknown", "evidence_hashes": []})
    credential = {
        "anonymous_entity_id": anonymous["anonymous_entity_id"],
        "candidate_identity": "new-tx",
        "evidence_sources": ["s1", "s2"],
        "independent_sources": True,
        "conflicts": [],
        "confidence": 0.99,
        "valid_until_ms": 1000.0,
        "registration_authorized": True,
        "signature": "external",
    }
    authorization = authorize_registration(anonymous, credential, now_ms=1.0)
    support = [
        {
            "linkage_mode": "verified_physical",
            "emission_event_id": f"F{index}",
            "physical_sample_id": f"P{index}",
            "candidate_identity": "new-tx",
        }
        for index in range(5)
    ]
    receipt = build_fresh_k_bridge(authorization, support, k=5)
    assert receipt["state"] == "FRESH_K_READY_FOR_STAGE2_C"
    reused = copy.deepcopy(support)
    reused[0]["emission_event_id"] = "U1"
    with pytest.raises(ValueError, match="historical unknown"):
        build_fresh_k_bridge(authorization, reused, k=5)
    duplicate = copy.deepcopy(support)
    duplicate[1]["physical_sample_id"] = duplicate[0]["physical_sample_id"]
    with pytest.raises(ValueError, match="unique"):
        build_fresh_k_bridge(authorization, duplicate, k=5)
