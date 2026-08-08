from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cvsrffi.phase3_care_poe import (
    EvidenceError,
    FusionConfig,
    PHYSICAL_BINDING_SCHEMA,
    SCHEMA,
    authorize_registration,
    bind_verified_physical_evidence,
    build_fresh_k_bridge,
    canonical_json,
    create_anonymous_entity,
    fuse_event,
    physical_binding_root,
    read_jsonl,
    run_abcd_matrix,
    score_predictions,
    seal_local_evidence,
    seal_physical_binding,
    validate_local_evidence,
    write_jsonl,
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
    binding = seal_physical_binding(
        {
            "schema_version": PHYSICAL_BINDING_SCHEMA,
            "binding_receipt_id": "TEST-BINDING-RECEIPT",
            "base_manifest_id": "M1",
            "source_satellite_reception_id": f"source-{event_id}-{node}",
            "emission_event_id": event_id,
            "satellite_reception_id": f"{event_id}-{node}",
            "node_id": node,
            "correlation_group_id": group,
            "delay_ms": delay,
            "deadline_ms": 10.0,
            "sealed_at_ms": 1.0,
            "created_before_label_access": True,
        }
    )
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
            "z_id": [0.1, 0.2],
            "z_dom": [0.3, 0.4],
            "d_class": [0.2, 0.8],
            "e_unknown": float(probabilities[-1]),
            "p_local": list(probabilities),
            "q": q,
            "correlation_group_id": group,
            "delay_ms": delay,
            "deadline_ms": 10.0,
            "local_decision": decision,
            "local_label": label,
            "reason_code": "LOCAL",
            "sealed_at_ms": 1.0,
            "physical_binding_receipt_id": binding["binding_receipt_id"],
            "physical_binding_hash": binding["binding_hash"],
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


def bindings_for(records):
    return [
        seal_physical_binding(
            {
                "schema_version": PHYSICAL_BINDING_SCHEMA,
                "binding_receipt_id": record["physical_binding_receipt_id"],
                "base_manifest_id": record["base_manifest_id"],
                "source_satellite_reception_id": f"source-{record['emission_event_id']}-{record['node_id']}",
                "emission_event_id": record["emission_event_id"],
                "satellite_reception_id": record["satellite_reception_id"],
                "node_id": record["node_id"],
                "correlation_group_id": record["correlation_group_id"],
                "delay_ms": record["delay_ms"],
                "deadline_ms": record["deadline_ms"],
                "sealed_at_ms": record["sealed_at_ms"],
                "created_before_label_access": True,
            }
        )
        for record in records
    ]


def proxy_evidence(*, index: int, node: str, probabilities=(0.05, 0.05, 0.90)):
    return seal_local_evidence(
        {
            "schema_version": SCHEMA,
            "linkage_mode": "proxy_unverified",
            "proxy_group_id": f"P-{index}",
            "satellite_reception_id": f"proxy-reception-{index}",
            "node_id": node,
            "base_manifest_id": "M1",
            "bundle_id": "base",
            "class_handles": ["a", "b"],
            "p_local": list(probabilities),
            "q": 1.0,
            "correlation_group_id": node,
            "delay_ms": 0.0,
            "deadline_ms": 10.0,
            "local_decision": "unknown",
            "local_label": None,
            "reason_code": "LOCAL_UNKNOWN",
            "sealed_at_ms": 0.0,
            "z_id": [0.1, 0.2],
            "z_dom": [0.3, 0.4],
            "d_class": [0.8, 0.9],
            "e_unknown": 0.9,
        }
    )


def physical_binding(record, *, event_id: str = "EVT-CAPTURE-1", reception_id: str | None = None, group: str | None = None):
    return seal_physical_binding(
        {
            "schema_version": PHYSICAL_BINDING_SCHEMA,
            "binding_receipt_id": "CAPTURE-RECEIPT-1",
            "base_manifest_id": record["base_manifest_id"],
            "source_satellite_reception_id": record["satellite_reception_id"],
            "emission_event_id": event_id,
            "satellite_reception_id": reception_id or f"{event_id}-{record['node_id']}",
            "node_id": record["node_id"],
            "correlation_group_id": group or f"RF-{record['node_id']}",
            "delay_ms": 1.0,
            "deadline_ms": 10.0,
            "sealed_at_ms": 2.0,
            "created_before_label_access": True,
        }
    )


@pytest.mark.parametrize("forbidden", ["true_label", "role", "raw_iq", "source_cache", "member_ids"])
def test_forbidden_truth_and_role_are_rejected(forbidden):
    raw = evidence()
    raw.pop("evidence_hash")
    raw[forbidden] = "forbidden"
    with pytest.raises(EvidenceError, match="forbidden|unexpected"):
        seal_local_evidence(raw)


def test_local_evidence_requires_open_world_vectors():
    raw = evidence()
    raw.pop("evidence_hash")
    raw.pop("z_dom")
    with pytest.raises(EvidenceError, match="missing local evidence fields"):
        seal_local_evidence(raw)
    malformed = evidence()
    malformed.pop("evidence_hash")
    malformed["d_class"] = [0.1]
    with pytest.raises(EvidenceError, match="one distance per registered class"):
        seal_local_evidence(malformed)


def test_verified_physical_cannot_bypass_binding_sidecar_or_schema_v3():
    row = evidence()
    assert fuse_event([row], FusionConfig())["reason_code"] == "EVENT_INTEGRITY_FAILURE"
    forged = dict(row)
    forged.pop("evidence_hash")
    forged["physical_binding_hash"] = "0" * 64
    forged = seal_local_evidence(forged)
    assert fuse_event(
        [forged],
        FusionConfig(),
        physical_bindings=bindings_for([row]),
    )["reason_code"] == "EVENT_INTEGRITY_FAILURE"
    legacy = dict(row)
    legacy.pop("evidence_hash")
    legacy["schema_version"] = "cvs.phase3.local_evidence.v2"
    with pytest.raises(EvidenceError, match="re-emit as v3"):
        seal_local_evidence(legacy)


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


def test_physical_binding_bridge_produces_valid_same_event_evidence():
    proxy = [proxy_evidence(index=1, node="SAT-01"), proxy_evidence(index=2, node="SAT-02")]
    bindings = [physical_binding(row) for row in proxy]
    rebound = bind_verified_physical_evidence(proxy, bindings)
    assert {row["linkage_mode"] for row in rebound} == {"verified_physical"}
    assert {row["emission_event_id"] for row in rebound} == {"EVT-CAPTURE-1"}
    assert {row["satellite_reception_id"] for row in rebound} == {
        "EVT-CAPTURE-1-SAT-01",
        "EVT-CAPTURE-1-SAT-02",
    }
    assert all(row["physical_binding_receipt_id"] == "CAPTURE-RECEIPT-1" for row in rebound)
    assert rebound[0]["z_id"] == proxy[0]["z_id"]
    result = fuse_event(
        rebound,
        FusionConfig(),
        physical_bindings=bindings,
        node_order={"SAT-01": 0, "SAT-02": 1},
    )
    assert result["event_key"] == "EVT-CAPTURE-1"
    assert result["decision"] == "unknown"
    assert result["shot_count"] == 1
    assert result["valid_reception_count"] == 2


def test_physical_binding_bridge_rejects_truth_tamper_and_partial_coverage():
    proxy = [proxy_evidence(index=1, node="SAT-01"), proxy_evidence(index=2, node="SAT-02")]
    raw = dict(physical_binding(proxy[0]))
    raw.pop("binding_hash")
    raw["role"] = "unknown"
    with pytest.raises(EvidenceError, match="unexpected physical binding fields"):
        seal_physical_binding(raw)

    tampered = physical_binding(proxy[0])
    tampered["emission_event_id"] = "TRUTH-DERIVED"
    with pytest.raises(EvidenceError, match="binding_hash mismatch"):
        bind_verified_physical_evidence(proxy[:1], [tampered])
    with pytest.raises(EvidenceError, match="exactly cover"):
        bind_verified_physical_evidence(proxy, [physical_binding(proxy[0])])

    mismatch = dict(physical_binding(proxy[0]))
    mismatch.pop("binding_hash")
    mismatch["node_id"] = "SAT-99"
    mismatch = seal_physical_binding(mismatch)
    with pytest.raises(EvidenceError, match="node_id mismatch"):
        bind_verified_physical_evidence(proxy[:1], [mismatch])


def test_physical_binding_bridge_rejects_duplicate_event_node_or_reception():
    first = proxy_evidence(index=1, node="SAT-01")
    second = proxy_evidence(index=2, node="SAT-01")
    with pytest.raises(EvidenceError, match="duplicate node evidence"):
        bind_verified_physical_evidence(
            [first, second],
            [
                physical_binding(first, reception_id="EVT-CAPTURE-1-RX-1"),
                physical_binding(second, reception_id="EVT-CAPTURE-1-RX-2"),
            ],
        )


def test_physical_binding_cli_end_to_end_and_no_overwrite(tmp_path):
    proxy = [proxy_evidence(index=1, node="SAT-01"), proxy_evidence(index=2, node="SAT-02")]
    bindings = [physical_binding(row) for row in proxy]
    evidence_path = tmp_path / "proxy.jsonl"
    binding_path = tmp_path / "binding.jsonl"
    output_path = tmp_path / "verified.jsonl"
    receipt_path = tmp_path / "receipt.json"
    write_jsonl(evidence_path, proxy)
    write_jsonl(binding_path, bindings)
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_bind_physical_evidence.py"
    command = [
        sys.executable,
        str(script),
        "--input-evidence",
        str(evidence_path),
        "--binding-jsonl",
        str(binding_path),
        "--output-jsonl",
        str(output_path),
        "--receipt-out",
        str(receipt_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert len(read_jsonl(output_path)) == 2
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["event_count"] == 1
    assert receipt["node_count"] == 2
    assert receipt["truth_or_role_opened"] is False
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr


def test_fixture_uses_binding_bridge_and_runs_abcd_predictor(tmp_path):
    fixture_script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_care_poe_fixture.py"
    predict_script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_care_poe_predict.py"
    score_script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_care_poe_score.py"
    fixture_dir = tmp_path / "fixture"
    prediction_dir = tmp_path / "prediction"
    subprocess.run(
        [sys.executable, str(fixture_script), "--output-dir", str(fixture_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((fixture_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["binding_count"] == 15
    assert manifest["event_count"] == 3
    assert manifest["truth_or_role_in_binding"] is False
    assert len(read_jsonl(fixture_dir / "physical_bindings.jsonl")) == 15
    subprocess.run(
        [
            sys.executable,
            str(predict_script),
            "--base-evidence",
            str(fixture_dir / "base_evidence.jsonl"),
            "--new-evidence",
            str(fixture_dir / "new_evidence.jsonl"),
            "--physical-bindings",
            str(fixture_dir / "physical_bindings.jsonl"),
            "--expected-physical-binding-root",
            manifest["binding_root"],
            "--output-dir",
            str(prediction_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    prediction_manifest = json.loads((prediction_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction_manifest["prediction_rows"] == 60
    assert prediction_manifest["truth_sidecar_opened"] is False
    assert prediction_manifest["physical_binding_root"] == manifest["binding_root"]
    metrics_path = tmp_path / "metrics.json"
    score_command = [
        sys.executable,
        str(score_script),
        "--predictions",
        str(prediction_dir / "predictions.jsonl"),
        "--prediction-manifest",
        str(prediction_dir / "prediction_manifest.json"),
        "--truth-sidecar",
        str(fixture_dir / "truth_sidecar.jsonl"),
        "--output",
        str(metrics_path),
    ]
    subprocess.run(score_command, check=True, capture_output=True, text=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["rows"]) == 20
    assert metrics["prediction_sha256"] == prediction_manifest["prediction_sha256"]

    truncated_path = tmp_path / "truncated_predictions.jsonl"
    truncated_predictions = [
        row for row in read_jsonl(prediction_dir / "predictions.jsonl") if row["node_budget"] == 1
    ]
    write_jsonl(truncated_path, truncated_predictions)
    truncated_manifest = dict(prediction_manifest)
    truncated_manifest["budgets"] = [1]
    truncated_manifest["prediction_rows"] = len(truncated_predictions)
    truncated_manifest["prediction_sha256"] = hashlib.sha256(truncated_path.read_bytes()).hexdigest()
    truncated_manifest_path = tmp_path / "truncated_prediction_manifest.json"
    truncated_manifest_path.write_text(canonical_json(truncated_manifest) + "\n", encoding="utf-8")
    truncated_command = list(score_command)
    truncated_command[3] = str(truncated_path)
    truncated_command[5] = str(truncated_manifest_path)
    truncated_command[-1] = str(tmp_path / "truncated_metrics.json")
    truncated = subprocess.run(truncated_command, check=False, capture_output=True, text=True)
    assert truncated.returncode != 0
    assert "budgets must be exactly" in truncated.stderr

    prediction_path = prediction_dir / "predictions.jsonl"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"\n")
    tampered_command = list(score_command)
    tampered_command[-1] = str(tmp_path / "tampered_metrics.json")
    tampered = subprocess.run(tampered_command, check=False, capture_output=True, text=True)
    assert tampered.returncode != 0
    assert "prediction_manifest hash" in tampered.stderr

    first = proxy_evidence(index=3, node="SAT-01")
    second = proxy_evidence(index=4, node="SAT-02")
    duplicate_reception = "EVT-CAPTURE-1-RX"
    with pytest.raises(EvidenceError, match="globally unique"):
        bind_verified_physical_evidence(
            [first, second],
            [
                physical_binding(first, reception_id=duplicate_reception),
                physical_binding(second, reception_id=duplicate_reception),
            ],
        )


def test_single_node_is_exact_identity_and_one_shot():
    row = evidence()
    result = fuse_event([row], FusionConfig(), physical_bindings=bindings_for([row]))
    assert result["p_fused"] == row["p_local"]
    assert result["decision"] == row["local_decision"]
    assert result["label"] == row["local_label"]
    assert result["shot_count"] == 1


def test_same_correlation_copy_does_not_add_strength():
    first = evidence(node="SAT-01", group="G1", probabilities=(0.15, 0.10, 0.75), decision="unknown", label=None)
    second = evidence(node="SAT-02", group="G1", probabilities=(0.15, 0.10, 0.75), decision="unknown", label=None)
    third = evidence(node="SAT-03", group="G1", probabilities=(0.80, 0.10, 0.10), decision="registered", label="a")
    two = fuse_event([first, second], FusionConfig(), physical_bindings=bindings_for([first, second]))
    three = fuse_event(
        [first, second, third],
        FusionConfig(),
        physical_bindings=bindings_for([first, second, third]),
    )
    assert two["p_fused"] == pytest.approx(three["p_fused"])
    assert two["correlation_group_count"] == three["correlation_group_count"] == 1
    assert two["decision"] != "unknown" and three["decision"] != "unknown"


def test_independent_groups_can_support_unknown_consensus():
    rows = [
        evidence(node="SAT-01", group="G1", probabilities=(0.05, 0.05, 0.90), decision="unknown", label=None, q=1.0),
        evidence(node="SAT-02", group="G2", probabilities=(0.05, 0.05, 0.90), decision="unknown", label=None, q=1.0),
    ]
    result = fuse_event(rows, FusionConfig(tau_group_quality=1.0), physical_bindings=bindings_for(rows))
    assert result["decision"] == "unknown"
    assert result["correlation_group_count"] == 2


def test_late_missing_and_integrity_failure_defer():
    late = evidence(delay=20.0)
    assert fuse_event([late], FusionConfig(), physical_bindings=bindings_for([late]))["reason_code"] == "NO_VALID_RECEPTION"
    mixed = [evidence(event_id="E1"), evidence(event_id="E2", node="SAT-02")]
    mixed_bindings = bindings_for(mixed)
    assert fuse_event(mixed, FusionConfig(), physical_bindings=mixed_bindings)["reason_code"] == "EVENT_INTEGRITY_FAILURE"
    assert fuse_event([], FusionConfig())["decision"] == "defer"


def test_node_and_class_permutation_are_semantically_invariant():
    rows = [
        evidence(node="SAT-01", group="G1", probabilities=(0.8, 0.1, 0.1)),
        evidence(node="SAT-02", group="G2", probabilities=(0.7, 0.2, 0.1)),
    ]
    forward = fuse_event(rows, FusionConfig(), physical_bindings=bindings_for(rows))
    reverse = fuse_event(list(reversed(rows)), FusionConfig(), physical_bindings=bindings_for(rows))
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
    permuted = fuse_event(swapped, FusionConfig(), physical_bindings=bindings_for(swapped))
    assert permuted["decision"] == forward["decision"]
    assert permuted["label"] == forward["label"]
    assert permuted["p_fused"] == pytest.approx([forward["p_fused"][1], forward["p_fused"][0], forward["p_fused"][2]])


def test_abcd_same_input_all_budgets_and_n1_parity():
    base = event_bundle("base")
    new = event_bundle("new", probabilities=(0.85, 0.08, 0.07))
    bindings = bindings_for(base)
    rows = run_abcd_matrix(
        base,
        new,
        FusionConfig(),
        node_roster=NODES,
        physical_bindings=bindings,
        expected_physical_binding_root=physical_binding_root(bindings),
    )
    assert len(rows) == 20
    assert {row["node_budget"] for row in rows} == {1, 2, 3, 4, 5}
    lookup = {(row["arm"], row["node_budget"]): row for row in rows}
    for left, right in (("A", "C"), ("B", "D")):
        assert lookup[(left, 1)]["p_fused"] == lookup[(right, 1)]["p_fused"]
        assert lookup[(left, 1)]["decision"] == lookup[(right, 1)]["decision"]
    assert all(row["shot_count"] == 1 for row in rows)
    with pytest.raises(EvidenceError, match="binding root mismatch"):
        run_abcd_matrix(
            base,
            new,
            FusionConfig(),
            node_roster=NODES,
            physical_bindings=bindings,
            expected_physical_binding_root="0" * 64,
        )


def test_abcd_rejects_different_physical_reception_binding():
    base = event_bundle("base")
    new = event_bundle("new")
    unsigned = copy.deepcopy(new[0])
    unsigned.pop("evidence_hash")
    unsigned["satellite_reception_id"] = "DIFFERENT-RECEPTION"
    new[0] = seal_local_evidence(unsigned)
    with pytest.raises(EvidenceError, match="physical binding"):
        bindings = bindings_for(base)
        run_abcd_matrix(
            base,
            new,
            FusionConfig(),
            node_roster=NODES,
            physical_bindings=bindings,
            expected_physical_binding_root=physical_binding_root(bindings),
        )


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
    with pytest.raises(ValueError, match="coverage mismatch"):
        score_predictions(
            [prediction],
            [
                {"event_key": "E1", "role": "registered", "true_label": "a"},
                {"event_key": "E2", "role": "unknown", "true_label": "u"},
            ],
        )


def test_truth_sidecar_never_changes_prediction_bytes():
    base = event_bundle("base")
    new = event_bundle("new")
    bindings = bindings_for(base)
    rows = run_abcd_matrix(
        base,
        new,
        FusionConfig(),
        node_roster=NODES,
        physical_bindings=bindings,
        expected_physical_binding_root=physical_binding_root(bindings),
    )
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
