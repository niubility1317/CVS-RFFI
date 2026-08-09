from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cvsrffi.phase3_care_poe import SCHEMA as LOCAL_SCHEMA
from cvsrffi.phase3_care_poe import canonical_json, seal_local_evidence, sha256_json
from cvsrffi.phase3_cirf_track_v3 import (
    EVENT_AUTHORITY_SCHEMA,
    EVENT_CANDIDATE_SCHEMA,
    FACTOR_SCHEMA,
    FALLBACK_MAD_SCHEMA,
    FUSION_PLAN_SCHEMA,
    KERNEL_FIT_CELL_SCHEMA,
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    RISK_RECEIPT_SCHEMA,
    TECHNICAL_NO_PERFORMANCE,
    CapacityRejected,
    CIRFContractError,
    FrozenScheduler,
    ReplayGuard,
    ResourceBudget,
    SchedulerContract,
    SchedulerEventAuthority,
    TechnicalMHT,
    build_kernel_contract,
    capacity_preflight,
    class_conditional_block_max_nonconformity,
    clopper_pearson_upper_bound,
    conformal_vs_unknown_far_counterexample,
    constrained_psd_completion,
    deduplicate_evidence_units,
    dual_axis_qp,
    enumerate_transcript_prefixes,
    factorized_log_opinion,
    fault_response_contract,
    fuse_factorized_event,
    n1_passthrough_bytes,
    nested_prediction_set,
    noncompensating_decision_gates,
    n_min_zero_failure,
    project_to_feasible_weights,
    reachable_transcript_breakdown,
    reject_all_is_not_a_safe_claim,
    registered_defer_is_error,
    require_capacity,
    restore_reference_prior,
    same_event_certificate,
    seal_decision_risk_receipt,
    seal_event_authority_receipt,
    seal_event_ledger,
    seal_factorized_evidence,
    seal_fallback_mad_receipt,
    seal_fusion_plan,
    seal_four_split_ledger,
    seal_interval_contract,
    seal_kernel_fit_cell_receipt,
    seal_same_event_candidate,
    scheduler_replay_store_id,
    solve_lexicographic_qp,
    split_conformal_quantile,
    top_l_worst_omission_envelope,
    topology_kernel,
    transmission_opportunity_id,
    validate_g0_manifest,
    validate_manifest_external_receipt,
    validate_factorized_evidence,
    validate_scheduler_replay_receipt,
)


def _hex(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _ledger(
    *,
    node: str = "SAT-A",
    reception: str = "RX-A",
    origin: str = "ORIGIN-A",
    counter: int = 1,
    capture=(100.0, 100.2),
    revoked: bool = False,
):
    return seal_event_ledger(
        {
            "schema_version": LEDGER_SCHEMA,
            "reception_id": reception,
            "node_id": node,
            "roster_epoch": "ROSTER-1",
            "clock_state_id": f"CLOCK-{node}",
            "clock_error_bound": 0.01,
            "drift_bound": 0.001,
            "capture_time_interval": list(capture),
            "receiver_ephemeris_interval": [10.0, 10.1],
            "propagation_delay_interval": [2.0, 2.1],
            "carrier_frequency_interval": [1000.0, 1000.1],
            "doppler_residual_interval": [-0.01, 0.01],
            "beam_id": "BEAM-1",
            "visibility_cell": "VIS-1",
            "transmission_opportunity": {
                "roster_epoch": "ROSTER-1",
                "time_slot": "SLOT-1",
                "band": "BAND-1",
                "beam": "BEAM-1",
                "visibility_cell": "VIS-1",
                "schedule_epoch": "SCHED-1",
            },
            "waveform_digest": _hex("neutral-sync"),
            "nonce": f"NONCE-{node}-{counter}",
            "monotonic_counter": counter,
            "key_epoch": "KEY-1",
            "revocation_epoch": "REV-1",
            "evidence_origin_id": origin,
            "revoked": revoked,
        }
    )


def _local(*, node="SAT-A", reception="RX-A", labels=("a", "b"), decision="registered", label="a"):
    return seal_local_evidence(
        {
            "schema_version": LOCAL_SCHEMA,
            "linkage_mode": "proxy_unverified",
            "proxy_group_id": "PROXY-G0",
            "satellite_reception_id": reception,
            "node_id": node,
            "base_manifest_id": "M-G0",
            "bundle_id": "B-G0",
            "class_handles": list(labels),
            "z_id": [0.1, 0.2],
            "z_dom": [0.1, 0.2],
            "q": 0.9,
            "d_class": [0.2 for _ in labels],
            "e_unknown": 0.2,
            "p_local": [0.7] + [0.1] * (len(labels) - 1) + [0.2],
            "correlation_group_id": f"COMP-{node}",
            "delay_ms": 1.0,
            "deadline_ms": 20.0,
            "local_decision": decision,
            "local_label": label if decision == "registered" else None,
            "reason_code": "SYNTHETIC",
            "sealed_at_ms": 1.0,
        }
    )


def _factor(
    *,
    node="SAT-A",
    reception="RX-A",
    origin="ORIGIN-A",
    labels=("a", "b"),
    logits=(2.0, -1.0),
    prior=None,
    reference_id="REF-1",
    ledger=None,
    tau=1.0,
):
    local = _local(node=node, reception=reception, labels=labels, label=labels[0])
    ledger = ledger or _ledger(node=node, reception=reception, origin=origin)
    prior = prior or {labels[0]: 0.4, labels[1]: 0.4, "__unknown__": 0.2}
    gate = {"a": 1.0, "b": 0.0, "fit_asset_hash": _hex("gate-fit")}
    temperature = {"tau": tau, "fit_asset_hash": _hex("temp-fit")}
    codec = {"codec_id": "CODEC-1", "codec_hash": _hex("codec")}
    assets = {
        "base_checkpoint_hash": _hex("checkpoint"),
        "class_registry_hash": sha256_json(list(labels)),
        "unknown_converter_hash": _hex("unknown-converter"),
        "gate_calibrator_hash": sha256_json(gate),
        "registered_temperature_hash": sha256_json(temperature),
        "reference_prior_hash": sha256_json(prior),
        "tier_codec_hash": sha256_json(codec),
        "receiver_state_hash": _hex(f"state-{origin}"),
    }
    return seal_factorized_evidence(
        {
            "schema_version": FACTOR_SCHEMA,
            "local_evidence": local,
            "event_ledger": ledger,
            "evidence_origin_id": origin,
            "context_id": "CTX-1",
            "unknown_gate_raw_logit": -1.0,
            "registered_raw_logits": list(logits),
            "asset_hashes": assets,
            "gate_calibrator": gate,
            "registered_temperature": temperature,
            "reference_prior_id": reference_id,
            "reference_prior": prior,
            "tier_codec": codec,
        }
    )


def _fit_blocks(origins, *, axis="R", count=3):
    return [
        {
            "block_id": f"F{index}",
            "split": "fit",
            "axis": axis,
            "context_id": "CTX-1",
            "availability": list(origins),
            "sealed_event_hash": _hex(f"{axis}-fit-event-{index}"),
            "residuals": {origin: base + 0.01 * origin_index for origin_index, origin in enumerate(origins)},
        }
        for index, base in enumerate((0.10, 0.20, 0.15)[:count], start=1)
    ]


def _fit_cell_receipt(origins, *, axis="R", blocks=None):
    blocks = blocks or _fit_blocks(origins, axis=axis)
    return seal_kernel_fit_cell_receipt(
        {
            "schema_version": KERNEL_FIT_CELL_SCHEMA,
            "axis": axis,
            "availability": list(origins),
            "context_id": "CTX-1",
            "fit_event_hash_by_block": {block["block_id"]: block["sealed_event_hash"] for block in blocks},
        }
    )


def _kernels(origins, components=None):
    components = components or {origin: f"C-{origin}" for origin in origins}
    blocks_r = _fit_blocks(origins, axis="R")
    blocks_u = _fit_blocks(origins, axis="U")
    kwargs = {
        "availability": list(origins),
        "context_id": "CTX-1",
        "component_by_origin": components,
    }
    return (
        build_kernel_contract(axis="R", fit_blocks=blocks_r, fit_cell_receipt=_fit_cell_receipt(origins, axis="R", blocks=blocks_r), **kwargs),
        build_kernel_contract(axis="U", fit_blocks=blocks_u, fit_cell_receipt=_fit_cell_receipt(origins, axis="U", blocks=blocks_u), **kwargs),
        components,
    )


def _authority_and_candidate(ledgers, *, physical_constraint="physical-v1", collision="collision-v1"):
    opportunity = transmission_opportunity_id(ledgers[0])
    ledger_hashes = sorted(ledger["ledger_hash"] for ledger in ledgers)
    physical_hash = _hex(physical_constraint)
    collision_hash = _hex(collision)
    authority = seal_event_authority_receipt(
        {
            "schema_version": EVENT_AUTHORITY_SCHEMA,
            "roster_epoch": ledgers[0]["roster_epoch"],
            "opportunity_id": opportunity,
            "ledger_hashes": ledger_hashes,
            "physical_constraint_hash": physical_hash,
            "collision_receipt_hash": collision_hash,
            "revocation_receipt_hash": _hex("revocation-v1"),
            "replay_policy_hash": _hex("replay-policy-v1"),
            "scheduler_replay_store_id": _hex("scheduler-replay-store-v1"),
            "collision_gate_passed": True,
        }
    )
    candidate = seal_same_event_candidate(
        {
            "schema_version": EVENT_CANDIDATE_SCHEMA,
            "candidate_id": "PHYSICAL-CANDIDATE-1",
            "opportunity_id": opportunity,
            "ledger_hashes": ledger_hashes,
            "physical_constraint_hash": physical_hash,
            "collision_receipt_hash": collision_hash,
        }
    )
    return authority, candidate


def _fusion_plan(factors, kernel_r, kernel_u, *, priors=None, context="CTX-1"):
    by_origin = {factor["evidence_origin_id"]: factor for factor in factors}
    availability = sorted(by_origin)
    first = by_origin[availability[0]]
    labels = list(first["local_evidence"]["class_handles"])
    priors = priors or [{labels[0]: 0.4, labels[1]: 0.4, "__unknown__": 0.2}]
    return seal_fusion_plan(
        {
            "schema_version": FUSION_PLAN_SCHEMA,
            "opportunity_id": transmission_opportunity_id(first["event_ledger"]),
            "availability": availability,
            "context_id": context,
            "common_reference_prior_id": first["reference_prior_id"],
            "expected_assets_by_origin": {origin: by_origin[origin]["asset_hashes"] for origin in availability},
            "expected_roster_state_hashes": {
                factor["local_evidence"]["node_id"]: factor["asset_hashes"]["receiver_state_hash"] for factor in factors
            },
            "operational_priors": priors,
            "kernel_r_hash": kernel_r["kernel_hash"],
            "kernel_u_hash": kernel_u["kernel_hash"],
            "cap_by_origin": {origin: 1.0 for origin in availability},
            "component_caps": {component: 1.0 for component in set(kernel_r["component_by_origin"].values())},
            "frozen_transforms": {},
        }
    )


def _scheduler_catalog(origins, *, retries=0):
    return {
        f"{origin}-{tier}-R{retry}": {
            "origin_id": origin,
            "tier": tier,
            "retransmission_index": retry,
            "message_bytes": 3 if tier == "T1" else 5,
            "energy_upper": 1.0 if tier == "T1" else 2.0,
            "worst_delay_ms": 4.0 if tier == "T1" else 7.0,
            "shrinkage_lower_bound": 1.0 if tier == "T1" else 0.5,
        }
        for origin in origins
        for tier in ("T1", "T2")
        for retry in range(retries + 1)
    }


def _scheduler_contract(origins=("A", "B"), *, retries=0, hard_deadline_ms=100.0, delay_bucket_count=1):
    return SchedulerContract(
        origins=tuple(origins),
        component_by_origin={origin: f"C-{origin}" for origin in origins},
        roster_epoch="ROSTER-1",
        delay_bucket_count=delay_bucket_count,
        quantization_levels=1,
        interval_bin_counts=(1, 1, 1),
        max_retransmissions=retries,
        hard_deadline_ms=hard_deadline_ms,
        message_catalog=_scheduler_catalog(origins, retries=retries),
    )


def _scheduler_authority(
    contract,
    root_template,
    budget,
    store_root,
    *,
    tag="scheduler-event",
    prior_replay_receipt=None,
):
    receipt = seal_event_authority_receipt(
        {
            "schema_version": EVENT_AUTHORITY_SCHEMA,
            "roster_epoch": contract.roster_epoch,
            "opportunity_id": _hex(f"{tag}-opportunity"),
            "ledger_hashes": [_hex(f"{tag}-ledger")],
            "physical_constraint_hash": _hex(f"{tag}-physical"),
            "collision_receipt_hash": _hex(f"{tag}-collision"),
            "revocation_receipt_hash": _hex(f"{tag}-revocation"),
            "replay_policy_hash": _hex(f"{tag}-replay"),
            "scheduler_replay_store_id": scheduler_replay_store_id(store_root),
            "collision_gate_passed": True,
        }
    )
    return SchedulerEventAuthority(
        contract,
        event_authority_receipt=receipt,
        root_template=root_template,
        initial_budget=budget,
        replay_store_root=store_root,
        prior_replay_receipt=prior_replay_receipt,
    )


def _split_rows():
    return [
        {
            "split": split,
            "emission_event_id": f"E-{index}",
            "physical_sample_id": f"P-{index}",
            "risk_cluster_id": f"R-{index}",
            "event_opportunity_block_id": f"B-{index}",
            "identity_handle": "registered-a",
            "population": "registered",
        }
        for index, split in enumerate(("fit", "interval_calibration", "conformal_calibration", "formal_test"), 1)
    ]


class _GuardedMapping(dict):
    def __init__(self, *args, guarded: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.guarded = guarded
        self.accessed = False

    def __getitem__(self, key):
        if key == self.guarded:
            self.accessed = True
            raise AssertionError(f"forbidden value accessed: {key}")
        return super().__getitem__(key)


@pytest.mark.parametrize("field", ["true_label", "role", "class_quota", "global_reassignment"])
def test_predictor_rejects_forbidden_query_fields_without_reading_values(field):
    payload = _factor()
    payload[field] = "forbidden"
    guarded = _GuardedMapping(payload, guarded=field)
    with pytest.raises(CIRFContractError, match="forbidden predictor fields"):
        seal_factorized_evidence(guarded)
    assert not guarded.accessed


def test_event_ledger_identity_blind_certificate_collision_and_counter_guards():
    left = _ledger()
    right = _ledger(node="SAT-B", reception="RX-B", origin="ORIGIN-B")
    authority, candidate = _authority_and_candidate([left, right])
    certificate = same_event_certificate(
        [left, right], candidate_hypotheses=[candidate], authority_receipt=authority, replay_guard=ReplayGuard()
    )
    assert certificate["certificate_status"] == "CERTIFIED"
    assert certificate["shot_count"] == 1
    assert certificate == same_event_certificate(
        [left, right], candidate_hypotheses=[candidate], authority_receipt=authority, replay_guard=ReplayGuard()
    )
    # A same-event certificate is unavailable without the sealed authority or
    # stateful replay guard; neither can be recovered from a candidate alone.
    assert same_event_certificate([left, right], candidate_hypotheses=[candidate])["reason_code"] == "REPLAY_GUARD_REQUIRED"
    assert same_event_certificate(
        [left, right], candidate_hypotheses=[candidate], replay_guard=ReplayGuard()
    )["reason_code"] == "EVENT_AUTHORITY_RECEIPT_REQUIRED"
    second_candidate = copy.deepcopy(candidate)
    second_candidate["candidate_id"] = "PHYSICAL-CANDIDATE-2"
    second_candidate.pop("candidate_receipt_hash")
    second_candidate = seal_same_event_candidate(second_candidate)
    assert same_event_certificate(
        [left, right],
        candidate_hypotheses=[candidate, second_candidate],
        authority_receipt=authority,
        replay_guard=ReplayGuard(),
    )["certificate_status"] == "NO_CERTIFICATE"
    unbound = copy.deepcopy(candidate)
    unbound["physical_constraint_hash"] = _hex("other-physical-constraint")
    unbound.pop("candidate_receipt_hash")
    unbound = seal_same_event_candidate(unbound)
    assert "CANDIDATE_AUTHORITY_BINDING_MISMATCH" in same_event_certificate(
        [left, right], candidate_hypotheses=[unbound], authority_receipt=authority, replay_guard=ReplayGuard()
    )["reason_code"]
    collision_unbound = copy.deepcopy(candidate)
    collision_unbound["collision_receipt_hash"] = _hex("other-collision-receipt")
    collision_unbound.pop("candidate_receipt_hash")
    collision_unbound = seal_same_event_candidate(collision_unbound)
    assert "CANDIDATE_AUTHORITY_BINDING_MISMATCH" in same_event_certificate(
        [left, right], candidate_hypotheses=[collision_unbound], authority_receipt=authority, replay_guard=ReplayGuard()
    )["reason_code"]
    collision_failed = copy.deepcopy(authority)
    collision_failed["collision_gate_passed"] = False
    collision_failed.pop("authority_receipt_hash")
    with pytest.raises(CIRFContractError, match="collision gate"):
        seal_event_authority_receipt(collision_failed)
    bad = copy.deepcopy(left)
    bad["transmission_opportunity"]["class_id"] = "forbidden"
    with pytest.raises(CIRFContractError):
        seal_event_ledger(bad)
    nested_bad = _GuardedMapping(
        {"geometry": _GuardedMapping({"class_id": "forbidden"}, guarded="class_id")},
        guarded="unused",
    )
    result = same_event_certificate(
        [left, right], candidate_hypotheses=[nested_bad], authority_receipt=authority, replay_guard=ReplayGuard()
    )
    assert result["certificate_status"] == "NO_CERTIFICATE"
    assert "unexpected same-event candidate fields" in result["reason_code"]
    assert not nested_bad["geometry"].accessed
    guard = ReplayGuard()
    guard.accept(left)
    with pytest.raises(CIRFContractError, match="REPLAY"):
        guard.accept(left)
    fork = _ledger(counter=1)
    fork["nonce"] = "DIFFERENT-NONCE"
    fork.pop("ledger_hash")
    fork = seal_event_ledger(fork)
    with pytest.raises(CIRFContractError, match="COUNTER_FORK"):
        guard.accept(fork)
    revoked = _ledger(node="SAT-C", reception="RX-C", origin="ORIGIN-C", revoked=True)
    revoked_authority, revoked_candidate = _authority_and_candidate([left, revoked])
    assert same_event_certificate(
        [left, revoked],
        candidate_hypotheses=[revoked_candidate],
        authority_receipt=revoked_authority,
        replay_guard=ReplayGuard(),
    )["certificate_status"] == "NO_CERTIFICATE"
    transactional_guard = ReplayGuard()
    assert same_event_certificate(
        [left, right],
        candidate_hypotheses=[candidate, second_candidate],
        authority_receipt=authority,
        replay_guard=transactional_guard,
    )["certificate_status"] == "NO_CERTIFICATE"
    assert same_event_certificate(
        [left, right], candidate_hypotheses=[candidate], authority_receipt=authority, replay_guard=transactional_guard
    )["certificate_status"] == "CERTIFIED"
    assert "REPLAY_DETECTED" in same_event_certificate(
        [left, right], candidate_hypotheses=[candidate], authority_receipt=authority, replay_guard=transactional_guard
    )["reason_code"]


def test_factorized_opinion_is_finite_simplex_and_assets_are_bound():
    factor = _factor()
    opinion = factorized_log_opinion(factor)
    assert abs(sum(__import__("math").exp(value) for value in opinion["log_probability"].values()) - 1.0) < 1e-12
    tampered = copy.deepcopy(factor)
    tampered["unknown_gate_raw_logit"] = float("inf")
    tampered.pop("factor_hash")
    with pytest.raises(CIRFContractError, match="finite"):
        seal_factorized_evidence(tampered)
    expected = dict(factor["asset_hashes"])
    expected["base_checkpoint_hash"] = _hex("different")
    with pytest.raises(CIRFContractError, match="asset hash drift"):
        validate_factorized_evidence(factor, expected_assets=expected)
    with pytest.raises(CIRFContractError):
        seal_factorized_evidence({**factor, "unknown_score": 0.2})
    tau_opinion = factorized_log_opinion(_factor(logits=(2.0, 0.0), tau=0.25))
    # tau is the frozen inverse temperature 1/T: it multiplies raw logits.
    registered_log_odds = tau_opinion["log_probability"]["a"] - tau_opinion["log_probability"]["b"]
    assert registered_log_odds == pytest.approx(0.5)


def test_reference_prior_restoration_requires_frozen_transform_and_is_label_symmetric():
    first = _factor(reference_id="REF-A")
    opinion = factorized_log_opinion(first)
    with pytest.raises(CIRFContractError, match="restoration"):
        restore_reference_prior(opinion, common_reference_prior_id="REF-B")
    transform = {
        "from_id": "REF-A",
        "to_id": "REF-B",
        "from_hash": opinion["reference_prior_hash"],
        "offsets": {"a": 0.0, "b": 0.0, "__unknown__": 0.0},
    }
    transform["transform_hash"] = sha256_json(transform)
    restored = restore_reference_prior(opinion, common_reference_prior_id="REF-B", frozen_transform=transform)
    assert restored["prior_corrected_log_evidence"] == opinion["prior_corrected_log_evidence"]
    permuted = _factor(labels=("b", "a"), logits=(-1.0, 2.0), prior={"b": 0.4, "a": 0.4, "__unknown__": 0.2})
    permuted_opinion = factorized_log_opinion(permuted)
    assert opinion["prior_corrected_log_evidence"]["a"] == pytest.approx(permuted_opinion["prior_corrected_log_evidence"]["a"])
    assert opinion["prior_corrected_log_evidence"]["b"] == pytest.approx(permuted_opinion["prior_corrected_log_evidence"]["b"])


def test_fusion_plan_receipt_binds_opportunity_assets_priors_kernels_caps_and_context():
    ledger_a = _ledger()
    ledger_b = _ledger(node="SAT-B", reception="RX-B", origin="ORIGIN-B")
    factors = [
        _factor(node="SAT-A", reception="RX-A", origin="ORIGIN-A", ledger=ledger_a),
        _factor(node="SAT-B", reception="RX-B", origin="ORIGIN-B", ledger=ledger_b),
    ]
    authority, candidate = _authority_and_candidate([ledger_a, ledger_b])
    certificate = same_event_certificate(
        [ledger_a, ledger_b], candidate_hypotheses=[candidate], authority_receipt=authority, replay_guard=ReplayGuard()
    )
    kernel_r, kernel_u, _ = _kernels(["ORIGIN-A", "ORIGIN-B"], {"ORIGIN-A": "A", "ORIGIN-B": "B"})
    plan = _fusion_plan(factors, kernel_r, kernel_u)
    result = fuse_factorized_event(factors, certificate=certificate, fusion_plan=plan, kernel_r=kernel_r, kernel_u=kernel_u)
    assert result["fusion_plan_hash"] == plan["fusion_plan_hash"]
    with pytest.raises(CIRFContractError, match="FusionPlan receipt"):
        fuse_factorized_event(factors, certificate=certificate, kernel_r=kernel_r, kernel_u=kernel_u)
    context_swapped = copy.deepcopy(factors[1])
    context_swapped["context_id"] = "CTX-SWAPPED"
    context_swapped.pop("factor_hash")
    context_swapped = seal_factorized_evidence(context_swapped)
    with pytest.raises(CIRFContractError, match="context binding"):
        fuse_factorized_event([factors[0], context_swapped], certificate=certificate, fusion_plan=plan, kernel_r=kernel_r, kernel_u=kernel_u)
    asset_swapped = copy.deepcopy(factors[1])
    asset_swapped["asset_hashes"]["base_checkpoint_hash"] = _hex("unbound-checkpoint")
    asset_swapped.pop("factor_hash")
    asset_swapped = seal_factorized_evidence(asset_swapped)
    with pytest.raises(CIRFContractError, match="asset hash drift"):
        fuse_factorized_event([factors[0], asset_swapped], certificate=certificate, fusion_plan=plan, kernel_r=kernel_r, kernel_u=kernel_u)
    for field_name, mutation in (
        ("context_id", "CTX-SWAPPED"),
        ("kernel_r_hash", _hex("other-R-kernel")),
        ("kernel_u_hash", _hex("other-U-kernel")),
        ("cap_by_origin", {"ORIGIN-A": 0.5, "ORIGIN-B": 1.0}),
        ("operational_priors", [{"a": 0.2, "b": 0.6, "__unknown__": 0.2}]),
        ("frozen_transforms", {"REF-1": {"unbound": "transform"}}),
    ):
        swapped = copy.deepcopy(plan)
        swapped[field_name] = mutation
        with pytest.raises(CIRFContractError, match="fusion plan hash mismatch"):
            fuse_factorized_event(factors, certificate=certificate, fusion_plan=swapped, kernel_r=kernel_r, kernel_u=kernel_u)
    with pytest.raises(CIRFContractError, match="kernel binding"):
        fuse_factorized_event(factors, certificate=certificate, fusion_plan=plan, kernel_r=kernel_u, kernel_u=kernel_r)


def test_kernel_contract_psd_components_and_split_merge_replication_contracts():
    origins = ["A", "B", "C"]
    top, groups = topology_kernel(origins, {"A": "X", "B": "X", "C": "Y"})
    assert top[0][1] == 1.0 and top[0][2] == 0.0 and groups == [["A", "B"], ["C"]]
    r, u, components = _kernels(origins, {"A": "X", "B": "X", "C": "Y"})
    assert r["kernel"][0][1] == pytest.approx(1.0)
    q = dual_axis_qp(
        r,
        u,
        active_origins=origins,
        cap_by_origin={origin: 1.0 for origin in origins},
        component_caps={"X": 1.0, "Y": 1.0},
    )
    assert sum(q["R"]["beta"]) == pytest.approx(1.0)
    assert sum(q["U"]["beta"]) == pytest.approx(1.0)
    full_r, full_u, full_components = _kernels(["A", "B"], {"A": "X", "B": "X"})
    full = dual_axis_qp(
        full_r,
        full_u,
        active_origins=["A", "B"],
        cap_by_origin={"A": 1.0, "B": 1.0},
        component_caps={"X": 1.0},
    )
    assert full["R"]["nu"] == pytest.approx(1.0)
    with pytest.raises(CIRFContractError, match="empty"):
        dual_axis_qp(
            full_r,
            full_u,
            active_origins=["A", "B"],
            cap_by_origin={"A": 0.4, "B": 0.4},
            component_caps={"X": 1.0},
        )
    factor = _factor()
    assert len(deduplicate_evidence_units([factor, factor])) == 1


def test_kernel_fit_cell_receipts_are_axis_context_event_exact_and_fallback_mad_is_provenanced():
    origins = ["A", "B"]
    components = {"A": "CA", "B": "CB"}
    blocks = _fit_blocks(origins, axis="R", count=1)
    receipt = _fit_cell_receipt(origins, axis="R", blocks=blocks)
    with pytest.raises(CIRFContractError, match="fallback MAD provenance"):
        build_kernel_contract(
            axis="R",
            availability=origins,
            context_id="CTX-1",
            component_by_origin=components,
            fit_blocks=blocks,
            fit_cell_receipt=receipt,
        )
    fallback = seal_fallback_mad_receipt(
        {
            "schema_version": FALLBACK_MAD_SCHEMA,
            "axis": "R",
            "availability": origins,
            "context_id": "CTX-1",
            "mad_scales": {"A": 1e-3, "B": 1e6},
            "source_fit_cell_receipt_hashes": [receipt["fit_cell_receipt_hash"]],
        }
    )
    contract = build_kernel_contract(
        axis="R",
        availability=origins,
        context_id="CTX-1",
        component_by_origin=components,
        fit_blocks=blocks,
        fit_cell_receipt=receipt,
        fallback_mad_receipt=fallback,
    )
    assert contract["mode"] == "TOPOLOGY_FALLBACK_INSUFFICIENT_BLOCKS"
    assert contract["fallback_mad_receipt_hash"] == fallback["fallback_mad_receipt_hash"]
    wrong_event = copy.deepcopy(blocks)
    wrong_event[0]["sealed_event_hash"] = _hex("different-sealed-event")
    with pytest.raises(CIRFContractError, match="not bound"):
        build_kernel_contract(
            axis="R",
            availability=origins,
            context_id="CTX-1",
            component_by_origin=components,
            fit_blocks=wrong_event,
            fit_cell_receipt=receipt,
            fallback_mad_receipt=fallback,
        )
    wrong_axis_receipt = copy.deepcopy(receipt)
    wrong_axis_receipt["axis"] = "U"
    wrong_axis_receipt.pop("fit_cell_receipt_hash")
    wrong_axis_receipt = seal_kernel_fit_cell_receipt(wrong_axis_receipt)
    with pytest.raises(CIRFContractError, match="availability/context/axis mismatch"):
        build_kernel_contract(
            axis="R",
            availability=origins,
            context_id="CTX-1",
            component_by_origin=components,
            fit_blocks=blocks,
            fit_cell_receipt=wrong_axis_receipt,
            fallback_mad_receipt=fallback,
        )


def test_two_stage_lexicographic_qp_handles_correlated_singular_scales_and_permutation():
    origins = ["A", "B", "C"]
    caps = {origin: 1.0 for origin in origins}
    components = {origin: origin for origin in origins}
    component_caps = {origin: 1.0 for origin in origins}
    reference = [0.15, 0.30, 0.55]
    fully_correlated = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    fully = solve_lexicographic_qp(
        covariance=fully_correlated,
        reference_weights=reference,
        origins=origins,
        cap_by_origin=caps,
        component_by_origin=components,
        component_caps=component_caps,
    )
    assert fully["solver"] == "two_stage_active_set_exact_float_spectrum_v3"
    assert fully["beta"] == pytest.approx(fully["beta0"])
    singular = solve_lexicographic_qp(
        covariance=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        reference_weights=reference,
        origins=origins,
        cap_by_origin=caps,
        component_by_origin=components,
        component_caps=component_caps,
    )
    assert sum(singular["beta"]) == pytest.approx(1.0, abs=1e-10)
    assert all(weight >= -1e-10 for weight in singular["beta"])
    permutation = ["C", "A", "B"]
    indices = [origins.index(origin) for origin in permutation]
    permuted = solve_lexicographic_qp(
        covariance=[[fully_correlated[i][j] for j in indices] for i in indices],
        reference_weights=[reference[index] for index in indices],
        origins=permutation,
        cap_by_origin={origin: 1.0 for origin in permutation},
        component_by_origin={origin: origin for origin in permutation},
        component_caps={origin: 1.0 for origin in permutation},
    )
    assert {origin: value for origin, value in zip(origins, fully["beta"])} == pytest.approx(
        {origin: value for origin, value in zip(permutation, permuted["beta"])}
    )
    near_singular = solve_lexicographic_qp(
        covariance=[[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]],
        reference_weights=[0.9, 0.1],
        origins=["A", "B"],
        cap_by_origin={"A": 1.0, "B": 1.0},
        component_by_origin={"A": "A", "B": "B"},
        component_caps={"A": 1.0, "B": 1.0},
    )
    # The antisymmetric eigenvalue is tiny but strictly positive.  Stage 1
    # therefore has the unique optimum (0.5, 0.5); stage 2 may not replace it
    # with the asymmetric reference point.
    assert near_singular["beta"] == pytest.approx([0.5, 0.5], abs=1e-7)
    adjacent_positive = solve_lexicographic_qp(
        covariance=[[1.0, 1.0 - 2**-53], [1.0 - 2**-53, 1.0]],
        reference_weights=[1.0, 0.0],
        origins=["A", "B"],
        cap_by_origin={"A": 1.0, "B": 1.0},
        component_by_origin={"A": "A", "B": "B"},
        component_caps={"A": 1.0, "B": 1.0},
    )
    # The smallest distinct off-diagonal representable below one still gives
    # strict antisymmetric curvature.  Its objective increment is below one
    # ulp of the shared baseline, so a rounded-float objective comparison would
    # incorrectly admit the asymmetric reference as a stage-2 null move.
    assert adjacent_positive["beta"] == pytest.approx([0.5, 0.5], abs=1e-12)
    r_blocks = _fit_blocks(["A", "B"], axis="R", count=1)
    u_blocks = _fit_blocks(["A", "B"], axis="U", count=1)
    r_receipt = _fit_cell_receipt(["A", "B"], axis="R", blocks=r_blocks)
    u_receipt = _fit_cell_receipt(["A", "B"], axis="U", blocks=u_blocks)
    def fallback(axis, receipt):
        return seal_fallback_mad_receipt(
            {
                "schema_version": FALLBACK_MAD_SCHEMA,
                "axis": axis,
                "availability": ["A", "B"],
                "context_id": "CTX-1",
                "mad_scales": {"A": 1e-3, "B": 1e6},
                "source_fit_cell_receipt_hashes": [receipt["fit_cell_receipt_hash"]],
            }
        )
    r = build_kernel_contract(axis="R", availability=["A", "B"], context_id="CTX-1", component_by_origin={"A": "A", "B": "B"}, fit_blocks=r_blocks, fit_cell_receipt=r_receipt, fallback_mad_receipt=fallback("R", r_receipt))
    u = build_kernel_contract(axis="U", availability=["A", "B"], context_id="CTX-1", component_by_origin={"A": "A", "B": "B"}, fit_blocks=u_blocks, fit_cell_receipt=u_receipt, fallback_mad_receipt=fallback("U", u_receipt))
    extreme = dual_axis_qp(r, u, active_origins=["A", "B"], cap_by_origin={"A": 1.0, "B": 1.0}, component_caps={"A": 1.0, "B": 1.0})
    assert all(math.isfinite(value) for value in extreme["R"]["beta"] + extreme["U"]["beta"])


def test_qp_all_nonempty_active_subsets_and_n1_byte_identity():
    origins = ["A", "B", "C"]
    for active_count in range(1, len(origins) + 1):
        for active in itertools.combinations(origins, active_count):
            r, u, components = _kernels(list(active))
            result = dual_axis_qp(
                r,
                u,
                active_origins=list(active),
                cap_by_origin={origin: 1.0 for origin in active},
                component_caps={components[origin]: 1.0 for origin in active},
            )
            if active_count == 1:
                assert result["mode"] == "DEGRADED_N1_NONCOLLABORATIVE"
            else:
                assert result["mode"] == "DUAL_AXIS_QP"
                assert all(weight >= -1e-10 for weight in result["R"]["beta"])
    local = _local()
    assert n1_passthrough_bytes(local) == canonical_json(local).encode("utf-8")
    original = ("  " + canonical_json(local) + "\n").encode("utf-8")
    assert n1_passthrough_bytes(original) == original
    n1 = fuse_factorized_event(
        [{"unvalidated_v3_wrapper": True}],
        certificate=None,
        n1_local_artifact_bytes=original,
    )
    assert n1["n1_local_artifact"].encode("utf-8") == original
    assert n1["n1_local_artifact_sha256"] == __import__("hashlib").sha256(original).hexdigest()


def test_four_split_interval_and_conformal_hash_isolation():
    split = seal_four_split_ledger(_split_rows())
    same_split_repeated_cluster = _split_rows() + [
        {
            **_split_rows()[0],
            "emission_event_id": "E-fit-second",
            "physical_sample_id": "P-fit-second",
        }
    ]
    seal_four_split_ledger(same_split_repeated_cluster)
    same_split_unknown = _split_rows() + [
        {
            **_split_rows()[0],
            "emission_event_id": "E-u1",
            "physical_sample_id": "P-u1",
            "risk_cluster_id": "R-u",
            "event_opportunity_block_id": "B-u",
            "identity_handle": "unknown-entity-a",
            "population": "unknown",
        },
        {
            **_split_rows()[0],
            "emission_event_id": "E-u2",
            "physical_sample_id": "P-u2",
            "risk_cluster_id": "R-u",
            "event_opportunity_block_id": "B-u",
            "identity_handle": "unknown-entity-a",
            "population": "unknown",
        },
    ]
    seal_four_split_ledger(same_split_unknown)
    duplicate = _split_rows()
    duplicate[-1]["physical_sample_id"] = duplicate[0]["physical_sample_id"]
    with pytest.raises(CIRFContractError, match="isolation"):
        seal_four_split_ledger(duplicate)
    interval = seal_interval_contract(
        four_split_ledger=split,
        delta_event=0.05,
        origin_count=2,
        class_plus_unknown_count=3,
        context_count=1,
        stochastic_error_sources=1,
        deterministic_envelope={"quant": "fixed"},
        p_lower_function_hash=_hex("p-lower"),
    )
    assert interval["N_atomic"] == 6
    original_hash = interval["interval_hash"]
    changed_later_rows = _split_rows()
    changed_later_rows[2]["identity_handle"] = "registered-renamed-only-in-conformal"
    changed_later_rows[3]["identity_handle"] = "registered-renamed-only-in-formal"
    changed_later_ledger = seal_four_split_ledger(changed_later_rows)
    changed_later_interval = seal_interval_contract(
        four_split_ledger=changed_later_ledger,
        delta_event=0.05,
        origin_count=2,
        class_plus_unknown_count=3,
        context_count=1,
        stochastic_error_sources=1,
        deterministic_envelope={"quant": "fixed"},
        p_lower_function_hash=_hex("p-lower"),
    )
    assert changed_later_interval["interval_hash"] == original_hash
    q = split_conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.25)
    permuted_q = split_conformal_quantile([0.4, 0.3, 0.2, 0.1], 0.25)
    assert q == permuted_q and interval["interval_hash"] == original_hash
    nested = nested_prediction_set(
        [{"a": 0.1, "b": 0.2}, {"a": 0.2, "b": 0.3}], {"a": 0.15, "b": 0.25}
    )
    assert set(nested[1]).issubset(set(nested[0]))
    block_max = class_conditional_block_max_nonconformity(
        [
            {"block_id": "B1", "class_label": "a", "context_id": "C", "nonconformity": 0.1, "split": "conformal_calibration"},
            {"block_id": "B1", "class_label": "a", "context_id": "C", "nonconformity": 0.4, "split": "conformal_calibration"},
            {"block_id": "B2", "class_label": "a", "context_id": "C", "nonconformity": 0.2, "split": "conformal_calibration"},
        ]
    )
    assert block_max[("a", "C")]["B1"] == 0.4
    envelope = top_l_worst_omission_envelope(
        {"a": 0.8, "b": 0.2}, omitted_probability_upper=0.1, interval_contract=interval
    )
    assert envelope["a"] == pytest.approx(0.7) and envelope["b"] == pytest.approx(0.1)


def test_conformal_and_cp_risk_are_not_interchangeable_and_reject_all_is_invalid():
    witness = conformal_vs_unknown_far_counterexample()
    assert witness["conformal_cell_has_finite_q"]
    assert not witness["unknown_far_gate_passes"]
    assert clopper_pearson_upper_bound(0, 59, 0.05) <= 0.05
    assert n_min_zero_failure(0.05, 0.05) == 59
    assert registered_defer_is_error("defer") == 1
    assert registered_defer_is_error("unknown") == 1
    assert reject_all_is_not_a_safe_claim(["defer", "unknown"])
    def receipt(name, losses):
        return seal_decision_risk_receipt(
            {
                "schema_version": RISK_RECEIPT_SCHEMA,
                "risk_name": name,
                "split": "formal_test",
                "alpha": 0.05,
                "delta": 0.05,
                "loss_range": [0.0, 1.0],
                "block_max_losses": losses,
            }
        )
    block_losses = {f"B-{index}": 0.0 for index in range(59)}
    receipts = {
        name: receipt(name, block_losses)
        for name in ("R_known_id", "R_unknown_FA", "R_false_binding", "R_false_nonopportunity", "R_deadline")
    }
    gates = noncompensating_decision_gates(conformal_singleton=True, risk_receipts=receipts)
    assert gates["all_gates_pass"]
    receipts["R_deadline"] = receipt("R_deadline", {"B-0": 1.0, **{f"B-{index}": 0.0 for index in range(1, 59)}})
    assert not noncompensating_decision_gates(conformal_singleton=True, risk_receipts=receipts)["all_gates_pass"]
    with pytest.raises(CIRFContractError, match=r"exactly \[0,1\]"):
        seal_decision_risk_receipt(
            {
                "schema_version": RISK_RECEIPT_SCHEMA,
                "risk_name": "R_deadline",
                "split": "formal_test",
                "alpha": 0.05,
                "delta": 0.05,
                "loss_range": [0.0, 0.5],
                "block_max_losses": {"B": 0.0},
            }
        )


def test_frozen_transcript_enumeration_hard_budget_and_unknown_state(tmp_path):
    contract = _scheduler_contract(delay_bucket_count=2)
    assert contract.resource_envelope()["max_request_plus_ack_bytes"] >= 3
    breakdown = reachable_transcript_breakdown(contract)
    assert breakdown["reachable_transcript_count"] > 0
    from cvsrffi.phase3_cirf_track_v3 import enumerate_transcript_prefixes

    states = enumerate_transcript_prefixes(contract)
    root_template = next(state for state in states if state["tier_path"] == ["T0"])
    refusal_budget = ResourceBudget(bytes_remaining=0, energy_remaining=0.0, deadline_slack_ms=0.0)
    refusal_authority = _scheduler_authority(contract, root_template, refusal_budget, tmp_path / "refusal", tag="refusal")
    refusal_scheduler = FrozenScheduler(contract, authority=refusal_authority)
    request = refusal_scheduler.request_next(
        refusal_authority.root_state, refusal_budget,
        ["B-T1-R0"],
    )
    assert request["reason_code"] == "HARD_BUDGET_PRE_SEND_REFUSAL"
    initial_budget = ResourceBudget(bytes_remaining=100, energy_remaining=100.0, deadline_slack_ms=100.0)
    authority = _scheduler_authority(contract, root_template, initial_budget, tmp_path / "permitted", tag="permitted")
    scheduler = FrozenScheduler(contract, authority=authority)
    permitted = scheduler.request_next(
        authority.root_state, initial_budget, ["B-T1-R0"]
    )
    assert permitted["action"] == "REQUEST"
    assert permitted["message_id"] == "B-T1-R0"
    assert permitted["next_state"]["elapsed_ms"] == pytest.approx(permitted["reserved_max_resources"]["delay_ms"])
    deadline_state = authority.root_state
    deadline_state["elapsed_ms"] = 99.0
    assert scheduler.request_next(
        deadline_state, ResourceBudget(100, 100.0, 100.0), ["B-T1-R0"]
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"
    with pytest.raises(CIRFContractError, match="absent from frozen catalog"):
        catalog_authority = _scheduler_authority(contract, root_template, initial_budget, tmp_path / "catalog", tag="catalog")
        FrozenScheduler(contract, authority=catalog_authority).request_next(
            catalog_authority.root_state, initial_budget, ["QUERY-TIME-INSERTED"]
        )
    unknown_authority = _scheduler_authority(contract, root_template, initial_budget, tmp_path / "unknown", tag="unknown")
    unknown = unknown_authority.root_state
    unknown["active_origins"] = ["NOT-IN-ROSTER"]
    assert FrozenScheduler(contract, authority=unknown_authority).request_next(
        unknown, initial_budget, []
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"


def test_scheduler_state_chain_rejects_elapsed_reset_budget_reset_and_replay(tmp_path):
    contract = _scheduler_contract(retries=1)
    root_template = next(state for state in enumerate_transcript_prefixes(contract) if state["tier_path"] == ["T0"])
    initial_budget = ResourceBudget(100, 100.0, 100.0)
    authority = _scheduler_authority(contract, root_template, initial_budget, tmp_path / "chain")
    root = authority.root_state
    scheduler = FrozenScheduler(contract, authority=authority)
    first = scheduler.request_next(root, initial_budget, ["B-T1-R0"])
    assert first["action"] == "REQUEST"
    next_state = first["next_state"]
    remaining = ResourceBudget(**{
        "bytes_remaining": first["remaining_after_request"]["bytes"],
        "energy_remaining": first["remaining_after_request"]["energy"],
        "deadline_slack_ms": first["remaining_after_request"]["deadline_slack_ms"],
    })

    # A consumed state is not replayable, even with the originally bound budget.
    assert scheduler.request_next(root, initial_budget, ["B-T1-R0"])["reason_code"] == "UNKNOWN_NETWORK_STATE"
    # Reconstructing only the scheduler cannot reset the event authority.
    assert FrozenScheduler(contract, authority=authority).request_next(
        root, initial_budget, ["B-T1-R0"]
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"

    # Resetting ordinary elapsed time is rejected by the cumulative receipt and
    # state digest, rather than being treated as a fresh deadline.
    reset = copy.deepcopy(next_state)
    reset["elapsed_ms"] = 0.0
    reset["state_hash"] = sha256_json({key: value for key, value in reset.items() if key != "state_hash"})
    assert scheduler.request_next(reset, remaining, ["B-T2-R1"])["reason_code"] == "UNKNOWN_NETWORK_STATE"

    parent_tampered = copy.deepcopy(next_state)
    parent_tampered["parent_state_hash"] = _hex("forged-parent")
    parent_tampered["state_hash"] = sha256_json(
        {key: value for key, value in parent_tampered.items() if key != "state_hash"}
    )
    assert scheduler.request_next(parent_tampered, remaining, ["B-T2-R1"])["reason_code"] == "UNKNOWN_NETWORK_STATE"

    cumulative_tampered = copy.deepcopy(next_state)
    cumulative_tampered["cumulative_reserved_resources"]["bytes"] += 1
    cumulative_tampered["state_hash"] = sha256_json(
        {key: value for key, value in cumulative_tampered.items() if key != "state_hash"}
    )
    assert scheduler.request_next(cumulative_tampered, remaining, ["B-T2-R1"])["reason_code"] == "UNKNOWN_NETWORK_STATE"

    # The caller may not restore the original resource budget on a valid state.
    assert scheduler.request_next(next_state, initial_budget, ["B-T2-R1"])["reason_code"] == "UNKNOWN_NETWORK_STATE"
    second = scheduler.request_next(next_state, remaining, ["B-T2-R1"])
    assert second["action"] == "REQUEST"
    assert second["next_state"]["parent_state_hash"] == next_state["state_hash"]
    assert second["next_state"]["request_history"] == ["B-T1-R0", "B-T2-R1"]

    receipt = authority.replay_receipt()
    assert receipt["event_authority_receipt_hash"] == root["event_authority_receipt_hash"]
    assert receipt["initial_budget_hash"] == root["initial_budget_hash"]
    assert root["state_hash"] in receipt["consumed_reason_by_state_hash"]
    assert next_state["state_hash"] in receipt["consumed_reason_by_state_hash"]


def test_scheduler_root_is_authority_issued_deep_copied_and_all_terminal_states_are_consumed(tmp_path):
    contract = _scheduler_contract(retries=1)
    template = next(state for state in enumerate_transcript_prefixes(contract) if state["tier_path"] == ["T0"])
    budget = ResourceBudget(100, 100.0, 100.0)
    store_root = tmp_path / "terminal"
    authority = _scheduler_authority(contract, template, budget, store_root, tag="terminal")
    scheduler = FrozenScheduler(contract, authority=authority)

    # An unbound enumerator template is not a runtime root.
    assert scheduler.request_next(template, budget, ["B-T1-R0"])["reason_code"] == "UNKNOWN_NETWORK_STATE"
    caller_copy = authority.root_state
    caller_copy["request_history"].append("B-T1-R0")
    pristine = authority.root_state
    assert pristine["request_history"] == []

    terminal = scheduler.request_next(pristine, budget, [])
    assert terminal["reason_code"] == "NO_POSITIVE_FROZEN_SHRINKAGE"
    assert FrozenScheduler(contract, authority=authority).request_next(
        pristine, budget, ["B-T1-R0"]
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"
    with pytest.raises(CIRFContractError, match="requires its latest external replay receipt"):
        _scheduler_authority(contract, template, budget, store_root, tag="terminal")
    rebuilt_authority = _scheduler_authority(
        contract,
        template,
        budget,
        store_root,
        tag="terminal",
        prior_replay_receipt=authority.replay_receipt(),
    )
    assert FrozenScheduler(contract, authority=rebuilt_authority).request_next(
        pristine, budget, ["B-T1-R0"]
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"
    assert rebuilt_authority.replay_receipt() == authority.replay_receipt()
    sealed_replay = validate_scheduler_replay_receipt(authority.replay_receipt())
    tampered_replay = copy.deepcopy(sealed_replay)
    tampered_replay["ledger_generation"] += 1
    with pytest.raises(CIRFContractError, match="receipt hash mismatch"):
        validate_scheduler_replay_receipt(tampered_replay)

    refusal_budget = ResourceBudget(0, 0.0, 0.0)
    refusal_authority = _scheduler_authority(
        contract, template, refusal_budget, tmp_path / "terminal-budget", tag="terminal-budget"
    )
    refusal_root = refusal_authority.root_state
    refusal_scheduler = FrozenScheduler(contract, authority=refusal_authority)
    assert refusal_scheduler.request_next(refusal_root, refusal_budget, ["B-T1-R0"])["reason_code"] == "HARD_BUDGET_PRE_SEND_REFUSAL"
    assert FrozenScheduler(contract, authority=refusal_authority).request_next(
        refusal_root, refusal_budget, ["B-T1-R0"]
    )["reason_code"] == "UNKNOWN_NETWORK_STATE"


def test_scheduler_append_only_ledger_blocks_same_session_replay_across_processes(tmp_path):
    store_root = tmp_path / "cross-process-ledger"
    child_env = dict(os.environ)
    code_root = str(Path(__file__).resolve().parents[1])
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (code_root, child_env.get("PYTHONPATH", "")) if part
    )
    program = r'''
import hashlib
import json
import sys
from pathlib import Path
from cvsrffi.phase3_cirf_track_v3 import (
    EVENT_AUTHORITY_SCHEMA, FrozenScheduler, ResourceBudget, SchedulerContract,
    SchedulerEventAuthority, enumerate_transcript_prefixes,
    scheduler_replay_store_id, seal_event_authority_receipt,
)
def h(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
store = sys.argv[1]
receipt_path = sys.argv[2]
receipt_file = Path(receipt_path)
contract = SchedulerContract(
    origins=("A", "B"),
    component_by_origin={"A": "C-A", "B": "C-B"},
    roster_epoch="ROSTER-1",
    delay_bucket_count=1,
    quantization_levels=1,
    interval_bin_counts=(1, 1, 1),
    max_retransmissions=1,
    hard_deadline_ms=100.0,
    message_catalog={
        f"{origin}-{tier}-R{retry}": {
            "origin_id": origin, "tier": tier, "retransmission_index": retry,
            "message_bytes": 3 if tier == "T1" else 5,
            "energy_upper": 1.0 if tier == "T1" else 2.0,
            "worst_delay_ms": 4.0 if tier == "T1" else 7.0,
            "shrinkage_lower_bound": 1.0 if tier == "T1" else 0.5,
        }
        for origin in ("A", "B") for tier in ("T1", "T2") for retry in (0, 1)
    },
)
budget = ResourceBudget(100, 100.0, 100.0)
template = next(state for state in enumerate_transcript_prefixes(contract) if state["tier_path"] == ["T0"])
receipt = seal_event_authority_receipt({
    "schema_version": EVENT_AUTHORITY_SCHEMA,
    "roster_epoch": contract.roster_epoch,
    "opportunity_id": h("cross-process-opportunity"),
    "ledger_hashes": [h("cross-process-ledger")],
    "physical_constraint_hash": h("cross-process-physical"),
    "collision_receipt_hash": h("cross-process-collision"),
    "revocation_receipt_hash": h("cross-process-revocation"),
    "replay_policy_hash": h("cross-process-replay"),
    "scheduler_replay_store_id": scheduler_replay_store_id(store),
    "collision_gate_passed": True,
})
authority = SchedulerEventAuthority(
    contract,
    event_authority_receipt=receipt,
    root_template=template,
    initial_budget=budget,
    replay_store_root=store,
    prior_replay_receipt=(json.loads(receipt_file.read_text(encoding="utf-8")) if receipt_file.exists() else None),
)
result = FrozenScheduler(contract, authority=authority).request_next(
    authority.root_state, budget, ["B-T1-R0"]
)
replay_receipt = authority.replay_receipt()
if not receipt_file.exists():
    with open(receipt_path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(replay_receipt, sort_keys=True, separators=(",", ":")) + "\n")
print(json.dumps({"result": result, "receipt": replay_receipt}, sort_keys=True))
'''
    receipt_path = tmp_path / "authoritative-latest-replay-receipt.json"
    first = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program, str(store_root), str(receipt_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_env,
    )
    second = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program, str(store_root), str(receipt_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_env,
    )
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    assert first_payload["result"]["action"] == "REQUEST"
    assert second_payload["result"]["reason_code"] == "UNKNOWN_NETWORK_STATE"
    assert second_payload["receipt"] == first_payload["receipt"]
    generations = sorted((store_root / first_payload["receipt"]["scheduler_session_hash"]).glob("*.json"))
    assert [path.name for path in generations] == ["00000000000000000000.json", "00000000000000000001.json"]
    # Removing the latest generation cannot roll authority back to the still
    # self-consistent root generation because the separately sealed latest
    # replay receipt anchors generation 1.
    generations[-1].unlink()
    rolled_back = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program, str(store_root), str(receipt_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_env,
    )
    assert rolled_back.returncode != 0
    assert "does not anchor the latest ledger head" in rolled_back.stderr


def test_psd_completion_and_fault_boundaries_are_conservative():
    completed = constrained_psd_completion([[1.0, 1.5], [1.5, 1.0]], components=[[0], [1]])
    assert all(0.0 <= value <= 1.0 for row in completed["kernel"] for value in row)
    from cvsrffi.phase3_cirf_track_v3 import is_psd

    assert is_psd(completed["kernel"])
    assert fault_response_contract(fault_kind="fail_silent", independent_component_count=1)["action"] == "REBUILD_ACTIVE_SET_OR_DEFER"
    assert fault_response_contract(fault_kind="bounded_numeric", independent_component_count=1)["action"] == "USE_FROZEN_INTERVAL_OR_DEFER"
    assert fault_response_contract(fault_kind="authenticated_byzantine", independent_component_count=2, conflicting_components=1)["action"] == "CONFLICT_DEFER_NO_BYZANTINE_CLAIM"
    assert fault_response_contract(fault_kind="authenticated_byzantine", independent_component_count=3, leave_one_component_invariant=True)["action"] == "TECHNICAL_BYZANTINE_CONTRACT_SATISFIED"


def test_technical_mht_visibility_oos_late_and_z_track_rejection():
    mht = TechnicalMHT()
    unknown = {
        "event_hash": _hex("unknown-1"),
        "decision": "unknown",
        "event_time": 10.0,
        "arrival_time": 10.0,
        "opportunity_index": 0,
        "possible_independent_components": 1,
        "associations": [],
    }
    created = mht.process_event(unknown)
    before = len(mht.tracks)
    registered = {**unknown, "event_hash": _hex("registered-1"), "decision": "registered", "event_time": 11.0, "arrival_time": 11.0, "opportunity_index": 1}
    assert mht.process_event(registered)["status"] == "REGISTERED_EVENT_NO_TRACK_MUTATION"
    assert len(mht.tracks) == before
    assert mht.visibility_opportunity(possible_independent_components=0, now=100.0)["status"] == "NON_OPPORTUNITY_NO_MISS"
    assert mht.online_as_of_revision(9.0) is None
    first_revision = mht.online_as_of_revision(10.0)
    oos = {**unknown, "event_hash": _hex("unknown-oos"), "event_time": 9.0, "arrival_time": 11.0, "opportunity_index": 0}
    oos_result = mht.process_event(oos)
    assert oos_result["oos_within_lag"] and oos_result["revision"]["revision_scope"] == "ONLINE_AS_OF"
    assert mht.online_as_of_revision(10.0)["revision_hash"] == first_revision["revision_hash"]
    assert mht.online_as_of_revision(11.0)["revision_hash"] == oos_result["revision"]["revision_hash"]
    late = {**unknown, "event_hash": _hex("late"), "event_time": 1.0, "arrival_time": 200.0, "opportunity_index": 0}
    assert mht.process_event(late)["status"] == "LATE_EVENT_AUDIT_ONLY"
    opportunity_late = {**unknown, "event_hash": _hex("opportunity-late"), "event_time": 11.0, "arrival_time": 11.0, "opportunity_index": 6}
    mht.process_event(opportunity_late)
    old_window = {**unknown, "event_hash": _hex("old-window"), "event_time": 11.1, "arrival_time": 11.1, "opportunity_index": 0}
    assert mht.process_event(old_window)["status"] == "N_SCAN_FINALIZED_AUDIT_ONLY"
    bad = {**unknown, "z_track": [1.0]}
    with pytest.raises(CIRFContractError, match="forbidden"):
        mht.process_event(bad)
    archived = mht.visibility_opportunity(possible_independent_components=0, now=90000.0)
    assert archived["archived_tracks"]
    assert any(track["alive"] for track in mht.tracks.values())
    reappeared = mht.process_event(
        {
            **unknown,
            "event_hash": _hex("post-archive"),
            "event_time": 90001.0,
            "arrival_time": 90001.0,
            "opportunity_index": 7,
            "associations": [{"track_id": created["birth_track_id"], "physical_log_likelihood": 10.0}],
        }
    )
    assert reappeared["birth_track_id"] != created["birth_track_id"]
    assert created["event_decision_rewritten"] is False


def test_technical_mht_fixed_boundaries_high_clutter_and_capacity_pruning():
    mht = TechnicalMHT()
    last = None
    for index in range(12):
        associations = [
            {"track_id": track_id, "physical_log_likelihood": float(rank)}
            for rank, track_id in enumerate(sorted(mht.tracks), start=1)
        ]
        last = mht.process_event(
            {
                "event_hash": _hex(f"clutter-{index}"),
                "decision": "defer",
                "event_time": float(index),
                "arrival_time": float(index),
                "opportunity_index": index,
                "possible_independent_components": 1,
                "associations": associations,
            }
        )
    assert last["status"] == "CAPACITY_PRUNED"
    death_mht = TechnicalMHT()
    death_mht.process_event(
        {
            "event_hash": _hex("death-base"),
            "decision": "unknown",
            "event_time": 0.0,
            "arrival_time": 0.0,
            "opportunity_index": 0,
            "possible_independent_components": 1,
            "associations": [],
        }
    )
    for tick in range(3):
        death_mht.visibility_opportunity(possible_independent_components=1, now=float(tick + 1))
    assert all(track["alive"] for track in death_mht.tracks.values())
    death_mht.visibility_opportunity(possible_independent_components=1, now=4.0)
    assert all(not track["alive"] for track in death_mht.tracks.values())


def test_mht_n_scan_finalization_rejects_late_reassociation_inside_time_lag():
    mht = TechnicalMHT()
    events = []
    for index in range(4):
        event = {
            "event_hash": _hex(f"scan-{index}"),
            "decision": "unknown",
            "event_time": float(index),
            "arrival_time": float(index),
            "opportunity_index": index,
            "possible_independent_components": 1,
            "associations": [],
        }
        events.append(event)
        mht.process_event(event)
    assert events[0]["event_hash"] in mht.revisions[-1]["n_scan_finalized_event_hashes"]
    assert mht.process_event(
        {**events[0], "event_hash": events[0]["event_hash"], "arrival_time": 4.0}
    )["status"] == "N_SCAN_FINALIZED_AUDIT_ONLY"
    before = {
        "tracks": copy.deepcopy(mht.tracks),
        "hypotheses": copy.deepcopy(mht.hypotheses),
        "history": copy.deepcopy(mht._event_history),
        "revision_count": len(mht.revisions),
        "watermark": mht._processing_watermark,
    }
    unseen_late = {
        **events[0],
        "event_hash": _hex("scan-unseen-but-finalized"),
        "arrival_time": 4.0,
    }
    result = mht.process_event(unseen_late)
    assert result["status"] == "N_SCAN_FINALIZED_AUDIT_ONLY"
    assert mht.tracks == before["tracks"]
    assert mht.hypotheses == before["hypotheses"]
    assert mht._event_history == before["history"]
    assert len(mht.revisions) == before["revision_count"]
    assert mht._processing_watermark == before["watermark"]


def test_capacity_preflight_exact_limits_and_pre_run_refusal():
    small_scheduler = _scheduler_contract()
    base = {
        "origin_count": 2,
        "class_plus_unknown_count": 3,
        "context_bucket_count": 1,
        "stochastic_error_source_count": 1,
        "prior_count": 1,
        "scheduler_contract": small_scheduler,
        "risk_cells": {"cell": {"alpha": 0.5, "delta": 0.05, "available_blocks": 100}},
    }
    approved = require_capacity(base)
    assert approved["status"] == "CAPACITY_APPROVED"
    assert {"N_atomic", "n_min_by_risk_cell", "reachable_transcript_count", "QP_solve_count", "primitive_operation_upper_bound", "peak_memory_upper_bound_bytes"}.issubset(approved)
    large_origins = ("A", "B", "C", "D", "E")
    large_scheduler = SchedulerContract(
        origins=large_origins,
        component_by_origin={origin: origin for origin in large_origins},
        roster_epoch="ROSTER-1",
        delay_bucket_count=4,
        quantization_levels=3,
        interval_bin_counts=(4, 4, 4),
        max_retransmissions=1,
        message_catalog=_scheduler_catalog(large_origins, retries=1),
    )
    over = dict(base)
    over.update(
        {
            "origin_count": 5,
            "class_plus_unknown_count": 32,
            "context_bucket_count": 12,
            "stochastic_error_source_count": 4,
            "prior_count": 4,
            "scheduler_contract": large_scheduler,
            "risk_cells": {"cell": {"alpha": 0.5, "delta": 0.05, "available_blocks": 0}},
        }
    )
    rejected = capacity_preflight(over)
    assert rejected["status"] == "CAPACITY_REJECTED_PRE_RUN"
    assert "TRANSCRIPT_CAP" in rejected["refusal_reasons"]
    with pytest.raises(CapacityRejected):
        require_capacity(over)


def test_cpu_cli_generates_non_overwriting_technical_only_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_cirf_track_v3_g0.py"
    output = tmp_path / "g0"
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-W", "error", str(script), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    manifest = json.loads((output / "g0_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "g0_manifest_receipt.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output / "g0_technical_artifacts.json").read_text(encoding="utf-8"))
    replay_receipt = json.loads((output / "scheduler_replay_receipt.json").read_text(encoding="utf-8"))
    assert TECHNICAL_NO_PERFORMANCE == manifest["evidence_level"] == artifacts["evidence_level"]
    assert manifest["performance_result"] is False and artifacts["truth_sidecar_opened"] is False
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["artifact_files"]["g0_technical_artifacts.json"] == hashlib.sha256((output / "g0_technical_artifacts.json").read_bytes()).hexdigest()
    assert manifest["artifact_files"]["scheduler_replay_receipt.json"] == hashlib.sha256((output / "scheduler_replay_receipt.json").read_bytes()).hexdigest()
    assert validate_scheduler_replay_receipt(replay_receipt) == replay_receipt
    assert replay_receipt == artifacts["scheduler_replay_receipt"]
    ledger_files = list((output / "scheduler_replay_store" / replay_receipt["scheduler_session_hash"]).glob("*.json"))
    assert sorted(path.name for path in ledger_files) == ["00000000000000000000.json", "00000000000000000001.json"]
    assert validate_g0_manifest(manifest)["manifest_self_hash"] == manifest["manifest_self_hash"]
    assert validate_manifest_external_receipt(receipt)["manifest_self_hash"] == manifest["manifest_self_hash"]
    assert receipt["manifest_content_sha256"] == hashlib.sha256((output / "g0_manifest.json").read_bytes()).hexdigest()
    assert "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT" in completed.stdout
    assert "Warning" not in completed.stderr
    repeated = subprocess.run(
        [sys.executable, "-X", "utf8", "-W", "error", str(script), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    assert repeated.returncode != 0
