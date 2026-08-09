#!/usr/bin/env python
"""Generate a CPU-only, truth-free CIRF-Track v3 G0 technical witness.

The output directory is created with ``exist_ok=False``.  Every emitted JSON
artifact states ``TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`` and deliberately
contains no score sidecar, query role, or performance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import SCHEMA as LOCAL_SCHEMA
from cvsrffi.phase3_care_poe import canonical_json, seal_local_evidence, sha256_json
from cvsrffi.phase3_cirf_track_v3 import (
    EVENT_AUTHORITY_SCHEMA,
    EVENT_CANDIDATE_SCHEMA,
    FACTOR_SCHEMA,
    FUSION_PLAN_SCHEMA,
    KERNEL_FIT_CELL_SCHEMA,
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    RISK_RECEIPT_SCHEMA,
    TECHNICAL_NO_PERFORMANCE,
    FrozenScheduler,
    ReplayGuard,
    ResourceBudget,
    SchedulerContract,
    SchedulerEventAuthority,
    TechnicalMHT,
    build_kernel_contract,
    capacity_preflight,
    conformal_vs_unknown_far_counterexample,
    enumerate_transcript_prefixes,
    factorized_log_opinion,
    fuse_factorized_event,
    nested_prediction_set,
    noncompensating_decision_gates,
    same_event_certificate,
    seal_decision_risk_receipt,
    seal_event_authority_receipt,
    seal_event_ledger,
    seal_factorized_evidence,
    seal_fusion_plan,
    seal_four_split_ledger,
    seal_interval_contract,
    seal_kernel_fit_cell_receipt,
    seal_same_event_candidate,
    scheduler_replay_store_id,
    split_conformal_quantile,
    transmission_opportunity_id,
    require_capacity,
    write_g0_manifest,
    write_json_artifact,
)


def _hex(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _local_evidence(node: str, reception: str, component: str) -> dict:
    return seal_local_evidence(
        {
            "schema_version": LOCAL_SCHEMA,
            "linkage_mode": "proxy_unverified",
            "proxy_group_id": "SYNTHETIC-G0-PROXY-GROUP",
            "satellite_reception_id": reception,
            "node_id": node,
            "base_manifest_id": "SYNTHETIC-CIRF-V3-G0",
            "bundle_id": "SYNTHETIC-FROZEN-BUNDLE",
            "class_handles": ["registered-a", "registered-b"],
            "z_id": [0.1, 0.2],
            "z_dom": [0.0, 0.1],
            "q": 0.9,
            "d_class": [0.2, 0.8],
            "e_unknown": 0.2,
            "p_local": [0.7, 0.1, 0.2],
            "correlation_group_id": component,
            "delay_ms": 1.0,
            "deadline_ms": 100.0,
            "local_decision": "registered",
            "local_label": "registered-a",
            "reason_code": "SYNTHETIC_LOCAL_TECHNICAL_ONLY",
            "sealed_at_ms": 1.0,
        }
    )


def _ledger(node: str, reception: str, origin: str, counter: int) -> dict:
    return seal_event_ledger(
        {
            "schema_version": LEDGER_SCHEMA,
            "reception_id": reception,
            "node_id": node,
            "roster_epoch": "SYNTHETIC-ROSTER-1",
            "clock_state_id": f"CLOCK-{node}",
            "clock_error_bound": 0.1,
            "drift_bound": 0.01,
            "capture_time_interval": [100.0 + counter * 0.01, 100.2 + counter * 0.01],
            "receiver_ephemeris_interval": [10.0, 10.1],
            "propagation_delay_interval": [2.0, 2.2],
            "carrier_frequency_interval": [1000.0, 1000.2],
            "doppler_residual_interval": [-0.05, 0.05],
            "beam_id": "BEAM-1",
            "visibility_cell": "VIS-1",
            "transmission_opportunity": {
                "roster_epoch": "SYNTHETIC-ROSTER-1",
                "time_slot": "SLOT-1",
                "band": "BAND-1",
                "beam": "BEAM-1",
                "visibility_cell": "VIS-1",
                "schedule_epoch": "SCHEDULE-1",
            },
            "waveform_digest": _hex("public-content-neutral-sync"),
            "nonce": f"NONCE-{node}",
            "monotonic_counter": counter,
            "key_epoch": "KEY-1",
            "revocation_epoch": "REV-1",
            "evidence_origin_id": origin,
            "revoked": False,
        }
    )


def _frozen_assets(origin: str, labels: list[str]) -> dict[str, str]:
    gate = {"a": 1.0, "b": 0.0, "fit_asset_hash": _hex("gate-fit-v1")}
    temperature = {"tau": 1.0, "fit_asset_hash": _hex("temperature-fit-v1")}
    prior = {labels[0]: 0.4, labels[1]: 0.4, "__unknown__": 0.2}
    codec = {"codec_id": "SYNTHETIC-TIER-CODEC", "codec_hash": _hex("tier-codec-v1")}
    return {
        "base_checkpoint_hash": _hex("checkpoint-v1"),
        "class_registry_hash": sha256_json(labels),
        "unknown_converter_hash": _hex("unknown-converter-v1"),
        "gate_calibrator_hash": sha256_json(gate),
        "registered_temperature_hash": sha256_json(temperature),
        "reference_prior_hash": sha256_json(prior),
        "tier_codec_hash": sha256_json(codec),
        "receiver_state_hash": _hex(f"receiver-state-{origin}"),
    }


def _factor(local: dict, ledger: dict, origin: str, raw_logits: list[float]) -> dict:
    labels = list(local["class_handles"])
    gate = {"a": 1.0, "b": 0.0, "fit_asset_hash": _hex("gate-fit-v1")}
    temperature = {"tau": 1.0, "fit_asset_hash": _hex("temperature-fit-v1")}
    prior = {labels[0]: 0.4, labels[1]: 0.4, "__unknown__": 0.2}
    codec = {"codec_id": "SYNTHETIC-TIER-CODEC", "codec_hash": _hex("tier-codec-v1")}
    assets = _frozen_assets(origin, labels)
    return seal_factorized_evidence(
        {
            "schema_version": FACTOR_SCHEMA,
            "local_evidence": local,
            "event_ledger": ledger,
            "evidence_origin_id": origin,
            "context_id": "SYNTHETIC-CONTEXT-1",
            "unknown_gate_raw_logit": -1.0,
            "registered_raw_logits": raw_logits,
            "asset_hashes": assets,
            "gate_calibrator": gate,
            "registered_temperature": temperature,
            "reference_prior_id": "SYNTHETIC-REF-PRIOR-1",
            "reference_prior": prior,
            "tier_codec": codec,
        }
    )


def _fit_blocks(axis: str, origins: list[str]) -> list[dict]:
    return [
        {
            "block_id": f"FIT-{index}",
            "split": "fit",
            "axis": axis,
            "context_id": "SYNTHETIC-CONTEXT-1",
            "availability": origins,
            "sealed_event_hash": _hex(f"{axis}-fit-event-{index}"),
            "residuals": {origin: base + 0.01 * origin_index for origin_index, origin in enumerate(origins)},
        }
        for index, base in enumerate((0.10, 0.20, 0.15), start=1)
    ]


def _fit_cell_receipt(axis: str, origins: list[str], blocks: list[dict]) -> dict:
    return seal_kernel_fit_cell_receipt(
        {
            "schema_version": KERNEL_FIT_CELL_SCHEMA,
            "axis": axis,
            "availability": origins,
            "context_id": "SYNTHETIC-CONTEXT-1",
            "fit_event_hash_by_block": {item["block_id"]: item["sealed_event_hash"] for item in blocks},
        }
    )


def _scheduler_catalog(origins: list[str]) -> dict[str, dict]:
    return {
        f"{origin}-{tier}-R0": {
            "origin_id": origin,
            "tier": tier,
            "retransmission_index": 0,
            "message_bytes": 10 if tier == "T1" else 20,
            "energy_upper": 1.0 if tier == "T1" else 2.0,
            "worst_delay_ms": 5.0 if tier == "T1" else 10.0,
            "shrinkage_lower_bound": 1.0 if tier == "T1" else 0.5,
        }
        for origin in origins
        for tier in ("T1", "T2")
    }


def _authority_and_candidate(ledgers: list[dict], *, replay_store_id: str) -> tuple[dict, dict]:
    opportunity = transmission_opportunity_id(ledgers[0])
    ledger_hashes = sorted(item["ledger_hash"] for item in ledgers)
    physical_constraint_hash = _hex("synthetic-physical-constraint-v1")
    collision_receipt_hash = _hex("synthetic-heldout-collision-gate-v1")
    authority = seal_event_authority_receipt(
        {
            "schema_version": EVENT_AUTHORITY_SCHEMA,
            "roster_epoch": "SYNTHETIC-ROSTER-1",
            "opportunity_id": opportunity,
            "ledger_hashes": ledger_hashes,
            "physical_constraint_hash": physical_constraint_hash,
            "collision_receipt_hash": collision_receipt_hash,
            "revocation_receipt_hash": _hex("synthetic-revocation-v1"),
            "replay_policy_hash": _hex("synthetic-replay-policy-v1"),
            "scheduler_replay_store_id": replay_store_id,
            "collision_gate_passed": True,
        }
    )
    candidate = seal_same_event_candidate(
        {
            "schema_version": EVENT_CANDIDATE_SCHEMA,
            "candidate_id": "SYNTHETIC-PHYSICAL-CANDIDATE-1",
            "opportunity_id": opportunity,
            "ledger_hashes": ledger_hashes,
            "physical_constraint_hash": physical_constraint_hash,
            "collision_receipt_hash": collision_receipt_hash,
        }
    )
    return authority, candidate


def _fusion_plan(*, ledgers: list[dict], local_a: dict, local_b: dict, kernel_r: dict, kernel_u: dict) -> dict:
    labels = list(local_a["class_handles"])
    expected_assets_by_origin = {
        "ORIGIN-A": _frozen_assets("ORIGIN-A", labels),
        "ORIGIN-B": _frozen_assets("ORIGIN-B", labels),
    }
    return seal_fusion_plan(
        {
            "schema_version": FUSION_PLAN_SCHEMA,
            "opportunity_id": transmission_opportunity_id(ledgers[0]),
            "availability": ["ORIGIN-A", "ORIGIN-B"],
            "context_id": "SYNTHETIC-CONTEXT-1",
            "common_reference_prior_id": "SYNTHETIC-REF-PRIOR-1",
            "expected_assets_by_origin": expected_assets_by_origin,
            "expected_roster_state_hashes": {
                "SAT-A": expected_assets_by_origin["ORIGIN-A"]["receiver_state_hash"],
                "SAT-B": expected_assets_by_origin["ORIGIN-B"]["receiver_state_hash"],
            },
            "operational_priors": [{labels[0]: 0.4, labels[1]: 0.4, "__unknown__": 0.2}],
            "kernel_r_hash": kernel_r["kernel_hash"],
            "kernel_u_hash": kernel_u["kernel_hash"],
            "cap_by_origin": {"ORIGIN-A": 1.0, "ORIGIN-B": 1.0},
            "component_caps": {"COMP-A": 1.0, "COMP-B": 1.0},
            "frozen_transforms": {},
        }
    )


def _split_assignments() -> list[dict]:
    rows = []
    for index, split in enumerate(("fit", "interval_calibration", "conformal_calibration", "formal_test"), start=1):
        rows.append(
            {
                "split": split,
                "emission_event_id": f"EV-{index}",
                "physical_sample_id": f"PHY-{index}",
                "risk_cluster_id": f"RISK-{index}",
                "event_opportunity_block_id": f"BLOCK-{index}",
                "identity_handle": "registered-a",
                "population": "registered",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="new immutable output directory")
    args = parser.parse_args()
    output = Path(args.output_dir)

    # Capacity closes before the entry point creates any run artifact or
    # synthesises evidence.  A rejected configuration therefore cannot mutate
    # a previous output directory or silently shrink a finite contract.
    origins = ["ORIGIN-A", "ORIGIN-B"]
    components = {"ORIGIN-A": "COMP-A", "ORIGIN-B": "COMP-B"}
    scheduler_contract = SchedulerContract(
        origins=tuple(origins),
        component_by_origin=components,
        roster_epoch="SYNTHETIC-ROSTER-1",
        delay_bucket_count=1,
        quantization_levels=1,
        interval_bin_counts=(1, 1, 1),
        max_retransmissions=0,
        hard_deadline_ms=100.0,
        message_catalog=_scheduler_catalog(origins),
    )
    capacity_config = {
        "origin_count": 2,
        "class_plus_unknown_count": 3,
        "context_bucket_count": 1,
        "stochastic_error_source_count": 1,
        "prior_count": 1,
        "scheduler_contract": scheduler_contract,
        "risk_cells": {"synthetic": {"alpha": 0.5, "delta": 0.05, "available_blocks": 100}},
    }
    capacity = require_capacity(capacity_config)
    output.mkdir(parents=True, exist_ok=False)

    local_a = _local_evidence("SAT-A", "RX-A", "COMP-A")
    local_b = _local_evidence("SAT-B", "RX-B", "COMP-B")
    ledger_a = _ledger("SAT-A", "RX-A", "ORIGIN-A", 1)
    ledger_b = _ledger("SAT-B", "RX-B", "ORIGIN-B", 1)
    scheduler_store_root = output / "scheduler_replay_store"
    authority, candidate = _authority_and_candidate(
        [ledger_a, ledger_b], replay_store_id=scheduler_replay_store_id(scheduler_store_root)
    )
    certificate = same_event_certificate(
        [ledger_a, ledger_b],
        candidate_hypotheses=[candidate],
        authority_receipt=authority,
        replay_guard=ReplayGuard(),
    )
    factor_a = _factor(local_a, ledger_a, "ORIGIN-A", [2.0, -1.0])
    factor_b = _factor(local_b, ledger_b, "ORIGIN-B", [1.8, -0.8])
    factors = [factor_a, factor_b]

    fit_blocks_r = _fit_blocks("R", origins)
    fit_blocks_u = _fit_blocks("U", origins)
    kernel_r = build_kernel_contract(
        axis="R",
        availability=origins,
        context_id="SYNTHETIC-CONTEXT-1",
        component_by_origin=components,
        fit_blocks=fit_blocks_r,
        fit_cell_receipt=_fit_cell_receipt("R", origins, fit_blocks_r),
    )
    kernel_u = build_kernel_contract(
        axis="U",
        availability=origins,
        context_id="SYNTHETIC-CONTEXT-1",
        component_by_origin=components,
        fit_blocks=fit_blocks_u,
        fit_cell_receipt=_fit_cell_receipt("U", origins, fit_blocks_u),
    )
    fusion_plan = _fusion_plan(
        ledgers=[ledger_a, ledger_b], local_a=local_a, local_b=local_b, kernel_r=kernel_r, kernel_u=kernel_u
    )
    fusion = fuse_factorized_event(
        factors,
        certificate=certificate,
        fusion_plan=fusion_plan,
        kernel_r=kernel_r,
        kernel_u=kernel_u,
    )
    split_ledger = seal_four_split_ledger(_split_assignments())
    interval = seal_interval_contract(
        four_split_ledger=split_ledger,
        delta_event=0.05,
        origin_count=2,
        class_plus_unknown_count=3,
        context_count=1,
        stochastic_error_sources=1,
        deterministic_envelope={"quantization": "SYNTHETIC-FROZEN", "top_l": "SYNTHETIC-FROZEN"},
        p_lower_function_hash=_hex("p-lower-v1"),
    )
    conformal = split_conformal_quantile([0.01, 0.02, 0.03, 0.04], 0.25)
    nested = nested_prediction_set(
        [{"registered-a": 0.01, "registered-b": 0.04}, {"registered-a": 0.02, "registered-b": 0.06}],
        {"registered-a": 0.03, "registered-b": 0.05},
    )

    transcripts = enumerate_transcript_prefixes(scheduler_contract)
    root_template = next(state for state in transcripts if state["tier_path"] == ["T0"])
    initial_budget = ResourceBudget(bytes_remaining=100, energy_remaining=10.0, deadline_slack_ms=100.0)
    scheduler_authority = SchedulerEventAuthority(
        scheduler_contract,
        event_authority_receipt=authority,
        root_template=root_template,
        initial_budget=initial_budget,
        replay_store_root=scheduler_store_root,
    )
    scheduler = FrozenScheduler(scheduler_contract, authority=scheduler_authority)
    scheduler_action = scheduler.request_next(
        scheduler_authority.root_state,
        initial_budget,
        ["ORIGIN-B-T1-R0"],
    )
    mht = TechnicalMHT()
    unknown_event = {
        "event_hash": _hex("unknown-event-1"),
        "decision": "unknown",
        "event_time": 10.0,
        "arrival_time": 10.0,
        "opportunity_index": 0,
        "possible_independent_components": 1,
        "associations": [],
    }
    registered_event = {
        "event_hash": _hex("registered-event-1"),
        "decision": "registered",
        "event_time": 11.0,
        "arrival_time": 11.0,
        "opportunity_index": 1,
        "possible_independent_components": 1,
        "associations": [],
    }
    mht_results = {
        "unknown": mht.process_event(unknown_event),
        "registered": mht.process_event(registered_event),
        "non_opportunity": mht.visibility_opportunity(possible_independent_components=0, now=86411.0),
    }
    risk_receipts = {
        risk_name: seal_decision_risk_receipt(
            {
                "schema_version": RISK_RECEIPT_SCHEMA,
                "risk_name": risk_name,
                "split": "formal_test",
                "alpha": 0.5,
                "delta": 0.05,
                "loss_range": [0.0, 1.0],
                "block_max_losses": {f"SYNTHETIC-BLOCK-{index}": 0.0 for index in range(1, 101)},
            }
        )
        for risk_name in ("R_known_id", "R_unknown_FA", "R_false_binding", "R_false_nonopportunity", "R_deadline")
    }
    noncompensating_gates = noncompensating_decision_gates(
        conformal_singleton=True,
        risk_receipts=risk_receipts,
    )

    scheduler_replay_receipt = scheduler_authority.replay_receipt()
    artifacts = {
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "truth_sidecar_opened": False,
        "event_ledger_rows": [ledger_a, ledger_b],
        "event_authority_receipt": authority,
        "same_event_candidate_receipt": candidate,
        "same_event_certificate": certificate,
        "factor_hashes": [item["factor_hash"] for item in factors],
        "factor_opinions": [factorized_log_opinion(item) for item in factors],
        "kernel_R": kernel_r,
        "kernel_U": kernel_u,
        "fusion_plan": fusion_plan,
        "technical_fusion": fusion,
        "four_split_ledger": split_ledger,
        "interval_contract": interval,
        "conformal": conformal,
        "nested_prediction_sets": nested,
        "conformal_vs_risk_counterexample": conformal_vs_unknown_far_counterexample(),
        "decision_risk_receipts": risk_receipts,
        "noncompensating_decision_gates": noncompensating_gates,
        "transcript_count": len(transcripts),
        "scheduler_action": scheduler_action,
        "scheduler_replay_receipt": scheduler_replay_receipt,
        "mht": mht_results,
        "capacity_preflight": capacity,
    }
    artifact_hash = write_json_artifact(output / "g0_technical_artifacts.json", artifacts)
    replay_receipt_hash = write_json_artifact(output / "scheduler_replay_receipt.json", scheduler_replay_receipt)
    manifest_write = write_g0_manifest(
        output,
        {
        "schema_version": MANIFEST_SCHEMA,
        "evidence_level": TECHNICAL_NO_PERFORMANCE,
        "technical_synthetic": True,
        "performance_result": False,
        "operational_claim": False,
        "truth_sidecar_opened": False,
        "cpu_only": True,
        "output_non_overwriting": True,
        "artifact_files": {
            "g0_technical_artifacts.json": artifact_hash,
            "scheduler_replay_receipt.json": replay_receipt_hash,
        },
        "external_receipt_filename": "g0_manifest_receipt.json",
        },
    )
    print(json.dumps(manifest_write["manifest"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
