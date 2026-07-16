from __future__ import annotations

import copy

import pytest

from cvsrffi.phase2_runtime_contract import (
    PHASE2_FULL_CONTRACT,
    PHASE2_SINGLE_OBSERVATION_CONTRACT,
    Phase2ContractError,
    classify_legacy_phase2_record,
    validate_phase2_contract,
    validate_predictor_request,
)


def _descriptor(role: str, path: str):
    return {
        "relative_path": path,
        "sha256": "a" * 64,
        "size_bytes": 10,
        "artifact_role": role,
        "schema": f"cvs.phase2.{role}.v1",
    }


def _pre_run_evidence():
    return {
        "sealed_inference_package_sha256": "a" * 64,
        "package_root_sha256": "2" * 64,
        "runtime_code_sha256": "3" * 64,
        "artifact_member_allowlist_sha256": "d" * 64,
        "os_isolation_mode": "bwrap_readonly_mounts",
        "os_isolation_attestation_sha256": "e" * 64,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": "f" * 64,
        "predict_score_process_isolation": True,
    }


def _request():
    return {
        **PHASE2_FULL_CONTRACT,
        "schema_version": "cvs.phase2.predict_request.v2",
        "request_id": "req-1",
        "row_id": "row-1",
        "stage": "stage2c",
        "receiver": "20-1",
        "scenarios": [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ],
        "k_shot": 1,
        "satellite_seed": 713101,
        "candidate_lock_sha256": "1" * 64,
        "package_root_sha256": "2" * 64,
        "runtime_code_sha256": "3" * 64,
        "registered_class_count": 2,
        "registered_classes": [
            {"class_index": 0, "class_handle": "cls_" + "0" * 32},
            {"class_index": 1, "class_handle": "cls_" + "1" * 32},
        ],
        "support_artifacts": [
            _descriptor(f"support:{scenario}", f"support_{scenario}.npz")
            for scenario in (
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            )
        ],
        "query_artifacts": [
            _descriptor(f"query:{scenario}", f"query_{scenario}.npz")
            for scenario in (
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            )
        ],
        "checkpoint_artifact": _descriptor("checkpoint", "checkpoint.bin"),
        "adapter_artifact": _descriptor("adapter", "adapter.bin"),
        "head_artifact": _descriptor("head", "head.npz"),
        "tta_policy": {"base_views": 1, "max_views": 5},
        "tta_policy_sha256": "4" * 64,
        "output_contract": {
            "schema": "cvs.phase2.prediction.v2",
            "relative_path": "prediction.npz",
            "sealed_immutable_required": True,
        },
        "phase2_runtime_isolation_evidence": _pre_run_evidence(),
    }


def test_full_contract_includes_clean_query_and_source_reachability_fields():
    assert len(PHASE2_FULL_CONTRACT) == 26
    assert PHASE2_FULL_CONTRACT["phase2_query_true_batch_class_count_access"] is False
    assert PHASE2_FULL_CONTRACT["phase2_source_sample_access"] is False
    assert PHASE2_FULL_CONTRACT["phase2_external_source_adapter_access"] is False


def test_full_contract_includes_exact_single_observation_policy():
    assert PHASE2_SINGLE_OBSERVATION_CONTRACT == {
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": "immutable_preoverlay_lineage_token",
        "phase2_query_post_reception_view_fit_access": False,
    }
    assert {
        key: PHASE2_FULL_CONTRACT[key]
        for key in PHASE2_SINGLE_OBSERVATION_CONTRACT
    } == PHASE2_SINGLE_OBSERVATION_CONTRACT


@pytest.mark.parametrize("field", PHASE2_SINGLE_OBSERVATION_CONTRACT)
def test_manifest_contract_missing_single_observation_field_fails_closed(field):
    manifest = copy.deepcopy(PHASE2_FULL_CONTRACT)
    manifest.pop(field)
    with pytest.raises(Phase2ContractError, match=rf"missing:{field}"):
        validate_phase2_contract(manifest, evidence_phase="none")


@pytest.mark.parametrize("field", PHASE2_SINGLE_OBSERVATION_CONTRACT)
def test_predictor_request_missing_single_observation_field_fails_closed(field):
    request = _request()
    request.pop(field)
    with pytest.raises(Phase2ContractError, match=rf"missing:{field}"):
        validate_predictor_request(request)


def test_registered_class_count_is_legal_predictor_state():
    validate_predictor_request(_request())


@pytest.mark.parametrize(
    "key,value",
    [
        ("query_truth", [0, 1]),
        ("evaluation_role", ["old", "new"]),
        ("query_per_tx", 20),
        ("query_class_histogram", {"0": 1}),
        ("truth_sidecar", "score/truth_sidecar.npz"),
        ("build_spec", "cache_specs/target.json"),
    ],
)
def test_predictor_request_rejects_truth_role_count_quota_and_build_signals(key, value):
    request = _request()
    request[key] = value
    with pytest.raises(Phase2ContractError):
        validate_predictor_request(request)


def test_predictor_request_requires_every_declared_key():
    request = _request()
    request.pop("scenarios")
    with pytest.raises(Phase2ContractError, match="predictor_request_schema"):
        validate_predictor_request(request)


@pytest.mark.parametrize("bad_path", ["/absolute/query.npz", "../query.npz", "a\\b.npz"])
def test_predictor_request_rejects_unsafe_artifact_paths(bad_path):
    request = _request()
    request["query_artifacts"][0]["relative_path"] = bad_path
    with pytest.raises(Phase2ContractError, match="artifact_relative_path_invalid"):
        validate_predictor_request(request)


def test_predictor_request_cross_checks_runtime_digests():
    request = _request()
    request["phase2_runtime_isolation_evidence"]["package_root_sha256"] = "9" * 64
    with pytest.raises(Phase2ContractError, match="cross_digest_mismatch"):
        validate_predictor_request(request)


def test_registered_class_handles_must_be_opaque_and_ordered():
    request = _request()
    request["registered_classes"][0]["class_handle"] = "14-10"
    with pytest.raises(Phase2ContractError, match="not_opaque"):
        validate_predictor_request(request)


def test_deprecated_ambiguous_field_never_satisfies_current_contract():
    request = _request()
    request["phase2_query_class_count_access"] = False
    with pytest.raises(Phase2ContractError, match="deprecated_field"):
        validate_phase2_contract(request)


def test_pre_run_evidence_does_not_claim_future_filesystem_ledger():
    request = _request()
    request["phase2_runtime_isolation_evidence"]["filesystem_access_audit_sha256"] = "e" * 64
    with pytest.raises(Phase2ContractError, match="unexpected_runtime_evidence"):
        validate_predictor_request(request)


def test_post_run_evidence_requires_access_and_prediction_seals():
    request = _request()
    with pytest.raises(Phase2ContractError, match="filesystem_access_audit_sha256"):
        validate_phase2_contract(request, evidence_phase="post_run")
    request["phase2_runtime_isolation_evidence"].update(
        {
            "filesystem_access_audit_sha256": "5" * 64,
            "filesystem_access_audit_status": "PASS",
            "prediction_artifact_sha256": "6" * 64,
            "prediction_seal_sha256": "7" * 64,
        }
    )
    validate_phase2_contract(request, evidence_phase="post_run")


def test_runtime_evidence_hashes_must_be_canonical_sha256():
    request = _request()
    request["phase2_runtime_isolation_evidence"]["preopen_audit_receipt_sha256"] = "pass"
    with pytest.raises(Phase2ContractError, match="invalid_runtime_evidence_sha256"):
        validate_phase2_contract(request)


def test_historical_missing_evidence_is_unverified_not_automatically_invalid():
    record = copy.deepcopy(PHASE2_FULL_CONTRACT)
    assert classify_legacy_phase2_record(record) == "UNVERIFIED_UNDER_CURRENT_PROTOCOL"
    record["clean_sample_access"] = True
    assert classify_legacy_phase2_record(record) == "PROTOCOL_INVALID_FOR_PHASE2"


def test_confirmed_role_or_quota_oracle_is_deployment_invalid():
    record = copy.deepcopy(PHASE2_FULL_CONTRACT)
    record["phase2_query_role_oracle_access"] = True
    assert classify_legacy_phase2_record(record) == "PROTOCOL_INVALID_FOR_DEPLOYMENT"
