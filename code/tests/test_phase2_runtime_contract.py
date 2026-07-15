from __future__ import annotations

import copy

import pytest

from cvsrffi.phase2_runtime_contract import (
    PHASE2_FULL_CONTRACT,
    Phase2ContractError,
    classify_legacy_phase2_record,
    validate_phase2_contract,
    validate_predictor_request,
)


def _evidence():
    return {
        "sealed_inference_package_sha256": "a" * 64,
        "package_root_sha256": "b" * 64,
        "runtime_code_sha256": "c" * 64,
        "artifact_member_allowlist_sha256": "d" * 64,
        "filesystem_access_audit_sha256": "e" * 64,
        "os_isolation_mode": "container_readonly_mounts",
        "preopen_audit_status": "PASS",
        "predict_score_process_isolation": True,
    }


def _request():
    return {
        **PHASE2_FULL_CONTRACT,
        "schema_version": "cvs.phase2.predict_request.v1",
        "request_id": "req-1",
        "row_id": "row-1",
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "k_shot": 1,
        "satellite_seed": 713101,
        "candidate_lock_sha256": "1" * 64,
        "package_root_sha256": "2" * 64,
        "runtime_code_sha256": "3" * 64,
        "registered_class_count": 2,
        "registered_classes": [
            {"class_index": 0, "class_label": "opaque-class-0"},
            {"class_index": 1, "class_label": "opaque-class-1"},
        ],
        "support_artifact": "artifacts/support.npz",
        "query_artifact": "artifacts/query.npz",
        "checkpoint_artifact": "artifacts/checkpoint.bin",
        "adapter_artifact": "artifacts/adapter.bin",
        "head_artifact": "artifacts/head.npz",
        "tta_policy": {"base_views": 1, "max_views": 5},
        "tta_policy_sha256": "4" * 64,
        "output_contract": "cvs.phase2.prediction.v1",
        "phase2_runtime_isolation_evidence": _evidence(),
    }


def test_full_contract_has_three_base_four_clean_and_five_query_fields():
    assert len(PHASE2_FULL_CONTRACT) == 12
    assert PHASE2_FULL_CONTRACT["phase2_query_true_batch_class_count_access"] is False


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


def test_deprecated_ambiguous_field_never_satisfies_current_contract():
    request = _request()
    request["phase2_query_class_count_access"] = False
    with pytest.raises(Phase2ContractError, match="deprecated_field"):
        validate_phase2_contract(request)


def test_runtime_reachability_false_requires_runtime_evidence():
    request = _request()
    request.pop("phase2_runtime_isolation_evidence")
    with pytest.raises(Phase2ContractError, match="runtime_isolation_evidence"):
        validate_phase2_contract(request)


def test_runtime_evidence_hashes_must_be_canonical_sha256():
    request = _request()
    request["phase2_runtime_isolation_evidence"]["filesystem_access_audit_sha256"] = "pass"
    with pytest.raises(Phase2ContractError, match="invalid_runtime_evidence_sha256"):
        validate_phase2_contract(request)


def test_historical_missing_evidence_is_unverified_not_automatically_invalid():
    record = copy.deepcopy(PHASE2_FULL_CONTRACT)
    assert classify_legacy_phase2_record(record) == "PHASE2_RUNTIME_ISOLATION_UNVERIFIED"
    record["clean_sample_access"] = True
    assert classify_legacy_phase2_record(record) == "PROTOCOL_INVALID_FOR_PHASE2"
