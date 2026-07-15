from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import cvsrffi.stage2_predictor_entry as entry
from cvsrffi.phase2_runtime_contract import Phase2ContractError


def test_invalid_request_is_rejected_before_any_package_open(monkeypatch) -> None:
    opened = []

    def forbidden_open(*_args, **_kwargs):
        opened.append(True)
        raise AssertionError("package was opened before request validation")

    monkeypatch.setattr(entry, "preflight_stage2_predictor_package", forbidden_open)
    with pytest.raises(Phase2ContractError):
        entry.prepare_role_blind_prediction(
            {"schema_version": "invalid"},
            predictor_package_root=Path("does-not-exist"),
            detached_seal_path=Path("does-not-exist.seal"),
            expected_seal_sha256="0" * 64,
            device=torch.device("cpu"),
        )
    assert opened == []


def test_strict_entry_source_has_no_dataset_training_or_scorer_import() -> None:
    source = inspect.getsource(entry).lower()
    for forbidden in (
        "leo_weak_cache",
        "dataset_wigsig",
        "class_incremental",
        "train_ssdg",
        "scoring_sidecar",
        "paper_reproduction",
    ):
        assert forbidden not in source


def test_v2_resource_receipt_counts_deployment_and_formal_head_state() -> None:
    receipt = {"persistent_state_bytes": 100_000}
    state = {
        "candidate_head_deployment_state_bytes_fp16": 14_768,
        "candidate_head_evaluation_comparator_state_bytes_fp16": 3_156,
        "candidate_head_formal_dual_stream_state_bytes_fp16": 17_924,
        "candidate_head_deployment_live_array_bytes": 29_536,
        "candidate_head_evaluation_comparator_live_array_bytes": 6_312,
        "candidate_head_formal_dual_stream_live_array_bytes": 35_848,
        "candidate_head_live_array_bytes": 35_848,
        "candidate_head_evidence_deployment_state_bytes_fp16": 104,
        "candidate_head_evidence_evaluation_comparator_state_bytes_fp16": 24,
        "candidate_head_evidence_formal_dual_stream_state_bytes_fp16": 128,
        "persistent_state_bytes_total": 114_768,
        "formal_dual_stream_persistent_state_bytes_total": 117_924,
    }

    entry._add_formal_head_resource_accounting(receipt, state)

    assert receipt["adapter_persistent_state_bytes"] == 100_000
    assert receipt["persistent_state_bytes"] == 114_768
    assert receipt["candidate_head_evidence_deployment_state_bytes_fp16"] == 104
    assert receipt["candidate_head_evidence_formal_dual_stream_state_bytes_fp16"] == 128
    assert receipt["candidate_head_deployment_live_array_bytes"] == 29_536
    assert receipt["candidate_head_evaluation_comparator_live_array_bytes"] == 6_312
    assert receipt["candidate_head_formal_dual_stream_live_array_bytes"] == 35_848
    assert receipt["formal_dual_stream_persistent_state_bytes_total"] == 117_924


def test_v1_resource_receipt_schema_is_unchanged() -> None:
    receipt = {"persistent_state_bytes": 100_000}
    entry._add_formal_head_resource_accounting(receipt, {"candidate_after": {}})
    assert receipt == {"persistent_state_bytes": 100_000}
