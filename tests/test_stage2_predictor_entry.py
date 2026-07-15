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
