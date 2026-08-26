from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi.target_only_progressive_runner import (
    load_sf_tapft_bundle_strict,
    run_sf_tapft_no_query,
)
from test_target_only_progressive_adapt import _ToyModel


def _write_support(path: Path) -> None:
    np.savez(
        path,
        received_iq=np.asarray(
            [
                [2.0, 0.0, 0.2, 0.0],
                [1.7, 0.1, 0.0, 0.2],
                [0.0, 2.0, 0.0, 0.2],
                [0.1, 1.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
        support_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        support_physical_ids=np.asarray(["p0", "p1", "p2", "p3"]),
        support_groups=np.asarray(["g0", "g1", "g0", "g1"]),
    )


def test_no_query_runner_writes_nonformal_bundle_and_consumer_can_reload(tmp_path: Path) -> None:
    support = tmp_path / "support.npz"
    _write_support(support)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"loader-owned fixture")
    output = tmp_path / "output"
    base = _ToyModel()

    config = {
        "candidate_id": "SF_TAPFT_V1_SMOKE",
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_path": str(checkpoint),
        "support_path": str(support),
        "sf_tapft": {
            "phase_steps": [1, 1, 1],
            "warmup_ratio": 0.0,
            "checkpoint_average_top_k": 1,
            "adapter_rank": 2,
            "seed": 23,
        },
    }

    receipt = run_sf_tapft_no_query(
        config,
        output,
        device="cpu",
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(base).to(device),
    )
    assert receipt["status"] == "SMOKE_PASS"
    assert receipt["permission"] == "DIAGNOSTIC_NON_FORMAL"
    assert receipt["protocol_schema"] == "p2_min_v1"
    assert receipt["phase2_data_status"] == "VALIDATED_ONCE"
    assert receipt["source_opened"] is False
    assert receipt["query_input_capability"] is False
    assert receipt["query_opened"] is False
    assert receipt["target_eval_opened"] is False
    assert receipt["total_steps"] == 3
    assert (output / "sf_tapft_bundle.pt").is_file()
    assert json.loads((output / "smoke.json").read_text(encoding="utf-8"))["status"] == "SMOKE_PASS"

    model, head, audit = load_sf_tapft_bundle_strict(
        output / "sf_tapft_bundle.pt",
        device="cpu",
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(base).to(device),
    )
    assert audit["schema"] == "cvs.sf_tapft.v1"
    assert audit["permission"] == "DIAGNOSTIC_NON_FORMAL"
    assert audit["checkpoint_selection_role"] == "target_train_loss_single"
    assert head.class_ids == (0, 1)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())

    with pytest.raises(FileExistsError):
        run_sf_tapft_no_query(
            config,
            output,
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(base).to(device),
        )


def test_runner_rejects_formal_permission_and_unknown_config_fields(tmp_path: Path) -> None:
    support = tmp_path / "support.npz"
    _write_support(support)
    base = _ToyModel()
    config = {
        "candidate_id": "bad",
        "method": "sf_tapft_v1",
        "permission": "FORMAL_PHASE2",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
        "support_path": str(support),
        "sf_tapft": {"phase_steps": [1, 1, 1], "unknown": 1},
    }
    with pytest.raises(ValueError, match="DIAGNOSTIC_NON_FORMAL"):
        run_sf_tapft_no_query(
            config,
            tmp_path / "output",
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(base).to(device),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_schema", "p2_other", "p2_min_v1"),
        ("phase2_data_status", "UNVALIDATED", "VALIDATED_ONCE"),
        ("capsule_id", "", "non-empty"),
        ("split_id", "", "non-empty"),
    ],
)
def test_runner_rejects_invalid_phase2_data_binding(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    support = tmp_path / "support.npz"
    _write_support(support)
    config = {
        "candidate_id": "bad-binding",
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
        "support_path": str(support),
        "sf_tapft": {"phase_steps": [1, 1, 1]},
    }
    config[field] = value
    with pytest.raises(ValueError, match=message):
        run_sf_tapft_no_query(
            config,
            tmp_path / "output",
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        )
