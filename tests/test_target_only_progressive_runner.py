from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import cvsrffi.target_only_progressive_runner as runner_module

from cvsrffi.target_only_progressive_runner import (
    load_sf_tapft_bundle_strict,
    run_sf_tapft_no_query,
)
from cvsrffi.sf_tapft_phase1_binding import SFTAPFTPhase1Binding
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


def _phase1_bundle_config() -> dict[str, str]:
    return {
        "package_root": "formal-package",
        "detached_seal_path": "detached-seal.json",
        "expected_detached_seal_sha256": "1" * 64,
        "signature_envelope_path": "signature-envelope.json",
        "expected_signature_envelope_sha256": "2" * 64,
        "expected_checkpoint_lineage_sha256": "3" * 64,
        "expected_runtime_sha256": "4" * 64,
        "expected_component_pre_sign_content_root_sha256": "5" * 64,
        "expected_class_handle_binding_sha256": "6" * 64,
        "expected_parity_receipt_sha256": "7" * 64,
        "expected_generation_lock_sha256": "8" * 64,
        "expected_method_lock_sha256": "9" * 64,
        "expected_generation_config_sha256": "a" * 64,
        "expected_generation_code_sha256": "b" * 64,
        "expected_outer_content_root_sha256": "c" * 64,
    }


def _phase1_binding(*, handles: tuple[str, ...]) -> SFTAPFTPhase1Binding:
    return SFTAPFTPhase1Binding(
        outer_content_root_sha256="c" * 64,
        checkpoint_lineage_sha256="3" * 64,
        runtime_sha256="4" * 64,
        class_handle_binding_sha256="6" * 64,
        class_handles=handles,
        component_pre_sign_content_root_sha256="5" * 64,
    )


def _r0_config(checkpoint: Path, support: Path) -> dict[str, object]:
    return {
        "candidate_id": "SF_TAPFT_R0_SMOKE",
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_path": str(checkpoint),
        "support_path": str(support),
        "phase1_bundle": _phase1_bundle_config(),
        "sf_tapft": {
            "phase_steps": [1, 1, 1],
            "warmup_ratio": 0.0,
            "checkpoint_average_top_k": 1,
            "adapter_rank": 2,
            "seed": 23,
        },
    }


def test_r0_runner_rejects_out_of_range_support_label_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support = tmp_path / "support.npz"
    _write_support(support)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"loader-owned fixture")
    monkeypatch.setattr(
        runner_module,
        "load_sf_tapft_phase1_binding",
        lambda *_args, **_kwargs: _phase1_binding(handles=("tx0",)),
    )
    monkeypatch.setattr(
        runner_module,
        "fit_sf_tapft",
        lambda *_args, **_kwargs: pytest.fail("fit_sf_tapft must not run before label validation"),
    )

    with pytest.raises(ValueError, match="ordered Phase1 class registry"):
        run_sf_tapft_no_query(
            _r0_config(checkpoint, support),
            tmp_path / "output",
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        )


def test_r0_runner_copies_phase1_binding_into_receipt_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support = tmp_path / "support.npz"
    _write_support(support)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"loader-owned fixture")
    binding = _phase1_binding(handles=("tx0", "tx1"))
    monkeypatch.setattr(
        runner_module,
        "load_sf_tapft_phase1_binding",
        lambda *_args, **_kwargs: binding,
    )
    output = tmp_path / "output"
    receipt = run_sf_tapft_no_query(
        _r0_config(checkpoint, support),
        output,
        device="cpu",
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )

    assert receipt["phase1_binding"]["outer_content_root_sha256"] == "c" * 64
    assert receipt["phase1_binding"]["checkpoint_lineage_sha256"] == "3" * 64
    assert receipt["phase1_binding"]["runtime_sha256"] == "4" * 64
    assert receipt["phase1_binding"]["class_handle_binding_sha256"] == "6" * 64
    payload = torch.load(output / "sf_tapft_bundle.pt", map_location="cpu", weights_only=True)
    assert payload["phase1_binding"] == receipt["phase1_binding"]


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
    assert audit["capsule_id"] == "capsule-test"
    assert audit["split_id"] == "split-test"
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


def test_runner_accepts_validated_support_without_embedded_physical_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support = tmp_path / "support-minimal.npz"
    np.savez(
        support,
        received_iq=np.asarray(
            [[2.0, 0.0, 0.2, 0.0], [1.7, 0.1, 0.0, 0.2], [0.0, 2.0, 0.0, 0.2], [0.1, 1.8, 0.2, 0.0]],
            dtype=np.float32,
        ),
        support_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    config = {
        "candidate_id": "minimal-support",
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "checkpoint_path": str(tmp_path / "checkpoint.pth"),
        "support_path": str(support),
        "sf_tapft": {
            "phase_steps": [1, 1, 1],
            "warmup_ratio": 0.0,
            "checkpoint_average_top_k": 1,
            "adapter_rank": 2,
        },
    }
    monkeypatch.setattr(
        runner_module.torch,
        "from_numpy",
        lambda _array: (_ for _ in ()).throw(TypeError("simulated NumPy bridge mismatch")),
    )
    receipt = run_sf_tapft_no_query(
        config,
        tmp_path / "output-minimal",
        device="cpu",
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )
    assert receipt["status"] == "SMOKE_PASS"
    assert receipt["support_physical_id_origin"] == "validated_support_row_index"


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
