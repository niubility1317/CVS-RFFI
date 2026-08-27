from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import cvsrffi.target_only_progressive_runner as runner_module

from cvsrffi.target_only_progressive_runner import (
    SF_TAPFT_CLEAN_SINGLE_BUNDLE_SCHEMA,
    load_sf_tapft_bundle_strict,
    load_sf_tapft_clean_single_bundle_strict,
    run_sf_tapft_deploy_no_query,
    run_sf_tapft_grouped_selection,
    run_sf_tapft_no_query,
)
from cvsrffi.sf_tapft_phase1_binding import SFTAPFTPhase1Binding
from cvsrffi.target_only_progressive_adapt import GroupedTargetCVSelector
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


def _expected_target_binding() -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "support_count": 4,
        "per_class_counts": [
            {"class_id": 0, "count": 2},
            {"class_id": 1, "count": 2},
        ],
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


def test_r0_no_query_receipt_carries_phase1_binding_but_v1_bundle_does_not(
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
    assert "phase1_binding" not in payload
    payload["phase1_binding"] = receipt["phase1_binding"]
    forged = tmp_path / "v1-with-extra-binding.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match="top-level allowlist mismatch"):
        load_sf_tapft_bundle_strict(
            forged,
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        )


def test_r0_grouped_selection_writes_strict_full_support_clean_single_bundle(
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
    monkeypatch.setattr(
        GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    output = tmp_path / "selection-output"

    config = _r0_config(checkpoint, support)
    config["sf_tapft"]["trainability_profile"] = "p1_head_norm"
    config["sf_tapft"]["oof_temperature_calibration"] = True
    config["sf_tapft"]["validation_steps"] = [1, 2, 3]
    receipt = run_sf_tapft_grouped_selection(
        config,
        output,
        device="cpu",
        folds=2,
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )

    bundle_path = output / "sf_tapft_clean_single_bundle.pt"
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "schema",
        "method",
        "permission",
        "model_role",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "base_checkpoint_path",
        "phase1_bundle",
        "phase1_binding",
        "config",
        "selected_phase_steps",
        "support_count",
        "per_class_counts",
        "fold0_as_final",
        "query_input_capability",
        "class_ids",
        "model_state",
        "head_state",
        "state_change_audit",
    }
    assert payload["schema"] == SF_TAPFT_CLEAN_SINGLE_BUNDLE_SCHEMA
    assert payload["model_role"] == "clean_single_full_support_refit"
    assert payload["selected_phase_steps"] == [1, 1, 1]
    assert payload["support_count"] == 4
    assert payload["per_class_counts"] == [
        {"class_id": 0, "count": 2},
        {"class_id": 1, "count": 2},
    ]
    assert payload["fold0_as_final"] is False
    assert payload["state_change_audit"]["training_sample_count"] == 4
    assert payload["state_change_audit"]["checkpoint_selection_role"] == "fixed_final_step"
    assert payload["state_change_audit"]["nonpermitted_changed_names"] == []
    assert payload["config"]["inference_temperature"] == pytest.approx(
        receipt["temperature_calibration"]["temperature"]
    )
    assert payload["config"]["validation_steps"] == ()
    assert receipt["oof_selection"]["selected"] == "adapted"
    assert receipt["final_full_support_refit"] == {
        "model_role": "clean_single_full_support_refit",
        "support_count": 4,
        "per_class_counts": [
            {"class_id": 0, "count": 2},
            {"class_id": 1, "count": 2},
        ],
        "selected_phase_steps": [1, 1, 1],
        "fold0_as_final": False,
        "checkpoint_selection_role": "fixed_final_step",
        "bundle_path": str(bundle_path),
        "delta_bundle_path": str(output / "sf_tapft_delta_bundle.pt"),
        "delta_bundle_bytes": (output / "sf_tapft_delta_bundle.pt").stat().st_size,
    }
    assert receipt["resource_audit"]["head_polish_steps"] == 0
    assert receipt["resource_audit"]["cached_head_forward_steps"] == 0
    assert receipt["resource_audit"]["trainable_delta_ema_decay"] == 0.0
    assert receipt["resource_audit"]["class_adaptive_rho"] == [0.5, 0.5]
    assert receipt["resource_audit"]["class_reliability"] == [0.0, 0.0]
    persisted_receipt = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert persisted_receipt["oof_selection"] == receipt["oof_selection"]
    assert persisted_receipt["final_full_support_refit"] == receipt["final_full_support_refit"]

    model, head, audit = load_sf_tapft_clean_single_bundle_strict(
        bundle_path,
        device="cpu",
        expected_target_binding=_expected_target_binding(),
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        phase1_binding_loader=lambda *_args, **_kwargs: binding,
    )
    assert audit["schema"] == SF_TAPFT_CLEAN_SINGLE_BUNDLE_SCHEMA
    assert audit["model_role"] == "clean_single_full_support_refit"
    assert audit["support_count"] == 4
    assert audit["fold0_as_final"] is False
    assert head.class_ids == (0, 1)
    assert head.scale == pytest.approx(
        payload["config"]["prototype_scale"]
        / payload["config"]["inference_temperature"]
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())

    delta_path = output / "sf_tapft_delta_bundle.pt"
    assert delta_path.stat().st_size < 10_000
    delta_model, delta_head, delta_audit = runner_module.load_sf_tapft_delta_bundle_strict(
        delta_path,
        device="cpu",
        expected_target_binding=_expected_target_binding(),
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )
    assert delta_audit["support_count"] == 4
    assert torch.allclose(delta_head.weight, head.weight, atol=1.0e-3, rtol=1.0e-3)
    for name in delta_audit["updated_parameter_names"]:
        assert torch.allclose(
            dict(delta_model.named_parameters())[name],
            dict(model.named_parameters())[name],
            atol=1.0e-3,
            rtol=1.0e-3,
        )

    payload["config"]["phase_steps"] = (1, 0, 0)
    payload["config"]["validation_steps"] = (1, 2, 3)
    payload["selected_phase_steps"] = [1, 0, 0]
    payload["state_change_audit"]["total_steps"] = 1
    payload["state_change_audit"]["phase_steps"] = [1, 0, 0]
    payload["state_change_audit"]["selected_checkpoint_steps"] = [1]
    historical = tmp_path / "historical-clean-single-with-research-validation.pt"
    torch.save(payload, historical)
    _, historical_head, historical_audit = load_sf_tapft_clean_single_bundle_strict(
        historical,
        device="cpu",
        expected_target_binding=_expected_target_binding(),
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        phase1_binding_loader=lambda *_args, **_kwargs: binding,
    )
    assert historical_audit["selected_phase_steps"] == (1, 0, 0)
    assert historical_head.scale == pytest.approx(head.scale)

    with pytest.raises(FileExistsError):
        run_sf_tapft_grouped_selection(
            _r0_config(checkpoint, support),
            output,
            device="cpu",
            folds=2,
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        )


def test_deploy_runner_fits_full_support_once_without_grouped_cv(
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
    monkeypatch.setattr(
        GroupedTargetCVSelector,
        "split",
        lambda *_args, **_kwargs: pytest.fail("deployment runner must not construct folds"),
    )
    config = _r0_config(checkpoint, support)
    config["candidate_id"] = "H6_DEPLOY_TEST"
    config["sf_tapft"].update(
        {
            "trainability_profile": "p1_head_norm",
            "norm_rules": [["t3", "weight_bias"]],
            "phase_steps": [2, 0, 0],
            "inference_temperature": 1.7,
            "hard_pair_weight": 0.03,
            "hard_pair_margin": 0.2,
        }
    )
    output = tmp_path / "deploy-output"

    receipt = run_sf_tapft_deploy_no_query(
        config,
        output,
        device="cpu",
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )

    assert receipt["status"] == "DEPLOY_ADAPT_COMPLETE"
    assert receipt["research_selection_executed"] is False
    assert receipt["folds"] == 0
    assert receipt["support_physical_sample_count"] == 4
    assert receipt["query_input_capability"] is False
    assert receipt["query_opened"] is False
    assert receipt["resource_audit"]["hard_pair_weight"] == pytest.approx(0.03)
    assert receipt["resource_audit"]["prefix_cache_build_forward_steps"] == 0
    assert receipt["delta_bundle_bytes"] < 10_000
    payload = torch.load(
        output / "sf_tapft_clean_single_bundle.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert payload["selected_phase_steps"] == [2, 0, 0]
    assert payload["fold0_as_final"] is False
    assert payload["config"]["validation_steps"] == ()
    assert payload["config"]["inference_temperature"] == pytest.approx(1.7)
    persisted = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert persisted == receipt

def test_clean_single_loader_accepts_pre_slimming_config_defaults(
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
    monkeypatch.setattr(
        GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    output = tmp_path / "selection-output"
    run_sf_tapft_grouped_selection(
        _r0_config(checkpoint, support),
        output,
        device="cpu",
        folds=2,
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )
    payload = torch.load(
        output / "sf_tapft_clean_single_bundle.pt", map_location="cpu", weights_only=True
    )
    for field in (
        "norm_scope",
        "norm_affine",
        "scheduler_reference_steps",
        "fast_tail_start_step",
        "fast_tail_steps",
        "fast_tail_lr_head_start",
        "fast_tail_lr_head_end",
        "fast_tail_lr_norm_start",
        "fast_tail_lr_norm_end",
        "head_polish_steps",
        "head_polish_lr",
        "trainable_delta_ema_decay",
        "use_class_adaptive_rho",
        "class_adaptive_rho_min",
        "class_adaptive_rho_max",
        "class_adaptive_rho_temperature",
        "head_anchor_weight",
    ):
        payload["config"].pop(field)
    legacy = tmp_path / "pre-slimming.pt"
    torch.save(payload, legacy)

    model, head, audit = load_sf_tapft_clean_single_bundle_strict(
        legacy,
        device="cpu",
        expected_target_binding=_expected_target_binding(),
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
        phase1_binding_loader=lambda *_args, **_kwargs: binding,
    )
    assert audit["support_count"] == 4
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())


@pytest.mark.parametrize(
    ("family", "mutate", "message"),
    [
        (
            "phase1_outer_root",
            lambda payload: payload["phase1_binding"].__setitem__("outer_content_root_sha256", "d" * 64),
            "Phase1 binding mismatch",
        ),
        (
            "phase1_checkpoint_lineage",
            lambda payload: payload["phase1_binding"].__setitem__("checkpoint_lineage_sha256", "d" * 64),
            "Phase1 binding mismatch",
        ),
        (
            "phase1_runtime",
            lambda payload: payload["phase1_binding"].__setitem__("runtime_sha256", "d" * 64),
            "Phase1 binding mismatch",
        ),
        (
            "phase1_ordered_classes",
            lambda payload: payload["phase1_binding"].__setitem__("class_handles", ["tx1", "tx0"]),
            "Phase1 binding mismatch",
        ),
        (
            "phase1_class_binding",
            lambda payload: payload["phase1_binding"].__setitem__("class_handle_binding_sha256", "d" * 64),
            "Phase1 binding mismatch",
        ),
        (
            "phase1_aggregate_component",
            lambda payload: payload["phase1_binding"].__setitem__(
                "component_pre_sign_content_root_sha256", "d" * 64
            ),
            "Phase1 binding mismatch",
        ),
        (
            "target_protocol",
            lambda payload: payload.__setitem__("protocol_schema", "p2_other"),
            "target data binding mismatch",
        ),
        (
            "target_status",
            lambda payload: payload.__setitem__("phase2_data_status", "UNVALIDATED"),
            "target data binding mismatch",
        ),
        (
            "target_capsule",
            lambda payload: payload.__setitem__("capsule_id", "other-capsule"),
            "trusted target binding mismatch",
        ),
        (
            "target_split",
            lambda payload: payload.__setitem__("split_id", "other-split"),
            "trusted target binding mismatch",
        ),
    ],
)
def test_clean_single_loader_rejects_each_binding_family_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    mutate,
    message: str,
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
    monkeypatch.setattr(
        GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    output = tmp_path / "selection-output"
    run_sf_tapft_grouped_selection(
        _r0_config(checkpoint, support),
        output,
        device="cpu",
        folds=2,
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )
    payload = torch.load(
        output / "sf_tapft_clean_single_bundle.pt", map_location="cpu", weights_only=True
    )
    mutate(payload)
    mutated = tmp_path / f"mutated-{family}.pt"
    torch.save(payload, mutated)

    with pytest.raises(ValueError, match=message):
        load_sf_tapft_clean_single_bundle_strict(
            mutated,
            device="cpu",
            expected_target_binding=_expected_target_binding(),
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
            phase1_binding_loader=lambda *_args, **_kwargs: binding,
        )


def test_clean_single_loader_rejects_internally_consistent_forged_target_binding(
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
    monkeypatch.setattr(
        GroupedTargetCVSelector,
        "choose",
        staticmethod(lambda *, frozen, adapted: "adapted"),
    )
    output = tmp_path / "selection-output"
    run_sf_tapft_grouped_selection(
        _r0_config(checkpoint, support),
        output,
        device="cpu",
        folds=2,
        checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
    )
    payload = torch.load(
        output / "sf_tapft_clean_single_bundle.pt", map_location="cpu", weights_only=True
    )
    payload["capsule_id"] = "forged-capsule"
    payload["split_id"] = "forged-split"
    payload["support_count"] = 6
    payload["per_class_counts"] = [
        {"class_id": 0, "count": 3},
        {"class_id": 1, "count": 3},
    ]
    payload["state_change_audit"]["training_sample_count"] = 6
    forged = tmp_path / "internally-consistent-forged-target.pt"
    torch.save(payload, forged)

    with pytest.raises(ValueError, match="trusted target binding mismatch"):
        load_sf_tapft_clean_single_bundle_strict(
            forged,
            device="cpu",
            expected_target_binding=_expected_target_binding(),
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
            phase1_binding_loader=lambda *_args, **_kwargs: binding,
        )

    with pytest.raises(TypeError, match="expected_target_binding"):
        load_sf_tapft_clean_single_bundle_strict(
            forged,
            device="cpu",
            checkpoint_loader=lambda _path, *, device: copy.deepcopy(_ToyModel()).to(device),
            phase1_binding_loader=lambda *_args, **_kwargs: binding,
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
