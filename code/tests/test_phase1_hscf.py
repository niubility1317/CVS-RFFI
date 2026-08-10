from __future__ import annotations

import ast
import gc
import inspect
import re
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import SSDG.train_ssdg as train_ssdg  # noqa: E402
from cvsrffi.phase1_hscf import (  # noqa: E402
    FROZEN_HSCF_BATCH_SIZE,
    FROZEN_HSCF_CLASS_COUNT,
    FROZEN_HSCF_CLASS_IDS,
    FROZEN_HSCF_GLOBAL_DENOMINATOR,
    FROZEN_HSCF_LAMBDA,
    FROZEN_HSCF_OPTIMIZER_TYPE,
    FROZEN_HSCF_SCENARIOS,
    HSCF_RECEIPT_SCHEMA,
    HSCFConfig,
    HSCFConfigurationError,
    HSCFRuntimeError,
    add_hscf_to_loss,
    bind_hscf_optimizer_initial_state,
    bind_hscf_source_data_order,
    finalize_hscf_amp_overflow_skip,
    hscf_aux_gradient_audit,
    hscf_config_receipt,
    hscf_loss,
    hscf_scaled_backward_and_classify,
    hscf_shared_encoder_and_head_parameters,
    release_hscf_retained_graph_roots,
    record_hscf_material_nonfinite_receipt,
    remap_hscf_local_labels_to_head_rows,
    resolve_hscf_classifier_weight,
    resolve_hscf_local_head_class_binding,
    strict_hscf_warm_start,
    update_hscf_common_batch_sequence_receipt,
    update_hscf_gradient_audit_receipt,
    update_hscf_amp_overflow_receipt,
    update_hscf_optimizer_step_receipt,
    update_hscf_receipt,
    validate_hscf_args,
    validate_hscf_binding,
    validate_hscf_terminal_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_hscf12_20260810.sh"


class _RecoveringScaler:
    """CPU scaler double with the public skip/backoff behavior used by HSCF."""

    def __init__(self, scale: float = 1.0e5) -> None:
        self._scale = float(scale)
        self._found_nonfinite = False
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, value: torch.Tensor) -> torch.Tensor:
        self.scale_calls += 1
        return value * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscale_calls += 1
        self._found_nonfinite = False
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.grad.div_(self._scale)
                if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
                    self._found_nonfinite = True

    def step(self, optimizer: torch.optim.Optimizer):
        self.step_calls += 1
        if self._found_nonfinite:
            return None
        return optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self._found_nonfinite:
            self._scale *= 0.5


class _SavedTensorToken:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor: torch.Tensor) -> None:
        # The token observes ownership by autograd's saved-tensor slot without
        # itself pointing back into the graph under test.
        self.tensor = tensor.detach()


class _FiniteForwardNonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("nan"))


def _frozen_args(*, enabled: bool = True, epochs: int = 40, batch_size: int = 128) -> SimpleNamespace:
    """Arguments consumed by the frozen HSCF validator, with all peers off."""

    return SimpleNamespace(
        phase1_hscf_frozen_mode=True,
        phase1_hscf_enabled=enabled,
        lambda_hscf=FROZEN_HSCF_LAMBDA if enabled else 0.0,
        batch_size=batch_size,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        amp=True,
        id_feature_key="feat_joint",
        epochs=epochs,
        label_epochs=epochs,
        pseudo_epochs=0,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(FROZEN_HSCF_SCENARIOS),
        sat_view_schedule="",
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


def _disable_peer_flags(args: SimpleNamespace) -> None:
    peers = (
        "ccpc_leo",
        "pamr",
        "cb_sfce",
        "gd_proto_nll",
        "icmt",
        "cagm",
        "rcrmd",
        "rcat",
        "recte",
        "cp_sfce",
    )
    for peer in peers:
        setattr(args, f"phase1_{peer}_frozen_mode", False)
        setattr(args, f"phase1_{peer}_enabled", False)
        setattr(args, f"lambda_{peer}", 0.0)


class _BindingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(4, 4, bias=False)
        self.id_backbone.cls_head = torch.nn.Module()
        # Bias is optional in the frozen contract; using the bias-free exact
        # head keeps the expected head-bias VJP scope explicitly absent.
        self.id_backbone.cls_head.head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.id_backbone.encoder.weight.copy_(torch.eye(4))
            self.id_backbone.cls_head.head.weight.copy_(torch.eye(4))

    def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        z_id = self.id_backbone.encoder(x)
        logits = self.id_backbone.cls_head.head(z_id)
        return {"z_id": z_id, "z_id_key": "feat_joint", "tx_logits": logits}


def _labels() -> torch.Tensor:
    return torch.arange(FROZEN_HSCF_BATCH_SIZE, dtype=torch.long) % FROZEN_HSCF_CLASS_COUNT


def _metadata() -> dict[str, torch.Tensor]:
    return {
        "base_index": torch.arange(FROZEN_HSCF_BATCH_SIZE),
        "sig_i": torch.arange(FROZEN_HSCF_BATCH_SIZE) + 10000,
    }


def _config(*, enabled: bool = True) -> HSCFConfig:
    return HSCFConfig(True, enabled, FROZEN_HSCF_LAMBDA if enabled else 0.0)


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = hscf_config_receipt(_config(enabled=enabled))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "source_receiver_ids": [],
            "optimizer_type": FROZEN_HSCF_OPTIMIZER_TYPE,
            "optimizer_initial_state_sha256": "e" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": list(FROZEN_HSCF_CLASS_IDS),
            "common_l_base_head_input_path_verified": True,
            "common_batch_sequence_sha256": "f" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scene: 0 for scene in FROZEN_HSCF_SCENARIOS},
        }
    )
    return receipt


def _pending_hscf_amp_receipt(*, batch_index: int = 1) -> dict[str, object]:
    receipt = hscf_config_receipt(_config(enabled=True))
    receipt["hscf_g_batch_aux"] = [
        {
            "epoch": 1,
            "batch_index": int(batch_index),
            "scenario": FROZEN_HSCF_SCENARIOS[0],
        }
    ]
    return receipt


def _append_pending_hscf_amp_batch(receipt: dict[str, object], *, batch_index: int) -> dict[str, object]:
    result = dict(receipt)
    events = list(result["hscf_g_batch_aux"])
    events.append(
        {
            "epoch": 1,
            "batch_index": int(batch_index),
            "scenario": FROZEN_HSCF_SCENARIOS[0],
        }
    )
    result["hscf_g_batch_aux"] = events
    return result


def _bind_common(receipt: dict[str, object], *, index: int, scenario: str) -> dict[str, object]:
    return update_hscf_common_batch_sequence_receipt(
        receipt,
        epoch=1,
        batch_index=index,
        scenario=scenario,
        source_tx_labels=_labels(),
        metadata=_metadata(),
    )


def _build_g_terminal_receipt(
    *, include_effective_optimizer_steps: bool = True
) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(23)
    model = _BindingModel().train()
    labels = _labels()
    clean_input = torch.randn(FROZEN_HSCF_BATCH_SIZE, 4)
    leo_input = clean_input + 0.25 * torch.randn(FROZEN_HSCF_BATCH_SIZE, 4)
    out_clean = model.paired_output(clean_input)
    out_leo = model.paired_output(leo_input)
    validate_hscf_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        expected_class_ids=FROZEN_HSCF_CLASS_IDS,
    )
    aux, info = hscf_loss(out_clean["tx_logits"], out_leo["tx_logits"])
    groups = hscf_shared_encoder_and_head_parameters(model)
    audit = hscf_aux_gradient_audit(aux, out_clean["tx_logits"], out_leo["tx_logits"], groups)
    receipt = _sealed_receipt(enabled=True)
    for index, scenario in enumerate(FROZEN_HSCF_SCENARIOS, start=1):
        receipt = _bind_common(receipt, index=index, scenario=scenario)
        receipt = update_hscf_receipt(
            receipt, info, scenario=scenario, epoch=1, batch_index=index
        )
        receipt = update_hscf_gradient_audit_receipt(receipt, audit, scenario=scenario)
        if include_effective_optimizer_steps:
            receipt = update_hscf_optimizer_step_receipt(receipt)
    return receipt, info


def test_fixed_constants_and_manual_double_centering() -> None:
    assert FROZEN_HSCF_BATCH_SIZE == 128
    assert FROZEN_HSCF_CLASS_COUNT == 4
    assert FROZEN_HSCF_GLOBAL_DENOMINATOR == 512
    clean = torch.zeros((128, 4), requires_grad=True)
    leo = torch.zeros((128, 4), requires_grad=True)
    with torch.no_grad():
        leo[0, 0] = 1.0
    loss, info = hscf_loss(clean, leo)
    # P[1,0,0,0]=(3/4,-1/4,-1/4,-1/4), then batch-centering gives
    # sum_i ||r_i||^2=(127/128)*(3/4); the fixed denominator is 128*4.
    expected = (127.0 / 128.0) * (3.0 / 4.0) / 512.0
    assert float(loss.detach()) == pytest.approx(expected, abs=2e-8)
    assert info["rows"] == 128 and info["class_count"] == 4
    assert info["global_denominator"] == 512 and info["fixed_scale"] == pytest.approx(1.0 / 512.0)
    assert info["positive_components"] > 0 and info["no_active_renormalization"] is True


def test_zero_set_common_centered_configuration_and_clean_stopgrad() -> None:
    clean = torch.zeros((128, 4), requires_grad=True)
    leo = torch.zeros((128, 4), requires_grad=True)
    with torch.no_grad():
        leo.add_(torch.tensor([1.0, -1.0, 0.0, 0.0]))
    loss, info = hscf_loss(clean, leo)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert info["positive_components"] == 0 and info["positive_batch"] is False
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) == 0

    zero = torch.zeros((128, 4), requires_grad=True)
    zero_loss, zero_info = hscf_loss(zero, zero.clone().requires_grad_(True))
    assert float(zero_loss.detach()) == pytest.approx(0.0, abs=1e-12)
    assert zero_info["positive_components"] == 0


@pytest.mark.parametrize(
    "clean_shape,leo_shape",
    [((127, 4), (127, 4)), ((128, 5), (128, 5)), ((128, 4), (128, 5))],
)
def test_wrong_batch_or_local4_shape_fails_closed(clean_shape: tuple[int, int], leo_shape: tuple[int, int]) -> None:
    with pytest.raises(HSCFRuntimeError, match="128|local4|shape"):
        hscf_loss(torch.zeros(clean_shape), torch.zeros(leo_shape))


def test_nonfinite_logits_fail_closed() -> None:
    clean = torch.zeros((128, 4))
    leo = torch.zeros((128, 4))
    clean[0, 0] = float("nan")
    with pytest.raises(HSCFRuntimeError, match="non-finite"):
        hscf_loss(clean, leo)
    leo[0, 0] = float("inf")
    with pytest.raises(HSCFRuntimeError, match="non-finite"):
        hscf_loss(torch.zeros((128, 4)), leo)


def test_class_and_batch_centering_invariance_and_order_pairing() -> None:
    torch.manual_seed(7)
    clean = torch.randn(128, 4)
    leo = torch.randn(128, 4)
    direct, _ = hscf_loss(clean, leo)
    common = torch.tensor([2.0, -1.0, 3.0, 4.0])
    shifted, _ = hscf_loss(clean + common, leo + common)
    assert torch.allclose(direct, shifted, atol=1e-6)
    order = torch.randperm(128)
    reordered, _ = hscf_loss(clean.index_select(0, order), leo.index_select(0, order))
    assert torch.allclose(direct, reordered, atol=1e-6)

    receipt = _sealed_receipt(enabled=False)
    bound = _bind_common(receipt, index=1, scenario=FROZEN_HSCF_SCENARIOS[0])
    event = bound["hscf_common_batches"][0]
    assert event["same_physical_clean_leo"] is True
    assert event["same_order_clean_leo"] is True
    assert event["fixed_batch_size"] == 128 and event["fixed_local_class_count"] == 4
    assert event["global_denominator"] == 512
    bad = dict(receipt)
    with pytest.raises(HSCFRuntimeError, match="metadata|physical"):
        update_hscf_common_batch_sequence_receipt(
            bad,
            epoch=1,
            batch_index=1,
            scenario=FROZEN_HSCF_SCENARIOS[0],
            source_tx_labels=_labels(),
            metadata={},
        )
    with pytest.raises(HSCFRuntimeError, match="sequence"):
        update_hscf_common_batch_sequence_receipt(
            receipt,
            epoch=1,
            batch_index=2,
            scenario=FROZEN_HSCF_SCENARIOS[2],
            source_tx_labels=_labels(),
            metadata=_metadata(),
        )


def test_common_control_vs_g_auxiliary_and_three_scene_terminal_closure() -> None:
    control = _sealed_receipt(enabled=False)
    for index, scenario in enumerate(FROZEN_HSCF_SCENARIOS, start=1):
        control = _bind_common(control, index=index, scenario=scenario)
    c_terminal = validate_hscf_terminal_receipt(control)
    assert c_terminal["schema"] == HSCF_RECEIPT_SCHEMA
    assert c_terminal["hscf_terminal_contract_passed"] is True
    assert c_terminal["hscf_terminal_contract"].startswith("CONTROL_ARM")
    assert int(c_terminal["hscf_batches"]) == 0 and float(c_terminal["hscf_loss_sum"]) == 0.0
    assert c_terminal["hscf_effective_optimizer_steps"] == 0
    assert c_terminal["hscf_amp_overflow_raw_finite_batches"] == 0

    receipt, info = _build_g_terminal_receipt()
    assert info["positive_batch"] is True
    terminal = validate_hscf_terminal_receipt(receipt)
    assert terminal["hscf_terminal_contract_passed"] is True
    assert int(terminal["hscf_total_rows"]) == 3 * 128
    assert set(terminal["hscf_scenes"]) == set(FROZEN_HSCF_SCENARIOS)
    assert all(terminal["hscf_scenes"][scene]["positive_batches"] > 0 for scene in FROZEN_HSCF_SCENARIOS)
    assert set(terminal["hscf_gradient_audit_scenes"]) == set(FROZEN_HSCF_SCENARIOS)
    assert terminal["hscf_optimizer_step_attempts"] == terminal["hscf_batches"]
    assert terminal["hscf_effective_optimizer_steps"] == terminal["hscf_batches"]
    assert len(terminal["hscf_optimizer_events"]) == terminal["hscf_batches"]
    assert all(event["action"] == "EFFECTIVE_OPTIMIZER_STEP" for event in terminal["hscf_optimizer_events"])


def test_raw_finite_scaled_overflow_uses_one_gradscaler_skip_backoff_without_update() -> None:
    torch.manual_seed(29)
    model = _BindingModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = _RecoveringScaler()
    logits = model.paired_output(torch.ones((128, 4)))["tx_logits"]
    # The raw float32 gradient is finite; only multiplication by the captured
    # AMP scale overflows the backward buffer.
    loss = logits.sum() * 1.0e33
    before = [parameter.detach().clone() for parameter in model.parameters()]
    overflow = hscf_scaled_backward_and_classify(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
    )
    assert overflow["amp_overflow_detected"] is True
    assert overflow["amp_overflow_recoverable"] is True
    assert overflow["amp_overflow_kind"] == "COMBINED_SCALED_OVERFLOW_RAW_FINITE"
    assert overflow["scaled_backward_count"] == 1
    assert overflow["optimizer_unscale_count"] == 1
    assert scaler.scale_calls == 1 and scaler.unscale_calls == 1
    finalized = finalize_hscf_amp_overflow_skip(
        optimizer=optimizer,
        scaler=scaler,
        overflow=overflow,
    )
    assert finalized["optimizer_state_unchanged"] is True
    assert finalized["optimizer_step_applied"] is False
    assert 0.0 < finalized["post_scale"] < finalized["pre_scale"]
    assert all(torch.equal(before_value, after.detach()) for before_value, after in zip(before, model.parameters()))
    receipt = update_hscf_amp_overflow_receipt(
        _pending_hscf_amp_receipt(),
        overflow=overflow,
        finalized_skip=finalized,
    )
    assert receipt["hscf_amp_overflow_raw_finite_batches"] == 1
    assert receipt["hscf_optimizer_step_attempts"] == 1
    assert receipt["hscf_effective_optimizer_steps"] == 0
    assert receipt["hscf_last_amp_overflow"]["optimizer_step_skipped"] is True
    assert receipt["hscf_persistent_amp_overflow"] is True


def test_normal_hscf_batch_uses_no_raw_material_vjp_or_second_backward(monkeypatch) -> None:
    torch.manual_seed(30)
    model = _BindingModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    loss = model.paired_output(torch.ones((128, 4)))["tx_logits"].square().mean()

    def _unexpected_raw_vjp(*args, **kwargs):
        raise AssertionError("normal HSCF batch must not run a raw material VJP")

    monkeypatch.setattr(torch.autograd, "grad", _unexpected_raw_vjp)
    scaler = _RecoveringScaler(scale=1.0)
    info = hscf_scaled_backward_and_classify(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
    )
    assert info["amp_overflow_detected"] is False
    assert info["scaled_backward_count"] == 1
    assert info["optimizer_unscale_count"] == 1
    assert scaler.scale_calls == 1 and scaler.unscale_calls == 1
    helper_source = inspect.getsource(hscf_scaled_backward_and_classify)
    assert helper_source.count(".backward(") == 1


@pytest.mark.parametrize("scaled_overflow", [False, True])
def test_retained_graph_release_drops_saved_tensors_before_next_forward(
    scaled_overflow: bool,
) -> None:
    saved_tokens: list[weakref.ReferenceType[_SavedTensorToken]] = []

    def pack(tensor: torch.Tensor) -> _SavedTensorToken:
        token = _SavedTensorToken(tensor)
        saved_tokens.append(weakref.ref(token))
        return token

    def unpack(token: _SavedTensorToken) -> torch.Tensor:
        return token.tensor

    def run_one_batch() -> tuple[dict[str, object], dict[str, object]]:
        torch.manual_seed(300 + int(scaled_overflow))
        model = _BindingModel().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = _RecoveringScaler(scale=1.0e5 if scaled_overflow else 1.0)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            logits = model.paired_output(torch.ones((128, 4)))["tx_logits"]
            loss = logits.sum() * 1.0e33 if scaled_overflow else logits.square().mean()
            backward_info = hscf_scaled_backward_and_classify(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss=loss,
            )
            if scaled_overflow:
                finalize_hscf_amp_overflow_skip(
                    optimizer=optimizer,
                    scaler=scaler,
                    overflow=backward_info,
                )
            else:
                scaler.step(optimizer)
                scaler.update()
        roots = {"out_l": {"tx_logits": logits}, "loss": loss}
        del logits, loss
        return roots, backward_info

    roots, backward_info = run_one_batch()
    assert saved_tokens
    assert backward_info["amp_overflow_detected"] is scaled_overflow
    if not scaled_overflow:
        # retain_graph=True must still own saved activations before the explicit
        # train-loop boundary; otherwise this test would be vacuous.
        assert any(reference() is not None for reference in saved_tokens)
    release_hscf_retained_graph_roots(roots)
    assert roots == {}
    # Production performs no forced GC or allocator flush.  CPython refcount
    # release at the clear boundary must be sufficient before the next forward.
    assert all(reference() is None for reference in saved_tokens)
    gc.collect()
    assert all(reference() is None for reference in saved_tokens)


def test_raw_and_material_nonfinite_paths_remain_fail_closed() -> None:
    torch.manual_seed(31)
    model = _BindingModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    logits = model.paired_output(torch.ones((128, 4)))["tx_logits"]
    overflow = hscf_scaled_backward_and_classify(
        model=model,
        optimizer=optimizer,
        scaler=_RecoveringScaler(),
        loss=_FiniteForwardNonfiniteBackward.apply(logits).sum(),
    )
    assert overflow["amp_overflow_detected"] is True
    assert overflow["amp_overflow_recoverable"] is False
    assert overflow["amp_overflow_kind"] == "COMBINED_RAW_NONFINITE_OR_DISCONNECTED"
    assert overflow["raw_material_nonfinite_parameter_names"]
    with pytest.raises(HSCFRuntimeError, match="raw-finite"):
        finalize_hscf_amp_overflow_skip(
            optimizer=optimizer,
            scaler=_RecoveringScaler(),
            overflow=overflow,
        )
    receipt = update_hscf_amp_overflow_receipt(
        _pending_hscf_amp_receipt(),
        overflow=overflow,
    )
    assert receipt["hscf_amp_overflow_raw_nonfinite_batches"] == 1

    material_receipt = record_hscf_material_nonfinite_receipt(
        hscf_config_receipt(_config(enabled=True)),
        reason="total_loss_nonfinite",
    )
    assert material_receipt["hscf_amp_overflow_material_nonfinite_batches"] == 1
    material_model = _BindingModel().train()
    with pytest.raises(HSCFRuntimeError, match="material loss is non-finite"):
        hscf_scaled_backward_and_classify(
            model=material_model,
            optimizer=torch.optim.AdamW(material_model.parameters(), lr=1e-3),
            scaler=_RecoveringScaler(),
            loss=torch.tensor(float("nan"), requires_grad=True),
        )


def test_persistent_overflow_and_zero_effective_steps_are_terminal_rejections() -> None:
    overflow = {
        "amp_overflow_detected": True,
        "amp_overflow_recoverable": True,
        "amp_overflow_kind": "COMBINED_SCALED_OVERFLOW_RAW_FINITE",
        "raw_material_vjp_finite": True,
        "raw_material_nonfinite_parameter_names": (),
        "raw_material_missing_parameter_names": (),
        "scaled_nonfinite_parameter_names": ("id_backbone.encoder.weight",),
    }
    finalized = {
        "amp_overflow_kind": "COMBINED_SCALED_OVERFLOW_RAW_FINITE",
        "pre_scale": 65536.0,
        "post_scale": 32768.0,
        "optimizer_state_unchanged": True,
        "optimizer_step_applied": False,
    }
    receipt = update_hscf_amp_overflow_receipt(
        _pending_hscf_amp_receipt(),
        overflow=overflow,
        finalized_skip=finalized,
    )
    assert receipt["hscf_consecutive_amp_overflow_skips"] == 1
    assert receipt["hscf_persistent_amp_overflow"] is True
    recovered = update_hscf_optimizer_step_receipt(
        _append_pending_hscf_amp_batch(receipt, batch_index=2)
    )
    assert recovered["hscf_consecutive_amp_overflow_skips"] == 0
    assert recovered["hscf_persistent_amp_overflow"] is False
    assert recovered["hscf_optimizer_step_attempts"] == 2
    assert recovered["hscf_effective_optimizer_steps"] == 1
    receipt = update_hscf_amp_overflow_receipt(
        _append_pending_hscf_amp_batch(receipt, batch_index=2),
        overflow=overflow,
        finalized_skip=finalized,
    )
    assert receipt["hscf_persistent_amp_overflow"] is True
    assert receipt["hscf_max_consecutive_amp_overflow_skips"] == 2

    persistent_terminal, _ = _build_g_terminal_receipt()
    persistent_terminal["hscf_consecutive_amp_overflow_skips"] = 2
    persistent_terminal["hscf_max_consecutive_amp_overflow_skips"] = 2
    persistent_terminal["hscf_persistent_amp_overflow"] = True
    with pytest.raises(HSCFRuntimeError, match="persistent AMP overflow"):
        validate_hscf_terminal_receipt(persistent_terminal)

    zero_step_terminal, _ = _build_g_terminal_receipt(include_effective_optimizer_steps=False)
    with pytest.raises(HSCFRuntimeError, match="zero or incomplete effective optimizer steps"):
        validate_hscf_terminal_receipt(zero_step_terminal)


def test_vjp_scope_clean_none_leo_encoder_head_live_and_bias_zero_or_none() -> None:
    torch.manual_seed(11)
    model = _BindingModel().train()
    labels = _labels()
    clean = model.paired_output(torch.randn(128, 4))
    leo = model.paired_output(torch.randn(128, 4))
    validate_hscf_binding(
        model=model,
        out_clean=clean,
        out_leo=leo,
        tx_labels=labels,
        expected_class_ids=FROZEN_HSCF_CLASS_IDS,
    )
    loss, _ = hscf_loss(clean["tx_logits"], leo["tx_logits"])
    groups = hscf_shared_encoder_and_head_parameters(model)
    audit = hscf_aux_gradient_audit(loss, clean["tx_logits"], leo["tx_logits"], groups)
    for name in ("leo_raw_logits", "shared_encoder", "head_weight"):
        assert audit[name]["norm"] > 0.0 and torch.isfinite(torch.tensor(audit[name]["norm"]))
    assert audit["clean_raw_logits"]["nonzero_parameters"] == 0.0
    assert audit["head_bias"]["nonzero_parameters"] == 0.0
    assert audit["raw_unscaled"] is True and audit["diagnostic_only"] is True
    assert resolve_hscf_classifier_weight(model) is model.id_backbone.cls_head.head.weight


def test_binding_local4_mapping_warm_start_and_new_adamw() -> None:
    binding = resolve_hscf_local_head_class_binding(
        local_class_order=["20-15", "20-19", "6-15", "8-20"],
        source_train_tx=["20-15", "20-19", "6-15", "8-20"],
        checkpoint_train_tx=["20-15", "20-19", "6-15", "8-20"],
        dataset_class_order=["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    mapped = remap_hscf_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"])
    assert torch.equal(mapped, torch.tensor([3, 0, 2]))

    model = _BindingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    receipt = bind_hscf_optimizer_initial_state(_sealed_receipt(enabled=True), optimizer)
    assert receipt["optimizer_type"] == FROZEN_HSCF_OPTIMIZER_TYPE
    assert receipt["optimizer_initial_state_empty"] is True
    bound = bind_hscf_source_data_order(
        _sealed_receipt(enabled=True),
        {"labeled_indices_sha256": "1" * 64, "split_manifest_sha256": "2" * 64},
    )
    assert bound["source_labeled_indices_sha256"] == "1" * 64
    warm = strict_hscf_warm_start(
        model,
        _BindingModel().state_dict(),
        baseline_path="base.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    with pytest.raises(HSCFConfigurationError, match="AdamW"):
        bind_hscf_optimizer_initial_state(_sealed_receipt(enabled=True), torch.optim.SGD(model.parameters(), lr=1e-3))


def test_configuration_rejects_wrong_frozen_values_and_old_candidates() -> None:
    args = _frozen_args(enabled=True)
    _disable_peer_flags(args)
    assert validate_hscf_args(args) == _config(enabled=True)
    args_c = _frozen_args(enabled=False)
    _disable_peer_flags(args_c)
    assert validate_hscf_args(args_c) == _config(enabled=False)
    for name, value in (
        ("batch_size", 64),
        ("epochs", 39),
        ("amp", False),
        ("lambda_hscf", 0.01),
        ("phase1_rcat_enabled", True),
        ("phase1_recte_frozen_mode", True),
        ("use_unlabeled", True),
    ):
        bad = _frozen_args(enabled=True)
        _disable_peer_flags(bad)
        setattr(bad, name, value)
        with pytest.raises(HSCFConfigurationError):
            validate_hscf_args(bad)


def test_c_arm_identity_and_u_v_permissions_are_zero_feedback() -> None:
    base = torch.tensor(1.5, requires_grad=True)
    assert add_hscf_to_loss(base, None, _config(enabled=False)) is base
    with pytest.raises(HSCFRuntimeError, match="auxiliary"):
        add_hscf_to_loss(base, None, _config(enabled=True))
    receipt = hscf_config_receipt(_config(enabled=True))
    assert receipt["uses_target_rows"] is False
    assert receipt["uses_proxy_rows"] is False
    assert receipt["uses_held_rows"] is False
    assert receipt["uses_unlabeled_rows"] is False
    assert "ZERO_ITERATE_ZERO_FORWARD_ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER" in receipt["u_loader_common_trainer_boundary"]
    assert "ZERO_LOSS_ZERO_BACKWARD_ZERO_OPTIMIZER" in receipt["v_common_trainer_boundary"]
    assert "query" not in inspect.signature(hscf_loss).parameters
    assert "rx" not in inspect.signature(hscf_loss).parameters
    assert "day" not in inspect.signature(hscf_loss).parameters


def test_train_integration_uses_hscf_only_on_g_and_validates_terminal() -> None:
    source = inspect.getsource(train_ssdg.train)
    module_source = Path(train_ssdg.__file__).read_text(encoding="utf-8")
    assert "hscf_loss(" in source
    assert "hscf_aux_gradient_audit(" in module_source
    assert "hscf_scaled_backward_and_classify(" in source
    assert "finalize_hscf_amp_overflow_skip(" in source
    assert "update_hscf_optimizer_step_receipt(" in source
    assert "release_hscf_retained_graph_roots(" in source
    assert "validate_hscf_terminal_receipt" in source
    assert "phase1_hscf_enabled" in source
    assert "query" not in source[source.index("hscf_loss(") : source.index("hscf_loss(") + 500].lower()

    tree = ast.parse(source)
    root_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "hscf_retained_graph_roots" for target in node.targets)
    )
    release_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "release_hscf_retained_graph_roots"
    )
    mapped_roots = {
        key.value
        for key in root_assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    deleted_roots = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete) and root_assignment.lineno < node.lineno < release_call.lineno
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }
    assert root_assignment.lineno < release_call.lineno
    assert mapped_roots == deleted_roots
    assert {
        "out_l",
        "out_sat",
        "core_losses",
        "z_id_l",
        "loss_hscf_l",
        "loss_sat_cons_l",
        "loss_closed_l",
        "loss_open_l",
        "loss_l",
        "loss_closed",
        "loss_open",
        "scaled_closed_loss",
        "scaled_open_loss",
        "loss",
    } <= mapped_roots
    append_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "epoch_logs"
    )
    assert append_line < root_assignment.lineno


def test_launcher_has_exact_12_arm_frozen_matrix_and_dry_run() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_hscf12_20260810_v1}"' in launcher_text
    assert "--phase1_hscf_frozen_mode true" in launcher_text
    assert "--batch_size 128" in launcher_text
    assert "--phase1_hscf_enabled false --lambda_hscf 0" in launcher_text
    assert "--phase1_hscf_enabled true --lambda_hscf 0.02" in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12 and {arm for _, arm, _ in calls} == {"C", "G"}
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    assert sum(arm == "C" for _, arm, _ in calls) == 6
    assert sum(arm == "G" for _, arm, _ in calls) == 6
    for flag in ("ccpc_leo", "pamr", "cb_sfce", "gd_proto_nll", "icmt", "cagm", "rcrmd", "rcat", "recte", "cp_sfce"):
        assert f"--phase1_{flag}_enabled false" in launcher_text
        assert f"--lambda_{flag} 0" in launcher_text
    relative_launcher = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative_launcher], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative_launcher, "--dry-run"], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(lines) == 12
    assert all("phase1_hscf12_20260810_v1" in line for line in lines)
    assert sum("--phase1_hscf_enabled false" in line for line in lines) == 6
    assert sum("--phase1_hscf_enabled true" in line for line in lines) == 6
    assert all("--phase1_rcat_enabled false" in line for line in lines)
    assert all("--phase1_recte_enabled false" in line for line in lines)
