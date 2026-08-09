from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import SSDG.train_ssdg as train_ssdg  # noqa: E402
from cvsrffi.phase1_cp_sfce import (  # noqa: E402
    CPSFCEConfig,
    CPSFCEConfigurationError,
    CPSFCERuntimeError,
    FROZEN_CP_SFCE_GAMMA,
    FROZEN_CP_SFCE_LAMBDA,
    FROZEN_CP_SFCE_SCENARIOS,
    cp_sfce_capture_optimizer_steps_for_model,
    cp_sfce_config_receipt,
    cp_sfce_loss,
    cp_sfce_parameter_scopes,
    cp_sfce_scaled_backward_and_project,
    remap_cp_sfce_local_labels_to_head_rows,
    resolve_cp_sfce_classifier_weight,
    resolve_cp_sfce_local_head_class_binding,
    strict_cp_sfce_warm_start,
    update_cp_sfce_coverage_receipt,
    update_cp_sfce_optimizer_step_receipt,
    update_cp_sfce_projection_receipt,
    validate_cp_sfce_args,
    validate_cp_sfce_logit_binding,
    validate_cp_sfce_terminal_receipt,
    write_cp_sfce_failure_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_cp_sfce12_20260809.sh"


class _FixedScaler:
    """CPU-equivalent scaler used to prove scaled-VJP/divide-back semantics."""

    def __init__(self, scale: float = 1024.0):
        self._scale = float(scale)

    def get_scale(self):
        return self._scale

    def scale(self, value):
        return value * self._scale

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self._scale)

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        return None


class _TinyModel(torch.nn.Module):
    def __init__(self, class_count: int = 4):
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(2, 2, bias=False)
        self.id_backbone.cls_head = torch.nn.Module()
        self.id_backbone.cls_head.head = torch.nn.Linear(2, class_count, bias=False)
        # These must stay outside the CP scope and receive None/zero auxiliary VJP.
        self.id_backbone.cls_head.imp_merge = torch.nn.Linear(2, 2)
        self.id_backbone.cls_head.dac_head = torch.nn.Linear(2, 1)
        self.id_backbone.cls_head.pa_head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.id_backbone.cls_head.head(self.id_backbone.encoder(x))


def _frozen_args(*, enabled: bool = True):
    return SimpleNamespace(
        phase1_cp_sfce_frozen_mode=True,
        phase1_cp_sfce_enabled=enabled,
        lambda_cp_sfce=FROZEN_CP_SFCE_LAMBDA if enabled else 0.0,
        cp_sfce_gamma=FROZEN_CP_SFCE_GAMMA,
        phase1_cb_sfce_frozen_mode=False,
        phase1_cb_sfce_enabled=False,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
        epochs=40,
        checkpoint_selection="final_only",
        phase1_source_val_selection_only=True,
        use_sat_consistency=True,
        lambda_sat_cons=0.10,
        lambda_sat_cls=0.0,
        sat_cons_start_epoch=1,
        sat_view_prob=1.0,
        sat_train_scenarios=",".join(FROZEN_CP_SFCE_SCENARIOS),
        sat_view_schedule="",
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        phase1_ccpc_leo_frozen_mode=False,
        phase1_ccpc_leo_enabled=False,
        phase1_ccpc_leo_gradient_audit_only=False,
        lambda_ccpc_leo=0.0,
        phase1_pamr_frozen_mode=False,
        phase1_pamr_enabled=False,
        phase1_pamr_audit_only=False,
        lambda_pamr=0.0,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


def _data():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]])
    y = torch.tensor([0, 1, 2, 3])
    return x, y


def _projection_receipt(conflict: bool = True):
    group = {
        "parameter_count": 1,
        "base_norm": 1.0,
        "base_norm_sq": 1.0,
        "aux_norm": 0.1,
        "aux_norm_sq": 0.01,
        "dot": -0.1 if conflict else 0.0,
        "base_zero": False,
        "conflict": conflict,
        "projection_coefficient": -0.1 if conflict else None,
        "projected_aux_norm": 0.0 if conflict else 0.1,
        "projected_dot": 0.0,
        "projected_dot_tolerance": 1e-6,
        "finite": True,
    }
    return {
        "raw_unscaled": True,
        "diagnostic_only": False,
        "captured_scale": 1024.0,
        "base_scaled_backward_count": 1,
        "optimizer_unscale_count": 1,
        "scaled_aux_vjp_count": 1,
        "unprojected_sfce_backward_count": 0,
        "all_trainable_parameter_count": 2,
        "outside_scope_parameter_count": 0,
        "outside_scope_aux_none_or_zero": True,
        "shared_encoder": dict(group),
        "classifier_head": dict(group),
    }


def _coverage(class_id: int):
    key = str(class_id)
    return {
        "rows": 1,
        "classes": 1,
        "per_tx_rows": {key: 1},
        "per_tx_loss": {key: 0.2},
        "per_tx_finite": {key: True},
        "per_tx_nonzero_logit_gradient": {key: True},
    }


def test_cp_sfce_keeps_exact_cb_focal_loss_and_rejects_stacked_route():
    logits = torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    loss, info = cp_sfce_loss(logits, labels)
    per_row = (1.0 - logits.detach().softmax(dim=1).gather(1, labels[:, None]).squeeze(1)) * F.cross_entropy(
        logits.detach(), labels, reduction="none"
    )
    assert torch.allclose(loss.detach(), 0.5 * (per_row[:2].mean() + per_row[2:].mean()), atol=1e-7)
    assert info["present_tx_equal_aggregation"] is True
    signature = inspect.signature(cp_sfce_loss)
    assert "clean" not in " ".join(signature.parameters).lower()
    assert "rx" not in " ".join(signature.parameters).lower()
    assert "domain" not in " ".join(signature.parameters).lower()

    assert validate_cp_sfce_args(_frozen_args()).enabled is True
    assert validate_cp_sfce_args(_frozen_args(enabled=False)).enabled is False
    stacked = _frozen_args()
    stacked.phase1_cb_sfce_enabled = True
    with pytest.raises(CPSFCEConfigurationError, match="stacking"):
        validate_cp_sfce_args(stacked)
    drift = _frozen_args()
    drift.lambda_cp_sfce = 0.05
    with pytest.raises(CPSFCEConfigurationError, match="lambda_cp_sfce"):
        validate_cp_sfce_args(drift)


def test_scaled_aux_vjp_divides_by_captured_scale_and_exactly_projects_conflict():
    torch.manual_seed(7)
    model = _TinyModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = _FixedScaler(4096.0)
    x, labels = _data()
    sfce, _ = cp_sfce_loss(model(x), labels)
    # b=-a makes the exact no-epsilon projection remove all additional a.
    base = -FROZEN_CP_SFCE_LAMBDA * sfce
    scopes = cp_sfce_parameter_scopes(model)
    expected = torch.autograd.grad(
        base,
        (*scopes["shared_encoder"], *scopes["classifier_head"]),
        retain_graph=True,
    )
    info = cp_sfce_scaled_backward_and_project(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        base_loss=base,
        sfce_loss=sfce,
        loss_weight=FROZEN_CP_SFCE_LAMBDA,
    )
    assert info["captured_scale"] == 4096.0
    assert info["base_scaled_backward_count"] == 1
    assert info["optimizer_unscale_count"] == 1
    assert info["scaled_aux_vjp_count"] == 1
    assert info["unprojected_sfce_backward_count"] == 0
    for group in ("shared_encoder", "classifier_head"):
        assert info[group]["conflict"] is True
        assert info[group]["projected_dot"] >= -info[group]["projected_dot_tolerance"]
    for parameter, expected_grad in zip(
        (*scopes["shared_encoder"], *scopes["classifier_head"]), expected
    ):
        assert torch.allclose(parameter.grad, expected_grad, atol=2e-6, rtol=2e-5)
    before = info["optimizer_state_before"]
    scaler.step(optimizer)
    after = cp_sfce_capture_optimizer_steps_for_model(model, optimizer)
    receipt = cp_sfce_config_receipt(CPSFCEConfig(True, True, 0.10, 1.0))
    receipt = update_cp_sfce_optimizer_step_receipt(receipt, before=before, after=after)
    assert receipt["cp_sfce_optimizer_state_step_batches"] == 1


def test_zero_base_gradient_is_legal_but_missing_nonfinite_and_outside_aux_fail_closed():
    torch.manual_seed(11)
    model = _TinyModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    x, labels = _data()
    logits = model(x)
    sfce, _ = cp_sfce_loss(logits, labels)
    info = cp_sfce_scaled_backward_and_project(
        model=model,
        optimizer=optimizer,
        scaler=_FixedScaler(),
        base_loss=(logits * 0.0).sum(),
        sfce_loss=sfce,
        loss_weight=0.10,
    )
    for group in ("shared_encoder", "classifier_head"):
        assert info[group]["base_zero"] is True
        assert info[group]["conflict"] is False

    model = _TinyModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    logits = model(x)
    with pytest.raises(CPSFCERuntimeError, match="missing or disconnected"):
        cp_sfce_scaled_backward_and_project(
            model=model,
            optimizer=optimizer,
            scaler=_FixedScaler(),
            base_loss=F.cross_entropy(logits, labels),
            sfce_loss=torch.tensor(1.0, requires_grad=True),
            loss_weight=0.10,
        )
    with pytest.raises(CPSFCERuntimeError, match="non-finite"):
        cp_sfce_scaled_backward_and_project(
            model=model,
            optimizer=optimizer,
            scaler=_FixedScaler(),
            base_loss=(logits * float("nan")).sum(),
            sfce_loss=F.cross_entropy(logits, labels),
            loss_weight=0.10,
        )
    model = _TinyModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    logits = model(x)
    sfce, _ = cp_sfce_loss(logits, labels)
    with pytest.raises(CPSFCERuntimeError, match="outside projection scope"):
        cp_sfce_scaled_backward_and_project(
            model=model,
            optimizer=optimizer,
            scaler=_FixedScaler(),
            base_loss=F.cross_entropy(logits, labels),
            sfce_loss=sfce + 0.1 * model.id_backbone.cls_head.imp_merge.weight.sum(),
            loss_weight=0.10,
        )
    receipt = cp_sfce_config_receipt(CPSFCEConfig(True, True, 0.10, 1.0))
    with pytest.raises(CPSFCERuntimeError, match="did not advance"):
        update_cp_sfce_optimizer_step_receipt(
            receipt,
            before={"shared_encoder": (1.0,), "classifier_head": (1.0,)},
            after={"shared_encoder": (1.0,), "classifier_head": (1.0,)},
        )


def test_local4_binding_receipt_terminal_and_failure_persistence_are_fail_closed(tmp_path, monkeypatch, capsys):
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_cp_sfce_local_head_class_binding(
        local_class_order=local_order,
        source_train_tx=local_order,
        checkpoint_train_tx=local_order,
        dataset_class_order=global_order,
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert torch.equal(
        remap_cp_sfce_local_labels_to_head_rows(torch.tensor([3, 0]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0]),
    )

    receipt = cp_sfce_config_receipt(CPSFCEConfig(True, True, 0.10, 1.0))
    receipt.update(binding)
    batch_index = 0
    for scenario in FROZEN_CP_SFCE_SCENARIOS:
        for class_id in range(4):
            batch_index += 1
            receipt = update_cp_sfce_coverage_receipt(receipt, _coverage(class_id), scenario=scenario)
            receipt = update_cp_sfce_projection_receipt(
                receipt,
                _projection_receipt(),
                epoch=1 if batch_index == 1 else 2,
                batch_index=batch_index,
            )
            receipt = update_cp_sfce_optimizer_step_receipt(
                receipt,
                before={"shared_encoder": (float(batch_index - 1),), "classifier_head": (float(batch_index - 1),)},
                after={"shared_encoder": (float(batch_index),), "classifier_head": (float(batch_index),)},
            )
    terminal = validate_cp_sfce_terminal_receipt(receipt)
    assert terminal["cp_sfce_terminal_contract_passed"] is True
    for count_key in (
        "cp_sfce_projection_batches",
        "cp_sfce_outside_aux_zero_or_none_batches",
        "cp_sfce_optimizer_state_step_batches",
    ):
        incomplete = dict(receipt)
        incomplete[count_key] = int(incomplete[count_key]) - 1
        with pytest.raises(CPSFCERuntimeError, match="batchwise"):
            validate_cp_sfce_terminal_receipt(incomplete)
    no_step = dict(receipt)
    no_step["cp_sfce_optimizer_state_no_step_batches"] = 1
    with pytest.raises(CPSFCERuntimeError, match="optimizer-step"):
        validate_cp_sfce_terminal_receipt(no_step)
    impossible_scope_count = dict(receipt)
    impossible_scope_count["cp_sfce_base_zero_batches"] = {
        "shared_encoder": int(receipt["cp_sfce_batches"]) + 1,
        "classifier_head": 0,
    }
    with pytest.raises(CPSFCERuntimeError, match="exceeds observed"):
        validate_cp_sfce_terminal_receipt(impossible_scope_count)
    inert = dict(receipt)
    inert["cp_sfce_conflict_batches"] = {"shared_encoder": 0, "classifier_head": int(receipt["cp_sfce_batches"])}
    with pytest.raises(CPSFCERuntimeError, match="inert"):
        validate_cp_sfce_terminal_receipt(inert)

    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    warm = strict_cp_sfce_warm_start(
        target,
        source.state_dict(),
        baseline_path="geosat_c_final.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="final",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False
    original = CPSFCERuntimeError("P1-CP-SFCE projected dot violates tolerance")
    written = write_cp_sfce_failure_receipt(
        tmp_path,
        candidate_id="F1G",
        run_id="cp-sfce12",
        receipt=receipt,
        error=original,
        failure_stage="test",
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["error_fingerprint"] == "CP_SFCE_PROJECTION_FAILURE"
    assert "projected dot violates" not in written.read_text(encoding="utf-8")

    def writer_failure(*args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(train_ssdg, "write_cp_sfce_failure_receipt", writer_failure)
    with pytest.raises(CPSFCERuntimeError) as caught:
        try:
            raise original
        except CPSFCERuntimeError as error:
            train_ssdg._persist_cp_sfce_failure_receipt(
                out_dir=tmp_path,
                args=SimpleNamespace(candidate_id="F1G", run_id="cp-sfce12"),
                cp_sfce_receipt=receipt,
                error=error,
                failure_stage="test",
            )
            raise
    assert caught.value is original
    assert "[P1-CP-SFCE-FAILURE-RECEIPT] persistence_failed writer_exception_type=OSError" in capsys.readouterr().out


def test_lite_d_no_query_forward_backward_smoke_and_exact_head_binding():
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(17)
    model = build_dual_model(
        num_classes=4,
        num_domains=1,
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    labels = torch.tensor([0, 1, 2, 3])
    clean = torch.randn(4, 2, 128)
    satellite = torch.randn(4, 2, 128)
    out_clean = model(clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_satellite = model(satellite, y_tx=labels, domain_labels=None, return_aux=True)
    assert resolve_cp_sfce_classifier_weight(model) is model.id_backbone.cls_head.head.weight
    validate_cp_sfce_logit_binding(
        model=model,
        tx_logits=out_satellite["tx_logits"],
        tx_labels=labels,
        expected_class_ids=[0, 1, 2, 3],
    )
    sfce, _ = cp_sfce_loss(out_satellite["tx_logits"], labels)
    base = F.cross_entropy(out_clean["tx_logits"], labels) + 0.10 * F.kl_div(
        F.log_softmax(out_satellite["tx_logits"], dim=1),
        out_clean["tx_logits"].detach().softmax(dim=1),
        reduction="batchmean",
    )
    info = cp_sfce_scaled_backward_and_project(
        model=model,
        optimizer=optimizer,
        scaler=_FixedScaler(128.0),
        base_loss=base,
        sfce_loss=sfce,
        loss_weight=0.10,
    )
    assert info["outside_scope_aux_none_or_zero"] is True
    assert info["all_trainable_parameter_count"] >= 2
    assert all(parameter.grad is not None for parameter in cp_sfce_parameter_scopes(model)["shared_encoder"])


def test_train_integration_and_launcher_freeze_the_single_cp_backward_path():
    source = inspect.getsource(train_ssdg.train)
    assert "cp_sfce_scaled_backward_and_project(" in source
    assert "scaled_base_backward_unscale_aux_vjp_projection" in source
    assert "post_scaler_step_optimizer_state_increment" in source
    assert "unprojected_sfce_backward_count" in Path(
        CODE_ROOT / "cvsrffi" / "phase1_cp_sfce.py"
    ).read_text(encoding="utf-8")
    assert "add_cp_sfce_to_loss" not in source

    text = LAUNCHER.read_text(encoding="utf-8")
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", text, flags=re.MULTILINE)
    assert calls == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert "phase1_cp_sfce12_20260809_v1" in text
    assert "GEOSAT_CKPT_ROOT:-${PROJECT_ROOT}/runs/phase1_loto_clsgeo12_20260808_v1" in text
    assert "--phase1_cb_sfce_enabled false" in text
    assert "--lambda_cp_sfce 0.10" in text and "--lambda_cp_sfce 0" in text
    completed = subprocess.run(
        ["bash", f"scripts/{LAUNCHER.name}", "--dry-run"],
        cwd=str(CODE_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.count("[DRY-RUN]") == 12
