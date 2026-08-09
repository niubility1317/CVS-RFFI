from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_rcrmd import (  # noqa: E402
    FROZEN_RCRMD_CLASS_IDS,
    FROZEN_RCRMD_LAMBDA,
    FROZEN_RCRMD_OPTIMIZER_TYPE,
    FROZEN_RCRMD_SCENARIOS,
    FROZEN_RCRMD_SOURCE_RECEIVER_IDS,
    RCRMDConfig,
    RCRMDConfigurationError,
    RCRMDRuntimeError,
    RCRMD_RECEIPT_SCHEMA,
    bind_rcrmd_optimizer_initial_state,
    bind_rcrmd_source_data_order,
    remap_rcrmd_local_labels_to_head_rows,
    rcrmd_aux_gradient_audit,
    rcrmd_config_receipt,
    rcrmd_loss,
    rcrmd_shared_encoder_and_head_parameters,
    resolve_rcrmd_classifier_weight,
    resolve_rcrmd_local_head_class_binding,
    strict_rcrmd_warm_start,
    update_rcrmd_common_batch_sequence_receipt,
    update_rcrmd_gradient_audit_receipt,
    update_rcrmd_receipt,
    validate_rcrmd_args,
    validate_rcrmd_binding,
    validate_rcrmd_terminal_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_rcrmd12_20260810.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        phase1_rcrmd_frozen_mode=True,
        phase1_rcrmd_enabled=enabled,
        lambda_rcrmd=FROZEN_RCRMD_LAMBDA if enabled else 0.0,
        from_scratch=False,
        baseline_ckpt="geosat_c_final.pth",
        freeze_backbone=False,
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
        sat_train_scenarios=",".join(FROZEN_RCRMD_SCENARIOS),
        sat_view_schedule="",
        use_concat_sat_channel_aug=False,
        use_unlabeled=False,
        use_tx_rx_balanced_sampler=False,
        use_aug=False,
        use_mixstyle=False,
        reject_head=False,
        phase1_ccpc_leo_frozen_mode=False,
        phase1_ccpc_leo_enabled=False,
        lambda_ccpc_leo=0.0,
        phase1_pamr_frozen_mode=False,
        phase1_pamr_enabled=False,
        lambda_pamr=0.0,
        phase1_cb_sfce_frozen_mode=False,
        phase1_cb_sfce_enabled=False,
        lambda_cb_sfce=0.0,
        phase1_gd_proto_nll_frozen_mode=False,
        phase1_gd_proto_nll_enabled=False,
        lambda_gd_proto_nll=0.0,
        phase1_icmt_frozen_mode=False,
        phase1_icmt_enabled=False,
        lambda_icmt=0.0,
        phase1_cagm_frozen_mode=False,
        phase1_cagm_enabled=False,
        lambda_cagm=0.0,
        phase1_cp_sfce_frozen_mode=False,
        phase1_cp_sfce_enabled=False,
        lambda_cp_sfce=0.0,
        use_ema_teacher=False,
        teacher_ckpt="",
        lambda_teacher_clean_kl=0.0,
        lambda_teacher_sat_kl=0.0,
        lambda_teacher_zid_mse=0.0,
    )


class _BindingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.encoder = torch.nn.Linear(4, 4, bias=False)
        self.id_backbone.cls_head = torch.nn.Module()
        self.id_backbone.cls_head.joint_proj = torch.nn.Linear(4, 4, bias=False)
        self.id_backbone.cls_head.head = torch.nn.Linear(4, 4, bias=False)
        self.id_backbone.cls_head.imp_merge = torch.nn.Linear(4, 4)
        self.id_backbone.cls_head.dac_head = torch.nn.Linear(4, 1)
        self.id_backbone.cls_head.pa_head = torch.nn.Linear(4, 1)
        with torch.no_grad():
            eye = torch.eye(4)
            self.id_backbone.encoder.weight.copy_(eye)
            self.id_backbone.cls_head.joint_proj.weight.copy_(eye)
            self.id_backbone.cls_head.head.weight.copy_(eye)

    def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        z_id = self.id_backbone.cls_head.joint_proj(self.id_backbone.encoder(x))
        return {
            "z_id": z_id,
            "z_id_key": "feat_joint",
            "tx_logits": self.id_backbone.cls_head.head(z_id),
        }


def _logits_from_margins(labels: torch.Tensor, margins: list[float]) -> torch.Tensor:
    log_three = float(torch.log(torch.tensor(3.0)).item())
    rows: list[torch.Tensor] = []
    for label, margin in zip(labels.tolist(), margins):
        row = torch.zeros(4, dtype=torch.float32)
        row[int(label)] = float(margin) + log_three
        rows.append(row)
    return torch.stack(rows).requires_grad_(True)


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = rcrmd_config_receipt(RCRMDConfig(True, enabled, 0.02 if enabled else 0.0))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "source_receiver_ids": list(FROZEN_RCRMD_SOURCE_RECEIVER_IDS),
            "source_receiver_count": len(FROZEN_RCRMD_SOURCE_RECEIVER_IDS),
            "source_receiver_ids_sha256": "e" * 64,
            "optimizer_type": FROZEN_RCRMD_OPTIMIZER_TYPE,
            "optimizer_initial_state_sha256": "f" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": list(FROZEN_RCRMD_CLASS_IDS),
            "common_batch_sequence_sha256": "1" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RCRMD_SCENARIOS},
        }
    )
    return receipt


def _cell_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor(list(FROZEN_RCRMD_CLASS_IDS) * len(FROZEN_RCRMD_SOURCE_RECEIVER_IDS))
    receivers = torch.tensor(
        [receiver for receiver in FROZEN_RCRMD_SOURCE_RECEIVER_IDS for _ in FROZEN_RCRMD_CLASS_IDS]
    )
    clean = _logits_from_margins(labels, [1.0] * int(labels.numel()))
    leo = _logits_from_margins(labels, [0.0] * int(labels.numel()))
    return clean, leo, labels, receivers


def _common_metadata(rows: int, offset: int = 0) -> dict[str, torch.Tensor]:
    return {
        "base_index": torch.arange(offset, offset + rows),
        "rx_i": torch.zeros(rows, dtype=torch.long),
        "day_i": torch.arange(rows) + 2021,
        "target": torch.arange(rows),
        "proxy": torch.arange(rows),
    }


def _build_g_terminal_receipt(*, margin_drop: float) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    model = _BindingModel().train()
    _, _, labels, receivers = _cell_batch()
    log_three = float(torch.log(torch.tensor(3.0)).item())
    x_clean = (log_three + margin_drop) * torch.eye(4)[labels]
    x_leo = log_three * torch.eye(4)[labels]
    out_clean = model.paired_output(x_clean)
    out_leo = model.paired_output(x_leo)
    validate_rcrmd_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RCRMD_CLASS_IDS,
        expected_receiver_ids=FROZEN_RCRMD_SOURCE_RECEIVER_IDS,
    )
    loss, info = rcrmd_loss(
        out_clean["tx_logits"],
        out_leo["tx_logits"],
        labels,
        receivers,
        FROZEN_RCRMD_SOURCE_RECEIVER_IDS,
    )
    audit = rcrmd_aux_gradient_audit(loss, rcrmd_shared_encoder_and_head_parameters(model))
    receipt = _sealed_receipt(enabled=True)
    receipt = update_rcrmd_gradient_audit_receipt(receipt, audit)
    for index, scenario in enumerate(FROZEN_RCRMD_SCENARIOS, start=1):
        receipt = update_rcrmd_common_batch_sequence_receipt(
            receipt,
            epoch=1,
            batch_index=index,
            scenario=scenario,
            source_tx_labels=labels,
            source_rx_labels=receivers,
            metadata={"base_index": torch.arange(28) + index * 100},
        )
        receipt = update_rcrmd_receipt(receipt, info, scenario=scenario, epoch=1, batch_index=index)
    return receipt, info, audit


def test_formula_is_hand_calculable_one_over_28_and_empty_cells_keep_denominator() -> None:
    labels = torch.tensor([0, 1])
    receivers = torch.tensor([0, 1])
    clean = _logits_from_margins(labels, [2.0, 3.0])
    leo = _logits_from_margins(labels, [1.0, 1.0])
    loss, info = rcrmd_loss(clean, leo, labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
    assert float(loss.detach()) == pytest.approx(5.0 / 28.0, abs=1e-6)
    assert info["global_denominator"] == 28
    assert info["fixed_scale"] == pytest.approx(1.0 / 28.0)
    assert info["cells"]["rx0|tx0"]["n_rc"] == 1
    assert info["cells"]["rx0|tx0"]["g_rc"] == pytest.approx(1.0)
    assert info["cells"]["rx1|tx1"]["g_rc"] == pytest.approx(4.0)
    empty = info["cells"]["rx6|tx3"]
    assert empty["n_rc"] == 0 and empty["g_rc"] == 0.0 and empty["loss_contribution"] == 0.0
    assert info["no_active_renormalization"] is True
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) > 0


def test_rx_class_permutation_and_sample_reordering_preserve_equal_weight_loss() -> None:
    torch.manual_seed(19)
    labels = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3, 3, 0, 2, 1, 0, 1, 2, 3])
    receivers = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1])
    clean = _logits_from_margins(labels, [0.5 + 0.1 * i for i in range(labels.numel())])
    leo = _logits_from_margins(labels, [-0.2 + 0.03 * i for i in range(labels.numel())])
    direct, _ = rcrmd_loss(clean, leo, labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
    class_permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(class_permutation)
    rx_permutation = torch.tensor([4, 6, 0, 5, 1, 3, 2])
    order = torch.tensor([7, 2, 15, 0, 11, 5, 13, 1, 9, 4, 14, 3, 10, 6, 8, 12])
    permuted_labels = class_permutation[labels].index_select(0, order)
    permuted_receivers = rx_permutation[receivers].index_select(0, order)
    permuted_clean = clean.detach().index_select(1, inverse).index_select(0, order).requires_grad_(True)
    permuted_leo = leo.detach().index_select(1, inverse).index_select(0, order).requires_grad_(True)
    permuted, _ = rcrmd_loss(
        permuted_clean,
        permuted_leo,
        permuted_labels,
        permuted_receivers,
        FROZEN_RCRMD_SOURCE_RECEIVER_IDS,
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)


def test_nonfinite_q_zero_and_frozen_seven_receiver_allowlist_fail_closed() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    receivers = torch.tensor([0, 1, 2, 3])
    clean = _logits_from_margins(labels, [1.0] * 4)
    leo = _logits_from_margins(labels, [2.0] * 4)
    zero_loss, zero_info = rcrmd_loss(clean, leo, labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
    assert float(zero_loss.detach()) == pytest.approx(0.0, abs=1e-8)
    assert zero_info["active_q"] == 0 and zero_info["finite_q"] == 4
    zero_loss.backward()
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) == 0

    nan_clean = clean.detach().clone()
    nan_clean[0, 0] = float("nan")
    with pytest.raises(RCRMDRuntimeError, match="non-finite"):
        rcrmd_loss(nan_clean.requires_grad_(True), leo.detach().requires_grad_(True), labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
    with pytest.raises(RCRMDRuntimeError, match="non-finite"):
        rcrmd_loss(clean.detach().requires_grad_(True), torch.full_like(leo, float("nan")).requires_grad_(True), labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
    with pytest.raises(RCRMDConfigurationError, match="frozen F1C source receivers"):
        rcrmd_loss(clean, leo, labels, receivers, tuple(range(6)))
    with pytest.raises(RCRMDConfigurationError, match="frozen F1C source receivers"):
        rcrmd_loss(clean, leo, labels, receivers, tuple(range(1, 8)))


def test_configuration_freezes_c_g_and_forbids_stacked_routes() -> None:
    assert validate_rcrmd_args(_frozen_args(enabled=True)) == RCRMDConfig(True, True, 0.02)
    assert validate_rcrmd_args(_frozen_args(enabled=False)) == RCRMDConfig(True, False, 0.0)
    for name, value in (("lambda_rcrmd", 0.01), ("epochs", 39), ("phase1_cagm_enabled", True), ("use_unlabeled", True)):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(RCRMDConfigurationError):
            validate_rcrmd_args(bad)
    bad_scenarios = _frozen_args()
    bad_scenarios.sat_train_scenarios = "leo_clear_weak,leo_rain_weak"
    with pytest.raises(RCRMDConfigurationError, match="sat_train_scenarios"):
        validate_rcrmd_args(bad_scenarios)


def test_c_common_coverage_closes_but_auxiliary_fields_stay_na_or_zero_and_metadata_is_blind() -> None:
    _, _, labels, receivers = _cell_batch()
    receipt = _sealed_receipt(enabled=False)
    for index, scenario in enumerate(FROZEN_RCRMD_SCENARIOS, start=1):
        receipt = update_rcrmd_common_batch_sequence_receipt(
            receipt,
            epoch=1,
            batch_index=index,
            scenario=scenario,
            source_tx_labels=labels,
            source_rx_labels=receivers,
            metadata=_common_metadata(28, offset=index * 10),
        )
    terminal = validate_rcrmd_terminal_receipt(receipt)
    assert terminal["schema"] == RCRMD_RECEIPT_SCHEMA
    assert terminal["rcrmd_terminal_contract_passed"] is True
    assert terminal["rcrmd_batches"] == 0
    assert terminal["rcrmd_total_rows"] == 0
    assert terminal["rcrmd_active_q"] == 0
    assert terminal["rcrmd_loss_sum"] == 0.0
    assert terminal["rcrmd_scenes"] == {}
    assert terminal["rcrmd_g_batch_aux"] == []
    assert terminal["rcrmd_gradient_audit_completed"] is False
    assert all("day_i" not in event and "target" not in event and "proxy" not in event for event in terminal["rcrmd_common_batch_cells"])
    assert "day" not in inspect.signature(rcrmd_loss).parameters
    assert "target" not in inspect.signature(rcrmd_loss).parameters
    assert "proxy" not in inspect.signature(rcrmd_loss).parameters
    missing_physical = _sealed_receipt(enabled=False)
    with pytest.raises(RCRMDRuntimeError, match="physical batch sequence metadata"):
        update_rcrmd_common_batch_sequence_receipt(
            missing_physical,
            epoch=1,
            batch_index=1,
            scenario=FROZEN_RCRMD_SCENARIOS[0],
            source_tx_labels=labels,
            source_rx_labels=receivers,
            metadata={"day_i": torch.ones(28), "target": torch.ones(28), "proxy": torch.ones(28)},
        )


def test_g_active_q_vjp_and_eighty_four_cell_terminal_contract() -> None:
    receipt, info, audit = _build_g_terminal_receipt(margin_drop=5.0)
    assert info["active_q"] == 28 and info["finite_q"] == 28
    assert audit["raw_unscaled"] is True and audit["diagnostic_only"] is True
    assert audit["touches_amp_optimizer_rng"] is False
    assert audit["shared_encoder"]["norm"] > 0.0
    assert audit["classifier_head"]["norm"] > 0.0
    ledger = float(receipt["rcrmd_loss_sum"])
    cell_total = sum(
        float(cell["loss_sum"])
        for scene_cells in receipt["rcrmd_scenes"].values()
        for cell in scene_cells.values()
    )
    assert ledger == pytest.approx(75.0000057220459, abs=1e-6)
    assert cell_total == pytest.approx(75.0, abs=1e-12)
    assert ledger - cell_total > 1e-8 * max(1.0, abs(cell_total))
    terminal = validate_rcrmd_terminal_receipt(receipt)
    assert terminal["rcrmd_terminal_contract_passed"] is True
    assert terminal["rcrmd_total_rows"] == 84
    assert terminal["rcrmd_active_q"] == 84
    assert terminal["rcrmd_gradient_audit_completed"] is True
    assert all(len(terminal["rcrmd_scenes"][scene]) == 28 for scene in FROZEN_RCRMD_SCENARIOS)
    assert all(
        cell["rows"] == 1
        for scene in terminal["rcrmd_scenes"].values()
        for cell in scene.values()
    )


def test_g_terminal_rejects_material_loss_ledger_drift() -> None:
    receipt, _, _ = _build_g_terminal_receipt(margin_drop=5.0)
    receipt["rcrmd_loss_sum"] = float(receipt["rcrmd_loss_sum"]) + 1.0
    with pytest.raises(RCRMDRuntimeError, match="batch/active/loss counters"):
        validate_rcrmd_terminal_receipt(receipt)


def test_lite_d_no_query_smoke_and_common_class_optimizer_binding() -> None:
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(31)
    model = build_dual_model(
        num_classes=4, num_domains=1, dataset="wisig", input_len=128, model_variant="lite_d"
    ).train()
    x_clean = torch.randn(8, 2, 128)
    x_leo = torch.randn(8, 2, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    receivers = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    out_clean = model(x_clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_leo = model(x_leo, y_tx=labels, domain_labels=None, return_aux=True)
    assert out_clean["z_id_key"] == "feat_joint" and out_leo["z_id_key"] == "feat_joint"
    validate_rcrmd_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RCRMD_CLASS_IDS,
        expected_receiver_ids=FROZEN_RCRMD_SOURCE_RECEIVER_IDS,
    )
    loss, info = rcrmd_loss(
        out_clean["tx_logits"], out_leo["tx_logits"], labels, receivers, FROZEN_RCRMD_SOURCE_RECEIVER_IDS
    )
    assert torch.isfinite(loss.detach()) and info["finite"] is True
    assert "query" not in inspect.signature(rcrmd_loss).parameters

    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_rcrmd_local_head_class_binding(
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
        remap_rcrmd_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )
    receipt = bind_rcrmd_source_data_order(
        _sealed_receipt(enabled=True),
        {"labeled_indices_sha256": "1" * 64, "split_manifest_sha256": "2" * 64, "source_receivers": list(range(7))},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    receipt = bind_rcrmd_optimizer_initial_state(receipt, optimizer)
    assert receipt["optimizer_type"] == FROZEN_RCRMD_OPTIMIZER_TYPE
    assert receipt["optimizer_initial_state_empty"] is True
    target = _BindingModel()
    warm = strict_rcrmd_warm_start(
        target,
        _BindingModel().state_dict(),
        baseline_path="base.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False


def test_rcrmd_launcher_is_frozen_twelve_arm_dry_run_only() -> None:
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert "RUN_ID=\"${RUN_ID:-phase1_rcrmd12_20260810_v1}\"" in launcher_text
    assert "phase1_rcrmd_frozen_mode true" in launcher_text
    assert "--phase1_rcrmd_enabled false --lambda_rcrmd 0" in launcher_text
    assert "--phase1_rcrmd_enabled true --lambda_rcrmd 0.02" in launcher_text
    assert "phase1_rcrmd12_20260810_v2" not in launcher_text
    for flag in ("phase1_cagm", "phase1_icmt", "phase1_gd_proto_nll", "phase1_cb_sfce", "phase1_cp_sfce"):
        assert f"--{flag}_frozen_mode true" not in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12
    assert {arm for _, arm, _ in calls} == {"C", "G"}
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    relative_launcher = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative_launcher], cwd=CODE_ROOT, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative_launcher, "--dry-run"], cwd=CODE_ROOT, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(lines) == 12
    assert all("phase1_rcrmd12_20260810_v1" in line for line in lines)
    assert all("--phase1_rcrmd_frozen_mode true" in line for line in lines)
    assert sum("--phase1_rcrmd_enabled false" in line for line in lines) == 6
    assert sum("--phase1_rcrmd_enabled true" in line for line in lines) == 6
    assert all("--phase1_cagm_enabled false" in line for line in lines)
    assert all("--phase1_icmt_enabled false" in line for line in lines)
