from __future__ import annotations

import inspect
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

from cvsrffi.phase1_rcat import (  # noqa: E402
    FROZEN_RCAT_CLASS_IDS,
    FROZEN_RCAT_LAMBDA,
    FROZEN_RCAT_OPTIMIZER_TYPE,
    FROZEN_RCAT_SCENARIOS,
    FROZEN_RCAT_SOURCE_RECEIVER_IDS,
    RCATConfig,
    RCATConfigurationError,
    RCATRuntimeError,
    RCAT_RECEIPT_SCHEMA,
    add_rcat_to_loss,
    bind_rcat_optimizer_initial_state,
    bind_rcat_source_data_order,
    rcat_aux_gradient_audit,
    rcat_config_receipt,
    rcat_loss,
    rcat_shared_encoder_and_head_parameters,
    remap_rcat_local_labels_to_head_rows,
    resolve_rcat_classifier_weight,
    resolve_rcat_local_head_class_binding,
    strict_rcat_warm_start,
    totalized_l2,
    update_rcat_common_batch_sequence_receipt,
    update_rcat_gradient_audit_receipt,
    update_rcat_receipt,
    validate_rcat_args,
    validate_rcat_binding,
    validate_rcat_terminal_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_rcat12_20260810.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    """Arguments consumed by the frozen RCAT validator, with all peers off."""

    return SimpleNamespace(
        phase1_rcat_frozen_mode=True,
        phase1_rcat_enabled=enabled,
        lambda_rcat=FROZEN_RCAT_LAMBDA if enabled else 0.0,
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
        sat_train_scenarios=",".join(FROZEN_RCAT_SCENARIOS),
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
        phase1_rcrmd_frozen_mode=False,
        phase1_rcrmd_enabled=False,
        lambda_rcrmd=0.0,
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


def _config(*, enabled: bool = True) -> RCATConfig:
    return RCATConfig(True, enabled, FROZEN_RCAT_LAMBDA if enabled else 0.0)


def _all_cells() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(list(FROZEN_RCAT_CLASS_IDS) * len(FROZEN_RCAT_SOURCE_RECEIVER_IDS))
    receivers = torch.tensor(
        [receiver for receiver in FROZEN_RCAT_SOURCE_RECEIVER_IDS for _ in FROZEN_RCAT_CLASS_IDS]
    )
    return labels, receivers


def _one_hot_features(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct same-shape clean/LEO features with strictly positive q."""

    clean = torch.eye(4, dtype=torch.float32).index_select(0, labels)
    other = torch.roll(clean, shifts=1, dims=1)
    leo = clean + 0.5 * other
    return clean, leo


def _metadata(rows: int, offset: int = 0) -> dict[str, torch.Tensor]:
    # base_index and sig_i identify the same physical records; rx_i is retained
    # as a cross-check and never acts as a method selector.
    return {
        "base_index": torch.arange(offset, offset + rows),
        "sig_i": torch.arange(offset, offset + rows) + 10000,
        "rx_i": torch.repeat_interleave(
            torch.arange(len(FROZEN_RCAT_SOURCE_RECEIVER_IDS)), len(FROZEN_RCAT_CLASS_IDS)
        )[:rows],
        "same_physical": torch.ones(rows, dtype=torch.bool),
    }


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = rcat_config_receipt(_config(enabled=enabled))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "source_receiver_ids": list(FROZEN_RCAT_SOURCE_RECEIVER_IDS),
            "source_receiver_count": len(FROZEN_RCAT_SOURCE_RECEIVER_IDS),
            "source_receiver_ids_sha256": "e" * 64,
            "optimizer_type": FROZEN_RCAT_OPTIMIZER_TYPE,
            "optimizer_initial_state_sha256": "f" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": list(FROZEN_RCAT_CLASS_IDS),
            "common_l_base_head_input_path_verified": True,
            "common_batch_sequence_sha256": "1" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RCAT_SCENARIOS},
        }
    )
    return receipt


def _common_bind(receipt: dict[str, object], *, epoch: int, batch_index: int, scenario: str) -> dict[str, object]:
    labels, receivers = _all_cells()
    return update_rcat_common_batch_sequence_receipt(
        receipt,
        epoch=epoch,
        batch_index=batch_index,
        scenario=scenario,
        source_tx_labels=labels,
        source_rx_labels=receivers,
        metadata=_metadata(int(labels.numel()), offset=batch_index * 100),
    )


def _build_g_terminal_receipt() -> tuple[dict[str, object], dict[str, object]]:
    model = _BindingModel().train()
    labels, receivers = _all_cells()
    x_clean, x_leo = _one_hot_features(labels)
    out_clean = model.paired_output(x_clean)
    out_leo = model.paired_output(x_leo)
    validate_rcat_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RCAT_CLASS_IDS,
        expected_receiver_ids=FROZEN_RCAT_SOURCE_RECEIVER_IDS,
    )
    loss, info = rcat_loss(
        out_clean["z_id"], out_leo["z_id"], labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS
    )
    audit = rcat_aux_gradient_audit(
        loss, out_leo["z_id"], rcat_shared_encoder_and_head_parameters(model)
    )
    receipt = _sealed_receipt(enabled=True)
    for index, scenario in enumerate(FROZEN_RCAT_SCENARIOS, start=1):
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
        receipt = update_rcat_receipt(receipt, info, scenario=scenario, epoch=1, batch_index=index)
    receipt = update_rcat_gradient_audit_receipt(receipt, audit)
    return receipt, info


def _cell_counts(info: dict[str, object]) -> dict[str, int]:
    values = {
        key: value.get("n_rc", -1)
        for key, value in dict(info.get("cells", {})).items()
        if isinstance(value, dict)
    }
    if not isinstance(values, dict):
        raise AssertionError("RCAT info must expose n_rc or cell_counts")
    return {str(key): int(value) for key, value in values.items()}


def test_totalized_l2_normalizes_positive_zero_and_rejects_nonfinite() -> None:
    labels = torch.tensor([0, 1])
    receivers = torch.tensor([0, 1])
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    leo = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    clean_t = totalized_l2(clean)
    leo_t = totalized_l2(leo)
    assert torch.allclose(clean_t[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(clean_t[1], torch.zeros(4))
    loss, info = rcat_loss(clean, leo, labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS)
    # q=[2,1] after T(z)=z/||z||; rcat_loss is raw and add_rcat_to_loss applies .02.
    assert float(loss.detach()) == pytest.approx(3.0 / 28.0, abs=1e-7)
    assert float(info["cells"]["rx0|tx0"]["g_rc"]) == pytest.approx(2.0)
    assert float(info["cells"]["rx1|tx1"]["g_rc"]) == pytest.approx(1.0)
    assert int(info["clean_zero_rows"]) == 1
    assert int(info["leo_zero_rows"]) == 0
    assert int(info["positive_q"]) == 2
    assert float(info["fixed_scale"]) == pytest.approx(1.0 / 28.0)
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None and int(torch.count_nonzero(leo.grad).item()) > 0

    zero = torch.zeros((2, 4), requires_grad=True)
    zero_loss, zero_info = rcat_loss(
        zero, zero.clone().requires_grad_(True), labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS
    )
    assert float(zero_loss.detach()) == pytest.approx(0.0, abs=1e-9)
    assert int(zero_info["positive_q"]) == 0
    assert int(zero_info["clean_zero_rows"]) == 2 and int(zero_info["leo_zero_rows"]) == 2

    nonfinite = clean.detach().clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(RCATRuntimeError, match="non-finite"):
        rcat_loss(
            nonfinite.requires_grad_(True),
            leo.detach().requires_grad_(True),
            labels,
            receivers,
            FROZEN_RCAT_SOURCE_RECEIVER_IDS,
        )


def test_fixed_28_cell_scale_keeps_empty_cells_zero_without_active_renormalization() -> None:
    labels = torch.tensor([0, 1])
    receivers = torch.tensor([0, 1])
    clean = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    leo = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], requires_grad=True)
    loss, info = rcat_loss(clean, leo, labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS)
    assert float(loss.detach()) == pytest.approx(4.0 / 28.0, abs=1e-7)
    counts = _cell_counts(info)
    assert len(counts) == 28 and counts["rx0|tx0"] == 1 and counts["rx1|tx1"] == 1
    assert counts["rx6|tx3"] == 0
    assert bool(info.get("no_active_renormalization", True)) is True
    assert float(info.get("global_denominator", info.get("denominator", 28))) == 28
    assert float(info["cells"]["rx6|tx3"]["g_rc"]) == 0.0


def test_rx_and_local4_permutations_preserve_totalized_l2_loss() -> None:
    torch.manual_seed(19)
    labels, receivers = _all_cells()
    clean, leo = _one_hot_features(labels)
    direct, _ = rcat_loss(
        clean.requires_grad_(True), leo.requires_grad_(True), labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS
    )
    class_permutation = torch.tensor([2, 0, 3, 1])
    rx_permutation = torch.tensor([4, 6, 0, 5, 1, 3, 2])
    feature_permutation = torch.tensor([2, 0, 3, 1])
    order = torch.tensor([7, 2, 27, 0, 11, 5, 13, 1, 9, 4, 14, 3, 10, 6, 8, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    permuted, _ = rcat_loss(
        clean.detach().index_select(1, feature_permutation).index_select(0, order).requires_grad_(True),
        leo.detach().index_select(1, feature_permutation).index_select(0, order).requires_grad_(True),
        class_permutation[labels].index_select(0, order),
        rx_permutation[receivers].index_select(0, order),
        FROZEN_RCAT_SOURCE_RECEIVER_IDS,
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)


def test_configuration_common_c_vs_g_and_peer_routes_are_frozen() -> None:
    assert validate_rcat_args(_frozen_args(enabled=True)) == _config(enabled=True)
    assert validate_rcat_args(_frozen_args(enabled=False)) == _config(enabled=False)
    for name, value in (
        ("lambda_rcat", 0.01),
        ("epochs", 39),
        ("phase1_gd_proto_nll_enabled", True),
        ("phase1_icmt_enabled", True),
        ("phase1_cagm_enabled", True),
        ("phase1_rcrmd_enabled", True),
        ("use_unlabeled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(RCATConfigurationError):
            validate_rcat_args(bad)


def test_feat_joint_binding_encoder_vjp_and_head_aux_are_na_or_zero() -> None:
    model = _BindingModel().train()
    labels, receivers = _all_cells()
    x_clean, x_leo = _one_hot_features(labels)
    out_clean = model.paired_output(x_clean)
    out_leo = model.paired_output(x_leo)
    validate_rcat_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RCAT_CLASS_IDS,
        expected_receiver_ids=FROZEN_RCAT_SOURCE_RECEIVER_IDS,
    )
    assert resolve_rcat_classifier_weight(model) is model.id_backbone.cls_head.head.weight
    aux, _ = rcat_loss(
        out_clean["z_id"], out_leo["z_id"], labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS
    )
    audit = rcat_aux_gradient_audit(
        aux, out_leo["z_id"], rcat_shared_encoder_and_head_parameters(model)
    )
    assert audit["raw_unscaled"] is True and audit["diagnostic_only"] is True
    assert audit["shared_encoder"]["norm"] > 0.0
    head = audit.get("classifier_head", audit.get("head_aux", {}))
    if isinstance(head, dict):
        assert float(head.get("nonzero_parameters", 0.0)) == 0.0
    aux.backward()
    assert model.id_backbone.cls_head.head.weight.grad is None


def test_common_c_has_no_aux_and_g_closes_84_cells_with_positive_q() -> None:
    labels, receivers = _all_cells()
    control = _sealed_receipt(enabled=False)
    for index, scenario in enumerate(FROZEN_RCAT_SCENARIOS, start=1):
        control = _common_bind(control, epoch=1, batch_index=index, scenario=scenario)
    c_terminal = validate_rcat_terminal_receipt(control)
    assert c_terminal["schema"] == RCAT_RECEIPT_SCHEMA
    assert c_terminal["rcat_terminal_contract_passed"] is True
    assert int(c_terminal.get("rcat_batches", 0)) == 0
    assert float(c_terminal.get("rcat_loss_sum", 0.0)) == 0.0
    assert not c_terminal.get("rcat_scenes")
    assert not c_terminal.get("rcat_gradient_audit")

    receipt, info = _build_g_terminal_receipt()
    assert int(info["positive_q"]) > 0
    terminal = validate_rcat_terminal_receipt(receipt)
    assert terminal["rcat_terminal_contract_passed"] is True
    assert int(terminal["rcat_total_rows"]) == 84
    assert int(terminal["rcat_positive_q"]) > 0
    assert all(len(terminal["rcat_scenes"][scene]) == 28 for scene in FROZEN_RCAT_SCENARIOS)


def test_material_receipt_mismatch_fails_closed() -> None:
    receipt, _ = _build_g_terminal_receipt()
    drifted = dict(receipt)
    drifted["rcat_loss_sum"] = float(drifted.get("rcat_loss_sum", 0.0)) + 1.0
    with pytest.raises(RCATRuntimeError, match="loss|counter|receipt|ledger"):
        validate_rcat_terminal_receipt(drifted)
    sequence_drifted = dict(receipt)
    sequence_drifted["common_batch_sequence_rows"] = int(sequence_drifted["common_batch_sequence_rows"]) + 1
    with pytest.raises(RCATRuntimeError, match="row|sequence|coverage|receipt"):
        validate_rcat_terminal_receipt(sequence_drifted)


def test_lite_d_no_query_smoke_uses_feat_joint_and_rcat_has_no_query_argument() -> None:
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
    validate_rcat_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RCAT_CLASS_IDS,
        expected_receiver_ids=FROZEN_RCAT_SOURCE_RECEIVER_IDS,
    )
    loss, info = rcat_loss(
        out_clean["z_id"], out_leo["z_id"], labels, receivers, FROZEN_RCAT_SOURCE_RECEIVER_IDS
    )
    assert torch.isfinite(loss.detach()) and int(info["nonfinite_rows"]) == 0
    assert "query" not in inspect.signature(rcat_loss).parameters


def test_common_receipt_binds_same_physical_and_optimizer_warm_start() -> None:
    labels, receivers = _all_cells()
    control = _sealed_receipt(enabled=False)
    control = _common_bind(control, epoch=1, batch_index=1, scenario=FROZEN_RCAT_SCENARIOS[0])
    events = list(control.get("rcat_common_batch_cells", control.get("common_batch_cells", [])))
    assert events
    event = events[0]
    assert event["same_physical_clean_leo"] is True
    assert len(event["n_rc"]) == 28 and sum(event["n_rc"].values()) == int(labels.numel())
    assert set(event["effective_weights"]) == set(event["n_rc"])

    missing_physical = _sealed_receipt(enabled=False)
    with pytest.raises(RCATRuntimeError, match="physical|metadata|same"):
        update_rcat_common_batch_sequence_receipt(
            missing_physical,
            epoch=1,
            batch_index=1,
            scenario=FROZEN_RCAT_SCENARIOS[0],
            source_tx_labels=labels,
            source_rx_labels=receivers,
            metadata={"same_physical": torch.ones(labels.numel(), dtype=torch.bool)},
        )

    model = _BindingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    receipt = bind_rcat_optimizer_initial_state(_sealed_receipt(enabled=True), optimizer)
    assert receipt["optimizer_type"] == FROZEN_RCAT_OPTIMIZER_TYPE
    assert receipt["optimizer_initial_state_empty"] is True
    bound = bind_rcat_source_data_order(
        _sealed_receipt(enabled=True),
        {
            "labeled_indices_sha256": "1" * 64,
            "split_manifest_sha256": "2" * 64,
            "source_receivers": list(FROZEN_RCAT_SOURCE_RECEIVER_IDS),
        },
    )
    assert bound["source_labeled_indices_sha256"] == "1" * 64
    warm = strict_rcat_warm_start(
        model,
        _BindingModel().state_dict(),
        baseline_path="base.pth",
        baseline_sha256="a" * 64,
        checkpoint_epoch=40,
        checkpoint_role="training_final_only",
    )
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False


def test_local4_binding_and_launcher_dry_run_are_frozen() -> None:
    global_order = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    local_order = ["20-15", "20-19", "6-15", "8-20"]
    binding = resolve_rcat_local_head_class_binding(
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
        remap_rcat_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]),
        torch.tensor([3, 0, 2]),
    )

    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_rcat12_20260810_v1}"' in launcher_text
    assert "--phase1_rcat_frozen_mode true" in launcher_text
    assert "--phase1_rcat_enabled false --lambda_rcat 0" in launcher_text
    assert "--phase1_rcat_enabled true --lambda_rcat 0.02" in launcher_text
    for flag in ("phase1_gd_proto_nll", "phase1_icmt", "phase1_cagm", "phase1_rcrmd"):
        assert f"--{flag}_enabled false" in launcher_text
        assert f"--lambda_{flag.removeprefix('phase1_')} 0" in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12 and {arm for _, arm, _ in calls} == {"C", "G"}
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    relative_launcher = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative_launcher], cwd=CODE_ROOT, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative_launcher, "--dry-run"], cwd=CODE_ROOT, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(lines) == 12
    assert all("phase1_rcat12_20260810_v1" in line for line in lines)
    assert sum("--phase1_rcat_enabled false" in line for line in lines) == 6
    assert sum("--phase1_rcat_enabled true" in line for line in lines) == 6
    assert all("--phase1_rcrmd_enabled false" in line for line in lines)
    assert all("--phase1_icmt_enabled false" in line for line in lines)
    assert all("--phase1_cagm_enabled false" in line for line in lines)


def test_loss_addition_keeps_control_identity() -> None:
    base = torch.tensor(1.5, requires_grad=True)
    assert add_rcat_to_loss(base, None, _config(enabled=False)) is base
