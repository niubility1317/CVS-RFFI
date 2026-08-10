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

from cvsrffi.phase1_recte import (  # noqa: E402
    FROZEN_RECTE_CLASS_IDS,
    FROZEN_RECTE_CELL_COUNT,
    FROZEN_RECTE_LAMBDA,
    FROZEN_RECTE_OPTIMIZER_TYPE,
    FROZEN_RECTE_PAIR_DENOMINATOR,
    FROZEN_RECTE_SCENARIOS,
    FROZEN_RECTE_SOURCE_RECEIVER_IDS,
    RECTEConfig,
    RECTEConfigurationError,
    RECTERuntimeError,
    RECTE_RECEIPT_SCHEMA,
    add_recte_to_loss,
    bind_recte_optimizer_initial_state,
    bind_recte_source_data_order,
    recte_aux_gradient_audit,
    recte_config_receipt,
    recte_loss,
    recte_shared_encoder_and_head_parameters,
    remap_recte_local_labels_to_head_rows,
    resolve_recte_classifier_weight,
    resolve_recte_local_head_class_binding,
    strict_recte_warm_start,
    update_recte_common_batch_sequence_receipt,
    update_recte_gradient_audit_receipt,
    update_recte_receipt,
    validate_recte_args,
    validate_recte_binding,
    validate_recte_terminal_receipt,
)


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_recte12_20260810.sh"


def _frozen_args(*, enabled: bool = True, epochs: int = 40) -> SimpleNamespace:
    """Arguments consumed by the frozen RECTE validator, with all peers off."""

    return SimpleNamespace(
        phase1_recte_frozen_mode=True,
        phase1_recte_enabled=enabled,
        lambda_recte=FROZEN_RECTE_LAMBDA if enabled else 0.0,
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
        sat_train_scenarios=",".join(FROZEN_RECTE_SCENARIOS),
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
        phase1_rcat_frozen_mode=False,
        phase1_rcat_enabled=False,
        lambda_rcat=0.0,
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
            if self.id_backbone.cls_head.head.bias is not None:
                self.id_backbone.cls_head.head.bias.zero_()

    def paired_output(self, x: torch.Tensor) -> dict[str, torch.Tensor | str]:
        z_id = self.id_backbone.cls_head.joint_proj(self.id_backbone.encoder(x))
        logits = self.id_backbone.cls_head.head(z_id)
        # RECTE audits the exact head-row functional path against live logits.
        return {
            "z_id": z_id,
            "z_id_key": "feat_joint",
            "tx_logits": logits,
            "tx_logits_live": logits,
            "tx_logits_functional": logits,
        }


def _config(*, enabled: bool = True) -> RECTEConfig:
    return RECTEConfig(True, enabled, FROZEN_RECTE_LAMBDA if enabled else 0.0)


def _logits_from_margins(labels: torch.Tensor, margins: list[float]) -> torch.Tensor:
    """Set local4 true-vs-other margins exactly (true logit=m+log(3))."""

    log_three = float(torch.log(torch.tensor(3.0)).item())
    rows: list[torch.Tensor] = []
    for label, margin in zip(labels.tolist(), margins):
        row = torch.zeros(4, dtype=torch.float32)
        row[int(label)] = float(margin) + log_three
        rows.append(row)
    return torch.stack(rows).requires_grad_(True)


def _occupied_two() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor([0, 1])
    receivers = torch.tensor([0, 1])
    # delta=(clean margin-leo margin)=(2,0), so q_ab=4 for one pair.
    clean = _logits_from_margins(labels, [2.0, 3.0])
    leo = _logits_from_margins(labels, [0.0, 3.0])
    return clean, leo, labels, receivers


def _recte_call(
    clean_logits: torch.Tensor,
    leo_feat_joint: torch.Tensor,
    labels: torch.Tensor,
    receivers: torch.Tensor,
    *,
    live_leo_logits: torch.Tensor | None = None,
    classifier_head: torch.nn.Module | None = None,
    expected_receiver_ids: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Call the frozen public RECTE functional-head loss contract."""

    if classifier_head is None:
        classifier_head = torch.nn.Linear(4, 4, bias=True).to(
            device=leo_feat_joint.device, dtype=leo_feat_joint.dtype
        )
        with torch.no_grad():
            classifier_head.weight.copy_(torch.eye(4, dtype=leo_feat_joint.dtype, device=leo_feat_joint.device))
            classifier_head.bias.zero_()
    if live_leo_logits is None:
        live_leo_logits = leo_feat_joint.detach().clone()
    return recte_loss(
        clean_logits,
        leo_feat_joint,
        live_leo_logits,
        labels,
        receivers,
        FROZEN_RECTE_SOURCE_RECEIVER_IDS if expected_receiver_ids is None else expected_receiver_ids,
        classifier_head=classifier_head,
    )


def _all_cells() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor(list(FROZEN_RECTE_CLASS_IDS) * len(FROZEN_RECTE_SOURCE_RECEIVER_IDS))
    receivers = torch.tensor(
        [receiver for receiver in FROZEN_RECTE_SOURCE_RECEIVER_IDS for _ in FROZEN_RECTE_CLASS_IDS]
    )
    return labels, receivers


def _metadata(rows: int, offset: int = 0) -> dict[str, torch.Tensor]:
    return {
        "base_index": torch.arange(offset, offset + rows),
        "sig_i": torch.arange(offset, offset + rows) + 10000,
        "rx_i": torch.repeat_interleave(
            torch.arange(len(FROZEN_RECTE_SOURCE_RECEIVER_IDS)), len(FROZEN_RECTE_CLASS_IDS)
        )[:rows],
        "same_physical": torch.ones(rows, dtype=torch.bool),
    }


def _sealed_receipt(*, enabled: bool) -> dict[str, object]:
    receipt = recte_config_receipt(_config(enabled=enabled))
    receipt.update(
        {
            "baseline_sha256": "a" * 64,
            "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": "b" * 64,
            "source_labeled_indices_sha256": "c" * 64,
            "source_split_manifest_sha256": "d" * 64,
            "source_receiver_ids": list(FROZEN_RECTE_SOURCE_RECEIVER_IDS),
            "source_receiver_count": len(FROZEN_RECTE_SOURCE_RECEIVER_IDS),
            "source_receiver_ids_sha256": "e" * 64,
            "optimizer_type": FROZEN_RECTE_OPTIMIZER_TYPE,
            "optimizer_initial_state_sha256": "f" * 64,
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "expected_tx_class_ids": list(FROZEN_RECTE_CLASS_IDS),
            "common_l_base_head_input_path_verified": True,
            "common_batch_sequence_sha256": "1" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scenario: 0 for scenario in FROZEN_RECTE_SCENARIOS},
        }
    )
    return receipt


def _common_bind(receipt: dict[str, object], *, epoch: int, batch_index: int, scenario: str) -> dict[str, object]:
    labels, receivers = _all_cells()
    return update_recte_common_batch_sequence_receipt(
        receipt,
        epoch=epoch,
        batch_index=batch_index,
        scenario=scenario,
        source_tx_labels=labels,
        source_rx_labels=receivers,
        metadata=_metadata(int(labels.numel()), offset=batch_index * 100),
    )


def _build_g_terminal_receipt(
    *, audit_scenes: tuple[str, ...] | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    model = _BindingModel().train()
    labels, receivers = _all_cells()
    # Cell-wise deltas are strictly varied, so every scene has positive pairs.
    clean = _logits_from_margins(labels, [1.0 + 0.01 * i for i in range(labels.numel())])
    leo = _logits_from_margins(labels, [1.0 - 0.01 * i for i in range(labels.numel())])
    out_clean = model.paired_output(clean)
    out_leo = model.paired_output(leo)
    validate_recte_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RECTE_CLASS_IDS,
        expected_receiver_ids=FROZEN_RECTE_SOURCE_RECEIVER_IDS,
    )
    aux, info = _recte_call(
        out_clean["tx_logits"],
        out_leo["z_id"],
        labels,
        receivers,
        live_leo_logits=out_leo["tx_logits"],
        classifier_head=model.id_backbone.cls_head.head,
    )
    receipt = _sealed_receipt(enabled=True)
    selected_audits = set(FROZEN_RECTE_SCENARIOS if audit_scenes is None else audit_scenes)
    for index, scenario in enumerate(FROZEN_RECTE_SCENARIOS, start=1):
        receipt = _common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
        receipt = update_recte_receipt(receipt, info, scenario=scenario, epoch=1, batch_index=index)
        if scenario in selected_audits:
            audit = recte_aux_gradient_audit(
                aux, out_leo["z_id"], recte_shared_encoder_and_head_parameters(model)
            )
            receipt = update_recte_gradient_audit_receipt(
                receipt, audit, scenario=scenario
            )
    return receipt, info


def test_formula_hand_calculable_pair_denominator_and_low_delta_only_gradient() -> None:
    clean, leo, labels, receivers = _occupied_two()
    loss, info = _recte_call(clean, leo, labels, receivers)
    assert float(loss.detach()) == pytest.approx(4.0 / FROZEN_RECTE_PAIR_DENOMINATOR, abs=1e-6)
    assert int(info["global_denominator"]) == FROZEN_RECTE_PAIR_DENOMINATOR == 378
    assert float(info["fixed_scale"]) == pytest.approx(1.0 / FROZEN_RECTE_PAIR_DENOMINATOR)
    assert int(info["occupied_unordered_pair_count"]) == 1
    assert int(info["positive_tail_pair_count"]) == 1
    assert info["no_active_pair_renormalization"] is True
    assert info["functional_logits_equal_live"] is True
    assert info["functional_head_readout_count"] == 1
    assert info["functional_head_parameters_stopgrad"] is True
    assert info["tail_only_lower_delta_gradient"] is True
    assert info["empty_pair_zero_contribution"] is True
    loss.backward()
    assert clean.grad is None
    assert leo.grad is not None
    # low-delta endpoint (row 0: delta=-2) receives VJP; high endpoint is stopgrad.
    assert int(torch.count_nonzero(leo.grad[0]).item()) > 0
    assert int(torch.count_nonzero(leo.grad[1]).item()) == 0


def test_empty_pairs_keep_fixed_378_scale_and_zero_contribution() -> None:
    clean, leo, labels, receivers = _occupied_two()
    loss, info = _recte_call(clean, leo, labels, receivers)
    cells = info["cells"]
    assert len(cells) == FROZEN_RECTE_CELL_COUNT == 28
    assert cells["rx0|tx0"]["n_rc"] == 1
    assert cells["rx1|tx1"]["n_rc"] == 1
    assert cells["rx6|tx3"]["n_rc"] == 0
    assert cells["rx6|tx3"].get("loss_contribution", 0.0) == 0.0
    assert int(info["global_denominator"]) == FROZEN_RECTE_PAIR_DENOMINATOR == 378
    assert float(loss.detach()) == pytest.approx(4.0 / FROZEN_RECTE_PAIR_DENOMINATOR, abs=1e-6)


def test_rx_class_permutation_and_sample_reordering_preserve_loss() -> None:
    clean, leo, labels, receivers = _occupied_two()
    direct, _ = _recte_call(clean, leo, labels, receivers)
    class_perm = torch.tensor([2, 0, 3, 1])
    rx_perm = torch.tensor([4, 6, 0, 5, 1, 3, 2])
    order = torch.tensor([1, 0])
    permuted_clean = clean.detach().index_select(1, torch.argsort(class_perm)).index_select(0, order).requires_grad_(True)
    permuted_leo = leo.detach().index_select(1, torch.argsort(class_perm)).index_select(0, order).requires_grad_(True)
    permuted, _ = _recte_call(
        permuted_clean,
        permuted_leo,
        class_perm[labels].index_select(0, order),
        rx_perm[receivers].index_select(0, order),
    )
    assert torch.allclose(direct.detach(), permuted.detach(), atol=1e-6)


def test_nonfinite_and_bad_allowlist_fail_closed() -> None:
    clean, leo, labels, receivers = _occupied_two()
    bad_clean = clean.detach().clone()
    bad_clean[0, 0] = float("nan")
    with pytest.raises(RECTERuntimeError, match="non-finite|nonfinite"):
        _recte_call(bad_clean.requires_grad_(True), leo.detach().requires_grad_(True), labels, receivers)
    bad_leo = torch.full_like(leo, float("inf"))
    with pytest.raises(RECTERuntimeError, match="non-finite|nonfinite"):
        _recte_call(clean.detach().requires_grad_(True), bad_leo.requires_grad_(True), labels, receivers)
    with pytest.raises(RECTEConfigurationError, match="receiver|frozen"):
        _recte_call(clean, leo, labels, receivers, expected_receiver_ids=tuple(range(6)))


def test_configuration_freezes_c_g_and_disables_legacy_routes() -> None:
    assert validate_recte_args(_frozen_args(enabled=True)) == _config(enabled=True)
    assert validate_recte_args(_frozen_args(enabled=False)) == _config(enabled=False)
    for name, value in (
        ("lambda_recte", 0.01),
        ("epochs", 39),
        ("phase1_gd_proto_nll_enabled", True),
        ("phase1_icmt_enabled", True),
        ("phase1_cagm_enabled", True),
        ("phase1_rcrmd_enabled", True),
        ("phase1_rcat_enabled", True),
        ("use_unlabeled", True),
    ):
        bad = _frozen_args()
        setattr(bad, name, value)
        with pytest.raises(RECTEConfigurationError):
            validate_recte_args(bad)


def test_functional_live_head_binding_and_aux_head_zero_vjp() -> None:
    model = _BindingModel().train()
    labels, receivers = _all_cells()
    clean = _logits_from_margins(labels, [1.0] * labels.numel())
    leo = _logits_from_margins(labels, [0.01 * i for i in range(labels.numel())])
    out_clean, out_leo = model.paired_output(clean), model.paired_output(leo)
    validate_recte_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RECTE_CLASS_IDS,
        expected_receiver_ids=FROZEN_RECTE_SOURCE_RECEIVER_IDS,
    )
    assert resolve_recte_classifier_weight(model) is model.id_backbone.cls_head.head.weight
    aux, _ = _recte_call(
        out_clean["tx_logits"],
        out_leo["z_id"],
        labels,
        receivers,
        live_leo_logits=out_leo["tx_logits"],
        classifier_head=model.id_backbone.cls_head.head,
    )
    audit = recte_aux_gradient_audit(aux, out_leo["z_id"], recte_shared_encoder_and_head_parameters(model))
    assert audit["raw_unscaled"] is True and audit["diagnostic_only"] is True
    assert float(audit["shared_encoder"]["norm"]) > 0.0
    head = audit.get("classifier_head", audit.get("head_aux", {}))
    assert float(head.get("nonzero_parameters", 0.0)) == 0.0
    aux.backward()
    assert model.id_backbone.cls_head.head.weight.grad is None

    mismatch = out_leo["tx_logits"].detach() + 1.0
    with pytest.raises(RECTERuntimeError, match="functional|live|head|logit"):
        _recte_call(
            out_clean["tx_logits"],
            out_leo["z_id"],
            labels,
            receivers,
            live_leo_logits=mismatch,
            classifier_head=model.id_backbone.cls_head.head,
        )


def test_common_c_has_zero_aux_and_g_closes_three_scene_28_cell_terminals() -> None:
    control = _sealed_receipt(enabled=False)
    for index, scenario in enumerate(FROZEN_RECTE_SCENARIOS, start=1):
        control = _common_bind(control, epoch=1, batch_index=index, scenario=scenario)
    terminal = validate_recte_terminal_receipt(control)
    assert terminal["schema"] == RECTE_RECEIPT_SCHEMA
    assert terminal["recte_terminal_contract_passed"] is True
    assert int(terminal.get("recte_batches", 0)) == 0
    assert float(terminal.get("recte_loss_sum", 0.0)) == 0.0
    assert not terminal.get("recte_scenes")
    assert not terminal.get("recte_gradient_audit_scenes")

    receipt, info = _build_g_terminal_receipt()
    assert int(info["positive_tail_pair_count"]) > 0
    terminal = validate_recte_terminal_receipt(receipt)
    assert terminal["recte_terminal_contract_passed"] is True
    assert int(terminal["recte_total_rows"]) == 84
    assert all(len(terminal["recte_common_cells"][scene]) == 28 for scene in FROZEN_RECTE_SCENARIOS)
    assert all(set(terminal["recte_scenes"][scene]) >= {
        "batches", "rows", "occupied_unordered_pair_count", "positive_tail_pair_count",
        "functional_logits_equal_live_batches",
    } for scene in FROZEN_RECTE_SCENARIOS)
    assert all(
        int(terminal["recte_scenes"][scene]["positive_tail_pair_count"]) > 0
        for scene in FROZEN_RECTE_SCENARIOS
    )


def test_material_receipt_mismatch_fails_closed() -> None:
    receipt, _ = _build_g_terminal_receipt()
    drifted = dict(receipt)
    drifted["recte_loss_sum"] = float(drifted.get("recte_loss_sum", 0.0)) + 1.0
    with pytest.raises(RECTERuntimeError, match="loss|counter|receipt|ledger"):
        validate_recte_terminal_receipt(drifted)


def test_terminal_rejects_three_positive_scenes_with_only_clear_vjp_audit() -> None:
    receipt, _ = _build_g_terminal_receipt(audit_scenes=(FROZEN_RECTE_SCENARIOS[0],))
    assert all(
        int(receipt["recte_scenes"][scene]["positive_tail_pair_count"]) > 0
        for scene in FROZEN_RECTE_SCENARIOS
    )
    assert set(receipt["recte_gradient_audit_scenes"]) == {FROZEN_RECTE_SCENARIOS[0]}
    with pytest.raises(RECTERuntimeError, match="per-scene|VJP|audit|incomplete"):
        validate_recte_terminal_receipt(receipt)


@pytest.mark.parametrize("missing_scene", FROZEN_RECTE_SCENARIOS)
def test_terminal_rejects_any_missing_scene_vjp_audit(missing_scene: str) -> None:
    selected = tuple(scene for scene in FROZEN_RECTE_SCENARIOS if scene != missing_scene)
    receipt, _ = _build_g_terminal_receipt(audit_scenes=selected)
    assert missing_scene not in receipt["recte_gradient_audit_scenes"]
    with pytest.raises(RECTERuntimeError, match="per-scene|VJP|audit|incomplete"):
        validate_recte_terminal_receipt(receipt)


def test_terminal_rejects_tampered_scene_vjp_audit() -> None:
    receipt, _ = _build_g_terminal_receipt()
    tampered = dict(receipt)
    tampered["recte_gradient_audit_scenes"] = {
        str(scene): dict(audit)
        for scene, audit in dict(receipt["recte_gradient_audit_scenes"]).items()
    }
    tampered_scene = FROZEN_RECTE_SCENARIOS[1]
    tampered_audit = dict(tampered["recte_gradient_audit_scenes"][tampered_scene])
    tampered_audit["shared_encoder"] = dict(tampered_audit["shared_encoder"])
    tampered_audit["shared_encoder"]["norm"] = 0.0
    tampered["recte_gradient_audit_scenes"][tampered_scene] = tampered_audit
    with pytest.raises(RECTERuntimeError, match="VJP|zero|non-finite|audit"):
        validate_recte_terminal_receipt(tampered)


def test_duplicate_same_scene_vjp_audit_fails_closed() -> None:
    receipt, _ = _build_g_terminal_receipt()
    scene = FROZEN_RECTE_SCENARIOS[0]
    audit = receipt["recte_gradient_audit_scenes"][scene]
    with pytest.raises(RECTERuntimeError, match="once|scene|audit"):
        update_recte_gradient_audit_receipt(receipt, audit, scenario=scene)


def test_common_binding_and_narrow_lite_d_no_query_regression() -> None:
    from model_dual_cvsincnet import build_dual_model

    torch.manual_seed(31)
    model = build_dual_model(num_classes=4, num_domains=1, dataset="wisig", input_len=128, model_variant="lite_d").train()
    x_clean, x_leo = torch.randn(8, 2, 128), torch.randn(8, 2, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    receivers = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    out_clean = model(x_clean, y_tx=labels, domain_labels=None, return_aux=True)
    out_leo = model(x_leo, y_tx=labels, domain_labels=None, return_aux=True)
    assert out_clean["z_id_key"] == "feat_joint" and out_leo["z_id_key"] == "feat_joint"
    validate_recte_binding(
        model=model,
        out_clean=out_clean,
        out_leo=out_leo,
        tx_labels=labels,
        source_rx_labels=receivers,
        expected_class_ids=FROZEN_RECTE_CLASS_IDS,
        expected_receiver_ids=FROZEN_RECTE_SOURCE_RECEIVER_IDS,
    )
    loss, info = _recte_call(
        out_clean["tx_logits"],
        out_leo["z_id"],
        labels,
        receivers,
        live_leo_logits=out_leo["tx_logits"],
        classifier_head=model.id_backbone.cls_head.head,
    )
    assert torch.isfinite(loss.detach()) and info["functional_logits_equal_live"] is True
    audit = recte_aux_gradient_audit(loss, out_leo["z_id"], recte_shared_encoder_and_head_parameters(model))
    assert float(audit["shared_encoder"]["norm"]) > 0.0
    assert float(audit["classifier_head"]["nonzero_parameters"]) == 0.0
    with pytest.raises(RECTERuntimeError, match="functional|live|logit"):
        _recte_call(
            out_clean["tx_logits"],
            out_leo["z_id"],
            labels,
            receivers,
            live_leo_logits=out_leo["tx_logits"].detach() + 1.0,
            classifier_head=model.id_backbone.cls_head.head,
        )
    assert "query" not in inspect.signature(recte_loss).parameters


def test_common_receipt_binds_physical_cells_and_optimizer_warm_start() -> None:
    labels, receivers = _all_cells()
    control = _sealed_receipt(enabled=False)
    control = _common_bind(control, epoch=1, batch_index=1, scenario=FROZEN_RECTE_SCENARIOS[0])
    events = list(control.get("recte_common_batch_cells", control.get("common_batch_cells", [])))
    assert events
    event = events[0]
    assert event["same_physical_clean_leo"] is True
    expected_order = [
        f"rx{receiver}|tx{class_id}"
        for receiver in FROZEN_RECTE_SOURCE_RECEIVER_IDS
        for class_id in FROZEN_RECTE_CLASS_IDS
    ]
    assert event["cell_order"] == expected_order
    assert len(event["n_rc"]) == FROZEN_RECTE_CELL_COUNT
    assert sum(event["n_rc"].values()) == int(labels.numel())
    assert event["occupied"] == {key: int(event["n_rc"][key]) > 0 for key in expected_order}
    missing = _sealed_receipt(enabled=False)
    with pytest.raises(RECTERuntimeError, match="physical|metadata|same"):
        update_recte_common_batch_sequence_receipt(
            missing, epoch=1, batch_index=1, scenario=FROZEN_RECTE_SCENARIOS[0], source_tx_labels=labels, source_rx_labels=receivers, metadata={"same_physical": torch.ones(labels.numel(), dtype=torch.bool)}
        )
    model = _BindingModel()
    receipt = bind_recte_optimizer_initial_state(_sealed_receipt(enabled=True), torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4))
    assert receipt["optimizer_type"] == FROZEN_RECTE_OPTIMIZER_TYPE and receipt["optimizer_initial_state_empty"] is True
    bound = bind_recte_source_data_order(_sealed_receipt(enabled=True), {"labeled_indices_sha256": "1" * 64, "split_manifest_sha256": "2" * 64, "source_receivers": list(FROZEN_RECTE_SOURCE_RECEIVER_IDS)})
    assert bound["source_labeled_indices_sha256"] == "1" * 64
    warm = strict_recte_warm_start(model, _BindingModel().state_dict(), baseline_path="base.pth", baseline_sha256="a" * 64, checkpoint_epoch=40, checkpoint_role="training_final_only")
    assert warm["strict_model_keys"] is True and warm["optimizer_state_restored"] is False


def test_local4_binding_and_launcher_dry_run_are_frozen() -> None:
    binding = resolve_recte_local_head_class_binding(
        local_class_order=["20-15", "20-19", "6-15", "8-20"],
        source_train_tx=["20-15", "20-19", "6-15", "8-20"],
        checkpoint_train_tx=["20-15", "20-19", "6-15", "8-20"],
        dataset_class_order=["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        local_data_class_count=4,
        checkpoint_head_class_count=4,
        live_head_class_count=4,
    )
    assert binding["local_to_dataset_class_ids"] == [2, 3, 4, 5]
    assert torch.equal(remap_recte_local_labels_to_head_rows(torch.tensor([3, 0, 2]), binding["local_to_head_class_ids"]), torch.tensor([3, 0, 2]))

    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_recte12_20260810_v1}"' in launcher_text
    assert "--phase1_recte_frozen_mode true" in launcher_text
    assert "--phase1_recte_enabled false --lambda_recte 0" in launcher_text
    assert "--phase1_recte_enabled true --lambda_recte 0.02" in launcher_text
    for flag in ("phase1_gd_proto_nll", "phase1_icmt", "phase1_cagm", "phase1_rcrmd", "phase1_rcat"):
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
    assert all("phase1_recte12_20260810_v1" in line for line in lines)
    assert sum("--phase1_recte_enabled false" in line for line in lines) == 6
    assert sum("--phase1_recte_enabled true" in line for line in lines) == 6
    assert all("--phase1_rcrmd_enabled false" in line for line in lines)
    assert all("--phase1_rcat_enabled false" in line for line in lines)
    assert all("--phase1_icmt_enabled false" in line for line in lines)
    assert all("--phase1_cagm_enabled false" in line for line in lines)


def test_loss_addition_keeps_control_identity() -> None:
    base = torch.tensor(1.5, requires_grad=True)
    assert add_recte_to_loss(base, None, _config(enabled=False)) is base


def test_u_v_boundaries_are_zero_forward_and_zero_reflow() -> None:
    receipt = recte_config_receipt(_config(enabled=False))
    assert "ZERO_FORWARD" in str(receipt["u_loader_common_trainer_boundary"])
    assert "ZERO_LOSS_ZERO_BACKWARD" in str(receipt["v_common_trainer_boundary"])
    assert receipt["uses_target_rows"] is False
    assert receipt["uses_proxy_rows"] is False
