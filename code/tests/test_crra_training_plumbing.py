import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import (  # noqa: E402
    _crra_satellite_aux_regularizers_active,
    _crra_satellite_kl_active,
    _resolve_phase1_terminal_status,
    _resolve_sat_training_mode,
    build_optimizer_with_crra_groups,
    build_arg_parser,
    resolve_crra_satellite_kl_weight,
    set_crra_optimizer_learning_rate,
    split_tx_rx_day_1_7_2_roles,
)
from cvsrffi.crra_training import (  # noqa: E402
    crra_gate_scale,
    validate_crra_phase1_config,
    validate_crra_phase1_scenarios,
)
from cvsrffi.losses import (  # noqa: E402
    crra_nuisance_huber_loss,
    crra_satellite_shell_loss,
)


def test_crra_schedule_has_identity_ramp_and_fixed_tail():
    assert crra_gate_scale(1) == 0.0
    assert crra_gate_scale(16) == 0.0
    assert 0.0 < crra_gate_scale(30) < 1.0
    assert crra_gate_scale(47) == 1.0


def test_phase1_crra_legacy_defaults_and_leo_controls_are_explicit():
    args = build_arg_parser().parse_args(["--output_dir", "x"])
    assert args.sat_train_scenario == "mixed_orbit"
    assert args.crra_scenario == "mixed_orbit"
    assert args.crra_target_adapter is False
    assert args.lambda_crra_pair == pytest.approx(0.05)
    assert args.lambda_crra_sat_kl == pytest.approx(0.0)
    assert args.lambda_crra_energy == pytest.approx(0.001)
    assert args.lambda_crra_nuisance == pytest.approx(0.02)
    assert args.lambda_crra_condition_tx_adv == pytest.approx(0.02)
    assert args.crra_s3_lr_scale == pytest.approx(0.25)
    assert args.crra_support_tau == pytest.approx(1.0)


def test_phase1_crra_accepts_leo_weak_family_and_rejects_target_adapter():
    validate_crra_phase1_config(SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=False))
    validate_crra_phase1_scenarios(
        ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        crra_scenario="leo_weak",
    )
    with pytest.raises(ValueError, match="target adapter"):
        validate_crra_phase1_config(SimpleNamespace(crra_scenario="leo_weak", crra_target_adapter=True))


def test_phase1_crra_rejects_scenarios_outside_the_selected_leo_family():
    with pytest.raises(ValueError, match="leo_weak"):
        validate_crra_phase1_scenarios(
            ["leo_clear_weak", "leo_low_elev_weak", "mixed_orbit"],
            crra_scenario="leo_weak",
        )


def test_nuisance_loss_uses_valid_same_view_metadata_only():
    pred = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [2.0, 4.0, 6.0]])
    valid = torch.tensor([True, False, True])
    loss, info = crra_nuisance_huber_loss(pred, target, valid)
    assert info["valid_count"] == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_nuisance_loss_is_zero_with_no_valid_same_view_targets():
    pred = torch.zeros(2, 3, requires_grad=True)
    loss, info = crra_nuisance_huber_loss(pred, None, None)
    assert float(loss.detach()) == 0.0
    assert info["valid_count"] == 0
    loss.backward()
    assert pred.grad is not None


def test_crra_satellite_kl_is_active_only_after_its_stage_scale():
    assert not _crra_satellite_kl_active(
        0.0,
        use_crra=True,
        crra_stage_scale=0.0,
        crra_sat_kl_weight=0.05,
    )
    assert _crra_satellite_kl_active(
        0.0,
        use_crra=True,
        crra_stage_scale=0.5,
        crra_sat_kl_weight=0.05,
    )
    assert _crra_satellite_kl_active(
        0.05,
        use_crra=False,
        crra_stage_scale=0.0,
        crra_sat_kl_weight=0.0,
    )


def test_crra_satellite_kl_has_one_effective_weight():
    assert resolve_crra_satellite_kl_weight(0.05, 0.0, use_crra=True) == pytest.approx(0.05)
    assert resolve_crra_satellite_kl_weight(0.0, 0.05, use_crra=True) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="only one"):
        resolve_crra_satellite_kl_weight(0.05, 0.05, use_crra=True)


def test_crra_optimizer_enters_quarter_rate_at_epoch_47_only():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Linear(2, 2)
            self.crra_adapter = nn.Linear(2, 2)

    optimizer = build_optimizer_with_crra_groups(
        TinyModel(),
        base_lr=2e-4,
        weight_decay=1e-4,
        use_crra=True,
    )
    set_crra_optimizer_learning_rate(
        optimizer,
        epoch=46,
        base_lr=2e-4,
        start_epoch=17,
        ramp_epochs=30,
        s3_lr_scale=0.25,
    )
    rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"core": pytest.approx(2e-4), "crra": pytest.approx(2e-4)}
    set_crra_optimizer_learning_rate(
        optimizer,
        epoch=47,
        base_lr=2e-4,
        start_epoch=17,
        ramp_epochs=30,
        s3_lr_scale=0.25,
    )
    rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"core": pytest.approx(2e-4), "crra": pytest.approx(5e-5)}


def test_nonformal_valid_checkpoint_becomes_artifacts_complete_despite_diagnostic_gates():
    status = _resolve_phase1_terminal_status(
        tail_stopped=False,
        export_failed=False,
        final_blocked=False,
        selected_checkpoint_exists=True,
        heldout_eval_status="COMPLETE",
        p0_mechanisms_ready=False,
        p1_mechanisms_ready=False,
        endpoint_export_ready=False,
        enforce_promotion_gates=False,
    )
    assert status == "ARTIFACTS_COMPLETE"


def test_nuisance_loss_rejects_field_dimension_drift():
    with pytest.raises(ValueError, match="same fixed field dimension"):
        crra_nuisance_huber_loss(
            torch.zeros(2, 8),
            torch.zeros(2, 9),
            torch.tensor([True, True]),
        )


def test_latest_guidance_resolves_pair_masked_concat_to_clean_anchor_plus_sat_aux():
    args = SimpleNamespace(
        sat_training_mode="concat_masked",
        use_concat_sat_channel_aug=False,
        concat_sat_ce_only=False,
    )
    assert _resolve_sat_training_mode(args) == "concat_masked"
    assert args.use_concat_sat_channel_aug is True
    assert args.concat_sat_ce_only is True
    assert not _crra_satellite_aux_regularizers_active(args)


def test_satellite_shell_loss_allows_bounded_class_shift():
    clean_z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    clean_y = torch.tensor([0, 1, 0, 1])
    sat_z = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    sat_y = torch.tensor([0, 1])
    loss, info = crra_satellite_shell_loss(
        clean_z,
        clean_y,
        sat_z,
        sat_y,
        shell_width_rad=0.01,
    )
    assert float(loss.detach()) > 0.0
    assert info["valid_count"] == 2
    loss.backward()
    assert sat_z.grad is not None


def test_current_phase1_source_roles_are_four_way_and_disjoint():
    rows = []
    for tx_i in range(2):
        for rx_i in range(2):
            for sig_i in range(100):
                rows.append(
                    SimpleNamespace(
                        tx_i=tx_i,
                        rx_i=rx_i,
                        day_i=0,
                        eq_i=0,
                        sig_i=sig_i,
                    )
                )
    dataset = SimpleNamespace(index=rows)
    labeled, unlabeled, v_cal, v_select = split_tx_rx_day_1_7_2_roles(
        dataset,
        labeled_ratio=0.07,
        unlabeled_ratio=0.63,
        source_cal_ratio=0.15,
        source_select_ratio=0.15,
    )
    buckets = [set(labeled), set(unlabeled), set(v_cal), set(v_select)]
    assert sum(map(len, buckets)) == len(rows)
    assert all(not (buckets[i] & buckets[j]) for i in range(4) for j in range(i + 1, 4))
    assert len(v_cal) > 0
    assert len(v_select) > 0
    assert len(labeled) / (len(labeled) + len(unlabeled)) <= 0.1
