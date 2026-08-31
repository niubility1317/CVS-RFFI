from __future__ import annotations

import math
import inspect
from types import SimpleNamespace

import pytest
import torch

from SSDG.train_ssdg import (
    _build_bicad_xr_concat_augmenter,
    _bicad_xr_cv2_build_optimizers,
    _bicad_xr_cv2_batch_physical_ids,
    _bicad_xr_cv2_batch_counts,
    _bicad_xr_cv2_coverage_snapshot,
    _bicad_xr_cv2_coverage_warmup_complete,
    _bicad_xr_cv2_coverage_plan,
    _bicad_xr_cv2_ema_update,
    _bicad_xr_cv2_margin_values,
    _bicad_xr_cv2_make_plateau_scheduler,
    _bicad_xr_cv2_no_early_freeze_audit,
    _bicad_xr_cv2_require_real_leo_view,
    _bicad_xr_cv2_resolve_epochs,
    _bicad_xr_cv2_select_validation_candidate,
    _bicad_xr_cv2_source_metrics,
    _bicad_xr_cv2_terminal_status,
    _bicad_xr_cv2_dataset_u_sample_ids,
    _bicad_xr_cv2_validation_role_audit,
    _bicad_xr_cv2_vector_drift,
    _resolve_unlabeled_batch_size,
    _train_bicad_xr,
)
from cvsrffi.phase1_bicad_xr.convergence import CoverageLedger


def test_cv2_coverage_plan_uses_half_u_cycle_with_500_update_floor() -> None:
    short = _bicad_xr_cv2_coverage_plan(
        unlabeled_physical_count=16_000,
        source_receiver_count=4,
    )
    long = _bicad_xr_cv2_coverage_plan(
        unlabeled_physical_count=64_000,
        source_receiver_count=4,
    )

    assert short == {
        "unlabeled_physical_count": 16_000,
        "source_receiver_count": 4,
        "unlabeled_per_four_updates": 120,
        "u_cycle_updates": 534,
        "eval_interval_updates": 500,
        "min_activation_updates": 1_600,
        "safety_updates": 6_400,
    }
    assert long == {
        "unlabeled_physical_count": 64_000,
        "source_receiver_count": 4,
        "unlabeled_per_four_updates": 120,
        "u_cycle_updates": 2_134,
        "eval_interval_updates": 1_067,
        "min_activation_updates": 6_400,
        "safety_updates": 25_600,
    }


@pytest.mark.parametrize("count,receiver_count", [(0, 4), (10, 0), (-1, 4)])
def test_cv2_coverage_plan_rejects_invalid_physical_counts(
    count: int,
    receiver_count: int,
) -> None:
    with pytest.raises(ValueError):
        _bicad_xr_cv2_coverage_plan(
            unlabeled_physical_count=count,
            source_receiver_count=receiver_count,
        )


def test_cv2_coverage_plan_accounts_for_five_receiver_structured_batches() -> None:
    plan = _bicad_xr_cv2_coverage_plan(
        unlabeled_physical_count=11_400,
        source_receiver_count=5,
    )

    assert plan["unlabeled_per_four_updates"] == 114
    assert plan["u_cycle_updates"] == 400
    assert plan["min_activation_updates"] == 1_200
    assert plan["safety_updates"] == 4_800


@pytest.mark.parametrize("unlabeled_count", [18, 24, 32])
def test_strict_pair_loader_accepts_frozen_cv2_unlabeled_batch_sizes(
    unlabeled_count: int,
) -> None:
    args = SimpleNamespace(
        strict_pair_concat=True,
        muse_unlabeled_batch_size=unlabeled_count,
    )

    assert _resolve_unlabeled_batch_size(args) == unlabeled_count


def test_cv2_plateau_scheduler_is_frozen_factor_patience_and_floor() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.0e-4)

    scheduler = _bicad_xr_cv2_make_plateau_scheduler(optimizer)

    assert math.isclose(scheduler.factor, 0.3)
    assert scheduler.patience == 3
    assert scheduler.min_lrs == [1.0e-6]


def test_cv2_safety_stop_is_not_scientific_completion() -> None:
    scientific = SimpleNamespace(
        status="SCIENTIFICALLY_CONVERGED",
        should_stop=True,
        scientifically_converged=True,
    )
    safety = SimpleNamespace(
        status="NOT_CONVERGED_SAFETY_STOP",
        should_stop=True,
        scientifically_converged=False,
    )

    assert _bicad_xr_cv2_terminal_status(scientific) == {
        "status": "SCIENTIFICALLY_CONVERGED",
        "scientifically_converged": True,
        "artifacts_allowed": True,
    }
    assert _bicad_xr_cv2_terminal_status(safety) == {
        "status": "NOT_CONVERGED_SAFETY_STOP",
        "scientifically_converged": False,
        "artifacts_allowed": True,
    }


def test_cv2_source_metrics_use_clean_leo_floor_and_margin_penalty() -> None:
    scenarios = {
        "clean": {
            "accuracy": 80.0,
            "floor": 60.0,
            "negative_margin_rate": 0.10,
            "margin_q10": 0.20,
        },
        "leo_clear_weak": {"accuracy": 70.0, "floor": 50.0},
        "leo_low_elev_weak": {"accuracy": 60.0, "floor": 40.0},
        "leo_rain_weak": {"accuracy": 50.0, "floor": 30.0},
    }

    metrics = _bicad_xr_cv2_source_metrics(scenarios)

    assert metrics["clean_bal"] == pytest.approx(0.80)
    assert metrics["leo_scene_floor_bal"] == pytest.approx(0.50)
    assert metrics["receiver_floor"] == pytest.approx(0.80)
    assert metrics["receiver_std"] == 0.0
    assert metrics["negative_margin_rate"] == pytest.approx(0.10)
    assert metrics["margin_q10"] == pytest.approx(0.20)
    assert metrics["s_dg"] == pytest.approx(3.0 / 4.5 - 0.005)


def test_cv2_margin_values_are_true_logit_minus_best_competitor() -> None:
    logits = torch.tensor([[3.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    labels = torch.tensor([0, 0, 1])

    margins = _bicad_xr_cv2_margin_values(logits, labels)

    assert margins.tolist() == pytest.approx([2.0, -1.0, -1.0])


def test_cv2_vector_drift_is_normalized_l2_change() -> None:
    previous = torch.tensor([1.0, 2.0])
    current = torch.tensor([2.0, 4.0])

    assert _bicad_xr_cv2_vector_drift(current, previous) == pytest.approx(1.0)
    assert _bicad_xr_cv2_vector_drift(previous, None) == math.inf


def test_cv2_adversarial_optimizers_are_disjoint_and_use_1p5_lr() -> None:
    encoder_parameter = torch.nn.Parameter(torch.tensor(1.0))
    discriminator_parameter = torch.nn.Parameter(torch.tensor(2.0))

    class Trainer:
        def adversarial_parameter_groups(self):
            return {
                "encoder": (encoder_parameter,),
                "discriminator": (discriminator_parameter,),
            }

    optimizers = _bicad_xr_cv2_build_optimizers(
        Trainer(),
        lr=2.0e-4,
        weight_decay=1.0e-4,
    )

    assert optimizers.encoder.param_groups[0]["lr"] == pytest.approx(2.0e-4)
    assert optimizers.discriminator.param_groups[0]["lr"] == pytest.approx(3.0e-4)
    assert optimizers.encoder.param_groups[0]["params"] == [encoder_parameter]
    assert optimizers.discriminator.param_groups[0]["params"] == [
        discriminator_parameter
    ]


def test_cv2_training_route_reaches_discriminator_then_encoder_steps() -> None:
    source = inspect.getsource(_train_bicad_xr)

    assert "_bicad_xr_cv2_build_optimizers" in source
    assert "if config.adversarial_two_time_scale:" in source
    assert "discriminator_optimizer.step()" in source
    assert source.index("discriminator_optimizer.step()") < source.index(
        "optimizer.step()"
    )


def test_cv2_batch_counts_switch_to_structured_every_four_updates() -> None:
    assert _bicad_xr_cv2_batch_counts(update=1, source_receiver_count=4) == (16, 32)
    assert _bicad_xr_cv2_batch_counts(update=4, source_receiver_count=4) == (24, 24)
    assert _bicad_xr_cv2_batch_counts(update=8, source_receiver_count=5) == (30, 18)


def test_cv2_training_contract_forces_e200_and_disables_all_early_stops() -> None:
    assert _bicad_xr_cv2_resolve_epochs(epochs=3) == 200

    source = inspect.getsource(_train_bicad_xr)
    assert "optimizer_update_stop" in source
    assert "coverage_stop" in source
    assert "wall_clock_stop" in source
    assert "if update >= total_updates:" not in source
    assert 'if source_loro_state["stopped_early"]:' not in source


def test_strict_pair_schedule_reaches_all_real_leo_weak_scenarios_at_epoch_one() -> None:
    augmenter = _build_bicad_xr_concat_augmenter(
        SimpleNamespace(strict_pair_concat=True, seed=392002, sat_view_seed=392002)
    )
    _, stage = augmenter.stage_for_epoch(1)

    assert stage.start_epoch == 1
    assert stage.view_prob == pytest.approx(1.0)
    assert set(stage.scenarios) == {
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }


@pytest.mark.parametrize(
    "view",
    [
        {"applied": False, "scenario": "clean_duplicate"},
        {"applied": True, "scenario": "clean_duplicate"},
        {"applied": True, "scenario": "unknown"},
    ],
)
def test_strict_pair_rejects_non_real_leo_second_views(view) -> None:
    with pytest.raises(ValueError, match="real LEO_WEAK"):
        _bicad_xr_cv2_require_real_leo_view(view)


@pytest.mark.parametrize("scenario", ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"])
def test_strict_pair_accepts_each_real_leo_second_view(scenario: str) -> None:
    assert _bicad_xr_cv2_require_real_leo_view(
        {"applied": True, "scenario": scenario}
    ) == scenario


def test_cv2_coverage_snapshot_uses_batch_physical_ids_and_fails_closed() -> None:
    metadata = {
        "tx_i": torch.tensor([0, 1, 0]),
        "rx_i": torch.tensor([1, 1, 3]),
        "day_i": torch.tensor([1, 1, 2]),
        "eq_i": torch.tensor([0, 0, 0]),
        "sig_i": torch.tensor([10, 11, 12]),
    }
    physical_ids = (
        (0, 1, 1, 0, 10),
        (1, 1, 1, 0, 11),
        (0, 3, 2, 0, 12),
    )
    assert _bicad_xr_cv2_batch_physical_ids(metadata, expected_count=3, role="U") == physical_ids

    ledger = CoverageLedger(
        u_sample_ids=physical_ids,
        l_groups=((0, 1, 1), (1, 1, 1)),
    )
    ledger.record_u(physical_ids[:2])
    ledger.record_l(((0, 1, 1), (1, 1, 1)))
    snapshot = _bicad_xr_cv2_coverage_snapshot(ledger)

    assert snapshot["u_cumulative_visits"] == 2
    assert snapshot["u_unique_samples"] == 2
    assert snapshot["u_unique_coverage"] == pytest.approx(2.0 / 3.0)
    assert snapshot["l_min_exposure"] == 1
    assert len(snapshot["l_group_exposures"]) == 2

    with pytest.raises(ValueError, match="physical sample ID"):
        _bicad_xr_cv2_batch_physical_ids(
            {"tx_i": torch.tensor([0]), "rx_i": torch.tensor([1])},
            expected_count=1,
            role="U",
        )


def test_cv2_coverage_uses_label_free_base_index_for_muse_u_batches() -> None:
    metadata = {
        "base_index": torch.tensor([17, 23, 41]),
        "rx_i": torch.tensor([1, 1, 3]),
        "day_i": torch.tensor([1, 1, 2]),
        "eq_i": torch.tensor([0, 0, 0]),
        "sig_i": torch.tensor([10, 11, 12]),
        "tx_label_visible": torch.tensor([False, False, False]),
    }
    expected = (
        ("base_index", 17),
        ("base_index", 23),
        ("base_index", 41),
    )

    assert _bicad_xr_cv2_batch_physical_ids(metadata, expected_count=3, role="U") == expected

    subset = SimpleNamespace(selected=[17, 23, 41])
    muse_view = SimpleNamespace(base=subset)
    assert _bicad_xr_cv2_dataset_u_sample_ids(muse_view) == expected

    ledger = CoverageLedger(u_sample_ids=expected, l_groups=((0, 1, 1),))
    ledger.record_u(expected)
    assert _bicad_xr_cv2_coverage_snapshot(ledger)["u_unique_coverage"] == pytest.approx(1.0)


def test_cv2_coverage_warmup_precedes_plateau_scheduler() -> None:
    ledger = CoverageLedger(
        u_sample_ids=("u0", "u1"),
        l_groups=((0, 1, 1),),
    )
    assert not _bicad_xr_cv2_coverage_warmup_complete(ledger)
    ledger.record_u(("u0", "u1"))
    assert not _bicad_xr_cv2_coverage_warmup_complete(ledger)
    ledger.record_l(((0, 1, 1),))
    assert _bicad_xr_cv2_coverage_warmup_complete(ledger)


def test_cv2_validation_roles_are_physical_disjoint() -> None:
    records = [
        SimpleNamespace(tx_i=0, rx_i=4, day_i=1, eq_i=0, sig_i=index)
        for index in range(4)
    ]
    cal_loader = SimpleNamespace(dataset=SimpleNamespace(index=records[:2]))
    select_loader = SimpleNamespace(dataset=SimpleNamespace(index=records[2:]))

    audit = _bicad_xr_cv2_validation_role_audit(cal_loader, select_loader)

    assert audit["same_loader"] is False
    assert audit["physical_id_overlap_count"] == 0
    assert audit["v_cal_size"] == 2
    assert audit["v_select_size"] == 2

    with pytest.raises(ValueError, match="V_cal/V_select"):
        _bicad_xr_cv2_validation_role_audit(cal_loader, cal_loader)


def test_cv2_ema_candidate_and_one_shot_validation_selection() -> None:
    first = _bicad_xr_cv2_ema_update(
        None,
        {"weight": torch.tensor([1.0]), "step": torch.tensor(1, dtype=torch.long)},
        decay=0.5,
    )
    second = _bicad_xr_cv2_ema_update(
        first,
        {"weight": torch.tensor([3.0]), "step": torch.tensor(2, dtype=torch.long)},
        decay=0.5,
    )

    assert second["weight"].item() == pytest.approx(2.0)
    assert second["step"].item() == 2
    selected, scores = _bicad_xr_cv2_select_validation_candidate(
        {"final": 0.40, "ema": 0.55, "swad": 0.50}
    )
    assert selected == "ema"
    assert scores == {"final": 0.40, "ema": 0.55, "swad": 0.50}


def test_cv2_no_early_freeze_audit_is_observable_and_fails_closed() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    audit = _bicad_xr_cv2_no_early_freeze_audit(
        [parameter], enabled=True, epoch=1
    )
    assert audit["enabled"] is True
    assert audit["all_trainable"] is True
    assert audit["frozen_parameter_count"] == 0

    parameter.requires_grad = False
    with pytest.raises(RuntimeError, match="no_early_freeze"):
        _bicad_xr_cv2_no_early_freeze_audit(
            [parameter], enabled=True, epoch=2
        )
