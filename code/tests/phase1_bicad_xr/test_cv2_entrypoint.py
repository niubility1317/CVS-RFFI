from __future__ import annotations

import math
import inspect
from types import SimpleNamespace

import pytest
import torch

from SSDG.train_ssdg import (
    _bicad_xr_cv2_build_optimizers,
    _bicad_xr_cv2_batch_counts,
    _bicad_xr_cv2_coverage_plan,
    _bicad_xr_cv2_margin_values,
    _bicad_xr_cv2_make_plateau_scheduler,
    _bicad_xr_cv2_source_metrics,
    _bicad_xr_cv2_terminal_status,
    _bicad_xr_cv2_vector_drift,
    _resolve_unlabeled_batch_size,
    _train_bicad_xr,
)


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
    assert "discriminator_optimizer.step()" in source
    assert source.index("discriminator_optimizer.step()") < source.index(
        "optimizer.step()"
    )


def test_cv2_batch_counts_switch_to_structured_every_four_updates() -> None:
    assert _bicad_xr_cv2_batch_counts(update=1, source_receiver_count=4) == (16, 32)
    assert _bicad_xr_cv2_batch_counts(update=4, source_receiver_count=4) == (24, 24)
    assert _bicad_xr_cv2_batch_counts(update=8, source_receiver_count=5) == (30, 18)
