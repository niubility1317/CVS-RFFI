from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cvsrffi.ssdg_guard import (
    detect_one_epoch_drop,
    detect_paic_variance_guard,
    guard_minimums_from_args,
    joint_safe_score,
    missing_joint_safe_metrics,
    protected_metric_snapshot,
    sat_protocol_requirement_satisfied,
)


def test_joint_safe_score_prefers_balanced_satellite_epoch_over_clean_only_peak():
    clean_only_peak = {
        "val_tx": 98.095,
        "overall_tx": 89.049,
        "strict_udu": 85.182,
        "receiver_floor": 75.775,
        "sat_mean_tx": 58.380,
        "sat_strict_mean": 57.172,
    }
    balanced_sat_epoch = {
        "val_tx": 98.381,
        "overall_tx": 89.220,
        "strict_udu": 83.270,
        "receiver_floor": 79.525,
        "sat_mean_tx": 77.580,
        "sat_strict_mean": 70.422,
    }

    assert joint_safe_score(balanced_sat_epoch) > joint_safe_score(clean_only_peak)


def test_protected_metric_snapshot_extracts_satellite_mean_and_floors():
    metrics = protected_metric_snapshot(
        val_stats={"tx_acc": 98.0},
        test_stats={"tx_acc": 89.0},
        named_test_stats={
            "test_unseen_day_seen_rx": {"tx_acc": 88.0},
            "test_seen_day_unseen_rx": {"tx_acc": 79.0},
            "test_unseen_day_unseen_rx": {"tx_acc": 83.0},
        },
        sat_test_stats={
            "leo_clear_weak": {
                "aggregate": {"tx_acc": 78.0},
                "strict_udu": 71.0,
                "named": {
                    "test_rx_9": {"tx_acc": 75.0},
                    "test_unseen_day_rx_9": {"tx_acc": 68.0},
                },
            },
            "leo_low_elev_weak": {
                "aggregate": {"tx_acc": 76.0},
                "strict_udu": 69.0,
                "named": {
                    "test_rx_9": {"tx_acc": 73.0},
                    "test_unseen_day_rx_9": {"tx_acc": 66.0},
                },
            },
            "leo_rain_weak": {
                "aggregate": {"tx_acc": 77.0},
                "strict_udu": 70.0,
                "named": {
                    "test_rx_9": {"tx_acc": 74.0},
                    "test_unseen_day_rx_9": {"tx_acc": 67.0},
                },
            },
        },
    )

    assert metrics["strict_udu"] == 83.0
    assert metrics["receiver_floor"] == 79.0
    assert metrics["sat_mean_tx"] == 77.0
    assert metrics["sat_floor_tx"] == 76.0
    assert metrics["sat_strict_mean"] == 70.0
    assert metrics["sat_strict_floor"] == 69.0
    assert metrics["sat_receiver_floor"] == 66.0
    assert metrics["sat_receiver_seen_day_floor"] == 73.0
    assert metrics["sat_receiver_strict_floor"] == 66.0


def test_satellite_protocol_readiness_uses_requirement_implication():
    assert sat_protocol_requirement_satisfied(required=False, actual_disjoint=False)
    assert sat_protocol_requirement_satisfied(required=False, actual_disjoint=True)
    assert sat_protocol_requirement_satisfied(required=True, actual_disjoint=True)
    assert not sat_protocol_requirement_satisfied(required=True, actual_disjoint=False)


def test_joint_safe_requires_satellite_metrics_when_guarded():
    missing_satellite = {
        "val_tx": 98.0,
        "overall_tx": 89.0,
        "strict_udu": 83.0,
        "receiver_floor": 76.0,
    }

    gaps = missing_joint_safe_metrics(missing_satellite, require_satellite=True)

    assert "sat_mean_tx" in gaps
    assert "sat_strict_mean" in gaps
    assert joint_safe_score(missing_satellite, require_satellite=True) == float("-inf")
    assert joint_safe_score(missing_satellite, require_satellite=False) > 0.0


def test_one_epoch_drop_guard_catches_gpu0_style_cliff():
    previous = {
        "strict_udu": 83.270,
        "receiver_floor": 79.525,
        "sat_mean_tx": 77.580,
        "sat_strict_mean": 70.422,
    }
    current = {
        "strict_udu": 77.382,
        "receiver_floor": 74.933,
        "sat_mean_tx": 72.636,
        "sat_strict_mean": 64.568,
    }

    decision = detect_one_epoch_drop(current, previous, threshold_pp=2.0)

    assert decision.fired
    assert decision.details["strict_udu_drop_pp"] > 5.0
    assert decision.details["sat_strict_mean_drop_pp"] > 5.0


def test_paic_variance_guard_catches_high_variance_without_pseudo_precision_collapse():
    previous = {
        "train/w_loss_sat_cls_labeled": 1.811,
        "train/w_loss_sat_cons_labeled": 0.0217,
        "train/loss_domain_labeled": 0.403,
        "train/grad_total": 44.10,
        "train/reliable_ratio": 0.904,
    }
    current = {
        "train/w_loss_sat_cls_labeled": 1.982,
        "train/w_loss_sat_cons_labeled": 0.0235,
        "train/loss_domain_labeled": 0.440,
        "train/grad_total": 47.98,
        "train/reliable_ratio": 0.888,
    }

    decision = detect_paic_variance_guard(
        current,
        previous,
        sat_ce_delta=0.12,
        grad_delta=3.0,
        reliable_drop=0.01,
    )

    assert decision.fired
    assert decision.details["sat_ce_delta"] > 0.12
    assert decision.details["grad_total_delta"] > 3.0
    assert decision.details["pseudo_reliable_drop"] > 0.01


def test_guard_minimums_from_args_ignores_zero_defaults():
    args = SimpleNamespace(
        joint_guard_min_strict_udu=84.0,
        joint_guard_min_receiver_floor=0.0,
        joint_guard_min_sat_mean=70.0,
        joint_guard_min_sat_floor=0.0,
        joint_guard_min_sat_strict_mean=0.0,
        joint_guard_min_sat_strict_floor=0.0,
    )

    assert guard_minimums_from_args(args) == {"strict_udu": 84.0, "sat_mean_tx": 70.0}
