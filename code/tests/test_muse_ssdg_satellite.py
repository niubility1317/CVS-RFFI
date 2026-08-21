from types import SimpleNamespace

import torch

from SSDG import train_ssdg
from cvsrffi import muse_ssdg


def test_satellite_choice_is_stable_for_sample_and_epoch():
    keys = [(1, 2, 3, 4, 5), (1, 2, 3, 4, 6)]

    first = muse_ssdg.select_satellite_student_mask(
        keys, epoch=41, probability=0.5, seed=392002
    )
    second = muse_ssdg.select_satellite_student_mask(
        list(reversed(keys)), epoch=41, probability=0.5, seed=392002
    )

    assert first.tolist() == list(reversed(second.tolist()))


def test_stable_sample_keys_preserve_metadata_identity_order():
    extra = {
        "rx_i": torch.tensor([2, 3]),
        "day_i": torch.tensor([4, 5]),
        "eq_i": torch.tensor([6, 7]),
        "sig_i": torch.tensor([8, 9]),
        "base_index": torch.tensor([10, 11]),
    }

    assert muse_ssdg.stable_sample_keys(extra) == [(2, 4, 6, 8, 10), (3, 5, 7, 9, 11)]


def test_u_satellite_policy_matches_adv3b02_core90_epoch_boundaries():
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(1) == (
        0.30,
        ("leo_clear_weak",),
    )
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(40) == (
        0.30,
        ("leo_clear_weak",),
    )
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(41) == (
        0.60,
        ("leo_low_elev_weak", "leo_rain_weak"),
    )
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(90) == (
        0.60,
        ("leo_low_elev_weak", "leo_rain_weak"),
    )
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(91) == (
        0.80,
        ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"),
    )
    assert muse_ssdg.adv3b02_core90_u_satellite_policy(200)[0] == 0.80


def test_muse_telemetry_exposes_fixed_satellite_and_reliability_fields():
    args = SimpleNamespace(
        output_dir="out",
        seed=392002,
        muse_unlabeled_prototype_weight=0.075,
    )
    row = train_ssdg._build_ssdg_epoch_telemetry_row(
        args=args,
        epoch=41,
        epochs=200,
        lr=1e-3,
        epoch_time_s=1.0,
        phase="pseudo",
        train_logs={
            "muse/high_ratio": 0.50,
            "muse/mid_ratio": 0.25,
            "muse/low_ratio": 0.25,
            "muse/effective_weight": 0.625,
            "muse/head_js": 0.125,
            "muse/proto_update_weight": 0.075,
            "muse/pseudo_precision_diagnostic": "N/A",
        },
        val_stats={},
        test_stats={},
        named_test_stats={},
        sat_test_stats={},
        stage_state={},
        mixstyle_state={},
        aug_state=None,
        loss_weights={},
        best_score=0.0,
        best_val=0.0,
        best_test=0.0,
        best_epoch=41,
        latest_path="latest",
        best_path="best",
        is_best=False,
    )

    assert row["train_muse_high_ratio"] == 0.50
    assert row["train_muse_mid_ratio"] == 0.25
    assert row["train_muse_low_ratio"] == 0.25
    assert row["train_muse_effective_weight"] == 0.625
    assert row["train_muse_head_js"] == 0.125
    assert row["train_muse_proto_update_weight"] == 0.075
    assert row["train_muse_pseudo_precision_diagnostic"] == "N/A"
