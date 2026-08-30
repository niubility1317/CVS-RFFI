from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from cvsrffi.phase1_bicad_xr.config import (
    BiCADXRConfig,
    BiCADXRStage,
    candidate_config,
    candidate_diff,
    stage_for_update,
)


def test_v1_alias_is_d5_plus_sparse_xdc_and_tail_only() -> None:
    cfg = candidate_config("ADV3B02-BiCAD-XDC-V1")

    assert isinstance(cfg, BiCADXRConfig)
    assert cfg.factorized_domains and cfg.conditional_cdan
    assert cfg.zdom_tx_adversary and cfg.conditional_xcov
    assert cfg.gradient_firewall and cfg.sparse_xdc and cfg.margin_tail
    assert not cfg.task_protected_gradient
    assert not cfg.xdc_kd and not cfg.paired_satellite
    assert cfg.receiver_tangent == "off"
    assert not cfg.swad


def test_v1_freezes_protocol_weights_and_runtime_intervals() -> None:
    cfg = candidate_config("ADV3B02-BiCAD-XDC-V1")

    assert cfg.batch_size == 96
    assert cfg.xdc_interval == 4
    assert cfg.pair_interval == 4
    assert cfg.lambda_sat_cls == pytest.approx(0.68)
    assert cfg.lambda_sat_cons == pytest.approx(0.0)
    assert cfg.lambda_cond_xcov == pytest.approx(0.02)
    assert cfg.gradient_firewall_scale == pytest.approx(0.05)
    assert cfg.concat_sat_ce_only
    assert cfg.concat_sat_start_epoch == 80
    assert cfg.sat_train_scenarios == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "use_fasttrust",
        "use_pseudo_label",
        "use_csd",
        "use_hcf_transport",
        "use_content_lodo",
        "use_hdro",
        "use_proxy_unknown",
        "use_soft_unknown_mixup",
        "use_open_world_feature_loss",
        "use_fishr",
        "use_generic_mixup",
        "use_mixstyle",
    ],
)
def test_forbidden_legacy_features_fail_closed(legacy_flag: str) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        candidate_config("D5", overrides={legacy_flag: True})


def test_candidate_overrides_are_scoped_and_do_not_mutate_registry() -> None:
    base = candidate_config("D5")
    tuned = candidate_config("D5", overrides={"batch_size": 128, "lambda_cond_xcov": 0.04})

    assert base.batch_size == 96
    assert base.lambda_cond_xcov == pytest.approx(0.02)
    assert tuned.batch_size == 128
    assert tuned.lambda_cond_xcov == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("left", "right", "changed"),
    [
        ("D0", "D1", {"factorized_domains"}),
        ("D1", "D2", {"conditional_cdan"}),
        ("D2", "D3", {"zdom_tx_adversary"}),
        ("D3", "D4", {"conditional_xcov"}),
        ("D4", "D5", {"gradient_firewall"}),
        ("D5", "D6", {"task_protected_gradient"}),
        ("D6", "E0", {"task_protected_gradient"}),
        ("E0", "E1", {"sparse_xdc"}),
        ("E1", "E2", {"xdc_kd"}),
        ("E2", "E3", {"paired_satellite"}),
        ("E3", "E4", {"margin_tail"}),
        ("E4", "F0", set()),
        ("F0", "F1", {"receiver_tangent"}),
        ("F1", "F2", {"receiver_tangent"}),
        ("F2", "F3", {"swad"}),
    ],
)
def test_d0_to_f3_adjacent_candidates_change_only_declared_fields(
    left: str, right: str, changed: set[str]
) -> None:
    diff = candidate_diff(left, right)

    assert set(diff) == changed
    for field_name, (left_value, right_value) in diff.items():
        assert left_value != right_value
        assert getattr(candidate_config(left), field_name) == left_value
        assert getattr(candidate_config(right), field_name) == right_value


def test_candidate_alias_diff_exposes_its_two_composed_changes() -> None:
    diff = candidate_diff("D5", "ADV3B02-BiCAD-XDC-V1")

    assert set(diff) == {"sparse_xdc", "margin_tail"}
    assert diff["sparse_xdc"] == (False, True)
    assert diff["margin_tail"] == (False, True)


@pytest.mark.parametrize(
    ("update", "stage"),
    [
        (1, "stage0"),
        (500, "stage0"),
        (501, "stage1"),
        (1750, "stage1"),
        (1751, "stage2"),
        (3500, "stage2"),
        (3501, "stage3"),
        (4500, "stage3"),
        (4501, "stage4"),
        (5000, "stage4"),
    ],
)
def test_five_stage_boundaries(update: int, stage: str) -> None:
    result = stage_for_update(update, 5000)

    assert isinstance(result, BiCADXRStage)
    assert result.name == stage
    assert result.value == stage


@pytest.mark.parametrize("update", [0, -1, 5001])
def test_stage_scheduler_rejects_updates_outside_inclusive_range(update: int) -> None:
    with pytest.raises(ValueError, match=r"\[1,total_updates\]"):
        stage_for_update(update, 5000)


@pytest.mark.parametrize("total_updates", [0, -1])
def test_stage_scheduler_rejects_non_positive_total(total_updates: int) -> None:
    with pytest.raises(ValueError, match="total_updates"):
        stage_for_update(1, total_updates)


def test_config_is_frozen_and_contains_all_candidate_switches() -> None:
    assert is_dataclass(BiCADXRConfig)
    assert {field.name for field in fields(BiCADXRConfig)} >= {
        "candidate_id",
        "factorized_domains",
        "conditional_cdan",
        "zdom_tx_adversary",
        "conditional_xcov",
        "gradient_firewall",
        "task_protected_gradient",
        "sparse_xdc",
        "xdc_kd",
        "paired_satellite",
        "margin_tail",
        "receiver_tangent",
        "swad",
    }
    with pytest.raises((AttributeError, TypeError)):
        candidate_config("D0").batch_size = 128  # type: ignore[misc]
