from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from cvsrffi.phase1_bicad_xr.config import (
    BiCADXRConfig,
    BiCADXRStage,
    CV2_CANDIDATE_IDS,
    CANDIDATE_IDS,
    candidate_config,
    candidate_diff,
    method_lock_payload,
    stage_for_update,
)


P_CANDIDATE_IDS = ("P0", "P1", "P2", "P3", "P4")
CV2_IDS = tuple(
    f"CV2-{family}{index}"
    for family in ("B", "D", "T")
    for index in range(4)
)


def test_cv2_registry_has_exactly_twelve_namespaced_static_candidates() -> None:
    assert CV2_CANDIDATE_IDS == CV2_IDS
    assert set(CV2_IDS).issubset(CANDIDATE_IDS)
    assert all(candidate_config(candidate).candidate_id == candidate for candidate in CV2_IDS)


@pytest.mark.parametrize(
    ("left", "right", "changed"),
    [
        ("CV2-B2", "CV2-B3", {"swad"}),
        (
            "CV2-D0",
            "CV2-D1",
            {"conditional_cdan", "detached_adversarial", "adversarial_two_time_scale"},
        ),
        (
            "CV2-D1",
            "CV2-D2",
            {"conditional_xcov", "zdom_tx_adversary"},
        ),
        (
            "CV2-D2",
            "CV2-D3",
            {"dynamic_adversarial_dose", "task_protected_gradient"},
        ),
        ("CV2-T0", "CV2-T1", {"pair_identity"}),
        ("CV2-T0", "CV2-T2", {"margin_tail"}),
        ("CV2-T0", "CV2-T3", {"pair_identity", "margin_tail"}),
    ],
)
def test_cv2_adjacent_candidates_differ_only_by_declared_static_switches(
    left: str, right: str, changed: set[str]
) -> None:
    assert set(candidate_diff(left, right)) == changed


@pytest.mark.parametrize("candidate_id", ["CV2-D0", "CV2-T0"])
def test_cv2_stage0_candidates_are_static_copies_of_b3_except_candidate_id(
    candidate_id: str,
) -> None:
    assert candidate_diff(candidate_id, "CV2-B3") == {}
    assert candidate_config(candidate_id).candidate_id == candidate_id
    assert candidate_config("CV2-B3").candidate_id == "CV2-B3"


def test_cv2_stage0_copy_does_not_rewrite_historical_d0() -> None:
    historical = candidate_config("D0")

    assert historical.candidate_id == "D0"
    assert historical.optimizer_updates == 5000
    assert historical.batch_size == 96
    assert historical.concat_sat_start_epoch == 80
    assert historical.satellite_supervision_mode == "ce_only"
    assert historical.factorized_domains is False
    assert historical.gradient_firewall is False
    assert historical.swad is False


@pytest.mark.parametrize("candidate_id", CV2_IDS)
def test_cv2_method_lock_is_source_only_and_deferred_features_are_disabled(
    candidate_id: str,
) -> None:
    cfg = candidate_config(candidate_id)
    lock = method_lock_payload(cfg)

    assert lock["candidate_id"] == candidate_id
    assert lock["frozen"] is True
    assert lock["dynamic_alias"] is False
    assert lock["source_only"] is True
    assert lock["target_access"] is False
    assert lock["phase2_access"] is False
    assert lock["support_access"] is False
    assert lock["query_access"] is False
    assert lock["truth_access"] is False
    assert all(value is False for value in lock["deferred_features"].values())


@pytest.mark.parametrize("deferred", ["ema_teacher", "hard_leo_mining", "iq_mixup"])
def test_cv2_deferred_overrides_fail_closed(deferred: str) -> None:
    with pytest.raises(ValueError, match="deferred|disabled|incompatible"):
        candidate_config("CV2-T0", overrides={deferred: True})


def test_pairbicad_p0_to_p4_are_registered_with_shared_frozen_config() -> None:
    assert set(P_CANDIDATE_IDS).issubset(CANDIDATE_IDS)

    for candidate_id in P_CANDIDATE_IDS:
        cfg = candidate_config(candidate_id)

        assert cfg.optimizer_updates == 4000
        assert cfg.batch_size == 48
        assert cfg.concat_sat_start_epoch == 1
        assert cfg.satellite_supervision_mode == "ce_only_plus_pair_selfsup"
        assert cfg.strict_pair_concat
        assert cfg.pair_projector_dim == 128
        assert cfg.factor_interaction_dim == 24
        assert cfg.lambda_sat_cls_start == pytest.approx(0.5)
        assert cfg.lambda_sat_cls_end == pytest.approx(1.0)


def test_pairbicad_p0_enables_only_strict_pair_concat() -> None:
    cfg = candidate_config("P0")

    assert cfg.strict_pair_concat
    assert not cfg.factorized_domains
    assert not cfg.gradient_firewall
    assert not cfg.conditional_cdan
    assert not cfg.zdom_tx_adversary
    assert not cfg.pair_identity
    assert not cfg.pair_vicreg
    assert not cfg.pair_delta
    assert not cfg.dynamic_adversarial_dose


@pytest.mark.parametrize(
    ("left", "right", "changed"),
    [
        ("P0", "P1", {"factorized_domains", "gradient_firewall"}),
        ("P1", "P2", {"conditional_cdan", "zdom_tx_adversary"}),
        ("P2", "P3", {"pair_identity", "pair_vicreg"}),
        ("P3", "P4", {"pair_delta", "dynamic_adversarial_dose"}),
    ],
)
def test_pairbicad_adjacent_candidates_change_only_intended_fields(
    left: str, right: str, changed: set[str]
) -> None:
    assert set(candidate_diff(left, right)) == changed


@pytest.mark.parametrize("candidate_id", P_CANDIDATE_IDS)
def test_pairbicad_candidates_keep_excluded_mechanisms_disabled(
    candidate_id: str,
) -> None:
    cfg = candidate_config(candidate_id)

    assert not cfg.conditional_xcov
    assert not cfg.task_protected_gradient
    assert not cfg.sparse_xdc
    assert not cfg.xdc_kd
    assert not cfg.paired_satellite
    assert not cfg.margin_tail
    assert cfg.receiver_tangent == "off"
    assert not cfg.swad
    assert not any(
        getattr(cfg, legacy_flag)
        for legacy_flag in (
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
        )
    )


def test_legacy_candidates_keep_original_schedule_and_satellite_contract() -> None:
    for candidate_id in (
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "F0",
        "F1",
        "F2",
        "F3",
        "ADV3B02-BiCAD-XDC-V1",
    ):
        cfg = candidate_config(candidate_id)

        assert cfg.optimizer_updates == 5000
        assert cfg.batch_size == 96
        assert cfg.concat_sat_start_epoch == 80
        assert cfg.lambda_sat_cls == pytest.approx(0.68)
        assert cfg.lambda_sat_cons == pytest.approx(0.0)


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
    tuned = candidate_config("D5", overrides={"lambda_cond_xcov": 0.04})

    assert base.lambda_cond_xcov == pytest.approx(0.02)
    assert tuned.lambda_cond_xcov == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("candidate_id", "D6"),
        ("factorized_domains", False),
        ("conditional_cdan", False),
        ("zdom_tx_adversary", False),
        ("conditional_xcov", False),
        ("gradient_firewall", False),
        ("task_protected_gradient", True),
        ("sparse_xdc", True),
        ("xdc_kd", True),
        ("paired_satellite", True),
        ("margin_tail", True),
        ("receiver_tangent", "factual"),
        ("swad", True),
        ("phase1_method", "other"),
        ("optimizer_updates", 6000),
        ("batch_size", 128),
        ("xdc_interval", 8),
        ("pair_interval", 8),
        ("lambda_sat_cls", 0.5),
        ("lambda_sat_cons", 0.1),
        ("lambda_orth", 0.01),
        ("gradient_firewall_scale", 0.1),
        ("concat_sat_ce_only", False),
        ("concat_sat_start_epoch", 79),
        ("satellite_supervision_mode", "ce_only_plus_pair_selfsup"),
        ("strict_pair_concat", True),
        ("pair_identity", True),
        ("pair_vicreg", True),
        ("pair_delta", True),
        ("dynamic_adversarial_dose", True),
        ("pair_projector_dim", 64),
        ("factor_interaction_dim", 12),
        ("lambda_sat_cls_start", 0.5),
        ("lambda_sat_cls_end", 1.0),
        (
            "sat_train_scenarios",
            ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"),
        ),
        ("xdc_ridge", 0.02),
        ("xdc_temperature", 3.0),
        ("xdc_min_support_accuracy", 0.30),
        ("xdc_microepisode_tx", 8),
        ("xdc_microepisode_receivers", 5),
        ("xdc_samples_per_cell", 3),
        ("margin_tail_cvar_fraction", 0.25),
        ("margin_tail_weights", (0.6, 0.3, 0.1)),
        ("margin_tail_ema", 0.8),
        ("receiver_tangent_rank", 2),
        ("receiver_tangent_start_progress", 0.60),
        ("stage4_domain_scale", 0.5),
        ("stage4_shared_stem_lr_scale", 0.2),
    ],
)
def test_candidate_overrides_reject_every_field_outside_source_search_whitelist(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValueError, match="frozen"):
        candidate_config("D5", overrides={field_name: value})


def test_sat_train_scenarios_requires_an_exact_tuple() -> None:
    with pytest.raises(ValueError, match="tuple"):
        BiCADXRConfig(
            candidate_id="CUSTOM",
            sat_train_scenarios=[
                "leo_clear_weak",
                "leo_low_elev_weak",
                "leo_rain_weak",
            ],  # type: ignore[arg-type]
        )


def test_margin_tail_weights_require_an_exact_tuple() -> None:
    with pytest.raises(ValueError, match="tuple"):
        BiCADXRConfig(
            candidate_id="CUSTOM",
            margin_tail_weights=[0.6, 0.3, 0.1],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="exactly"):
        BiCADXRConfig(
            candidate_id="CUSTOM",
            margin_tail_weights=(0.5, 0.4, 0.1),
        )


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


@pytest.mark.parametrize(
    ("total_updates", "expected"),
    [
        (
            10,
            (
                "stage0",
                "stage1",
                "stage1",
                "stage2",
                "stage2",
                "stage2",
                "stage2",
                "stage3",
                "stage3",
                "stage4",
            ),
        ),
        (
            7,
            (
                "stage1",
                "stage1",
                "stage2",
                "stage2",
                "stage3",
                "stage3",
                "stage4",
            ),
        ),
    ],
)
def test_stage_boundaries_scale_with_non_5000_totals(
    total_updates: int, expected: tuple[str, ...]
) -> None:
    actual = tuple(
        stage_for_update(update, total_updates).name
        for update in range(1, total_updates + 1)
    )

    assert actual == expected


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
        "strict_pair_concat",
        "pair_identity",
        "pair_vicreg",
        "pair_delta",
        "dynamic_adversarial_dose",
    }
    with pytest.raises((AttributeError, TypeError)):
        candidate_config("D0").batch_size = 128  # type: ignore[misc]


def test_cv2_e200_configs_use_epoch_termination_and_no_legacy_update_budget() -> None:
    for candidate_id in CV2_IDS:
        cfg = candidate_config(candidate_id)

        assert getattr(cfg, "epochs", None) == 200
        assert getattr(cfg, "termination_mode", None) == "epochs"
        assert cfg.optimizer_updates != 6500


def test_cv2_configs_expose_consumable_runtime_contracts() -> None:
    for candidate_id in CV2_IDS:
        cfg = candidate_config(candidate_id)
        no_freeze = getattr(cfg, "no_early_freeze_contract", None)
        two_time_scale = getattr(cfg, "adversarial_two_time_scale_contract", None)

        assert no_freeze is not None
        assert no_freeze.enabled is cfg.no_early_freeze
        if cfg.no_early_freeze:
            assert no_freeze.freeze_until_epoch == 200
            assert no_freeze.audit == "requires_grad_per_epoch"
        else:
            assert no_freeze.freeze_until_epoch == 0
            assert no_freeze.audit == "disabled"

        assert two_time_scale is not None
        assert two_time_scale.enabled is cfg.adversarial_two_time_scale
        if cfg.adversarial_two_time_scale:
            assert two_time_scale.optimizer_mode == "separate"
            assert two_time_scale.discriminator_lr_ratio == pytest.approx(1.5)
            assert two_time_scale.phase_order == ("discriminator", "encoder")
            assert two_time_scale.one_backbone_forward is True
        else:
            assert two_time_scale.optimizer_mode == "disabled"
            assert two_time_scale.discriminator_lr_ratio == pytest.approx(1.0)


def test_cv2_method_lock_serializes_epoch_contract_without_update_budget() -> None:
    lock = method_lock_payload(candidate_config("CV2-D1"))

    assert lock["runtime_contracts"]["termination"] == {
        "mode": "epochs",
        "epochs": 200,
    }
    assert lock["runtime_contracts"]["no_early_freeze"]["freeze_until_epoch"] == 200
    assert lock["runtime_contracts"]["adversarial_two_time_scale"] == {
        "enabled": True,
        "optimizer_mode": "separate",
        "discriminator_lr_ratio": pytest.approx(1.5),
        "phase_order": ["discriminator", "encoder"],
        "one_backbone_forward": True,
    }
    assert "optimizer_updates" not in lock["configuration"]
