from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_e0d_slim as slim
from cvsrffi.stage2_d92_cauchy_scatter_oas import D92CauchyScatterOASNumericalError
from cvsrffi.stage2_d92_cross_class_offblock_consensus import D92CCOCNumericalError
from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT
from cvsrffi.stage2_d92_e0d_slim import (
    D92_E0D_ARMS,
    build_d92_e0d_fit,
    expected_total_component_fit_count,
)


def _ground() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(92_051)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    return basis, weights, {
        "d81_basis_sha256": "a" * 64,
        "d81_spectral_weight_sha256": "b" * 64,
        "d81_participation_ratio_effective_rank": 2.6,
        "d81_retained_rank": 3,
        "d81_rank_policy": "ceil_participation_ratio_effective_rank",
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
    }


def _support(*, class_count: int, k_shot: int, repeated: bool = False):
    rng = np.random.default_rng(92_100 + class_count * 100 + k_shot)
    labels = np.repeat(np.arange(class_count), k_shot)
    means = rng.normal(size=(class_count, 288))
    if repeated:
        return means[labels].astype(np.float32), labels
    rows = (
        means[labels] + 0.08 * rng.normal(size=(class_count * k_shot, 288))
    ).astype(np.float32)
    return rows, labels


def _measure(call: Callable[[], Any]):
    return call(), {
        "schema": "cvs.phase2.registration_resource_receipt.v1",
        "registration_wall_time_ns": 5_000,
        "registration_process_cpu_time_ns": 4_000,
        "registration_baseline_rss_bytes": 100,
        "registration_peak_rss_bytes": 180,
        "registration_incremental_peak_working_set_bytes": 80,
        "rss_sampler": "synthetic",
    }


def _run(arm_id: str, *, class_count: int, k_shot: int, repeated: bool = False):
    basis, weights, ground_audit = _ground()
    fit, call_records, transform_records = build_d92_e0d_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id=arm_id,
        resource_measure=_measure,
    )
    rows, labels = _support(
        class_count=class_count, k_shot=k_shot, repeated=repeated
    )
    coefficient, intercept, audit = fit(rows, labels, class_count, k_shot)
    return coefficient, intercept, audit, call_records, transform_records


def test_arm_registry_locks_the_five_frozen_e0d_graphs():
    """Would fail if an arm changed Fisher or its registered D mode."""

    expected_historical = {
        "D92_FULL": ("d92_e0d_d92_full", "fusion_loo", True, True),
        "E0_FUSION": ("d92_e0d_e0_fusion", "fusion_loo", True, False),
        "E0_FULL_ONLY": (
            "d92_e0d_e0_full_only",
            "full_only",
            True,
            False,
        ),
        "E0_BLOCK_ONLY": (
            "d92_e0d_e0_block_only",
            "block_only",
            True,
            False,
        ),
        "E0_FIXED50": ("d92_e0d_e0_fixed50", "fixed50", True, False),
    }
    assert {
        key: (
            D92_E0D_ARMS[key].candidate_id,
            D92_E0D_ARMS[key].registered_d_mode,
            D92_E0D_ARMS[key].b_enabled,
            D92_E0D_ARMS[key].e_enabled,
        )
        for key in expected_historical
    } == expected_historical


def test_ocf25_registered_mode_has_the_frozen_public_identity():
    """Would fail if the new OCF25 arm or its fixed mode drifted or vanished."""

    arm = D92_E0D_ARMS["E0_OCF25"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == ("d92_e0ocf_e0_ocf25", "ocf25", True, False)


def test_floorboost_arm_locks_the_single_after_state_full_block_graph():
    """Would fail if floorboost drifted from its frozen identity, constants, or two fits."""

    arm = D92_E0D_ARMS["E0_FULL_MAXMIN_FLOORBOOST"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.ocf_lambda,
        arm.floorboost_quantile,
        arm.floorboost_kappa,
    ) == ("d92_e0_full_maxmin_floorboost", "floorboost", 0.25, 0.20, 0.35)
    _, _, audit, _, _ = _run(
        "E0_FULL_MAXMIN_FLOORBOOST", class_count=11, k_shot=5
    )
    inventory = audit["d92_e0d_actual_component_inventory"]
    assert audit["d92_e0d_actual_component_fit_count"] == 2
    assert audit["d92_e0d_total_component_fit_count"] == 4
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 1
    assert audit["d92_e0d_floorboost_active"] is True
    assert audit["d92_e0d_floorboost_new_rows_byte_exact"] is True
    assert audit["d92_e0d_floorboost_persistent_state_bytes_delta"] == 0
    assert audit["d92_e0d_query_macs"] == 11 * 288


def test_ocf25_registered_state_keeps_new_rows_byte_exact_to_full_only():
    """Would fail if OCF touched a registered new-class affine row."""

    full_coefficient, full_intercept, _, _, _ = _run(
        "E0_FULL_ONLY", class_count=11, k_shot=5
    )
    coefficient, intercept, _, _, _ = _run(
        "E0_OCF25", class_count=11, k_shot=5
    )
    assert coefficient[6:].tobytes() == full_coefficient[6:].tobytes()
    assert intercept[6:].tobytes() == full_intercept[6:].tobytes()


def test_ocf25_registered_state_preserves_full_only_old_group_means():
    """Would fail if OCF changed either full-component old-group mean."""

    full_coefficient, full_intercept, _, _, _ = _run(
        "E0_FULL_ONLY", class_count=11, k_shot=5
    )
    coefficient, intercept, audit, _, _ = _run(
        "E0_OCF25", class_count=11, k_shot=5
    )
    tolerance = audit["d92_ocf_affine_invariant_tolerance"]
    assert tolerance > 0.0
    np.testing.assert_allclose(
        coefficient[:6].mean(axis=0),
        full_coefficient[:6].mean(axis=0),
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        intercept[:6].mean(), full_intercept[:6].mean(), rtol=0.0, atol=tolerance
    )


def test_ocf_receipt_crosscheck_rejects_active_or_lambda_drift(monkeypatch):
    """Would fail if slim accepted an OCF base receipt from a different frozen arm."""

    def fake_builder(*_args, **_kwargs):
        def fake_fit(_rows, _labels, class_count, k_shot):
            return (
                np.zeros((class_count, 288), dtype=np.float32),
                np.zeros(class_count, dtype=np.float32),
                {
                    "d92_component_fit_count": 2,
                    "d92_component_fit_inventory": {
                        "actual_component_fit_count": 2,
                    },
                    "d92_ground_center_active": True,
                    "d92_fisher_residual_pareto_active": False,
                    "d92_registered_d_mode_effective": "ocf25",
                    "d92_k1_k2_exact_full_alias": False,
                    "d92_ocf_active": True,
                    "d92_ocf_lambda": 0.50,
                },
            )

        return fake_fit, [], []

    monkeypatch.setattr(slim.d92_probe, "build_d92_fit", fake_builder)
    basis, weights, ground_audit = _ground()
    fit, _, _ = slim.build_d92_e0d_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id="E0_OCF25",
        resource_measure=_measure,
    )
    rows, labels = _support(class_count=11, k_shot=5)
    with pytest.raises(slim.D92E0DSlimError, match="OCF"):
        fit(rows, labels, 11, 5)


def test_ocf_receipt_crosscheck_rejects_low_k_activity(monkeypatch):
    """Would fail if an OCF receipt stayed active for an exact full alias."""

    def fake_builder(*_args, **_kwargs):
        def fake_fit(_rows, _labels, class_count, _k_shot):
            return (
                np.zeros((class_count, 288), dtype=np.float32),
                np.zeros(class_count, dtype=np.float32),
                {
                    "d92_component_fit_count": 3,
                    "d92_component_fit_inventory": {
                        "actual_component_fit_count": 3,
                    },
                    "d92_ground_center_active": True,
                    "d92_fisher_residual_pareto_active": True,
                    "d92_registered_d_mode_effective": "d92_full_alias",
                    "d92_k1_k2_exact_full_alias": True,
                    "d92_ocf_active": True,
                    "d92_ocf_lambda": 0.25,
                },
            )

        return fake_fit, [], []

    monkeypatch.setattr(slim.d92_probe, "build_d92_fit", fake_builder)
    basis, weights, ground_audit = _ground()
    fit, _, _ = slim.build_d92_e0d_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id="E0_OCF25",
        resource_measure=_measure,
    )
    rows, labels = _support(class_count=11, k_shot=2)
    with pytest.raises(slim.D92E0DSlimError, match="OCF"):
        fit(rows, labels, 11, 2)


def test_registered_k5_arms_emit_frozen_counts_and_actual_inventory():
    """Would fail if a slim arm still performed an unlisted D component fit."""

    expected = {
        "D92_FULL": (48, 24, "d92_full_alias"),
        "E0_FUSION": (24, 12, "fusion_loo"),
        "E0_FULL_ONLY": (2, 1, "full_only"),
        "E0_BLOCK_ONLY": (2, 1, "block_only"),
        "E0_FIXED50": (4, 2, "fixed50"),
        "E0_OCF25": (4, 2, "ocf25"),
        "E0_OCF50": (4, 2, "ocf50"),
    }
    for arm_id, (two_state_count, actual_count, mode) in expected.items():
        coefficient, intercept, audit, _, _ = _run(
            arm_id, class_count=11, k_shot=5
        )
        inventory = audit["d92_e0d_actual_component_inventory"]
        assert coefficient.shape == (11, 288)
        assert intercept.shape == (11,)
        assert np.isfinite(coefficient).all()
        assert np.isfinite(intercept).all()
        assert audit["d92_e0d_registered_d_mode_effective"] == mode
        assert audit["d92_e0d_total_component_fit_count"] == two_state_count
        assert audit["d92_e0d_actual_component_fit_count"] == actual_count
        assert inventory["actual_component_fit_count"] == actual_count
        assert audit["d92_e0d_query_macs"] == 11 * 288
        assert audit["d92_e0d_query_fit_access"] is False
        assert audit["d92_e0d_query_update_access"] is False
        assert audit["d92_e0d_query_selection_access"] is False
        assert audit["registration_incremental_peak_working_set_bytes"] == 80
        if arm_id.startswith("E0_OCF"):
            assert inventory["full_component_fit_count"] == 1
            assert inventory["block3_component_fit_count"] == 1
            assert inventory["loo_component_fit_count"] == 0
            assert audit["d92_e0d_ocf_lambda"] == (
                0.25 if arm_id == "E0_OCF25" else 0.50
            )
            assert audit["d92_e0d_ocf_new_rows_byte_exact"] is True
            assert audit["d92_e0d_ocf_support_alignment_macs_upper_bound"] > 0


def test_k10_total_count_uses_the_frozen_mode_formula():
    expected = {
        "D92_FULL": 88,
        "E0_FUSION": 44,
        "E0_FULL_ONLY": 2,
        "E0_BLOCK_ONLY": 2,
        "E0_FIXED50": 4,
        "E0_OCF25": 4,
        "E0_OCF50": 4,
    }
    for arm_id, count in expected.items():
        _, _, audit, _, _ = _run(arm_id, class_count=11, k_shot=10)
        assert expected_total_component_fit_count(10, arm_id=arm_id) == count
        assert audit["d92_e0d_total_component_fit_count"] == count


@pytest.mark.parametrize(
    ("k_shot", "expected_affine", "expected_mix", "expected_total"),
    (
        (5, 103_680, 8_670, 112_350),
        (10, 207_360, 8_670, 216_030),
    ),
)
@pytest.mark.parametrize("arm_id", ("E0_OCF25", "E0_OCF50"))
def test_ocf_k5_k10_slim_receipt_forwards_frozen_mac_parts(
    arm_id,
    k_shot,
    expected_affine,
    expected_mix,
    expected_total,
):
    """Would fail if slim dropped or changed either frozen OCF MAC term."""

    _, _, audit, _, _ = _run(arm_id, class_count=11, k_shot=k_shot)
    affine_formula = 2 * (OLD_CLASS_COUNT * k_shot) * OLD_CLASS_COUNT * 288
    mix_formula = 5 * OLD_CLASS_COUNT * (288 + 1)
    assert OLD_CLASS_COUNT == 6
    assert (affine_formula, mix_formula, affine_formula + mix_formula) == (
        expected_affine,
        expected_mix,
        expected_total,
    )
    assert (
        audit["d92_e0d_ocf_support_alignment_affine_macs_upper_bound"]
        == affine_formula
    )
    assert (
        audit["d92_e0d_ocf_support_alignment_contrast_mix_macs_upper_bound"]
        == mix_formula
    )
    assert audit["d92_e0d_ocf_support_alignment_macs_upper_bound"] == (
        affine_formula + mix_formula
    )


def test_before_and_k1_k2_states_are_exact_d92_full_aliases_across_arms():
    """Would fail if an E0 arm changed a registration-before or low-K state."""

    for class_count, k_shot in ((6, 5), (11, 1), (11, 2)):
        heads = [
            _run(
                arm_id,
                class_count=class_count,
                k_shot=k_shot,
                repeated=k_shot <= 2,
            )[:3]
            for arm_id in D92_E0D_ARMS
        ]
        for coefficient, intercept, audit in heads[1:]:
            np.testing.assert_array_equal(coefficient, heads[0][0])
            np.testing.assert_array_equal(intercept, heads[0][1])
            assert audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
            if k_shot <= 2:
                assert audit["d92_e0d_k1_k2_exact_full_alias"] is True
        if k_shot == 1:
            assert {head[2]["d92_e0d_total_component_fit_count"] for head in heads} == {3}
            assert {head[2]["d92_e0d_actual_component_fit_count"] for head in heads} == {3}


def test_newguard_arm_is_single_full_fit_and_keeps_low_k_full_alias():
    """Would fail if the NewGuard arm added a BLOCK/LOO fit or changed K2."""

    arm = D92_E0D_ARMS["E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_bidirectional_newguard_maxmin",
        "newguard_maxmin",
        True,
        False,
    )
    _, _, active_audit, _, _ = _run(
        "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN", class_count=11, k_shot=5
    )
    assert active_audit["d92_e0d_actual_component_fit_count"] == 1
    assert active_audit["d92_e0d_total_component_fit_count"] == 2
    assert active_audit["d92_e0d_newguard_full_component_fit_count"] == 1
    assert active_audit["d92_e0d_newguard_active"] is False
    assert active_audit["d92_e0d_newguard_fallback_active"] is True
    assert active_audit["d92_e0d_newguard_deployment_strength_scale"] is None
    assert active_audit["d92_e0d_newguard_deployment_candidate_count"] == 1
    assert active_audit["d92_e0d_newguard_deployment_full_head_byte_exact"] is True
    assert active_audit["d92_e0d_newguard_deployment_codec_roundtrip_count"] == 2
    assert active_audit["d92_e0d_newguard_deployment_codec_macs_upper_bound"] > 0
    _, _, k2_audit, _, _ = _run(
        "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN",
        class_count=11,
        k_shot=2,
        repeated=True,
    )
    assert k2_audit["d92_e0d_k1_k2_exact_full_alias"] is True
    assert k2_audit["d92_e0d_newguard_fallback_reason"] == (
        "K1_K2_EXACT_D92_FULL_ALIAS"
    )


def test_pareto_distill_arm_wires_shared_two_component_receipt_and_low_k_alias():
    """Would fail if slim dropped Pareto's 4/2 inventory or K2 exact alias."""

    arm = D92_E0D_ARMS["E0_FULL_BLOCK_PARETO_DISTILL"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == ("d92_e0_full_block_pareto_distill", "pareto_distill", True, False)
    _, _, audit, _, transforms = _run(
        "E0_FULL_BLOCK_PARETO_DISTILL", class_count=11, k_shot=5
    )
    inventory = audit["d92_e0d_actual_component_inventory"]
    assert expected_total_component_fit_count(5, arm_id=arm.arm_id) == 4
    assert audit["d92_e0d_total_component_fit_count"] == 4
    assert audit["d92_e0d_actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 1
    assert inventory["loo_component_fit_count"] == 0
    assert len(transforms) == 1
    assert audit["d92_e0d_pareto_distill_covariance_estimation_count"] == 1
    assert audit["d92_e0d_pareto_distill_robust_center_transform_count"] == 1
    assert audit["d92_e0d_pareto_distill_query_rows_used"] == 0
    assert audit["d92_e0d_pareto_distill_query_macs"] == 11 * 288
    assert audit["d92_e0d_pareto_distill_query_fit_access"] is False
    assert audit["d92_e0d_pareto_distill_query_update_access"] is False
    assert audit["d92_e0d_pareto_distill_query_selection_access"] is False
    if audit["d92_e0d_pareto_distill_active"]:
        assert audit["d92_e0d_pareto_distill_deployed_support_constraints_pass"] is True
        assert audit["d92_e0d_pareto_distill_deployed_full_head_byte_exact"] is False
        assert audit[
            "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
        ] > 0.0
        assert (
            audit[
                "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
            ]
            >= audit[
                "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
            ]
        )
        assert audit[
            "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"
        ] is True
    else:
        assert audit["d92_e0d_pareto_distill_fallback_active"] is True
        assert audit["d92_e0d_pareto_distill_local_valid"] is False
        assert audit["d92_e0d_pareto_distill_deployed_support_constraints_pass"] is False
        assert audit["d92_e0d_pareto_distill_deployed_full_head_byte_exact"] is True
        assert audit[
            "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"
        ] is False

    _, _, k2_audit, _, _ = _run(
        "E0_FULL_BLOCK_PARETO_DISTILL", class_count=11, k_shot=2, repeated=True
    )
    assert k2_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
    assert k2_audit["d92_e0d_pareto_distill_active"] is False
    assert k2_audit["d92_e0d_pareto_distill_fallback_active"] is False
    assert k2_audit["d92_e0d_pareto_distill_fallback_reason"] == (
        "K1_K2_EXACT_D92_FULL_ALIAS"
    )
    assert k2_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
    ] is None
    assert k2_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
    ] is None
    assert k2_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"
    ] is None

    _, _, k1_audit, _, _ = _run(
        "E0_FULL_BLOCK_PARETO_DISTILL", class_count=11, k_shot=1
    )
    assert k1_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
    assert k1_audit["d92_e0d_pareto_distill_active"] is False
    assert k1_audit["d92_e0d_pareto_distill_fallback_active"] is False
    assert k1_audit["d92_e0d_pareto_distill_fallback_reason"] == (
        "K1_K2_EXACT_D92_FULL_ALIAS"
    )
    assert k1_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_change_max_abs"
    ] is None
    assert k1_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_margin_quantum"
    ] is None
    assert k1_audit[
        "d92_e0d_pareto_distill_deployment_cross_group_quantum_pass"
    ] is None


def test_tpce_arm_reuses_the_single_full_fit_and_low_k_alias() -> None:
    """TPCE is a D42 state postprocess, not another covariance fit."""

    arm = D92_E0D_ARMS["E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_d42_tail_pair_code_exchange",
        "full_only",
        True,
        False,
    )
    assert expected_total_component_fit_count(10, arm_id=arm.arm_id) == 2
    _, _, k1_audit, _, _ = _run(
        arm.arm_id, class_count=11, k_shot=1, repeated=True
    )
    assert k1_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
    assert k1_audit["d92_e0d_total_component_fit_count"] == 3
    assert k1_audit["d92_e0d_actual_component_fit_count"] == 3


def test_tcra_arm_reuses_one_full_fit_and_adds_no_postprocess_fit() -> None:
    """TCRA changes the compiled q2 state, never the FULL fit inventory."""

    arm = D92_E0D_ARMS["E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_d42_tail_class_row_ascent",
        "full_only",
        True,
        False,
    )
    assert expected_total_component_fit_count(10, arm_id=arm.arm_id) == 2
    _, _, audit, _, _ = _run(arm.arm_id, class_count=11, k_shot=5)
    inventory = audit["d92_e0d_actual_component_inventory"]
    assert audit["d92_e0d_total_component_fit_count"] == 2
    assert audit["d92_e0d_actual_component_fit_count"] == 1
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert inventory["loo_component_fit_count"] == 0
    assert audit["d92_e0d_query_truth_access"] is False

    for k_shot in (1, 2):
        _, _, low_audit, _, _ = _run(
            arm.arm_id, class_count=11, k_shot=k_shot, repeated=True
        )
        assert low_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
        assert low_audit["d92_e0d_k1_k2_exact_full_alias"] is True


def test_qic_arm_reuses_one_full_fit_and_preserves_low_k_aliases() -> None:
    """Would fail if QIC changed the frozen FULL lifecycle or K1/K2 route."""

    arm = D92_E0D_ARMS["E0_FULL_D42_QUANTIZATION_INTERCEPT_CLOSURE"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_d42_quantization_intercept_closure",
        "full_only",
        True,
        False,
    )
    assert expected_total_component_fit_count(10, arm_id=arm.arm_id) == 2
    _, _, audit, _, _ = _run(arm.arm_id, class_count=11, k_shot=5)
    assert audit["d92_e0d_actual_component_fit_count"] == 1
    assert audit["d92_e0d_total_component_fit_count"] == 2
    for k_shot in (1, 2):
        _, _, low_audit, _, _ = _run(
            arm.arm_id, class_count=11, k_shot=k_shot, repeated=True
        )
        assert low_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
        assert low_audit["d92_e0d_k1_k2_exact_full_alias"] is True


def test_afcp_arm_reuses_one_full_fit_and_preserves_low_k_aliases() -> None:
    """Would fail if AFCP changed the frozen FULL lifecycle or K1/K2 route."""

    arm = D92_E0D_ARMS["E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_d42_allclass_fold_consensus_plane",
        "full_only",
        True,
        False,
    )
    assert expected_total_component_fit_count(10, arm_id=arm.arm_id) == 2
    _, _, audit, _, _ = _run(arm.arm_id, class_count=11, k_shot=5)
    assert audit["d92_e0d_actual_component_fit_count"] == 1
    assert audit["d92_e0d_total_component_fit_count"] == 2
    for k_shot in (1, 2):
        _, _, low_audit, _, _ = _run(
            arm.arm_id, class_count=11, k_shot=k_shot, repeated=True
        )
        assert low_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
        assert low_audit["d92_e0d_k1_k2_exact_full_alias"] is True


def test_csoas_arm_has_one_active_full_fit_and_an_explicit_low_k_e0_alias():
    """Would fail if CSOAS added a LOO/BLOCK fit or changed the frozen K2 route."""

    arm = D92_E0D_ARMS["E0_FULL_CSOAS"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == ("d92_e0_full_csoas", "csoas_full", True, False)
    assert expected_total_component_fit_count(5, arm_id=arm.arm_id) == 2
    coefficient, intercept, audit, _, _ = _run(
        arm.arm_id, class_count=11, k_shot=5
    )
    inventory = audit["d92_e0d_actual_component_inventory"]
    assert coefficient.shape == (11, 288)
    assert intercept.shape == (11,)
    assert audit["d92_e0d_registered_d_mode_effective"] == "csoas_full"
    assert audit["d92_e0d_total_component_fit_count"] == 2
    assert audit["d92_e0d_actual_component_fit_count"] == 1
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert audit["d92_csoas_active"] is True
    assert audit["d92_csoas_fallback_active"] is False
    assert audit["d92_csoas_paired_e0_codec_state_equal"] is None
    _, _, low_k_audit, _, _ = _run(
        arm.arm_id, class_count=11, k_shot=2, repeated=True
    )
    assert low_k_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
    assert low_k_audit["d92_e0d_k1_k2_exact_full_alias"] is True
    assert low_k_audit["d92_csoas_active"] is False
    assert low_k_audit["d92_csoas_fallback_active"] is False
    assert low_k_audit["d92_csoas_fallback_reason"] == (
        "K1_K2_EXACT_D92_FULL_ALIAS"
    )


def test_csoas_numeric_fallback_reports_two_after_fits_and_is_not_g0_eligible(
    monkeypatch,
):
    """Would fail if a fallback was counted as one active CSOAS fit or passed G0."""

    def injected_numeric_failure(_statistics):
        raise D92CauchyScatterOASNumericalError("injected_csoas_numeric_failure")

    monkeypatch.setattr(
        slim.d92_probe, "compile_cauchy_scatter_oas_affine", injected_numeric_failure
    )
    basis, weights, ground_audit = _ground()
    fit, _, _ = slim.build_d92_e0d_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id="E0_FULL_CSOAS",
        resource_measure=_measure,
    )
    rows, labels = _support(class_count=11, k_shot=5)
    coefficient, intercept, audit = fit(rows, labels, 11, 5)

    inventory = audit["d92_e0d_actual_component_inventory"]
    assert coefficient.shape == (11, 288)
    assert intercept.shape == (11,)
    assert audit["d92_csoas_active"] is False
    assert audit["d92_csoas_fallback_active"] is True
    assert audit["d92_e0d_actual_component_fit_count"] == 2
    assert inventory["actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 2
    assert audit["d92_e0d_total_component_fit_count"] == 3
    assert audit["d92_e0d_csoas_g0_eligible"] is False


def test_ccoc_arm_has_one_active_full_fit_and_low_k_exact_aliases():
    """Would fail if CCOC changed its frozen arm, count, or K1/K2 route."""

    arm = D92_E0D_ARMS["E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"]
    assert (
        arm.candidate_id,
        arm.registered_d_mode,
        arm.b_enabled,
        arm.e_enabled,
    ) == (
        "d92_e0_full_cross_class_offblock_consensus",
        "ccoc_full",
        True,
        False,
    )
    assert expected_total_component_fit_count(10, arm_id=arm.arm_id) == 2
    coefficient, intercept, audit, _, transforms = _run(
        arm.arm_id, class_count=11, k_shot=10
    )
    inventory = audit["d92_e0d_actual_component_inventory"]
    assert coefficient.shape == (11, 288)
    assert intercept.shape == (11,)
    assert audit["d92_e0d_registered_d_mode_effective"] == "ccoc_full"
    assert audit["d92_e0d_actual_component_fit_count"] == 1
    assert audit["d92_e0d_total_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 1
    assert inventory["block3_component_fit_count"] == 0
    assert len(transforms) == 1
    assert audit["d92_e0d_ccoc_active"] is True
    assert audit["d92_e0d_ccoc_fallback_active"] is False
    assert audit["d92_e0d_ccoc_g0_eligible"] is True
    for shots in (1, 2):
        _, _, low_audit, _, _ = _run(
            arm.arm_id, class_count=11, k_shot=shots, repeated=True
        )
        assert low_audit["d92_e0d_registered_d_mode_effective"] == "d92_full_alias"
        assert low_audit["d92_e0d_k1_k2_exact_full_alias"] is True
        assert low_audit["d92_e0d_ccoc_active"] is False
        assert low_audit["d92_e0d_ccoc_fallback_active"] is False
        assert low_audit["d92_e0d_ccoc_fallback_reason"] == (
            "K1_K2_EXACT_D92_FULL_ALIAS"
        )


def test_ccoc_numeric_fallback_reports_real_three_fit_two_state_inventory(
    monkeypatch,
):
    """Would fail if CCOC fallback fabricated a normal two-fit receipt or G0."""

    def injected_numeric_failure(_d42, _statistics):
        raise D92CCOCNumericalError("injected_ccoc_numeric_failure")

    monkeypatch.setattr(
        slim.d92_probe,
        "compile_cross_class_offblock_consensus_affine",
        injected_numeric_failure,
        raising=False,
    )
    basis, weights, ground_audit = _ground()
    fit, _, _ = slim.build_d92_e0d_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        resource_measure=_measure,
    )
    rows, labels = _support(class_count=11, k_shot=10)
    _, _, audit = fit(rows, labels, 11, 10)

    inventory = audit["d92_e0d_actual_component_inventory"]
    assert audit["d92_e0d_ccoc_active"] is False
    assert audit["d92_e0d_ccoc_fallback_active"] is True
    assert audit["d92_e0d_actual_component_fit_count"] == 2
    assert audit["d92_e0d_total_component_fit_count"] == 3
    assert inventory["actual_component_fit_count"] == 2
    assert inventory["full_component_fit_count"] == 2
    assert audit["d92_e0d_ccoc_g0_eligible"] is False
    assert audit["d92_e0d_ccoc_g0_block_reason"] == "NUMERIC_FALLBACK_EXACT_E0"


@pytest.mark.parametrize("delta", (-1, 1))
def test_ccoc_slim_rejects_frozen_k10_workspace_byte_drift(delta):
    """Would fail if the immutable 334336-byte K10 workspace became a loose bound."""

    arm = D92_E0D_ARMS["E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"]
    _, _, audit, _, _ = _run(
        arm.arm_id,
        class_count=11,
        k_shot=10,
    )
    tampered = dict(audit)
    field = "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound"
    tampered[field] = int(tampered[field]) + delta

    with pytest.raises(slim.D92E0DSlimError, match="CCOC statistic receipt drift"):
        slim._ccoc_receipt(
            tampered,
            arm=arm,
            registered=True,
            k_shot=10,
            class_count=11,
        )
