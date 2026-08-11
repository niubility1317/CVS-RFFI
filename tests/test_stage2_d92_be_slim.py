from __future__ import annotations

from typing import Any, Callable

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d92_be_slim import (
    D92_BE_ARMS,
    build_d92_be_fit,
    expected_total_component_fit_count,
)
from scripts import probe_d92_registration_balanced_covariance as d92_probe


def _ground() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(9211)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    weights = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    audit = {
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
    return basis, weights, audit


def _support(*, class_count: int, k_shot: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(920_000 + class_count * 100 + k_shot)
    labels = np.repeat(np.arange(class_count), k_shot)
    means = rng.normal(size=(class_count, 288))
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


def _run(arm_id: str, *, class_count: int, k_shot: int):
    basis, weights, ground_audit = _ground()
    fit, call_records, transform_records = build_d92_be_fit(
        d42,
        basis,
        weights,
        ground_audit,
        arm_id=arm_id,
        resource_measure=_measure,
    )
    rows, labels = _support(class_count=class_count, k_shot=k_shot)
    coefficient, intercept, audit = fit(rows, labels, class_count, k_shot)
    return coefficient, intercept, audit, call_records, transform_records


def test_arm_registry_freezes_only_registered_B_and_E_switches():
    assert {
        key: (value.b_enabled, value.e_enabled, value.candidate_id)
        for key, value in D92_BE_ARMS.items()
    } == {
        "FULL": (True, True, "d92_be_full"),
        "B0": (False, True, "d92_be_b0"),
        "E0": (True, False, "d92_be_e0"),
        "B0E0": (False, False, "d92_be_b0e0"),
    }


def test_expected_fit_count_is_halved_only_when_E_is_disabled():
    assert expected_total_component_fit_count(5, e_enabled=True) == 48
    assert expected_total_component_fit_count(10, e_enabled=True) == 88
    assert expected_total_component_fit_count(5, e_enabled=False) == 24
    assert expected_total_component_fit_count(10, e_enabled=False) == 44


def test_registered_four_arms_close_switches_counts_and_resource_receipt():
    expected = {
        "FULL": (48, True, True, True),
        "B0": (48, False, True, False),
        "E0": (24, True, False, True),
        "B0E0": (24, False, False, False),
    }
    for arm_id, (
        total_fit_count,
        b_enabled,
        e_enabled,
        ground_transform_expected,
    ) in expected.items():
        coefficient, intercept, audit, call_records, transforms = _run(
            arm_id, class_count=11, k_shot=5
        )
        assert coefficient.shape == (11, 288)
        assert intercept.shape == (11,)
        assert np.isfinite(coefficient).all()
        assert np.isfinite(intercept).all()
        assert audit["d92_be_arm_id"] == arm_id
        assert audit["d92_be_B_enabled"] is b_enabled
        assert audit["d92_be_E_enabled"] is e_enabled
        assert audit["d92_be_B_effective"] is b_enabled
        assert audit["d92_be_E_effective"] is e_enabled
        assert audit["d92_be_total_component_fit_count"] == total_fit_count
        assert audit["d92_be_base_component_fit_count"] == 24
        assert audit["d92_be_fisher_component_fit_count"] == (
            24 if e_enabled else 0
        )
        assert bool(transforms) is ground_transform_expected
        assert bool(call_records) is e_enabled
        assert audit["registration_wall_time_ns"] == 5_000
        assert audit["registration_incremental_peak_working_set_bytes"] == 80
        assert audit["d92_be_query_macs"] == 11 * 288
        assert audit["d92_be_query_fit_access"] is False
        assert audit["d92_be_query_update_access"] is False
        assert audit["d92_be_query_selection_access"] is False


def test_registered_k10_fit_count_is_88_or_44_from_real_call_records():
    for arm_id, expected in (("FULL", 88), ("E0", 44)):
        _, _, audit, _, _ = _run(arm_id, class_count=11, k_shot=10)
        assert audit["d92_be_total_component_fit_count"] == expected


def test_old_only_da0_reg0_head_is_identical_across_all_arms():
    heads = [
        _run(arm_id, class_count=6, k_shot=5)[:2]
        for arm_id in D92_BE_ARMS
    ]
    for coefficient, intercept in heads[1:]:
        np.testing.assert_array_equal(coefficient, heads[0][0])
        np.testing.assert_array_equal(intercept, heads[0][1])
    for _, _, audit, _, _ in (
        _run(arm_id, class_count=6, k_shot=5) for arm_id in D92_BE_ARMS
    ):
        assert audit["d92_be_B_effective"] is True
        assert audit["d92_be_E_effective"] is True


def test_k1_registered_head_is_exact_alias_and_global_drift_flag_is_restored():
    original_policy = d92_probe.d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT
    heads = [
        _run(arm_id, class_count=11, k_shot=1)[:3]
        for arm_id in D92_BE_ARMS
    ]
    for coefficient, intercept, audit in heads[1:]:
        np.testing.assert_array_equal(coefficient, heads[0][0])
        np.testing.assert_array_equal(intercept, heads[0][1])
        assert audit["d92_be_k1_k2_exact_full_alias"] is True
        assert audit["d92_be_B_effective"] is True
        assert audit["d92_be_E_effective"] is True
    assert d92_probe.d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT is original_policy
