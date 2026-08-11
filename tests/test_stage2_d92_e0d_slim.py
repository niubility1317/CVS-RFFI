from __future__ import annotations

from typing import Any, Callable

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
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

    assert {
        key: (
            value.candidate_id,
            value.registered_d_mode,
            value.b_enabled,
            value.e_enabled,
        )
        for key, value in D92_E0D_ARMS.items()
    } == {
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


def test_registered_k5_arms_emit_frozen_counts_and_actual_inventory():
    """Would fail if a slim arm still performed an unlisted D component fit."""

    expected = {
        "D92_FULL": (48, 24, "d92_full_alias"),
        "E0_FUSION": (24, 12, "fusion_loo"),
        "E0_FULL_ONLY": (2, 1, "full_only"),
        "E0_BLOCK_ONLY": (2, 1, "block_only"),
        "E0_FIXED50": (4, 2, "fixed50"),
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


def test_k10_total_count_uses_the_frozen_mode_formula():
    expected = {
        "D92_FULL": 88,
        "E0_FUSION": 44,
        "E0_FULL_ONLY": 2,
        "E0_BLOCK_ONLY": 2,
        "E0_FIXED50": 4,
    }
    for arm_id, count in expected.items():
        _, _, audit, _, _ = _run(arm_id, class_count=11, k_shot=10)
        assert expected_total_component_fit_count(10, arm_id=arm_id) == count
        assert audit["d92_e0d_total_component_fit_count"] == count


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
