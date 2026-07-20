from __future__ import annotations

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d93_query_evaluation import (
    _build_guarded_d93_base_fit,
    build_d93_top_level_fit,
    predict_d93,
    score_d93,
)


class D43ProbeError(RuntimeError):
    pass


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_d93_wraps_int8_target_head_and_scores_all_classes() -> None:
    rng = np.random.default_rng(31)
    old_classes = tuple(f"old-{index}" for index in range(6))
    new_classes = tuple(f"new-{index}" for index in range(2))
    canonical = _normalize(rng.normal(size=(6, 160)))
    nuisance = np.linalg.qr(rng.normal(size=(160, 4)))[0]
    ground = np.stack(
        [
            _normalize(
                canonical + rng.normal(scale=0.04, size=(6, 4)) @ nuisance.T
            )
            for _ in range(14)
        ]
    )
    mask = np.ones((14, 6), dtype=np.uint8)
    target_old_z = _normalize(canonical + rng.normal(scale=0.035, size=(6, 4)) @ nuisance.T)
    target_new_z = _normalize(rng.normal(size=(2, 160)))

    def registered(z: np.ndarray, repeats: int) -> np.ndarray:
        primary = np.repeat(z, repeats, axis=0)
        primary = _normalize(primary + rng.normal(scale=0.01, size=primary.shape))
        auxiliary = _normalize(rng.normal(size=(len(primary), 128)))
        return _normalize(np.concatenate([primary, 4.0 * auxiliary], axis=1)).astype(
            np.float32
        )

    old_x = registered(target_old_z, 5)
    new_x = registered(target_new_z, 5)
    old_y = tuple(handle for handle in old_classes for _ in range(5))
    new_y = tuple(handle for handle in new_classes for _ in range(5))
    wrapper = build_d93_top_level_fit(
        d42.fit_d42_unified_shrinkage_lda,
        ground_prototypes=ground[:, ::-1],
        ground_mask=mask[:, ::-1],
        ground_classes=old_classes[::-1],
        target_old_tx_labels=old_classes,
        ground_audit={
            "component_manifest_sha256": "1" * 64,
            "component_npz_sha256": "2" * 64,
            "ground_int8_component_logical_state_bytes": 5816,
            "ground_component_input_count": 84,
        },
        include_nuisance_scale=True,
    )
    result = wrapper(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=713101,
        device="cpu",
    )
    scores = score_d93(result.state, np.concatenate([old_x, new_x]))
    predictions = predict_d93(result.state, np.concatenate([old_x, new_x]))
    assert scores.shape == (40, 8)
    assert predictions.shape == (40,)
    assert result.state.base_state.is_int8
    assert result.geometry_audit["formal_old_target_vectors_residual_int8"] is True
    assert result.geometry_audit["formal_new_target_vectors_residual_int8"] is True
    assert result.geometry_audit["d93_ground_direct_query_score_access"] is False
    assert (
        result.geometry_audit["d93_transport_audit"][
            "ground_registry_reordered_to_target_old"
        ]
        is True
    )
    assert result.geometry_audit["d93_transport_audit"][
        "ground_to_target_binding_policy"
    ] == "registered_target_old_tx_label_order_to_opaque_class_index"
    assert result.resource_audit["d93_optimizer_steps"] == 0
    assert result.resource_audit["persistent_state_bytes"] < 256 * 1024
    assert result.resource_audit["trainable_parameters"] < 80_000


def test_d93_guarded_base_uses_locked_d42_only_for_exact_pd_failure() -> None:
    class Module:
        @staticmethod
        def _fit_equal_prior_lda(*args, **kwargs):
            del args, kwargs
            return np.ones((2, 3)), np.zeros(2), {"covariance_policy": "locked_d42"}

    class D62Probe:
        @staticmethod
        def build_d62_fit(module):
            del module

            def fail(*args, **kwargs):
                del args, kwargs
                raise D43ProbeError(
                    "D43 structured covariance is not positive definite"
                )

            return fail, []

    guarded, records = _build_guarded_d93_base_fit(Module, D62Probe)
    coefficients, intercept, audit = guarded(None, None, 2, 10)
    assert records == []
    assert coefficients.shape == (2, 3)
    assert intercept.shape == (2,)
    assert audit["d93_d43_nonpositive_definite_observed"] is True
    assert audit["d93_fallback_query_rows_used"] == 0
