from __future__ import annotations

import numpy as np

from paper_reproduction.cvs_aligned.k1_symmetric_head import (
    apply_diagonal_alignment,
    calibrate_symmetric_k1_head,
    fit_diagonal_alignment,
    fit_gram_score_transform,
    fit_locked_symmetric_support_head,
    fit_symmetric_k1_head,
    leave_one_support_view_out_scores,
    quantize_symmetric_head_fp16,
    persist_and_reload_symmetric_head_fp16,
    score_symmetric_head,
)


def _support(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_centers = np.asarray(
        [
            [1.0, 0.35, 0.0, 0.0, 0.0, 0.0],
            [0.72, 0.69, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 1.0, 0.3],
        ],
        dtype=np.float32,
    )
    return np.stack(
        [class_centers + rng.normal(0.0, 0.035, class_centers.shape) for _ in range(15)]
    ).astype(np.float32)


def test_diagonal_alignment_uses_support_stats_and_round_trips_reference() -> None:
    support = _support()
    source_mean = np.linspace(-0.2, 0.2, support.shape[-1], dtype=np.float32)
    source_std = np.linspace(0.7, 1.2, support.shape[-1], dtype=np.float32)
    state = fit_diagonal_alignment(
        support, source_mean=source_mean, source_std=source_std
    )
    transformed = apply_diagonal_alignment(support, state).reshape(-1, support.shape[-1])
    np.testing.assert_allclose(transformed.mean(axis=0), source_mean, atol=1.0e-5)
    np.testing.assert_allclose(transformed.std(axis=0), source_std, atol=2.0e-4)


def test_calibration_is_query_free_and_role_free() -> None:
    result = calibrate_symmetric_k1_head(_support())
    assert result["selection_source"] == "support_view_leave_one_out_only"
    assert result["query_rows_used"] == 0
    assert result["role_labels_used"] is False
    assert result["class_quota_used"] is False
    assert result["physical_shots_per_class"] == 1
    assert len(result["candidates"]) == 30

    selected = result["selected"]
    scores = leave_one_support_view_out_scores(
        _support(),
        use_alignment=selected["use_alignment"],
        prototype_rule=selected["prototype_rule"],
        ridge=selected["ridge"],
    )
    assert scores.shape == (15, 4, 4)


def test_class_permutation_is_equivariant() -> None:
    support = _support()
    head = fit_symmetric_k1_head(
        support,
        prototype_rules=("trimmed20",),
        ridges=(0.1,),
        allow_alignment=False,
    )
    query = support[0] + 0.01
    original = np.argmax(score_symmetric_head(query, head), axis=1)

    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted_head = fit_symmetric_k1_head(
        support[:, permutation],
        prototype_rules=("trimmed20",),
        ridges=(0.1,),
        allow_alignment=False,
    )
    permuted = np.argmax(score_symmetric_head(query, permuted_head), axis=1)
    decoded = permutation[permuted]
    np.testing.assert_array_equal(decoded, original)


def test_fp16_deployment_state_preserves_predictions() -> None:
    support = _support()
    head = fit_symmetric_k1_head(support)
    quantized = quantize_symmetric_head_fp16(head)
    query = support[0] + 0.003
    np.testing.assert_array_equal(
        np.argmax(score_symmetric_head(query, head), axis=1),
        np.argmax(score_symmetric_head(query, quantized), axis=1),
    )


def test_fp16_state_is_actually_saved_reloaded_and_scored(tmp_path) -> None:
    head = fit_symmetric_k1_head(_support())
    reloaded, audit = persist_and_reload_symmetric_head_fp16(
        head, tmp_path / "head_state.npz"
    )
    assert audit["prediction_parity_pass"] is True
    assert audit["storage_dtype"] == "fp16"
    query = _support()[0]
    np.testing.assert_array_equal(
        np.argmax(score_symmetric_head(query, quantize_symmetric_head_fp16(head)), axis=-1),
        np.argmax(score_symmetric_head(query, reloaded), axis=-1),
    )


def test_locked_head_uses_same_source_selected_rule_for_k5() -> None:
    base = _support()[:3]
    observations = np.repeat(base, 5, axis=0)
    head = fit_locked_symmetric_support_head(
        observations,
        physical_shots_per_class=5,
        selected={
            "use_alignment": False,
            "prototype_rule": "trimmed20",
            "ridge": 0.1,
        },
    )
    assert head.calibration["selection_source"] == "source_receiver_holdout_locked"
    assert head.calibration["physical_shots_per_class"] == 5
    assert head.prototype_rule == "trimmed20"
    assert head.ridge == 0.1


def test_gram_transform_deconfuses_correlated_prototype_responses() -> None:
    prototypes = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )
    transform = fit_gram_score_transform(prototypes, ridge=0.03)
    raw = prototypes @ prototypes.T
    corrected = raw @ transform
    np.testing.assert_array_equal(
        np.argmax(corrected, axis=1), np.arange(len(prototypes))
    )
    assert float(np.mean(np.diag(corrected))) > float(
        np.mean(corrected[~np.eye(3, dtype=bool)])
    )


def test_state_and_compute_remain_extreme_light_for_26_classes() -> None:
    rng = np.random.default_rng(11)
    support = rng.normal(size=(15, 26, 256)).astype(np.float32)
    head = fit_symmetric_k1_head(
        support,
        prototype_rules=("mean",),
        ridges=(0.1,),
        allow_alignment=True,
    )
    # Calibration may legally reject the affine.  These are upper bounds when
    # it is retained; the no-affine state is even smaller.
    assert head.persistent_state_bytes_fp16 <= 15688
    assert head.extra_macs_per_query <= 932
    assert head.persistent_state_bytes_fp16 < 32 * 1024
    assert head.extra_macs_per_query < 2_000
