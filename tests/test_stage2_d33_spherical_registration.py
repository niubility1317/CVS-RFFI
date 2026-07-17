from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d33_spherical_registration import (
    D33SphericalRegistrationConfig,
    D33SphericalRegistrationError,
    MAX_ACTIVE_PARAMETERS,
    fit_d33_spherical_registration,
    score_d33_spherical_registration,
)


def _support(
    classes: tuple[str, ...],
    k_shot: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(classes), 288)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, class_name in enumerate(classes):
        class_scale = 0.025 + 0.006 * (index % 4)
        for _ in range(k_shot):
            row = centers[index] + class_scale * rng.normal(size=288).astype(np.float32)
            row /= np.linalg.norm(row)
            rows.append(row)
            labels.append(class_name)
    return np.stack(rows), np.asarray(labels)


def _fit(new_count: int, k_shot: int, *, policy: str = "B_balanced"):
    old_classes = tuple(f"old_{index}" for index in range(6))
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k_shot, seed=10 + new_count)
    new_x, new_y = _support(new_classes, k_shot, seed=20 + new_count)
    log_diag = np.linspace(-0.08, 0.08, 288, dtype=np.float32)
    result = fit_d33_spherical_registration(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        log_diag,
        config=D33SphericalRegistrationConfig(selection_policy=policy),
    )
    return result, old_x, new_x, log_diag


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_all_required_new_class_scales_are_closed_form_and_under_50k(new_count: int) -> None:
    result, _, _, _ = _fit(new_count, 2)
    class_count = 6 + new_count

    assert result.state.centroids_qint8.shape == (class_count, 288)
    assert result.state.centroids_qint8.dtype == np.int8
    assert result.state.centroid_scales.shape == (class_count,)
    assert result.state.centroid_scales.dtype == np.float32
    assert result.state.radii.shape == (class_count,)
    assert result.state.active_parameters == 288 + class_count * 288 + 2 * class_count
    assert result.state.active_parameters < MAX_ACTIVE_PARAMETERS
    assert result.state.optimizer_steps == 0
    assert len(result.selection_trace) == 3 * 4 * 3
    assert sum(bool(row["selected"]) for row in result.selection_trace) == 1
    assert result.resource_audit["active_parameter_cap_pass"] is True
    assert result.resource_audit["dense_query_graph_bytes"] == 0
    assert result.resource_audit["query_rows_used_for_fit"] == 0
    assert result.resource_audit["query_role_oracle_access"] is False
    assert result.resource_audit["query_class_quota_access"] is False
    assert result.resource_audit["clean_sample_access"] is False
    assert result.resource_audit["source_sample_access"] is False
    assert result.resource_audit["train_deploy_score_surface_identical"] is True
    assert result.resource_audit["old_new_shared_radius_rule"] is True
    assert result.resource_audit["old_new_shared_centroid_quantization"] == (
        "symmetric_int8_per_class_scale"
    )
    assert result.resource_audit["resident_fp32_centroid_count"] == 0
    assert result.resource_audit["persistent_state_bytes"] == (
        288 * 4 + class_count * 288 + class_count * 4 + class_count * 4
    )


@pytest.mark.parametrize(
    "policy", ["A_overall_first", "B_balanced", "C_floor_first"]
)
def test_three_selection_policies_are_deterministic_and_radius_capped(policy: str) -> None:
    first, _, _, _ = _fit(5, 5, policy=policy)
    second, _, _, _ = _fit(5, 5, policy=policy)

    assert first.state.radii.tobytes() == second.state.radii.tobytes()
    assert (
        first.state.centroids_qint8.tobytes()
        == second.state.centroids_qint8.tobytes()
    )
    assert (
        first.state.centroid_scales.tobytes()
        == second.state.centroid_scales.tobytes()
    )
    assert first.state.selection_policy == policy
    ratio = float(np.max(first.state.radii) / np.min(first.state.radii))
    assert ratio <= first.state.selected_ratio_cap**2 + 1.0e-5
    assert not first.state.radii.flags.writeable
    assert not first.state.centroids_qint8.flags.writeable
    assert not first.state.centroid_scales.flags.writeable


def test_k1_uses_uniform_radius_and_is_exactly_constant_shifted_cosine() -> None:
    result, old_x, new_x, log_diag = _fit(10, 1)
    state = result.state
    rows = np.concatenate((old_x, new_x), axis=0)
    transformed = rows * np.exp(log_diag)[None, :]
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    cosine = transformed @ state.dequantized_centroids().T
    scores = score_d33_spherical_registration(state, rows)

    np.testing.assert_array_equal(state.radii, np.ones(len(state.classes), dtype=np.float32))
    np.testing.assert_allclose(scores, cosine - 1.0, atol=2.0e-6, rtol=0.0)
    assert len(result.selection_trace) == 1
    assert result.selection_trace[0]["selection_mode"] == (
        "k1_uniform_radius_pure_cosine_bypass"
    )
    assert result.selection_trace[0]["query_rows_used"] == 0


def test_all_classes_use_same_transformed_spherical_centroid_rule() -> None:
    result, old_x, new_x, log_diag = _fit(2, 5)
    rows = np.concatenate((old_x, new_x), axis=0)
    transformed = rows * np.exp(log_diag)[None, :]
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    manual = []
    for index in range(len(result.state.classes)):
        centroid = np.mean(transformed[index * 5 : (index + 1) * 5], axis=0)
        centroid /= np.linalg.norm(centroid)
        manual.append(centroid)
    manual = np.stack(manual).astype(np.float32)
    scales = np.max(np.abs(manual), axis=1).astype(np.float32) / np.float32(127.0)
    quantized = np.clip(
        np.rint(manual / scales[:, None]), -127, 127
    ).astype(np.int8)
    restored = quantized.astype(np.float32) * scales[:, None]
    restored /= np.linalg.norm(restored, axis=1, keepdims=True)
    np.testing.assert_array_equal(result.state.centroids_qint8, quantized)
    np.testing.assert_allclose(result.state.centroid_scales, scales, atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(result.state.dequantized_centroids(), restored, atol=1.0e-7)
    distances = np.clip(1.0 - transformed @ restored.T, 0.0, 2.0)
    manual_scores = (
        -distances / result.state.radii[None, :]
        - np.log(result.state.radii)[None, :]
    )
    np.testing.assert_allclose(
        score_d33_spherical_registration(result.state, rows),
        manual_scores,
        atol=2.0e-6,
    )


def test_rejects_k_mismatch_overlap_and_grid_drift() -> None:
    old_classes = ("old_a", "old_b")
    new_classes = ("new_a", "new_b")
    old_x, old_y = _support(old_classes, 2, seed=40)
    new_x, new_y = _support(new_classes, 1, seed=41)
    with pytest.raises(D33SphericalRegistrationError, match="matched K-shot"):
        fit_d33_spherical_registration(
            old_x,
            old_y,
            old_classes,
            new_x,
            new_y,
            new_classes,
            np.zeros(288, dtype=np.float32),
        )
    with pytest.raises(D33SphericalRegistrationError, match="fixed radius grid drift"):
        D33SphericalRegistrationConfig(radius_quantiles=(0.5, 0.75))
