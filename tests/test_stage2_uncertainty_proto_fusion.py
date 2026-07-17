from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ciaf import Int8DomainClassComponent
from cvsrffi.stage2_uncertainty_proto_fusion import (
    FEATURE_DIM,
    UncertaintyFusionConfig,
    UncertaintyProtoFusionError,
    append_new_classes,
    fit_old,
    predict_one,
    score_one,
)


OLD = ("old-a", "old-b", "old-c")


def _direction(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[index] = 1.0
    return value


def _component() -> Int8DomainClassComponent:
    q = np.zeros((3, len(OLD), FEATURE_DIM), dtype=np.int8)
    scale = np.full((3, len(OLD)), 1.0 / 127.0, dtype=np.float16)
    mask = np.ones((3, len(OLD)), dtype=np.uint8)
    for domain in range(3):
        for class_index in range(len(OLD)):
            q[domain, class_index, class_index] = 127
            q[domain, class_index, 10 + domain] = domain - 1
    return Int8DomainClassComponent(q, scale, mask, OLD)


def _support(classes: tuple[str, ...], k: int, *, offset: int = 0) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(classes):
        for rank in range(k):
            row = _direction(offset + class_index)
            if k > 1:
                row = row + np.float32(0.01 * (rank - (k - 1) / 2.0)) * _direction(
                    40 + class_index
                )
            rows.append(row)
            labels.append(label)
    return np.stack(rows), labels


def _state(k: int = 5):
    features, labels = _support(OLD, k)
    return fit_old(
        _component(),
        features,
        labels,
        config=UncertaintyFusionConfig(r0=0.07, r_min=1.0e-3, separation_margin=0.02),
    )


def test_closed_form_matches_inverse_uncertainty_reference() -> None:
    component = _component()
    features, labels = _support(OLD, 5)
    config = UncertaintyFusionConfig(r0=0.07, r_min=1.0e-3, separation_margin=0.02)
    state = fit_old(component, features, labels, config=config)

    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    target = normalized[np.asarray(labels) == OLD[0]]
    target_center = np.mean(target, axis=0)
    target_center /= np.linalg.norm(target_center)
    target_radius = np.quantile(np.clip(1.0 - target @ target_center, 0.0, 2.0), 0.90)
    ground = component.dequantized_class_anchors(0)
    ground_center = np.mean(ground, axis=0)
    ground_center /= np.linalg.norm(ground_center)
    ground_radius = np.quantile(np.clip(1.0 - ground @ ground_center, 0.0, 2.0), 0.90)
    w_ground = 1.0 / max(ground_radius, config.r_min) ** 2
    w_target = 5.0 / max(target_radius, config.r_min) ** 2
    lam = w_target / (w_ground + w_target)
    expected = (1.0 - lam) * ground_center + lam * target_center
    expected /= np.linalg.norm(expected)
    expected_radius = np.sqrt(1.0 / (w_ground + w_target))
    expected_radius += lam * (1.0 - lam) * np.clip(
        1.0 - ground_center @ target_center, 0.0, 2.0
    )

    np.testing.assert_allclose(state.old_ground_radius[0], ground_radius, atol=2e-7)
    np.testing.assert_allclose(state.old_target_radius[0], target_radius, atol=2e-7)
    np.testing.assert_allclose(state.old_target_weight[0], lam, atol=2e-7)
    np.testing.assert_allclose(state.prototypes[0], expected, atol=2e-7)
    np.testing.assert_allclose(state.radius[0], expected_radius, atol=2e-7)


def test_ground_component_is_readonly_and_no_fp32_ground_anchor_is_persisted() -> None:
    component = _component()
    q_before = component.domain_class_q.tobytes()
    state = _state()
    assert component.domain_class_q.tobytes() == q_before
    assert not component.domain_class_q.flags.writeable
    assert not hasattr(state, "ground_centers")
    assert not hasattr(state, "ground_anchors")
    assert state.old_target_prototypes.dtype == np.float32
    assert state.prototypes.dtype == np.float32
    assert not state.prototypes.flags.writeable
    with pytest.raises(ValueError):
        state.prototypes[0, 0] = 0.0


@pytest.mark.parametrize("k", (1, 5, 10))
def test_k1_r0_and_target_only_new_registration(k: int) -> None:
    state = _state(k)
    new_classes = ("new-x", "new-y")
    new_features, new_labels = _support(new_classes, k, offset=20)
    after = append_new_classes(state, new_features, new_labels)
    assert after.classes == OLD + new_classes
    assert after.prototypes.dtype == np.float32
    assert after.old_ground_radius.shape == (len(OLD),)
    assert after.old_target_weight.shape == (len(OLD),)
    if k == 1:
        np.testing.assert_array_equal(state.old_target_radius, np.float32(0.07))
        np.testing.assert_array_equal(after.radius[len(OLD) :], np.float32(0.07))
    np.testing.assert_allclose(
        after.prototypes[len(OLD) :],
        np.stack([_direction(20), _direction(21)]),
        atol=1.0e-6,
    )


def test_stage2c_append_freezes_old_bytes_hash_and_score_columns() -> None:
    before = _state()
    probe = _direction(0) + 0.2 * _direction(20)
    scores_before = score_one(before, probe)
    payload_before = b"".join(
        value.tobytes()
        for value in (
            before.prototypes,
            before.radius,
            before.support_count_by_class,
            before.old_target_prototypes,
            before.old_ground_radius,
            before.old_target_radius,
            before.old_target_weight,
        )
    )
    new_features, new_labels = _support(("new-x", "new-y"), 5, offset=20)
    after = append_new_classes(
        before,
        new_features,
        new_labels,
        registered_classes=("new-y", "new-x"),
    )
    payload_after = b"".join(
        value.tobytes()
        for value in (
            after.prototypes[: len(OLD)],
            after.radius[: len(OLD)],
            after.support_count_by_class[: len(OLD)],
            after.old_target_prototypes,
            after.old_ground_radius,
            after.old_target_radius,
            after.old_target_weight,
        )
    )
    assert after.old_prefix_sha256 == before.old_prefix_sha256
    assert payload_after == payload_before
    np.testing.assert_array_equal(score_one(after, probe)[: len(OLD)], scores_before)


def test_geometry_audit_reports_all_pairs_and_collision_gap() -> None:
    state = _state(1)
    colliding = np.stack([state.prototypes[0]])
    after = append_new_classes(state, colliding, ["new-collision"])
    audit = after.geometry_audit()
    assert audit["pair_count"] == after.class_count * (after.class_count - 1) // 2
    assert audit["pass"] is False
    pair = next(
        value
        for value in audit["collision_pairs"]
        if value["right_class"] == "new-collision" and value["left_class"] == OLD[0]
    )
    assert pair["left_role"] == "old"
    assert pair["right_role"] == "new"
    assert pair["gap"] <= 0.0
    assert pair["center_cosine_distance"] <= pair["required_distance"]
    assert audit["support_derived_only"] is True
    assert audit["query_rows_used"] == 0


def test_resource_audit_counts_ground_target_fusion_and_query_macs() -> None:
    state = _state()
    new_features, new_labels = _support(("new-x", "new-y"), 5, offset=20)
    after = append_new_classes(state, new_features, new_labels)
    audit = after.resource_audit()
    assert audit["int8_ground_component_state_bytes"] == _component().state_bytes
    assert audit["target_fp32_state_bytes"] == after.target_fp32_state_bytes
    assert audit["fusion_metadata_state_bytes"] == after.fusion_metadata_state_bytes
    assert audit["persistent_state_bytes"] == after.persistent_state_bytes
    assert audit["persistent_state_limit_pass"] is True
    assert audit["estimated_macs_per_query"] == after.class_count * FEATURE_DIM
    assert audit["persistent_full_precision_ground_anchor_count"] == 0
    assert audit["trainable_parameters"] == 0
    assert audit["adaptation_epochs"] == 0
    assert audit["dense_query_graph_bytes"] == 0
    assert audit["query_rows_used_for_fit"] == 0


def test_scoring_is_single_sample_all_registered_and_readonly() -> None:
    state = _state()
    new_features, new_labels = _support(("new-x",), 5, offset=20)
    after = append_new_classes(state, new_features, new_labels)
    label, scores = predict_one(after, _direction(20))
    assert label == "new-x"
    assert scores.shape == (after.class_count,)
    assert not scores.flags.writeable
    with pytest.raises(UncertaintyProtoFusionError, match="exactly one"):
        score_one(after, np.stack([_direction(0), _direction(1)]))


def test_public_fit_api_has_no_query_or_forbidden_oracle_surface() -> None:
    forbidden = ("query", "truth", "role", "quota", "assignment", "source", "clean")
    for function in (fit_old, append_new_classes, score_one, predict_one):
        names = inspect.signature(function).parameters
        assert not any(token in name.lower() for name in names for token in forbidden)


def test_registration_rejects_overlap_unbalanced_or_k_drift() -> None:
    state = _state(5)
    with pytest.raises(UncertaintyProtoFusionError, match="overlap"):
        append_new_classes(state, np.stack([_direction(0)] * 5), [OLD[0]] * 5)
    with pytest.raises(UncertaintyProtoFusionError, match="K-shot"):
        append_new_classes(state, np.stack([_direction(20)] * 4), ["new-x"] * 4)
    rows = np.stack([_direction(20)] * 5 + [_direction(21)] * 4)
    labels = ["new-x"] * 5 + ["new-y"] * 4
    with pytest.raises(UncertaintyProtoFusionError, match="K-shot"):
        append_new_classes(state, rows, labels)
