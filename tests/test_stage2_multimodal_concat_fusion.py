from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ciaf import Int8DomainClassComponent
from cvsrffi.stage2_multimodal_concat_fusion import (
    FEATURE_DIM,
    FFT_DIM,
    RF_DIM,
    Z_DIM,
    MultimodalConcatConfig,
    MultimodalConcatFusionError,
    append_new_classes_concat,
    build_concat288,
    fit_old_concat,
    predict_one,
    score_one,
)


OLD = ("old-a", "old-b", "old-c")
NEW = ("new-x", "new-y")
DEFAULT_ENERGY = (5.0 / 9.0, 1.0 / 3.0, 1.0 / 9.0)


def _direction(dimension: int, index: int) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float32)
    value[index] = 1.0
    return value


def _component(*, shifted: bool = False, domains: int = 4) -> Int8DomainClassComponent:
    q = np.zeros((domains, len(OLD), Z_DIM), dtype=np.int8)
    scale = np.full((domains, len(OLD)), 1.0 / 127.0, dtype=np.float16)
    mask = np.ones((domains, len(OLD)), dtype=np.uint8)
    for domain in range(domains):
        for class_index in range(len(OLD)):
            primary = 20 + class_index if shifted else class_index
            q[domain, class_index, primary] = 127
            q[domain, class_index, 40 + domain] = domain + 1
    return Int8DomainClassComponent(q, scale, mask, OLD)


def _support(
    classes: tuple[str, ...],
    k: int,
    *,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(classes):
        for rank in range(k):
            z = _direction(Z_DIM, offset + class_index)
            fft = _direction(FFT_DIM, offset + class_index)
            rf = _direction(RF_DIM, offset + class_index)
            if k > 1:
                delta = np.float32(0.01 * (rank - (k - 1) / 2.0))
                z = z + delta * _direction(Z_DIM, 80 + class_index)
                fft = fft + delta * _direction(FFT_DIM, 60 + class_index)
                rf = rf + delta * _direction(RF_DIM, 20 + class_index)
            z_rows.append(z)
            fft_rows.append(fft)
            rf_rows.append(rf)
            labels.append(label)
    return np.stack(z_rows), np.stack(fft_rows), np.stack(rf_rows), labels


def _old_state(k: int = 5, *, component: Int8DomainClassComponent | None = None):
    z, fft, rf, labels = _support(OLD, k, offset=0)
    return fit_old_concat(
        _component() if component is None else component,
        z,
        fft,
        rf,
        labels,
        config=MultimodalConcatConfig(
            r0_by_block=(0.07, 0.08, 0.09),
            separation_margin=0.01,
        ),
    )


def test_concat_is_288d_readonly_unit_norm_and_dimension_energy_balanced() -> None:
    z, fft, rf, _ = _support(OLD, 2, offset=0)
    feature = build_concat288(z, fft, rf)
    assert feature.shape == (len(z), FEATURE_DIM)
    assert FEATURE_DIM == Z_DIM + FFT_DIM + RF_DIM == 288
    assert feature.dtype == np.float32
    assert not feature.flags.writeable
    np.testing.assert_allclose(np.linalg.norm(feature, axis=1), 1.0, atol=1.0e-6)

    block_energy = np.stack(
        [
            np.sum(feature[:, :Z_DIM] ** 2, axis=1),
            np.sum(feature[:, Z_DIM : Z_DIM + FFT_DIM] ** 2, axis=1),
            np.sum(feature[:, Z_DIM + FFT_DIM :] ** 2, axis=1),
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        block_energy,
        np.tile(np.asarray(DEFAULT_ENERGY, dtype=np.float32), (len(feature), 1)),
        atol=1.0e-6,
    )


def test_energy_audit_removes_historical_w4_auxiliary_domination() -> None:
    state = _old_state()
    audit = state.resource_audit()
    np.testing.assert_allclose(audit["block_energy"], DEFAULT_ENERGY, atol=1.0e-12)
    d25_auxiliary_energy = audit["block_energy"][1] + audit["block_energy"][2]
    historical_w4_auxiliary_energy = 16.0 / 17.0
    assert d25_auxiliary_energy == pytest.approx(4.0 / 9.0)
    assert d25_auxiliary_energy < 0.5
    assert historical_w4_auxiliary_energy > 0.94
    assert d25_auxiliary_energy < historical_w4_auxiliary_energy


def test_concat_rejects_misaligned_blocks_and_each_zero_norm_block() -> None:
    z, fft, rf, _ = _support(OLD, 2, offset=0)
    with pytest.raises(MultimodalConcatFusionError, match="aligned row"):
        build_concat288(z, fft[:-1], rf)
    for name in ("z", "fft", "rf"):
        changed = {"z": z.copy(), "fft": fft.copy(), "rf": rf.copy()}
        changed[name][0] = 0.0
        with pytest.raises(MultimodalConcatFusionError, match="zero-norm"):
            build_concat288(changed["z"], changed["fft"], changed["rf"])


def test_ground_component_changes_only_identity_block() -> None:
    reference = _old_state(component=_component(shifted=False))
    shifted = _old_state(component=_component(shifted=True))
    assert not np.array_equal(reference.prototype_z, shifted.prototype_z)
    np.testing.assert_array_equal(reference.prototype_fft, shifted.prototype_fft)
    np.testing.assert_array_equal(reference.prototype_rf, shifted.prototype_rf)
    np.testing.assert_array_equal(reference.radius_fft, shifted.radius_fft)
    np.testing.assert_array_equal(reference.radius_rf, shifted.radius_rf)
    assert not hasattr(reference, "ground_fft")
    assert not hasattr(reference, "ground_rf")
    assert not hasattr(reference, "ground_prototypes")


def test_k1_uses_locked_target_radius_in_all_three_blocks() -> None:
    state = _old_state(1)
    np.testing.assert_array_equal(state.old_target_radius_z, np.float32(0.07))
    np.testing.assert_array_equal(state.radius_fft, np.float32(0.08))
    np.testing.assert_array_equal(state.radius_rf, np.float32(0.09))

    z, fft, rf, labels = _support(NEW, 1, offset=10)
    after = append_new_classes_concat(state, z, fft, rf, labels)
    old_count = state.old_class_count
    np.testing.assert_array_equal(after.radius_z[old_count:], np.float32(0.07))
    np.testing.assert_array_equal(after.radius_fft[old_count:], np.float32(0.08))
    np.testing.assert_array_equal(after.radius_rf[old_count:], np.float32(0.09))
    assert np.isfinite(after.radius_z).all()


def test_new_classes_are_pure_target_block_prototypes() -> None:
    state = _old_state()
    z, fft, rf, labels = _support(NEW, state.k_shot, offset=10)
    after = append_new_classes_concat(state, z, fft, rf, labels)
    expected_z = []
    expected_fft = []
    expected_rf = []
    label_array = np.asarray(labels)
    for label in NEW:
        for rows, output in ((z, expected_z), (fft, expected_fft), (rf, expected_rf)):
            normalized = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            center = normalized[label_array == label].mean(axis=0)
            output.append(center / np.linalg.norm(center))
    suffix = slice(state.old_class_count, None)
    np.testing.assert_allclose(after.prototype_z[suffix], expected_z, atol=1.0e-7)
    np.testing.assert_allclose(after.prototype_fft[suffix], expected_fft, atol=1.0e-7)
    np.testing.assert_allclose(after.prototype_rf[suffix], expected_rf, atol=1.0e-7)
    assert after.old_ground_radius_z.shape == state.old_ground_radius_z.shape
    assert after.old_target_weight_z.shape == state.old_target_weight_z.shape


def test_append_freezes_old_prefix_payload_and_score_columns_bitwise() -> None:
    before = _old_state()
    probe = build_concat288(
        _direction(Z_DIM, 0)[None, :],
        _direction(FFT_DIM, 0)[None, :],
        _direction(RF_DIM, 0)[None, :],
        block_energy=before.config.block_energy,
    )[0]
    scores_before = score_one(before, probe)
    old_payload_before = b"".join(
        value.tobytes()
        for value in (
            before.prototype_z,
            before.prototype_fft,
            before.prototype_rf,
            before.radius_z,
            before.radius_fft,
            before.radius_rf,
            before.support_count_by_class,
            before.old_target_z,
            before.old_ground_radius_z,
            before.old_target_radius_z,
            before.old_target_weight_z,
        )
    )
    z, fft, rf, labels = _support(NEW, before.k_shot, offset=10)
    after = append_new_classes_concat(
        before,
        z,
        fft,
        rf,
        labels,
        registered_classes=("new-y", "new-x"),
    )
    count = before.old_class_count
    old_payload_after = b"".join(
        value.tobytes()
        for value in (
            after.prototype_z[:count],
            after.prototype_fft[:count],
            after.prototype_rf[:count],
            after.radius_z[:count],
            after.radius_fft[:count],
            after.radius_rf[:count],
            after.support_count_by_class[:count],
            after.old_target_z,
            after.old_ground_radius_z,
            after.old_target_radius_z,
            after.old_target_weight_z,
        )
    )
    assert after.old_prefix_sha256 == before.old_prefix_sha256
    assert old_payload_after == old_payload_before
    np.testing.assert_array_equal(score_one(after, probe)[:count], scores_before)


def test_geometry_audit_covers_old_old_old_new_and_new_new_roles() -> None:
    before = _old_state(1)
    z, fft, rf, labels = _support(NEW, 1, offset=10)
    after = append_new_classes_concat(before, z, fft, rf, labels)
    audit = after.geometry_audit()
    assert audit["pair_count"] == after.class_count * (after.class_count - 1) // 2
    roles = {(row["left_role"], row["right_role"]) for row in audit["pairs"]}
    assert roles == {("old", "old"), ("old", "new"), ("new", "new")}
    assert all("gap" in row and isinstance(row["pass"], bool) for row in audit["pairs"])
    assert audit["support_derived_only"] is True
    assert audit["query_rows_used"] == 0


def test_resource_audit_accounts_for_blocks_ground_scratch_and_no_training() -> None:
    component = _component(domains=4)
    state = _old_state(component=component)
    audit = state.resource_audit()
    assert audit["feature_dimension"] == 288
    assert audit["block_dimensions"] == [160, 96, 32]
    assert audit["max_active_ground_domain_count"] == 4
    assert audit["int8_ground_component_state_bytes"] == component.state_bytes
    assert audit["target_fp32_state_bytes"] == state.target_fp32_state_bytes
    assert audit["persistent_state_bytes"] == state.persistent_state_bytes
    assert audit["persistent_state_limit_pass"] is True
    assert audit["registered_prototype_dot_macs_per_query"] == state.class_count * 288
    assert audit["estimated_head_macs_per_query"] == 288 + state.class_count * 288
    assert audit["fft96_calls_per_physical_sample"] == 1
    assert audit["rf32_calls_per_physical_sample"] == 1
    assert audit["fft96_rf32_included_in_head_macs"] is False
    assert audit["trainable_parameters"] == 0
    assert audit["adaptation_epochs"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["support_row_multiplicity"] == 1
    assert audit["additional_leo_overlay_count"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["dense_query_graph_bytes"] == 0


def test_scoring_is_single_sample_all_registered_and_readonly() -> None:
    state = _old_state()
    z, fft, rf, labels = _support(("new-x",), state.k_shot, offset=10)
    after = append_new_classes_concat(state, z, fft, rf, labels)
    feature = build_concat288(z[:1], fft[:1], rf[:1], block_energy=state.config.block_energy)[0]
    label, scores = predict_one(after, feature)
    assert label == "new-x"
    assert scores.shape == (after.class_count,)
    assert not scores.flags.writeable
    with pytest.raises(MultimodalConcatFusionError, match="exactly one"):
        score_one(after, np.stack([feature, feature]))


def test_public_fit_and_append_signatures_have_no_forbidden_oracle_inputs() -> None:
    forbidden = ("query", "truth", "role", "quota", "assignment", "source", "clean")
    for function in (fit_old_concat, append_new_classes_concat):
        names = inspect.signature(function).parameters
        assert not any(token in name.lower() for name in names for token in forbidden)


def test_append_rejects_overlap_k_drift_and_unbalanced_support() -> None:
    state = _old_state(5)
    z, fft, rf, _ = _support((OLD[0],), 5, offset=0)
    with pytest.raises(MultimodalConcatFusionError, match="overlap"):
        append_new_classes_concat(state, z, fft, rf, [OLD[0]] * 5)

    z, fft, rf, labels = _support(("new-x",), 4, offset=10)
    with pytest.raises(MultimodalConcatFusionError, match="K-shot"):
        append_new_classes_concat(state, z, fft, rf, labels)

    zx, fftx, rfx, labels_x = _support(("new-x",), 5, offset=10)
    zy, ffty, rfy, labels_y = _support(("new-y",), 4, offset=11)
    with pytest.raises(MultimodalConcatFusionError, match="K-shot"):
        append_new_classes_concat(
            state,
            np.concatenate([zx, zy]),
            np.concatenate([fftx, ffty]),
            np.concatenate([rfx, rfy]),
            labels_x + labels_y,
        )


def test_pure_target_old_fit_preserves_explicit_nonlexical_registry_order() -> None:
    classes = ("old-c", "old-a", "old-b")
    z, fft, rf, labels = _support(classes, 2, offset=0)
    state = fit_old_concat(
        None,
        z,
        fft,
        rf,
        labels,
        registered_classes=classes,
        config=MultimodalConcatConfig(use_ground_identity_fusion=False),
    )
    assert state.classes == classes
    with pytest.raises(MultimodalConcatFusionError, match="registry"):
        fit_old_concat(
            None,
            z,
            fft,
            rf,
            labels,
            registered_classes=("old-a", "old-b", "missing"),
            config=MultimodalConcatConfig(use_ground_identity_fusion=False),
        )
