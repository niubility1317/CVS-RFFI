from __future__ import annotations

import numpy as np
import pytest

import cvsrffi.stage2_d36_compiled_joint_int8 as d36
from cvsrffi.stage2_d36_compiled_joint_int8 import (
    D36CompiledJointConfig,
    D36CompiledJointInt8Error,
    base_score_d36_compiled_joint_int8,
    fit_d36_compiled_joint_int8,
    margin_features_d36_compiled_joint_int8,
    score_d36_compiled_joint_int8,
    with_oof_calibration_d36_compiled_joint_int8,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return np.asarray(rows / np.linalg.norm(rows, axis=1, keepdims=True), dtype=np.float32)


def _fixture(k: int = 2):
    rng = np.random.default_rng(36 + k)
    old_classes = ("old-a", "old-b")
    new_classes = ("new-a", "new-b")
    centers = np.zeros((4, 288), dtype=np.float32)
    centers[np.arange(4), np.arange(4)] = 1.0
    rows = []
    labels = []
    for i, name in enumerate(old_classes + new_classes):
        for _ in range(k):
            rows.append(centers[i] + 0.025 * rng.normal(size=288).astype(np.float32))
            labels.append(name)
    rows = _unit(np.stack(rows))
    return (
        rows[: 2 * k],
        np.asarray(labels[: 2 * k]),
        old_classes,
        rows[2 * k :],
        np.asarray(labels[2 * k :]),
        new_classes,
        np.linspace(-0.03, 0.03, 288, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("arm", "rank", "kind", "calibration_size"),
    [("A", 0, "none", 0), ("B", 2, "constant", 1), ("C", 2, "margin6_irls", 6)],
)
def test_three_arms_compile_one_int8_head_without_fp32_target_prototypes(
    arm: str, rank: int, kind: str, calibration_size: int
) -> None:
    result = fit_d36_compiled_joint_int8(
        *_fixture(), config=D36CompiledJointConfig(arm=arm)
    )
    state = result.state

    assert state.compiled_qint8.shape == (4, 288)
    assert state.compiled_qint8.dtype == np.int8
    assert state.compiled_scales_fp16.dtype == np.float16
    assert state.compiled_inverse_norms_fp16.dtype == np.float16
    assert state.radii_fp16.dtype == np.float16
    assert state.calibration_kind == kind
    assert state.calibration_fp16.size == calibration_size
    assert result.geometry_audit["rank"] == rank
    assert len(result.training_trace) == 12
    assert [row["phase"] for row in result.training_trace[:6]] == ["Stage2-B"] * 6
    assert [row["phase"] for row in result.training_trace[6:]] == ["Stage2-C"] * 6
    assert result.resource_audit["optimizer_steps"] == 12
    assert result.resource_audit["optimizer_persistent_state_bytes"] == 0
    assert result.resource_audit["resident_fp32_target_prototype_count"] == 0
    assert result.resource_audit["target_old_int8_prototype_count"] == 2
    assert result.resource_audit["target_new_int8_prototype_count"] == 2
    assert result.resource_audit["query_dot_macs"] == 4 * 288
    assert result.resource_audit["dense_query_graph_bytes"] == 0
    assert result.resource_audit["phase2_query_true_batch_class_count_access"] is False
    assert result.before_state.classes == ("old-a", "old-b")
    assert result.before_state.old_class_count == 2
    assert result.before_state.calibration_kind == "none"
    assert result.before_state.compiled_qint8.shape == (2, 288)


def test_score_is_finite_per_sample_over_all_registered_classes() -> None:
    values = _fixture()
    result = fit_d36_compiled_joint_int8(
        *values, config=D36CompiledJointConfig(arm="C")
    )
    query = np.concatenate((values[0], values[3]), axis=0)
    together = score_d36_compiled_joint_int8(result.state, query)
    separate = np.concatenate(
        [score_d36_compiled_joint_int8(result.state, row[None, :]) for row in query]
    )

    assert together.shape == (len(query), 4)
    assert np.isfinite(together).all()
    np.testing.assert_array_equal(together, separate)
    assert not together.flags.writeable
    before = score_d36_compiled_joint_int8(
        result.before_state, values[0]
    )
    assert before.shape == (len(values[0]), 2)
    assert np.isfinite(before).all()


def test_public_margin6_and_oof_calibration_replace_only_calibrator() -> None:
    values = _fixture()
    result = fit_d36_compiled_joint_int8(
        *values, config=D36CompiledJointConfig(arm="C")
    )
    support = np.concatenate((values[0], values[3]), axis=0)
    roles = np.concatenate(
        (np.zeros(len(values[0])), np.ones(len(values[3])))
    ).astype(np.int64)
    base = base_score_d36_compiled_joint_int8(result.state, support)
    psi = margin_features_d36_compiled_joint_int8(result.state, support)
    replaced = with_oof_calibration_d36_compiled_joint_int8(
        result.state, psi, roles
    )

    assert base.shape == (len(support), 4)
    assert psi.shape == (len(support), 6)
    assert not base.flags.writeable
    assert not psi.flags.writeable
    np.testing.assert_array_equal(replaced.compiled_qint8, result.state.compiled_qint8)
    np.testing.assert_array_equal(replaced.radii_fp16, result.state.radii_fp16)
    assert replaced.calibration_kind == "margin6_irls"
    assert replaced.calibration_fp16.shape == (6,)


def test_ground_anchor_is_optional_read_only_old_z_fusion() -> None:
    values = _fixture()
    anchors = _unit(np.eye(2, 160, dtype=np.float32))
    result = fit_d36_compiled_joint_int8(
        *values,
        config=D36CompiledJointConfig(arm="B"),
        ground_anchor_z=anchors,
        ground_anchor_radius=np.asarray([0.1, 0.2], dtype=np.float32),
    )

    assert result.geometry_audit["ground_anchor_z_used"] is True
    assert result.state.persistent_state_bytes < 50_000


def test_ground_anchor_preserves_target_z_block_energy_scale() -> None:
    prototype = np.zeros((2, 288), dtype=np.float32)
    prototype[:, 0] = 0.25
    prototype[:, 160] = np.sqrt(1.0 - 0.25**2)
    anchors = np.zeros((2, 160), dtype=np.float32)
    anchors[:, 1] = 1.0
    fused = d36._fuse_ground_z(
        prototype,
        np.asarray([0.05, 0.06], dtype=np.float32),
        5,
        2,
        anchors,
        np.asarray([0.1, 0.1], dtype=np.float32),
    )

    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), 1.0, atol=1.0e-6)
    assert np.all(np.linalg.norm(fused[:, :160], axis=1) < 0.30)


def test_k1_disables_rank2_residual_by_fixed_shot_shrinkage() -> None:
    result = fit_d36_compiled_joint_int8(
        *_fixture(k=1), config=D36CompiledJointConfig(arm="C")
    )

    assert result.geometry_audit["rank"] == 2
    assert result.geometry_audit["shot_shrinkage"] == 0.0
    assert len(result.training_trace) == 12


def test_rejects_nonunit_input_and_fisher_drift() -> None:
    values = list(_fixture())
    values[0] = values[0] * np.float32(1.01)
    with pytest.raises(D36CompiledJointInt8Error, match="unit"):
        fit_d36_compiled_joint_int8(*values)

    values = list(_fixture())
    values[-1] = values[-1].astype(np.float64)
    with pytest.raises(D36CompiledJointInt8Error, match="fisher_log_diag"):
        fit_d36_compiled_joint_int8(*values)


def test_fit_does_not_use_torch_from_numpy_abi_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reject_bridge(*_args, **_kwargs):
        raise AssertionError("torch.from_numpy must not be used by D36")

    monkeypatch.setattr(d36.torch, "from_numpy", _reject_bridge)
    result = fit_d36_compiled_joint_int8(
        *_fixture(), config=D36CompiledJointConfig(arm="A")
    )

    assert result.state.compiled_qint8.shape == (4, 288)
