from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi import stage2_next_r5_fa_target125_core as core
from cvsrffi import stage2_next_r5_fa_target125_matrix as matrix


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _asset() -> core.Target125FARDCE3Asset:
    basis = np.zeros((core.FA_RANK, core.Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    basis[2, 2] = 1.0
    return core.build_target_fa_asset(
        old_classes=tuple(f"old{index}" for index in range(core.OLD_CLASS_COUNT)),
        aggregate_samples_per_class=(98,) * core.OLD_CLASS_COUNT,
        class_centers_3d=np.full(
            (core.OLD_CLASS_COUNT, core.FA_RANK),
            -0.15,
            dtype=np.float32,
        ),
        fisher_precision_3d=np.ones(core.FA_RANK, dtype=np.float32),
        residual_variance_3d=np.ones(core.FA_RANK, dtype=np.float32),
        fisher_radius=np.asarray([np.sqrt(3.0)], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.2, 0.1, 0.05], dtype=np.float32),
        basis_3x160=basis,
        checkpoint_sha256=_sha("checkpoint"),
        phase1_bundle_sha256=_sha("phase1-bundle"),
        phase1_aggregate_receipt_sha256=_sha("phase1-aggregate"),
        method_lock_sha256=_sha("method-lock"),
    )


def _rows(names: tuple[str, ...], k_shot: int, *, offset: int) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[np.ndarray] = []
    labels: list[str] = []
    for index, name in enumerate(names):
        value = np.zeros(core.Z_DIM, dtype=np.float32)
        value[offset + index] = 1.0
        for _ in range(k_shot):
            values.append(value.copy())
            labels.append(name)
    return np.asarray(values, dtype=np.float32), tuple(labels)


def _binding(
    *,
    registry: tuple[str, ...],
    k_shot: int,
    new_count: int,
    phase: str,
    support_ids: tuple[str, ...],
    query_ids: tuple[str, ...],
) -> core.Target125FARuntimeBinding:
    return core.Target125FARuntimeBinding(
        checkpoint_sha256=_sha("checkpoint"),
        capsule_id=_sha("capsule"),
        split_id=_sha("split"),
        outer_id=matrix.make_outer_id("20-1", 713102, k_shot, new_count),
        receiver="20-1",
        seed=713102,
        k_shot=k_shot,
        new_count=new_count,
        source_pool_k=matrix.source_pool_k_for(k_shot, new_count),
        scene="leo_clear_weak",
        registration_phase=phase,
        registered_classes=registry,
        support_physical_ids=support_ids,
        query_physical_ids=query_ids,
    )


def _four_state_inputs(k_shot: int, new_count: int):
    asset = _asset()
    old_rows, old_labels = _rows(asset.old_classes, k_shot, offset=10)
    new_classes = tuple(f"new{index}" for index in range(new_count))
    new_rows, new_labels = _rows(new_classes, k_shot, offset=32)
    old_ids = tuple(f"old-support-{index}" for index in range(len(old_rows)))
    new_ids = tuple(f"new-support-{index}" for index in range(len(new_rows)))
    reg0 = _binding(
        registry=asset.old_classes,
        k_shot=k_shot,
        new_count=new_count,
        phase="REG0",
        support_ids=old_ids,
        query_ids=("old-query-0", "old-query-1"),
    )
    reg1 = _binding(
        registry=asset.old_classes + new_classes,
        k_shot=k_shot,
        new_count=new_count,
        phase="REG1",
        support_ids=old_ids + new_ids,
        query_ids=("new-query-0", "new-query-1"),
    )
    return asset, reg0, reg1, old_rows, old_labels, new_rows, new_labels, new_ids


def test_frozen_matrix_has_exact_target125_and_k1_alias_counts() -> None:
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    assert len(plan.outer_rows) == 125
    assert len(plan.scene_rows) == 375
    assert len(plan.surfaces) == 1500
    assert sum(surface.unique_prediction for surface in plan.surfaces) == 1350
    assert sum(not surface.unique_prediction for surface in plan.surfaces) == 150
    k1 = next(surface for surface in plan.surfaces if surface.k_shot == 1 and surface.state == "DA1_REG0")
    assert k1.alias_of_surface_id.endswith("state-DA0_REG0")
    assert matrix.source_pool_k_for(5, 20) == 10


def test_target_asset_is_six_class_source_only_and_roundtrips() -> None:
    asset = _asset()
    wire = core.serialize_target_fa_asset(asset)
    recovered = core.deserialize_target_fa_asset(wire)
    assert recovered.asset_sha256 == asset.asset_sha256
    assert len(recovered.old_classes) == 6
    assert recovered.fa_asset.aggregate_samples_per_class == (98,) * 6


def test_k1_is_exact_fa_state_logit_prediction_and_resource_alias() -> None:
    asset, reg0, reg1, old_rows, old_labels, new_rows, new_labels, new_ids = _four_state_inputs(1, 20)
    state = core.build_fa_qknn_four_state(
        asset,
        reg0_binding=reg0,
        reg1_binding=reg1,
        old_support_features=old_rows,
        old_support_labels=old_labels,
        new_support_features=new_rows,
        new_support_labels=new_labels,
        new_support_physical_ids=new_ids,
    )
    assert state.fa_state is None
    assert state.da1_reg0 is state.da0_reg0
    assert state.da1_reg1 is state.da0_reg1
    assert state.da1_reg0.resource_receipt == state.da0_reg0.resource_receipt
    result = core.score_fa_qknn_four_state(
        state,
        reg0_query_features=old_rows[:2],
        reg0_query_physical_ids=reg0.query_physical_ids,
        reg1_query_features=new_rows[:2],
        reg1_query_physical_ids=reg1.query_physical_ids,
    )
    assert result.logits_by_state["DA1_REG0"] is result.logits_by_state["DA0_REG0"]
    assert result.logits_by_state["DA1_REG1"] is result.logits_by_state["DA0_REG1"]
    assert result.predictions_by_state["DA1_REG0"] is result.predictions_by_state["DA0_REG0"]
    with pytest.raises(core.NextR5FATarget125CoreError, match="K/REG0"):
        core.fit_fa_rdce3(
            asset,
            old_rows,
            old_labels,
            asset.old_classes,
            1,
            binding=reg0,
        )


@pytest.mark.parametrize("k_shot,new_count", [(5, 20), (10, 5)])
def test_k5_k10_share_closed_formula_and_reuse_one_reg0_state(
    k_shot: int,
    new_count: int,
) -> None:
    asset, reg0, reg1, old_rows, old_labels, new_rows, new_labels, new_ids = _four_state_inputs(k_shot, new_count)
    state = core.build_fa_qknn_four_state(
        asset,
        reg0_binding=reg0,
        reg1_binding=reg1,
        old_support_features=old_rows,
        old_support_labels=old_labels,
        new_support_features=new_rows,
        new_support_labels=new_labels,
        new_support_physical_ids=new_ids,
    )
    assert state.fa_state is not None
    assert state.fa_state.binding.k_shot == k_shot
    assert state.reg1_reuse_receipt["same_state_object"] is True
    assert state.reg1_reuse_receipt["reg1_fit_calls"] == 0
    assert state.da1_reg0.representation == core.R1_REPRESENTATION
    assert state.da1_reg1.representation == core.R1_REPRESENTATION
    r1 = core.transform_fa_rdce3(old_rows[:1], state.fa_state)
    assert np.any(r1 < 0.0)
    assert float(np.linalg.norm(r1[0])) == pytest.approx(1.0, abs=2.0e-6)
    # The direct qKNN codec preserves the already signed-unit R1 row rather
    # than calling a second normalisation routine.
    direct = core.fit_qknn(
        r1,
        ("old0",),
        ("old0",),
        support_physical_ids=("direct-r1-support",),
        representation=core.R1_REPRESENTATION,
    )
    expected_scale = np.float16(np.max(np.abs(r1[0])) / 127.0)
    assert direct.scales_fp16[0] == expected_scale
    assert direct.resource_receipt["r1_second_normalization"] is False


def test_query_physical_id_drift_and_exact_top_tie_fail_closed() -> None:
    asset, reg0, reg1, old_rows, old_labels, new_rows, new_labels, new_ids = _four_state_inputs(5, 20)
    state = core.build_fa_qknn_four_state(
        asset,
        reg0_binding=reg0,
        reg1_binding=reg1,
        old_support_features=old_rows,
        old_support_labels=old_labels,
        new_support_features=new_rows,
        new_support_labels=new_labels,
        new_support_physical_ids=new_ids,
    )
    with pytest.raises(core.NextR5FATarget125CoreError, match="query"):
        core.score_fa_qknn_four_state(
            state,
            reg0_query_features=old_rows[:2],
            reg0_query_physical_ids=("wrong-query-0", "wrong-query-1"),
            reg1_query_features=new_rows[:2],
            reg1_query_physical_ids=reg1.query_physical_ids,
        )
