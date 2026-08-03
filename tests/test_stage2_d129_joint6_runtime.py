from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d129_joint6_da as da
from cvsrffi import stage2_d129_joint6_heads as heads
from cvsrffi import stage2_d129_joint6_matrix as matrix
from cvsrffi import stage2_d129_joint6_runtime as runtime
from cvsrffi import stage2_zid_student_t_qknn as qknn


def _lock(k_shot: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=4.0,
        kernel_effective_dim=160,
        kernel_volume_gamma=1.0,
        shared_h0=0.45,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _assets_and_row(k_shot: int):
    rng = np.random.default_rng(9129 + k_shot)
    phase1 = rng.normal(size=(7, 6, 4, 160)).astype(np.float32)
    receiver_shift = rng.normal(size=(7, 1, 1, 160)).astype(np.float32) * 0.2
    class_shift = rng.normal(size=(1, 6, 1, 160)).astype(np.float32) * 0.3
    phase1 = phase1 * np.float32(0.08) + receiver_shift + class_shift
    classes = tuple([f"tx{i}" for i in range(5)] + ["tx_new"])
    receivers = tuple(f"rx{i}" for i in range(7))
    cells = {
        (receiver, class_id): tuple(
            f"p-{receiver}-{class_id}-{index}" for index in range(14)
        )
        for receiver in receivers
        for class_id in classes
    }

    def ordered(receiver, class_id):
        return tuple(
            sorted(
                cells[(receiver, class_id)],
                key=lambda physical_id: hashlib.sha256(
                    f"{da.LOCO_SALT}|{receiver}|{class_id}|{physical_id}".encode()
                ).hexdigest(),
            )
        )

    loco = da.build_d129_loco_plan(
        da.D129LOCORecord(receiver, class_id, physical_id)
        for (receiver, class_id), physical_ids in cells.items()
        for physical_id in physical_ids
    )
    held_receiver = "rx0"
    held_class = "tx_new"
    retained = classes[:-1]
    row_k1 = matrix.Joint6LocoRow(
        row_id=f"rx={held_receiver}|held={held_class}|K=1",
        held_receiver=held_receiver,
        held_class=held_class,
        active_k=1,
        retained_classes=retained,
        registered_classes=classes,
    )
    row_k5 = matrix.Joint6LocoRow(
        row_id=f"rx={held_receiver}|held={held_class}|K=5",
        held_receiver=held_receiver,
        held_class=held_class,
        active_k=5,
        retained_classes=retained,
        registered_classes=classes,
    )
    fold = next(
        value
        for value in loco.folds
        if value.held_receiver == held_receiver and value.held_class == held_class
    )
    phase1_fit_ids = tuple(
        physical_id
        for receiver in receivers
        if receiver != held_receiver
        for class_id in classes
        if class_id != held_class
        for physical_id in ordered(receiver, class_id)
    )
    support5_by_class = {
        class_id: ordered(held_receiver, class_id)[:5] for class_id in classes
    }
    support1_by_class = {
        class_id: values[:1] for class_id, values in support5_by_class.items()
    }
    query_by_class = {
        class_id: ordered(held_receiver, class_id)[5:] for class_id in classes
    }
    binding = matrix.bind_joint6_physical_ids(
        row_k1=row_k1,
        row_k5=row_k5,
        loco_fold_receipt=fold.as_dict(),
        phase1_fit_ids=phase1_fit_ids,
        k1_support_ids_by_class=support1_by_class,
        k5_support_ids_by_class=support5_by_class,
        query_ids_by_class=query_by_class,
    )
    assets = da.build_d129_phase1_assets(
        phase1,
        checkpoint_sha256="3" * 64,
        phase1_seal_sha256=binding["phase1_seal_sha256"],
    )
    centers = rng.normal(size=(6, 160)).astype(np.float32)
    support = np.vstack(
        [
            centers[index]
            + np.float32(0.1) * rng.normal(size=(k_shot, 160)).astype(np.float32)
            for index in range(6)
        ]
    ).astype(np.float32)
    if k_shot == 5:
        nuisance = da.decode_cspar2_basis(assets[0])[:, 0].astype(np.float32)
        coefficients = np.asarray([-0.45, -0.2, 0.0, 0.2, 0.45], dtype=np.float32)
        support = support.reshape(6, 5, 160)
        support += coefficients[None, :, None] * nuisance[None, None, :]
        support = support.reshape(30, 160)
    query = np.vstack(
        [
            centers[index]
            + np.float32(0.12) * rng.normal(size=(9, 160)).astype(np.float32)
            for index in range(6)
        ]
    ).astype(np.float32)
    labels = tuple(class_id for class_id in classes for _ in range(k_shot))
    support_ids_by_class = support1_by_class if k_shot == 1 else support5_by_class
    support_ids = tuple(
        physical_id for class_id in classes for physical_id in support_ids_by_class[class_id]
    )
    query_ids = tuple(
        physical_id for class_id in classes for physical_id in query_by_class[class_id]
    )
    row = row_k1 if k_shot == 1 else row_k5
    common_r0 = heads.build_d129_common_r0(
        base_support_zid=support,
        base_query_zid=query,
        support_labels=labels,
        registered_classes=classes,
        old_class_count=5,
        partition_semantics="phase1_seen_class_loco_directional_proxy",
        opaque_query_ids=query_ids,
        qknn_lock=_lock(k_shot),
    )
    return (
        assets,
        classes,
        support,
        query,
        labels,
        support_ids,
        query_ids,
        binding,
        row,
        common_r0,
    )


@pytest.mark.parametrize("candidate_index", [0, 1])
@pytest.mark.parametrize("k_shot", [1, 5])
def test_candidate_runtime_closes_two_caches_six_heads_and_no_truth(
    candidate_index: int, k_shot: int
) -> None:
    (
        assets,
        classes,
        support,
        query,
        labels,
        support_ids,
        query_ids,
        binding,
        _row,
        common_r0,
    ) = _assets_and_row(k_shot)
    result = runtime.run_d129_candidate_joint6(
        asset=assets[candidate_index],
        base_support_zid160=support,
        base_query_zid160=query,
        support_labels=labels,
        support_physical_ids=support_ids,
        registered_classes=classes,
        retained_class_count=5,
        opaque_query_ids=query_ids,
        qknn_lock=_lock(k_shot),
        fold_binding=binding,
        common_r0=common_r0,
    )
    assert tuple(arm.arm_id for arm in result.six_arm.arms) == matrix.ARM_IDS
    assert result.runtime_receipt["representation_cache_count"] == 2
    assert result.runtime_receipt["heads_per_representation"] == 3
    assert result.runtime_receipt["common_r0_head_fit_calls_in_candidate_runtime"] == 0
    assert result.runtime_receipt["query_rows_used_for_fit"] == 0
    assert result.runtime_receipt["query_state_updates"] == 0
    assert result.runtime_receipt["truth_input_exists"] is False
    assert result.runtime_receipt["source_runtime_access"] is False
    assert result.query_read_only_receipt["protocol_closed"] is True
    assert result.smoke_receipt["feature_change_alone_is_sufficient"] is False
    assert result.smoke_receipt["smoke_pass"] is True


def test_prediction_row_is_truth_free_and_held_proxy_is_registry_suffix() -> None:
    (
        assets,
        classes,
        support,
        query,
        labels,
        support_ids,
        query_ids,
        binding,
        row,
        common_r0,
    ) = _assets_and_row(5)
    result = runtime.run_d129_candidate_joint6(
        asset=assets[0],
        base_support_zid160=support,
        base_query_zid160=query,
        support_labels=labels,
        support_physical_ids=support_ids,
        registered_classes=classes,
        retained_class_count=5,
        opaque_query_ids=query_ids,
        qknn_lock=_lock(5),
        fold_binding=binding,
        common_r0=common_r0,
    )
    payload = runtime.build_joint6_prediction_row(row, result)
    assert payload["registered_classes"][-1] == payload["held_class"]
    assert set(payload["arms"]) == set(matrix.ARM_IDS)
    assert "truth" not in repr(dict(payload)).lower()


def test_runtime_has_no_truth_role_quota_or_query_fit_api() -> None:
    parameters = set(inspect.signature(runtime.run_d129_candidate_joint6).parameters)
    assert not parameters & {
        "truth",
        "query_labels",
        "query_role",
        "class_quota",
        "source_rows",
        "clean_rows",
        "optimizer",
        "old_class_count",
    }
