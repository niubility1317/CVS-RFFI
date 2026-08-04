from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi import stage2_next_r5_fa_target125_matrix as matrix
from cvsrffi import stage2_next_r5_fa_target125_core as core
from cvsrffi import stage2_next_r5_fa_target125_runtime as runtime
from cvsrffi.stage2_next_r5_fa_target125_runtime import (
    NextR5FATarget125RuntimeError,
    Target125ConditionInput,
    Target125RegistrationInput,
    build_target125_runtime_bindings,
    query_isolation_receipt,
)


def _rows(count: int, *, offset: int = 0) -> np.ndarray:
    value = np.zeros((count, matrix.FEATURE_DIM), dtype=np.float32)
    for index in range(count):
        value[index, (offset + index) % matrix.FEATURE_DIM] = 1.0
    return value


def _condition() -> Target125ConditionInput:
    outer = next(row for row in matrix.freeze_next_r5_fa_target125_matrix().outer_rows if row.k_shot == 1)
    old = tuple(f"old-{index}" for index in range(matrix.OLD_CLASS_COUNT))
    new = tuple(f"new-{index}" for index in range(outer.new_count))
    old_ids = tuple(f"old-support-{index}" for index in range(len(old)))
    reg0 = Target125RegistrationInput(
        registration_phase="REG0",
        registered_classes=old,
        support_zid160=_rows(len(old)),
        support_labels=old,
        support_physical_ids=old_ids,
        query_zid160=_rows(len(old), offset=40),
        query_physical_ids=tuple(f"old-query-{index}" for index in range(len(old))),
    )
    reg1 = Target125RegistrationInput(
        registration_phase="REG1",
        registered_classes=old + new,
        support_zid160=np.vstack((_rows(len(old)), _rows(len(new), offset=20))).astype(np.float32),
        support_labels=old + new,
        support_physical_ids=old_ids + tuple(f"new-support-{index}" for index in range(len(new))),
        query_zid160=_rows(len(old) + len(new), offset=80),
        query_physical_ids=tuple(f"reg1-query-{index}" for index in range(len(old) + len(new))),
    )
    source_row = {
        "outer_id": "d108-source-row",
        "receiver": outer.receiver,
        "seed": outer.seed,
        "k_shot": outer.k_shot,
        "active_k": outer.k_shot,
        "new_count": outer.new_count,
        "source_pool_k": outer.source_pool_k,
        "packages": {},
        "authority_bundle": {},
    }
    return Target125ConditionInput(
        outer_row=outer,
        scene=matrix.SCENES[0],
        source_row=source_row,
        reg0=reg0,
        reg1=reg1,
    )


def test_condition_preserves_reg0_old_support_order_and_builds_core_bindings() -> None:
    condition = _condition()
    token = hashlib.sha256(b"binding").hexdigest()
    source_plan = {
        "identity": {
            "d92_matrix_manifest": {"sha256": token},
            "checkpoint": {"sha256": token},
            "d92_sealed_runtime_sha256": token,
        },
        "plan_receipt_sha256": token,
    }
    reg0, reg1 = build_target125_runtime_bindings(source_plan=source_plan, condition=condition)
    assert reg0.support_physical_ids == condition.reg0.support_physical_ids
    assert reg1.support_physical_ids == condition.reg0.support_physical_ids + tuple(
        condition.reg1.support_physical_ids[matrix.OLD_CLASS_COUNT :]
    )
    assert reg0.query_physical_ids == condition.reg0.query_physical_ids
    assert reg1.query_physical_ids == condition.reg1.query_physical_ids


def test_query_isolation_and_reg1_old_support_reorder_fail_closed() -> None:
    condition = _condition()
    assert query_isolation_receipt()["query_truth_access"] is False
    assert query_isolation_receipt()["phase2_optimizer_steps"] == 0
    reordered = Target125RegistrationInput(
        registration_phase="REG1",
        registered_classes=condition.reg1.registered_classes,
        support_zid160=condition.reg1.support_zid160,
        support_labels=(condition.reg1.support_labels[1], condition.reg1.support_labels[0], *condition.reg1.support_labels[2:]),
        support_physical_ids=(condition.reg1.support_physical_ids[1], condition.reg1.support_physical_ids[0], *condition.reg1.support_physical_ids[2:]),
        query_zid160=condition.reg1.query_zid160,
        query_physical_ids=condition.reg1.query_physical_ids,
    )
    with pytest.raises(NextR5FATarget125RuntimeError, match="byte-preserve"):
        Target125ConditionInput(
            outer_row=condition.outer_row,
            scene=condition.scene,
            source_row=condition.source_row,
            reg0=condition.reg0,
            reg1=reordered,
        )


def test_fa_asset_method_lock_mismatch_fails_closed_on_reload(tmp_path) -> None:
    def sha(value: str) -> str:
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    basis = np.zeros((core.FA_RANK, core.Z_DIM), dtype=np.float32)
    basis[0, 0] = basis[1, 1] = basis[2, 2] = 1.0
    asset = core.build_target_fa_asset(
        old_classes=tuple(f"old-{index}" for index in range(core.OLD_CLASS_COUNT)),
        aggregate_samples_per_class=(98,) * core.OLD_CLASS_COUNT,
        class_centers_3d=np.zeros((core.OLD_CLASS_COUNT, core.FA_RANK), dtype=np.float32),
        fisher_precision_3d=np.ones(core.FA_RANK, dtype=np.float32),
        residual_variance_3d=np.ones(core.FA_RANK, dtype=np.float32),
        fisher_radius=np.asarray([np.sqrt(3.0)], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.2, 0.1, 0.05], dtype=np.float32),
        basis_3x160=basis,
        checkpoint_sha256=sha("checkpoint"),
        phase1_bundle_sha256=sha("phase1-bundle"),
        phase1_aggregate_receipt_sha256=sha("aggregate"),
        method_lock_sha256=sha("asset-method-lock"),
    )
    wire = core.serialize_target_fa_asset(asset)
    path = tmp_path / "fa.wire"
    path.write_bytes(wire)
    plan = {
        "identity": {
            "fa_asset": {"path": str(path), "sha256": hashlib.sha256(wire).hexdigest()},
            "checkpoint_sha256": sha("checkpoint"),
            "method_lock": {"sha256": sha("different-method-lock")},
        }
    }
    with pytest.raises(NextR5FATarget125RuntimeError, match="method-lock binding drift"):
        runtime._load_target_asset(plan)  # noqa: SLF001 - negative release-boundary test.
