from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_d106_rdce_asset as rdce_asset
import cvsrffi.stage2_d106_rdce_runtime as rdce
from cvsrffi import stage2_next_r3_rdce_tsl_runtime as runtime
from cvsrffi import stage2_next_r3_tsl160 as tsl
from cvsrffi.stage2_d106_phase1_tap import (
    D106Phase1TapRows,
    TAP_ARCHIVE_NAME,
    TAP_MEMBERS,
    TAP_RECEIPT_SCHEMA,
)
from cvsrffi.stage2_d106_rdce_asset import (
    CANDIDATE_ID,
    D104_SPLIT_ID,
    D106RDCEBuildLock,
    build_d106_rdce_asset,
)
from cvsrffi.stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_cluster(class_index: int, count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(0.02, 0.025, size=(count, 160)).astype(np.float32)
    value[:, class_index] += np.float32(4.0)
    value[:, 32 + class_index] += np.float32(0.7)
    return value


def _d106_source_tap() -> D106Phase1TapRows:
    rows: list[np.ndarray] = []
    tx: list[str] = []
    receiver: list[str] = []
    day: list[str] = []
    for receiver_index in range(7):
        for day_index in range(4):
            for class_index in range(6):
                count = 4 if (receiver_index + day_index + class_index) % 2 == 0 else 3
                for sample_index in range(count):
                    row = np.zeros(160, dtype=np.float32)
                    row[class_index] = 1.2
                    centred = float(sample_index) - float(count - 1) / 2.0
                    row[20] = 0.22 + 0.018 * (receiver_index - 3) + 0.004 * centred
                    row[21] = 0.19 + 0.016 * day_index + 0.003 * centred
                    row[22] = 0.17 + 0.008 * (receiver_index - 3) * (day_index * 2 - 3)
                    rows.append(row)
                    tx.append(f"tx_{class_index}")
                    receiver.append(f"rx_{receiver_index}")
                    day.append(f"day_{day_index}")
    pre_relu = np.stack(rows).astype(np.float32)
    arrays = {
        "pre_relu": pre_relu,
        "z_dom": np.zeros_like(pre_relu),
        "tx_labels": np.asarray(tx, dtype="<U16"),
        "receiver_ids": np.asarray(receiver, dtype="<U16"),
        "day_ids": np.asarray(day, dtype="<U16"),
        "physical_ids": np.asarray(
            [f"phase1-{index:04d}" for index in range(len(pre_relu))], dtype="<U20"
        ),
        "scenario_names": np.full(len(pre_relu), "leo_clear_weak", dtype="<U20"),
        "observation_ids": np.asarray(
            [f"obs-{index:04d}" for index in range(len(pre_relu))], dtype="<U20"
        ),
    }
    receipt = {
        "schema": TAP_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": D104_SPLIT_ID,
        "protocol_schema": "p2_min_v1",
        "selected_iq_archive_sha256": _sha("selected-iq"),
        "selected_iq_receipt_sha256": _sha("selected-receipt"),
        "storage_validator_receipt_sha256": _sha("storage-validator"),
        "storage_validation_binding": {
            "schema": rdce_asset.LS_IQ_VALIDATOR_SCHEMA,
            "storage_validation_root_sha256": _sha("storage-root"),
            "selected_content_root_sha256": _sha("selected-root"),
            "all_8400x3_storage_semantics_verified": True,
        },
        "extraction_binding": {
            "schema": rdce_asset.LS_IQ_RECEIPT_SCHEMA,
            "row_count": len(pre_relu),
            "selection_salt_sha256": _sha("selection-salt"),
            "selected_content_root_sha256": _sha("selected-root"),
            "input_ls_archive_sha256": _sha("input-ls"),
            "execution_root_sha256": _sha("execution-root"),
        },
        "input_ls_archive_sha256": _sha("input-ls"),
        "checkpoint_sha256": _sha("checkpoint"),
        "runtime_sha256": _sha("runtime"),
        "tap_archive_name": TAP_ARCHIVE_NAME,
        "tap_archive_sha256": _sha("placeholder-tap"),
        "tap_archive_members": list(TAP_MEMBERS),
        "array_sha256": {
            name: rdce_asset._tap_array_sha256(value) for name, value in arrays.items()
        },
        "row_count": len(pre_relu),
        "physical_id_root_sha256": rdce_asset._tap_ordered_id_root(arrays["physical_ids"]),
    }
    return D106Phase1TapRows(
        **arrays,
        z_id=np.maximum(pre_relu, np.float32(0.0)).astype(np.float32, copy=False),
        receipt=receipt,
    )


def _formal_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "tap.npz"
    receipt = tmp_path / "tap.receipt.json"
    archive.write_bytes(b"formal-tap-archive")
    receipt.write_bytes(b"formal-tap-receipt")
    archive_sha, receipt_sha = _file_sha(archive), _file_sha(receipt)
    tap = _d106_source_tap()
    loader_tap = replace(tap, receipt={**tap.receipt, "tap_archive_sha256": archive_sha})
    monkeypatch.setattr(rdce_asset, "load_d106_phase1_ls_tap", lambda *_a, **_k: loader_tap)
    return build_d106_rdce_asset(
        archive,
        receipt,
        expected_tap_archive_sha256=archive_sha,
        expected_tap_receipt_sha256=receipt_sha,
        build_lock=D106RDCEBuildLock(
            method_lock_sha256=_sha("method-lock"),
            construction_code_sha256=_sha("construction-code"),
        ),
    )


def _qknn_lock(active_k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=active_k,
        student_nu=4.0,
        kernel_effective_dim=16,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256=_sha("qknn-lodo"),
        quantization_margin_audit_sha256=_sha("qknn-margin"),
    )


def _write_row_authority(tmp_path: Path, support: rdce.D106RDCESupportRows):
    physical = tuple(str(value) for value in support.support_physical_ids.tolist())
    document = {
        "schema": rdce.ROW_AUTHORITY_SCHEMA,
        "capsule_id": support.split_handle.capsule_id,
        "split_id": support.split_handle.split_id,
        "validator_receipt_sha256": support.split_handle.validator_receipt_sha256,
        "row_id": support.row_id,
        "seed": support.seed,
        "active_k": support.qknn_bank.active_k,
        "registered_classes": list(support.qknn_bank.classes),
        "support_z_id_receipt": rdce._array_receipt(support.support_z_id),
        "support_labels_receipt": rdce._array_receipt(support.support_labels),
        "support_physical_ids_receipt": rdce._array_receipt(support.support_physical_ids),
        "ordered_support_physical_ids_sha256": rdce._ordered_physical_root(physical),
        "qknn_bank_sha256": support.qknn_bank.bank_receipt_sha256,
        "support_physical_root_sha256": support.split_handle.support_physical_root_sha256,
        "query_physical_root_sha256": support.split_handle.query_physical_root_sha256,
        "protocol_schema": support.split_handle.protocol_schema,
        "phase2_data_status": support.split_handle.phase2_data_status,
        "support_query_disjoint": True,
    }
    path = tmp_path / "r3-row-authority.json"
    path.write_bytes(rdce._canonical_bytes(document))
    return rdce.load_d106_rdce_row_authority(
        path, expected_authority_sha256=_file_sha(path)
    )


def _build_prior(asset) -> tuple[tsl.TSL160Phase1Prior, tsl.TSL160RuntimeBinding]:
    old_classes = tuple(f"c{index}" for index in range(5))

    def cell(receiver: str, label: str, class_index: int, seed: int) -> tsl.TSL160Phase1Cell:
        rows = _raw_cluster(class_index, 6, seed=seed)
        return tsl.TSL160Phase1Cell(
            receiver_id=receiver,
            class_handle=label,
            physical_ids=tuple(f"phase1-{receiver}-{label}-{index}" for index in range(6)),
            zid160=rows,
        )

    cells = [cell("outer-hold", "c0", 0, 10), cell("p1", "pseudo-new", 5, 11)]
    for receiver_index, receiver in enumerate(("p1", "p2")):
        cells.extend(
            cell(receiver, class_id, class_index, 100 + receiver_index * 10 + class_index)
            for class_index, class_id in enumerate(old_classes)
        )
    source = tuple(cells)
    fold_cells = tuple(
        next(item for item in source if item.receiver_id == "p1" and item.class_handle == class_id)
        for class_id in old_classes
    )
    fold = tsl.TSL160PhysicalLOOFold(
        fold_id="p1-c1-physical-loo",
        receiver_id="p1",
        class_handle="c1",
        registered_classes=old_classes,
        support_zid160=np.concatenate(tuple(cell.zid160[:5] for cell in fold_cells), axis=0),
        support_labels=tuple(class_id for class_id in old_classes for _ in range(5)),
        support_physical_ids=tuple(
            physical_id for cell in fold_cells for physical_id in cell.physical_ids[:5]
        ),
        validation_zid160=np.concatenate(tuple(cell.zid160[5:] for cell in fold_cells), axis=0),
        validation_labels=old_classes,
        validation_physical_ids=tuple(cell.physical_ids[5] for cell in fold_cells),
    )
    eligible = tuple(
        item
        for item in source
        if item.receiver_id != "outer-hold" and item.class_handle != "pseudo-new"
    )
    binding = tsl.TSL160RuntimeBinding(
        outer_fold_id="outer-hold/pseudo-new",
        checkpoint_sha256=asset.checkpoint_sha256,
        representation_rule_sha256=_sha("representation-rule"),
        phase1_physical_id_root_sha256=tsl.phase1_physical_id_root(eligible),
        phase1_seal_sha256=_sha("phase1-seal"),
    )
    built = tsl.build_tsl160_phase1_prior(
        source,
        (fold,),
        binding=binding,
        held_receiver="outer-hold",
        held_class="pseudo-new",
    )
    return built.prior, binding


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active_k: int):
    asset = _formal_asset(tmp_path, monkeypatch)
    old_classes = tuple(f"c{index}" for index in range(5))
    all_classes = old_classes + ("c5",)
    raw_old_support = np.concatenate(
        tuple(_raw_cluster(index, active_k, seed=300 + index) for index in range(5)), axis=0
    )
    canonical_old_support = tsl.canonical_d106_relu_zid160(raw_old_support)
    old_labels = tuple(class_id for class_id in old_classes for _ in range(active_k))
    old_ids = tuple(
        f"support-{class_id}-{shot}" for class_id in old_classes for shot in range(active_k)
    )
    lock = _qknn_lock(active_k)
    bank = build_typed_zid_support_bank(
        canonical_old_support, old_labels, old_classes, config=lock
    )
    split_handle = TypedValidatedOnceP2SplitHandle(
        capsule_id=_sha("capsule"),
        split_id=_sha("split"),
        validator_receipt_sha256=_sha("validator"),
        support_physical_root_sha256=rdce._physical_root(old_ids),
        query_physical_root_sha256=_sha("d106-query-root"),
        support_query_disjoint=True,
    )
    support = rdce.D106RDCESupportRows(
        support_z_id=canonical_old_support,
        support_labels=np.asarray(old_labels, dtype="<U8"),
        support_physical_ids=np.asarray(old_ids, dtype="<U32"),
        qknn_bank=bank,
        split_handle=split_handle,
        row_id="r3-local-row",
        seed=104713,
    )
    state = rdce.fit_d106_rdce_runtime(
        asset, support, row_authority=_write_row_authority(tmp_path, support)
    )
    prior, tsl_binding = _build_prior(asset)
    received_root = _sha("received-iq-root")
    raw_old_query = np.concatenate(
        tuple(_raw_cluster(index, 1, seed=400 + index) for index in range(5)), axis=0
    )
    old_query_ids = tuple(f"query-c{index}-0" for index in range(5))
    reg0 = runtime.NextR3RegistrationInput(
        registration_state="REG0",
        received_iq_root_sha256=received_root,
        support_pre_relu160=raw_old_support,
        query_pre_relu160=raw_old_query,
        support_labels=old_labels,
        registered_classes=old_classes,
        support_physical_ids=old_ids,
        query_physical_ids=old_query_ids,
    )
    raw_new_support = _raw_cluster(5, active_k, seed=305)
    raw_new_query = _raw_cluster(5, 1, seed=405)
    reg1 = runtime.NextR3RegistrationInput(
        registration_state="REG1",
        received_iq_root_sha256=received_root,
        support_pre_relu160=np.concatenate((raw_old_support, raw_new_support), axis=0),
        query_pre_relu160=np.concatenate((raw_old_query, raw_new_query), axis=0),
        support_labels=old_labels + ("c5",) * active_k,
        registered_classes=all_classes,
        support_physical_ids=old_ids + tuple(f"support-c5-{shot}" for shot in range(active_k)),
        query_physical_ids=old_query_ids + ("query-c5-0",),
    )
    bridge = runtime.NextR3RDCEBridgeBinding(
        checkpoint_sha256=asset.checkpoint_sha256,
        capsule_id=state.capsule_id,
        split_id=state.split_id,
        row_id=state.row_id,
        seed=state.seed,
        received_iq_root_sha256=received_root,
        tap_sha256=asset.tap_sha256,
        representation_rule_sha256=tsl_binding.representation_rule_sha256,
        phase1_physical_id_root_sha256=tsl_binding.phase1_physical_id_root_sha256,
        phase1_seal_sha256=tsl_binding.phase1_seal_sha256,
        outer_fold_id=tsl_binding.outer_fold_id,
    )
    return bridge, state, reg0, reg1, lock, prior, tsl_binding


def test_formal_rdce_tsl_four_state_closure_and_fail_closed_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge, state, reg0, reg1, lock, prior, tsl_binding = _case(
        tmp_path, monkeypatch, 5
    )
    result = runtime.execute_next_r3_four_state(
        bridge=bridge,
        da1_reg0_state=state,
        reg0=reg0,
        reg1=reg1,
        qknn_lock=lock,
        tsl_prior=prior,
        tsl_runtime_binding=tsl_binding,
    )

    assert tuple(result.four_state) == runtime.FOUR_STATE_IDS
    assert result.da1_reg0_state_sha256 == state.runtime_receipt_sha256
    assert result.da1_reg1_state_sha256 == state.runtime_receipt_sha256
    assert result.reg0.receipt["metric_availability"]["seen_new_acc"] == "N/A"
    assert result.reg0.receipt["metric_availability"]["H_old_new"] == "N/A"
    assert result.reg0.receipt["head_fit_calls"]["R0F"] == 0
    assert result.reg1.receipt["old_class_count"] == 5
    assert result.runtime_receipt["r1_tsl_prior_semantics"] == tsl.PRIOR_SEMANTICS
    assert result.runtime_receipt["r1_tsl_prior_transported_by_rdce"] is False
    assert result.runtime_receipt["r1_tsl_covariance_claim"] is False
    for registration in (result.reg0, result.reg1):
        for representation in ("R0", "R1"):
            cache = registration.caches[representation]
            l_arm = registration.arms[f"{representation}L"]
            l_receipt = l_arm.receipt["head_receipt"]
            assert l_arm.cache is cache
            assert l_receipt["representation_mode"] == (
                tsl.CANONICAL_R0 if representation == "R0" else tsl.RDCE_R1_SIGNED_UNIT
            )
            assert l_receipt["tsl_support_cache_sha256"] == tsl.tsl160_cache_sha256(
                cache.support_zid160
            )
            assert l_receipt["tsl_query_cache_sha256"] == tsl.tsl160_cache_sha256(
                cache.query_zid160
            )
            assert l_receipt["prior_semantics"] == tsl.PRIOR_SEMANTICS
            assert l_receipt["prior_transported_by_rdce"] is False
            assert l_receipt["r1_covariance_claim"] is False
            assert registration.arms[f"{representation}Q"].cache is cache
            assert registration.arms[f"{representation}F"].cache is cache
    assert result.reg1.head_fits["R0F"] is not None
    assert result.reg1.head_fits["R1F"] is not None
    np.testing.assert_array_equal(
        result.reg1.caches["R1"].support_zid160[: len(reg0.support_labels)],
        result.reg0.caches["R1"].support_zid160,
    )

    with pytest.raises(runtime.NextR3RDCETSLRuntimeError, match="bridge drift"):
        runtime.execute_next_r3_four_state(
            bridge=replace(bridge, checkpoint_sha256=_sha("wrong-checkpoint")),
            da1_reg0_state=state,
            reg0=reg0,
            reg1=reg1,
            qknn_lock=lock,
            tsl_prior=prior,
            tsl_runtime_binding=tsl_binding,
        )
    changed_support = reg1.support_pre_relu160.copy()
    changed_support[0, 7] += np.float32(0.25)
    with pytest.raises(runtime.NextR3RDCETSLRuntimeError, match="byte-preserve"):
        runtime.execute_next_r3_four_state(
            bridge=bridge,
            da1_reg0_state=state,
            reg0=reg0,
            reg1=replace(reg1, support_pre_relu160=changed_support),
            qknn_lock=lock,
            tsl_prior=prior,
            tsl_runtime_binding=tsl_binding,
        )
    reverse = np.arange(len(reg1.query_physical_ids) - 1, -1, -1)
    reordered = replace(
        reg1,
        query_pre_relu160=reg1.query_pre_relu160[reverse].copy(),
        query_physical_ids=tuple(reg1.query_physical_ids[index] for index in reverse),
    )
    reordered_result = runtime.execute_next_r3_four_state(
        bridge=bridge,
        da1_reg0_state=state,
        reg0=reg0,
        reg1=reordered,
        qknn_lock=lock,
        tsl_prior=prior,
        tsl_runtime_binding=tsl_binding,
    )
    assert reordered_result.da1_reg0_state_sha256 == result.da1_reg0_state_sha256
    assert reordered_result.da1_reg1_state_sha256 == result.da1_reg1_state_sha256
    for arm_id in ("R0F", "R0L", "R1F", "R1L"):
        assert (
            reordered_result.reg1.head_fits[arm_id].state.state_sha256
            == result.reg1.head_fits[arm_id].state.state_sha256
        )
    tied = np.zeros(
        (len(result.reg0.caches["R0"].query_physical_ids), len(reg0.registered_classes)),
        dtype=np.float32,
    )
    with pytest.raises(runtime.NextR3RDCETSLRuntimeError, match="top-tie"):
        runtime._strict_arm(
            cache=result.reg0.caches["R0"], head="Q", logits=tied, head_receipt={}
        )
