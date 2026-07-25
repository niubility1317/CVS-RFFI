from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from cvsrffi.rxid_metabias4_phase1_trainer import (
    CANDIDATE_ID,
    D103R1Config,
    D103R1Phase1Trainer,
    D103R1TeacherState,
    D103R1TrainingError,
    K_VALUES,
    Operation,
    OuterMaskSpec,
    PermissionLedger,
    SplitRole,
    build_outer_masks,
    build_training_data,
    build_tx_projector,
    export_teacher_arrays,
)


def _synthetic_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(103713)
    receivers = ("r0", "r1", "r2")
    days = ("d0", "d1", "d2")
    classes = tuple(f"tx{i}" for i in range(6))
    rows_per_cell = 9

    ls_receiver: list[str] = []
    ls_day: list[str] = []
    ls_tx: list[str] = []
    ls_physical: list[str] = []
    ls_z: list[np.ndarray] = []
    ls_pre: list[np.ndarray] = []
    tx_directions = rng.standard_normal((len(classes), 160)).astype(np.float32)
    receiver_directions = rng.standard_normal((len(receivers), 160)).astype(np.float32)
    day_directions = rng.standard_normal((len(days), 160)).astype(np.float32)
    for receiver_index, receiver in enumerate(receivers):
        for day_index, day in enumerate(days):
            for class_index, label in enumerate(classes):
                for repeat in range(rows_per_cell):
                    ls_receiver.append(receiver)
                    ls_day.append(day)
                    ls_tx.append(label)
                    ls_physical.append(
                        f"ls-{receiver}-{day}-{label}-{repeat:02d}"
                    )
                    noise = rng.standard_normal(160).astype(np.float32) * 0.03
                    ls_z.append(
                        tx_directions[class_index]
                        + 0.5 * receiver_directions[receiver_index]
                        + 0.2 * day_directions[day_index]
                        + noise
                    )
                    ls_pre.append(
                        rng.standard_normal(160).astype(np.float32)
                        + 0.05 * tx_directions[class_index]
                    )

    us_receiver: list[str] = []
    us_day: list[str] = []
    us_physical: list[str] = []
    us_z: list[np.ndarray] = []
    for receiver_index, receiver in enumerate(receivers):
        for day_index, day in enumerate(days):
            for repeat in range(4):
                us_receiver.append(receiver)
                us_day.append(day)
                us_physical.append(f"us-{receiver}-{day}-{repeat:02d}")
                us_z.append(
                    0.5 * receiver_directions[receiver_index]
                    + 0.2 * day_directions[day_index]
                    + rng.standard_normal(160).astype(np.float32) * 0.03
                )

    labeled = {
        "z_dom": np.asarray(ls_z, dtype=np.float32),
        "pre_relu": np.asarray(ls_pre, dtype=np.float32),
        "receiver_ids": np.asarray(ls_receiver),
        "day_ids": np.asarray(ls_day),
        "tx_labels": np.asarray(ls_tx),
        "physical_ids": np.asarray(ls_physical),
    }
    unlabeled = {
        "z_dom": np.asarray(us_z, dtype=np.float32),
        "receiver_ids": np.asarray(us_receiver),
        "day_ids": np.asarray(us_day),
        "physical_ids": np.asarray(us_physical),
    }
    source_val = {"row_count": 300, "content_sha256": "a" * 64}
    return labeled, unlabeled, source_val


def _data():
    labeled, unlabeled, source_val = _synthetic_payloads()
    return build_training_data(labeled, unlabeled, source_val)


def test_singleton_configuration_rejects_any_selection_or_constant_drift() -> None:
    config = D103R1Config()
    assert config.total_meta_steps == 400
    assert config.k_values == K_VALUES
    assert config.performance_selection is False
    assert config.early_stopping is False
    assert config.outer_results_read is False

    with pytest.raises(D103R1TrainingError, match="singleton configuration drift"):
        D103R1Config(learning_rate=2.0e-3)
    with pytest.raises(D103R1TrainingError, match="singleton configuration drift"):
        D103R1Config(k_values=(1,))
    with pytest.raises(D103R1TrainingError, match="singleton configuration drift"):
        D103R1Config(performance_selection=True)


def test_split_member_allowlists_fail_closed_for_tx_and_source_val_features() -> None:
    labeled, unlabeled, source_val = _synthetic_payloads()

    bad_unlabeled = dict(unlabeled)
    bad_unlabeled["tx_labels"] = np.asarray(["hidden"] * len(unlabeled["z_dom"]))
    with pytest.raises(D103R1TrainingError, match="U_s member closure drift"):
        build_training_data(labeled, bad_unlabeled, source_val)

    bad_unlabeled = dict(unlabeled)
    bad_unlabeled["pre_relu"] = unlabeled["z_dom"]
    with pytest.raises(D103R1TrainingError, match="U_s member closure drift"):
        build_training_data(labeled, bad_unlabeled, source_val)

    bad_source_val = dict(source_val)
    bad_source_val["z_dom"] = np.zeros((1, 160), dtype=np.float32)
    with pytest.raises(D103R1TrainingError, match="source-val member closure drift"):
        build_training_data(labeled, unlabeled, bad_source_val)

    missing_tx = dict(labeled)
    missing_tx.pop("tx_labels")
    with pytest.raises(D103R1TrainingError, match="L_s member closure drift"):
        build_training_data(missing_tx, unlabeled, source_val)


def test_ls_us_physical_overlap_is_rejected() -> None:
    labeled, unlabeled, source_val = _synthetic_payloads()
    bad_unlabeled = dict(unlabeled)
    physical = np.asarray(unlabeled["physical_ids"]).astype("<U64")
    physical[0] = np.asarray(labeled["physical_ids"])[0]
    bad_unlabeled["physical_ids"] = physical
    with pytest.raises(D103R1TrainingError, match="physical IDs overlap"):
        build_training_data(labeled, bad_unlabeled, source_val)


def test_outer_day_class_masks_never_join_hidden_tx_into_us() -> None:
    data = _data()
    ledger = PermissionLedger()
    spec = OuterMaskSpec(
        held_receiver="r2", held_day="d2", held_class="tx5"
    )
    masks = build_outer_masks(data, spec, ledger)
    assert not np.any(data.labeled.receiver_ids[masks.labeled_train] == "r2")
    assert not np.any(data.labeled.day_ids[masks.labeled_train] == "d2")
    assert not np.any(data.labeled.tx_labels[masks.labeled_train] == "tx5")
    assert not np.any(masks.unlabeled_train)
    assert masks.hidden_tx_unlabeled_policy == (
        "exclude_all_U_s_in_class_LOCO_no_TX_join"
    )
    assert all(event.role != SplitRole.SOURCE_VALIDATION.value for event in ledger.events)


def test_permission_ledger_denies_tx_meta_bank_and_any_source_val_training() -> None:
    ledger = PermissionLedger()
    forbidden = (
        Operation.TX_PROJECTOR,
        Operation.TX_MMD,
        Operation.CLASS_BALANCED_BANK,
        Operation.METABIAS_META,
    )
    for operation in forbidden:
        with pytest.raises(D103R1TrainingError, match="permission denied"):
            ledger.authorize(
                SplitRole.UNLABELED_SOURCE,
                operation,
                ("z_dom",),
                1,
            )
    with pytest.raises(D103R1TrainingError, match="permission denied"):
        ledger.authorize(
            SplitRole.SOURCE_VALIDATION,
            Operation.VICREG,
            ("z_dom",),
            1,
        )
    assert ledger.denied_attempts == 5
    receipt = ledger.receipt()
    assert receipt["source_val_array_access"] is False
    assert receipt["performance_selection_access"] is False


def test_tx_projector_uses_only_ls_and_annihilates_rank_five_basis() -> None:
    data = _data()
    ledger = PermissionLedger()
    masks = build_outer_masks(data, OuterMaskSpec(), ledger)
    projector, receipt = build_tx_projector(data, masks, ledger)
    assert projector.shape == (160, 160)
    assert receipt["tx_null_rank"] == 5
    assert receipt["unlabeled_rows_used"] == 0
    assert receipt["source_val_rows_used"] == 0
    assert receipt["null_residual"] <= 1.0e-8
    np.testing.assert_allclose(projector, projector.T, atol=2.0e-6)
    np.testing.assert_allclose(projector @ projector, projector, atol=2.0e-5)


def test_class_loco_projector_has_rank_four_and_uses_no_us() -> None:
    data = _data()
    ledger = PermissionLedger()
    masks = build_outer_masks(
        data, OuterMaskSpec(held_receiver="r2", held_class="tx5"), ledger
    )
    _, receipt = build_tx_projector(data, masks, ledger)
    assert receipt["tx_class_count"] == 5
    assert receipt["tx_null_rank"] == 4
    assert not np.any(masks.unlabeled_train)


def test_mechanical_train_step_runs_all_k_and_preserves_role_permissions() -> None:
    torch.set_num_threads(1)
    trainer = D103R1Phase1Trainer(_data(), device="cpu")
    before = trainer.model.basis.detach().clone()
    receipt = trainer.step()
    after = trainer.model.basis.detach().clone()

    assert receipt.candidate_id == CANDIDATE_ID
    assert receipt.step_index == 0
    assert receipt.k_values == (1, 5, 10)
    assert receipt.episode_support_receiver != receipt.episode_query_receiver
    assert receipt.optimizer_step_completed is True
    assert receipt.performance_metrics_computed is False
    assert receipt.source_val_rows_used == 0
    assert receipt.target_access is False
    assert receipt.formal_query_access is False
    assert math.isfinite(receipt.total_loss)
    assert math.isfinite(receipt.meta_loss)
    assert not torch.equal(before, after)
    assert trainer.completed_steps == 1

    us_operations = {
        event.operation
        for event in trainer.ledger.events
        if event.role == SplitRole.UNLABELED_SOURCE.value
    }
    assert us_operations == {
        Operation.FOLD_MASK.value,
        Operation.RX_SELF_SUPERVISION.value,
        Operation.VICREG.value,
    }
    assert all(
        event.role != SplitRole.SOURCE_VALIDATION.value
        for event in trainer.ledger.events
    )
    ledger_receipt = trainer.ledger.receipt()
    assert ledger_receipt["denied_attempts"] == 0
    assert receipt.ledger_receipt_sha256 == ledger_receipt["receipt_sha256"]


def test_receiver_self_supervision_fails_when_cross_day_pairs_are_removed() -> None:
    labeled, unlabeled, source_val = _synthetic_payloads()
    labeled_day = np.asarray(labeled["day_ids"])
    unlabeled_day = np.asarray(unlabeled["day_ids"])
    labeled_keep = labeled_day == "d0"
    unlabeled_keep = unlabeled_day == "d0"
    one_day_labeled = {
        key: np.asarray(value)[labeled_keep] for key, value in labeled.items()
    }
    one_day_unlabeled = {
        key: np.asarray(value)[unlabeled_keep] for key, value in unlabeled.items()
    }
    data = build_training_data(one_day_labeled, one_day_unlabeled, source_val)
    trainer = D103R1Phase1Trainer(data, device="cpu")
    with pytest.raises(D103R1TrainingError, match="cross-day positives"):
        trainer.step()


def test_final_state_and_export_refuse_incomplete_fit() -> None:
    trainer = D103R1Phase1Trainer(_data(), device="cpu")
    with pytest.raises(D103R1TrainingError, match="exactly 400"):
        trainer.final_state()
    with pytest.raises(D103R1TrainingError, match="exactly 400"):
        trainer.export_teacher_arrays()


def test_final_aggregation_requires_two_physical_samples_per_class_cell() -> None:
    labeled, unlabeled, source_val = _synthetic_payloads()
    receiver = np.asarray(labeled["receiver_ids"])
    day = np.asarray(labeled["day_ids"])
    label = np.asarray(labeled["tx_labels"])
    target = (receiver == "r0") & (day == "d0") & (label == "tx0")
    keep = ~target
    keep[np.flatnonzero(target)[0]] = True
    sparse_labeled = {
        key: np.asarray(value)[keep] for key, value in labeled.items()
    }
    data = build_training_data(sparse_labeled, unlabeled, source_val)
    with pytest.raises(D103R1TrainingError, match="lacks frozen sample count"):
        D103R1Phase1Trainer(data, device="cpu")


def test_completed_teacher_export_has_only_arrays_and_anonymous_receipts() -> None:
    trainer = D103R1Phase1Trainer(_data(), device="cpu")
    # The 400-step numeric schedule is exercised by the runner.  This focused
    # unit test places the trainer at the already-completed boundary solely to
    # test aggregation and payload closure without repeating 400 CPU steps.
    trainer.completed_steps = trainer.config.total_meta_steps
    state = trainer.final_state()
    assert type(state) is D103R1TeacherState
    exported = trainer.export_teacher_arrays()
    assert set(export_teacher_arrays(state)) == set(exported)
    assert set(exported) == {
        "U",
        "B",
        "bank_g",
        "bank_t",
        "bank_precision",
        "bank_sigma",
        "aggregation_receipt",
        "access_receipt",
    }
    assert exported["U"].shape == (32, 160)
    assert exported["B"].shape == (160, 4)
    assert exported["bank_g"].shape[1] == 32
    assert exported["bank_t"].shape == (exported["bank_g"].shape[0], 4)
    assert exported["bank_precision"].shape == exported["bank_t"].shape
    assert exported["bank_sigma"].shape == (exported["bank_g"].shape[0],)
    for name in ("U", "B", "bank_g", "bank_t", "bank_precision", "bank_sigma"):
        assert exported[name].flags.writeable is False

    aggregation = exported["aggregation_receipt"]
    assert aggregation["completed_meta_steps"] == 400
    assert aggregation["minimum_physical_samples_per_class_cell"] == 9
    assert aggregation["unlabeled_rows_used"] == 0
    assert aggregation["source_val_rows_used"] == 0
    assert aggregation["contains_receiver_values"] is False
    assert aggregation["contains_day_values"] is False
    assert aggregation["contains_class_values"] is False
    assert aggregation["contains_physical_ids"] is False
    assert aggregation["contains_optimizer"] is False

    receipts = json.loads(
        json.dumps(
            {
                "aggregation_receipt": dict(exported["aggregation_receipt"]),
                "access_receipt": dict(exported["access_receipt"]),
            },
            sort_keys=True,
        )
    )

    def string_values(value: object) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, dict):
            result: set[str] = set()
            for child in value.values():
                result.update(string_values(child))
            return result
        if isinstance(value, list):
            result = set()
            for child in value:
                result.update(string_values(child))
            return result
        return set()

    labeled, _, _ = _synthetic_payloads()
    forbidden_values = set(np.asarray(labeled["receiver_ids"]).tolist())
    forbidden_values.update(np.asarray(labeled["day_ids"]).tolist())
    forbidden_values.update(np.asarray(labeled["tx_labels"]).tolist())
    forbidden_values.update(np.asarray(labeled["physical_ids"]).tolist())
    assert forbidden_values.isdisjoint(string_values(receipts))
    assert "optimizer_state" not in json.dumps(receipts, sort_keys=True)
