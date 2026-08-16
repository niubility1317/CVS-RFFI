"""Behavioral checks for role-safe Phase1 MIRAGE source data views."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import re

import pytest

from cvsrffi.phase1_mirage.protocol import SourcePartition


def _data_api():
    """Import the Task 2 public API so RED proves the missing data boundary."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.data")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.data":
            pytest.fail("missing cvs rffi phase1 mirage data boundary")
        raise
    return (
        module.LabeledView,
        module.SourceInventoryRow,
        module.SourceProtocolError,
        module.UnlabeledView,
        module.ValidationView,
        module.build_source_split,
        module.materialize_labeled,
        module.materialize_unlabeled,
        module.materialize_validation,
    )


def _rows(
    *,
    count: int,
    receiver_id: str = "source-rx-1",
    tx_label: int = 7,
    day_id: str = "day-1",
    sample_prefix: str = "sample",
):
    _, SourceInventoryRow, _, _, _, _, _, _, _ = _data_api()
    return tuple(
        SourceInventoryRow(
            physical_sample_id=f"{sample_prefix}-{index:03d}",
            tx_label=tx_label,
            receiver_id=receiver_id,
            day_id=day_id,
            iq_index=index,
        )
        for index in range(count)
    )


def _rows_100_per_group():
    return _rows(count=100)


def test_split_uses_approved_policy_per_group_and_preserves_physical_id_disjointness():
    rows_100_per_group = _rows_100_per_group()
    _, _, _, _, _, build_source_split, _, _, _ = _data_api()

    split = build_source_split(rows_100_per_group, seed=817001)

    assert tuple(map(len, (split.l_ids, split.u_ids, split.v_cal_ids, split.v_select_ids))) == (7, 63, 15, 15)
    partition_ids = (set(split.l_ids), set(split.u_ids), set(split.v_cal_ids), set(split.v_select_ids))
    assert not any(left & right for index, left in enumerate(partition_ids) for right in partition_ids[index + 1 :])
    assert split.group_counts == {(7, "source-rx-1", "day-1"): (7, 63, 15, 15)}
    assert split.receiver_registry == ("source-rx-1",)
    assert split.tx_registry == (7,)
    assert set(split.id_sha256) == set(SourcePartition)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in split.id_sha256.values())
    assert split.split_schema == "phase1_mirage_source_7_63_15_15_v1"


def test_split_is_deterministic_for_a_seed_and_changes_for_a_different_seed():
    rows_100_per_group = _rows_100_per_group()
    _, _, _, _, _, build_source_split, _, _, _ = _data_api()

    first = build_source_split(rows_100_per_group, seed=817001)
    repeated = build_source_split(rows_100_per_group, seed=817001)
    changed = build_source_split(rows_100_per_group, seed=817002)

    assert repeated == first
    assert (changed.l_ids, changed.u_ids, changed.v_cal_ids, changed.v_select_ids) != (
        first.l_ids,
        first.u_ids,
        first.v_cal_ids,
        first.v_select_ids,
    )


def test_split_partitions_each_tx_receiver_day_group_without_forbidding_shared_tx_identity():
    first_group = _rows(count=100, receiver_id="source-rx-a", tx_label=7, day_id="day-1")
    second_group = _rows(
        count=100,
        receiver_id="source-rx-b",
        tx_label=7,
        day_id="day-2",
        sample_prefix="second",
    )
    _, _, _, _, _, build_source_split, _, _, _ = _data_api()

    split = build_source_split(first_group + second_group, seed=817001)

    assert tuple(map(len, (split.l_ids, split.u_ids, split.v_cal_ids, split.v_select_ids))) == (14, 126, 30, 30)
    assert split.group_counts == {
        (7, "source-rx-a", "day-1"): (7, 63, 15, 15),
        (7, "source-rx-b", "day-2"): (7, 63, 15, 15),
    }
    assert split.tx_registry == (7,)


def test_target_receiver_is_rejected_before_duplicate_id_validation_or_split():
    rows_100_per_group = _rows_100_per_group()
    _, SourceInventoryRow, SourceProtocolError, _, _, build_source_split, _, _, _ = _data_api()
    target_with_reused_id = rows_100_per_group + (
        SourceInventoryRow(
            physical_sample_id=rows_100_per_group[0].physical_sample_id,
            tx_label=99,
            receiver_id="20-1",
            day_id="target-day",
            iq_index=0,
        ),
    )

    with pytest.raises(SourceProtocolError, match="target receiver"):
        build_source_split(target_with_reused_id, seed=817001, forbidden_receivers={"20-1"})


def test_duplicate_physical_sample_id_is_rejected():
    rows_100_per_group = _rows_100_per_group()
    _, SourceInventoryRow, SourceProtocolError, _, _, build_source_split, _, _, _ = _data_api()
    duplicate_id_rows = rows_100_per_group + (
        SourceInventoryRow(
            physical_sample_id=rows_100_per_group[0].physical_sample_id,
            tx_label=7,
            receiver_id="source-rx-1",
            day_id="day-2",
            iq_index=100,
        ),
    )

    with pytest.raises(SourceProtocolError, match="duplicate physical_sample_id"):
        build_source_split(duplicate_id_rows, seed=817001)


def test_unlabeled_materialization_has_no_tx_label_in_structure_or_attributes():
    rows_100_per_group = _rows_100_per_group()
    _, _, _, UnlabeledView, _, build_source_split, _, materialize_unlabeled, _ = _data_api()
    split = build_source_split(rows_100_per_group, seed=817001)

    views = materialize_unlabeled(rows_100_per_group, split.u_ids[:2])

    assert isinstance(views, tuple)
    assert tuple(view.physical_sample_id for view in views) == split.u_ids[:2]
    assert {field.name for field in dataclasses.fields(UnlabeledView)} == {
        "physical_sample_id",
        "receiver_id",
        "day_id",
        "iq_index",
    }
    assert all(not hasattr(view, "tx_label") for view in views)
    assert all("tx_label" not in dataclasses.asdict(view) for view in views)
    with pytest.raises(dataclasses.FrozenInstanceError):
        views[0].receiver_id = "mutated"


def test_labeled_and_validation_views_expose_only_their_approved_fields():
    rows_100_per_group = _rows_100_per_group()
    LabeledView, _, SourceProtocolError, _, ValidationView, build_source_split, materialize_labeled, _, materialize_validation = _data_api()
    split = build_source_split(rows_100_per_group, seed=817001)

    labeled = materialize_labeled(rows_100_per_group, split.l_ids)
    validation = materialize_validation(rows_100_per_group, split.v_cal_ids, split_role="val_cal")

    assert isinstance(labeled, tuple)
    assert all(isinstance(view, LabeledView) and isinstance(view.tx_label, int) for view in labeled)
    assert {field.name for field in dataclasses.fields(LabeledView)} == {
        "physical_sample_id",
        "tx_label",
        "receiver_id",
        "day_id",
        "iq_index",
    }
    assert all(isinstance(view, ValidationView) and view.split_role == "val_cal" for view in validation)
    assert {field.name for field in dataclasses.fields(ValidationView)} == {
        "physical_sample_id",
        "tx_label",
        "receiver_id",
        "day_id",
        "iq_index",
        "split_role",
    }
    with pytest.raises(SourceProtocolError, match="split_role"):
        materialize_validation(rows_100_per_group, split.v_select_ids, split_role="not-a-validation-role")


def test_manifest_receipt_hashes_match_each_returned_id_list():
    rows_100_per_group = _rows_100_per_group()
    _, _, _, _, _, build_source_split, _, _, _ = _data_api()
    split = build_source_split(rows_100_per_group, seed=817001)

    for partition, identifiers in (
        (SourcePartition.L_S, split.l_ids),
        (SourcePartition.U_S, split.u_ids),
        (SourcePartition.V_CAL, split.v_cal_ids),
        (SourcePartition.V_SELECT, split.v_select_ids),
    ):
        expected = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
        assert split.id_sha256[partition] == expected
