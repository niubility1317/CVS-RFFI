from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from cvsrffi.stage2_d110_sourceheld_split import (
    CELL_COUNT,
    D110SourceHeldSplitError,
    EXPECTED_HELD_ROWS,
    HELD_PER_CELL,
    partition_d110_sourceheld_rows,
    publish_d110_sourceheld_split,
    validate_source_feature_pool,
)


def _metadata():
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    historical: list[str] = []
    old_held: list[str] = []
    d110_ls: list[str] = []
    cell = 0
    for receiver_index in range(7):
        receiver = f"rx{receiver_index}"
        for tx_index in range(6):
            tx = f"tx{tx_index}"
            for day_index in range(4):
                day = f"d{day_index}"
                local = [
                    f"{receiver}|{tx}|{day}|{sample:02d}" for sample in range(50)
                ]
                labels.extend([tx] * 50)
                receivers.extend([receiver] * 50)
                days.extend([day] * 50)
                physical.extend(local)
                old_held.extend(local[:15])
                historical.extend(local[15:29])
                if cell < 126:
                    historical.append(local[29])
                d110_ls.extend(local[30:33])
                if cell < 84:
                    d110_ls.append(local[33])
                cell += 1
    assert len(historical) == 2478
    assert len(old_held) == 2520
    assert len(d110_ls) == 588
    return tuple(
        np.asarray(value)
        for value in (labels, receivers, days, physical)
    ) + (tuple(historical), tuple(old_held), tuple(d110_ls))


def test_d110_split_is_deterministic_balanced_and_fully_excluded() -> None:
    labels, receivers, days, physical, historical, old_held, d110_ls = _metadata()
    first, receipt = partition_d110_sourceheld_rows(
        labels,
        receivers,
        days,
        physical,
        historical_query_ids=historical,
        d104_held_ids=old_held,
        d110_ls_ids=d110_ls,
    )
    second, second_receipt = partition_d110_sourceheld_rows(
        labels,
        receivers,
        days,
        physical,
        historical_query_ids=tuple(reversed(historical)),
        d104_held_ids=tuple(reversed(old_held)),
        d110_ls_ids=tuple(reversed(d110_ls)),
    )
    assert np.array_equal(first, second)
    assert receipt == second_receipt
    assert len(first) == EXPECTED_HELD_ROWS == 1176
    assert receipt["cell_count"] == CELL_COUNT == 168
    assert receipt["held_per_cell"] == HELD_PER_CELL == 7
    assert receipt["receiver_tx_group_count"] == 42
    assert receipt["held_per_receiver_tx"] == 28
    assert receipt["truth_values_persisted_in_selection_receipt"] is False
    assert receipt["performance_computed"] is False
    selected = set(physical[first].tolist())
    assert not selected.intersection(historical)
    assert not selected.intersection(old_held)
    assert not selected.intersection(d110_ls)


def test_d110_split_is_row_order_equivariant() -> None:
    labels, receivers, days, physical, historical, old_held, d110_ls = _metadata()
    base, _ = partition_d110_sourceheld_rows(
        labels,
        receivers,
        days,
        physical,
        historical_query_ids=historical,
        d104_held_ids=old_held,
        d110_ls_ids=d110_ls,
    )
    permutation = np.random.default_rng(110813).permutation(len(physical))
    changed, _ = partition_d110_sourceheld_rows(
        labels[permutation],
        receivers[permutation],
        days[permutation],
        physical[permutation],
        historical_query_ids=historical,
        d104_held_ids=old_held,
        d110_ls_ids=d110_ls,
    )
    assert set(physical[base].tolist()) == set(physical[permutation][changed].tolist())


def test_d110_split_fails_when_one_cell_has_less_than_seven_rows() -> None:
    labels, receivers, days, physical, historical, old_held, d110_ls = _metadata()
    historical = list(historical)
    excluded = set(historical) | set(old_held) | set(d110_ls)
    replacements = [
        value
        for value in physical.tolist()
        if value.startswith("rx0|tx0|d0|") and value not in excluded
    ][:10]
    assert len(replacements) == 10
    for index, replacement in enumerate(replacements):
        historical[-(index + 1)] = replacement
    with pytest.raises(D110SourceHeldSplitError, match="lacks seven"):
        partition_d110_sourceheld_rows(
            labels,
            receivers,
            days,
            physical,
            historical_query_ids=historical,
            d104_held_ids=old_held,
            d110_ls_ids=d110_ls,
        )


def test_d110_source_feature_pool_requires_exact_d103_d105_parity() -> None:
    labels, receivers, days, physical, *_ = _metadata()
    z_id = np.zeros((8400, 160), dtype=np.float32)
    z_dom = np.ones((8400, 160), dtype=np.float32)
    pre_relu = np.zeros((8400, 160), dtype=np.float32)
    dual = {
        "z_id": z_id,
        "z_dom": z_dom,
        "tx_logits": np.zeros((8400, 6), dtype=np.float32),
        "labels": labels,
        "receiver_ids": receivers,
        "day_ids": days,
        "physical_ids": physical,
        "scenario_names": np.asarray(["leo_weak"] * 8400),
        "class_ids": np.asarray([f"tx{index}" for index in range(6)]),
        "observation_ids": np.asarray([f"obs{index}" for index in range(8400)]),
    }
    strict = {
        "pre_relu": pre_relu,
        "z_dom": z_dom.copy(),
        "labels": labels.copy(),
        "receiver_ids": receivers.copy(),
        "physical_ids": physical.copy(),
    }
    pool, receipt = validate_source_feature_pool(dual, strict)
    assert tuple(pool) == (
        "z_id",
        "z_dom",
        "pre_relu",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
        "class_ids",
    )
    assert receipt["relu_pre_relu_equals_z_id_exact"] is True
    assert receipt["z_dom_equal_exact"] is True
    changed = copy.copy(strict)
    changed["pre_relu"] = strict["pre_relu"].copy()
    changed["pre_relu"][0, 0] = np.float32(1.0)
    with pytest.raises(D110SourceHeldSplitError, match="pre_relu/D103 z_id"):
        validate_source_feature_pool(dual, changed)


def test_d110_publish_is_scorer_compatible_truth_free_and_immutable(tmp_path) -> None:
    labels, receivers, days, physical, historical, old_held, d110_ls = _metadata()
    z_id = np.zeros((8400, 160), dtype=np.float32)
    z_dom = np.ones((8400, 160), dtype=np.float32)
    dual = {
        "z_id": z_id,
        "z_dom": z_dom,
        "tx_logits": np.zeros((8400, 6), dtype=np.float32),
        "labels": labels,
        "receiver_ids": receivers,
        "day_ids": days,
        "physical_ids": physical,
        "scenario_names": np.asarray(["leo_weak"] * 8400),
        "class_ids": np.asarray([f"tx{index}" for index in range(6)]),
        "observation_ids": np.asarray([f"obs{index}" for index in range(8400)]),
    }
    strict = {
        "pre_relu": np.zeros((8400, 160), dtype=np.float32),
        "z_dom": z_dom.copy(),
        "labels": labels.copy(),
        "receiver_ids": receivers.copy(),
        "physical_ids": physical.copy(),
    }
    pool, validation = validate_source_feature_pool(dual, strict)
    output = tmp_path / "split"
    result = publish_d110_sourceheld_split(
        pool,
        historical_query_ids=historical,
        d104_held_ids=old_held,
        d110_ls_ids=d110_ls,
        validation_receipt=validation,
        input_files={"synthetic": {"sha256": "0" * 64}},
        output_dir=output,
    )
    assert result["row_count"] == 1176
    manifest = json.loads(
        (output / "scorer_only" / "source_val" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["row_count"] == 1176
    assert manifest["d106_prepare_member_compatible"] is True
    with np.load(output / "scorer_only" / "source_val" / "features.npz") as archive:
        assert set(archive.files) == {
            "z_id",
            "z_dom",
            "pre_relu",
            "labels",
            "receiver_ids",
            "day_ids",
            "physical_ids",
            "scenario_names",
            "observation_ids",
            "class_ids",
        }
        assert archive["physical_ids"].shape == (1176,)
    receipt_text = (output / "selection_receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["truth_values_persisted_in_selection_receipt"] is False
    assert physical[0] not in receipt_text
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_d110_sourceheld_split(
            pool,
            historical_query_ids=historical,
            d104_held_ids=old_held,
            d110_ls_ids=d110_ls,
            validation_receipt=validation,
            input_files={"synthetic": {"sha256": "0" * 64}},
            output_dir=output,
        )
