from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_held_execution import canonical_sha256
from cvsrffi.stage2_d104_source_split import (
    D104SourceSplitError,
    HISTORICAL_QUERY_CANONICAL_ROOT_SHA256,
    load_d104_exclusion_manifest,
    partition_d104_source_rows,
)


def _metadata():
    labels = []
    receivers = []
    days = []
    physical = []
    for receiver in (f"rx{index}" for index in range(7)):
        for tx in (f"tx{index}" for index in range(6)):
            for day in (f"d{index}" for index in range(4)):
                for sample in range(50):
                    labels.append(tx)
                    receivers.append(receiver)
                    days.append(day)
                    physical.append(f"{receiver}|{tx}|{day}|{sample:02d}")
    historical = []
    for receiver in (f"rx{index}" for index in range(7)):
        for tx in (f"tx{index}" for index in range(6)):
            for day in (f"d{index}" for index in range(4)):
                local = [
                    f"{receiver}|{tx}|{day}|{sample:02d}"
                    for sample in range(15)
                ]
                historical.extend(local)
    historical = historical[:-42]
    assert len(physical) == 8400
    assert len(set(historical)) == 2478
    return (
        np.asarray(labels),
        np.asarray(receivers),
        np.asarray(days),
        np.asarray(physical),
        tuple(sorted(historical)),
    )


def _with_test_root(monkeypatch, historical):
    root = canonical_sha256(list(historical))
    monkeypatch.setattr(
        "cvsrffi.stage2_d104_source_split."
        "HISTORICAL_QUERY_CANONICAL_ROOT_SHA256",
        root,
    )
    return root


def test_d104_split_exact_deterministic_and_closed(monkeypatch) -> None:
    labels, receivers, days, physical, historical = _metadata()
    root = _with_test_root(monkeypatch, historical)
    first, receipt = partition_d104_source_rows(
        labels,
        receivers,
        days,
        physical,
        historical_query_ids=historical,
    )
    second, second_receipt = partition_d104_source_rows(
        labels,
        receivers,
        days,
        physical,
        historical_query_ids=tuple(reversed(historical)),
    )
    assert root != HISTORICAL_QUERY_CANONICAL_ROOT_SHA256
    assert {name: len(rows) for name, rows in first.items()} == {
        "L_s": 588,
        "U_s": 5292,
        "source_val": 2520,
    }
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert receipt == second_receipt
    assert receipt["cell_count"] == 168
    assert receipt["held_per_cell"] == 15
    assert receipt["four_day_labeled_range"] == [2, 4]
    assert receipt["leave_day_labeled_range"] == [10, 12]
    assert receipt["overlap_count"] == 0
    assert receipt["source_labels_used_for_stratified_split"] is True
    assert receipt["query_truth_used_for_method_selection"] is False
    assert receipt["query_truth_used_for_performance_selection"] is False
    assert receipt["source_val_performance_computed"] is False
    selected = set(np.concatenate(tuple(first.values())).tolist())
    excluded_rows = {
        index for index, value in enumerate(physical.tolist()) if value in historical
    }
    assert len(selected) == 8400
    assert not set(first["source_val"].tolist()).intersection(excluded_rows)


def test_d104_split_row_order_equivariance(monkeypatch) -> None:
    labels, receivers, days, physical, historical = _metadata()
    _with_test_root(monkeypatch, historical)
    base, _ = partition_d104_source_rows(
        labels, receivers, days, physical, historical_query_ids=historical
    )
    permutation = np.random.default_rng(104713).permutation(len(labels))
    changed, _ = partition_d104_source_rows(
        labels[permutation],
        receivers[permutation],
        days[permutation],
        physical[permutation],
        historical_query_ids=historical,
    )
    for role in base:
        expected_ids = set(physical[base[role]].tolist())
        actual_ids = set(physical[permutation][changed[role]].tolist())
        assert actual_ids == expected_ids


@pytest.mark.parametrize("mode", ("count", "duplicate", "foreign", "root"))
def test_d104_split_historical_commitment_fail_closed(monkeypatch, mode) -> None:
    labels, receivers, days, physical, historical = _metadata()
    _with_test_root(monkeypatch, historical)
    changed = list(historical)
    if mode == "count":
        changed.pop()
    elif mode == "duplicate":
        changed[-1] = changed[0]
    elif mode == "foreign":
        changed[-1] = "foreign"
    else:
        monkeypatch.setattr(
            "cvsrffi.stage2_d104_source_split."
            "HISTORICAL_QUERY_CANONICAL_ROOT_SHA256",
            "0" * 64,
        )
    with pytest.raises(D104SourceSplitError):
        partition_d104_source_rows(
            labels,
            receivers,
            days,
            physical,
            historical_query_ids=changed,
        )


def test_d104_split_metadata_fail_closed(monkeypatch) -> None:
    labels, receivers, days, physical, historical = _metadata()
    _with_test_root(monkeypatch, historical)
    bad = copy.copy(physical)
    bad[0] = bad[1]
    with pytest.raises(D104SourceSplitError):
        partition_d104_source_rows(
            labels, receivers, days, bad, historical_query_ids=historical
        )


def test_exclusion_manifest_consumer_recomputes_file_and_content_roots(
    monkeypatch, tmp_path
) -> None:
    query_ids = [f"q{index:04d}" for index in range(2478)]
    support_ids = [f"s{index:02d}" for index in range(42)]
    body = {
        "schema": "cvs.d104_r1.historical_query_exclusion_manifest.v2",
        "candidate_id": "D104-R1-ANGQ-RXID-MB4",
        "split_id": "d104_source_seed104713_v2",
        "status": "ACTIVE_REPRODUCIBLE_EXCLUSION_CONTROL",
        "active_query_physical_id_root_sha256": canonical_sha256(query_ids),
        "query_physical_id_count": 2478,
        "query_physical_ids_sorted": query_ids,
        "support_physical_id_count": 42,
        "support_physical_ids_sorted": support_ids,
        "support_query_intersection_count": 0,
        "derivation_code": {"builder_script": {"sha256": "b" * 64}},
        "source_val_labels_used_for_package_reconstruction": True,
        "query_truth_passed_to_predictor": False,
        "query_truth_used_for_scoring": False,
        "performance_computed": False,
        "target_access": False,
        "packages": [{"K": 1} for _ in range(7)],
    }
    content_root = canonical_sha256(body)
    value = {**body, "manifest_content_root_sha256": content_root}
    path = tmp_path / "manifest.json"
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    monkeypatch.setattr(
        "cvsrffi.stage2_d104_source_split."
        "HISTORICAL_QUERY_CANONICAL_ROOT_SHA256",
        canonical_sha256(query_ids),
    )
    monkeypatch.setattr(
        "cvsrffi.stage2_d104_source_split."
        "EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256",
        content_root,
    )
    monkeypatch.setattr(
        "cvsrffi.stage2_d104_source_split.EXCLUSION_BUILDER_SHA256",
        "b" * 64,
    )
    monkeypatch.setattr(
        "cvsrffi.stage2_d104_source_split.EXCLUSION_MANIFEST_FILE_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    assert load_d104_exclusion_manifest(path)["query_physical_id_count"] == 2478
    path.write_bytes(raw + b" ")
    with pytest.raises(D104SourceSplitError, match="file SHA drift"):
        load_d104_exclusion_manifest(path)
