from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_source_archive import (
    D103R1SourceArchiveError,
    LABELED_MEMBERS,
    POOL_MEMBERS,
    SCORER_MEMBERS,
    UNLABELED_MEMBERS,
    partition_source_pool,
    publish_source_split_archives,
)


SHA = hashlib.sha256(b"d103-source-archive-test").hexdigest()


def _pool(rows_per_cell: int = 50) -> dict[str, np.ndarray]:
    labels = [f"tx-{index}" for index in range(6)]
    receivers = [f"rx-{index}" for index in range(7)]
    days = [f"day-{index}" for index in range(4)]
    metadata: list[tuple[str, str, str, str]] = []
    for label in labels:
        for receiver in receivers:
            for day in days:
                for sample in range(rows_per_cell):
                    physical = f"source|{label}|{receiver}|{day}|{sample:04d}"
                    metadata.append((label, receiver, day, physical))
    count = len(metadata)
    rng = np.random.default_rng(103713)
    pre_relu = rng.normal(size=(count, 160)).astype(np.float32)
    return {
        "z_id": np.maximum(pre_relu, np.float32(0.0)),
        "z_dom": rng.normal(size=(count, 160)).astype(np.float32),
        "pre_relu": pre_relu,
        "labels": np.asarray([row[0] for row in metadata], dtype=np.str_),
        "receiver_ids": np.asarray([row[1] for row in metadata], dtype=np.str_),
        "day_ids": np.asarray([row[2] for row in metadata], dtype=np.str_),
        "physical_ids": np.asarray([row[3] for row in metadata], dtype=np.str_),
        "scenario_names": np.asarray(
            ["leo_clear_weak"] * count, dtype=np.str_
        ),
        "observation_ids": np.asarray(
            [f"obs-{index:08d}" for index in range(count)], dtype=np.str_
        ),
        "class_ids": np.asarray(labels, dtype=np.str_),
    }


def test_partition_is_deterministic_disjoint_and_structurally_tx_blind() -> None:
    first = partition_source_pool(_pool())
    second = partition_source_pool(_pool())
    labeled, unlabeled, scorer, receipt = first
    assert tuple(labeled) == LABELED_MEMBERS
    assert tuple(unlabeled) == UNLABELED_MEMBERS
    assert tuple(scorer) == SCORER_MEMBERS
    assert "tx_labels" not in unlabeled
    assert "pre_relu" not in unlabeled
    assert receipt["cell_count"] == 168
    assert receipt["receiver_tx_group_count"] == 42
    assert receipt["counts"] == {"L_s": 588, "U_s": 5292, "source_val": 2520}
    assert receipt["leave_one_day_k10_reachable"] is True
    assert receipt["overlap_count"] == 0
    assert receipt["union_complete"] is True
    assert receipt == second[3]
    for left, right in zip(first[:3], second[:3]):
        for name in left:
            assert np.array_equal(left[name], right[name])
    ids = [
        set(labeled["physical_ids"].astype(str)),
        set(unlabeled["physical_ids"].astype(str)),
        set(scorer["physical_ids"].astype(str)),
    ]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_publish_exact_fit_manifests_and_scorer_separation(tmp_path: Path) -> None:
    root = tmp_path / "split"
    result = publish_source_split_archives(
        _pool(),
        output_dir=root,
        checkpoint_sha256=SHA,
        runtime_sha256=SHA,
        cache_set_sha256=SHA,
        selection_salt_receipt_sha256=SHA,
    )
    assert result["status"] == "FORMAL_PHASE1_SOURCE_SPLIT_COMPLETE"
    with np.load(root / "L_s" / "features.npz", allow_pickle=False) as archive:
        assert tuple(archive.files) == LABELED_MEMBERS
    with np.load(root / "U_s" / "features.npz", allow_pickle=False) as archive:
        assert tuple(archive.files) == UNLABELED_MEMBERS
    with np.load(
        root / "scorer_only" / "source_val" / "features.npz", allow_pickle=False
    ) as archive:
        assert tuple(archive.files) == SCORER_MEMBERS
    source_val_manifest = json.loads(
        (root / "source_val.manifest.json").read_text(encoding="utf-8")
    )
    assert source_val_manifest["archive_sha256"] is None
    assert source_val_manifest["tx_visibility"] == "scorer_only"
    assert not (root / "source_val" / "features.npz").exists()
    with pytest.raises(FileExistsError):
        publish_source_split_archives(
            _pool(),
            output_dir=root,
            checkpoint_sha256=SHA,
            runtime_sha256=SHA,
            cache_set_sha256=SHA,
            selection_salt_receipt_sha256=SHA,
        )


def test_rejects_relu_drift_and_duplicate_physical_id() -> None:
    pool = _pool()
    pool["z_id"][0, 0] += np.float32(1.0)
    with pytest.raises(D103R1SourceArchiveError, match="ReLU"):
        partition_source_pool(pool)


def test_rejects_cell_capacity_that_cannot_reserve_all_roles() -> None:
    pool = _pool()
    labels = pool["labels"].astype(str)
    receivers = pool["receiver_ids"].astype(str)
    days = pool["day_ids"].astype(str)
    source = np.flatnonzero(
        (labels == "tx-0") & (receivers == "rx-0") & (days == "day-0")
    )
    target = np.flatnonzero(
        (labels == "tx-0") & (receivers == "rx-0") & (days == "day-1")
    )
    for index in source[3:]:
        pool["day_ids"][index] = pool["day_ids"][target[0]]
    with pytest.raises(D103R1SourceArchiveError, match="reserve capacity"):
        partition_source_pool(pool)
    pool = _pool()
    pool["physical_ids"][1] = pool["physical_ids"][0]
    with pytest.raises(D103R1SourceArchiveError, match="unique"):
        partition_source_pool(pool)


def test_pool_member_order_is_closed() -> None:
    pool = _pool()
    assert tuple(pool) == POOL_MEMBERS
    reordered = {name: pool[name] for name in reversed(POOL_MEMBERS)}
    with pytest.raises(D103R1SourceArchiveError, match="member order"):
        partition_source_pool(reordered)
