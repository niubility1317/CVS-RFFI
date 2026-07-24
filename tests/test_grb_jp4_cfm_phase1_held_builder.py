from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cvsrffi.grb_jp4_cfm_phase1_held_builder import (
    GRBJP4HeldBuilderError,
    build_phase1_method_lock,
    build_phase1_qknn_locks,
    build_source_aggregate,
)


def _tap() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(123)
    classes = tuple(f"c{index}" for index in range(6))
    rows = []
    metadata = []
    index = 0
    for receiver_index in range(4):
        for day_index in range(2):
            for class_index, class_id in enumerate(classes):
                for local in range(12):
                    vector = np.zeros(160, dtype=np.float64)
                    vector[class_index] = 5.0
                    vector += 0.01 * generator.normal(size=160)
                    vector /= np.linalg.norm(vector)
                    pre = vector + 0.02
                    hidden = generator.normal(size=320)
                    rows.append((vector, hidden, pre))
                    metadata.append(
                        (
                            class_id,
                            f"r{receiver_index}",
                            f"d{day_index}",
                            f"p{index}",
                            "leo_clear_weak",
                            f"o{index}",
                        )
                    )
                    index += 1
    z_id = np.asarray([row[0] for row in rows], dtype=np.float32)
    hidden = np.asarray([row[1] for row in rows], dtype=np.float32)
    pre = np.asarray([row[2] for row in rows], dtype=np.float32)
    return {
        "z_id": z_id,
        "hidden": hidden,
        "pre_relu": pre,
        "joint_weight": generator.normal(size=(160, 320)).astype(np.float32),
        "labels": np.asarray([row[0] for row in metadata]),
        "receiver_ids": np.asarray([row[1] for row in metadata]),
        "day_ids": np.asarray([row[2] for row in metadata]),
        "physical_ids": np.asarray([row[3] for row in metadata]),
        "scenario_names": np.asarray([row[4] for row in metadata]),
        "class_ids": np.asarray(classes),
        "observation_ids": np.asarray([row[5] for row in metadata]),
    }


def test_source_aggregate_is_deterministic_source_only_and_complete():
    tap = _tap()
    locks = build_phase1_qknn_locks()
    first = build_source_aggregate(tap, qknn_locks=locks)
    second = build_source_aggregate(tap, qknn_locks=locks)
    assert len(first["ground_multiprototypes"]) == 6
    assert all(len(item["prototypes"]) == 3 for item in first["ground_multiprototypes"])
    for class_record in first["ground_multiprototypes"]:
        for prototype in class_record["prototypes"]:
            receipt = prototype["aggregation_receipt"]
            assert receipt["distinct_physical_sample_count"] >= 2
            assert receipt["aggregation_radius"] >= 0.0
            assert receipt["member_ids_included"] is False
    assert np.array_equal(
        first["receiver_day_means"], second["receiver_day_means"]
    )
    assert np.array_equal(
        first["phase1_qknn_margin_receipt"]["margins"],
        second["phase1_qknn_margin_receipt"]["margins"],
    )
    assert len(first["phase1_qknn_margin_receipt"]["margins"]) >= 2


def test_source_aggregate_numeric_state_is_opaque_label_permutation_equivariant():
    tap = _tap()
    mapping = {
        value: replacement
        for value, replacement in zip(
            tap["class_ids"].astype(str).tolist(),
            ("opaque-z", "opaque-v", "opaque-x", "opaque-u", "opaque-y", "opaque-w"),
        )
    }
    renamed = {
        name: np.array(value, copy=True)
        for name, value in tap.items()
    }
    renamed["labels"] = np.asarray(
        [mapping[value] for value in tap["labels"].astype(str).tolist()]
    )
    renamed["class_ids"] = np.asarray(
        [mapping[value] for value in tap["class_ids"].astype(str).tolist()]
    )
    locks = build_phase1_qknn_locks()
    original = build_source_aggregate(tap, qknn_locks=locks)
    permuted = build_source_aggregate(renamed, qknn_locks=locks)
    assert np.array_equal(
        original["receiver_day_means"], permuted["receiver_day_means"]
    )
    assert np.array_equal(
        original["receiver_day_mask"], permuted["receiver_day_mask"]
    )
    assert np.array_equal(
        original["receiver_day_physical_counts"],
        permuted["receiver_day_physical_counts"],
    )
    assert np.array_equal(
        original["phase1_qknn_margin_receipt"]["margins"],
        permuted["phase1_qknn_margin_receipt"]["margins"],
    )
    for original_class, permuted_class in zip(
        original["ground_multiprototypes"],
        permuted["ground_multiprototypes"],
    ):
        assert all(
            np.array_equal(left["vector"], right["vector"])
            for left, right in zip(
                original_class["prototypes"],
                permuted_class["prototypes"],
            )
        )


def test_three_k_locks_are_distinct_and_method_lock_binds_all():
    locks = build_phase1_qknn_locks()
    digests = {k: value.lock_digest for k, value in locks.items()}
    assert len(set(digests.values())) == 3
    method = build_phase1_method_lock(
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        class_handle_binding_sha256=hashlib.sha256(b"registry").hexdigest(),
        qknn_locks=locks,
    )
    assert method["qknn_lock_sha256_by_k"] == {
        str(k): digests[k] for k in (1, 5, 10)
    }
    assert method["target25_release_authorized"] is False


def test_duplicate_physical_identity_or_missing_k_fails_closed():
    tap = _tap()
    tap["physical_ids"] = tap["physical_ids"].copy()
    tap["physical_ids"][1] = tap["physical_ids"][0]
    with pytest.raises(GRBJP4HeldBuilderError, match="class/physical"):
        build_source_aggregate(tap, qknn_locks=build_phase1_qknn_locks())
    locks = build_phase1_qknn_locks()
    del locks[10]
    with pytest.raises(GRBJP4HeldBuilderError, match="K1/K5/K10"):
        build_phase1_method_lock(
            checkpoint_sha256="0" * 64,
            class_handle_binding_sha256="1" * 64,
            qknn_locks=locks,
        )
