from __future__ import annotations

import io

import numpy as np
import pytest

from paper_reproduction.cvs_aligned.support_only_multiprototype_head import (
    fit_support_only_multiprototype_head,
    pack_support_only_multiprototype_head,
    predict_support_only_multiprototype_head,
    score_support_only_multiprototype_head,
    unpack_support_only_multiprototype_head,
)


def _bimodal_support() -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(
        [
            [1.0, 0.08, 0.0], [0.96, -0.04, 0.0],
            [-1.0, 0.08, 0.0], [-0.96, -0.04, 0.0],
            [0.08, 1.0, 0.0], [-0.04, 0.96, 0.0],
            [0.08, -1.0, 0.0], [-0.04, -0.96, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return rows, labels


def test_multi_prototype_preserves_bimodal_classes_that_centroids_collapse() -> None:
    support, labels = _bimodal_support()
    head = fit_support_only_multiprototype_head(
        support,
        labels,
        class_count=2,
        max_prototypes_per_class=2,
        residual_shrinkage=1.0,
        hubness_weight=0.0,
    )
    query = np.asarray(
        [[0.99, 0.0, 0.0], [-0.99, 0.0, 0.0], [0.0, 0.99, 0.0], [0.0, -0.99, 0.0]],
        dtype=np.float32,
    )
    assert predict_support_only_multiprototype_head(query, head).tolist() == [0, 0, 1, 1]


def test_scoring_is_per_sample_and_batch_composition_invariant() -> None:
    support, labels = _bimodal_support()
    head = fit_support_only_multiprototype_head(support, labels, class_count=2)
    query = np.asarray([[0.8, 0.1, 0.0], [0.1, -0.8, 0.0]], dtype=np.float32)
    together = score_support_only_multiprototype_head(query, head)
    separate = np.concatenate(
        [score_support_only_multiprototype_head(row[None, :], head) for row in query],
        axis=0,
    )
    np.testing.assert_allclose(together, separate, rtol=0.0, atol=0.0)


def test_resource_state_is_small_and_recomputed_from_tensors() -> None:
    rng = np.random.default_rng(7)
    support = rng.normal(size=(26 * 10, 160)).astype(np.float32)
    labels = np.repeat(np.arange(26, dtype=np.int64), 10)
    head = fit_support_only_multiprototype_head(support, labels, class_count=26)
    assert head.prototype_count == 52
    assert head.persistent_state_bytes_fp16 < 256 * 1024
    assert head.extra_macs_per_query > 0
    assert head.support_audit["query_rows_used"] == 0
    assert head.support_audit["dense_query_graph_used"] is False


def test_requires_every_registered_class_and_finite_support() -> None:
    support, labels = _bimodal_support()
    with pytest.raises(ValueError, match="every registered class"):
        fit_support_only_multiprototype_head(support, labels, class_count=3)
    broken = support.copy()
    broken[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        fit_support_only_multiprototype_head(broken, labels, class_count=2)


def test_query_requires_finite_rank_two_rows() -> None:
    support, labels = _bimodal_support()
    head = fit_support_only_multiprototype_head(support, labels, class_count=2)
    with pytest.raises(ValueError, match="shape"):
        score_support_only_multiprototype_head(np.zeros((1, 1, 3), dtype=np.float32), head)
    broken = np.zeros((1, 3), dtype=np.float32)
    broken[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        score_support_only_multiprototype_head(broken, head)


def test_pickle_free_fp16_capsule_round_trip_preserves_predictions() -> None:
    support, labels = _bimodal_support()
    head = fit_support_only_multiprototype_head(support, labels, class_count=2)
    packed = pack_support_only_multiprototype_head(head)
    buffer = io.BytesIO()
    np.savez(buffer, **packed)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        restored = unpack_support_only_multiprototype_head(
            {key: np.asarray(archive[key]) for key in archive.files}
        )
    query = np.asarray(
        [[0.99, 0.0, 0.0], [0.0, -0.99, 0.0], [0.1, 0.8, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        predict_support_only_multiprototype_head(query, restored),
        predict_support_only_multiprototype_head(query, head),
    )
    assert restored.persistent_state_bytes_fp16 == head.persistent_state_bytes_fp16
    assert restored.support_audit == head.support_audit


def test_capsule_rejects_schema_or_member_drift() -> None:
    support, labels = _bimodal_support()
    packed = pack_support_only_multiprototype_head(
        fit_support_only_multiprototype_head(support, labels, class_count=2)
    )
    with_extra = dict(packed)
    with_extra["query_labels"] = np.asarray([0], dtype=np.int64)
    with pytest.raises(ValueError, match="members mismatch"):
        unpack_support_only_multiprototype_head(with_extra)
    wrong_schema = dict(packed)
    wrong_schema["schema_utf8"] = np.frombuffer(b"wrong", dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported head schema"):
        unpack_support_only_multiprototype_head(wrong_schema)
