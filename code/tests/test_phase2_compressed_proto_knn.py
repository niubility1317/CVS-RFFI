import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts"))

from phase2_compressed_proto_knn_sweep import (  # noqa: E402
    build_quantized_knn_memory,
    build_compressed_memory,
    predict_quantized_knn_memory,
    predict_compressed_memory,
)


def test_compressed_memory_keeps_class_statistics_not_support_rows():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "new-b", "new-b"], dtype=object)

    memory = build_compressed_memory(support, labels, old_labels={"old-a"}, prototypes_per_class=1)

    assert memory.prototype_matrix.shape == (2, 2)
    assert memory.prototype_labels.tolist() == ["new-b", "old-a"]
    assert memory.counts == {"new-b": 2, "old-a": 2}
    assert "support" not in memory.__dict__
    assert "support_features" not in memory.__dict__
    assert "support_labels" not in memory.__dict__


def test_old_bias_rescues_old_query_with_compressed_prototypes():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.58, 0.82],
            [0.62, 0.78],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "new-b", "new-b"], dtype=object)
    query = np.asarray([[0.82, 0.55]], dtype=np.float64)
    memory = build_compressed_memory(support, labels, old_labels={"old-a"}, prototypes_per_class=1)

    assert predict_compressed_memory(memory, query, old_bias=0.0).tolist() == ["new-b"]
    assert predict_compressed_memory(memory, query, old_bias=0.16).tolist() == ["old-a"]


def test_medoid_anchor_mode_keeps_bounded_representative_embeddings():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.7, 0.7],
            [0.72, 0.68],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "new-b", "new-b", "new-b", "new-b"], dtype=object)

    memory = build_compressed_memory(
        support,
        labels,
        old_labels={"old-a"},
        prototypes_per_class=2,
        prototype_mode="medoid",
    )

    normalized_support = support / np.linalg.norm(support, axis=1, keepdims=True)
    assert memory.prototype_matrix.shape == (4, 2)
    assert memory.counts == {"new-b": 4, "old-a": 2}
    assert all(
        np.any(np.all(np.isclose(anchor, normalized_support, atol=1e-8), axis=1))
        for anchor in memory.prototype_matrix
    )
    assert "support_features" not in memory.__dict__


def test_boundary_medoid_keeps_near_boundary_anchor_without_full_support():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.96, 0.04],
            [0.62, 0.78],
            [0.0, 1.0],
            [0.04, 0.96],
            [0.78, 0.62],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"], dtype=object)

    memory = build_compressed_memory(
        support,
        labels,
        old_labels={"old-a"},
        prototypes_per_class=2,
        prototype_mode="boundary_medoid",
    )

    normalized_support = support / np.linalg.norm(support, axis=1, keepdims=True)
    near_boundary_old = normalized_support[2]
    near_boundary_new = normalized_support[5]

    assert memory.prototype_matrix.shape == (4, 2)
    assert any(np.allclose(anchor, near_boundary_old, atol=1e-8) for anchor in memory.prototype_matrix)
    assert any(np.allclose(anchor, near_boundary_new, atol=1e-8) for anchor in memory.prototype_matrix)
    assert predict_compressed_memory(memory, np.asarray([[0.70, 0.70]], dtype=np.float64)).tolist() in (
        ["old-a"],
        ["new-b"],
    )
    assert "support_features" not in memory.__dict__


def test_weighted_anchor_memory_uses_counts_without_storing_support():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.92, 0.08],
            [0.0, 1.0],
            [0.08, 0.92],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "old-a", "old-a", "new-b", "new-b"], dtype=object)

    memory = build_compressed_memory(
        support,
        labels,
        old_labels={"old-a"},
        prototypes_per_class=1,
        prototype_weight_mode="assigned_count",
    )

    assert memory.prototype_weights.tolist() == [2.0, 4.0]
    query = np.asarray([[0.64, 0.65]], dtype=np.float64)

    assert predict_compressed_memory(memory, query).tolist() == ["new-b"]
    assert predict_compressed_memory(
        memory,
        query,
        weight_scale=0.12,
    ).tolist() == ["old-a"]
    assert "support_features" not in memory.__dict__
    assert "support_labels" not in memory.__dict__


def test_loo_knn1_agreement_downweights_anchor_that_disagrees_with_support_teacher():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.18, 0.98],
            [0.0, 1.0],
            [0.05, 0.99],
            [0.10, 0.96],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "old-a", "new-b", "new-b", "new-b"], dtype=object)

    memory = build_compressed_memory(
        support,
        labels,
        old_labels={"old-a"},
        prototypes_per_class=2,
        prototype_mode="medoid",
        prototype_weight_mode="loo_knn1_agreement",
    )

    old_weights = memory.prototype_weights[memory.prototype_labels == "old-a"]

    assert old_weights.min() < 1.0
    assert old_weights.max() > 1.0
    assert "support_features" not in memory.__dict__
    assert "support_labels" not in memory.__dict__


def test_quantized_knn_memory_matches_knn1_without_raw_support_storage():
    support = np.asarray(
        [
            [1.0, 0.0],
            [0.92, 0.08],
            [0.0, 1.0],
            [0.08, 0.92],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["old-a", "old-a", "new-b", "new-b"], dtype=object)
    query = np.asarray([[0.93, 0.07], [0.07, 0.93]], dtype=np.float64)

    memory = build_quantized_knn_memory(support, labels, old_labels={"old-a"}, quant_bits=8)

    assert memory.quantized_matrix.dtype == np.int8
    assert memory.quantized_matrix.shape == support.shape
    assert memory.stored_support_count == 0
    assert memory.stored_quantized_count == support.shape[0]
    assert predict_quantized_knn_memory(memory, query, k=1).tolist() == ["old-a", "new-b"]
    assert "support_features" not in memory.__dict__
    assert "support_labels" not in memory.__dict__
