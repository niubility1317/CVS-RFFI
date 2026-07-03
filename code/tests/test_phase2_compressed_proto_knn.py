import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts"))

from phase2_compressed_proto_knn_sweep import (  # noqa: E402
    build_compressed_memory,
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
