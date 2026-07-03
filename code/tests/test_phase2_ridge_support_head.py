import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts"))

from phase2_ridge_support_head_sweep import predict_ridge_head, train_ridge_head  # noqa: E402


def test_ridge_head_stores_weights_not_support_rows():
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

    model = train_ridge_head(support, labels, old_labels={"old-a"}, l2=0.01)

    assert model.weight_matrix.shape == (3, 2)
    assert model.class_labels.tolist() == ["new-b", "old-a"]
    assert "support_features" not in model.__dict__
    assert "support_labels" not in model.__dict__


def test_ridge_head_old_bias_can_rescue_borderline_old_query():
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
    query = np.asarray([[0.8, 0.54]], dtype=np.float64)
    model = train_ridge_head(support, labels, old_labels={"old-a"}, l2=0.1)

    assert predict_ridge_head(model, query, old_bias=0.0).tolist() == ["new-b"]
    assert predict_ridge_head(model, query, old_bias=0.25).tolist() == ["old-a"]
