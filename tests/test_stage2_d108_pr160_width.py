from __future__ import annotations

import numpy as np

from cvsrffi.stage2_d108_target125_runner import _coerce_materialized_state


def test_typed_materializer_accepts_160_width_without_changing_default_contract() -> None:
    state = {
        "support_features": (np.ones((2, 160), dtype=np.float32) / np.sqrt(160.0)).astype(np.float32),
        "support_labels": ("a", "b"),
        "registered_classes": ("a", "b"),
        "support_physical_ids": ("s-a", "s-b"),
        "query_features": (np.ones((1, 160), dtype=np.float32) / np.sqrt(160.0)).astype(np.float32),
        "query_physical_ids": ("q-a",),
    }
    request = {"k_shot": 1}
    result = _coerce_materialized_state(state, request=request, feature_width=160)
    assert result.support_features.shape == (2, 160)
    assert result.query_features.shape == (1, 160)
